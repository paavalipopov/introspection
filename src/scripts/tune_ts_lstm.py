# pylint: disable-all
import argparse

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback
from apto.utils.report import get_classification_report
from catalyst import utils
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.settings import LOGS_ROOT, UTCNOW
from src.ts import load_ABIDE1, TSQuantileTransformer

import wandb


class LSTM(nn.Module):
    def __init__(
        self,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        bidirectional: bool = False,
        **kwargs,
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(hidden_size=hidden_size, bidirectional=bidirectional, **kwargs)
        self.fc = nn.Sequential(
            nn.Dropout(p=fc_dropout),
            nn.Linear(2 * hidden_size if bidirectional else hidden_size, 1),
        )

    def forward(self, x):
        lstm_output, _ = self.lstm(x)

        if self.bidirectional:
            out_forward = lstm_output[:, -1, : self.hidden_size]
            out_reverse = lstm_output[:, 0, self.hidden_size :]
            lstm_output = torch.cat((out_forward, out_reverse), 1)
        else:
            lstm_output = lstm_output[:, -1, :]

        fc_output = self.fc(lstm_output)
        return fc_output


class Experiment(IExperiment):
    def __init__(self, quantile: bool, max_epochs: int, logdir: str) -> None:
        super().__init__()
        assert not quantile, "Not implemented yet"
        self._quantile: bool = quantile
        self._trial: optuna.Trial = None
        self.max_epochs = max_epochs
        self.logdir = logdir

    def on_tune_start(self):
        features, labels = load_ABIDE1()
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        if self._quantile:
            n_quantiles = 10
            n_offset = 3  # 0 - pad, 1 - cls, 2 - mask
            transform = TSQuantileTransformer(n_quantiles=n_quantiles, random_state=42)
            transform = transform.fit(X_train)
            X_train = transform.transform(X_train) + n_offset
            X_test = transform.transform(X_test) + n_offset

        X_train = np.swapaxes(X_train, 1, 2)  # [n_samples; seq_len; n_features]
        X_test = np.swapaxes(X_test, 1, 2)

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )

    def on_experiment_start(self, exp: "IExperiment"):
        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(project="tune_lstm", name="{UTCNOW}-lstm")

        super().on_experiment_start(exp)
        # setup experiment
        self.num_epochs = self._trial.suggest_int("exp.num_epochs", 1, self.max_epochs)
        # setup data
        self.batch_size = self._trial.suggest_int("data.batch_size", 32, 64, log=True)
        self.datasets = {
            "train": DataLoader(
                self._train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True
            ),
            "valid": DataLoader(
                self._valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
            ),
        }
        # setup model
        hidden_size = self._trial.suggest_int("lstm.hidden_size", 32, 256, log=True)
        num_layers = self._trial.suggest_int("lstm.num_layers", 1, 4)
        bidirectional = self._trial.suggest_categorical("lstm.bidirectional", [True, False])
        fc_dropout = self._trial.suggest_uniform("lstm.fc_dropout", 0.1, 0.9)
        self.model = LSTM(
            input_size=53,  # PRIOR
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            fc_dropout=fc_dropout,
        )
        lr = self._trial.suggest_float("adam.lr", 1e-5, 1e-3, log=True)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
        )
        # setup callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=False,
                patience=5,
                dataset_key="valid",
                metric_key="score",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="model",
                logdir=f"{self.logdir}/{self._trial.number:04d}",
                dataset_key="valid",
                metric_key="score",
                minimize=False,
            ),
        }
        self.wandb_logger.config.update(
            {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "bidirectional": bidirectional,
                "fc_dropout": fc_dropout,
                "lr": lr,
            }
        )

    def run_dataset(self) -> None:
        all_scores, all_targets = [], []
        total_loss, total_accuracy = 0.0, 0.0
        self.model.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(self.dataset)):
                self.optimizer.zero_grad()
                score = self.model(data).view(-1)
                loss = self.criterion(score, target)
                score = torch.sigmoid(score)
                pred = (score > 0.5).to(torch.int32)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()
                total_accuracy += pred.eq(target.view_as(pred)).sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step
        total_accuracy /= self.dataset_batch_step * self.batch_size

        y_test = np.hstack(all_targets)
        y_score = np.hstack(all_scores)
        y_pred = (y_score > 0.5).astype(np.int32)
        report = get_classification_report(y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5)
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key:
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))

        self.dataset_metrics = {
            "score": report["auc"].loc["weighted"],
            "accuracy": total_accuracy,
            "loss": total_loss,
        }
        if self.is_train_dataset:
            self.wandb_logger.log(
                {
                    "train_score": self.dataset_metrics["score"],
                    "train_accuracy": self.dataset_metrics["accuracy"],
                    "train_loss": self.dataset_metrics["loss"],
                }
            )
        else:
            self.wandb_logger.log(
                {
                    "valid_score": self.dataset_metrics["score"],
                    "valid_accuracy": self.dataset_metrics["accuracy"],
                    "valid_loss": self.dataset_metrics["loss"],
                }
            )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.callbacks["early-stop"].best_score
        self.wandb_logger.log(
            {
                "best_score": self._score,
            }
        )
        self.wandb_logger.finish()

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()

        return self._score

    def tune(self, n_trials: int):
        self.on_tune_start()
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)
        logfile = f"{self.logdir}/optuna.csv"
        df = self.study.trials_dataframe()
        df.to_csv(logfile, index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    utils.boolean_flag(parser, "quantile", default=False)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--num-trials", type=int, default=1)
    args = parser.parse_args()
    Experiment(
        quantile=args.quantile,
        max_epochs=args.max_epochs,
        logdir=f"{LOGS_ROOT}/{UTCNOW}-ts-lstm-q{args.quantile}/",
    ).tune(n_trials=args.num_trials)

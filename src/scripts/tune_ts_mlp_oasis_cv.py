# pylint: disable-all
import argparse

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback
from apto.utils.report import get_classification_report
from catalyst import utils
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.settings import LOGS_ROOT, UTCNOW
from src.ts import load_balanced_OASIS, TSQuantileTransformer

import wandb


class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 0,
    ):
        super(MLP, self).__init__()
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(p=dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(
                ResidualBlock(
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(p=dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.ReLU(),
                    )
                )
            )
        layers.append(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, output_size),
            )
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        bs, ln, fs = x.shape
        fc_output = self.fc(x.reshape(-1, fs))
        fc_output = fc_output.reshape(bs, ln, -1).mean(1)  # .squeeze(1)
        return fc_output


class Experiment(IExperiment):
    def __init__(self, quantile: bool, max_epochs: int, logdir: str) -> None:
        super().__init__()
        assert not quantile, "Not implemented yet"
        self._quantile: bool = quantile
        self._trial: optuna.Trial = None
        self.max_epochs = max_epochs
        self.logdir = logdir

    def on_tune_start(self, trial, k):
        self.k = k
        self.trial = trial
        features, labels = load_balanced_OASIS()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + trial)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        X_train = np.swapaxes(X_train, 1, 2)  # [n_samples; seq_len; n_features]
        X_test = np.swapaxes(X_test, 1, 2)

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        )

    def on_experiment_start(self, exp: "IExperiment"):
        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project="mlp_oasis_cv_1", name=f"{UTCNOW}-k_{self.k}-trial_{self.trial}"
        )

        super().on_experiment_start(exp)
        # # setup experiment
        # self.num_epochs = self._trial.suggest_int("exp.num_epochs", 20, self.max_epochs)
        # # setup data
        # self.batch_size = self._trial.suggest_int("data.batch_size", 4, 32, log=True)
        # self.datasets = {
        #     "train": DataLoader(
        #         self._train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True
        #     ),
        #     "valid": DataLoader(
        #         self._valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        #     ),
        # }
        # # setup model
        # hidden_size = self._trial.suggest_int("mlp.hidden_size", 32, 256, log=True)
        # num_layers = self._trial.suggest_int("mlp.num_layers", 0, 4)
        # dropout = self._trial.suggest_uniform("mlp.dropout", 0.1, 0.9)
        # self.model = MLP(
        #     input_size=53,  # PRIOR
        #     output_size=2,  # PRIOR
        #     hidden_size=hidden_size,
        #     num_layers=num_layers,
        #     dropout=dropout,
        # )

        # best tune
        # model = MLP(
        #     input_size=53,  # PRIOR
        #     output_size=2,  # PRIOR
        #     hidden_size=997,
        #     num_layers=4,
        #     dropout=0.20352535084272705,
        # )

        # best cv
        self.num_epochs = 32
        # setup data
        self.batch_size = 6
        self.datasets = {
            "train": DataLoader(
                self._train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True
            ),
            "valid": DataLoader(
                self._valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
            ),
        }
        # setup model
        hidden_size = 142
        num_layers = 2
        dropout = 0.15847198018446662
        self.model = MLP(
            input_size=53,  # PRIOR
            output_size=2,  # PRIOR
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

        # lr = self._trial.suggest_float("adam.lr", 1e-5, 1e-3, log=True)
        lr = 0.0002222585782420201
        self.criterion = nn.CrossEntropyLoss()
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
                "dropout": dropout,
                "lr": lr,
            }
        )

    def run_dataset(self) -> None:
        all_scores, all_targets = [], []
        total_loss = 0.0
        self.model.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(self.dataset)):
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
        report = get_classification_report(y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5)
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key:
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))

        self.dataset_metrics = {
            "score": report["auc"].loc["weighted"],
            "loss": total_loss,
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        self.wandb_logger.log(
            {
                "train_score": self.epoch_metrics["train"]["score"],
                "train_loss": self.epoch_metrics["train"]["loss"],
                "valid_score": self.epoch_metrics["valid"]["score"],
                "valid_loss": self.epoch_metrics["valid"]["loss"],
            },
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.callbacks["early-stop"].best_score

        wandb.summary["valid_score"] = self._score
        self.wandb_logger.finish()

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()

        return self._score

    def tune(self, n_trials: int):
        for trial in range(n_trials):
            for k in range(5):
                self.on_tune_start(trial, k)
                self.study = optuna.create_study(direction="maximize")
                self.study.optimize(self._objective, n_trials=1, n_jobs=1)
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
        logdir=f"{LOGS_ROOT}/{UTCNOW}-ts-mlp-oasis-q{args.quantile}/",
    ).tune(n_trials=args.num_trials)

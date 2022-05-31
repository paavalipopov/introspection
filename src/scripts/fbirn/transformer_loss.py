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
from src.ts import load_FBIRN, TSQuantileTransformer

import wandb
import json


class Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        input_len: int,
        fc_dropout: float = 0.5,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_heads: int = 8,
    ):
        super(Transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            transformer_encoder,
        ]
        self.transformer = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(input_len * hidden_size),
            nn.Dropout(p=fc_dropout),
            nn.Linear(input_len * hidden_size, 2),
        )

    def forward(self, x):
        fc_output = self.transformer(x)
        fc_output = self.fc(fc_output)
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
        features, labels = load_FBIRN()

        # print("Features shape: ", features.shape)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # print("Raw train shape: ", X_train.shape)
        # print("Test shape: ", X_test.shape)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=int(X_train.shape[0] / 4),
            random_state=42 + trial,
            stratify=y_train,
        )

        X_train = np.swapaxes(X_train, 1, 2)  # [n_samples; seq_len; n_features]
        X_val = np.swapaxes(X_val, 1, 2)  # [n_samples; seq_len; n_features]
        X_test = np.swapaxes(X_test, 1, 2)

        # print("Train shape: ", X_train.shape)
        # print("Val shape: ", X_val.shape)

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        )
        self._test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        )

    def on_experiment_start(self, exp: "IExperiment"):
        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project="fbirn_transf_loss", name=f"{UTCNOW}-k_{self.k}-trial_{self.trial}"
        )

        super().on_experiment_start(exp)

        # # setup experiment
        # self.num_epochs = self._trial.suggest_int("exp.num_epochs", 10, self.max_epochs)
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
        # hidden_size = self._trial.suggest_int("transformer.hidden_size", 4, 128, log=True)
        # num_heads = self._trial.suggest_int("transformer.num_heads", 1, 4)
        # num_layers = self._trial.suggest_int("transformer.num_layers", 1, 4)
        # fc_dropout = self._trial.suggest_uniform("transformer.fc_dropout", 0.2, 0.8)
        # lr = self._trial.suggest_float("adam.lr", 1e-5, 1e-3, log=True)

        # best cv
        self.num_epochs = 64
        self.batch_size = 8
        self.datasets = {
            "train": DataLoader(
                self._train_ds, batch_size=self.batch_size, num_workers=0, shuffle=True
            ),
            "valid": DataLoader(
                self._valid_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
            ),
        }
        # setup model
        hidden_size = 80
        num_heads = 1
        num_layers = 2
        fc_dropout = 0.4374500259545924
        lr = 0.00018054194996419817

        self.model = Transformer(
            input_size=53,  # PRIOR
            input_len=140,  # PRIOR
            hidden_size=hidden_size * num_heads,
            num_layers=num_layers,
            num_heads=num_heads,
            fc_dropout=fc_dropout,
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        # loss callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=True,
                patience=16,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="model",
                logdir=f"{self.logdir}/{self._trial.number:04d}",
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

        # # score callbacks
        # self.callbacks = {
        #     "early-stop": EarlyStoppingCallback(
        #         minimize=False,
        #         patience=16,
        #         dataset_key="valid",
        #         metric_key="score",
        #         min_delta=0.001,
        #     ),
        #     "checkpointer": TorchCheckpointerCallback(
        #         exp_attr="model",
        #         logdir=f"{self.logdir}/{self._trial.number:04d}",
        #         dataset_key="valid",
        #         metric_key="score",
        #         minimize=False,
        #     ),
        # }

        self.wandb_logger.config.update(
            {
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "fc_dropout": fc_dropout,
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

    def run_test_dataset(self) -> None:

        test_ds = DataLoader(
            self._test_ds, batch_size=self.batch_size, num_workers=0, shuffle=False
        )

        f = open(
            f"{LOGS_ROOT}/{UTCNOW}-ts-transformer-oasis-q{args.quantile}/0000/model.storage.json"
        )
        logpath = json.load(f)["storage"][0]["logpath"]
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint)
        self.model.train(False)
        self.model.zero_grad()

        all_scores, all_targets = [], []
        total_loss = 0.0

        with torch.set_grad_enabled(False):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(test_ds)):
                self.optimizer.zero_grad()
                logits = self.model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()

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

        print("AUC ", report["auc"].loc["weighted"])
        print("Loss ", total_loss)
        self.wandb_logger.log(
            {
                "test_score": report["auc"].loc["weighted"],
                "test_loss": total_loss,
            },
        )

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        self._score = self.callbacks["early-stop"].best_score

        print("Run test dataset")
        self.run_test_dataset()

        wandb.summary["valid_loss"] = self._score
        # wandb.summary["valid_score"] = self._score
        self.wandb_logger.finish()

    def _objective(self, trial) -> float:
        self._trial = trial
        self.run()

        return self._score

    def tune(self, n_trials: int):
        for k in range(5):
            for trial in range(n_trials):
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
        logdir=f"{LOGS_ROOT}/{UTCNOW}-ts-transformer-oasis-q{args.quantile}/",
    ).tune(n_trials=args.num_trials)

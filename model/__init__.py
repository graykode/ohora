from typing import Any, Dict, Tuple

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam

from metric import Metric
from model.lstm import LSTMClassifier


class Model(pl.LightningModule):
    def __init__(self, conf: omegaconf.DictConfig, num_class: list) -> None:
        super().__init__()
        self.conf = conf
        self.save_hyperparameters(conf)

        self.model = LSTMClassifier(conf)

        self.criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor([1 - (x / sum(num_class)) for x in num_class])
        )
        self.eval_acc = Metric(dist_sync_on_step=False)

    def configure_optimizers(self) -> Any:
        optimizer = Adam(self.parameters(), **self.conf.train.optim_conf)
        return optimizer

    def step(
        self, batch: Tuple[Dict[str, Any], Dict[str, Any]], batch_index: int, mode: str
    ) -> torch.Tensor:
        X, y = batch
        output = self.model(X)
        total_loss = self.criterion(output, y)

        if mode != "train":
            self.validation(logits=output, labels=y)

        # logging
        on_step = mode == "train"
        on_epoch = mode != "train"
        self.log(
            f"{mode}/total_loss",
            total_loss,
            on_step=on_step,
            on_epoch=on_epoch,
            sync_dist=True,
        )
        return total_loss

    def training_step(self, batch, batch_index):  # type: ignore
        return self.step(batch, batch_index, "train")

    def validation_step(self, batch, batch_index):  # type: ignore
        return self.step(batch, batch_index, "dev")

    def validation(self, logits, labels):  # type: ignore
        pred = torch.argmax(logits, dim=-1)
        self.eval_acc.update(len(logits), torch.sum(pred == labels).item())

    def validation_epoch_end(self, outputs):
        total_eval_acc = self.eval_acc.compute()
        self.eval_acc.reset()
        self.log(
            f"dev/acc", total_eval_acc, on_step=False, on_epoch=True, sync_dist=True
        )

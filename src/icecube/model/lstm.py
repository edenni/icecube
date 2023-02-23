import logging
from math import sqrt
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy

from icecube.metrics.angle import MeanAngularError
from icecube.model.base import Model
from icecube.utils.coordinate import (bins2angles, create_angle_bins,
                                      create_bins)

logger = logging.getLogger(__name__)


class LSTMClassifier(LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        bias: bool = False,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        optimizer: Callable = None,
        scheduler: Callable = None,
    ):
        super(LSTMClassifier, self).__init__()
        self.save_hyperparameters()
        self.num_bins = int(sqrt(output_size))

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # TODO: create multiple fc layers
        self.linear = nn.Linear(hidden_size, output_size)

        # TODO: build criterion from config
        self.criterion = nn.CrossEntropyLoss()

        # Metircs
        self.acc = Accuracy(task="multiclass", num_classes=output_size)
        self.mae = MeanAngularError()

        # Create bins
        azimuth_bins, zenith_bins = create_bins(self.num_bins)
        azimuth_bins = torch.as_tensor(azimuth_bins)
        zenith_bins = torch.as_tensor(zenith_bins)

        self.angle_bins = torch.as_tensor(create_angle_bins(
            azimuth_bins, zenith_bins, self.num_bins
        ))

    def forward(self, x):
        # lstm_out = (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1] if self.hparams.batch_first else lstm_out[-1]
        y_pred = self.linear(last)
        return y_pred

    def training_step(self, batch, batch_idx):
        x, _, y_oh = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y_oh)
        self.log("train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_oh = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y_oh)
        self.acc(y_hat, y_oh)
        self.log("val/loss", loss)

        azimuth, zenith = bins2angles(y_hat, self.angle_bins, self.num_bins)
        preds = torch.stack([azimuth, zenith], axis=-1)
        self.mae(preds, y)

        return loss
    

    def on_validation_start(self) -> None:
        if self.angle_bins.device != self.device:
            logger.info(
                f"Start validation. Move angle bin vertors to <{self.device}>"
            )
            self.angle_bins = self.angle_bins.to(self.device)    

    def on_validation_epoch_end(self) -> None:
        acc = self.acc.compute()
        mae = self.mae.compute()
        self.log("val/acc", acc)
        self.log("val/mae", mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

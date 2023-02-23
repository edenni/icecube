import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from icecube.model.base import Model

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
        optimizer: optim.Optimizer = None,
        scheduler: optim.lr_scheduler._LRScheduler = None,
        *args,
        **kwargs,
    ):
        super(LSTMClassifier, self).__init__(args, kwargs)
        self.save_hyperparameters()

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

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1] if self.hparams.batch_first else lstm_out[-1]
        y_pred = self.linear(last)
        return y_pred

    def _shared_step(self, x, target):
        preds = self(x)
        loss = self.criterion(preds, target)

        return loss

    def training_step(self, batch, batch_idx):
        x, target = batch
        loss = self._shared_step(x, target)

        return loss

    def validation_step(self, batch, batch_idx):
        x, target = batch
        loss = self._shared_step(x, target)

        return loss

    def configure_optimizers(self):
        # TODO: build optimizer from config
        return optim.Adam(self.parameters(), lr=self.hparams.lr)

import logging
from math import sqrt
from typing import Callable

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

from icecube.metrics.angle import MeanAngularError
from icecube.utils.coordinate import (
    bins2angles,
    create_angle_bins,
    create_bins,
)

logger = logging.getLogger(__name__)


class LSTM(LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_bins: int,
        num_layers: int = 1,
        bias: bool = False,
        batch_first: bool = True,
        dropout: float = 0,
        bidirectional: bool = False,
        optimizer: Callable = None,
        scheduler: Callable = None,
        criterion: nn.Module = None,
        task: str = "clf",
        net_name: str = "lstm",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # Checks
        assert task in (
            "clf",
            "rgr",
        ), f"Task must be one of clf or rgr but got {task}"

        net_name = net_name.upper()
        assert net_name in ("GRU", "LSTM"), "Inner model should be either LSTM or GRU"

        output_size = num_bins**2

        net_class = getattr(nn, net_name)

        self.lstm = net_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        fc_input = hidden_size * 2 if bidirectional else hidden_size
        self.prj = nn.Linear(fc_input, 256)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(256, output_size)

        self.criterion = criterion

        self.mae = MeanAngularError()

        if task == "clf":
            self.acc = Accuracy(task="multiclass", num_classes=output_size)
            azimuth_bins, zenith_bins = create_bins(self.hparams.num_bins)
            azimuth_bins = torch.as_tensor(azimuth_bins)
            zenith_bins = torch.as_tensor(zenith_bins)

            self.angle_bins = torch.as_tensor(
                create_angle_bins(
                    azimuth_bins, zenith_bins, self.hparams.num_bins
                )
            )

    def forward(self, x, lengths=None):
        # lstm_out = (batch_size, seq_len, hidden_size)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=self.hparams.batch_first,
                enforce_sorted=False,
            )
        lstm_out, _ = self.lstm(x)

        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )

        last = lstm_out[:, -1] if self.hparams.batch_first else lstm_out[-1]
        y_pred = self.fc(self.relu(self.prj(last)))
        return y_pred

    def _shared_step(self, x, y, y_code, lengths=None):
        y_hat = self(x, lengths)

        if self.hparams.task == "clf":
            loss = self.criterion(y_hat, y_code)
        else:
            loss = self.criterion(y_hat, y)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        x, y, y_code = batch
        loss, _ = self._shared_step(x, y, y_code)
        self.log("train/loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_code = batch
        loss, y_hat = self._shared_step(x, y, y_code)
        self.log("val/loss", loss)

        if self.hparams.task == "clf":
            self.acc(y_hat, y_code)
            azimuth, zenith = bins2angles(
                y_hat.softmax(dim=-1),
                self.angle_bins,
                self.hparams.num_bins,
            )
            y_hat = torch.stack([azimuth, zenith], axis=-1)
        self.mae(y_hat, y)

        return loss

    def on_validation_start(self) -> None:
        if (
            self.angle_bins.device != self.device
            and self.hparams.task == "clf"
        ):
            logger.info(
                f"Start validation. Move angle bin vertors to <{self.device}>"
            )
            self.angle_bins = self.angle_bins.to(self.device)

    def on_validation_epoch_end(self) -> None:
        if self.hparams.task == "clf":
            acc = self.acc.compute()
            self.log("val/acc", acc)
        mae = self.mae.compute()
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

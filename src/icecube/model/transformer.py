import logging
from math import sqrt
from typing import Callable

import numpy as np
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


class TransformerEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, feedforward_dim):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.linear_1 = torch.nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = torch.nn.Linear(feedforward_dim, embed_dim)
        self.layernorm_1 = torch.nn.LayerNorm(embed_dim)
        self.layernorm_2 = torch.nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        attn_out, _ = self.attn(x_in, x_in, x_in)
        x = self.layernorm_1(x_in + attn_out)
        ff_out = self.linear_2(torch.nn.functional.relu(self.linear_1(x)))
        x = self.layernorm_2(x + ff_out)
        return x


class TransformerAutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_inputs,
        num_subspaces=8,
        embed_dim=128,
        num_heads=8,
        dropout=0,
        feedforward_dim=512,
        emphasis=0.75,
        mask_loss_weight=2,
    ):
        super().__init__()
        self.num_subspaces = num_subspaces
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.stem = nn.Linear(in_features=num_inputs, out_features=embed_dim)

        self.encoder_1 = TransformerEncoder(
            embed_dim, num_heads, dropout, feedforward_dim
        )
        self.encoder_2 = TransformerEncoder(
            embed_dim, num_heads, dropout, feedforward_dim
        )
        self.encoder_3 = TransformerEncoder(
            embed_dim, num_heads, dropout, feedforward_dim
        )

    def forward(self, x):
        x = torch.nn.functional.relu(self.stem(x))

        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        return torch.concat([x1, x2, x3], dim=-1)


class AddPositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_time: int,
        times,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_time = max_time
        positional_encoding_weight: torch.Tensor = self._initialize_weight()
        self.register_buffer(
            "positional_encoding_weight", positional_encoding_weight
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.positional_encoding_weight[:seq_len, :].unsqueeze(0)

    def _get_positional_encoding(self, pos: int, i: int) -> float:
        w = pos / (10000 ** (((2 * i) // 2) / self.embed_dim))
        if i % 2 == 0:
            return np.sin(w)
        else:
            return np.cos(w)

    def _initialize_weight(self) -> torch.Tensor:
        positional_encoding_weight = [
            [
                self._get_positional_encoding(pos, i)
                for i in range(1, self.embed_dim + 1)
            ]
            for pos in range(1, self.max_time + 1)
        ]
        return torch.tensor(positional_encoding_weight).float()


class TransformerClassifier(LightningModule):
    def __init__(
        self,
        input_size: int,
        embed_dim: int,
        output_size: int,
        nhead: int,
        feedforward_dim: int,
        batch_first: bool = True,
        dropout: float = 0.1,
        task: str = "rgr",
        optimizer: Callable = None,
        scheduler: Callable = None,
        criterion: nn.Module = None,
    ):
        super(TransformerClassifier, self).__init__()
        self.save_hyperparameters(ignore=["criterion"])
        self.num_bins = int(sqrt(output_size))

        self.transformer = TransformerAutoEncoder(
            num_inputs=input_size,
            embed_dim=embed_dim,
            num_heads=nhead,
            dropout=dropout,
            feedforward_dim=feedforward_dim,
        )

        # TODO: create multiple fc layers
        self.linear = nn.Linear(3 * embed_dim * 128, output_size)

        # TODO: build criterion from config
        self.criterion = criterion

        # Metircs
        self.mae = MeanAngularError()

        # Create bins
        if self.hparams.task == "clf":
            self.acc = Accuracy(task="multiclass", num_classes=output_size)
            azimuth_bins, zenith_bins = create_bins(self.num_bins)
            azimuth_bins = torch.as_tensor(azimuth_bins)
            zenith_bins = torch.as_tensor(zenith_bins)

            self.angle_bins = torch.as_tensor(
                create_angle_bins(azimuth_bins, zenith_bins, self.num_bins)
            )

    def forward(self, x):
        x = self.transformer(x)
        x = x.flatten(1)
        y_pred = self.linear(x)
        return y_pred

    def training_step(self, batch, batch_idx):
        if self.hparams.task == "clf":
            x, _, y = batch
        elif self.hparams.task == "rgr":
            x, y, _ = batch

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y, y_oh = batch
        y_hat = self(x)

        if self.hparams.task == "clf":
            loss = self.criterion(y_hat, y_oh)
        elif self.hparams.task == "rgr":
            loss = self.criterion(y_hat, y)

        self.log("val/loss", loss)

        if self.hparams.task == "clf":
            self.acc(y_hat, y_oh)

            azimuth, zenith = bins2angles(
                y_hat, self.angle_bins, self.num_bins
            )
            y_hat = torch.stack([azimuth, zenith], axis=-1)

        self.mae(y_hat, y)

        return loss

    def on_validation_start(self) -> None:
        if (
            self.hparams.task == "clf"
            and self.angle_bins.device != self.device
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

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data


class Model(LightningModule, ABC):
    """Base model class with `Trainer` integrated.
    """

    @abstractmethod
    def forward(
        self, x: Union[torch.Tensor, Data]
    ) -> Union[torch.Tensor, Data]:
        """Forward pass."""

    def _construct_trainers(
        self,
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        logger: Optional[Logger] = None,
        ckpt_path :Optional[str] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        if gpus:
            accelerator = "gpu"
            devices = gpus
        else:
            accelerator = "cpu"
            devices = None

        self._trainer = Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            logger=logger,
            gradient_clip_val=gradient_clip_val,
            strategy=distribution_strategy,
            resume_from_checkpoint=ckpt_path,
            **trainer_kwargs,
        )

    def fit(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        datamodule: Optional[LightningDataModule] = None,
        *,
        max_epochs: int = 10,
        gpus: Optional[Union[List[int], int]] = None,
        callbacks: Optional[List[Callback]] = None,
        ckpt_path: Optional[str] = None,
        logger: Optional[Logger] = None,
        log_every_n_steps: int = 1,
        gradient_clip_val: Optional[float] = None,
        distribution_strategy: Optional[str] = "ddp",
        **trainer_kwargs: Any,
    ) -> None:
        """Fit `Model` using `pytorch_lightning.Trainer`."""
        self.train(mode=True)

        self._construct_trainers(
            max_epochs=max_epochs,
            gpus=gpus,
            callbacks=callbacks,
            ckpt_path=ckpt_path,
            logger=logger,
            log_every_n_steps=log_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            distribution_strategy=distribution_strategy,
            **trainer_kwargs,
        )

        try:
            self._trainer.fit(
                self, train_dataloader, val_dataloader, datamodule
            )
        except KeyboardInterrupt:
            self.warning("[ctrl+c] Exiting gracefully.")
            pass

    def predict(
        self,
        dataloader: DataLoader,
    ) -> List[torch.Tensor]:
        """Return predictions for `dataloader`.

        Returns a list of Tensors, one for each model output.
        """
        self.train(mode=False)

        preds = self._trainer.predict(self, dataloader)
        assert len(preds), "Got no predictions"

        # TODO: Confirm the ouput shape
        return preds
    

    @property
    def best_model(self):
        return self._trainer.best_model

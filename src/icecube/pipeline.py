from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger

from icecube.data.utils import create_dataloaders
from icecube.model import Model


class Pipeline:
    def __init__(self, cfg: DictConfig) -> None:
        pass


def fit_one_cycle(cfg: DictConfig) -> str:
    """Fitting model one cycle.

    Returns:
        str: Best model path.
    """
    dm: LightningDataModule = instantiate(cfg.data)
    model: LightningModule = instantiate(cfg.model)

    callbacks: List[Callback] = (
        [
            instantiate(callback_cfg)
            for _, callback_cfg in cfg.callbacks.items()
        ]
        if cfg.get("callbacks")
        else None
    )
    logger: List[Logger] = (
        [instantiate(logger_cfg) for _, logger_cfg in cfg.logger.items()]
        if cfg.get("logger")
        else None
    )

    trainer: Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    dm.setup("train")
    trainer.fit(model=model, datamodule=dm)

    return [cb.best_model_path for cb in callbacks if isinstance(cb, ModelCheckpoint)][0]

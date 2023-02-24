from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger

from icecube.data.utils import create_dataloaders
from icecube.utils.logging import flatten_dict

def one_cycle(cfg: DictConfig, num_folds: int = None, k: int = None) -> str:
    """Fitting model one cycle.

    Returns:
        str: Best model path.
    """
    dm: LightningDataModule = instantiate(cfg.data, num_folds=num_folds, k=k)
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

    # Retrieve wandb logger
    wandb_logger = [l for l in logger if isinstance(l, WandbLogger)]
    wandb_logger: WandbLogger = wandb_logger[0] if len(wandb_logger) > 0 else None
    wandb_logger.experiment.config.update(flatten_dict(cfg))
    wandb_logger.experiment.watch(model)

    trainer: Trainer = instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    dm.setup("train")
    trainer.fit(model=model, datamodule=dm)

    wandb_logger.experiment.finish()

    return [
        cb.best_model_path
        for cb in callbacks
        if isinstance(cb, ModelCheckpoint)
    ][0]


def cv(cfg: DictConfig) -> None:
    assert cfg.num_folds > 0

def train(cfg):
    if cfg.cv and isinstance(cfg.cv, int):
        one_cycle(cfg, cfg.cv, cfg.fold)
    else:
        one_cycle(cfg)
      


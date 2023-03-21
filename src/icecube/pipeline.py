from typing import List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger

from icecube.data.utils import create_dataloaders
from icecube.utils.logging import flatten_dict


def one_cycle(
    cfg: DictConfig, num_folds: int = None, k: int = None
) -> Optional[str]:
    """Fitting model one cycle.

    Returns:
        str: Best model path.
    """
    dm: LightningDataModule = instantiate(cfg.data, num_folds=num_folds, k=k)
    model: LightningModule = instantiate(cfg.model)

    callbacks: Optional[List[Callback]] = (
        [
            instantiate(callback_cfg)
            for _, callback_cfg in cfg.callbacks.items()
        ]
        if cfg.get("callbacks")
        else None
    )
    logger: Optional[WandbLogger] = (
        instantiate(cfg.logger) if cfg.get("logger") else None
    )

    # Retrieve wandb logger
    if isinstance(logger, WandbLogger):
        logger.experiment.config.update(flatten_dict(cfg))
        logger.experiment.watch(model, log="all")

    trainer: Trainer = instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    trainer.fit(model=model, datamodule=dm)

    logger.experiment.finish()

    if trainer.checkpoint_callback:
        return trainer.checkpoint_callback.best_model_path
    else:
        return None


def train(cfg):
    if cfg.cv and isinstance(cfg.cv, int):
        one_cycle(cfg, cfg.cv, cfg.fold)
    else:
        one_cycle(cfg)

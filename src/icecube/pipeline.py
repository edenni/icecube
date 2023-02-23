from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig
from icecube.model import Model
from icecube.data.utils import create_dataloaders
from pytorch_lightning import LightningDataModule

class Pipeline:
    def __init__(self, cfg: DictConfig) -> None:
        pass


def fit_one_cycle(cfg: DictConfig) -> Path:
    """Fitting model one cycle.

    Returns:
        Path: Best model path.
    """
    dm: LightningDataModule = instantiate(cfg.data)
    dm.setup()
    # model: Model = instantiate(cfg.model)

    # model.fit(dataloaders[0], dataloaders[1], **cfg.trainer)

    # return model.best_model

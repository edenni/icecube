import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from graphnet.utilities.logging import get_logger
from icecube.constants import *

logger = get_logger(log_folder=log_dir)


import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from sklearn.model_selection import KFold
from torch.optim import SGD
from tqdm import tqdm

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCubeKaggle
from graphnet.models.gnn import DynEdge
from graphnet.models.graph_builders import KNNGraphBuilder
from graphnet.models.task.reconstruction import \
    DirectionReconstructionWithKappa
from graphnet.training.callbacks import PiecewiseLinearLR, ProgressBar
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss
from graphnet.training.utils import make_dataloader

torch.set_float32_matmul_precision("medium")


PULSEMAP = "pulse_table"
# DATABASE_PATH = database_dir / "batch_51_100.db"
DATABASE_PATH = Path("/media/eden/sandisk/projects/icecube/input/sqlite/batch_51_100.db")
# DATABASE_PATH = "/media/eden/sandisk/projects/icecube/input/sqlite/batch_1.db"
PULSE_THRESHOLD = 200
SEED = 42

# Training configs
MAX_EPOCHS = 100
LR = 1e-2
MOMENTUM = 0.87
BS = 1024
ES = 10
NUM_FOLDS = 5
NUM_WORKERS = 4

# Paths
FOLD_PATH = input_dir / "folds"
COUNT_PATH = FOLD_PATH / "batch51_100_counts.csv"
CV_PATH = FOLD_PATH / f"batch51_100_cv_max_{PULSE_THRESHOLD}_pulses.csv"
WANDB_DIR = log_dir
PROJECT_NAME = "icecube"
GROUP_NAME = "pretrain_batch_51_100_np_200"

CREATE_FOLDS = False


config = {
    "path": "/media/eden/sandisk/projects/icecube/input/sqlite/batch_51_100.db",
    "pulsemap": "pulse_table",
    "truth_table": "meta_table",
    "features": FEATURES.KAGGLE,
    "truth": TRUTH.KAGGLE,
    "index_column": "event_id",
    "batch_size": BS,
    "num_workers": NUM_WORKERS,
    "target": "direction",
    "run_name_tag": "batch_1_50",
    "early_stopping_patience": ES,
    "fit": {
        "max_epochs": MAX_EPOCHS,
        "gpus": [0],
        "distribution_strategy": None,
        "limit_train_batches": 1.0,  # debug
        "limit_val_batches": 1.0,
        "precision": 16,
    },
    "base_dir": "training",
    "wandb": {
        "project": PROJECT_NAME,
        "group": GROUP_NAME,
    },
    "lr": LR,
}


def build_model(
    config: Dict[str, Any], train_dataloader: Any
) -> StandardModel:
    """Builds GNN from config"""
    # Building model
    detector = IceCubeKaggle(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
        global_pooling_schemes=["min", "max", "mean"],
        dynedge_layer_sizes = [
                (
                    128,
                    256,
                ),
                (
                    336,
                    256,
                ),
                (
                    336,
                    256,
                    256,
                ),
                (
                    336,
                    256,
                    256,
                ),
            ]
    )
    gnn._activation = torch.nn.Mish()

    if config["target"] == "direction":
        task = DirectionReconstructionWithKappa(
            hidden_size=gnn.nb_outputs,
            target_labels=config["target"],
            loss_function=VonMisesFisher3DLoss(),
        )
        prediction_columns = [
            config["target"] + "_x",
            config["target"] + "_y",
            config["target"] + "_z",
            config["target"] + "_kappa",
        ]
        additional_attributes = ["zenith", "azimuth", "event_id"]

    model = StandardModel(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        optimizer_class=SGD,
        optimizer_kwargs={
            "lr": LR,
            "momentum": MOMENTUM,
            "nesterov": True,
            # "weight_decay": 1e-4,
        },
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader) / 2,
                len(train_dataloader) * config["fit"]["max_epochs"] / 10,
                len(train_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-03, 1, 1e-01, 1e-04],
        },
        scheduler_config={
            "interval": "step",
        },
    )
    model.prediction_columns = prediction_columns
    model.additional_attributes = additional_attributes

    return model


def load_pretrained_model(
    config: Dict[str, Any],
    state_dict_path: str = "/kaggle/input/dynedge-pretrained/dynedge_pretrained_batch_1_to_50/state_dict.pth",
) -> StandardModel:
    train_dataloader, _ = make_dataloaders(config=config)
    model = build_model(config=config, train_dataloader=train_dataloader)
    # model._inference_trainer = Trainer(config['fit'])
    state_dict = torch.load(state_dict_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.prediction_columns = [
        config["target"] + "_x",
        config["target"] + "_y",
        config["target"] + "_z",
        config["target"] + "_kappa",
    ]
    model.additional_attributes = ["zenith", "azimuth", "event_id"]
    return model


def make_dataloaders(config: Dict[str, Any], fold: int = 0) -> List[Any]:
    """Constructs training and validation dataloaders for training with early stopping."""

    train_idx = pd.read_csv("/media/eden/sandisk/projects/icecube/input/folds/batch51_100_train.csv")[config["index_column"]].to_list()
    val_idx = pd.read_csv("/media/eden/sandisk/projects/icecube/input/folds/batch51_100_test.csv")[config["index_column"]].to_list()

    # df_cv = pd.read_csv(CV_PATH)

    # val_idx = (
    #     df_cv[df_cv["fold"] == fold][config["index_column"]].ravel().tolist()
    # )
    # train_idx = (
    #     df_cv[~df_cv["fold"].isin([-1, fold])][config["index_column"]]
    #     .ravel()
    #     .tolist()
    # )

    logger.info(f"training samples: {len(train_idx)}")
    logger.info(f"val samples: {len(val_idx)}")

    train_dataloader = make_dataloader(
        db=config["path"],
        selection=train_idx,
        pulsemaps=config["pulsemap"],
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        labels={"direction": Direction()},
        index_column=config["index_column"],
        truth_table=config["truth_table"],
    )

    validate_dataloader = make_dataloader(
        db=config["path"],
        selection=val_idx,
        pulsemaps=config["pulsemap"],
        features=FEATURES.KAGGLE,
        truth=TRUTH.KAGGLE,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        labels={"direction": Direction()},
        index_column=config["index_column"],
        truth_table=config["truth_table"],
    )

    return train_dataloader, validate_dataloader


def train_dynedge(
    config: Dict[str, Any], fold: int = 0, resume: Path = None
) -> pd.DataFrame:
    """Builds(or resumes) and trains GNN according to config."""
    logger.info(f"features: {config['features']}")
    logger.info(f"truth: {config['truth']}")

    run_name = (
        f"dynedge_{config['target']}_{config['run_name_tag']}_np200_aux0.8_h256"
    )

    wandb_logger = WandbLogger(
        project=PROJECT_NAME,
        group=GROUP_NAME,
        name=run_name,
        save_dir=WANDB_DIR,
        log_model=True,
    )
    wandb_logger.experiment.config.update(config)

    train_dataloader, validate_dataloader = make_dataloaders(
        config=config, fold=fold
    )

    if not resume:
        model = build_model(config, train_dataloader)
    else:
        model = load_pretrained_model(config, state_dict_path=resume)

    # wandb_logger.experiment.watch(model, log="all")

    # Training model
    callbacks = [
        EarlyStopping(
            monitor="val/mae",
            patience=config["early_stopping_patience"],
        ),
        LearningRateMonitor(logging_interval="step"),
        ProgressBar(),
        ModelCheckpoint(
            filename="graphnet-{val/mae:.4f}-{epoch:02d}",
            monitor="val/mae",
            mode="min",
        ),
    ]

    profiler = AdvancedProfiler(dirpath=log_dir, filename="profile")

    model.fit(
        train_dataloader,
        validate_dataloader,
        callbacks=callbacks,
        logger=wandb_logger,
        profiler=profiler,
        **config["fit"],
    )

    wandb_logger.experiment.save(str(profiler.dirpath / profiler.filename))
    wandb_logger.experiment.finish()

    return model


def convert_to_3d(df: pd.DataFrame) -> pd.DataFrame:
    """Converts zenith and azimuth to 3D direction vectors"""
    df["true_x"] = np.cos(df["azimuth"]) * np.sin(df["zenith"])
    df["true_y"] = np.sin(df["azimuth"]) * np.sin(df["zenith"])
    df["true_z"] = np.cos(df["zenith"])
    return df


def calculate_angular_error(df: pd.DataFrame) -> pd.DataFrame:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    df["angular_error"] = np.arccos(
        df["true_x"] * df["direction_x"]
        + df["true_y"] * df["direction_y"]
        + df["true_z"] * df["direction_z"]
    )
    return df



if __name__ == "__main__":
    train_dynedge(
        config=config,
        # resume="/media/eden/sandisk/projects/icecube/models/graphnet/ft-epoch=52-mae=1.0905.ckpt",
    )
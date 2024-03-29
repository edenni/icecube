# %%
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
from pytorch_lightning.profiler import SimpleProfiler
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

torch.set_float32_matmul_precision("high")


PULSEMAP = "pulse_table"
DATABASE_PATH = database_dir / "batch_51_100.db"
# DATABASE_PATH = "/media/eden/sandisk/projects/icecube/input/sqlite/batch_1.db"
PULSE_THRESHOLD = 400
SEED = 42

# Training configs
MAX_EPOCHS = 100
LR = 5e-4
MOMENTUM = 0.9
BS = 256
ES = 10
NUM_FOLDS = 5
NUM_WORKERS = 16

# Paths
FOLD_PATH = input_dir / "folds"
COUNT_PATH = FOLD_PATH / "batch51_100_counts.csv"
CV_PATH = FOLD_PATH / f"batch51_100_cv_max_{PULSE_THRESHOLD}_pulses.csv"
WANDB_DIR = log_dir
PROJECT_NAME = "icecube"
GROUP_NAME = "pretrain_sub_5_batch_51_100_large_resume"

CREATE_FOLDS = False


def make_selection(
    df: pd.DataFrame, num_folds: int = 5, pulse_threshold: int = 200
) -> None:
    """Creates a validation and training selection (20 - 80). All events in both selections satisfies n_pulses <= 200 by default."""
    n_events = np.arange(0, len(df), 1)
    df["fold"] = 0

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
    for i, (_, val_idx) in enumerate(kf.split(n_events)):
        df.loc[val_idx, "fold"] = i

    # Remove events with large pulses from training and validation sample (memory)
    df["fold"][df["n_pulses"] > pulse_threshold] = -1

    df.to_csv(CV_PATH)
    return


def get_number_of_pulses(db: Path, event_id: int, pulsemap: str) -> int:
    with sqlite3.connect(str(db)) as con:
        query = f"select event_id from {pulsemap} where event_id = {event_id} limit 20000"
        data = con.execute(query).fetchall()
    return len(data)


def count_pulses(database: Path, pulsemap: str) -> pd.DataFrame:
    """Will count the number of pulses in each event and return a single dataframe that contains counts for each event_id."""
    with sqlite3.connect(str(database)) as con:
        query = "select event_id from meta_table"
        events = pd.read_sql(query, con)
    counts = {"event_id": [], "n_pulses": []}

    for event_id in tqdm(events["event_id"]):
        a = get_number_of_pulses(database, event_id, pulsemap)
        counts["event_id"].append(event_id)
        counts["n_pulses"].append(a)

    df = pd.DataFrame(counts)
    df.to_csv(COUNT_PATH)
    return df

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
    )

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
        },
        scheduler_class=PiecewiseLinearLR,
        scheduler_kwargs={
            "milestones": [
                0,
                len(train_dataloader) / 2,
                len(train_dataloader) * config["fit"]["max_epochs"],
            ],
            "factors": [1e-03, 1, 1e-03],
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
    df_cv = pd.read_csv(CV_PATH)

    val_idx = (
        df_cv[df_cv["fold"] == fold][config["index_column"]].ravel().tolist()
    )
    train_idx = (
        df_cv[~df_cv["fold"].isin([-1, fold])][config["index_column"]]
        .ravel()
        .tolist()
    )

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


# %%
config = {
    "inference_database_path": "/media/eden/sandisk/projects/icecube/input/sqlite/test_batch_641_660.db",
    "path": str(DATABASE_PATH),
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
        "limit_train_batches": 0.1,  # debug
        "limit_val_batches": 0.1,
    },
    "base_dir": "training",
    "wandb": {
        "project": PROJECT_NAME,
        "group": GROUP_NAME,
    },
    "lr": LR,
}


# %%
ckpt = "/media/eden/sandisk/projects/icecube/models/graphnet/ft_graphnet_50_100.ckpt"

# %%
run_name = f"dynedge_{config['target']}_{config['run_name_tag']}"

model = load_pretrained_model(config, state_dict_path=ckpt)

test_dataloader = make_dataloader(
    db = config['inference_database_path'],
    selection = None, # Entire database
    pulsemaps = config['pulsemap'],
    features = config["features"],
    truth = config["truth"],
    batch_size = config['batch_size'],
    num_workers = config['num_workers'],
    shuffle = False,
    labels = {'direction': Direction()},
    index_column = config['index_column'],
    truth_table = config['truth_table'],
)


# %%
from torchmetrics import Metric


class MeanAngularError(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "err",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.tensor(0, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.size(1) > target.size(1):
            preds = preds[:, : target.size(1)]

        assert (
            preds.shape == target.shape
        ), f"preds in size {preds.shape} doesn't match target in size {target.shape}"

        rt = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
        rp = torch.linalg.vector_norm(preds, dim=-1, keepdim=True)


        target = target / rt
        preds = preds / rp

        err = torch.arccos(
            target[:, 0] * preds[:, 0]
            + target[:, 1] * preds[:, 1]
            + target[:, 2] * preds[:, 2]
        )

        err = torch.clip(err, 0, 2*torch.pi)

        self.err += err.sum()
        self.total += preds.size(0)

    def compute(self):
        return self.err.item() / self.total.item()


# %%
model = model.to("cuda").eval()
mae = MeanAngularError()

# %%
model.predict_as_dataframe(
    dataloader=test_dataloader,
    gpus=[0],
    prediction_columns=model.prediction_columns,
    additional_attributes=model.additional_attributes,
)

# %%




import gc
import logging
import os
import pickle
from glob import glob
from typing import List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
# import webdataset as wbs
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

from icecube.data.utils import az_onehot
from icecube.utils.coordinate import create_bins

logger = logging.getLogger(__name__)


class EventDataModule(LightningDataModule):
    def __init__(
        self,
        num_bins: int,
        data_dir: str,
        train_files: str,
        val_files: str,
        batch_size: int,
        file_format: str,
        batch_ids: Tuple[int, int],
        num_workers: int = 8,
        val_size: float = 0.05,
        num_folds: int = None,
        k: int = None,
        shift_azimuth: bool = False,
        max_pulse_count: int = 96,
        n_features: int = 6,
    ) -> None:
        super(EventDataModule, self).__init__()

        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        if not stage or stage == "fit":
            # Read data
            logger.info("Setup datamodule")
            logger.info(
                f"Reading picked up points from batches: {self.hparams.batch_ids[0]} to {self.hparams.batch_ids[1]}"
            )

            batch_ids = list(
                range(self.hparams.batch_ids[0], self.hparams.batch_ids[1] + 1)
            )

            N_CHUNK = 200000
            # batch 660 has events less than 200000
            assert 660 not in batch_ids, "Batch 660 is not available"
            X: Optional[np.ndarray] = np.zeros(
                (
                    len(batch_ids) * N_CHUNK,
                    self.hparams.max_pulse_count,
                    self.hparams.n_features,
                ),
                dtype=np.float16,
            )
            y: Optional[np.ndarray] = np.zeros(
                (len(batch_ids) * N_CHUNK, 2), dtype=np.float16
            )

            for i, batch_id in enumerate(batch_ids):
                logger.info(f"Reading batch {batch_id}")
                train_data_file: np.ndarray = np.load(
                    self.hparams.file_format.format(batch_id=batch_id)
                )

                # X = (
                #     train_data_file["x"][:, :, :-1]
                #     if X is None
                #     else np.append(X, train_data_file["x"][:, :, :-1], axis=0)
                # )
                # y = (
                #     train_data_file["y"]
                #     if y is None
                #     else np.append(y, train_data_file["y"], axis=0)
                # )
                X[i * N_CHUNK : (i + 1) * N_CHUNK] = train_data_file["x"][
                    :, :, :-1
                ]
                y[i * N_CHUNK : (i + 1) * N_CHUNK] = train_data_file["y"]

                train_data_file.close()
                del train_data_file
                _ = gc.collect()

            logger.info("Convert azimuth and zenith to one-hot")
            self.azimuth_edges, self.zenith_edges = create_bins(
                self.hparams.num_bins
            )

            # Shift azimuth
            if self.hparams.shift_azimuth:
                y[:, 0] += (self.azimuth_edges[1] - self.azimuth_edges[0]) / 2
                y[:, 0][y[:, 0] > 2 * np.pi] -= 2 * np.pi

            # y_onehot = az_onehot(
            #     y, azimuth_edges, zenith_edges, self.hparams.num_bins
            # )
            y_onehot = self._y_to_angle_code(y)

            # Pre-processing
            logger.info("Standard normalization")
            # original_shape = X.shape
            # scaler = StandardScaler().fit(
            #     X[: min(original_shape[0], 100000)].reshape(
            #         -1, original_shape[-1]
            #     )
            # )
            # X = scaler.transform(X.reshape(-1, original_shape[-1])).reshape(
            #     original_shape
            # )
            X[:, :, 0] /= 1000  # time
            X[:, :, 1] /= 300  # charge
            X[:, :, 3:] /= 600  # space

            # Split dataset
            if self.hparams.num_folds and self.hparams.k:
                logger.info(
                    f"Use fold={self.hparams.k} by cv={self.hparams.num_folds}"
                )
                kf = KFold(n_splits=self.hparams.num_folds, shuffle=True)
                all_splits = list(kf.split(X, y))
                train_idx, val_idx = all_splits[self.hparams.k]
                X_train = X[train_idx]
                y_train = y[train_idx]
                y_onehot_train = y_onehot[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                y_onehot_val = y_onehot[val_idx]
            else:
                logger.info(
                    f"Split data with val_size={self.hparams.val_size}"
                )
                n = len(X)
                n_val = int(n * self.hparams.val_size)
                X_train = X[:-n_val]
                y_train = y[:-n_val]
                X_val = X[-n_val:]
                y_val = y[-n_val:]
                y_onehot_train = y_onehot[:-n_val]
                y_onehot_val = y_onehot[-n_val:]
                # (
                #     X_train,
                #     X_val,
                #     y_train,
                #     y_val,
                #     y_onehot_train,
                #     y_onehot_val,
                # ) = train_test_split(
                #     X,
                #     y,
                #     y_onehot,
                #     test_size=self.hparams.val_size,
                #     shuffle=True,
                # )

            self.trainset = TensorDataset(
                torch.as_tensor(X_train),
                torch.as_tensor(y_train),
                torch.as_tensor(y_onehot_train),
            )
            self.valset = TensorDataset(
                torch.as_tensor(X_val),
                torch.as_tensor(y_val),
                torch.as_tensor(y_onehot_val),
            )

            # self.trainset = (
            #     wbs.WebDataset(self.hparams.train_files)
            #     .map(self._preprocess)
            #     .shuffle(100000)
            # )

            # self.valset = wbs.WebDataset(self.hparams.val_files).map(
            #     self._preprocess
            # )

    def _preprocess(self, src):
        features, truth = pickle.loads(src["pickle"])

        x = np.concatenate(
            [features, np.zeros((features.shape[0], 1))], axis=1
        )
        dtype = [
            ("x", "float16"),
            ("y", "float16"),
            ("z", "float16"),
            ("time", "float16"),
            ("charge", "float16"),
            ("auxiliary", "float16"),
            ("rank", "short"),
        ]

        n_pulses = len(x)
        event_x = np.zeros(n_pulses, dtype)

        event_x["x"] = x[:, 0]
        event_x["y"] = x[:, 1]
        event_x["z"] = x[:, 2]
        event_x["time"] = x[:, 3] - x[:, 3].min()
        event_x["charge"] = x[:, 4]
        event_x["auxiliary"] = x[:, 5]

        if n_pulses > self.hparams.max_pulse_count:
            # Find valid time window
            t_peak = event_x["time"][event_x["time"].argmax()]
            t_valid_min = t_peak - 6199.700247193777
            t_valid_max = t_peak + 6199.700247193777

            t_valid = (event_x["time"] > t_valid_min) * (
                event_x["time"] < t_valid_max
            )

            # rank
            event_x["rank"] = 2 * (1 - event_x["auxiliary"]) + (t_valid)

            # sort by rank and charge (important goes to backward)
            event_x = np.sort(event_x, order=["rank", "charge"])

            # pick-up from backward
            event_x = event_x[-self.hparams.max_pulse_count :]

            # resort by time
            event_x = np.sort(event_x, order="time")

        event_x["x"] /= 600
        event_x["y"] /= 600
        event_x["z"] /= 600
        event_x["time"] /= 1000
        event_x["charge"] /= 300

        event_y = truth.astype(dtype="float16")
        code = self._y_to_angle_code(event_y[:, ::-1])[0]

        placeholder = np.zeros(
            (self.hparams.max_pulse_count, 6), dtype=np.float16
        )
        placeholder[:n_pulses, 0] = event_x["x"]
        placeholder[:n_pulses, 1] = event_x["y"]
        placeholder[:n_pulses, 2] = event_x["z"]
        placeholder[:n_pulses, 3] = event_x["time"]
        placeholder[:n_pulses, 4] = event_x["charge"]
        placeholder[:n_pulses, 5] = event_x["auxiliary"]

        y = np.zeros(2, dtype=np.float16)
        y[0] = event_y[0, 1]
        y[1] = event_y[0, 0]

        return placeholder, y, code, np.clip(len(event_x), a_min=0, a_max=96)

    def _y_to_angle_code(self, y):
        azimuth_code = (y[:, 0] > self.azimuth_edges[1:].reshape((-1, 1))).sum(
            axis=0
        )
        zenith_code = (y[:, 1] > self.zenith_edges[1:].reshape((-1, 1))).sum(
            axis=0
        )
        angle_code = self.hparams.num_bins * azimuth_code + zenith_code

        return angle_code

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.valset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

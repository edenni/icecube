import gc
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from icecube.data.utils import az_onehot
from icecube.utils.coordinate import create_bins

logger = logging.getLogger(__name__)


class EventDataModule(LightningDataModule):
    def __init__(
        self,
        num_bins: int,
        data_dir: str,
        batch_ids: List[int],
        file_format: str,
        batch_size: int,
        num_workers: int = 8,
        val_size: float = 0.05,
        num_folds: int = None,
        k: int = None,
        shift_azimuth: bool = False,
    ) -> None:
        super(EventDataModule, self).__init__()

        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        if not stage or stage == "fit":
            # Read data
            logger.info(
                f"Reading picked up points from batches: {self.hparams.batch_ids}"
            )

            X: Optional[np.ndarray] = None
            y: Optional[np.ndarray] = None

            for batch_id in self.hparams.batch_ids:
                logger.info(f"Reading batch {batch_id}")
                train_data_file: np.ndarray = np.load(
                    self.hparams.file_format.format(batch_id=batch_id)
                )

                X = (
                    train_data_file["x"]
                    if X is None
                    else np.append(X, train_data_file["x"], axis=0)
                )
                y = (
                    train_data_file["y"]
                    if y is None
                    else np.append(y, train_data_file["y"], axis=0)
                )

                train_data_file.close()
                del train_data_file
                _ = gc.collect()

            logger.info("Convert azimuth and zenith to one-hot")
            azimuth_edges, zenith_edges = create_bins(self.hparams.num_bins)

            # Shift azimuth
            if self.hparams.shift_azimuth:
                y[:, 0] += azimuth_edges[1] - azimuth_edges[0]
                y[:, 0][y[:, 0] > 2 * np.pi] -= 2 * np.pi

            y_onehot = az_onehot(
                y, azimuth_edges, zenith_edges, self.hparams.num_bins
            )

            # Pre-processing
            logger.info("Standard normalization")
            from time import time
            start = time()
            original_shape = X.shape
            scaler = StandardScaler().fit(
                X[: min(original_shape[0], 100000)].reshape(
                    -1, original_shape[-1]
                )
            )
            logging.info(f"Fitting toke {time()-start}s")
            X = scaler.transform(X.reshape(-1, original_shape[-1])).reshape(
                original_shape
            )

            # Split dataset
            if self.hparams.num_folds and self.hparams.k:
                logger.info(f"Use fold={self.hparams.k} by cv={self.hparams.num_folds}")
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
                logger.info(f"Split data with val_size={self.hparams.val_size}")
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    y_onehot_train,
                    y_onehot_val,
                ) = train_test_split(
                    X,
                    y,
                    y_onehot,
                    test_size=self.hparams.val_size,
                    shuffle=True,
                )

            self.trainset = TensorDataset(
                torch.as_tensor(X_train),
                torch.as_tensor(y_train),
                torch.as_tensor(y_onehot_train),
            )
            self.validset = TensorDataset(
                torch.as_tensor(X_val),
                torch.as_tensor(y_val),
                torch.as_tensor(y_onehot_val),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

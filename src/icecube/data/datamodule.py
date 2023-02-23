import gc
import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
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
    ) -> None:
        super(EventDataModule, self).__init__()

        self.num_bins = num_bins
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.file_format = file_format
        self.batch_ids = batch_ids
        self.num_workers = num_workers
        self.val_size = val_size

    def setup(self, stage: Optional[str] = None) -> None:
        if not stage or stage == "train":
            # Read data
            logger.info(
                f"Reading picked up points from batches: {self.batch_ids}"
            )

            X_train = None
            y_train = None
            for batch_id in self.batch_ids:
                logger.info(f"Reading batch {batch_id}")
                train_data_file = np.load(
                    self.file_format.format(batch_id=batch_id)
                )

                X_train = (
                    train_data_file["x"]
                    if X_train is None
                    else np.append(X_train, train_data_file["x"], axis=0)
                )
                y_train = (
                    train_data_file["y"]
                    if y_train is None
                    else np.append(y_train, train_data_file["y"], axis=0)
                )

                train_data_file.close()
                del train_data_file
                _ = gc.collect()

            # Pre-processing
            logger.info("Convert azimuth and zenith to one-hot")
            azimuth_edges, zenith_edges = create_bins(self.num_bins)
            y_onehot = az_onehot(
                y_train, azimuth_edges, zenith_edges, self.num_bins
            )

            logger.info("Stardard normalization")
            original_shape = X_train.shape
            scaler = StandardScaler().fit(
                X_train[: min(original_shape[0], 100000)].reshape(
                    -1, original_shape[-1]
                )
            )
            X_train = scaler.transform(
                X_train.reshape(-1, original_shape[-1])
            ).reshape(original_shape)

            # Split dataset
            (
                X_train,
                X_val,
                y_train,
                y_val,
                y_onehot_train,
                y_onehot_val,
            ) = train_test_split(
                X_train,
                y_train,
                y_onehot,
                test_size=self.val_size,
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
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

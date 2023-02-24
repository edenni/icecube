import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def create_dataloaders(cfg) -> Tuple[DataLoader, ...]:
    ...


def az_onehot(
    az: np.ndarray,
    azimuth_edges: np.ndarray,
    zenith_edges: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    # evaluate bin code
    azimuth_code = (az[:, 0] > azimuth_edges[1:].reshape((-1, 1))).sum(
        axis=0
    )
    zenith_code = (az[:, 1] > zenith_edges[1:].reshape((-1, 1))).sum(
        axis=0
    )
    angle_code = num_bins * azimuth_code + zenith_code

    # one-hot
    az_oh = np.zeros((angle_code.size, num_bins * num_bins))
    az_oh[np.arange(angle_code.size), angle_code] = 1

    return az_oh


def read_processed_data(data_dir, batch_ids):
    X_train = None
    y_train = None
    for batch_id in batch_ids:
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
from typing import Tuple

from torch.utils.data import DataLoader
import numpy as np
from functools import lru_cache

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

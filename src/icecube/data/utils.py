from typing import Tuple

from torch.utils.data import DataLoader
import numpy as np
from functools import lru_cache

def create_dataloaders(cfg) -> Tuple[DataLoader, ...]:
    ...

@lru_cache(1)
def create_bins(num_bins) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create bins to bucketize azimuth and zenith.
    Azimuth is in uniform and zenith sinusoidal.
    """
    azimuth_edges = np.linspace(0, 2 * np.pi, num_bins + 1)

    # zenith_edges_flat = np.linspace(0, np.pi, self.num_bins + 1)
    zenith_edges = list()
    zenith_edges.append(0)
    for bin_idx in range(1, num_bins):
        zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / num_bins)
        zenith_edges.append(zen_now)
    zenith_edges.append(np.pi)
    zenith_edges = np.array(zenith_edges)

    return azimuth_edges, zenith_edges


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

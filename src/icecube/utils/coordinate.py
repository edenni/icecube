from functools import lru_cache
from typing import Tuple, Union

import numpy as np
import torch


def az2xyz(
    azimuth: np.ndarray, zenith: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts zenith and azimuth to 3D direction vectors"""
    x = np.cos(azimuth) * np.sin(zenith)
    y = np.sin(azimuth) * np.sin(zenith)
    z = np.cos(zenith)
    return x, y, z


@lru_cache(1)
def create_bins(
    num_bins,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Create bins to bucketize azimuth and zenith.
    Azimuth is in uniform and zenith sinusoidal.
    """
    azimuth_edges = np.linspace(0, 2 * torch.pi, num_bins + 1)

    # zenith_edges_flat = np.linspace(0, np.pi, self.num_bins + 1)
    zenith_edges = list()
    zenith_edges.append(0)
    
    for _ in range(1, num_bins):
        zen_now = np.arccos(np.cos(zenith_edges[-1]) - 2 / num_bins)
        zenith_edges.append(zen_now)
    zenith_edges.append(np.pi)

    zenith_edges = np.array(zenith_edges)

    return azimuth_edges, zenith_edges


@lru_cache(1)
def create_angle_bins(
    azimuth_edges: np.ndarray,
    zenith_edges: np.ndarray,
    num_bins: int,
) -> np.ndarray:

    angle_bin_zenith0 = np.tile(zenith_edges[:-1], num_bins)
    angle_bin_zenith1 = np.tile(zenith_edges[1:], num_bins)
    angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], num_bins)
    angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], num_bins)

    angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * (
        np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1)
    )
    angle_bin_vector_sum_x = (
        np.sin(angle_bin_azimuth1) - np.sin(angle_bin_azimuth0)
    ) * (
        (angle_bin_zenith1 - angle_bin_zenith0) / 2
        - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0))
        / 4
    )
    angle_bin_vector_sum_y = (
        np.cos(angle_bin_azimuth0) - np.cos(angle_bin_azimuth1)
    ) * (
        (angle_bin_zenith1 - angle_bin_zenith0) / 2
        - (np.sin(2 * angle_bin_zenith1) - np.sin(2 * angle_bin_zenith0))
        / 4
    )
    angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * (
        (np.cos(2 * angle_bin_zenith0) - np.cos(2 * angle_bin_zenith1))
        / 4
    )

    angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
    angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
    angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

    angle_bin_vector = np.zeros((1, num_bins * num_bins, 3))
    angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
    angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
    angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z

    return angle_bin_vector


def bins2angles(pred: torch.Tensor, angle_bins, num_bins, epsilon=1e-8):
    # convert prediction to vector
    pred_vector = (
        pred.reshape((-1, num_bins * num_bins, 1)) * angle_bins
    ).sum(axis=1)

    # normalize
    pred_vector_norm = (pred_vector**2).sum(axis=1).sqrt()
    mask = pred_vector_norm < epsilon
    pred_vector_norm[mask] = 1

    # assign <1, 0, 0> to very small vectors (badly predicted)
    pred_vector /= pred_vector_norm.reshape((-1, 1))
    pred_vector[mask] = torch.tensor(
        [1.0, 0.0, 0.0], device="cuda", dtype=pred_vector.dtype
    )

    # convert to angle
    azimuth = torch.arctan2(pred_vector[:, 1], pred_vector[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi
    zenith = torch.arccos(pred_vector[:, 2])

    return azimuth, zenith

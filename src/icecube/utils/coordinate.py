from typing import Tuple

import numpy as np


def az2xyz(
    azimuth: np.ndarray, zenith: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts zenith and azimuth to 3D direction vectors"""
    x = np.cos(azimuth) * np.sin(zenith)
    y = np.sin(azimuth) * np.sin(zenith)
    z = np.cos(zenith)
    return x, y, z


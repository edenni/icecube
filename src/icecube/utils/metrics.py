import numpy as np

from icecube import constants
from icecube.utils.coordinate import az2xyz


def calculate_angular_error(
    pred: np.ndarray, target: np.ndarray
) -> float:
    """Calcualtes the opening angle (angular error) between true and reconstructed direction vectors"""
    assert pred.shape[-1] == 3 and target.shape[-1] == 3

    err = np.arccos((pred * target).sum(axis=1))

    return err.mean()


def angular_dist_score(
    az_true: np.ndarray,
    zen_true: np.ndarray,
    az_pred: np.ndarray,
    zen_pred: np.ndarray,
) -> np.ndarray:
    """
    Calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Args:
        az_true  (np.ndarray): true azimuth value(s) in radian
        zen_true (np.ndarray): true zenith value(s) in radian
        az_pred  (np.ndarray): predicted azimuth value(s) in radian
        zen_pred (np.ndarray): predicted zenith value(s) in radian

    Returns:
        float: mean over the angular distance(s) in radian
    """

    if not (
        np.all(np.isfinite(az_true))
        and np.all(np.isfinite(zen_true))
        and np.all(np.isfinite(az_pred))
        and np.all(np.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    # Pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # Scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # Scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # That might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # Convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))

from typing import Union

import numpy as np
import torch
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
        ae = angular_dist_score(
            target[:, 0], target[:, 1], preds[:, 0], preds[:, 1], avg=False
        )

        self.err += ae.sum()
        self.total += ae.shape[0]

    def compute(self):
        return self.err / self.total


def angular_dist_score(
    az_true: Union[np.ndarray, torch.Tensor],
    zen_true: Union[np.ndarray, torch.Tensor],
    az_pred: Union[np.ndarray, torch.Tensor],
    zen_pred: Union[np.ndarray, torch.Tensor],
    avg: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse
    cosine (arccos) thereof is then the angle between the two input vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    """

    t = np if isinstance(az_pred, np.ndarray) else torch
        

    if not (
        t.all(t.isfinite(az_true))
        and t.all(t.isfinite(zen_true))
        and t.all(t.isfinite(az_pred))
        and t.all(t.isfinite(zen_pred))
    ):
        raise ValueError("All arguments must be finite")

    # pre-compute all sine and cosine values
    sa1 = az_true.sin()
    ca1 = az_true.cos()
    sz1 = zen_true.sin()
    cz1 = zen_true.cos()

    sa2 = az_pred.sin()
    ca2 = az_pred.cos()
    sz2 = zen_pred.sin()
    cz2 = zen_pred.cos()

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = t.clip(scalar_prod, -1, 1)

    ae = t.abs(t.arccos(scalar_prod))

    # convert back to an angle (in radian)
    return t.average(ae) if avg else ae

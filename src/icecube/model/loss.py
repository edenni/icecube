import torch
import torch.nn as nn


class MeanAngularLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        az_true = target[:, 0]
        zen_true = target[:, 1]
        az_pred = preds[:, 0]
        zen_pred = preds[:, 1]
        if not (
            torch.all(torch.isfinite(az_true))
            and torch.all(torch.isfinite(zen_true))
            and torch.all(torch.isfinite(az_pred))
            and torch.all(torch.isfinite(zen_pred))
        ):
            print(preds, target)
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
        scalar_prod = torch.clip(scalar_prod, -1, 1)

        ae = scalar_prod.arccos().abs()

        # convert back to an angle (in radian)
        print(scalar_prod, ae)
        return ae.mean()

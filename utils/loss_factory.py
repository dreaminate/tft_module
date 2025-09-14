from torch.nn import MSELoss, SmoothL1Loss
import torch
from typing import List
from utils.weighted_bce import WeightedBinaryCrossEntropy


class _WeightedMSELoss(MSELoss):
    def __init__(self):
        # use reduction='none' to allow sample weights
        super().__init__(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        loss = super().forward(input, target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class _WeightedSmoothL1Loss(SmoothL1Loss):
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        loss = super().forward(input, target)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


def get_losses_by_targets(target_names: List[str]) -> List[object]:
    losses = []
    smooth_targets = [
        "logreturn", "logsharpe_ratio",
        "breakout_count", "trend_persistence",
        "max_drawdown", "fundflow_strength"
    ]
    for name in target_names:
        if any(key in name for key in [
            "binarytrend", "pullback", "sideway", "drawdown_prob", "outlier"
        ]):
            losses.append(WeightedBinaryCrossEntropy(auto_pos_weight=True))
        elif any(smooth_key in name for smooth_key in smooth_targets):
            losses.append(_WeightedSmoothL1Loss())
        else:
            losses.append(_WeightedMSELoss())
    return losses

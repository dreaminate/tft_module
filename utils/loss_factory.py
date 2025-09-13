from torch.nn import MSELoss, SmoothL1Loss
from typing import List
from utils.weighted_bce import WeightedBinaryCrossEntropy


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
            losses.append(SmoothL1Loss())
        else:
            losses.append(MSELoss())
    return losses


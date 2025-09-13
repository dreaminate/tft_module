from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, AveragePrecision
)
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torch.nn import MSELoss, SmoothL1Loss, CrossEntropyLoss
from tft.utils.weighted_bce import WeightedBinaryCrossEntropy
from typing import List




def get_losses_by_targets(target_names: List[str]) -> List[object]:
    losses = []

    # 使用 SmoothL1Loss 的推荐字段（适合 fat tail / 尖峰）
    smooth_targets = [
        "logreturn", "logsharpe_ratio",
        "breakout_count", "trend_persistence",
        "max_drawdown", "fundflow_strength"
    ]

    for name in target_names:
        if any(key in name for key in [
            "binarytrend", "pullback", "sideway", "drawdown_prob",
             "outlier"
        ]):
            # 二分类目标
            losses.append(WeightedBinaryCrossEntropy(auto_pos_weight=True))
        elif any(smooth_key in name for smooth_key in smooth_targets):
            # 尖峰容错回归目标
            losses.append(SmoothL1Loss())
        else:
            # 其他普通回归目标
            losses.append(MSELoss())
    return losses

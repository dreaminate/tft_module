# === utils/metric_factory.py ===
from typing import List, Tuple
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, AveragePrecision, Accuracy, Precision, Recall
)
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

def get_metrics_by_targets(targets: List[str]) -> List[Tuple[str, object]]:
    """
    根据目标字段名列表返回对应的 metrics 列表（(metric_name, metric_obj)）
    支持自动识别分类与回归目标
    """
    metrics_list = []

    for t in targets:
        if "binarytrend" in t:
            metrics_list += [
                (f"{t}_f1", BinaryF1Score()),
                (f"{t}_accuracy", Accuracy(task="binary")),
                (f"{t}_precision", Precision(task="binary")),
                (f"{t}_recall", Recall(task="binary")),
                (f"{t}_roc_auc", BinaryAUROC()),
                (f"{t}_ap", AveragePrecision(task="binary")),
            ]
        else:
            # 对于回归目标（不再考虑三分类）
            metrics_list += [
                (f"{t}_rmse", MeanSquaredError(squared=False)),
                (f"{t}_mae", MeanAbsoluteError()),
            ]

    return metrics_list

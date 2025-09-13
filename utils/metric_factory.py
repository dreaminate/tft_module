from typing import List, Tuple
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, AveragePrecision, Accuracy, Precision, Recall,
)
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

DEFAULT_HORIZONS = ["1h", "4h", "1d"]


def get_metrics_by_targets(
    targets: List[str], horizons: List[str] | None = None
) -> List[Tuple[str, object]]:
    metrics_list: List[Tuple[str, object]] = []
    horizons = horizons or DEFAULT_HORIZONS
    binary_keywords = {
        "binarytrend",
        "pullback_prob",
        "sideway_detect",
        "drawdown_prob",
        "return_outlier",
    }
    for t in targets:
        is_cls = any(k in t for k in binary_keywords)
        for h in horizons:
            suffix = f"@{h}"
            if is_cls:
                metrics_list += [
                    (f"{t}_f1{suffix}", BinaryF1Score()),
                    (f"{t}_accuracy{suffix}", Accuracy(task="binary")),
                    (f"{t}_precision{suffix}", Precision(task="binary")),
                    (f"{t}_recall{suffix}", Recall(task="binary")),
                    (f"{t}_roc_auc{suffix}", BinaryAUROC()),
                    (f"{t}_ap{suffix}", AveragePrecision(task="binary")),
                ]
            else:
                metrics_list += [
                    (f"{t}_rmse{suffix}", MeanSquaredError(squared=False)),
                    (f"{t}_mae{suffix}", MeanAbsoluteError()),
                ]
    return metrics_list


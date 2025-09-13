from typing import List, Tuple
from torchmetrics.classification import (
    BinaryF1Score, BinaryAUROC, AveragePrecision, Accuracy, Precision, Recall,
)
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

# 默认评估周期，可在调用时传自定义列表
DEFAULT_HORIZONS = ["1h", "4h", "1d"]


def get_metrics_by_targets(
    targets: List[str], horizons: List[str] | None = None
) -> List[Tuple[str, object]]:
    """根据目标字段名返回 (metric_name, metric_obj) 列表。

    每个目标会为 *每个周期* 生成一套指标：
        - 分类目标：F1 / Accuracy / Precision / Recall / ROC AUC / AP
        - 回归目标：RMSE / MAE
    产出名称示例：``val_target_binarytrend_f1@1h``、``val_target_logreturn_rmse@4h`` …
    """

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


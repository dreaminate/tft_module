# Re-export selected utils for cleaner package access
from .loss_factory import get_losses_by_targets
from .metric_factory import get_metrics_by_targets

__all__ = [
    "get_losses_by_targets",
    "get_metrics_by_targets",
]


import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, auto_pos_weight: bool = False, epsilon: float = 1e-8,
                 min_weight: float = 1e-2, max_weight: float = 10.0):
        super().__init__()
        self.auto_pos_weight = auto_pos_weight
        self.epsilon = epsilon
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.latest_stats: dict[str, float] = {}

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        y_true = y_true.float()
        base_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")

        class_weights = torch.ones_like(y_true)
        pos_weight_val = torch.tensor(1.0, device=y_true.device, dtype=y_true.dtype)
        neg_weight_val = torch.tensor(1.0, device=y_true.device, dtype=y_true.dtype)

        if self.auto_pos_weight:
            pos = (y_true >= 0.5).sum().clamp(min=1.0)
            neg = (y_true < 0.5).sum().clamp(min=1.0)
            pos_weight_val = (neg / (pos + self.epsilon)).clamp(min=self.min_weight, max=self.max_weight)
            neg_weight_val = (pos / (neg + self.epsilon)).clamp(min=self.min_weight, max=self.max_weight)
            class_weights = torch.where(y_true >= 0.5, pos_weight_val, neg_weight_val)

        combined_weights = class_weights
        if weight is not None:
            combined_weights = combined_weights * weight

        combined_sum = combined_weights.detach().sum().clamp(min=self.epsilon)
        norm_factor = combined_weights.numel() / combined_sum
        combined_weights = combined_weights * norm_factor

        loss = (base_loss * combined_weights).mean()

        self.latest_stats = {
            "pos_weight": float(pos_weight_val.detach().item()),
            "neg_weight": float(neg_weight_val.detach().item()),
            "mean_combined_weight": float(combined_weights.detach().mean().item()),
            "effective_weight_sum": float(combined_weights.detach().sum().item()),
            "norm_factor": float(norm_factor.detach().item()),
        }
        return loss


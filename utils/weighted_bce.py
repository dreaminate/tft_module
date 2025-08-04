# utils/weighted_bce.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedBinaryCrossEntropy(nn.Module):
    def __init__(self, auto_pos_weight: bool = False, epsilon: float = 1e-8):
        super().__init__()
        self.auto_pos_weight = auto_pos_weight
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
        y_true = y_true.float()
        if self.auto_pos_weight:
            pos = (y_true == 1).sum().clamp(min=1).float()
            neg = (y_true == 0).sum().clamp(min=1).float()
            pos_weight = (neg / (pos + self.epsilon)).clamp(min=1.0, max=10.0)
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none", pos_weight=pos_weight)
        else:
            loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
        if weight is not None:
            loss = loss * weight
        return loss.mean()

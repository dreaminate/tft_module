# utils/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, auto_pos_weight=False, epsilon=1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.auto_pos_weight = auto_pos_weight
        self.epsilon = epsilon

    def forward(self, y_pred, y_true, weight=None):
        y_true = y_true.float()
        bce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")

        probs = torch.sigmoid(y_pred)
        pt = torch.where(y_true == 1, probs, 1 - probs)
        focal_factor = (1 - pt).pow(self.gamma)

        if self.auto_pos_weight:
            pos = (y_true == 1).sum().clamp(min=1).float()
            neg = (y_true == 0).sum().clamp(min=1).float()
            pos_weight = (neg / (pos + self.epsilon)).clamp(min=1.0, max=10.0)
            alpha = torch.where(y_true == 1, pos_weight, 1.0)
        elif self.alpha is not None:
            alpha = torch.where(y_true == 1, self.alpha, 1 - self.alpha)
        else:
            alpha = 1.0

        loss = alpha * focal_factor * bce_loss
        if weight is not None:
            loss = loss * weight

        return loss.mean()

import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import brier_score_loss as brier_score

class TemperatureScaling(nn.Module):
    def __init__(self, temp=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temp)

    def forward(self, logits):
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Ensure logits is at least 2D for binary classification case
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def temperature_scale_binary(logits, labels, temp=1.0):
    model = TemperatureScaling(temp=temp)
    if torch.cuda.is_available():
        model.cuda()
        logits = logits.cuda()
        labels = labels.cuda()
    
    optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=50)
    criterion = nn.BCEWithLogitsLoss()

    def eval():
        optimizer.zero_grad()
        scaled_logits = model(logits)
        loss = criterion(scaled_logits, labels.float().unsqueeze(1))
        loss.backward()
        return loss
    
    optimizer.step(eval)
    
    scaled_logits = model(logits).cpu().detach()
    return scaled_logits, model.temperature.cpu().detach().item()


def compute_ece(preds, labels, n_bins=10):
    preds = np.asarray(preds).flatten()
    labels = np.asarray(labels).flatten()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # For binary classification, preds are for the positive class
    # Binarize labels to handle cases where they are continuous probabilities
    binary_labels = (labels > 0.5).astype(int)
    accuracies_all = ((preds > 0.5).astype(int) == binary_labels)
    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (preds > bin_lower) & (preds <= bin_upper)
        if np.sum(in_bin) > 0:
            prop_in_bin = np.mean(in_bin)
            avg_confidence_in_bin = np.mean(preds[in_bin])
            avg_accuracy_in_bin = np.mean(accuracies_all[in_bin])
            
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
            
    return ece

def reliability_curve(preds, labels, n_bins=10):
    """Computes the reliability curve using torch for consistency.
    
    Args:
        preds (torch.Tensor): The predictions (probabilities).
        labels (torch.Tensor): The true labels (0 or 1).
        n_bins (int): The number of bins to use.
        
    Returns:
        dict: A dictionary containing 'confidence' and 'accuracy' tensors.
    """
    device = preds.device
    preds = preds.flatten()
    labels = labels.flatten()

    # Binarize labels if they are not already, assuming they are probabilities.
    binary_labels = (labels > 0.5).long()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    accuracies_all = ((preds > 0.5).long() == binary_labels).float()
    
    confidences = []
    accuracies_out = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (preds > bin_lower) & (preds <= bin_upper)
        if in_bin.sum() > 0:
            avg_confidence_in_bin = preds[in_bin].mean()
            avg_accuracy_in_bin = accuracies_all[in_bin].mean()
            confidences.append(avg_confidence_in_bin)
            accuracies_out.append(avg_accuracy_in_bin)
            
    conf_tensor = torch.stack(confidences) if confidences else torch.tensor([], device=device)
    acc_tensor = torch.stack(accuracies_out) if accuracies_out else torch.tensor([], device=device)

    return {
        'confidence': conf_tensor,
        'accuracy': acc_tensor
    }

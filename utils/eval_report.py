from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import average_precision_score, mean_squared_error
import torch
from metrics.calibration import compute_ece, brier_score


def _symbol_name(symbol_classes: List[str], idx: int) -> str:
    if 0 <= idx < len(symbol_classes):
        return str(symbol_classes[idx])
    return str(idx)


def _period_name(period_map: Dict[int, str], idx: int) -> str:
    return str(period_map.get(idx, idx))


def build_eval_report(
    symbol_names: List[str],
    period_map: Dict[int, str],
    cls_storage: Dict[str, List[dict]],
    reg_storage: Dict[str, List[dict]],
    n_bins: int = 15,
) -> pd.DataFrame:
    """Aggregate per-symbol×period metrics for validation results."""
    records: List[dict] = []

    # Classification metrics
    for target_name, entries in cls_storage.items():
        if not entries:
            continue
        probs = torch.cat([item['probs'] for item in entries], dim=0).numpy()
        labels = torch.cat([item['labels'] for item in entries], dim=0).numpy()
        symbol_idx = torch.cat([item['symbol_idx'] for item in entries], dim=0).numpy()
        period_idx = torch.cat([item['period_idx'] for item in entries], dim=0).numpy()
        df = pd.DataFrame({
            'symbol_idx': symbol_idx,
            'period_idx': period_idx,
            'prob': probs,
            'label': labels,
        })
        for (sym_i, per_i), group in df.groupby(['symbol_idx', 'period_idx']):
            labels_g = group['label'].to_numpy()
            probs_g = group['prob'].to_numpy()
            if probs_g.size == 0:
                continue
            try:
                pr_auc = float(average_precision_score(labels_g, probs_g)) if np.unique(labels_g).size > 1 else float('nan')
            except ValueError:
                pr_auc = float('nan')
            ece_val = compute_ece(probs_g, labels_g, n_bins=n_bins)
            # Corrected argument order: y_true, y_prob
            binarized_labels = (labels_g > 0.5).astype(np.int32)
            brier_val = brier_score(binarized_labels, probs_g)
            records.append({
                'symbol': _symbol_name(symbol_names, int(sym_i)),
                'period': _period_name(period_map, int(per_i)),
                'target': target_name,
                'metric': 'pr_auc',
                'value': pr_auc,
            })
            records.append({
                'symbol': _symbol_name(symbol_names, int(sym_i)),
                'period': _period_name(period_map, int(per_i)),
                'target': target_name,
                'metric': 'ece',
                'value': ece_val,
            })
            records.append({
                'symbol': _symbol_name(symbol_names, int(sym_i)),
                'period': _period_name(period_map, int(per_i)),
                'target': target_name,
                'metric': 'brier',
                'value': brier_val,
            })

    # Regression metrics
    z_scores = {0.1: -1.2815515655446004, 0.5: 0.0, 0.9: 1.2815515655446004}
    for target_name, entries in reg_storage.items():
        if not entries:
            continue
        preds = torch.cat([item['preds'] for item in entries], dim=0).numpy()
        targets = torch.cat([item['targets'] for item in entries], dim=0).numpy()
        sigma = torch.cat([item.get('sigma', torch.zeros_like(item['preds'])) for item in entries], dim=0).numpy()
        symbol_idx = torch.cat([item['symbol_idx'] for item in entries], dim=0).numpy()
        period_idx = torch.cat([item['period_idx'] for item in entries], dim=0).numpy()
        df = pd.DataFrame({
            'symbol_idx': symbol_idx,
            'period_idx': period_idx,
            'pred': preds,
            'target': targets,
            'sigma': sigma,
        })
        for (sym_i, per_i), group in df.groupby(['symbol_idx', 'period_idx']):
            preds_g = group['pred'].to_numpy()
            targets_g = group['target'].to_numpy()
            sigma_g = group['sigma'].to_numpy()
            if preds_g.size == 0:
                continue
            rmse = float(np.sqrt(mean_squared_error(targets_g, preds_g)))
            records.append({
                'symbol': _symbol_name(symbol_names, int(sym_i)),
                'period': _period_name(period_map, int(per_i)),
                'target': target_name,
                'metric': 'rmse',
                'value': rmse,
            })
            for tau, z in z_scores.items():
                q = preds_g + z * sigma_g
                coverage = float(np.mean(targets_g <= q))
                diff = targets_g - q
                pinball = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
                records.append({
                    'symbol': _symbol_name(symbol_names, int(sym_i)),
                    'period': _period_name(period_map, int(per_i)),
                    'target': target_name,
                    'metric': f'coverage_{int(tau*100)}',
                    'value': coverage,
                })
                records.append({
                    'symbol': _symbol_name(symbol_names, int(sym_i)),
                    'period': _period_name(period_map, int(per_i)),
                    'target': target_name,
                    'metric': f'pinball_{int(tau*100)}',
                    'value': pinball,
                })

    return pd.DataFrame.from_records(records)

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from metrics.calibration import temperature_scale_binary, compute_ece, reliability_curve, brier_score
from utils.eval_report import build_eval_report
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.metrics import MultiLoss, MAE
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class HybridMultiLoss(nn.Module):
    def __init__(
        self,
        losses: nn.ModuleList,
        base_weights: torch.Tensor,
        loss_names: list[str] | None = None,
    ):
        super().__init__()
        assert len(losses) == base_weights.numel()
        self.losses = losses
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))
        self.register_buffer("w", base_weights.view(-1))
        self.loss_names = loss_names or [f"loss_{i}" for i in range(len(losses))]

    def forward(self, preds, targets, sample_weights=None, return_dict: bool = True):
        if sample_weights is None:
            sample_weights = [None] * len(self.losses)
        else:
            sample_weights = list(sample_weights)
            if len(sample_weights) < len(self.losses):
                sample_weights.extend([None] * (len(self.losses) - len(sample_weights)))
        total_loss = 0.0
        details: dict[str, dict] = {}
        sig2 = F.softplus(self.log_vars)
        sig2 = torch.clamp(sig2, min=1e-2)
        log_sig2 = torch.log(sig2 + 1e-8)
        for i, (pred, targ, fn) in enumerate(zip(preds, targets, self.losses)):
            weight = sample_weights[i] if i < len(sample_weights) else None
            li = fn(pred, targ, weight=weight) if weight is not None else fn(pred, targ)
            term = li / sig2[i] + log_sig2[i]
            weighted = self.w[i] * term
            total_loss = total_loss + weighted
            if return_dict:
                name = self.loss_names[i]
                details[name] = {"raw": li.detach(), "weighted": weighted.detach()}
        return (total_loss, details) if return_dict else total_loss


class MyTFTModule(LightningModule):
    def __init__(
        self,
        dataset,
        loss_list,
        weights,
        output_size,
        metrics_list,
        learning_rate: float = 1e-3,
        loss_schedule: dict | None = None,
        target_names: list[str] | None = None,
        period_map: dict | None = None,
        pct_start: float = 0.1,
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        norm_pack: dict | None = None,
        steps_per_epoch: int | None = None,
        output_head_cfg: dict | None = None,
        symbol_weight_map: dict | None = None,
        expert_name: str | None = None,
        period_name: str | None = None,
        modality_name: str | None = None,
        schema_version: str | None = None,
        data_version: str | None = None,
        expert_version: str | None = None,
        train_window_id: str | None = None,
        **tft_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "loss_schedule": loss_schedule or {},
            "pct_start": pct_start,
            "div_factor": div_factor,
            "final_div_factor": final_div_factor,
            "steps_per_epoch": steps_per_epoch,
            "output_head_cfg": output_head_cfg or {},
            "symbol_weight_map": symbol_weight_map or {},
            "expert_name": expert_name,
            "period_name": period_name,
            "modality_name": modality_name,
            "schema_version": schema_version,
            "data_version": data_version,
            "expert_version": expert_version,
            "train_window_id": train_window_id,
        })
        self.target_names = target_names or []
        self.loss_schedule = loss_schedule or {}
        self.regression_targets = [
            "target_logreturn",
            "target_logsharpe_ratio",
            "target_breakout_count",
            "target_max_drawdown",
            "target_trend_persistence",
        ]
        self.reg_indices = [self.target_names.index(t) for t in self.regression_targets if t in self.target_names]

        self.expert_name = expert_name or "unknown_expert"
        self.period_name = period_name or "_"
        self.modality_name = modality_name or "_"
        self.prediction_buffer = []
        self.prediction_file_counter = 0
        self.schema_version = schema_version or 'schema_v1'
        self.data_version = data_version or 'data_unknown'
        self.expert_version = expert_version or 'expert_unknown'
        self.train_window_id = train_window_id or 'window_unknown'
        self.val_cls_storage = defaultdict(list)
        self.val_reg_storage = defaultdict(list)
        self.calibration_temperature = {}
        self.output_head_cfg = output_head_cfg or {}
        self.symbol_weight_cfg = symbol_weight_map or {}
        self.head_type = str(self.output_head_cfg.get("type", "shared")).lower()
        self.head_apply_on = str(self.output_head_cfg.get("apply_on", "both")).lower()
        reg_cfg = (self.output_head_cfg.get("regularization") or {})
        self.reg_l2_scale = float(reg_cfg.get("l2_scale_from_one", 0.0) or 0.0)
        self.reg_l2_bias = float(reg_cfg.get("l2_bias", 0.0) or 0.0)
        clamp_cfg = (self.output_head_cfg.get("clamp_scale") or {})
        self.clamp_scale_min = clamp_cfg.get("min")
        self.clamp_scale_max = clamp_cfg.get("max")
        if self.clamp_scale_min is not None:
            self.clamp_scale_min = float(self.clamp_scale_min)
        if self.clamp_scale_max is not None:
            self.clamp_scale_max = float(self.clamp_scale_max)

        self.cls_target_idx = self._infer_classification_targets(self.target_names)
        self.reg_target_idx = [i for i in range(len(self.target_names)) if i not in self.cls_target_idx]

        if norm_pack is None:
            self.register_buffer("means_tbl", torch.zeros(1, 1, len(self.regression_targets)), persistent=True)
            self.register_buffer("stds_tbl", torch.ones(1, 1, len(self.regression_targets)), persistent=True)
            self.sym2idx = {}
            self.per2idx = {}
        else:
            means = torch.as_tensor(norm_pack["means"])
            stds = torch.as_tensor(norm_pack["stds"])
            self.register_buffer("means_tbl", means, persistent=True)
            self.register_buffer("stds_tbl", stds, persistent=True)
            self.sym2idx = norm_pack.get("sym2idx", {})
            self.per2idx = norm_pack.get("per2idx", {})

        # ensure TFT gets a MultiLoss when there are multiple targets to satisfy
        # pytorch_forecasting's internal assertion, even though we compute loss externally
        tft_kwargs.pop("loss", None)
        tft_kwargs.pop("output_size", None)
        n_targets = len(output_size) if isinstance(output_size, (list, tuple)) else 1
        safe_loss = MAE() if n_targets == 1 else MultiLoss([MAE()] * n_targets)
        self.model = TemporalFusionTransformer.from_dataset(
            dataset, loss=safe_loss, output_size=output_size, **tft_kwargs
        )
        for m in self.model.modules():
            if hasattr(m, "mask_bias"):
                m.mask_bias = -1e-4 * 1e4

        self.metrics_list = nn.ModuleList([m for _, m in metrics_list])
        self.metric_names = [n for n, _ in metrics_list]
        self.metric_target_idx = [
            next(i for i, t in enumerate(self.target_names)
                 if n.split("@")[0].startswith(t + "_") or n.startswith(t + "_"))
            for n in self.metric_names
        ]
        cls_suffix = ("_f1", "_precision", "_recall", "_accuracy", "_ap", "_auc")
        self.metric_is_cls = [any(n.split("@")[0].endswith(s) for s in cls_suffix) for n in self.metric_names]
        self.confmats = nn.ModuleDict({
            name: torchmetrics.classification.BinaryConfusionMatrix()
            for name, is_cls in zip(self.metric_names, self.metric_is_cls) if is_cls
        })

        self.period_group_idx = dataset.group_ids.index("period")
        self.symbol_group_idx = dataset.group_ids.index("symbol")
        if period_map is not None:
            self.period_map = period_map
        else:
            enc = dataset.categorical_encoders.get("period", None)
            classes_ = getattr(enc, "classes_", None)
            self.period_map = {i: c for i, c in enumerate(classes_)} if classes_ is not None else {}

        self.close_encoder_idx = None
        self.close_decoder_idx = None
        try:
            enc_reals = getattr(dataset, "reals", {}).get("encoder", [])
            if isinstance(enc_reals, (list, tuple)) and "close" in enc_reals:
                self.close_encoder_idx = enc_reals.index("close")
        except Exception:
            pass
        try:
            dec_reals = getattr(dataset, "reals", {}).get("decoder", [])
            if isinstance(dec_reals, (list, tuple)) and "close" in dec_reals:
                self.close_decoder_idx = dec_reals.index("close")
        except Exception:
            pass
        self._build_loss(loss_list, weights, self.target_names)
        self._init_symbol_structures(dataset, norm_pack)

    def _batch_sym_per_idx(self, x):
        g = x["groups"]
        sym_idx = g[:, self.symbol_group_idx].long()
        per_idx = g[:, self.period_group_idx].long()
        if isinstance(x, dict) and "symbol_idx" not in x:
            x["symbol_idx"] = sym_idx
        return sym_idx, per_idx

    def _build_loss(self, loss_list, weights, target_names):
        assert isinstance(loss_list, (list, nn.ModuleList)) and len(loss_list) > 0
        losses = nn.ModuleList(loss_list)
        base_w = torch.as_tensor(weights, dtype=torch.float32).view(-1)
        assert base_w.numel() == len(losses)
        if target_names and len(target_names) == len(losses):
            loss_names = [f"{t}_loss" for t in target_names]
        else:
            loss_names = [f"loss_{i}" for i in range(len(losses))]
        self.loss_fn = HybridMultiLoss(losses=losses, base_weights=base_w, loss_names=loss_names)

    def _standardize_y(self, y_enc, sym_idx, per_idx):
        if not self.reg_indices or self.means_tbl.numel() <= 1:
            return y_enc
        y_std = y_enc.clone()
        sym_idx = sym_idx.to(self.means_tbl.device)
        per_idx = per_idx.to(self.means_tbl.device)
        means_b = self.means_tbl[sym_idx, per_idx, :]
        stds_b = torch.clamp(self.stds_tbl[sym_idx, per_idx, :], min=1e-8)
        for local_t, global_t in enumerate(self.reg_indices):
            y_std[:, global_t] = (y_enc[:, global_t] - means_b[:, local_t]) / stds_b[:, local_t]
        return y_std

    def _destandardize_pred_and_true(self, preds_bt, y_enc, sym_idx, per_idx):
        if not self.reg_indices or self.means_tbl.numel() <= 1:
            return preds_bt, y_enc
        preds = preds_bt.clone()
        ys = y_enc.clone()
        sym_idx = sym_idx.to(self.means_tbl.device)
        per_idx = per_idx.to(self.means_tbl.device)
        means_b = self.means_tbl[sym_idx, per_idx, :]
        stds_b = torch.clamp(self.stds_tbl[sym_idx, per_idx, :], min=1e-8)
        for local_t, global_t in enumerate(self.reg_indices):
            preds[:, global_t] = preds_bt[:, global_t] * stds_b[:, local_t] + means_b[:, local_t]
            ys[:, global_t] = y_enc[:, global_t] * stds_b[:, local_t] + means_b[:, local_t]
        return preds, ys

    def _to_dev(self, obj, dev):
        if torch.is_tensor(obj):
            return obj.to(dev)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_dev(o, dev) for o in obj)
        if isinstance(obj, dict):
            return {k: self._to_dev(v, dev) for k, v in obj.items()}
        return obj

    def _stack_y(self, y_list, device=None):
        cat = torch.cat([t.view(-1, 1).float() for t in y_list], dim=1)
        return cat.to(device) if device else cat

    def _parse_y_to_list(self, y, n_targets_expected: int | None = None):
        """Convert various y formats from TimeSeriesDataSet to a list of 1D tensors (per-target).

        Handles:
        - y as Tensor with shapes [B], [B, 1], [B, T], [B, 1, T] (T = n_targets)
        - y as list/tuple of tensors (drops None if present)
        - y as tuple like (y, weights) -> uses first element
        """
        # unwrap (y, weights) style tuples
        if isinstance(y, tuple) and len(y) == 2:
            if torch.is_tensor(y[0]) and (y[1] is None or torch.is_tensor(y[1])):
                y = y[0]
            elif isinstance(y[0], (list, tuple)) and all((t is None or torch.is_tensor(t)) for t in y[0]):
                y = y[0]

        y_list: list[torch.Tensor] = []
        if torch.is_tensor(y):
            # normalize to [B, T]
            if y.dim() == 1:
                y2 = y.view(-1, 1)
            elif y.dim() == 2:
                y2 = y
            elif y.dim() == 3:
                # try squeeze prediction length dim
                if y.size(1) == 1:
                    y2 = y[:, 0, :]
                elif y.size(2) == 1:
                    y2 = y[:, :, 0]
                else:
                    # fallback: flatten middle dims and try to split later
                    y2 = y.view(y.size(0), -1)
            else:
                raise RuntimeError(f"Unsupported y tensor shape: {tuple(y.shape)}")
            # if single target, y2 could be [B, 1]
            if y2.dim() == 1:
                y2 = y2.view(-1, 1)
            # if expected number of targets known and matches, split columns
            n_cols = y2.size(1)
            if n_targets_expected is not None and n_cols != n_targets_expected:
                # try to coerce [B] -> [B, 1] handled above, else accept as-is
                pass
            if n_cols == 1 and (n_targets_expected == 1 or n_targets_expected is None):
                y_list = [y2[:, 0].contiguous().view(-1)]
            else:
                y_list = [y2[:, i].contiguous().view(-1) for i in range(n_cols)]
        elif isinstance(y, (list, tuple)):
            # treat as per-target list; drop None safely
            for t in y:
                if t is None:
                    continue
                if not torch.is_tensor(t):
                    raise RuntimeError(f"y contains non-tensor element: {type(t)}")
                tt = t
                # remove trivial dims like [B, 1]
                if tt.dim() >= 2:
                    # prefer squeezing last then second dim (prediction length)
                    tt = tt.squeeze(-1)
                    if tt.dim() >= 2:
                        tt = tt.squeeze(1)
                y_list.append(tt.contiguous().view(-1))
        else:
            raise RuntimeError(f"Unsupported y type: {type(y)}")

        if n_targets_expected is not None and len(y_list) != n_targets_expected:
            raise RuntimeError(
                f"Parsed y targets {len(y_list)} != expected {n_targets_expected}. Shapes/types may be inconsistent."
            )
        return y_list

    def forward(self, x):
        x = self._to_dev(x, self.device)
        out = self.model(x)["prediction"]
        if isinstance(out, list):
            return [t.flatten() for t in out]
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return [out[:, i].contiguous().view(-1) for i in range(out.size(1))]

    def _shared_step(self, batch, stage):
        x, y = batch
        for name in ("encoder_cont", "decoder_cont"):
            if name in x:
                ten = x[name]
                if not torch.isfinite(ten).all():
                    raise RuntimeError(f"[{stage}] {name} has NaN/Inf")
        # predictions list length equals number of targets when output_size is a list
        pred_list = self.forward(x)
        n_targets = len(pred_list)
        preds_bt = torch.stack(pred_list, dim=1).to(self.device)
        sym_idx, per_idx = self._batch_sym_per_idx(x)
        preds_bt = self._apply_symbol_affine(preds_bt, sym_idx)
        pred_list = [preds_bt[:, i].contiguous().view(-1) for i in range(preds_bt.size(1))]
        # robustly parse y from dataset into per-target vectors
        y_list = self._parse_y_to_list(y, n_targets_expected=n_targets)
        y_enc = self._stack_y(y_list, self.device)
        y_for_loss = self._standardize_y(y_enc, sym_idx, per_idx)
        sample_weights = None
        if hasattr(self, "sym_weight_tensor") and self.sym_weight_tensor is not None:
            w_batch = self.sym_weight_tensor.index_select(0, sym_idx).view(-1)
            sample_weights = [w_batch] * n_targets
        weighted_total, sub_losses = self.loss_fn(
            pred_list,
            [y_for_loss[:, i] for i in range(y_for_loss.size(1))],
            sample_weights=sample_weights,
            return_dict=True,
        )
        reg_term = self._symbol_head_regularization()
        if reg_term is not None:
            weighted_total = weighted_total + reg_term
            self.log(f"{stage}/affine_reg", reg_term, on_step=False, on_epoch=True, prog_bar=False, batch_size=y_enc.size(0))
        raws = torch.stack([v["raw"] for v in sub_losses.values()])
        w = self.loss_fn.w.to(raws)
        raw_total = (raws * w).sum()
        self.log(f"{stage}_loss_raw", raw_total, on_step=True, on_epoch=True, prog_bar=True, batch_size=y_enc.size(0))
        self.log(f"{stage}_loss", weighted_total, on_step=True, on_epoch=True, prog_bar=True, batch_size=y_enc.size(0))
        for name, item in sub_losses.items():
            self.log(f"{stage}/{name}_raw", item["raw"], on_step=True, on_epoch=True, batch_size=y_enc.size(0))
            self.log(f"{stage}/{name}_w", item["weighted"], on_step=True, on_epoch=True, batch_size=y_enc.size(0))
        return weighted_total, preds_bt, y_enc, x, sym_idx, per_idx

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_cls_storage = defaultdict(list)
        self.val_reg_storage = defaultdict(list)

    def _collect_validation_outputs(self, preds_logits, preds_eval, y_enc, y_eval, sym_idx, per_idx, batch_inputs):
        if not self.cls_target_idx and not self.reg_target_idx:
            return
        sym_cpu = sym_idx.detach().to('cpu', dtype=torch.long)
        per_cpu = per_idx.detach().to('cpu', dtype=torch.long)
        if self.cls_target_idx:
            logits = preds_logits[:, self.cls_target_idx].detach().to('cpu')
            probs = torch.sigmoid(logits)
            labels = y_enc[:, self.cls_target_idx].detach().to('cpu')
            for local_pos, target_idx in enumerate(self.cls_target_idx):
                target_name = self.target_names[target_idx]
                self.val_cls_storage[target_name].append({
                    'logits': logits[:, local_pos],
                    'probs': probs[:, local_pos],
                    'labels': labels[:, local_pos].float(),
                    'symbol_idx': sym_cpu,
                    'period_idx': per_cpu,
                })
        if self.reg_target_idx:
            preds_cpu = preds_eval[:, self.reg_target_idx].detach().to('cpu')
            targets_cpu = y_eval[:, self.reg_target_idx].detach().to('cpu')
            sigmas = torch.sqrt(F.softplus(self.loss_fn.log_vars.detach())).to('cpu')
            for local_pos, target_idx in enumerate(self.reg_target_idx):
                target_name = self.target_names[target_idx]
                sigma_val = float(sigmas[target_idx].item())
                sigma_series = torch.full_like(preds_cpu[:, local_pos], sigma_val)
                self.val_reg_storage[target_name].append({
                    'preds': preds_cpu[:, local_pos],
                    'targets': targets_cpu[:, local_pos],
                    'sigma': sigma_series,
                    'symbol_idx': sym_cpu,
                    'period_idx': per_cpu,
                })

    def training_step(self, batch, batch_idx):
        total, *_ = self._shared_step(batch, "train")
        return total

    def validation_step(self, batch, batch_idx):
        total, preds, y_enc, x, sym_idx, per_idx = self._shared_step(batch, "val")
        preds_eval, y_eval = self._destandardize_pred_and_true(preds, y_enc, sym_idx, per_idx)
        period_idx = x["groups"][:, self.period_group_idx].to(self.device)
        for idx, metric in enumerate(self.metrics_list):
            name = self.metric_names[idx]
            suffix = name.split("@")[-1]
            pid = next((k for k, v in self.period_map.items() if str(v) == str(suffix)), None)
            if pid is None:
                continue
            mask = (period_idx == pid)
            if not mask.any():
                continue
            col = self.metric_target_idx[idx]
            if self.metric_is_cls[idx]:
                prob = torch.sigmoid(preds[mask, col])
                t = y_enc[mask, col].float().clamp(0, 1).int()
                metric.update(prob, t)
                if name in self.confmats:
                    self.confmats[name].update((prob > 0.5).int(), t)
            else:
                metric.update(preds_eval[mask, col], y_eval[mask, col])
        self._collect_validation_outputs(preds, preds_eval, y_enc, y_eval, sym_idx, per_idx, x)
        return total

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        for name, metric in zip(self.metric_names, self.metrics_list):
            try:
                val = metric.compute()
            except Exception:
                metric.reset()
                continue
            self.log(f"val_{name}", val, on_epoch=True, batch_size=1)
            metric.reset()
        for tag, cm in self.confmats.items():
            try:
                cm_val = cm.compute().cpu().numpy()
            except Exception:
                cm.reset()
                continue
            if cm_val.size:
                fig, ax = plt.subplots(figsize=(3, 3))
                ConfusionMatrixDisplay(cm_val, display_labels=[0, 1]).plot(ax=ax, colorbar=False)
                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.add_figure(f"val/confmat_{tag}", fig, self.current_epoch)
                plt.close(fig)
            cm.reset()

        self._log_validation_calibration_and_reports()

    def on_train_epoch_start(self):
        if self.current_epoch in self.loss_schedule:
            vals = self.loss_schedule[self.current_epoch]
            expected = int(self.loss_fn.w.numel())
            device = self.loss_fn.w.device
            dtype = self.loss_fn.w.dtype

            w_new = None
            # allow dict mapping by target name
            if isinstance(vals, dict) and getattr(self, "target_names", None):
                w_arr = []
                for t in self.target_names[:expected]:
                    w_arr.append(float(vals.get(t, 1.0)))
                w_new = torch.tensor(w_arr, dtype=dtype, device=device)
            elif isinstance(vals, (list, tuple)):
                if len(vals) == expected:
                    w_new = torch.tensor([float(x) for x in vals], dtype=dtype, device=device)
                elif len(vals) == 1:
                    w_new = torch.full((expected,), float(vals[0]), dtype=dtype, device=device)
                else:
                    self.print(
                        f"[warn] loss_schedule length {len(vals)} != expected {expected}; keep previous weights."
                    )
            elif isinstance(vals, (int, float)):
                w_new = torch.full((expected,), float(vals), dtype=dtype, device=device)

            if w_new is not None and w_new.numel() == expected:
                self.loss_fn.w.copy_(w_new.view_as(self.loss_fn.w))

    def _log_validation_calibration_and_reports(self):
        if not hasattr(self, 'val_cls_storage'):
            return
        from datetime import datetime
        from pathlib import Path
        log_dir = None
        if self.logger and hasattr(self.logger, 'log_dir'):
            log_dir = Path(self.logger.log_dir)
        else:
            log_dir = Path('lightning_logs')
        results_dir = log_dir / 'eval_reports'
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
        rank = getattr(self.trainer, 'global_rank', 0)
        eval_rows = []

        # Calibration for classification targets
        for target_name, entries in self.val_cls_storage.items():
            if not entries:
                continue
            logits = torch.cat([e['logits'] for e in entries], dim=0)
            labels = torch.cat([e['labels'] for e in entries], dim=0)
            probs = torch.sigmoid(logits)
            temperature = temperature_scale_binary(logits, labels)
            scaled_probs = torch.sigmoid(logits / temperature)
            ece_value = compute_ece(scaled_probs, labels)
            brier_value = brier_score(scaled_probs, labels)
            self.calibration_temperature[target_name] = float(temperature)
            self.log(f"val_ece@{target_name}", float(ece_value), on_epoch=True, prog_bar=False, logger=True)
            self.log(f"val_brier@{target_name}", float(brier_value), on_epoch=True, prog_bar=False, logger=True)

            bins = reliability_curve(scaled_probs, labels)
            if bins['confidence'].numel() and hasattr(self.logger, 'experiment'):
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
                ax.plot(bins['confidence'].numpy(), bins['accuracy'].numpy(), marker='o', label='Empirical')
                ax.set_title(f'Reliability {target_name}')
                ax.set_xlabel('Predicted probability')
                ax.set_ylabel('Observed frequency')
                ax.set_ylim(0, 1)
                ax.set_xlim(0, 1)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')
                self.logger.experiment.add_figure(f'val/reliability_{target_name}', fig, self.current_epoch)
                plt.close(fig)

        z_scores = {0.1: -1.2815515655446004, 0.5: 0.0, 0.9: 1.2815515655446004}
        for target_name, entries in self.val_reg_storage.items():
            if not entries:
                continue
            preds = torch.cat([item['preds'] for item in entries], dim=0).numpy()
            targets = torch.cat([item['targets'] for item in entries], dim=0).numpy()
            sigma = torch.cat([item['sigma'] for item in entries], dim=0).numpy()
            coverages = {}
            pinball_losses = {}
            for tau, z in z_scores.items():
                q = preds + z * sigma
                coverages[tau] = float(np.mean(targets <= q))
                diff = targets - q
                pinball_losses[tau] = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
            if pinball_losses:
                mean_pinball = float(np.mean(list(pinball_losses.values())))
                self.log(f"val_pinball@{target_name}", mean_pinball, on_epoch=True, prog_bar=False, logger=True)
                for tau, loss in pinball_losses.items():
                    self.log(f"val_pinball{int(tau*100)}@{target_name}", loss, on_epoch=True, prog_bar=False, logger=True)
            for tau, cov in coverages.items():
                self.log(f"val_coverage{int(tau*100)}@{target_name}", cov, on_epoch=True, prog_bar=False, logger=True)

        # Build evaluation report
        symbol_names = getattr(self, 'symbol_classes', [])
        report_df = build_eval_report(
            symbol_names=symbol_names,
            period_map=self.period_map,
            cls_storage=self.val_cls_storage,
            reg_storage=self.val_reg_storage,
        )
        if not report_df.empty:
            filename = f'eval_report_{self.expert_name}_{self.period_name}_{timestamp}_rank{rank:02d}.csv'
            report_path = results_dir / filename
            report_df.to_csv(report_path, index=False)
            if hasattr(self.logger, 'experiment'):
                for row in report_df.itertuples():
                    tag = f"eval/{row.metric}/{row.target}/{row.symbol}/{row.period}"
                    try:
                        self.logger.experiment.add_scalar(tag, float(row.value), self.current_epoch)
                    except Exception:
                        pass

        self.val_cls_storage = defaultdict(list)
        self.val_reg_storage = defaultdict(list)

    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        return_standardized: bool = False,
        with_antilog: bool = True,
        with_future_price: bool = True,
    ):
        x, _ = batch
        sym_idx, per_idx = self._batch_sym_per_idx(x)
        pred_list = self.forward(x)
        preds_bt = torch.stack(pred_list, dim=1).float()
        preds_bt = self._apply_symbol_affine(preds_bt, sym_idx)
        preds_bt = preds_bt.contiguous()

        scale, bias = self._get_affine_params_for_batch(sym_idx)
        if scale is None:
            scale = torch.ones_like(preds_bt)
        else:
            scale = scale.to(preds_bt.device, dtype=preds_bt.dtype)
        if bias is None:
            bias = torch.zeros_like(preds_bt)
        else:
            bias = bias.to(preds_bt.device, dtype=preds_bt.dtype)

        preds_orig, _ = self._destandardize_pred_and_true(preds_bt, preds_bt, sym_idx, per_idx)
        preds_orig = preds_orig.contiguous()

        scores = preds_bt.clone()
        if self.cls_target_idx:
            scores[:, self.cls_target_idx] = torch.sigmoid(scores[:, self.cls_target_idx])
        if self.reg_target_idx:
            scores[:, self.reg_target_idx] = preds_orig[:, self.reg_target_idx]

        scores_cpu = scores.detach().to('cpu')
        uncertainty_cpu = torch.full_like(scores_cpu, float('nan'))
        if self.reg_target_idx:
            sigmas = torch.sqrt(F.softplus(self.loss_fn.log_vars.detach())).to(scores_cpu.device, dtype=scores_cpu.dtype)
            for target_idx in self.reg_target_idx:
                uncertainty_cpu[:, target_idx] = sigmas[target_idx].item()

        time_idx_tensor = x.get('decoder_time_idx')
        if time_idx_tensor is None:
            time_idx_cpu = torch.arange(scores_cpu.size(0), dtype=torch.long)
        else:
            time_idx_cpu = time_idx_tensor.view(scores_cpu.size(0), -1)[:, 0].detach().to('cpu', dtype=torch.long)

        meta = {
            'symbol_idx': sym_idx.detach().to('cpu', dtype=torch.long),
            'period_idx': per_idx.detach().to('cpu', dtype=torch.long),
            'time_idx': time_idx_cpu,
            'head_scale': scale.detach().to('cpu'),
            'head_bias': bias.detach().to('cpu'),
        }

        record = {
            'score': scores_cpu,
            'uncertainty': uncertainty_cpu,
            'meta': meta,
            'schema_ver': self.schema_version,
            'data_ver': self.data_version,
            'expert_ver': self.expert_version,
            'train_window_id': self.train_window_id,
        }

        preds_antilog = None
        future_price = None
        if with_antilog and not return_standardized:
            try:
                li = self.target_names.index('target_logreturn')
            except ValueError:
                li = None
            if li is not None:
                preds_antilog = torch.expm1(preds_orig[:, li]).detach().to('cpu')
                if with_future_price:
                    base_close = None
                    if self.close_encoder_idx is not None and 'encoder_cont' in x:
                        try:
                            base_close = x['encoder_cont'][:, -1, self.close_encoder_idx]
                        except Exception:
                            base_close = None
                    if base_close is None and self.close_decoder_idx is not None and 'decoder_cont' in x:
                        try:
                            base_close = x['decoder_cont'][:, 0, self.close_decoder_idx]
                        except Exception:
                            base_close = None
                    if base_close is not None:
                        future_price = (base_close * torch.exp(preds_orig[:, li])).detach().to('cpu')
        record['antilog_return'] = preds_antilog
        record['future_price'] = future_price

        self.prediction_buffer.append(record)
        return record

    def _infer_classification_targets(self, target_names):
        cls_keywords = ("binary", "prob", "detect", "outlier")
        idx_list = []
        for idx, name in enumerate(target_names or []):
            lower = name.lower()
            if any(k in lower for k in cls_keywords):
                idx_list.append(idx)
        return idx_list

    def _init_symbol_structures(self, dataset, norm_pack):
        enc = dataset.categorical_encoders.get("symbol") if hasattr(dataset, "categorical_encoders") else None
        classes = []
        if enc is not None:
            classes_attr = getattr(enc, "classes_", None)
            if classes_attr is not None:
                classes = [str(c) for c in classes_attr]
        if not classes and norm_pack:
            classes = [str(c) for c in norm_pack.get("symbol_classes", [])]
        classes = classes or []
        self.symbol_classes = classes
        self.symbol_to_idx = {s: i for i, s in enumerate(classes)}
        if classes:
            default_w = float(self.symbol_weight_cfg.get("default", 1.0))
            weight_vec = [float(self.symbol_weight_cfg.get(sym, default_w)) for sym in classes]
            tensor = torch.tensor(weight_vec, dtype=torch.float32)
        else:
            tensor = torch.ones(1, dtype=torch.float32)
            self.symbol_classes = ["__default__"]
            self.symbol_to_idx = {"__default__": 0}
        self.register_buffer("sym_weight_tensor", tensor, persistent=True)
        self.use_symbol_head = self.head_type == "per_symbol_affine" and len(self.target_names) > 0
        if not self.use_symbol_head:
            self.affine_scale = None
            self.affine_bias = None
            return
        n_symbols = len(self.symbol_classes)
        n_targets = max(1, len(self.target_names))
        init_cfg = self.output_head_cfg.get("init", {}) if isinstance(self.output_head_cfg, dict) else {}
        init_scale = float(init_cfg.get("scale", 1.0))
        init_bias = float(init_cfg.get("bias", 0.0))
        self.affine_scale = nn.Parameter(torch.full((n_symbols, n_targets), init_scale, dtype=torch.float32))
        self.affine_bias = nn.Parameter(torch.full((n_symbols, n_targets), init_bias, dtype=torch.float32))

    def _get_affine_params_for_batch(self, sym_idx):
        if not getattr(self, "use_symbol_head", False) or self.affine_scale is None:
            return None, None
        scale = self.affine_scale.index_select(0, sym_idx)
        bias = self.affine_bias.index_select(0, sym_idx)
        clamp_args = {}
        if self.clamp_scale_min is not None:
            clamp_args["min"] = self.clamp_scale_min
        if self.clamp_scale_max is not None:
            clamp_args["max"] = self.clamp_scale_max
        if clamp_args:
            scale = torch.clamp(scale, **clamp_args)
        return scale, bias

    def _apply_symbol_affine(self, preds_bt, sym_idx):
        if not getattr(self, "use_symbol_head", False) or self.affine_scale is None:
            return preds_bt
        scale, bias = self._get_affine_params_for_batch(sym_idx)
        if scale is None:
            return preds_bt
        scale = scale.to(preds_bt.device, dtype=preds_bt.dtype)
        bias = bias.to(preds_bt.device, dtype=preds_bt.dtype)
        if self.head_apply_on in ("logits", "both") and self.cls_target_idx:
            preds_bt[:, self.cls_target_idx] = preds_bt[:, self.cls_target_idx] * scale[:, self.cls_target_idx] + bias[:, self.cls_target_idx]
        if self.head_apply_on in ("regression", "both") and self.reg_target_idx:
            preds_bt[:, self.reg_target_idx] = preds_bt[:, self.reg_target_idx] * scale[:, self.reg_target_idx] + bias[:, self.reg_target_idx]
        return preds_bt

    def _symbol_head_regularization(self):
        if not getattr(self, "use_symbol_head", False) or self.affine_scale is None:
            return None
        terms = []
        if self.reg_l2_scale > 0:
            terms.append(self.reg_l2_scale * torch.mean((self.affine_scale - 1.0) ** 2))
        if self.reg_l2_bias > 0:
            terms.append(self.reg_l2_bias * torch.mean(self.affine_bias ** 2))
        if not terms:
            return None
        return sum(terms)

    def _log_symbol_head_stats(self):
        if not getattr(self, "use_symbol_head", False) or self.affine_scale is None:
            return
        scale = self.affine_scale.detach()
        bias = self.affine_bias.detach()
        self.log("head/scale_mean", scale.mean(), on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        self.log("head/scale_std", scale.std(), on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        self.log("head/bias_mean", bias.mean(), on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        self.log("head/bias_std", bias.std(), on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
        for idx, sym in enumerate(self.symbol_classes):
            self.log(f"head/{sym}_scale", scale[idx].mean(), on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)
            self.log(f"head/{sym}_bias", bias[idx].mean(), on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        self._log_symbol_head_stats()

    def on_predict_epoch_start(self):
        super().on_predict_epoch_start()
        self.prediction_buffer = []

    def on_predict_epoch_end(self, results):
        super().on_predict_epoch_end(results)
        if not getattr(self, "prediction_buffer", None):
            return
        try:
            import pandas as pd
        except ImportError:
            self.print("[warn] pandas not available; skip prediction logging.")
            self.prediction_buffer = []
            return
        from pathlib import Path
        from datetime import datetime

        scores = torch.cat([rec["score"] for rec in self.prediction_buffer], dim=0)
        uncertainty = torch.cat([rec["uncertainty"] for rec in self.prediction_buffer], dim=0)
        symbol_idx = torch.cat([rec["meta"]["symbol_idx"] for rec in self.prediction_buffer], dim=0)
        period_idx = torch.cat([rec["meta"]["period_idx"] for rec in self.prediction_buffer], dim=0)
        time_idx = torch.cat([rec["meta"]["time_idx"] for rec in self.prediction_buffer], dim=0)
        head_scale = torch.cat([rec["meta"]["head_scale"] for rec in self.prediction_buffer], dim=0)
        head_bias = torch.cat([rec["meta"]["head_bias"] for rec in self.prediction_buffer], dim=0)

        num_samples, num_targets = scores.shape
        target_names = self.target_names or [f"target_{i}" for i in range(num_targets)]
        symbol_classes = getattr(self, "symbol_classes", None) or []
        period_map = getattr(self, "period_map", {})

        rows = []
        for row in range(num_samples):
            sym_idx_val = int(symbol_idx[row].item())
            period_idx_val = int(period_idx[row].item())
            symbol_name = symbol_classes[sym_idx_val] if sym_idx_val < len(symbol_classes) else str(sym_idx_val)
            period_name = str(period_map.get(period_idx_val, period_idx_val))
            time_value = int(time_idx[row].item())
            for target_pos, target_name in enumerate(target_names):
                rows.append({
                    "symbol": symbol_name,
                    "symbol_idx": sym_idx_val,
                    "period": period_name,
                    "period_idx": period_idx_val,
                    "time_idx": time_value,
                    "target": target_name,
                    "score": float(scores[row, target_pos].item()),
                    "uncertainty": float(uncertainty[row, target_pos].item()),
                    "head_scale": float(head_scale[row, target_pos].item()),
                    "head_bias": float(head_bias[row, target_pos].item()),
                    "expert": self.expert_name,
                    "expert_ver": self.expert_version,
                    "schema_ver": self.schema_version,
                    "data_ver": self.data_version,
                    "train_window_id": self.train_window_id,
                    "period_config": self.period_name,
                    "modality": self.modality_name,
                    "epoch": int(self.current_epoch),
                    "global_step": int(self.global_step),
                })

        if not rows:
            self.prediction_buffer = []
            return

        df = pd.DataFrame.from_records(rows)
        if self.logger and hasattr(self.logger, "log_dir"):
            out_dir = Path(self.logger.log_dir) / "predictions"
        else:
            out_dir = Path("lightning_logs") / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        rank = getattr(self.trainer, "global_rank", 0)
        expert_tag = (self.expert_name or "expert").replace("/", "-")
        period_tag = (self.period_name or "period").replace("/", "-")
        filename = f"predictions_{expert_tag}_{period_tag}_{timestamp}_rank{rank:02d}.parquet"
        df.to_parquet(out_dir / filename, index=False)
        self.prediction_buffer = []

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))
        spe = self.hparams.get("steps_per_epoch", None)
        if isinstance(spe, (int, float)) and math.isfinite(spe) and spe > 0:
            sched = OneCycleLR(
                opt,
                max_lr=float(self.hparams.learning_rate),
                epochs=int(self.trainer.max_epochs),
                steps_per_epoch=int(spe),
                pct_start=float(self.hparams.pct_start),
                anneal_strategy="cos",
                div_factor=float(self.hparams.div_factor),
                final_div_factor=float(self.hparams.final_div_factor),
            )
        else:
            nb = getattr(self.trainer, "num_training_batches", None)
            acc = int(getattr(self.trainer, "accumulate_grad_batches", 1)) or 1
            max_epochs = int(getattr(self.trainer, "max_epochs", 1)) or 1
            if isinstance(nb, (int, float)) and math.isfinite(nb) and nb > 0:
                total_steps = max(1, (int(nb) // acc) * max_epochs)
            else:
                total_steps = int(getattr(self.trainer, "max_steps", 0)) \
                            or int(getattr(self.trainer, "estimated_stepping_batches", 0)) \
                            or 1000
                total_steps = max(1, int(total_steps) // acc)
            sched = OneCycleLR(
                opt,
                max_lr=float(self.hparams.learning_rate),
                total_steps=int(total_steps),
                pct_start=float(self.hparams.pct_start),
                anneal_strategy="cos",
                div_factor=float(self.hparams.div_factor),
                final_div_factor=float(self.hparams.final_div_factor),
            )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1}}



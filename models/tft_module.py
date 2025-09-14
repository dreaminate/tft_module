import torch
import math
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, preds, targets, return_dict: bool = True):
        total_loss = 0.0
        details: dict[str, dict] = {}
        sig2 = F.softplus(self.log_vars)
        sig2 = torch.clamp(sig2, min=1e-2)
        log_sig2 = torch.log(sig2 + 1e-8)
        for i, (pred, targ, fn) in enumerate(zip(preds, targets, self.losses)):
            li = fn(pred, targ)
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

    def _batch_sym_per_idx(self, x):
        g = x["groups"]
        sym_idx = g[:, self.symbol_group_idx].long()
        per_idx = g[:, self.period_group_idx].long()
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
        # robustly parse y from dataset into per-target vectors
        y_list = self._parse_y_to_list(y, n_targets_expected=n_targets)
        y_enc = self._stack_y(y_list, self.device)
        sym_idx, per_idx = self._batch_sym_per_idx(x)
        y_for_loss = self._standardize_y(y_enc, sym_idx, per_idx)
        weighted_total, sub_losses = self.loss_fn(
            pred_list,
            [y_for_loss[:, i] for i in range(y_for_loss.size(1))],
            return_dict=True,
        )
        raws = torch.stack([v["raw"] for v in sub_losses.values()])
        w = self.loss_fn.w.to(raws)
        raw_total = (raws * w).sum()
        self.log(f"{stage}_loss_raw", raw_total, on_step=True, on_epoch=True, prog_bar=True, batch_size=y_enc.size(0))
        self.log(f"{stage}_loss", weighted_total, on_step=True, on_epoch=True, prog_bar=True, batch_size=y_enc.size(0))
        for name, item in sub_losses.items():
            self.log(f"{stage}/{name}_raw", item["raw"], on_step=True, on_epoch=True, batch_size=y_enc.size(0))
            self.log(f"{stage}/{name}_w", item["weighted"], on_step=True, on_epoch=True, batch_size=y_enc.size(0))
        preds_bt = torch.stack(pred_list, dim=1)
        return weighted_total, preds_bt, y_enc, x, sym_idx, per_idx

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
        pred_list = self.forward(x)
        preds_bt = torch.stack(pred_list, dim=1).float()
        preds_bt = preds_bt.contiguous()
        out = {"preds_std": preds_bt}
        if return_standardized:
            out.update({"preds_orig": None, "preds_antilog": None, "future_price": None})
            return out
        sym_idx, per_idx = self._batch_sym_per_idx(x)
        preds_orig, _ = self._destandardize_pred_and_true(preds_bt, preds_bt, sym_idx, per_idx)
        preds_orig = preds_orig.contiguous()
        out["preds_orig"] = preds_orig
        preds_antilog = torch.full_like(preds_orig, float("nan"))
        future_price = None
        if with_antilog:
            try:
                li = self.target_names.index("target_logreturn")
            except ValueError:
                li = None
            if li is not None:
                preds_antilog[:, li] = torch.expm1(preds_orig[:, li])
                if with_future_price:
                    base_close = None
                    if self.close_encoder_idx is not None and "encoder_cont" in x:
                        try:
                            base_close = x["encoder_cont"][:, -1, self.close_encoder_idx]
                        except Exception:
                            base_close = None
                    if base_close is None and self.close_decoder_idx is not None and "decoder_cont" in x:
                        try:
                            base_close = x["decoder_cont"][:, 0, self.close_decoder_idx]
                        except Exception:
                            base_close = None
                    if base_close is not None:
                        future_price = (base_close * torch.exp(preds_orig[:, li])).contiguous()
        out["preds_antilog"] = preds_antilog.contiguous()
        out["future_price"] = future_price
        return out

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

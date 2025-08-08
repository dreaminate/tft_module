import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_forecasting.models import TemporalFusionTransformer
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# 1) 混合多目标加权损失（手动权重 × 不确定性自动加权）
#    total = Σ_i w_i * ( L_i / σ_i^2 + log σ_i^2 )
# ============================================================
class HybridMultiLoss(nn.Module):
    def __init__(
        self,
        losses: nn.ModuleList,
        base_weights: torch.Tensor,
        loss_names: list[str] | None = None,
    ):
        super().__init__()
        assert len(losses) == base_weights.numel(), (
            f"损失函数数量({len(losses)}) 与权重长度({base_weights.numel()}) 不匹配"
        )
        self.losses = losses
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))  # 可学习不确定性
        self.register_buffer("w", base_weights.view(-1))        # 固定手动权重
        self.loss_names = loss_names or [f"loss_{i}" for i in range(len(losses))]

    def forward(self, preds, targets, return_dict: bool = True):
        """
        preds / targets: list[T]，每个元素形状 [B]
        """
        total_loss = 0.0
        details: dict[str, dict] = {}

        # σ^2 = softplus(log_vars) 保证正；再加 clamp 避免过小
        sig2 = F.softplus(self.log_vars)
        sig2 = torch.clamp(sig2, min=1e-2)
        log_sig2 = torch.log(sig2 + 1e-8)

        for i, (pred, targ, fn) in enumerate(zip(preds, targets, self.losses)):
            li = fn(pred, targ)                 # 原始子损失（标量）
            term = li / sig2[i] + log_sig2[i]  # 不确定性加权项
            weighted = self.w[i] * term        # 结合手动权重
            total_loss = total_loss + weighted
            if return_dict:
                name = self.loss_names[i]
                details[name] = {"raw": li.detach(), "weighted": weighted.detach()}

        return (total_loss, details) if return_dict else total_loss


# ============================================================
# 2) LightningModule 封装 TFT
# ============================================================
class MyTFTModule(LightningModule):
    """
    集成：
      - TemporalFusionTransformer 主模型
      - 混合多目标损失
      - 分类/回归指标
      - OneCycleLR（按 step 更新）
    约定：
      - batch 的 y 为 (list_of_targets, None)，其中 list_of_targets 长度=T
      - metrics_list: List[Tuple[str, torchmetrics.Metric]]，如 ("target_binarytrend_f1@1h", F1())
    """
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

        # 保存重要超参
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
        # —— 回归目标索引 & 统计表 —— #
        self.regression_targets = [
            "target_logreturn",
            "target_logsharpe_ratio",
            "target_breakout_count",
            "target_max_drawdown",
            "target_trend_persistence",
        ]
        self.reg_indices = [self.target_names.index(t) for t in self.regression_targets if t in self.target_names]

        if norm_pack is None:
            # 兜底（不用标准化）
            self.register_buffer("means_tbl", torch.zeros(1, 1, len(self.regression_targets)), persistent=True)
            self.register_buffer("stds_tbl",  torch.ones(1, 1, len(self.regression_targets)),  persistent=True)
            self.sym2idx = {}
            self.per2idx = {}
        else:
            means = torch.as_tensor(norm_pack["means"])  # [S,P,T_r]
            stds  = torch.as_tensor(norm_pack["stds"])   # [S,P,T_r]
            self.register_buffer("means_tbl", means, persistent=True)
            self.register_buffer("stds_tbl",  stds,  persistent=True)
            self.sym2idx = norm_pack.get("sym2idx", {})
            self.per2idx = norm_pack.get("per2idx", {})

        # —— 构建 TFT —— #
        tft_kwargs.pop("loss", None)
        tft_kwargs.pop("output_size", None)
        
        self.model = TemporalFusionTransformer.from_dataset(
            dataset, loss=None, output_size=output_size, **tft_kwargs
        )
        # 抑制无效注意力
        for m in self.model.modules():
            if hasattr(m, "mask_bias"):
                m.mask_bias = -1e4

        # —— 指标 —— #
        self.metrics_list = nn.ModuleList([m for _, m in metrics_list])
        self.metric_names = [n for n, _ in metrics_list]

        # 每个指标对应的目标列索引
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

        # —— group 索引 —— #
        self.period_group_idx = dataset.group_ids.index("period")
        self.symbol_group_idx = dataset.group_ids.index("symbol")

        # —— 周期映射 —— #
        if period_map is not None:
            self.period_map = period_map
        else:
            enc = dataset.categorical_encoders.get("period", None)
            classes_ = getattr(enc, "classes_", None)
            self.period_map = {i: c for i, c in enumerate(classes_)} if classes_ is not None else {}

        # —— close 索引：优先 encoder，其次 decoder —— #
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
    # -------------------- 小工具 --------------------
    def _batch_sym_per_idx(self, x):
        g = x["groups"]  # [B, num_group_ids]
        sym_idx = g[:, self.symbol_group_idx].long()
        per_idx = g[:, self.period_group_idx].long()
        return sym_idx, per_idx
    def _build_loss(self, loss_list, weights, target_names):
        # 1) 统一为 ModuleList / Tensor
        assert isinstance(loss_list, (list, nn.ModuleList)) and len(loss_list) > 0, "loss_list 不能为空"
        losses = nn.ModuleList(loss_list)

        base_w = torch.as_tensor(weights, dtype=torch.float32).view(-1)
        assert base_w.numel() == len(losses), f"weights({base_w.numel()}) != losses({len(losses)})"

        # 2) 可读的 loss 名称（与 target_names 对齐更容易看日志）
        if target_names and len(target_names) == len(losses):
            loss_names = [f"{t}_loss" for t in target_names]
        else:
            loss_names = [f"loss_{i}" for i in range(len(losses))]

        # 3) 真正创建混合损失
        self.loss_fn = HybridMultiLoss(
            losses=losses,
            base_weights=base_w,     # 会被 register_buffer 存成 self.loss_fn.w
            loss_names=loss_names,
        )
    def _standardize_y(self, y_enc, sym_idx, per_idx):
        """
        y_enc: [B, T] 原始目标；仅回归列标准化
        """
        if not self.reg_indices or self.means_tbl.numel() <= 1:
            return y_enc
        y_std = y_enc.clone()
        means_b = self.means_tbl[sym_idx, per_idx, :]  # [B, Tr]
        stds_b  = torch.clamp(self.stds_tbl[sym_idx, per_idx, :], min=1e-8)
        for local_t, global_t in enumerate(self.reg_indices):
            y_std[:, global_t] = (y_enc[:, global_t] - means_b[:, local_t]) / stds_b[:, local_t]
        return y_std

    def _destandardize_pred_and_true(self, preds_bt, y_enc, sym_idx, per_idx):
        """
        仅回归列反标准化；分类列原样返回
        """
        if not self.reg_indices or self.means_tbl.numel() <= 1:
            return preds_bt, y_enc
        preds = preds_bt.clone()
        ys    = y_enc.clone()
        means_b = self.means_tbl[sym_idx, per_idx, :]
        stds_b  = torch.clamp(self.stds_tbl[sym_idx, per_idx, :], min=1e-8)
        for local_t, global_t in enumerate(self.reg_indices):
            preds[:, global_t] = preds_bt[:, global_t] * stds_b[:, local_t] + means_b[:, local_t]
            ys[:, global_t]    = y_enc[:, global_t]    * stds_b[:, local_t] + means_b[:, local_t]
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
        cat = torch.cat([t.view(-1, 1).float() for t in y_list], dim=1)  # [B, T]
        return cat.to(device) if device else cat

    # -------------------- 前向 --------------------
    def forward(self, x):
        x = self._to_dev(x, self.device)
        out = self.model(x)["prediction"]  # [B, T] or [B]
        if isinstance(out, list):
            return [t.flatten() for t in out]
        if out.dim() == 1:
            out = out.unsqueeze(1)
        return [out[:, i].contiguous().view(-1) for i in range(out.size(1))]

    # -------------------- 共享步骤 --------------------
    def _shared_step(self, batch, stage):
        x, y = batch

        # 输入 NaN/Inf 早检
        for name in ("encoder_cont", "decoder_cont"):
            if name in x:
                ten = x[name]
                if not torch.isfinite(ten).all():
                    raise RuntimeError(f"[{stage}] {name} has NaN/Inf")

        y_list = y[0] if isinstance(y, (list, tuple)) else [y]
        y_enc = self._stack_y(y_list, self.device)       # [B, T]
        pred_list = self.forward(x)                      # list[T]，每个 [B]

        # 标准化目标（仅回归列）
        sym_idx, per_idx = self._batch_sym_per_idx(x)
        y_for_loss = self._standardize_y(y_enc, sym_idx, per_idx)

        # —— Loss —— #
        weighted_total, sub_losses = self.loss_fn(
            pred_list,
            [y_for_loss[:, i] for i in range(y_for_loss.size(1))],
            return_dict=True,
        )
        raws = torch.stack([v["raw"] for v in sub_losses.values()])
        w = self.loss_fn.w.to(raws)
        raw_total = (raws * w).sum()

        self.log(f"{stage}_loss_raw", raw_total, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=y_enc.size(0))
        self.log(f"{stage}_loss", weighted_total, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=y_enc.size(0))

        for name, item in sub_losses.items():
            self.log(f"{stage}/{name}_raw", item["raw"],
                     on_step=True, on_epoch=True, batch_size=y_enc.size(0))
            self.log(f"{stage}/{name}_w", item["weighted"],
                     on_step=True, on_epoch=True, batch_size=y_enc.size(0))

        preds_bt = torch.stack(pred_list, dim=1)  # [B, T]
        return weighted_total, preds_bt, y_enc, x, sym_idx, per_idx

    # -------------------- 训练/验证 --------------------
    def training_step(self, batch, batch_idx):
        total, *_ = self._shared_step(batch, "train")
        return total

    def validation_step(self, batch, batch_idx):
        total, preds, y_enc, x, sym_idx, per_idx = self._shared_step(batch, "val")

        # 回归指标在“原尺度”上评估
        preds_eval, y_eval = self._destandardize_pred_and_true(preds, y_enc, sym_idx, per_idx)

        # —— 按 period 计算指标 —— #
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
                prob = torch.sigmoid(preds[mask, col])  # 分类用原 logits 过 sigmoid
                t = y_enc[mask, col].float().clamp(0, 1).int()
                metric.update(prob, t)
                if name in self.confmats:
                    self.confmats[name].update((prob > 0.5).int(), t)
            else:
                metric.update(preds_eval[mask, col], y_eval[mask, col])

        return total

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        # —— 汇总指标 —— #
        for name, metric in zip(self.metric_names, self.metrics_list):
            try:
                val = metric.compute()
            except Exception:
                metric.reset()
                continue
            self.log(f"val_{name}", val, on_epoch=True, batch_size=1)
            metric.reset()

        # —— 混淆矩阵（若有分类指标）—— #
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

    # -------------------- 动态权重切换 --------------------
    def on_train_epoch_start(self):
        if self.current_epoch in self.loss_schedule:
            new = torch.tensor(
                self.loss_schedule[self.current_epoch],
                dtype=self.loss_fn.w.dtype,
                device=self.loss_fn.w.device,
            )
            self.loss_fn.w.copy_(new.view_as(self.loss_fn.w))

    # -------------------- 预测：标准化/原尺度/反log/未来价 --------------------
    def predict_step(
        self,
        batch,
        batch_idx,
        dataloader_idx: int = 0,
        return_standardized: bool = False,
        with_antilog: bool = True,
        with_future_price: bool = True,
    ):
        """
        返回 dict：
          - preds_std: 标准化空间 [B, T]
          - preds_orig: 反标准化后的原尺度 [B, T]
          - preds_antilog: 对 log 目标（如 target_logreturn）做 expm1，其它列 NaN [B, T]
          - future_price: 若能拿到基价 close，则 future_close = close * exp(logreturn_pred)
        """
        x, _ = batch

        # 1) 标准化空间的预测
        pred_list = self.forward(x)                         # list[T]，[B]
        preds_bt = torch.stack(pred_list, dim=1).float()    # [B, T]
        preds_bt = preds_bt.contiguous()

        out = {"preds_std": preds_bt}

        if return_standardized:
            out.update({"preds_orig": None, "preds_antilog": None, "future_price": None})
            return out

        # 2) 反标准化到原尺度（仅回归列）
        sym_idx, per_idx = self._batch_sym_per_idx(x)
        preds_orig, _ = self._destandardize_pred_and_true(preds_bt, preds_bt, sym_idx, per_idx)
        preds_orig = preds_orig.contiguous()
        out["preds_orig"] = preds_orig

        # 3) 反 log（只对“log 目标”生效）
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
                    # 优先：encoder 最后一根 close
                    if self.close_encoder_idx is not None and "encoder_cont" in x:
                        try:
                            base_close = x["encoder_cont"][:, -1, self.close_encoder_idx]
                        except Exception:
                            base_close = None
                    # 其次：decoder 第一步 close（你如果把 close 放进了 decoder）
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

    # -------------------- 优化器/调度器 --------------------
    def configure_optimizers(self):
    
        opt = torch.optim.Adam(self.parameters(), lr=float(self.hparams.learning_rate))

        spe = self.hparams.get("steps_per_epoch", None)
        if isinstance(spe, (int, float)) and math.isfinite(spe) and spe > 0:
            # ✅ 有 steps_per_epoch：用 epochs + steps_per_epoch（最稳）
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
            # ❗兜底：算一个有限的 total_steps
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

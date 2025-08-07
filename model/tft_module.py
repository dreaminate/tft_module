import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_forecasting.models import TemporalFusionTransformer
import torchmetrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# ------------------------------------------------------------
# 1. 混合多目标加权损失（手动权重 × 不确定性自动加权）
# ------------------------------------------------------------
class HybridMultiLoss(nn.Module):
    """
    对多个子损失进行加权求和：
    total_loss = Σ_i [ w_i * (precision_i * L_i + log_var_i) ]
    precision_i = exp(-log_var_i)，log_var_i 为 σ_i 的对数
    支持 return_dict=True 返回每个子损失的明细（不影响回传总损失）
    """
    def __init__(self, losses: nn.ModuleList, base_weights: torch.Tensor, loss_names: list[str] | None = None):
        super().__init__()
        # 确保损失函数数量与权重长度一致
        assert len(losses) == base_weights.numel(), \
            f"损失函数数量({len(losses)}) 与权重长度({base_weights.numel()}) 不匹配"
        self.losses = losses
        # 初始化 log variance，可学习参数
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))
        # 保存手动权重
        self.register_buffer('w', base_weights.view(-1))
        # 子损失名称
        if loss_names is None:
            self.loss_names = [f"loss_{i}" for i in range(len(losses))]
        else:
            assert len(loss_names) == len(losses), "loss_names 长度需与 losses 相同"
            self.loss_names = loss_names

    def forward(self, preds, targets, return_dict: bool = True):
        total_loss = 0.0
        details: dict[str, dict] = {}
        # 对每个子损失计算加权项
        for i, (pred, targ, fn) in enumerate(zip(preds, targets, self.losses)):
            li = fn(pred, targ)
            precision = torch.exp(-self.log_vars[i])
            term = precision * li + self.log_vars[i]
            weighted = self.w[i] * term
            total_loss += weighted
            if return_dict:
                name = self.loss_names[i]
                details[name] = {"raw": li.detach(), "weighted": weighted.detach()}
        return (total_loss, details) if return_dict else total_loss

# ------------------------------------------------------------
# 2. LightningModule 封装 TFT（已移除复合评分逻辑）
# ------------------------------------------------------------
class MyTFTModule(LightningModule):
    """
    该模块集成：
    1) TemporalFusionTransformer 主模型
    2) 混合多目标加权损失
    3) 单目标及多目标指标计算（分类/回归）
    4) One-Cycle 学习率调度
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
        **tft_kwargs,
    ):
        super().__init__()
        # 保存超参数
        self.save_hyperparameters({
            "learning_rate": learning_rate,
            "loss_schedule": loss_schedule or {},
            "pct_start": pct_start,
            "div_factor": div_factor,
            "final_div_factor": final_div_factor,
        })
        self.target_names = target_names

        # 初始化混合多目标损失函数
        losses = nn.ModuleList(loss_list)
        base_w = torch.tensor(weights, dtype=torch.float32)
        self.loss_fn = HybridMultiLoss(losses, base_w, loss_names=target_names)
        self.loss_schedule = loss_schedule or {}

        # 构建 TFT 模型
        tft_kwargs.pop('loss', None)
        tft_kwargs.pop('output_size', None)
        tft_kwargs.pop("scheduler_cfg", None) 
        self.model = TemporalFusionTransformer.from_dataset(
            dataset, loss=None, output_size=output_size, **tft_kwargs
        )
        # 调整注意力遮罩偏置
        for m in self.model.modules():
            if hasattr(m, 'mask_bias'):
                m.mask_bias = -1e4

        # 指标和混淆矩阵
        self.metrics_list = nn.ModuleList([m for _, m in metrics_list])
        self.metric_names = [n for n, _ in metrics_list]
        self.metric_target_idx = [
            next(i for i, t in enumerate(target_names) if n.startswith(t + "_"))
            for n in self.metric_names
        ]
        cls_suffix = ("_f1", "_precision", "_recall", "_accuracy", "_ap", "_auc")
        self.metric_is_cls = [
            any(n.split('@')[0].endswith(s) for s in cls_suffix)
            for n in self.metric_names
        ]
        self.confmats = nn.ModuleDict({
            name: torchmetrics.classification.BinaryConfusionMatrix()
            for name, is_cls in zip(self.metric_names, self.metric_is_cls) if is_cls
        })

        # 周期映射
        self.period_group_idx = dataset.group_ids.index("period")
        self.period_map = period_map or {
            i: c for i, c in enumerate(dataset.categorical_encoders["period"].classes_)
        }

    def _to_dev(self, obj, dev):
        """递归将对象移动到指定设备"""
        if torch.is_tensor(obj): return obj.to(dev)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_dev(o, dev) for o in obj)
        if isinstance(obj, dict):
            return {k: self._to_dev(v, dev) for k, v in obj.items()}
        return obj

    def _stack_y(self, y_list, device=None):
        """将多个目标张量按列拼接"""
        cat = torch.cat([t.view(-1, 1).float() for t in y_list], dim=1)
        return cat.to(device) if device else cat

    def forward(self, x):
        """前向计算，返回每个目标的一维向量"""
        x = self._to_dev(x, self.device)
        outs = self.model(x)["prediction"]
        return [t.flatten() for t in outs]

    def _shared_step(self, batch, stage):
        """
        train/val 公共步骤：
        - 计算预测值
        - 计算混合多目标损失
        - 记录损失日志
        """
        x, y = batch
        y_enc = self._stack_y(y[0], self.device)
        preds = torch.stack(self.forward(x), dim=1)
        total, sub = self.loss_fn(
            [preds[:, i] for i in range(preds.size(1))],
            [y_enc[:, i] for i in range(y_enc.size(1))],
            return_dict=True
        )
        self.log(f"{stage}_loss", total, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=y_enc.size(0))
        for name, d in sub.items():
            self.log(f"{stage}/{name}_raw", d["raw"], on_step=True, on_epoch=True,
                     batch_size=y_enc.size(0))
            self.log(f"{stage}/{name}_w", d["weighted"], on_step=True, on_epoch=True,
                     batch_size=y_enc.size(0))
        return total, preds, y_enc, x

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        total, *_ = self._shared_step(batch, 'train')
        return total

    def validation_step(self, batch, batch_idx):
        """验证步骤：更新各指标"""
        total, preds, y_enc, x = self._shared_step(batch, 'val')
        period_idx = x['groups'][:, self.period_group_idx].to(self.device)
        for idx, metric in enumerate(self.metrics_list):
            name = self.metric_names[idx]
            suffix = name.split("@")[-1]
            pid = next(k for k,v in self.period_map.items() if v==suffix)
            mask = period_idx == pid
            if not mask.any(): continue
            col = self.metric_target_idx[idx]
            t_slice = y_enc[mask, col]
            p_slice = preds[mask, col]
            if self.metric_is_cls[idx]:
                prob = torch.sigmoid(p_slice)
                metric.update(prob, t_slice.long())
            else:
                metric.update(p_slice, t_slice)
        return total

    def on_validation_epoch_end(self):
        """验证 epoch 结束后：记录指标 & 混淆矩阵"""
        super().on_validation_epoch_end()
        # 记录指标
        for name, metric in zip(self.metric_names, self.metrics_list):
            try:
                val = metric.compute()
            except Exception:
                metric.reset()
                continue
            self.log(f"val_{name}", val, on_epoch=True, batch_size=1)
            metric.reset()
        # 绘制并记录混淆矩阵
        for tag, cm in self.confmats.items():
            try:
                cm_val = cm.compute().cpu().numpy()
            except Exception:
                cm.reset()
                continue
            if cm_val.size:
                fig, ax = plt.subplots(figsize=(3,3))
                ConfusionMatrixDisplay(cm_val, display_labels=[0,1]) \
                    .plot(ax=ax, colorbar=False)
                self.logger.experiment.add_figure(f"val/confmat_{tag}", fig, self.current_epoch)
                plt.close(fig)
            cm.reset()

    def on_train_epoch_start(self):
        """训练 epoch 开始时，根据计划调整权重"""
        if self.current_epoch in self.loss_schedule:
            new = torch.tensor(
                self.loss_schedule[self.current_epoch],
                dtype=self.loss_fn.w.dtype,
                device=self.loss_fn.w.device
            )
            self.loss_fn.w.copy_(new.view_as(self.loss_fn.w))

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.hparams.pct_start,
                anneal_strategy="cos",
                div_factor=self.hparams.div_factor,
                final_div_factor=self.hparams.final_div_factor
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

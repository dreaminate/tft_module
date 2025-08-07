from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from typing import Dict

class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, weights: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.weights = weights
        self._missing_warned = set()

    def __getstate__(self):
        # 标记为 stateless callback，绕过 parent 检查
        state = self.__dict__.copy()
        return state

    def _normalize(self, metric_name: str, value: float) -> float:
        name = metric_name.lower()
        if any(k in name for k in ["rmse", "mae", "val_loss"]):
            normed = 1.0 / (1.0 + value)
        else:
            normed = float(value)
        return max(0.0, min(1.0, normed))

    def _compute_custom_score(self, trainer, pl_module):
        score = 0.0
        count = 0

        # # —— DEBUG：打印实际记录的 metrics keys 和预期 keys —— 
        # print(f"[CustomCheckpoint] actual callback_metrics keys: {list(trainer.callback_metrics.keys())}")
        # print(f"[CustomCheckpoint] expected weight keys     : {list(self.weights.keys())}")

        for metric_name, weight in self.weights.items():
            if metric_name in trainer.callback_metrics:
                val = trainer.callback_metrics[metric_name]
                if torch.is_tensor(val):
                    val = val.item()
                normed = self._normalize(metric_name, val)
                score += weight * normed
                count += 1
            else:
                # 每个没匹配上的 key 只 warn 一次
                if metric_name not in self._missing_warned:
                    # print(f"[CustomCheckpoint] ❌ missing metric: {metric_name}")
                    self._missing_warned.add(metric_name)

        if count == 0:
            print("[CustomCheckpoint] ⚠️ no metrics matched, returning 0.0 instead of -inf")
            return 0.0

        return score

    def on_validation_epoch_end(self, trainer, pl_module):
        composite_score = self._compute_custom_score(trainer, pl_module)
        trainer.callback_metrics[self.monitor] = composite_score
        pl_module.log(self.monitor, composite_score, on_epoch=True, prog_bar=True, batch_size=1)

        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"]
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
            trainer.callback_metrics["val_loss_for_ckpt"] = val_loss
            

        super().on_validation_epoch_end(trainer, pl_module)

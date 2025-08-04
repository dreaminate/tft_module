# === custom_checkpoint.py ===
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from typing import Dict


class CustomCheckpoint(ModelCheckpoint):
    def __init__(self, weights: Dict[str, float], **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def __getstate__(self):
        # ✅ 标记为 stateless callback，绕过 is_overridden 的 parent 检查
        state = self.__dict__.copy()
        return state

    def _compute_custom_score(self, trainer, pl_module):
        score = 0.0
        count = 0
        for metric_name, weight in self.weights.items():
            if metric_name in trainer.callback_metrics:
                val = trainer.callback_metrics[metric_name]
                if torch.is_tensor(val):
                    val = val.item()
                score += weight * val
                count += 1
            else:
                print(f"[⚠️ Warning] Metric '{metric_name}' not found in callback_metrics.")
        return score if count > 0 else float("-inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        composite_score = self._compute_custom_score(trainer, pl_module)
        trainer.callback_metrics[self.monitor] = composite_score
        pl_module.log(self.monitor, composite_score, on_epoch=True, prog_bar=True)

        if "val_loss" in trainer.callback_metrics:
            val_loss = trainer.callback_metrics["val_loss"]
            if torch.is_tensor(val_loss):
                val_loss = val_loss.item()
            trainer.callback_metrics["val_loss_for_ckpt"] = val_loss
            pl_module.log("val_loss_for_ckpt", val_loss, on_epoch=True)

        super().on_validation_epoch_end(trainer, pl_module)

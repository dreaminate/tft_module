# === Part 1: 多阶段 Loss 权重调度器 ===
# ➤ 放在 utils/loss_scheduler.py

class MultiStageLossScheduler:
    """
    支持在指定 epoch 切换 loss 权重
    示例：
        scheduler = MultiStageLossScheduler({
            0: [1.0, 1.0, 1.0],       # 第 0~9 epoch
            10: [2.0, 1.0, 0.5],     # 第 10~19 epoch
            20: [0.5, 2.0, 1.0],     # 第 20+ epoch
        })
        new_weights = scheduler.get_weights(epoch)
    """
    def __init__(self, stage_weights: dict[int, list[float]]):
        self.stage_weights = dict(sorted(stage_weights.items()))

    def get_weights(self, epoch: int):
        applicable = [e for e in self.stage_weights if e <= epoch]
        if not applicable:
            return None
        return self.stage_weights[max(applicable)]

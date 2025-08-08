import torch
from collections import defaultdict

def compute_raw_means(model, loss_fn, dataloader, device=None, warmup_steps=100):
    """
    进行 Warm-up 计算各子损失的平均值 raw_means。

    参数:
    - model:    LightningModule 或 nn.Module，需有 forward(x) 返回 list of preds per task
    - loss_fn:  HybridMultiLoss 实例（不需要 raw_means）
    - dataloader: PyTorch DataLoader，用于迭代训练数据
    - device:   torch.device，对模型和数据进行移动
    - warmup_steps: int，最多取 warmup_steps 个 batch 进行统计

    返回:
    - raw_means: torch.Tensor，shape [n_tasks]，各子损失的平均 raw loss
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    # 初始化
    n_tasks = len(loss_fn.losses)
    sum_losses = torch.zeros(n_tasks, device=device)
    count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            # 前向
            preds = model.forward(x.to(device))  # list of [B] tensors
            # y[0] 假设是 list of target tensors
            y_list = y[0]
            # 计算 raw losses
            for i, loss_module in enumerate(loss_fn.losses):
                raw = loss_module(preds[i], y_list[i].to(device))
                sum_losses[i] += raw.mean().item()  # 按 batch 平均
            count += 1
            if count >= warmup_steps:
                break

    raw_means = sum_losses / count
    return raw_means

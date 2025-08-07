# utils/stage_summary.py
import os, yaml

def save_stage_summary(log_name: str,
                       best_score: float,
                       best_val_loss: float,
                       save_dir: str = "stage_logs"):
    """Save one-line yaml summary for this training stage."""
    # 1️⃣   生成真正的路径
    cfg_path = os.path.join(save_dir, f"{log_name}.yaml")

    # 2️⃣   若目录不存在就创建
    os.makedirs(save_dir, exist_ok=True)

    # 3️⃣   写入 / 更新
    summary = dict(best_composite_score=float(best_score),
                   best_val_loss=float(best_val_loss))
    with open(cfg_path, "w") as f:
        yaml.safe_dump(summary, f)

    return cfg_path

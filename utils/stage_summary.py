import yaml
import pandas as pd
import os


def save_stage_summary(stage_name, score, val_loss, config_path="logs/train_stages.csv"):
    """
    将当前训练阶段指标记录到 CSV 文件中
    """
    record = {
        "stage": stage_name,
        "composite_score": score,
        "val_loss": val_loss,
    }

    with open("configs/model_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    record.update({
        "sampler_mode": cfg.get("sampler_mode"),
        "focus_period": cfg.get("focus_period"),
        "batch_size": cfg.get("batch_size"),
        "log_name": cfg.get("log_name"),
    })

    df = pd.DataFrame([record])

    os.makedirs("logs", exist_ok=True)
    if os.path.exists(config_path):
        df_old = pd.read_csv(config_path)
        df = pd.concat([df_old, df], ignore_index=True)

    df.to_csv(config_path, index=False)
    print(f"[📋] 阶段训练日志已写入: {config_path}")

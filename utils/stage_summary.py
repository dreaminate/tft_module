import os, yaml

def save_stage_summary(log_name: str,
                       best_score: float,
                       best_val_loss: float,
                       save_dir: str = "stage_logs"):
    cfg_path = os.path.join(save_dir, f"{log_name}.yaml")
    os.makedirs(save_dir, exist_ok=True)
    summary = dict(best_monitor=float(best_score), best_val_loss=float(best_val_loss))
    with open(cfg_path, "w") as f:
        yaml.safe_dump(summary, f)
    return cfg_path



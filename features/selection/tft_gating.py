from __future__ import annotations
import argparse
import os
from typing import Dict
import pandas as pd
import torch

from data.load_dataset import get_dataloaders
from models.tft_module import MyTFTModule


def extract_gating_from_ckpt(ckpt_path: str, model_cfg_path: str = "configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml", weights_cfg_path: str = "configs/weights_config.yaml", max_batches: int = 50) -> pd.DataFrame:
    import yaml
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(weights_cfg_path, "r", encoding="utf-8") as f:
        weights_cfg = yaml.safe_load(f)
    train_loader, val_loader, target_names, train_ds, periods, norm_pack = get_dataloaders(
        data_path=model_cfg["data_path"],
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=model_cfg.get("num_workers", 4),
        val_mode=model_cfg.get("val_mode", "days"),
        val_days=model_cfg.get("val_days", 252),
        val_ratio=model_cfg.get("val_ratio", 0.2),
    )
    dummy_steps = max(1, len(train_loader))
    model = MyTFTModule.load_from_checkpoint(
        ckpt_path,
        dataset=train_ds,
        steps_per_epoch=dummy_steps,
        norm_pack=norm_pack,
        loss_list=[],
        weights=weights_cfg.get("custom_weights", [1.0] * len(target_names)),
        output_size=[1] * len(target_names),
        metrics_list=[],
        target_names=target_names,
        period_map={i: p for i, p in enumerate(periods)},
        learning_rate=model_cfg["learning_rate"],
        loss_schedule=model_cfg.get("loss_schedule", {}),
        hidden_size=model_cfg["hidden_size"],
        lstm_layers=model_cfg["lstm_layers"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batches = 0
    all_raw = []
    with torch.no_grad():
        for b in val_loader:
            x, y = b
            x = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            raw = model.model(x)
            all_raw.append(raw)
            batches += 1
            if batches >= max_batches:
                break
    raw0 = all_raw[0]
    inter = model.model.interpret_output(raw0, reduction="mean")
    scores: Dict[str, float] = {}
    for key in ("encoder_variables", "decoder_variables"):
        try:
            d = inter.get(key, {})
            for name, val in d.items():
                scores[name] = scores.get(name, 0.0) + float(val)
        except Exception:
            pass
    out = pd.DataFrame({"feature": list(scores.keys()), "score": list(scores.values())}).sort_values("score", ascending=False)
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract TFT gating (variable selection) scores from checkpoint")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--out", type=str, default="reports/feature_evidence/tft_gating.csv")
    ap.add_argument("--max-batches", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df = extract_gating_from_ckpt(args.ckpt, max_batches=args.max_batches)
    df.to_csv(args.out, index=False)
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()

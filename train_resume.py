import os
import yaml
import torch
import argparse
from typing import Optional
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from models.tft_module import MyTFTModule
from data.load_dataset import get_dataloaders
from utils.loss_factory import get_losses_by_targets
from utils.metric_factory import get_metrics_by_targets

import warnings

# 只忽略 torchmetrics 中 “compute before update” 的警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*compute.*before the.*update.*method.*"
)

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True


import os
def _load_targets(expert: str | None = None):
    import yaml
    tgt_path = os.path.join("configs", "targets.yaml")
    if not os.path.exists(tgt_path):
        return None
    with open(tgt_path, "r", encoding="utf-8") as f:
        all_cfg = yaml.safe_load(f) or {}
    experts = (all_cfg or {}).get("experts", {})
    default_expert = (all_cfg or {}).get("default_expert", None)
    exp = expert or os.environ.get("EXPERT") or default_expert
    if not exp:
        return None
    entry = experts.get(exp, None)
    if not entry:
        raise ValueError(f"Expert '{exp}' not found in configs/targets.yaml")
    model_type = entry.get("model_type", "tft")
    targets = entry.get("targets", [])
    return {"expert": exp, "model_type": model_type, "targets": targets}

def _resolve_weights(weight_cfg: dict, target_names: list[str]) -> list[float]:
    by_name = weight_cfg.get("custom_weights_by_target")
    if isinstance(by_name, dict):
        return [float(by_name.get(t, 1.0)) for t in target_names]
    lst = weight_cfg.get("custom_weights")
    if isinstance(lst, (list, tuple)) and len(lst) == len(target_names):
        return [float(x) for x in lst]
    return [1.0] * len(target_names)

def _parse_args():
    parser = argparse.ArgumentParser(description="Resume training with expert config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to expert leaf model_config.yaml under configs/experts/...",
    )
    parser.add_argument(
        "--expert",
        type=str,
        default=None,
        help="Override expert name to use from configs/targets.yaml",
    )
    return parser.parse_args()


def _extract_meta_from_cfg_path(cfg_path: str) -> dict:
    parts = os.path.normpath(cfg_path).split(os.sep)
    meta = {"expert": None, "period": None, "modality": None, "symbol": None}
    try:
        if len(parts) >= 5 and "experts" in parts:
            idx = parts.index("experts")
            meta["expert"] = parts[idx + 1]
            meta["period"] = parts[idx + 2] if len(parts) > idx + 2 else None
            meta["modality"] = parts[idx + 3] if len(parts) > idx + 3 else None
            maybe_symbol = parts[idx + 4] if len(parts) > idx + 4 else None
            if maybe_symbol and maybe_symbol != "model_config.yaml":
                meta["symbol"] = maybe_symbol
    except ValueError:
        pass
    return meta


def _find_nearby_config(basename: str, cfg_path: str, expert: Optional[str]) -> Optional[str]:
    leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
    cand = [os.path.join(leaf_dir, basename)]
    if expert:
        cand.append(os.path.join("configs", "experts", expert, basename))
    cand.append(os.path.join("configs", basename))
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


def _require_leaf_yaml(cfg_path: str, filenames: list[str]) -> dict:
    leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
    found = {}
    missing = []
    for name in filenames:
        p = os.path.join(leaf_dir, name)
        if os.path.exists(p):
            found[name] = p
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"缺少必要的叶子配置文件: {', '.join(missing)}. 请在 {leaf_dir} 下提供这些 YAML 文件。"
        )
    return found


def main():
    args = _parse_args()
    cfg_path = args.config
    # ===== 读取配置 =====
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    resume_ckpt = model_cfg.get("resume_ckpt")
    assert resume_ckpt and os.path.exists(resume_ckpt), f"Checkpoint 未找到: {resume_ckpt}"
    print(f"🔄 从 checkpoint 续训: {resume_ckpt}")

    inferred = _extract_meta_from_cfg_path(cfg_path)
    expert_name = args.expert or model_cfg.get("expert") or inferred.get("expert")
    exp_name = expert_name or "default"
    # 叶子 targets.yaml 决定训练目标
    leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
    tpath = os.path.join(leaf_dir, "targets.yaml")
    assert os.path.exists(tpath), f"缺少叶子 targets.yaml: {tpath}"
    with open(tpath, "r", encoding="utf-8") as f:
        leaf_targets_obj = yaml.safe_load(f) or {}
    model_type = leaf_targets_obj.get("model_type", "tft")
    targets_override = (leaf_targets_obj.get("targets", []) or None)
    if model_type != "tft":
        raise NotImplementedError(f"model_type '{model_type}' not supported yet")

    # ===== 数据加载 =====
    # 基于配置过滤周期/符号（若给出）
    cfg_period = model_cfg.get("period") or inferred.get("period")
    cfg_symbols = model_cfg.get("symbols") or ([] if inferred.get("symbol") is None else [inferred.get("symbol")])
    sel_feats_path = _find_nearby_config("selected_features.txt", cfg_path, expert_name)

    train_loader, val_loader, target_names, train_ds, periods ,norm_pack= get_dataloaders(
        data_path=model_cfg["data_path"],
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=model_cfg.get("num_workers", 4),
        val_mode=model_cfg.get("val_mode", "days"),
        val_days=model_cfg.get("val_days", 252),
        val_ratio=model_cfg.get("val_ratio", 0.2),
        targets_override=targets_override,
        periods=[str(cfg_period)] if cfg_period else None,
        symbols=[str(s) for s in cfg_symbols] if cfg_symbols else None,
        selected_features_path=sel_feats_path,
    )
    enc = train_ds.categorical_encoders.get("period", None)
    classes_ = getattr(enc, "classes_", None)
    if classes_ is not None:
        period_map = {i: c for i, c in enumerate(classes_)}
    else:
        period_map = {i: p for i, p in enumerate(periods)}
    # ===算有效 steps_per_epoch（考虑梯度累积）===
    steps_per_epoch = len(train_loader)  # 你这个 DataLoader 一定有 len()
    accum = int(model_cfg.get("accumulate", 1)) or 1
    steps_per_epoch_eff = max(1, steps_per_epoch // accum)
    # 权重：优先 targets.yaml -> weights.by_target/custom；否则回退叶子 weights_config.yaml；都没有则全1
    weights_section = leaf_targets_obj.get("weights", {}) or {}
    if isinstance(weights_section, dict) and weights_section.get("by_target"):
        weight_cfg = {"custom_weights_by_target": weights_section["by_target"]}
    elif isinstance(weights_section, dict) and isinstance(weights_section.get("custom"), (list, tuple)):
        weight_cfg = {"custom_weights": weights_section.get("custom")}
    else:
        leaf_weights_yaml = os.path.join(leaf_dir, "weights_config.yaml")
        weight_cfg = yaml.safe_load(open(leaf_weights_yaml, "r", encoding="utf-8")) if os.path.exists(leaf_weights_yaml) else {}
    # ===== 构建模型 =====
    out_sizes = [1] * len(target_names)
    output_size_arg = out_sizes[0] if len(out_sizes) == 1 else out_sizes
    model = MyTFTModule.load_from_checkpoint(
        checkpoint_path=resume_ckpt,
        steps_per_epoch=steps_per_epoch_eff,
        norm_pack=norm_pack, 
        dataset=train_ds,
        loss_list=get_losses_by_targets(target_names),
        weights=_resolve_weights(weight_cfg, target_names),
        output_size=output_size_arg,
        metrics_list=get_metrics_by_targets(target_names),
        target_names=target_names,
        period_map=period_map,
        learning_rate=model_cfg["learning_rate"],
        loss_schedule=model_cfg.get("loss_schedule", {}),
        hidden_size=model_cfg["hidden_size"],
        lstm_layers=model_cfg["lstm_layers"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        log_interval=model_cfg.get("log_interval", 50),
        log_val_interval=model_cfg.get("log_val_interval", 1),
        strict=False,
    )

    # ===== 日志 & 回调 =====
    
    period = model_cfg.get("period") or inferred.get("period") or "_"
    modality = model_cfg.get("modality_set") or model_cfg.get("modality") or inferred.get("modality") or "_"
    logs_root = os.path.join("lightning_logs", "experts", exp_name or "default", str(period), str(modality), model_type)
    os.makedirs(logs_root, exist_ok=True)
    logger = TensorBoardLogger(logs_root, name=model_cfg.get("log_name", "tft_multi"), flush_secs=10)
    # 监控指标：统一使用验证损失
    monitor_key = "val_loss_epoch"
    monitor_mode = "min"
    ckpt_cb = ModelCheckpoint(
        monitor=monitor_key,
        filename="epoch{epoch:02d}",
        save_top_k=3,
        save_last=True,
        mode=monitor_mode,
        dirpath=os.path.join("checkpoints", "experts", exp_name or "default", str(period), str(modality), model_type),
    )
    early_stopping = EarlyStopping(
        monitor=monitor_key,
        patience=model_cfg.get("early_stop_patience", 5),
        mode=monitor_mode,
    )

    # 保存配置到对应专家分桶
    cfg_dir = os.path.join(logs_root, "configs")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, "model_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f)
    used = {"expert": exp_name, "model_type": model_type, "targets": targets_override or []}
    with open(os.path.join(cfg_dir, "targets_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(used, f)

    # ===== 训练 =====
    trainer = Trainer(
        max_epochs=model_cfg["max_epochs"],
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        precision=model_cfg.get("precision", "16-mixed"),
        gradient_clip_val=model_cfg.get("grad_clip", 0.2),
        accumulate_grad_batches=model_cfg.get("accumulate", 1),
        callbacks=[early_stopping, ckpt_cb, LearningRateMonitor(logging_interval="step")],
        logger=logger,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # ===== 完成 =====
    print("🔚 续训完成")

if __name__ == "__main__":
    main()

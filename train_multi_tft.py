import os
import yaml
import torch
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before other imports
from typing import Optional
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from models.tft_module import MyTFTModule
from data.load_dataset import get_dataloaders
from utils.loss_factory import get_losses_by_targets
from utils.metric_factory import get_metrics_by_targets
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from utils.mp_start import ensure_start_method
import warnings
warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*compute.*before the.*update.*method.*"
    )
# -- PyTorch 全局加速 --
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_targets(expert: str | None = None):
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
        return None
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
    # 兜底：全 1
    return [1.0] * len(target_names)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-TFT by expert config")
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
        help="Override expert name（默认读取叶子 targets.yaml，必要时可指定全局 fallback）",
    )
    return parser.parse_args()


def _extract_meta_from_cfg_path(cfg_path: str) -> dict:
    """Try to infer {expert, period, modality, symbol} from the path under configs/experts.

    E.g., configs/experts/Alpha-Dir-TFT/1h/base/model_config.yaml
    """
    parts = os.path.normpath(cfg_path).split(os.sep)
    meta = {"expert": None, "period": None, "modality": None, "symbol": None}
    try:
        if len(parts) >= 5:
            # [..., configs, experts, <expert>, <period>, <modality>, (<symbol>), model_config.yaml]
            idx = parts.index("experts")
            expert = parts[idx + 1]
            period = parts[idx + 2] if len(parts) > idx + 2 else None
            modality = parts[idx + 3] if len(parts) > idx + 3 else None
            # optional symbol layer
            maybe_symbol = parts[idx + 4] if len(parts) > idx + 4 else None
            if maybe_symbol and maybe_symbol != "model_config.yaml":
                meta["symbol"] = maybe_symbol
            meta["expert"] = expert
            meta["period"] = period
            meta["modality"] = modality
    except ValueError:
        pass
    return meta


def _find_nearby_config(basename: str, cfg_path: str, expert: Optional[str]) -> Optional[str]:
    """按优先级查找同名配置文件：
    1) 叶子目录（cfg 文件所在目录）
    2) 专家根目录（configs/experts/<Expert>/）
    3) 全局 configs/
    """
    leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
    cand = [
        os.path.join(leaf_dir, basename),
    ]
    if expert:
        cand.append(os.path.join("configs", "experts", expert, basename))
    cand.append(os.path.join("configs", basename))
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


def _load_dataset_config(cfg_path: str, expert: Optional[str]) -> dict:
    """按优先级查找 datasets.yaml 并加载。"""
    p = _find_nearby_config("datasets.yaml", cfg_path, expert)
    if p and os.path.exists(p):
        return load_yaml(p)
    # Fallback for old structure with separate files
    modality = (_extract_meta_from_cfg_path(cfg_path) or {}).get("modality")
    if modality:
        basename = f"{modality}.yaml"
        p = _find_nearby_config(f"datasets/{basename}", cfg_path, expert)
        if p and os.path.exists(p):
            return load_yaml(p)
    return {}


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
    model_cfg = load_yaml(cfg_path)
    # 优先使用命令行 --expert，其次 config 中的 expert 字段，其次从路径推断，再次 default_expert
    inferred = _extract_meta_from_cfg_path(cfg_path)
    expert_name = args.expert or model_cfg.get("expert") or inferred.get("expert")
    
    # 新增：加载数据集配置
    dataset_cfg = _load_dataset_config(cfg_path, expert_name)
    pinned_features_cfg = dataset_cfg.get("pinned_features", {})

    # 使用叶子 targets.yaml 决定目标与类型
    exp_name = expert_name or "default"
    leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
    # 强制要求 targets.yaml 存在
    _require_leaf_yaml(cfg_path, ["targets.yaml"])
    with open(os.path.join(leaf_dir, "targets.yaml"), "r", encoding="utf-8") as f:
        leaf_targets_obj = yaml.safe_load(f) or {}
    model_type = leaf_targets_obj.get("model_type", "tft")
    targets_override = (leaf_targets_obj.get("targets", []) or None)
    output_head_cfg = leaf_targets_obj.get("output_head") or {}
    symbol_weight_cfg = leaf_targets_obj.get("symbol_weights") or {}
    symbol_limit = leaf_targets_obj.get("symbols") or None
    if model_type != "tft":
        raise NotImplementedError(f"model_type '{model_type}' not supported yet")

    # ===== 数据加载 =====
    # 过滤周期/符号（若配置中给出）
    cfg_period = model_cfg.get("period") or inferred.get("period")
    cfg_symbols = model_cfg.get("symbols") or ([] if inferred.get("symbol") is None else [inferred.get("symbol")])
    if symbol_limit:
        cfg_symbols = [str(s) for s in symbol_limit]
    # 权重来源：优先 targets.yaml -> weights.by_target/custom；否则回退叶子 weights_config.yaml；都没有则全1
    weights_section = leaf_targets_obj.get("weights", {}) or {}
    if isinstance(weights_section, dict) and weights_section.get("by_target"):
        weight_cfg = {"custom_weights_by_target": weights_section["by_target"]}
    elif isinstance(weights_section, dict) and isinstance(weights_section.get("custom"), (list, tuple)):
        weight_cfg = {"custom_weights": weights_section.get("custom")}
    else:
        leaf_weights_yaml = os.path.join(leaf_dir, "weights_config.yaml")
        weight_cfg = load_yaml(leaf_weights_yaml) if os.path.exists(leaf_weights_yaml) else {}
    sel_feats_path = _find_nearby_config("selected_features.txt", cfg_path, expert_name)

    train_loader, val_loader, target_names, train_ds, periods, norm_pack = get_dataloaders(
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
        pinned_features_cfg=pinned_features_cfg,
        modality=inferred.get("modality"),
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

    # ===== 构建模型 =====
    resolved_period = model_cfg.get("period") or inferred.get("period") or "_"
    resolved_modality = model_cfg.get("modality_set") or model_cfg.get("modality") or inferred.get("modality") or "_"

    schema_version = model_cfg.get("schema_version", "schema_v1")
    data_version = model_cfg.get("data_version", "data_unknown")
    expert_version = model_cfg.get("expert_version", "expert_unknown")
    train_window_id = model_cfg.get("train_window_id", "window_unknown")

    out_sizes = [1] * len(target_names)
    cls_keywords = ("binary", "prob", "detect", "outlier")
    cls_targets = [t for t in target_names if any(k in t for k in cls_keywords)]
    if cls_targets:
        primary_target = cls_targets[0]
        monitor_metric = f"val_{primary_target}_ap@{resolved_period}"
        monitor_mode = 'max'
    else:
        primary_target = target_names[0]
        monitor_metric = f"val_{primary_target}_rmse@{resolved_period}"
        monitor_mode = 'min'

    output_size_arg = out_sizes[0] if len(out_sizes) == 1 else out_sizes
    model = MyTFTModule(
        dataset=train_ds,
        steps_per_epoch=steps_per_epoch_eff,
        norm_pack=norm_pack,  
        loss_list=get_losses_by_targets(target_names),
        weights=_resolve_weights(weight_cfg or {}, target_names),
        output_size=output_size_arg,
        metrics_list=get_metrics_by_targets(target_names, horizons=[str(resolved_period)]),
        target_names=target_names,
        period_map=period_map,
        learning_rate=model_cfg["learning_rate"],
        loss_schedule=model_cfg.get("loss_schedule", {}),
        hidden_size=model_cfg["hidden_size"],
        lstm_layers=model_cfg["lstm_layers"],
        output_head_cfg=output_head_cfg,
        symbol_weight_map=symbol_weight_cfg,
        expert_name=exp_name,
        period_name=str(resolved_period),
        modality_name=str(resolved_modality),
        schema_version=schema_version,
        data_version=data_version,
        expert_version=expert_version,
        train_window_id=train_window_id,
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        log_interval=model_cfg.get("log_interval", 100),
        log_val_interval=model_cfg.get("log_val_interval", 1),
    )

    # ===== 日志 & 回调 =====
    # 组织日志路径：experts/<exp>/<period>/<modality>/<model_type>
    period = resolved_period
    modality = resolved_modality
    log_root = os.path.join("lightning_logs", "experts", exp_name or "default", str(period), str(modality), model_type)
    os.makedirs(log_root, exist_ok=True)
    logger = TensorBoardLogger(log_root, name=model_cfg.get("log_name", "tft_multi"))
    # 监控指标：统一使用验证损失（按你的要求）
    monitor_key = monitor_metric
    monitor_mode = monitor_mode
    ckpt_dir = os.path.join("checkpoints", "experts", exp_name or "default", str(period), str(modality), model_type)
    ckpt_cb = ModelCheckpoint(
        monitor=monitor_key,
        mode=monitor_mode,
        filename="epoch{epoch:02d}",
        save_top_k=3,
        save_last=True,
        dirpath=ckpt_dir,
    )
    early_stop = EarlyStopping(monitor=monitor_key, patience=model_cfg.get("early_stop_patience", 5), mode=monitor_mode)

    # 保存配置文件快照
    os.makedirs(os.path.join(log_root, "configs"), exist_ok=True)
    with open(os.path.join(log_root, "configs", "model_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(model_cfg, f)
    used = {"expert": exp_name, "model_type": model_type, "targets": targets_override or [], "symbols": cfg_symbols}
    with open(os.path.join(log_root, "configs", "targets_used.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(used, f)
    symbol_meta = {
        "symbol_classes": [str(s) for s in (norm_pack or {}).get("symbol_classes", [])],
        "sym2idx": {str(k): int(v) for k, v in ((norm_pack or {}).get("sym2idx", {}) or {}).items()},
        "symbol_weights": symbol_weight_cfg,
    }
    with open(os.path.join(log_root, "configs", "symbol_mapping.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(symbol_meta, f)
    
    # ===== 训练 =====
    devices_cfg = model_cfg.get("devices", 1)
    strategy = model_cfg.get("strategy")
    multi_device = False
    if isinstance(devices_cfg, int):
        multi_device = devices_cfg != 1
    elif isinstance(devices_cfg, (list, tuple)):
        multi_device = len(devices_cfg) > 1
    if strategy is None and multi_device:
        strategy = "ddp_find_unused_parameters_true"

    trainer_kwargs = dict(
        log_every_n_steps=int(model_cfg.get("log_every_n_steps", model_cfg.get("log_interval", 100)) or 100),
        max_epochs=model_cfg["max_epochs"],
        accelerator="gpu",
        devices=devices_cfg,
        precision=model_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=model_cfg.get("grad_clip", 0.2),
        accumulate_grad_batches=model_cfg.get("accumulate", 1),
        callbacks=[
            early_stop,
            ckpt_cb,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        enable_progress_bar=True,
    )
    if strategy is not None:
        trainer_kwargs["strategy"] = strategy
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    ensure_start_method()
    main()
#    python train_multi_tft.py


# === warm_start_train.py ===
# 多周期 TFT 多目标训练入口，仅加载模型权重（结构需兼容）
if __name__ == "__main__":
    import os
    import yaml
    import torch
    import argparse
    from lightning.pytorch import Trainer
    from models.tft_module import MyTFTModule
    from data.load_dataset import get_dataloaders
    from utils.loss_factory import get_losses_by_targets
    from utils.metric_factory import get_metrics_by_targets
    from utils.mp_start import ensure_start_method
    from utils.checkpoint_utils import load_partial_weights
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.loggers import TensorBoardLogger
    from utils.stage_summary import save_stage_summary
    from tft.utils.run_helper import prepare_run_dirs
    ensure_start_method()
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    # === 读取参数与配置 ===
    parser = argparse.ArgumentParser(description="Warm-start training with expert config")
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
        help="Override expert name（默认读取叶子 targets.yaml，可指定全局 fallback）",
    )
    args = parser.parse_args()
    cfg_path = args.config
    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    weight_cfg = {}

    # === 读取 targets.yaml（按专家） ===
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

    # 从路径推断元信息
    def _infer_meta_from_path(pth: str):
        parts = os.path.normpath(pth).split(os.sep)
        meta = {"expert": None, "period": None, "modality": None}
        try:
            if len(parts) >= 5 and "experts" in parts:
                idx = parts.index("experts")
                meta["expert"] = parts[idx + 1]
                meta["period"] = parts[idx + 2] if len(parts) > idx + 2 else None
                meta["modality"] = parts[idx + 3] if len(parts) > idx + 3 else None
        except ValueError:
            pass
        return meta

    inferred = _infer_meta_from_path(cfg_path)
    expert_name = args.expert or model_cfg.get("expert") or inferred.get("expert")
    exp_name = expert_name or "default"
    # 读取叶子 targets.yaml
    leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
    tpath = os.path.join(leaf_dir, "targets.yaml")
    assert os.path.exists(tpath), f"缺少叶子 targets.yaml: {tpath}"
    with open(tpath, "r", encoding="utf-8") as f:
        leaf_targets_obj = yaml.safe_load(f) or {}
    model_type = leaf_targets_obj.get("model_type", "tft")
    targets_override = (leaf_targets_obj.get("targets", []) or None)
    if model_type != "tft":
        raise NotImplementedError(f"model_type '{model_type}' not supported yet")

    output_head_cfg = leaf_targets_obj.get("output_head") or {}
    symbol_weight_cfg = leaf_targets_obj.get("symbol_weights") or {}

    # === 加载数据 ===
    resolved_period = model_cfg.get("period") or inferred.get("period") or "_"
    resolved_modality = model_cfg.get("modality_set") or model_cfg.get("modality") or inferred.get("modality") or "_"

    # 基于配置过滤周期/符号（若给出）
    cfg_period = model_cfg.get("period") or inferred.get("period")
    cfg_symbols = model_cfg.get("symbols") or ([] if inferred.get("symbol") is None else [inferred.get("symbol")])
    # 叶子/专家/全局优先查找特征白名单与权重
    def _find_nearby_config(basename: str, cfg_path: str, expert: str | None):
        leaf_dir = os.path.dirname(os.path.abspath(cfg_path))
        cand = [os.path.join(leaf_dir, basename)]
        if expert:
            cand.append(os.path.join("configs", "experts", expert, basename))
        cand.append(os.path.join("configs", basename))
        for p in cand:
            if p and os.path.exists(p):
                return p
        return None
    # 强制：叶子目录必须提供权重配置（YAML）
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
    sel_feats_path = _find_nearby_config("selected_features.txt", cfg_path, expert_name)
    # 权重：优先 targets.yaml -> weights.by_target/custom；否则回退叶子 weights_config.yaml；都没有则全1
    weights_section = leaf_targets_obj.get("weights", {}) or {}
    if isinstance(weights_section, dict) and weights_section.get("by_target"):
        weight_cfg = {"custom_weights_by_target": weights_section["by_target"]}
        weights_yaml_path = None
    elif isinstance(weights_section, dict) and isinstance(weights_section.get("custom"), (list, tuple)):
        weight_cfg = {"custom_weights": weights_section.get("custom")}
        weights_yaml_path = None
    else:
        weights_yaml_path = os.path.join(leaf_dir, "weights_config.yaml") if os.path.exists(os.path.join(leaf_dir, "weights_config.yaml")) else None

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
    )
    period_map = {i: p for i, p in enumerate(periods)}
    # === 获取 Loss 与 Metric ===
    all_losses = get_losses_by_targets(target_names)
    if weights_yaml_path and os.path.exists(weights_yaml_path):
        with open(weights_yaml_path, "r", encoding="utf-8") as f:
            weight_cfg = yaml.safe_load(f) or {}
    all_weights = (weight_cfg.get("custom_weights") if isinstance(weight_cfg, dict) else None) or [1.0] * len(target_names)
    metrics_list = get_metrics_by_targets(target_names)

    # === 初始化模型结构 ===
    steps_per_epoch = len(train_loader)
    accum = int(model_cfg.get("accumulate", 1)) or 1
    steps_per_epoch_eff = max(1, steps_per_epoch // accum)

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
        loss_list=all_losses,
        weights=all_weights,
        output_size=output_size_arg,
        metrics_list=metrics_list,
        target_names=target_names,
        period_map=period_map,
        learning_rate=model_cfg["learning_rate"],
        loss_schedule=model_cfg.get("loss_schedule", {}),
        hidden_size=model_cfg["hidden_size"],
        lstm_layers=model_cfg["lstm_layers"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        output_head_cfg=output_head_cfg,
        symbol_weight_map=symbol_weight_cfg,
        expert_name=exp_name,
        period_name=str(resolved_period),
        modality_name=str(resolved_modality),
        schema_version=schema_version,
        data_version=data_version,
        expert_version=expert_version,
        train_window_id=train_window_id,
    )

    # === 加载预训练模型权重（不恢复优化器/epoch） ===
    warm_start_ckpt = model_cfg.get("warm_start_ckpt")
    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        print(f"🔥 Warm start from checkpoint: {warm_start_ckpt}")
        load_partial_weights(model, warm_start_ckpt)
    else:
        raise FileNotFoundError(f"❌ warm_start_ckpt not found: {warm_start_ckpt}")

    # === 设置日志与 Callback ===
    run_dirs = prepare_run_dirs()
    period = resolved_period
    modality = resolved_modality
    logs_root = os.path.join("lightning_logs", "experts", exp_name or "default", str(period), str(modality), model_type)
    os.makedirs(logs_root, exist_ok=True)
    logger = TensorBoardLogger(logs_root, name=model_cfg.get("log_name", "tft_multi"))
    # 监控指标：统一使用验证损失
    monitor_key = monitor_metric
    monitor_mode = monitor_mode
    ckpt_cb = ModelCheckpoint(
        monitor=monitor_key,
        filename="epoch{epoch:02d}",
        save_top_k=3,
        save_last=True,
        mode=monitor_mode,
        dirpath=os.path.join("checkpoints", "experts", exp_name or "default", str(period), str(modality), model_type),
    )
    with open(f"{run_dirs['cfg']}/model_config.yaml", "w") as f:
        yaml.safe_dump(model_cfg, f)
    early_stopping = EarlyStopping(
        monitor=monitor_key,
        patience=model_cfg.get("early_stop_patience", 5),
        mode=monitor_mode,
    )

    # === 启动训练 ===
    trainer = Trainer(
        max_epochs=model_cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=model_cfg.get("precision", "16-mixed"),
        gradient_clip_val=model_cfg.get("grad_clip", 0.2),
        accumulate_grad_batches=model_cfg.get("accumulate", 1),
        callbacks=[early_stopping, ckpt_cb, LearningRateMonitor(logging_interval="step")],
        logger=logger,
        log_every_n_steps=int(model_cfg.get("log_every_n_steps", model_cfg.get("log_interval", 100)) or 100),
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        # 记录最佳 val_loss
        val_loss = ckpt.get("callbacks", {}).get("ModelCheckpoint", {}).get("best_model_score", float("nan"))
        save_stage_summary(model_cfg.get("log_name", "unnamed_stage"), float("nan"), float(val_loss) if val_loss is not None else float("nan"))

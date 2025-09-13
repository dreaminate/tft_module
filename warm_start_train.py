# === warm_start_train.py ===
# 多周期 TFT 多目标训练入口，仅加载模型权重（结构需兼容）
if __name__ == "__main__":
    import os
    import yaml
    import torch
    from lightning.pytorch import Trainer
    from tft.models import MyTFTModule
    from tft.data import get_dataloaders
    from tft.utils import get_losses_by_targets, get_metrics_by_targets
    from tft.utils.checkpoint_utils import load_partial_weights
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import TensorBoardLogger
    from tft.callbacks import CustomCheckpoint
    from tft.utils.stage_summary import save_stage_summary
    from tft.utils.composite import filter_weights_by_period
    from tft.utils.run_helper import prepare_run_dirs
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True
    # === 读取配置 ===
    with open("configs/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)

    with open("configs/weights_config.yaml", "r") as f:
        weight_cfg = yaml.safe_load(f)

    with open("configs/composite_score.yaml", "r") as f:
        composite_weights = yaml.safe_load(f)

    # === 加载数据 ===
    train_loader, val_loader, target_names, train_ds, periods, norm_pack = get_dataloaders(
        data_path=model_cfg["data_path"],
        sampler_mode=model_cfg.get("sampler_mode", "balanced"),
        focus_period=model_cfg.get("focus_period"),
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=model_cfg.get("num_workers", 4),
        val_mode=model_cfg.get("val_mode", "days"),
        val_days=model_cfg.get("val_days", 252),
        val_ratio=model_cfg.get("val_ratio", 0.2),
    )
    composite_weights = filter_weights_by_period(composite_weights, periods)
    period_map = {i + 1: p for i, p in enumerate(periods)}
    # === 获取 Loss 与 Metric ===
    all_losses = get_losses_by_targets(target_names)
    all_weights = weight_cfg["custom_weights"]
    metrics_list = get_metrics_by_targets(target_names)

    # === 初始化模型结构 ===
    model = MyTFTModule(
        dataset=train_ds,
        loss_list=all_losses,
        weights=all_weights,
        output_size=[1] * len(target_names),
        metrics_list=metrics_list,
        loss_schedule=model_cfg.get("loss_schedule", []),
        hidden_size=model_cfg["hidden_size"],
        lstm_layers=model_cfg["lstm_layers"],
        scheduler_cfg=model_cfg.get("scheduler", {}), 
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        learning_rate=model_cfg["learning_rate"],
        composite_weights=composite_weights,
        period_map=period_map,
        log_interval=0,
        log_val_interval=0,
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
    logger = TensorBoardLogger(run_dirs["log"], name=model_cfg.get("log_name", "tft_multi"))
    ckpt_cb = CustomCheckpoint(
        weights=composite_weights,
        dirpath=run_dirs["ckpt"],
        filename="tft-best-{epoch}-{val_composite_score:.4f}-{val_loss_for_ckpt:.4f}",
        save_top_k=3,
        save_last=True,
        monitor="val_composite_score",
        mode="max",
    )
    with open(f"{run_dirs['cfg']}/model_config.yaml", "w") as f:
        yaml.safe_dump(model_cfg, f)
    early_stopping = EarlyStopping(
        monitor="val_composite_score",
        patience=model_cfg.get("early_stop_patience", 5),
        mode="max"
    )

    # === 启动训练 ===
    trainer = Trainer(
        max_epochs=model_cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
        precision=model_cfg.get("precision", "16-mixed"),
        gradient_clip_val=model_cfg.get("grad_clip", 0.2),
        accumulate_grad_batches=model_cfg.get("accumulate", 1),
        callbacks=[early_stopping, ckpt_cb],
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        key = next(k for k in ckpt["callbacks"] if k.startswith("CustomCheckpoint"))
        score = ckpt["callbacks"][key]["best_model_score"].item()
        val_loss = ckpt["callbacks"][key].get("val_loss_for_ckpt", float("nan"))
        save_stage_summary(model_cfg.get("log_name", "unnamed_stage"), score, val_loss)

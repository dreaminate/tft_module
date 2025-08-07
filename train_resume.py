import os
import yaml
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from model.tft_module import MyTFTModule
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


def main():
    # ===== 读取配置 =====
    with open("configs/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/weights_config.yaml", "r") as f:
        weight_cfg = yaml.safe_load(f)

    resume_ckpt = model_cfg.get("resume_ckpt")
    assert resume_ckpt and os.path.exists(resume_ckpt), f"Checkpoint 未找到: {resume_ckpt}"
    print(f"🔄 从 checkpoint 续训: {resume_ckpt}")

    # ===== 数据加载 =====
    train_loader, val_loader, target_names, train_ds, periods = get_dataloaders(
        data_path=model_cfg["data_path"],
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=model_cfg.get("num_workers", 4),
        val_mode=model_cfg.get("val_mode", "days"),
        val_days=model_cfg.get("val_days", 252),
        val_ratio=model_cfg.get("val_ratio", 0.2),
    )
    period_map = {i + 1: p for i, p in enumerate(periods)}

    # ===== 构建模型 =====
    model = MyTFTModule.load_from_checkpoint(
        checkpoint_path=resume_ckpt,
        dataset=train_ds,
        loss_list=get_losses_by_targets(target_names),
        weights=weight_cfg["custom_weights"],
        output_size=[1] * len(target_names),
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
    
    logger = TensorBoardLogger("lightning_logs", name=model_cfg.get("log_name", "tft_multi"), flush_secs=10)
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss_epoch",
        filename="epoch{epoch:02d}-loss{val_loss_epoch:.4f}",
        save_top_k=3,
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss_epoch",
        patience=model_cfg.get("early_stop_patience", 5),
        mode="min",
    )

    # 保存配置
    yaml.safe_dump(model_cfg, open("logs/configs/model_config.yaml", "w"))

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

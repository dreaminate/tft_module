import os
import yaml
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from model.tft_module import MyTFTModule
from data.load_dataset import get_dataloaders
from utils.loss_factory import get_losses_by_targets
from utils.metric_factory import get_metrics_by_targets
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
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
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # ===== 读取配置 =====
    model_cfg = load_yaml("configs/model_config.yaml")
    weight_cfg = load_yaml("configs/weights_config.yaml")

    # ===== 数据加载 =====
    train_loader, val_loader, target_names, train_ds, periods,norm_pack = get_dataloaders(
        data_path=model_cfg["data_path"],
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=model_cfg.get("num_workers", 4),
        val_mode=model_cfg.get("val_mode", "days"),
        val_days=model_cfg.get("val_days", 252),
        val_ratio=model_cfg.get("val_ratio", 0.2),
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
    model = MyTFTModule(
        dataset=train_ds,
        steps_per_epoch=steps_per_epoch_eff,
        norm_pack=norm_pack,  
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
        log_interval=model_cfg.get("log_interval", 100),
        log_val_interval=model_cfg.get("log_val_interval", 1),
    )

    # ===== 日志 & 回调 =====
    
    logger = TensorBoardLogger("lightning_logs", name=model_cfg.get("log_name", "tft_multi"))
    ckpt_cb = ModelCheckpoint(
        monitor="val_loss_epoch",
        filename="epoch{epoch:02d}-loss{val_loss_epoch:.4f}",
        save_top_k=3,
        mode="min",
    )
    early_stop = EarlyStopping(
        monitor="val_loss_epoch",
        patience=model_cfg.get("early_stop_patience", 5),
        mode="min",
    )

    # 保存配置文件
    yaml.safe_dump(model_cfg, open("logs/configs/model_config.yaml", "w"))
    
    # ===== 训练 =====
    trainer = Trainer(
        
        log_every_n_steps=1,
        max_epochs=model_cfg["max_epochs"],
        accelerator="gpu",
        devices=1,
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
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
#    python train_multi_tft.py
# === resume_train.py ===
# 从 checkpoint 续训（包括 optimizer 状态、epoch 等）
#           
import os
import yaml
import torch
from lightning.pytorch import Trainer
from model.tft_module import MyTFTModule
from data.load_dataset import get_dataloaders
from utils.loss_factory import get_losses_by_targets
from utils.metric_factory import get_metrics_by_targets
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from callbacks.custom_checkpoint import CustomCheckpoint
from utils.stage_summary import save_stage_summary  # 添加此行以导入函数

# === 读取配置 ===
with open("configs/model_config.yaml", "r") as f:
    model_cfg = yaml.safe_load(f)

with open("configs/weights_config.yaml", "r") as f:
    weight_cfg = yaml.safe_load(f)

with open("configs/composite_score.yaml", "r") as f:
    composite_weights = yaml.safe_load(f)

resume_ckpt = model_cfg.get("resume_ckpt")
assert resume_ckpt and os.path.exists(resume_ckpt), f"❌ Checkpoint not found: {resume_ckpt}"
print(f"♻️ Resume training from checkpoint: {resume_ckpt}")

# === 加载数据 ===
data_path = model_cfg["data_path"]
train_loader, val_loader, target_names, _ = get_dataloaders(
    data_path=data_path,
    sampler_mode=model_cfg.get("sampler_mode", "balanced"),
    focus_period=model_cfg.get("focus_period"),
    batch_size=model_cfg.get("batch_size", 64),
    num_workers=model_cfg.get("num_workers", 4),
)

# === 获取 Loss 与 Metric ===
all_losses = get_losses_by_targets(target_names)
all_weights = weight_cfg["custom_weights"]
metrics_list = get_metrics_by_targets(target_names)

# === 加载模型（包括 optimizer/scheduler 状态）===
model = MyTFTModule.load_from_checkpoint(
    checkpoint_path=resume_ckpt,
    loss_list=all_losses,
    weights=all_weights,
    output_size=[1] * len(target_names),
    metrics_list=metrics_list,
    loss_schedule=model_cfg.get("loss_schedule", []), 
    hidden_size=model_cfg["hidden_size"],
    lstm_layers=model_cfg["lstm_layers"],
    attention_head_size=model_cfg["attention_head_size"],
    dropout=model_cfg["dropout"],
    learning_rate=model_cfg["learning_rate"],
    composite_weights=composite_weights,
    log_interval=0,
    log_val_interval=0,
)

# === 设置日志与 Callback ===
logger = TensorBoardLogger("lightning_logs", name=model_cfg.get("log_name", "tft_multi"), flush_secs=30)
ckpt_cb = CustomCheckpoint(
    weights=composite_weights,
    dirpath="checkpoints",
    filename="tft-best-{epoch}-{composite_score:.4f}-{val_loss_for_ckpt:.4f}",
    save_top_k=3,
    mode="max",
    save_last=True,
    monitor="composite_score",
)
early_stopping = EarlyStopping(
    monitor="composite_score",
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
    resume_from_checkpoint=resume_ckpt,
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
best_ckpt = ckpt_cb.best_model_path
if best_ckpt and os.path.exists(best_ckpt):
    ckpt = torch.load(best_ckpt, map_location="cpu")
    score = ckpt["callbacks"][type(ckpt_cb).__name__]["best_model_score"].item()
    val_loss = ckpt["callbacks"][type(ckpt_cb).__name__].get("val_loss_for_ckpt", float("nan"))
    save_stage_summary(model_cfg.get("log_name", "unnamed_stage"), score, val_loss)

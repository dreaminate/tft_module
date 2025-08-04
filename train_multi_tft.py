# === train_multi_tft.py ===
# 多周期 TFT 多目标训练入口（不含 resume / warm_start）
#   python train_multi_tft.py
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
from utils.stage_summary import save_stage_summary  # ✅
from model.tft_module import get_manual_input_sizes

if __name__ == "__main__":
    # === 加载配置 ===
    with open("configs/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/weights_config.yaml", "r") as f:
        weight_cfg = yaml.safe_load(f)
    with open("configs/composite_score.yaml", "r") as f:
        composite_weights = yaml.safe_load(f)

    # === 加载数据集 ===
    data_path = model_cfg["data_path"]
    train_loader, val_loader, target_names, train_ds = get_dataloaders(
        data_path=data_path,
        batch_size=model_cfg.get("batch_size", 64),
        num_workers=model_cfg.get("num_workers", 4),
        val_mode=model_cfg.get("val_mode", "days"),
        val_days=model_cfg.get("val_days", 252),
        val_ratio=model_cfg.get("val_ratio", 0.2)
    )
        
        

    # === Loss 和 Metric ===
    all_losses = get_losses_by_targets(target_names)
    all_weights = weight_cfg["custom_weights"]
    metrics_list = get_metrics_by_targets(target_names)
#     tft_kwargs = {
#     "embedding_sizes": {
#         "symbol": (3, 3),  # 设置 symbol 类别的嵌入维度为 3
#         "period": (3, 3)   # 设置 period 类别的嵌入维度为 3
#     }
# }
   
    # === 构建模型 ===
    model = MyTFTModule(
        dataset=train_ds,
        loss_list=all_losses,
        weights=all_weights,
        output_size=[1]*len(target_names),  # 每个目标输出1个值
        metrics_list=metrics_list,
        target_names=target_names,
        composite_weights=composite_weights,
        learning_rate=model_cfg["learning_rate"],
        loss_schedule=model_cfg.get("loss_schedule", []),
        hidden_size=model_cfg["hidden_size"],
        lstm_layers=model_cfg["lstm_layers"],
        attention_head_size=model_cfg["attention_head_size"],
        dropout=model_cfg["dropout"],
        log_interval=50,
        log_val_interval=1,
       
        # **tft_kwargs,  # 添加 tft_kwargs 参数
    )
   
    


    


    # === 日志与 Callback 设置 ===
    logger = TensorBoardLogger("lightning_logs", name=model_cfg.get("log_name", "tft_multi"), flush_secs=30)
    ckpt_cb = CustomCheckpoint(
        weights=composite_weights,
        dirpath="checkpoints",
        filename="tft-best-{epoch}-{composite_score:.4f}-{val_loss_for_ckpt:.4f}",
        save_top_k=3,
        save_last=True,
        monitor="composite_score",
        mode="max"
    )
    os.makedirs("logs/configs", exist_ok=True)
    with open("logs/configs/model_config.yaml", "w") as f:
        yaml.safe_dump(model_cfg, f)

    early_stopping = EarlyStopping(
        monitor="composite_score",
        patience=model_cfg.get("early_stop_patience", 5),
        mode="max"
    )
    print("Categorical features:", train_ds.categoricals)
    print("Real-valued features:", train_ds.reals)
    print("Target names:", train_ds.target_names)
    print("Time index column:", train_ds.time_idx)
    print("Group IDs:", train_ds.group_ids)
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

    # === 记录最优结果 ===
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt and os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        score = ckpt["callbacks"][type(ckpt_cb).__name__]["best_model_score"].item()
        val_loss = ckpt["callbacks"][type(ckpt_cb).__name__].get("val_loss_for_ckpt", float("nan"))
        save_stage_summary(model_cfg.get("log_name", "unnamed_stage"), score, val_loss)

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from pytorch_forecasting.models import TemporalFusionTransformer
from torch.nn import MSELoss, SmoothL1Loss, CrossEntropyLoss
from utils.weighted_bce import WeightedBinaryCrossEntropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from pytorch_forecasting.models import TemporalFusionTransformer
from torch.nn import MSELoss, SmoothL1Loss, CrossEntropyLoss
from utils.weighted_bce import WeightedBinaryCrossEntropy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

class ManualMultiLoss(nn.Module):
    def __init__(self, losses, weights, loss_names=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        self.loss_names = (
            loss_names if loss_names is not None else [f"loss_{i}" for i in range(len(losses))]
        )

    def forward(self, preds, targets, return_dict=False):
        assert len(preds) == len(targets) == len(self.losses)
        total = 0.0
        loss_dict = {}

        for i in range(len(preds)):
            assert preds[i].shape == targets[i].shape, f"Shape mismatch: {preds[i].shape} vs {targets[i].shape}"
            loss_i = self.losses[i](preds[i], targets[i])
            weighted_loss_i = self.weights[i] * loss_i
            total += weighted_loss_i
            loss_dict[self.loss_names[i]] = loss_i.detach().clone()

        if return_dict:
            return total, loss_dict
        return total

def get_manual_input_sizes(dataset):
    input_sizes = {
        "static_categoricals": [],
        "static_reals": [],
        "time_varying_known_categoricals": [],
        "time_varying_known_reals": [],
        "time_varying_unknown_categoricals": [],
        "time_varying_unknown_reals": [],
    }

    embedding_sizes = dict(dataset.embedding_sizes)


    for name in dataset.categoricals:
        if name in dataset.static_categoricals:
            input_sizes["static_categoricals"].append(embedding_sizes[name][1])
        elif name in dataset.time_varying_known_categoricals:
            input_sizes["time_varying_known_categoricals"].append(embedding_sizes[name][1])
        elif name in dataset.time_varying_unknown_categoricals:
            input_sizes["time_varying_unknown_categoricals"].append(embedding_sizes[name][1])

    for name in dataset.reals:
        if name in dataset.static_reals:
            input_sizes["static_reals"].append(1)
        elif name in dataset.time_varying_known_reals:
            input_sizes["time_varying_known_reals"].append(1)
        elif name in dataset.time_varying_unknown_reals:
            input_sizes["time_varying_unknown_reals"].append(1)

    return input_sizes
class MyTFTModule(LightningModule):
    def __init__(
        self,
        dataset, 
        loss_list: list,
        weights: list,
        output_size: list,
        metrics_list: list,
        composite_weights: dict[str, float],
        learning_rate: float = 1e-3,
        loss_schedule: dict[int, list] = None,
        target_names: list[str] = None,
        period_map: dict[int, str] = None,
        **tft_kwargs
    ):
        super().__init__()
        tft_kwargs.pop("loss", None)
        tft_kwargs.pop("output_size", None)

        self.loss_fn = ManualMultiLoss(loss_list, weights, loss_names=target_names)
        self.initial_weights = weights
        self.loss_schedule = loss_schedule or {}
        self.target_names = target_names

        self.model = TemporalFusionTransformer.from_dataset(
            dataset=dataset,
            loss=None,
            output_size=output_size,
            **tft_kwargs
        )

        # 简洁方案：在模型内部推断 static_input_size（用于日志/验证）
        input_sizes = self.get_manual_input_sizes(dataset)
        self.static_input_size = sum(input_sizes["static_categoricals"]) + sum(input_sizes["static_reals"])

        self.output_names = target_names
        self.metrics_list = nn.ModuleList([m for _, m in metrics_list])
        self.metric_names = [n for n, _ in metrics_list]
        self.metric_target_idx = []
        self.metric_is_classification = []

        for n in self.metric_names:
            found = False
            for i, tgt_name in enumerate(self.target_names):
                if n.startswith(tgt_name + "_"):
                    self.metric_target_idx.append(i)
                    found = True
                    self.metric_is_classification.append(
                        any(n.endswith(suffix) for suffix in ["_f1", "_precision", "_recall", "_accuracy", "_ap", "_auc"])
                    )
                    break
            if not found:
                raise ValueError(f"Cannot find matching target prefix for metric: {n}")

        provided_keys = set(composite_weights.keys())
        expected_keys = set(f"val_{n}" for n in self.metric_names)
        missing = expected_keys - provided_keys
        if missing:
            print(f"[⚠️ Warning] The following metrics will NOT be used in composite_score:\n{sorted(missing)}")

        self.hparams.composite_weights = composite_weights
        self.hparams.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["loss_fn", "metrics_list", "model", "logging_metrics"])

        self.period_map = period_map or {0: "1h", 1: "4h", 2: "1d"}
        print(f"✅ static_input_size={self.static_input_size}")
    def setup(self, stage: str):
        self.metrics_list = nn.ModuleList([m.to(self.device) for m in self.metrics_list])
        self.loss_fn.to(self.device)

    def forward(self, x):
        out = self.model(x)["prediction"]
        if self.current_epoch == 0:
            print("🧠 [forward] model output type:", type(out))
            if isinstance(out, list):
                print("🧠 output shapes:", [t.shape for t in out])
            else:
                print("⚠️ Unexpected output shape:", out.shape)
        return out

    @staticmethod
    def get_manual_input_sizes(dataset):
        input_sizes = {
            "static_categoricals": [],
            "static_reals": [],
            "time_varying_known_categoricals": [],
            "time_varying_known_reals": [],
            "time_varying_unknown_categoricals": [],
            "time_varying_unknown_reals": [],
        }

        if hasattr(dataset, "embedding_sizes"):
            embedding_sizes = dict(dataset.embedding_sizes)
        else:
            raise AttributeError("❌ 'dataset' does not contain 'embedding_sizes'. Make sure it's processed via TimeSeriesDataSet.")

        for name in dataset.categoricals:
            if name in dataset.static_categoricals:
                input_sizes["static_categoricals"].append(embedding_sizes[name][1])
            elif name in dataset.time_varying_known_categoricals:
                input_sizes["time_varying_known_categoricals"].append(embedding_sizes[name][1])
            elif name in dataset.time_varying_unknown_categoricals:
                input_sizes["time_varying_unknown_categoricals"].append(embedding_sizes[name][1])

        for name in dataset.reals:
            if name in dataset.static_reals:
                input_sizes["static_reals"].append(1)
            elif name in dataset.time_varying_known_reals:
                input_sizes["time_varying_known_reals"].append(1)
            elif name in dataset.time_varying_unknown_reals:
                input_sizes["time_varying_unknown_reals"].append(1)

        return input_sizes



    def on_train_start(self):
        text = "\n".join([f"{k}: {v:.3f}" for k, v in self.hparams.composite_weights.items()])
        self.logger.experiment.add_text("CompositeWeights", text, global_step=0)

    def on_train_epoch_start(self):
        if self.current_epoch in self.loss_schedule:
            new_weights = self.loss_schedule[self.current_epoch]
            self.loss_fn.weights = new_weights
            print(f"[🔁] Switched loss weights at epoch {self.current_epoch} → {new_weights}")
            for i, name in enumerate(self.loss_fn.loss_names):
                self.logger.experiment.add_scalar(f"LossWeight/{name}", new_weights[i], self.current_epoch)

    def stack_y(self, y_list):
        return torch.cat([t.reshape(-1, 1).float() for t in y_list], dim=1)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            val_loss = self.trainer.callback_metrics.get("val_loss")
            score = self.trainer.callback_metrics.get("val_composite_score")
            if self.logger and hasattr(self.logger, "experiment"):
                if val_loss is not None and score is not None:
                    self.logger.experiment.add_scalars(
                        "Validation", {"composite_score": score, "val_loss": val_loss}, self.current_epoch
                    )

                if self.current_epoch % 5 == 0:
                    batch = next(iter(self.trainer.val_dataloaders))
                    x, y = batch
                    out = self.forward(x)
                    preds = torch.stack([p.squeeze(1).squeeze(1) for p in out], dim=1).detach().cpu()
                    trues = self.stack_y(y[0]).detach().cpu()

                    for i in range(preds.shape[1]):
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.scatter(trues[:, i], preds[:, i], alpha=0.3)
                        ax.set_title(f"Prediction vs True ({self.loss_fn.loss_names[i]})")
                        ax.set_xlabel("True")
                        ax.set_ylabel("Predicted")
                        self.logger.experiment.add_figure(f"Scatter/{self.loss_fn.loss_names[i]}", fig, global_step=self.current_epoch)

                    for i, (name, idx, is_cls) in enumerate(zip(self.metric_names, self.metric_target_idx, self.metric_is_classification)):
                        if not is_cls:
                            continue
                        y_pred_label = preds[:, idx] > 0.5
                        y_true_label = trues[:, idx]
                        cm = confusion_matrix(y_true_label, y_pred_label)

                        fig, ax = plt.subplots(figsize=(3.5, 3))
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title(f"Confusion Matrix ({name})")
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("True")
                        self.logger.experiment.add_figure(f"Confusion/{name}", fig, global_step=self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_enc = self.stack_y(y[0]) if isinstance(y, (list, tuple)) else y
        out = self.forward(x)
        assert isinstance(out, list), f"[❌] model output expected list, got {type(out)}"
        preds = [p.squeeze(1) for p in out]
        pred = torch.stack(preds, dim=1)

        total_loss, loss_dict = self.loss_fn(
            [pred[:, i] for i in range(pred.shape[1])],
            [y_enc[:, i] for i in range(y_enc.shape[1])],
            return_dict=True
        )

        for name, val in loss_dict.items():
            self.log(f"train_loss_{name}", val, on_epoch=True, prog_bar=False)
            self.logger.experiment.add_scalar(f"Loss/{name}", val, global_step=self.global_step)

        self.log("train_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_enc = self.stack_y(y[0]) if isinstance(y, (list, tuple)) else y
        out = self.forward(x)
        preds = [p.squeeze(1) for p in out]
        pred = torch.stack(preds, dim=1)

        val_loss = self.loss_fn(
            [pred[:, i] for i in range(pred.shape[1])],
            [y_enc[:, i] for i in range(y_enc.shape[1])]
        )
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)

        period_labels = x["encoder_cat"][:, 0, 1].detach().cpu().numpy()
        period_idx = period_labels[0]
        period_name = self.period_map.get(period_idx, "unknown")

        composite_score = 0.0
        for name, idx, metric, is_cls in zip(self.metric_names, self.metric_target_idx, self.metrics_list, self.metric_is_classification):
            y_true = y_enc[:, idx].long() if is_cls else y_enc[:, idx]
            y_pred = pred[:, idx]
            if is_cls and y_pred.ndim == 2:
                val = metric(y_pred, y_true)
            else:
                val = metric(y_pred.squeeze(-1), y_true)

            key = f"val_{name}"
            self.log(key, val, on_epoch=True, prog_bar=False)

            composite_key = f"{key}@{period_name}"
            if composite_key not in self.hparams.composite_weights:
                print(f"[⚠️] Composite key missing: {composite_key}")
            weight = self.hparams.composite_weights.get(composite_key, 0.0)
            if weight != 0:
                composite_score += weight * val

        self.log("val_composite_score", composite_score, on_epoch=True, prog_bar=True)
        return val_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        out = self.forward(x)
        preds = [p.squeeze(1) for p in out]
        pred = torch.stack(preds, dim=1)

        return {
            "pred": pred.detach().cpu(),
            "true": self.stack_y(y[0]).detach().cpu(),
            "time_idx": x["decoder_time_idx"][:, 0].detach().cpu(),
            "symbol_idx": x["encoder_cat"][:, 0, 0].detach().cpu(),
        }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

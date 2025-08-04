# === plot_loss_weights.py ===
import os
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def extract_loss_weights(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()

    weight_tags = [tag for tag in ea.Tags()["scalars"] if tag.startswith("LossWeight/")]
    data = {}

    for tag in weight_tags:
        events = ea.Scalars(tag)
        name = tag.split("LossWeight/")[1]
        data[name] = [(e.step, e.value) for e in events]

    return data


def plot_weights(log_dir, save_path=None):
    weights = extract_loss_weights(log_dir)
    if not weights:
        print(f"[⚠️] No weight data found in {log_dir}")
        return

    plt.figure(figsize=(10, 5))
    for name, points in weights.items():
        steps, values = zip(*points)
        plt.plot(steps, values, label=name)

    plt.title("Loss Weights over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"[📊] Saved plot to: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str, help="Path to TensorBoard log directory")
    parser.add_argument("--save", type=str, default=None, help="Path to save figure (PNG)")
    args = parser.parse_args()

    plot_weights(args.log_dir, args.save)


if __name__ == "__main__":
    main()

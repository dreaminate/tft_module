import os
import glob
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def extract_scalar_from_event(event_path, keys=("val_composite_score", "val_loss")):
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    results = {}
    for key in keys:
        if key in ea.Tags().get("scalars", []):
            scalars = ea.Scalars(key)
            steps = [s.step for s in scalars]
            values = [s.value for s in scalars]
            results[key] = (steps, values)
    return results

def plot_logs(log_dirs, labels=None, keys=("val_composite_score", "val_loss"), save_path=None):
    if labels is None:
        labels = [os.path.basename(d) for d in log_dirs]

    for key in keys:
        plt.figure(figsize=(10, 4))
        for log_dir, label in zip(log_dirs, labels):
            event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
            if not event_files:
                print(f"[⚠️] No event file found in {log_dir}")
                continue
            data = extract_scalar_from_event(event_files[0], keys=keys)
            if key in data:
                steps, values = data[key]
                plt.plot(steps, values, label=label)

        plt.title(f"{key} across training stages")
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f"compare_{key}.png"))
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dirs", nargs='+', help="List of lightning_logs/version_* folders")
    parser.add_argument("--labels", nargs='+', default=None, help="Optional labels for each run")
    parser.add_argument("--save", type=str, default=None, help="Path to save figures")
    args = parser.parse_args()

    plot_logs(args.log_dirs, labels=args.labels, save_path=args.save)

if __name__ == "__main__":
    main()

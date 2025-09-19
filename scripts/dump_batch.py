import argparse
import pprint
import torch
from data.load_dataset import get_dataloaders


def dump_batch(data_path: str, batch_size: int, num_workers: int, val_ratio: float, val_mode: str):
    train_loader, val_loader, target_names, train_ds, periods, norm_pack = get_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        val_mode=val_mode,
        val_ratio=val_ratio,
    )
    batch = next(iter(val_loader))
    features, target = batch
    print("keys:", list(features.keys()))
    for k, v in features.items():
        if torch.is_tensor(v):
            print(f"{k}: shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
        else:
            print(f"{k}: type={type(v)}")
    if torch.is_tensor(target):
        print("target shape:", tuple(target.shape), "dtype=", target.dtype)
    else:
        print("target type:", type(target))
    meta = {
        "target_names": target_names,
        "periods": periods,
        "symbol_classes": norm_pack.get("symbol_classes") if norm_pack else None,
    }
    print("meta:")
    pprint.pprint(meta)


def parse_args():
    ap = argparse.ArgumentParser(description="Dump a sample batch from the validation loader")
    ap.add_argument("--data-path", default="data/pkl_merged/full_merged.pkl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--val-mode", choices=["ratio", "days"], default="ratio")
    return ap.parse_args()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    dump_batch(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        val_mode=args.val_mode,
    )

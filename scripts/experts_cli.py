import argparse
import os
import sys
import subprocess
from pathlib import Path


def find_leaf(experts_root: Path, expert: str, period: str, modality: str) -> Path:
    leaf = experts_root / expert / period / modality
    cfg = leaf / "model_config.yaml"
    tcfg = leaf / "targets.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"未找到配置文件: {cfg}")
    if not tcfg.exists():
        raise FileNotFoundError(f"未找到目标配置: {tcfg}")
    return cfg


def list_leaves(experts_root: Path):
    rows = []
    for p in experts_root.rglob("model_config.yaml"):
        try:
            expert, period, modality = p.parents[2].name, p.parents[1].name, p.parents[0].name
        except Exception:
            expert = p.parent.name
            period = modality = "_"
        rows.append((expert, period, modality, p))
    rows.sort()
    return rows


def run_train(script: str, cfg_path: Path, extra: list[str]):
    cmd = [sys.executable, script, "--config", str(cfg_path)] + extra
    print("运行:", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Experts CLI — 便捷按专家训练/续训/热启动")
    ap.add_argument("command", choices=["list", "train", "resume", "warm"], help="操作：列出/训练/续训/热启动")
    ap.add_argument("--experts-root", default="configs/experts", help="专家配置根目录")
    ap.add_argument("--expert", help="专家名，例如 Alpha-Dir-TFT")
    ap.add_argument("--period", help="周期：1h|4h|1d 等")
    ap.add_argument("--modality", default="base", help="模态：base|rich")
    ap.add_argument("--leaf", help="快捷指定叶子，如 '1h/base'（优先于 --period/--modality）")
    ap.add_argument("--", dest="passthrough", nargs=argparse.REMAINDER, help="透传给底层脚本的参数")
    args = ap.parse_args()

    experts_root = Path(args.experts_root)
    if args.command == "list":
        rows = list_leaves(experts_root)
        if not rows:
            print("未找到任何专家叶子配置 (model_config.yaml)")
            sys.exit(1)
        print("可用叶子配置：expert period modality -> path")
        for expert, period, modality, p in rows:
            print(f"- {expert:>24} {period:>4} {modality:>6} -> {p}")
        sys.exit(0)

    # 解析 leaf 参数
    if args.leaf:
        try:
            per, mod = args.leaf.split("/", 1)
            args.period = per
            args.modality = mod
        except ValueError:
            ap.error("--leaf 格式应为 'period/modality' 例如 '1h/base'")
    # train/resume/warm 需要 expert + period/modality
    missing = [k for k in ("expert", "period", "modality") if getattr(args, k, None) in (None, "")]
    if missing:
        ap.error(f"命令 {args.command} 需要参数: --expert --period --modality")

    cfg_path = find_leaf(experts_root, args.expert, args.period, args.modality)
    extra = (args.passthrough or [])

    if args.command == "train":
        rc = run_train("train_multi_tft.py", cfg_path, extra)
    elif args.command == "resume":
        rc = run_train("train_resume.py", cfg_path, extra)
    else:  # warm
        rc = run_train("warm_start_train.py", cfg_path, extra)
    sys.exit(rc)


if __name__ == "__main__":
    main()

# utils/run_helper.py 示例 (tft version)
from pathlib import Path
from datetime import datetime, timezone
import inspect

def prepare_run_dirs(base: str = "runs"):
    script = Path(inspect.getfile(inspect.currentframe().f_back)).stem
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    root = Path(base) / f"{script}-{ts}"
    ckpt = root / "checkpoints"
    log  = root / "lightning_logs"
    cfg  = root / "configs"
    for p in (ckpt, log, cfg):
        p.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in {"root": root, "ckpt": ckpt, "log": log, "cfg": cfg}.items()}


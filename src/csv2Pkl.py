"""Convert fused CSV (full or slim) to Pickle, aligned with expert folder layout.

Usage:
  python src/csv2Pkl.py --config configs/fuse_fundamentals.yaml --which full
  python src/csv2Pkl.py --config configs/fuse_fundamentals.yaml --which slim
"""

from pathlib import Path
import argparse
import pandas as pd
from typing import Any, Dict, Optional


def _load_yaml(path: str) -> Dict[str, Any]:
    # 轻量 YAML 解析（与 fuse 中一致的降级版）
    try:
        import yaml  # type: ignore
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f.read()) or {}
    except Exception:
        pass
    text = Path(path).read_text(encoding='utf-8')
    def _parse_value(v: str):
        vs = v.strip()
        if vs.lower() in ("true","yes","on"): return True
        if vs.lower() in ("false","no","off"): return False
        if vs.lower() in ("null","none","~"): return None
        if (vs.startswith('"') and vs.endswith('"')) or (vs.startswith("'") and vs.endswith("'")):
            return vs[1:-1]
        try:
            if vs.isdigit() or (vs.startswith('-') and vs[1:].isdigit()):
                return int(vs)
            return float(vs)
        except Exception:
            return vs
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip():
            continue
        line = raw.split('#',1)[0]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        line = line.strip()
        if ':' not in line:
            continue
        key, rest = line.split(':',1)
        key = key.strip(); rest = rest.strip()
        while stack and stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1] if stack else root
        if rest == '':
            d: Dict[str, Any] = {}
            parent[key] = d
            stack.append((indent, d))
        else:
            parent[key] = _parse_value(rest)
    return root


def _compose_paths(cfg: Dict[str, Any], which: str) -> tuple[Path, Path]:
    # 复刻 fuse 的路径推导：文件名后缀 + 分组文件夹
    extra = cfg.get('extra_field') or {}
    out_csv = cfg.get('out_csv', 'data/merged/full_merged_with_fundamentals.csv')
    slim_csv = cfg.get('slim_csv', 'data/merged/full_merged_slim.csv')
    path = Path(out_csv if which == 'full' else slim_csv)

    # suffix
    name = (extra.get('name') or '').strip()
    value = extra.get('value')
    how = str(extra.get('append_to_filename', 'name')).strip().lower()
    suffix = ''
    if name and how not in ('none', 'off', 'no', 'false'):
        if how == 'both' and value is not None:
            suffix = f'_{name}-{value}'
        elif how == 'value' and value is not None:
            suffix = f'_{value}'
        else:
            suffix = f'_{name}'
    if suffix:
        path = path.with_name(path.stem + suffix + path.suffix)

    # group folder
    if extra.get('group_to_folder'):
        mode = str(extra.get('folder_mode', how)).strip().lower()
        tag = ''
        if name:
            if mode == 'both' and value is not None:
                tag = f'{name}-{value}'
            elif mode == 'value' and value is not None:
                tag = f'{value}'
            else:
                tag = f'{name}'
        base_root = extra.get('folder_root')
        if base_root:
            path = Path(base_root) / tag / path.name
        elif tag:
            path = path.parent / tag / path.name

    # outputs next to CSV
    dst = path.with_suffix('.pkl')
    return path, dst


def convert_csv_to_pkl(src_file: Path, dst_file: Path, selected_periods: Optional[list[str]] = None):
    if not src_file.exists():
        print(f"[❌] 源文件不存在: {src_file}")
        return
    try:
        print(f"▶ 读取: {src_file}")
        df = pd.read_csv(src_file)
        if 'timestamp' not in df.columns:
            print('❌ 缺失 timestamp 字段'); return
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].max() > 1e12 else None, errors='coerce')
        if selected_periods and 'period' in df.columns:
            before = len(df); df = df[df['period'].isin(selected_periods)]; print(f"🔍 周期过滤: {before} → {len(df)}")
        if df.empty:
            print('⚠️ 文件数据为空（过滤后）'); return
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(dst_file)
        print(f"✅ 保存 PKL → {dst_file}")
    except Exception as e:
        print(f"❌ 转换失败: {e}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fuse_fundamentals.yaml')
    ap.add_argument('--which', choices=['full','slim'], default='full', help='转换 full 或 slim 输出')
    ap.add_argument('--periods', nargs='*', default=["1h","4h","1d"])
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    src, dst = _compose_paths(cfg, args.which)
    convert_csv_to_pkl(src, dst, args.periods)
    
    # 也顺手提示 slim/full 另一个文件位置
    other = 'slim' if args.which == 'full' else 'full'
    try:
        o_src, o_dst = _compose_paths(cfg, other)
        print(f"ℹ️ 另外一个输出: {other}: {o_src} -> {o_dst.with_suffix('.pkl')}")
    except Exception:
        pass

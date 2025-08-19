#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
nan_early_gap_check.py

只做“检查”：判定高 NaN 列是否属于“早期连续缺失、后期连续有”的情形。
- 输入固定：data\merged\full_merged.csv
- 不修改源数据；仅打印报告并输出一个审计 CSV。

运行：
  python nan_early_gap_check.py
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import re
import numpy as np
import pandas as pd

# ================= 固定输入路径（只读） =================
INPUT_PATH = Path(r"data\merged\full_merged.csv")
# ======================================================


# ---------- 列名规范化（仅用于识别时间列） ----------
def norm_col_name(name: str) -> str:
    s = str(name).lower()
    # 各种 dash（-，–，—，——、数学负号）和空白、斜杠替为下划线
    s = re.sub(r"[ \t\r\n/\\\-\u2012-\u2015\u2212]+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ---------- 安全把“数值秒/毫秒/日期”统一到毫秒 Int64（只在本脚本内部用） ----------
def _series_int64_from_float(num: pd.Series, tol: float = 1e-6) -> pd.Series:
    """
    仅将“近似整数”的浮点安全转为 Int64；其他置为 <NA>，避免 float→Int64 的 unsafe cast 报错。
    """
    num = pd.to_numeric(num, errors="coerce")
    mask = num.notna() & (np.abs(num - np.round(num)) <= tol)
    out = pd.Series(pd.array([pd.NA] * len(num), dtype="Int64"), index=num.index)
    if mask.any():
        out.loc[mask] = np.round(num[mask]).astype("int64", copy=False)
    return out

def ensure_ms_timestamp(s: pd.Series) -> pd.Series:
    """
    将任意时间表示统一为“毫秒 Int64（可空）”，不改源数据：
      1) 数值路径：先安全取整为 Int64；位数<=10 视为秒，*1000；
      2) 字符/日期路径：to_datetime(..., utc=True) → 毫秒；
      3) 取非空比例更高者。
    """
    # 数值路径
    num_raw = pd.to_numeric(s, errors="coerce")
    num_int = _series_int64_from_float(num_raw)
    if num_int.dropna().size:
        lens = num_int.dropna().astype("int64").astype(str).str.len()
        mult = 1000 if lens.median() <= 10 else 1
        ts_num = pd.Series(pd.array([pd.NA] * len(num_int), dtype="Int64"), index=num_int.index)
        ts_num.loc[num_int.notna()] = (num_int.dropna().astype("int64") * mult).values
    else:
        ts_num = pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)

    # 日期路径
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if dt.notna().any():
        ts_raw = dt.astype("int64", copy=False) // 10**6
        ts_dt = pd.Series(ts_raw, index=s.index).where(dt.notna()).astype("Int64")
    else:
        ts_dt = pd.Series(pd.array([pd.NA] * len(s), dtype="Int64"), index=s.index)

    return ts_num if ts_num.notna().mean() >= ts_dt.notna().mean() else ts_dt


# ---------- 自动识别时间列（遍历所有列名 + 值） ----------
TIME_NAME_WHITELIST = {
    "timestamp","time","datetime","date","open_time","close_time","start_time","end_time",
    "open_time_ms","close_time_ms","start_time_ms","end_time_ms",
}

def detect_time_column(df: pd.DataFrame) -> str:
    cand = []
    for col in df.columns:
        ts = ensure_ms_timestamp(df[col])
        ok = float(ts.notna().mean())

        ts_non = ts.dropna()
        mono = bool(ts_non.is_monotonic_increasing or ts_non.is_monotonic_decreasing)
        mono_score = 0.15 if mono else 0.0

        uniq_score = 0.0
        if ts_non.size > 0:
            uniq_ratio = ts_non.nunique() / ts_non.size
            uniq_score = 0.15 * max(0.0, min(1.0, uniq_ratio))

        name_norm = norm_col_name(col)
        looks_like = (
            name_norm in TIME_NAME_WHITELIST
            or "timestamp" in name_norm
            or re.search(r"(^|_)time($|_)", name_norm)
            or re.search(r"(^|_)date($|_)", name_norm)
        )
        name_score = 0.10 if looks_like else 0.0

        plausible = 0.0
        if ts_non.size > 0:
            mn, mx = int(ts_non.min()), int(ts_non.max())
            if 1230768000000 <= mn <= 4102444800000 and 1230768000000 <= mx <= 4102444800000:
                plausible = 0.10

        score = ok + mono_score + uniq_score + name_score + plausible
        cand.append((col, score, ok))

    if not cand:
        raise KeyError("无法识别时间列：没有可用列。")

    # 取最高分；并列优先列名更像 timestamp
    cand.sort(key=lambda x: (x[1], "timestamp" in norm_col_name(x[0])), reverse=True)
    best = cand[0][0]
    print(f"[time] 识别到时间列：{best}")
    return best


# ---------- 形态判定 ----------
@dataclass
class AuditRow:
    group_key: str
    col: str
    nan_ratio: float
    leading_nan_len: int
    inner_nan_count: int
    trailing_nan_len: int
    has_contiguous_tail: bool
    is_strict_early_gap: bool
    first_non_nan_dt: Optional[str]
    last_non_nan_dt: Optional[str]

def analyze_mask(mask: np.ndarray) -> Dict[str, object]:
    n = int(mask.shape[0])
    total_nan = int(mask.sum())

    # 前/后连续 NaN 段
    lead = 0
    while lead < n and mask[lead]:
        lead += 1
    trail = 0
    while trail < n and mask[n-1-trail]:
        trail += 1

    inner = int(mask[lead:n-trail].sum()) if n - lead - trail > 0 else 0

    # 是否存在“从某一行开始到结尾全为非 NaN”
    if n > 0:
        # suffix_nan[i] = i..end 是否仍存在 NaN
        suffix_nan = np.maximum.accumulate(mask[::-1].astype(np.int8))[::-1].astype(bool)
        # 候选 i：当前位置非 NaN 且 i..end 都没有 NaN
        cand_tail = (~mask) & (~suffix_nan)
        has_tail = bool(cand_tail.any())
    else:
        has_tail = False

    # “严格早期缺失”：前段有 NaN，且中段无 NaN，且末尾无 NaN
    strict_early = (lead > 0 and inner == 0 and trail == 0 and n > lead)

    return dict(
        leading=lead, inner=inner, trailing=trail,
        has_contiguous_tail=has_tail,
        is_strict_early_gap=strict_early,
        nan_ratio=(total_nan / n) if n > 0 else np.nan
    )


# ---------- 每组审计 ----------
def audit_one_group(df: pd.DataFrame, timecol: str, group_key: str) -> List[AuditRow]:
    out: List[AuditRow] = []
    if df.empty:
        return out

    # 确保时间列为毫秒 Int64，并按时间升序
    ts = ensure_ms_timestamp(df[timecol])
    df = df.copy()
    df[timecol] = ts
    df = df.sort_values(timecol)
    ts_dt = pd.to_datetime(df[timecol], unit="ms", utc=True)

    # 仅数值列（排除时间列）
    num_cols = list(df.select_dtypes(include=["number"]).columns)
    if timecol in num_cols:
        num_cols.remove(timecol)

    for col in num_cols:
        ser = pd.to_numeric(df[col], errors="coerce")
        mask = ser.isna().to_numpy()

        stats = analyze_mask(mask)
        lead = int(stats["leading"])
        inner = int(stats["inner"])
        trail = int(stats["trailing"])
        nan_ratio = float(stats["nan_ratio"])
        has_tail = bool(stats["has_contiguous_tail"])
        strict_early = bool(stats["is_strict_early_gap"])

        # 首/末个非 NaN 的时间
        first_non_dt = None
        last_non_dt = None
        if lead < len(df) and not mask[lead]:
            first_non_dt = str(ts_dt.iloc[lead])
        if trail < len(df) and not mask[len(df) - 1 - trail]:
            last_non_dt = str(ts_dt.iloc[len(df) - 1 - trail])

        out.append(AuditRow(
            group_key=group_key, col=col, nan_ratio=nan_ratio,
            leading_nan_len=lead, inner_nan_count=inner, trailing_nan_len=trail,
            has_contiguous_tail=has_tail, is_strict_early_gap=strict_early,
            first_non_nan_dt=first_non_dt, last_non_nan_dt=last_non_dt
        ))
    return out


# ---------- 主流程（只检查） ----------
def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_PATH.resolve()}")

    print(f"[info] 读取：{INPUT_PATH.resolve()}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[info] 形状：{df.shape}")

    # 自动识别时间列（仅用于排序/定位，不改源数据）
    time_col = detect_time_column(df)

    # 组键（若存在）
    group_keys = [k for k in ("symbol","period") if k in df.columns]
    audits: List[AuditRow] = []

    if group_keys:
        for gvals, sub in df.groupby(group_keys, dropna=False):
            if not isinstance(gvals, tuple):
                gvals = (gvals,)
            gkey = " | ".join(f"{k}={v}" for k,v in zip(group_keys, gvals))
            audits.extend(audit_one_group(sub, time_col, gkey))
    else:
        audits.extend(audit_one_group(df, time_col, "__ALL__"))

    res = pd.DataFrame([a.__dict__ for a in audits])
    out_csv = INPUT_PATH.parent / f"{INPUT_PATH.stem}__nan_early_gap_check.csv"
    res.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[ok] 检查完成，结果已写出：{out_csv}")

    if res.empty:
        print("[ok] 表内无可审计的数值列。")
        return

    # 打印 NaN 比例最高的 30 列，并标注是否属于“早期缺失导致”
    print("\n=== NaN 比例 TOP 30（含是否为早期缺失） ===")
    top = res.sort_values(["nan_ratio","col"], ascending=[False, True]).head(30).copy()
    top["early_cause?"] = np.where(
        (top["is_strict_early_gap"]) | ((top["leading_nan_len"] > 0) & (top["inner_nan_count"] == 0) & (top["has_contiguous_tail"])),
        "YES(早期缺失导致，后段连续有）", "NO/混合（中段或后段也缺）"
    )
    with pd.option_context("display.max_rows", 50, "display.max_colwidth", 100):
        print(top[[
            "group_key","col","nan_ratio","leading_nan_len","inner_nan_count","trailing_nan_len",
            "has_contiguous_tail","is_strict_early_gap","early_cause?",
            "first_non_nan_dt","last_non_nan_dt"
        ]])

    print("\n提示：若某列 `has_contiguous_tail=True` 且 `inner_nan_count=0`，基本可判断为“早期没有，后面一直有”。")


if __name__ == "__main__":
    main()
# 仅检查：判定高 NaN 列是否属于“早期连续缺失、后期连续有”的情形。
# 主要关注以下几种情况：
# - 早期缺失：leading_nan_len > 0
# - 中段缺失：inner_nan_count > 0
# - 后期缺失：trailing_nan_len > 0
#  python src/check_merged_data.py
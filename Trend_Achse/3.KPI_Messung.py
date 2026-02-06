# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/2.KPI_Messung.py

KPI-Messung für Trend-Regime (States) gegen OHLC.
Berechnet NUR Timeframes ab min_tf (Default H4).

OUTPUT (Ordnerstruktur, keine große Sammel-CSV):
  <ROOT>/1_Data_Center/Data/Regime/KPI/Trend/Variants/<variant_id>/<TF>.csv
    - enthält eine Zeile pro Symbol (KPIs)
    - sortiert nach symbol

SUMMARY:
  <ROOT>/1_Data_Center/Data/Regime/KPI/Trend/summary.json

Input:
  States (Trend):
    Base:
      <ROOT>/1_Data_Center/Data/Regime/States/Trend/<TF>/<SYMBOL>.csv
    Variants:
      <ROOT>/1_Data_Center/Data/Regime/States/Trend/Variants/<variant_id>/<TF>/<SYMBOL>.csv

  OHLC:
      <ROOT>/1_Data_Center/Data/Regime/Ohcl/<TF>/<SYMBOL>.csv

Fix:
- pandas datetime parse warning: format="ISO8601" (falls verfügbar) via parse_time_utc()

CLI Beispiele:
  python 4_Regime_Builder/Trend_Achse/2.KPI_Messung.py --once
  python 4_Regime_Builder/Trend_Achse/2.KPI_Messung.py --once --include-base --include-variants
  python 4_Regime_Builder/Trend_Achse/2.KPI_Messung.py --once --variants v1 v2
  python 4_Regime_Builder/Trend_Achse/2.KPI_Messung.py --once --min-tf H4 --horizons 1 5 20 50 --is-frac 0.7 --bootstrap-B 2000
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# ROOT / PATHS
# =========================

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "1_Data_Center").exists():
            return p
    return start.resolve().parents[1]


ROOT = find_project_root(Path(__file__))

OHLC_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Ohcl"
STATES_BASE_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "States" / "Trend"
STATES_VARIANTS_DIR = STATES_BASE_DIR / "Variants"

KPI_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "KPI" / "Trend"
KPI_VARIANTS_DIR = KPI_DIR / "Variants"
SUMMARY_JSON = KPI_DIR / "summary.json"


# =========================
# TIMEFRAME ORDER + FILTER
# =========================

TIMEFRAMES_PREFERRED = ["M1", "M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]


def filter_tfs_min(tf_list: List[str], min_tf: str = "H4") -> List[str]:
    order = {tf: i for i, tf in enumerate(TIMEFRAMES_PREFERRED)}
    if min_tf not in order:
        # keep only known timeframes if min_tf unknown
        return [tf for tf in tf_list if tf in order]

    min_i = order[min_tf]
    out: List[str] = []
    for tf in tf_list:
        i = order.get(tf, None)
        if i is not None and i >= min_i:
            out.append(tf)
    return out


# =========================
# Datetime parsing (FIX)
# =========================

def parse_time_utc(series: pd.Series) -> pd.Series:
    """
    Deterministic, fast datetime parsing for ISO8601-like timestamps.
    Uses pandas format="ISO8601" when available; falls back gracefully.
    """
    try:
        return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    except TypeError:
        return pd.to_datetime(series, utc=True, errors="coerce")


# =========================
# IO helpers
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


def atomic_write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass


# =========================
# Read helpers
# =========================

def read_ohlc(tf: str, symbol: str) -> pd.DataFrame:
    p = OHLC_DIR / tf / f"{symbol}.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty or "time" not in df.columns or "close" not in df.columns:
        return pd.DataFrame()

    df["time"] = parse_time_utc(df["time"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (
        df.dropna(subset=["time", "close"])
          .sort_values("time")
          .drop_duplicates("time", keep="last")
          .reset_index(drop=True)
    )
    return df


def read_states_from_path(p: Path, state_col: str = "trend_state") -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)
    if df.empty or "time" not in df.columns or state_col not in df.columns:
        return pd.DataFrame()

    df["time"] = parse_time_utc(df["time"])
    df[state_col] = pd.to_numeric(df[state_col], errors="coerce")

    # optional columns
    if "trend_score" in df.columns:
        df["trend_score"] = pd.to_numeric(df["trend_score"], errors="coerce")
    if "trend_label" in df.columns:
        df["trend_label"] = df["trend_label"].astype(str)

    df = (
        df.dropna(subset=["time", state_col])
          .sort_values("time")
          .drop_duplicates("time", keep="last")
          .reset_index(drop=True)
    )
    df[state_col] = df[state_col].astype(int).clip(-1, 1)
    return df


def list_timeframes_in_dir(base: Path) -> List[str]:
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


def list_symbols_in_tf_dir(tf_dir: Path) -> List[str]:
    if not tf_dir.exists():
        return []
    return sorted([p.stem for p in tf_dir.glob("*.csv")])


def list_variants() -> List[str]:
    if not STATES_VARIANTS_DIR.exists():
        return []
    return sorted([p.name for p in STATES_VARIANTS_DIR.iterdir() if p.is_dir()])


# =========================
# KPI core math
# =========================

def forward_log_return(close: pd.Series, h: int) -> pd.Series:
    h = int(h)
    nxt = close.shift(-h)
    ratio = nxt / close
    ratio = ratio.where((close > 0) & (nxt > 0))
    return np.log(ratio)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1)
    vb = np.var(b, ddof=1)
    denom = max(1, (len(a) + len(b) - 2))
    sp = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / denom)
    if sp == 0 or not np.isfinite(sp):
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / sp)


def state_balance(st: np.ndarray) -> Tuple[float, float, float]:
    n = len(st)
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    return (
        float(np.mean(st == 1)),
        float(np.mean(st == 0)),
        float(np.mean(st == -1)),
    )


def switch_stats(st: np.ndarray) -> Tuple[float, float, float]:
    """
    returns:
      switch_rate (fraction of transitions),
      switches_per_1000,
      avg_state_duration (bars)
    """
    n = len(st)
    if n <= 1:
        return (np.nan, np.nan, np.nan)

    changes = int(np.sum(st[1:] != st[:-1]))
    switch_rate = changes / float(n - 1)

    idx = np.flatnonzero(st[1:] != st[:-1]) + 1
    boundaries = np.r_[0, idx, n]
    durs = np.diff(boundaries).astype(float)
    avg_dur = float(np.mean(durs)) if len(durs) else float("nan")

    switches_per_1000 = (changes / float(max(1, n))) * 1000.0
    return (float(switch_rate), float(switches_per_1000), avg_dur)


def bootstrap_sep_mean(a: np.ndarray, b: np.ndarray, B: int, seed: int) -> Tuple[float, float, float]:
    """
    bootstrap distribution of mean(a)-mean(b)
    returns: (ci05, ci95, pneg)
    """
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2 or int(B) <= 0:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(int(seed))
    nA, nB = len(a), len(b)

    diffs = np.empty(int(B), dtype=float)
    for i in range(int(B)):
        sa = a[rng.integers(0, nA, size=nA)]
        sb = b[rng.integers(0, nB, size=nB)]
        diffs[i] = float(np.mean(sa) - np.mean(sb))

    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return (np.nan, np.nan, np.nan)

    ci05 = float(np.quantile(diffs, 0.05))
    ci95 = float(np.quantile(diffs, 0.95))
    pneg = float(np.mean(diffs < 0.0))
    return (ci05, ci95, pneg)


# =========================
# Merge + split
# =========================

def merge_ohlc_states(
    ohlc: pd.DataFrame,
    states: pd.DataFrame,
    state_col: str = "trend_state",
) -> Tuple[pd.DataFrame, float, int, int]:
    """
    Returns:
      merged_df (inner on time) with columns: time, close, trend_state, [trend_score], [trend_label]
      coverage = len(merged)/len(ohlc) (if ohlc non-empty)
      n_ohlc, n_merged
    """
    if ohlc.empty:
        return (pd.DataFrame(), np.nan, 0, 0)
    if states.empty:
        return (pd.DataFrame(), 0.0, int(len(ohlc)), 0)

    keep_cols = ["time", state_col]
    if "trend_score" in states.columns:
        keep_cols.append("trend_score")
    if "trend_label" in states.columns:
        keep_cols.append("trend_label")

    st = states[keep_cols].copy()

    m = pd.merge(
        ohlc[["time", "close"]].copy(),
        st,
        how="inner",
        on="time",
        validate="one_to_one",
    )

    m = (
        m.dropna(subset=["time", "close", state_col])
         .sort_values("time")
         .reset_index(drop=True)
    )

    n_ohlc = int(len(ohlc))
    n_m = int(len(m))
    coverage = float(n_m / n_ohlc) if n_ohlc > 0 else np.nan
    return (m, coverage, n_ohlc, n_m)


def compute_split_time(times: pd.Series, is_frac: float) -> Optional[pd.Timestamp]:
    if times is None or len(times) == 0:
        return None
    f = float(is_frac)
    if not np.isfinite(f) or f <= 0.0 or f >= 1.0:
        return None

    t = parse_time_utc(times).dropna().sort_values()
    if len(t) < 3:
        return None

    idx = int(np.floor(f * (len(t) - 1)))
    idx = int(np.clip(idx, 0, len(t) - 1))
    return pd.to_datetime(t.iloc[idx], utc=True)


def segment_df(m: pd.DataFrame, split_time: Optional[pd.Timestamp]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if m.empty:
        return (m.copy(), m.copy(), m.copy())
    if split_time is None:
        return (m.copy(), m.copy(), m.copy())
    t = pd.to_datetime(m["time"], utc=True)
    is_df = m.loc[t <= split_time].copy()
    oos_df = m.loc[t > split_time].copy()
    return (m.copy(), is_df, oos_df)


# =========================
# KPI per segment
# =========================

def kpis_for_segment(
    df: pd.DataFrame,
    horizons: List[int],
    bootstrap_B: int,
    bootstrap_seed: int,
    suffix: str,
    coverage_value: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    out[f"coverage{suffix}"] = float(coverage_value) if np.isfinite(coverage_value) else np.nan

    if df.empty or "trend_state" not in df.columns:
        out[f"n_rows{suffix}"] = 0.0
        return out

    st = df["trend_state"].to_numpy(dtype=int)
    b1, b0, bm1 = state_balance(st)
    out[f"balance_bull{suffix}"] = b1
    out[f"balance_neutral{suffix}"] = b0
    out[f"balance_bear{suffix}"] = bm1

    sw_rate, sw_1000, avg_dur = switch_stats(st)
    out[f"switch_rate{suffix}"] = sw_rate
    out[f"switches_per_1000{suffix}"] = sw_1000
    out[f"avg_state_duration{suffix}"] = avg_dur
    out[f"n_rows{suffix}"] = float(len(df))

    for h in horizons:
        col = f"fwd_lr_{int(h)}"
        if col not in df.columns:
            continue

        x = df[col].to_numpy(dtype=float)
        bull = x[(st == 1) & np.isfinite(x)]
        neu = x[(st == 0) & np.isfinite(x)]
        bear = x[(st == -1) & np.isfinite(x)]

        def _m(a: np.ndarray) -> float:
            return float(np.mean(a)) if len(a) else float("nan")

        def _md(a: np.ndarray) -> float:
            return float(np.median(a)) if len(a) else float("nan")

        def _sd(a: np.ndarray) -> float:
            return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")

        out[f"mean_bull_h{h}{suffix}"] = _m(bull)
        out[f"median_bull_h{h}{suffix}"] = _md(bull)
        out[f"std_bull_h{h}{suffix}"] = _sd(bull)
        out[f"n_bull_h{h}{suffix}"] = float(len(bull))

        out[f"mean_neutral_h{h}{suffix}"] = _m(neu)
        out[f"median_neutral_h{h}{suffix}"] = _md(neu)
        out[f"std_neutral_h{h}{suffix}"] = _sd(neu)
        out[f"n_neutral_h{h}{suffix}"] = float(len(neu))

        out[f"mean_bear_h{h}{suffix}"] = _m(bear)
        out[f"median_bear_h{h}{suffix}"] = _md(bear)
        out[f"std_bear_h{h}{suffix}"] = _sd(bear)
        out[f"n_bear_h{h}{suffix}"] = float(len(bear))

        if len(bull) and len(bear):
            sep_mean = float(np.mean(bull) - np.mean(bear))
            sep_med = float(np.median(bull) - np.median(bear))
        else:
            sep_mean = float("nan")
            sep_med = float("nan")

        out[f"sep_mean_bull_bear_h{h}{suffix}"] = sep_mean
        out[f"sep_median_bull_bear_h{h}{suffix}"] = sep_med
        out[f"cohen_d_bull_bear_h{h}{suffix}"] = cohen_d(bull, bear)

        out[f"hitrate_bull_pos_h{h}{suffix}"] = float(np.mean(bull > 0.0)) if len(bull) else float("nan")
        out[f"hitrate_bear_neg_h{h}{suffix}"] = float(np.mean(bear < 0.0)) if len(bear) else float("nan")

        if int(bootstrap_B) > 0:
            ci05, ci95, pneg = bootstrap_sep_mean(
                bull, bear,
                B=int(bootstrap_B),
                seed=int(bootstrap_seed) + int(h) * 10007
            )
            out[f"boot_ci05_sep_mean_h{h}{suffix}"] = ci05
            out[f"boot_ci95_sep_mean_h{h}{suffix}"] = ci95
            out[f"boot_pneg_sep_mean_h{h}{suffix}"] = pneg

    return out


# =========================
# Runner
# =========================

@dataclass
class Config:
    horizons: List[int]
    is_frac: float
    bootstrap_B: int
    bootstrap_seed: int
    include_base: bool
    include_variants: bool
    variants_filter: Optional[List[str]]
    min_tf: str


def resolve_variants(cfg: Config) -> List[str]:
    all_vars = list_variants()
    if not cfg.include_variants:
        return []
    if cfg.variants_filter and len(cfg.variants_filter) > 0:
        keep = set(cfg.variants_filter)
        return [v for v in all_vars if v in keep]
    return all_vars


def get_state_path(variant_id: str, tf: str, symbol: str) -> Path:
    if variant_id == "base":
        return STATES_BASE_DIR / tf / f"{symbol}.csv"
    return STATES_VARIANTS_DIR / variant_id / tf / f"{symbol}.csv"


def compute_one(
    tf: str,
    symbol: str,
    variant_id: str,
    cfg: Config,
) -> Tuple[str, Optional[Dict[str, object]]]:
    """
    Returns: status, row-dict or None
    row-dict is one KPI row for (variant_id, tf, symbol)
    """
    ohlc = read_ohlc(tf, symbol)
    if ohlc.empty:
        return ("missing_ohlc", None)

    sp = get_state_path(variant_id, tf, symbol)
    states = read_states_from_path(sp)
    if states.empty:
        return ("missing_states", {
            "variant_id": variant_id,
            "tf": tf,
            "symbol": symbol,
            "state_path": str(sp),
            "ohlc_rows": float(len(ohlc)),
            "merged_rows": 0.0,
            "coverage_all": 0.0,
            "split_time_utc": "",
            "from_utc": "",
            "to_utc": "",
            "updated_at_utc": utc_now_str(),
            "status": "missing_states",
        })

    merged, coverage, n_ohlc, n_merged = merge_ohlc_states(ohlc, states)
    if merged.empty:
        return ("no_overlap", {
            "variant_id": variant_id,
            "tf": tf,
            "symbol": symbol,
            "state_path": str(sp),
            "ohlc_rows": float(n_ohlc),
            "merged_rows": float(n_merged),
            "coverage_all": float(coverage),
            "split_time_utc": "",
            "from_utc": "",
            "to_utc": "",
            "updated_at_utc": utc_now_str(),
            "status": "no_overlap",
        })

    if not cfg.horizons:
        return ("no_horizons", None)

    max_h = int(max(cfg.horizons))
    for h in cfg.horizons:
        merged[f"fwd_lr_{int(h)}"] = forward_log_return(merged["close"], int(h))

    merged = merged.dropna(subset=[f"fwd_lr_{max_h}"]).reset_index(drop=True)
    if merged.empty:
        return ("insufficient_forward_history", {
            "variant_id": variant_id,
            "tf": tf,
            "symbol": symbol,
            "state_path": str(sp),
            "ohlc_rows": float(n_ohlc),
            "merged_rows": 0.0,
            "coverage_all": float(coverage),
            "split_time_utc": "",
            "from_utc": "",
            "to_utc": "",
            "updated_at_utc": utc_now_str(),
            "status": "insufficient_forward_history",
        })

    split_time = compute_split_time(merged["time"], cfg.is_frac)
    split_str = split_time.isoformat() if split_time is not None else ""

    all_df, is_df, oos_df = segment_df(merged, split_time)

    row: Dict[str, object] = {
        "variant_id": variant_id,
        "tf": tf,
        "symbol": symbol,
        "state_path": str(sp),
        "ohlc_rows": float(n_ohlc),
        "merged_rows": float(n_merged),
        "coverage_all": float(coverage),
        "split_time_utc": split_str,
        "from_utc": str(pd.to_datetime(all_df["time"], utc=True).min()),
        "to_utc": str(pd.to_datetime(all_df["time"], utc=True).max()),
        "updated_at_utc": utc_now_str(),
        "status": "ok",
    }

    row.update(kpis_for_segment(all_df, cfg.horizons, cfg.bootstrap_B, cfg.bootstrap_seed, suffix="_all", coverage_value=coverage))
    if split_time is not None:
        row.update(kpis_for_segment(is_df, cfg.horizons, cfg.bootstrap_B, cfg.bootstrap_seed, suffix="_is", coverage_value=coverage))
        row.update(kpis_for_segment(oos_df, cfg.horizons, cfg.bootstrap_B, cfg.bootstrap_seed, suffix="_oos", coverage_value=coverage))

    return ("ok", row)


def build_rows(cfg: Config) -> Tuple[List[Dict[str, object]], Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Computes KPI rows (dicts) and returns:
      rows_list
      status_counts_total
      status_counts_per_variant
    """
    ensure_dir(KPI_DIR)
    ensure_dir(KPI_VARIANTS_DIR)

    tfs = list_timeframes_in_dir(OHLC_DIR)
    if not tfs:
        raise RuntimeError(f"Keine Timeframe-Ordner in OHLC_DIR: {OHLC_DIR}")

    tfs = filter_tfs_min(tfs, min_tf=cfg.min_tf)
    if not tfs:
        raise RuntimeError(f"Keine Timeframes >= {cfg.min_tf} gefunden (nach Filter).")

    variants: List[str] = []
    if cfg.include_base:
        variants.append("base")
    variants += resolve_variants(cfg)

    status_counts_total: Dict[str, int] = {}
    per_variant_counts: Dict[str, Dict[str, int]] = {v: {} for v in variants}

    rows: List[Dict[str, object]] = []

    for tf in tfs:
        syms = list_symbols_in_tf_dir(OHLC_DIR / tf)
        if not syms:
            status_counts_total["no_symbols_in_ohlc"] = status_counts_total.get("no_symbols_in_ohlc", 0) + 1
            continue

        for variant_id in variants:
            for symbol in syms:
                status, row = compute_one(tf, symbol, variant_id, cfg)
                status_counts_total[status] = status_counts_total.get(status, 0) + 1
                per_variant_counts[variant_id][status] = per_variant_counts[variant_id].get(status, 0) + 1
                if row is not None:
                    rows.append(row)

    return rows, status_counts_total, per_variant_counts


def write_kpi_folders(rows: List[Dict[str, object]]) -> Dict[str, int]:
    """
    Writes KPI rows into:
      KPI/Trend/Variants/<variant_id>/<TF>.csv
    Returns write counters.
    """
    if not rows:
        return {"groups_written": 0, "rows_written": 0}

    df = pd.DataFrame(rows)
    if df.empty:
        return {"groups_written": 0, "rows_written": 0}

    ensure_dir(KPI_VARIANTS_DIR)

    groups_written = 0
    rows_written = 0

    # group by (variant_id, tf)
    for (variant_id, tf), g in df.groupby(["variant_id", "tf"]):
        out_dir = KPI_VARIANTS_DIR / str(variant_id)
        ensure_dir(out_dir)
        out_path = out_dir / f"{tf}.csv"

        g = g.sort_values(["symbol"]).reset_index(drop=True)
        atomic_write_csv(g, out_path)

        groups_written += 1
        rows_written += int(len(g))

    return {"groups_written": groups_written, "rows_written": rows_written}


def write_summary(cfg: Config, status_counts_total: Dict[str, int], per_variant_counts: Dict[str, Dict[str, int]], write_counts: Dict[str, int]) -> None:
    summary = {
        "meta": {
            "updated_at_utc": utc_now_str(),
            "root": str(ROOT.resolve()),
            "ohlc_dir": str(OHLC_DIR.resolve()),
            "states_base_dir": str(STATES_BASE_DIR.resolve()),
            "states_variants_dir": str(STATES_VARIANTS_DIR.resolve()),
            "kpi_dir": str(KPI_DIR.resolve()),
            "kpi_variants_dir": str(KPI_VARIANTS_DIR.resolve()),
        },
        "config": {
            "horizons": list(map(int, cfg.horizons)),
            "is_frac": float(cfg.is_frac),
            "bootstrap_B": int(cfg.bootstrap_B),
            "bootstrap_seed": int(cfg.bootstrap_seed),
            "include_base": bool(cfg.include_base),
            "include_variants": bool(cfg.include_variants),
            "variants_filter": cfg.variants_filter,
            "min_tf": str(cfg.min_tf),
        },
        "write_counts": write_counts,
        "status_counts_total": status_counts_total,
        "status_counts_per_variant": per_variant_counts,
    }
    atomic_write_json(summary, SUMMARY_JSON)


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=300.0)

    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 20, 50], help="Forward horizons in bars")
    ap.add_argument("--is-frac", type=float, default=0.7, help="IS fraction (0..1). If invalid -> no split.")
    ap.add_argument("--bootstrap-B", type=int, default=0, help="Bootstrap resamples (0 disables)")
    ap.add_argument("--bootstrap-seed", type=int, default=123, help="Bootstrap seed")

    ap.add_argument("--include-base", action="store_true", help="Include base states")
    ap.add_argument("--include-variants", action="store_true", help="Include variants discovered under States/Trend/Variants")
    ap.add_argument("--variants", nargs="*", default=None, help="Optional filter list of variant ids (only if include-variants)")

    ap.add_argument("--min-tf", type=str, default="H4", help="Minimaler Timeframe (z.B. H4)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.loop and args.once:
        raise ValueError("Use either --once or --loop (not both).")
    if not args.loop and not args.once:
        args.once = True

    # Default: wenn nichts gesetzt -> base+variants
    if not args.include_base and not args.include_variants:
        args.include_base = True
        args.include_variants = True

    cfg = Config(
        horizons=[int(x) for x in args.horizons],
        is_frac=float(args.is_frac),
        bootstrap_B=int(args.bootstrap_B),
        bootstrap_seed=int(args.bootstrap_seed),
        include_base=bool(args.include_base),
        include_variants=bool(args.include_variants),
        variants_filter=list(args.variants) if args.variants is not None and len(args.variants) > 0 else None,
        min_tf=str(args.min_tf),
    )

    print("[INFO] ROOT             =", ROOT.resolve())
    print("[INFO] OHLC_DIR          =", OHLC_DIR.resolve())
    print("[INFO] STATES_BASE_DIR   =", STATES_BASE_DIR.resolve())
    print("[INFO] STATES_VARIANTS   =", STATES_VARIANTS_DIR.resolve())
    print("[INFO] KPI_DIR           =", KPI_DIR.resolve())
    print("[INFO] KPI_VARIANTS_DIR  =", KPI_VARIANTS_DIR.resolve())
    print("[INFO] min_tf            =", cfg.min_tf)
    print("[INFO] horizons          =", cfg.horizons)
    print("[INFO] is_frac           =", cfg.is_frac)
    print("[INFO] bootstrap_B       =", cfg.bootstrap_B)
    print("[INFO] include_base      =", cfg.include_base)
    print("[INFO] include_variants  =", cfg.include_variants)
    print("[INFO] variants_filter   =", cfg.variants_filter)
    print("[INFO] mode              =", "loop" if args.loop else "once")

    def run_once() -> None:
        rows, counts_total, counts_per_variant = build_rows(cfg)

        write_counts = write_kpi_folders(rows)
        print("[DONE] wrote KPI folders ->", KPI_VARIANTS_DIR.resolve(), "write_counts=", write_counts)

        write_summary(cfg, counts_total, counts_per_variant, write_counts)
        print("[DONE] summary ->", SUMMARY_JSON.resolve())
        print("[INFO] status_counts_total:", counts_total)

    if args.once:
        run_once()
        return

    import time
    try:
        while True:
            run_once()
            print("[LOOP] updated_at_utc=", utc_now_str())
            time.sleep(float(args.sleep))
    except KeyboardInterrupt:
        print("[INFO] stopped.")


if __name__ == "__main__":
    main()

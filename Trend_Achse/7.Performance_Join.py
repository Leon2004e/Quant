# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/7.Performance_Join.py

Purpose:
  Take passed/selected regimes (Filter_System output),
  join Candidate trend states (H4+) onto H1 labeling,
  compute hourly + session performance per regime.

Fixes:
  - merge_asof now uses tolerance based on TF duration to prevent stale state carryover
  - stronger type coercion and NaN handling
  - summary includes coverage ratio (joined rows / label rows)
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple, List

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

FILTER_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Filter" / "Trend"
PASSED_PATH = FILTER_DIR / "passed.csv"
SELECTED_PATH = FILTER_DIR / "selected.csv"

STATES_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "States" / "Trend" / "Variants"
LABEL_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Labeling" / "H1"

OUT_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Performance" / "Trend"
OUT_PER_REGIME = OUT_DIR / "per_regime"
OUT_SUMMARY_CSV = OUT_DIR / "per_regime_summary.csv"
OUT_SUMMARY_JSON = OUT_DIR / "summary.json"


# =========================
# IO atomic
# =========================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


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


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time_utc_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


# =========================
# Config
# =========================

REQUIRED_FILTER_COLS = ["variant_id", "tf", "symbol"]
REQUIRED_LABEL_COLS = ["time", "ret_cc", "hour_utc", "session_block", "session_core"]
REQUIRED_STATE_COLS = ["time", "trend_state", "trend_label"]


@dataclass(frozen=True)
class RegimeKey:
    variant_id: str
    tf: str
    symbol: str


# =========================
# TF -> tolerance mapping
# =========================

def tf_to_timedelta(tf: str) -> pd.Timedelta:
    tfu = str(tf).strip().upper()
    if tfu == "H4":
        return pd.Timedelta(hours=4)
    if tfu == "H8":
        return pd.Timedelta(hours=8)
    if tfu == "H12":
        return pd.Timedelta(hours=12)
    if tfu == "D1":
        return pd.Timedelta(days=1)
    if tfu == "W1":
        return pd.Timedelta(days=7)
    if tfu == "MN1":
        return pd.Timedelta(days=31)
    if tfu == "Q":
        return pd.Timedelta(days=92)
    if tfu == "Y":
        return pd.Timedelta(days=365)
    # fallback conservative
    return pd.Timedelta(hours=24)


# =========================
# Loaders
# =========================

def load_filter_table(use_selected: bool) -> pd.DataFrame:
    path = SELECTED_PATH if use_selected and SELECTED_PATH.exists() else PASSED_PATH
    if not path.exists():
        raise RuntimeError(f"Filter file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"Filter file empty: {path}")

    missing = [c for c in REQUIRED_FILTER_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Filter file missing cols {missing}: {path}")

    for c in REQUIRED_FILTER_COLS:
        df[c] = df[c].astype(str)

    df = df.drop_duplicates(subset=REQUIRED_FILTER_COLS, keep="last").reset_index(drop=True)
    return df


def state_path(k: RegimeKey) -> Path:
    return STATES_DIR / k.variant_id / k.tf / f"{k.symbol}.csv"


def label_path(symbol: str) -> Path:
    return LABEL_DIR / f"{symbol}.csv"


def read_states(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df

    for c in REQUIRED_STATE_COLS:
        if c not in df.columns:
            raise ValueError(f"State file missing '{c}': {p}")

    df["time"] = parse_time_utc_series(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    df["trend_state"] = pd.to_numeric(df["trend_state"], errors="coerce")
    df = df.dropna(subset=["trend_state"]).reset_index(drop=True)
    df["trend_state"] = df["trend_state"].astype(int)

    df["trend_label"] = df["trend_label"].astype(str)
    return df


def read_labels(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if df.empty:
        return df

    for c in REQUIRED_LABEL_COLS:
        if c not in df.columns:
            raise ValueError(f"Label file missing '{c}': {p}")

    df["time"] = parse_time_utc_series(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

    df["ret_cc"] = pd.to_numeric(df["ret_cc"], errors="coerce")
    df["hour_utc"] = pd.to_numeric(df["hour_utc"], errors="coerce")
    df["session_core"] = pd.to_numeric(df["session_core"], errors="coerce").fillna(0).astype(int)
    df["session_block"] = df["session_block"].astype(str)

    # require hour 0..23
    df = df[(df["hour_utc"].notna()) & (df["hour_utc"] >= 0) & (df["hour_utc"] <= 23)].reset_index(drop=True)
    df["hour_utc"] = df["hour_utc"].astype(int)

    return df


# =========================
# Join + performance
# =========================

def join_state_to_h1(labels_h1: pd.DataFrame, states_tf: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    For each H1 bar, take last known state at or before that time.
    Uses tolerance to avoid carrying stale state across long gaps.
    """
    if labels_h1.empty or states_tf.empty:
        return pd.DataFrame()

    left = labels_h1.sort_values("time").copy()
    right = states_tf.sort_values("time").copy()

    tol = tf_to_timedelta(tf)

    joined = pd.merge_asof(
        left,
        right[["time", "trend_state", "trend_label"]],
        on="time",
        direction="backward",
        allow_exact_matches=True,
        tolerance=tol,
    )

    joined = joined.dropna(subset=["trend_state"]).reset_index(drop=True)
    joined["trend_state"] = joined["trend_state"].astype(int)
    joined["trend_label"] = joined["trend_label"].astype(str)
    return joined


def perf_agg(joined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Metrics:
      n, mean_ret, vol_ret, sharpe (mean/vol), hitrate, sum_ret
    """
    df = joined.copy()
    df = df.dropna(subset=["ret_cc", "hour_utc", "session_block"]).reset_index(drop=True)

    def _agg(g: pd.DataFrame) -> pd.Series:
        r = pd.to_numeric(g["ret_cc"], errors="coerce").to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        n = int(r.size)
        if n == 0:
            return pd.Series({"n": 0, "mean_ret": np.nan, "vol_ret": np.nan, "sharpe": np.nan, "hitrate": np.nan, "sum_ret": np.nan})
        mean = float(np.mean(r))
        vol = float(np.std(r, ddof=1)) if n > 1 else np.nan
        sharpe = float(mean / vol) if (np.isfinite(vol) and vol > 0) else np.nan
        hit = float(np.mean(r > 0.0))
        s = float(np.sum(r))
        return pd.Series({"n": n, "mean_ret": mean, "vol_ret": vol, "sharpe": sharpe, "hitrate": hit, "sum_ret": s})

    by_state = df.groupby(["trend_state", "trend_label"], dropna=False).apply(_agg).reset_index()

    by_state_hour = (
        df.groupby(["trend_state", "trend_label", "hour_utc"], dropna=False)
          .apply(_agg).reset_index()
          .sort_values(["trend_state", "hour_utc"])
          .reset_index(drop=True)
    )

    by_state_sess = (
        df.groupby(["trend_state", "trend_label", "session_block", "session_core"], dropna=False)
          .apply(_agg).reset_index()
          .sort_values(["trend_state", "session_block", "session_core"])
          .reset_index(drop=True)
    )

    return by_state, by_state_hour, by_state_sess


def run_one(k: RegimeKey) -> Tuple[str, Dict[str, object]]:
    sp = state_path(k)
    lp = label_path(k.symbol)

    states = read_states(sp)
    if states.empty:
        return "missing_states", {"variant_id": k.variant_id, "tf": k.tf, "symbol": k.symbol, "state_path": str(sp)}

    labels = read_labels(lp)
    if labels.empty:
        return "missing_labels", {"variant_id": k.variant_id, "tf": k.tf, "symbol": k.symbol, "label_path": str(lp)}

    joined = join_state_to_h1(labels, states, tf=k.tf)
    if joined.empty:
        return "empty_join", {"variant_id": k.variant_id, "tf": k.tf, "symbol": k.symbol}

    by_state, by_state_hour, by_state_sess = perf_agg(joined)

    out_dir = OUT_PER_REGIME / k.variant_id / k.tf
    ensure_dir(out_dir)

    def _tag(df: pd.DataFrame, section: str) -> pd.DataFrame:
        out = df.copy()
        out.insert(0, "section", section)
        return out

    combined = pd.concat([
        _tag(by_state, "by_state"),
        _tag(by_state_hour, "by_state_hour"),
        _tag(by_state_sess, "by_state_session"),
    ], ignore_index=True)

    outp = out_dir / f"{k.symbol}.csv"
    atomic_write_csv(combined, outp)

    info = {
        "variant_id": k.variant_id,
        "tf": k.tf,
        "symbol": k.symbol,
        "rows_labels": int(len(labels)),
        "rows_joined": int(len(joined)),
        "coverage_join": float(len(joined) / max(1, len(labels))),
        "from_utc": str(pd.to_datetime(joined["time"], utc=True).min()),
        "to_utc": str(pd.to_datetime(joined["time"], utc=True).max()),
        "out": str(outp),
        "updated_at_utc": utc_now_str(),
    }
    return "ok", info


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-selected", action="store_true",
                    help="use selected.csv if present; otherwise passed.csv")
    ap.add_argument("--max-n", type=int, default=0,
                    help="limit number of regimes for quick test (0=all)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    ensure_dir(OUT_DIR)
    ensure_dir(OUT_PER_REGIME)

    filt = load_filter_table(use_selected=bool(args.use_selected))

    keys: List[RegimeKey] = [
        RegimeKey(variant_id=str(r["variant_id"]), tf=str(r["tf"]), symbol=str(r["symbol"]))
        for _, r in filt.iterrows()
    ]

    if int(args.max_n) > 0:
        keys = keys[:int(args.max_n)]

    counts: Dict[str, int] = {}
    rows: List[Dict[str, object]] = []

    for k in keys:
        try:
            status, info = run_one(k)
            counts[status] = counts.get(status, 0) + 1
            rows.append({"status": status, **info})
        except Exception as e:
            counts["error"] = counts.get("error", 0) + 1
            rows.append({
                "status": "error",
                "variant_id": k.variant_id,
                "tf": k.tf,
                "symbol": k.symbol,
                "error": str(e),
            })

    summary_df = pd.DataFrame(rows)
    atomic_write_csv(summary_df, OUT_SUMMARY_CSV)

    summary = {
        "meta": {
            "updated_at_utc": utc_now_str(),
            "root": str(ROOT.resolve()),
            "filter_dir": str(FILTER_DIR.resolve()),
            "states_dir": str(STATES_DIR.resolve()),
            "label_dir": str(LABEL_DIR.resolve()),
            "out_dir": str(OUT_DIR.resolve()),
            "use_selected": bool(args.use_selected),
            "n_requested": int(len(keys)),
        },
        "status_counts": counts,
        "files": {
            "per_regime_summary_csv": str(OUT_SUMMARY_CSV),
            "per_regime_dir": str(OUT_PER_REGIME),
            "summary_json": str(OUT_SUMMARY_JSON),
        }
    }
    atomic_write_json(summary, OUT_SUMMARY_JSON)

    print("[INFO] status_counts:", counts)
    print("[DONE] per_regime_summary ->", OUT_SUMMARY_CSV.resolve())
    print("[DONE] summary_json        ->", OUT_SUMMARY_JSON.resolve())
    print("[DONE] per_regime dir      ->", OUT_PER_REGIME.resolve())


if __name__ == "__main__":
    main()

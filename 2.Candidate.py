# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/1.Candidate.py

UPDATED:
  - optional global start-time cutoff:
      --start-time-utc "2016-02-02 21:00:00+00:00"
  - applies cutoff to:
      trend features input, existing states, and final output

UPDATED (H4+ ONLY):
  - ignores all timeframes smaller than H4 (M1,M5,M15,H1 removed)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
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
    # fallback: 2 levels up
    return start.resolve().parents[1]


HERE = Path(__file__).resolve()
ROOT = find_project_root(HERE)

FEATURES_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Features"
TREND_FEAT_DIR = FEATURES_DIR / "Trend"

STATES_BASE_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "States" / "Trend"
STATES_VAR_DIR = STATES_BASE_DIR / "Variants"
SUMMARY_PATH = STATES_BASE_DIR / "summary.json"

DEFAULT_VARIANTS_JSON = HERE.parent / "trend_variants.json"


# =========================
# SETTINGS (H4+ ONLY)
# =========================

# Removed: M1, M5, M15, H1
TIMEFRAMES_PREFERRED = ["H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]

DEFAULT_LOOKBACK_BARS = 600

# Removed intraday handling
MIN_ROWS_DEFAULT = 200

DEFAULT_SMOOTH_SPAN = 10
DEFAULT_W_SPREAD = 0.5
DEFAULT_W_PRICE = 0.4
DEFAULT_W_SLOPE = 0.1
DEFAULT_T_ENTER = 0.80
DEFAULT_T_EXIT = 0.40

REQUIRED_TREND_COLS = ["ema50_ema200_atr", "price_ema200_atr", "ema200_slope_atr"]


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


def parse_time_utc_str(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    ts = pd.to_datetime(ss, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid --start-time-utc: {s}")
    return ts


# =========================
# Discovery
# =========================

def safe_symbol_from_path(p: Path) -> str:
    return p.stem


def load_timeframes() -> List[str]:
    """
    H4+ ONLY:
      Only returns TFs that are in TIMEFRAMES_PREFERRED and exist as folders.
      Does NOT include any other folders.
    """
    if not TREND_FEAT_DIR.exists():
        return []
    available = {p.name for p in TREND_FEAT_DIR.iterdir() if p.is_dir()}
    return [tf for tf in TIMEFRAMES_PREFERRED if tf in available]


def list_symbols_for_tf(tf: str) -> List[Path]:
    tf_dir = TREND_FEAT_DIR / tf
    if not tf_dir.exists():
        return []
    return sorted(tf_dir.glob("*.csv"))


def min_rows_save(tf: str) -> int:
    # unified for all allowed TFs
    return int(MIN_ROWS_DEFAULT)


# =========================
# Read helpers
# =========================

def read_trend_features(path: Path, start_time_utc: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "time" not in df.columns:
        raise ValueError(f"missing 'time' column: {path}")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for c in REQUIRED_TREND_COLS:
        if c not in df.columns:
            raise ValueError(f"missing required column '{c}': {path}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=REQUIRED_TREND_COLS).reset_index(drop=True)

    # global cutoff
    if start_time_utc is not None and not df.empty:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.loc[t >= start_time_utc].reset_index(drop=True)

    return df


def read_existing_states(path: Path, start_time_utc: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty or "time" not in df.columns:
        return pd.DataFrame()
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["time"])
          .sort_values("time")
          .drop_duplicates(subset=["time"], keep="last")
          .reset_index(drop=True)
    )

    # cutoff existing too
    if start_time_utc is not None and not df.empty:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.loc[t >= start_time_utc].reset_index(drop=True)

    return df


def slice_for_incremental(df: pd.DataFrame, last_time: Optional[pd.Timestamp], lookback_bars: int) -> pd.DataFrame:
    if df.empty:
        return df
    if last_time is None:
        return df

    t = pd.to_datetime(df["time"], utc=True)
    mask_new = t > last_time

    if not mask_new.any():
        start = max(0, len(df) - int(lookback_bars))
        return df.iloc[start:].copy()

    first_new_idx = int(mask_new.idxmax())
    start = max(0, first_new_idx - int(lookback_bars))
    return df.iloc[start:].copy()


def merge_on_time(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = new.copy()
    elif new is None or new.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new], ignore_index=True)

    if out.empty:
        return out

    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = (
        out.dropna(subset=["time"])
           .drop_duplicates(subset=["time"], keep="last")
           .sort_values("time")
           .reset_index(drop=True)
    )
    return out


# =========================
# Trend regime logic
# =========================

def compute_trend_score(
    df: pd.DataFrame,
    w_spread: float,
    w_price: float,
    w_slope: float,
    smooth_span: int,
) -> pd.Series:
    raw = (
        float(w_spread) * df["ema50_ema200_atr"] +
        float(w_price)  * df["price_ema200_atr"] +
        float(w_slope)  * df["ema200_slope_atr"]
    )

    span = int(smooth_span)
    if span <= 1:
        return raw.astype(float)

    return raw.ewm(span=span, adjust=False, min_periods=span).mean()


def apply_hysteresis_states(
    score: np.ndarray,
    t_enter: float,
    t_exit: float,
    initial_state: int = 0,
) -> np.ndarray:
    te = float(t_enter)
    tx = float(t_exit)
    if tx > te:
        tx = te

    st = np.zeros(len(score), dtype=np.int8)

    cur = int(initial_state)
    if cur not in (-1, 0, 1):
        cur = 0

    for i, s in enumerate(score):
        if np.isnan(s):
            st[i] = cur
            continue

        if cur == 0:
            if s >= te:
                cur = 1
            elif s <= -te:
                cur = -1
        elif cur == 1:
            if s <= tx:
                cur = 0
        else:
            if s >= -tx:
                cur = 0

        st[i] = cur

    return st


def label_from_state(x: int) -> str:
    if x == 1:
        return "bull"
    if x == -1:
        return "bear"
    return "neutral"


# =========================
# Paths for outputs
# =========================

def base_state_out_path(tf: str, symbol: str) -> Path:
    return STATES_BASE_DIR / tf / f"{symbol}.csv"


def variant_state_out_path(variant_name: str, tf: str, symbol: str) -> Path:
    return STATES_VAR_DIR / variant_name / tf / f"{symbol}.csv"


# =========================
# Compute one symbol incremental (generic for base/variant)
# =========================

def compute_symbol_incremental(
    tf: str,
    symbol: str,
    trend_feat_full: pd.DataFrame,
    outp: Path,
    lookback_bars: int,
    smooth_span: int,
    w_spread: float,
    w_price: float,
    w_slope: float,
    t_enter: float,
    t_exit: float,
    start_time_utc: Optional[pd.Timestamp],
) -> Tuple[str, Dict[str, object]]:
    existing = read_existing_states(outp, start_time_utc=start_time_utc)

    last_time = None
    last_state = 0
    if not existing.empty:
        last_time = pd.to_datetime(existing["time"].max(), utc=True)
        if "trend_state" in existing.columns:
            try:
                last_state = int(existing.loc[existing["time"] == last_time, "trend_state"].iloc[-1])
            except Exception:
                last_state = 0

    df_slice = slice_for_incremental(trend_feat_full, last_time, int(lookback_bars))
    if df_slice.empty:
        return "empty_input", {"file": str(outp)}

    df_new = pd.DataFrame({"time": df_slice["time"]}).copy()
    df_new["trend_score"] = compute_trend_score(
        df_slice,
        w_spread=float(w_spread),
        w_price=float(w_price),
        w_slope=float(w_slope),
        smooth_span=int(smooth_span),
    )

    df_new["time"] = pd.to_datetime(df_new["time"], utc=True, errors="coerce")
    df_new = df_new.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    df_valid = df_new.dropna(subset=["trend_score"]).reset_index(drop=True)
    if df_valid.empty:
        return "insufficient_history", {
            "file": str(outp),
            "rows_slice": int(len(df_slice)),
            "rows_new": int(len(df_new)),
            "rows_valid": 0,
        }

    if start_time_utc is not None:
        t = pd.to_datetime(df_valid["time"], utc=True, errors="coerce")
        df_valid = df_valid.loc[t >= start_time_utc].reset_index(drop=True)
        if df_valid.empty:
            return "empty_after_start_cut", {"file": str(outp), "start_time_utc": str(start_time_utc)}

    score_arr = df_valid["trend_score"].to_numpy(dtype=float)
    st = apply_hysteresis_states(
        score_arr,
        t_enter=float(t_enter),
        t_exit=float(t_exit),
        initial_state=int(last_state),
    )

    df_valid["trend_state"] = st.astype(np.int8)
    df_valid["trend_label"] = [label_from_state(int(x)) for x in df_valid["trend_state"].values]

    df_final = merge_on_time(existing, df_valid)
    df_final["time"] = pd.to_datetime(df_final["time"], utc=True, errors="coerce")
    df_final = df_final.dropna(subset=["time", "trend_score", "trend_state"]).reset_index(drop=True)

    if start_time_utc is not None and not df_final.empty:
        t = pd.to_datetime(df_final["time"], utc=True, errors="coerce")
        df_final = df_final.loc[t >= start_time_utc].reset_index(drop=True)

    if df_final.empty:
        return "no_valid_rows_after_merge", {"file": str(outp)}

    mr = min_rows_save(tf)
    if len(df_final) < mr:
        return "insufficient_data", {"file": str(outp), "rows_final": int(len(df_final)), "min_rows": mr}

    atomic_write_csv(df_final, outp)
    return "ok", {
        "file": str(outp),
        "rows_saved": int(len(df_final)),
        "from_utc": str(pd.to_datetime(df_final["time"], utc=True).min()),
        "to_utc": str(pd.to_datetime(df_final["time"], utc=True).max()),
        "updated_at_utc": utc_now_str(),
        "params": {
            "lookback_bars": int(lookback_bars),
            "smooth_span": int(smooth_span),
            "w_spread": float(w_spread),
            "w_price": float(w_price),
            "w_slope": float(w_slope),
            "t_enter": float(t_enter),
            "t_exit": float(t_exit),
            "start_time_utc": (str(start_time_utc) if start_time_utc is not None else None),
        }
    }


# =========================
# Variants loader
# =========================

def _sanitize_variant_name(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "", s)
    s = s.strip("._-")
    return s or "variant"


def load_variants(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []

    txt = path.read_text(encoding="utf-8", errors="ignore")
    if not txt or not txt.strip():
        return []

    obj = json.loads(txt)

    if isinstance(obj, list):
        raw_list = obj
    elif isinstance(obj, dict):
        raw_list = obj.get("variants", [])
    else:
        raw_list = []

    if not isinstance(raw_list, list) or len(raw_list) == 0:
        return []

    out: List[Dict[str, object]] = []
    seen = set()

    for v in raw_list:
        if not isinstance(v, dict):
            continue
        name = _sanitize_variant_name(v.get("name", "variant"))
        if name in seen:
            continue
        seen.add(name)

        def get_float(key: str, default: float) -> float:
            x = v.get(key, default)
            try:
                return float(x)
            except Exception:
                return float(default)

        def get_int(key: str, default: int) -> int:
            x = v.get(key, default)
            try:
                return int(x)
            except Exception:
                return int(default)

        lookback_bars = get_int("lookback_bars", DEFAULT_LOOKBACK_BARS)
        smooth_span = get_int("smooth_span", DEFAULT_SMOOTH_SPAN)
        w_spread = get_float("w_spread", DEFAULT_W_SPREAD)
        w_price = get_float("w_price", DEFAULT_W_PRICE)
        w_slope = get_float("w_slope", DEFAULT_W_SLOPE)
        t_enter = get_float("t_enter", DEFAULT_T_ENTER)
        t_exit = get_float("t_exit", DEFAULT_T_EXIT)

        if smooth_span < 1:
            smooth_span = 1
        if lookback_bars < 50:
            lookback_bars = 50

        ws = abs(w_spread) + abs(w_price) + abs(w_slope)
        if ws <= 1e-12:
            w_spread, w_price, w_slope = DEFAULT_W_SPREAD, DEFAULT_W_PRICE, DEFAULT_W_SLOPE
        else:
            w_spread = w_spread / ws
            w_price = w_price / ws
            w_slope = w_slope / ws

        if t_enter < 0:
            t_enter = abs(t_enter)
        if t_exit < 0:
            t_exit = abs(t_exit)
        if t_exit > t_enter:
            t_exit = t_enter

        out.append({
            "name": name,
            "lookback_bars": int(lookback_bars),
            "smooth_span": int(smooth_span),
            "w_spread": float(w_spread),
            "w_price": float(w_price),
            "w_slope": float(w_slope),
            "t_enter": float(t_enter),
            "t_exit": float(t_exit),
        })

    return out


# =========================
# Build cycle
# =========================

def build_cycle_base(
    lookback_bars: int,
    smooth_span: int,
    w_spread: float,
    w_price: float,
    w_slope: float,
    t_enter: float,
    t_exit: float,
    start_time_utc: Optional[pd.Timestamp],
) -> Dict[str, int]:
    ensure_dir(STATES_BASE_DIR)

    tfs = load_timeframes()
    if not tfs:
        raise RuntimeError(f"Keine H4+ Timeframe-Ordner gefunden unter: {TREND_FEAT_DIR}")

    counters: Dict[str, int] = {}

    for tf in tfs:
        csv_paths = list_symbols_for_tf(tf)
        if not csv_paths:
            counters["no_symbols"] = counters.get("no_symbols", 0) + 1
            continue

        for p in csv_paths:
            symbol = safe_symbol_from_path(p)
            try:
                trend_feat = read_trend_features(p, start_time_utc=start_time_utc)
                if trend_feat.empty:
                    counters["empty_trend_features"] = counters.get("empty_trend_features", 0) + 1
                    continue

                outp = base_state_out_path(tf, symbol)
                status, _info = compute_symbol_incremental(
                    tf=tf,
                    symbol=symbol,
                    trend_feat_full=trend_feat,
                    outp=outp,
                    lookback_bars=int(lookback_bars),
                    smooth_span=int(smooth_span),
                    w_spread=float(w_spread),
                    w_price=float(w_price),
                    w_slope=float(w_slope),
                    t_enter=float(t_enter),
                    t_exit=float(t_exit),
                    start_time_utc=start_time_utc,
                )
                counters[status] = counters.get(status, 0) + 1

            except Exception:
                counters["error"] = counters.get("error", 0) + 1

    return counters


def build_cycle_variants(
    variants: List[Dict[str, object]],
    start_time_utc: Optional[pd.Timestamp],
) -> Dict[str, Dict[str, int]]:
    ensure_dir(STATES_VAR_DIR)

    tfs = load_timeframes()
    if not tfs:
        raise RuntimeError(f"Keine H4+ Timeframe-Ordner gefunden unter: {TREND_FEAT_DIR}")

    per_variant_counts: Dict[str, Dict[str, int]] = {}

    for v in variants:
        vname = str(v["name"])
        per_variant_counts[vname] = {}

        for tf in tfs:
            csv_paths = list_symbols_for_tf(tf)
            if not csv_paths:
                per_variant_counts[vname]["no_symbols"] = per_variant_counts[vname].get("no_symbols", 0) + 1
                continue

            for p in csv_paths:
                symbol = safe_symbol_from_path(p)
                try:
                    trend_feat = read_trend_features(p, start_time_utc=start_time_utc)
                    if trend_feat.empty:
                        per_variant_counts[vname]["empty_trend_features"] = per_variant_counts[vname].get("empty_trend_features", 0) + 1
                        continue

                    outp = variant_state_out_path(vname, tf, symbol)
                    status, _info = compute_symbol_incremental(
                        tf=tf,
                        symbol=symbol,
                        trend_feat_full=trend_feat,
                        outp=outp,
                        lookback_bars=int(v["lookback_bars"]),
                        smooth_span=int(v["smooth_span"]),
                        w_spread=float(v["w_spread"]),
                        w_price=float(v["w_price"]),
                        w_slope=float(v["w_slope"]),
                        t_enter=float(v["t_enter"]),
                        t_exit=float(v["t_exit"]),
                        start_time_utc=start_time_utc,
                    )
                    per_variant_counts[vname][status] = per_variant_counts[vname].get(status, 0) + 1

                except Exception:
                    per_variant_counts[vname]["error"] = per_variant_counts[vname].get("error", 0) + 1

    return per_variant_counts


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=60.0)

    ap.add_argument("--run-base", action="store_true", help="write base states to States/Trend/<TF>/<SYMBOL>.csv")
    ap.add_argument("--run-variants", action="store_true", help="write variants to States/Trend/Variants/<NAME>/...")

    ap.add_argument("--variants-json", type=str, default=str(DEFAULT_VARIANTS_JSON), help="path to trend_variants.json")

    ap.add_argument("--lookback-bars", type=int, default=DEFAULT_LOOKBACK_BARS)
    ap.add_argument("--smooth-span", type=int, default=DEFAULT_SMOOTH_SPAN)
    ap.add_argument("--w-spread", type=float, default=DEFAULT_W_SPREAD)
    ap.add_argument("--w-price", type=float, default=DEFAULT_W_PRICE)
    ap.add_argument("--w-slope", type=float, default=DEFAULT_W_SLOPE)
    ap.add_argument("--t-enter", type=float, default=DEFAULT_T_ENTER)
    ap.add_argument("--t-exit", type=float, default=DEFAULT_T_EXIT)

    ap.add_argument("--start-time-utc", type=str, default="",
                    help='global cutoff, e.g. "2016-02-02 21:00:00+00:00"')

    return ap.parse_args()


def write_summary(
    run_base: bool,
    run_variants: bool,
    base_counts: Dict[str, int],
    variants_counts: Dict[str, Dict[str, int]],
    variants: List[Dict[str, object]],
    variants_json: Path,
    base_params: Dict[str, object],
    start_time_utc: Optional[pd.Timestamp],
) -> None:
    summary = {
        "meta": {
            "updated_at_utc": utc_now_str(),
            "root": str(ROOT.resolve()),
            "trend_features_dir": str(TREND_FEAT_DIR.resolve()),
            "states_base_dir": str(STATES_BASE_DIR.resolve()),
            "states_variants_dir": str(STATES_VAR_DIR.resolve()),
            "run_base": bool(run_base),
            "run_variants": bool(run_variants),
            "variants_json": str(variants_json.resolve()),
            "n_variants": int(len(variants)),
            "start_time_utc": (str(start_time_utc) if start_time_utc is not None else None),
            "allowed_timeframes": TIMEFRAMES_PREFERRED,
        },
        "base": {
            "params": base_params,
            "status_counts": base_counts,
        },
        "variants": {
            "definitions": variants,
            "status_counts": variants_counts,
        }
    }
    atomic_write_json(summary, SUMMARY_PATH)


def main() -> None:
    args = parse_args()

    if args.loop and args.once:
        raise ValueError("Use either --once or --loop (not both).")
    if not args.loop and not args.once:
        args.once = True

    if not TREND_FEAT_DIR.exists():
        raise RuntimeError(f"TREND_FEAT_DIR not found: {TREND_FEAT_DIR.resolve()}")

    start_time_utc = parse_time_utc_str(args.start_time_utc)

    variants_path = Path(args.variants_json).resolve()
    variants_exist = variants_path.exists()

    if not args.run_base and not args.run_variants:
        if variants_exist:
            args.run_variants = True
            print("[INFO] auto: --run-variants enabled (variants json exists)")
        else:
            args.run_base = True
            print("[INFO] auto: --run-base enabled (no variants json found)")

    variants: List[Dict[str, object]] = []
    if args.run_variants:
        try:
            variants = load_variants(variants_path)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in variants file: {variants_path} -> {e}") from e

        if len(variants) == 0:
            print(f"[WARN] variants json has 0 variants: {variants_path}")

    print("[INFO] ROOT           =", ROOT.resolve())
    print("[INFO] TREND_FEAT_DIR =", TREND_FEAT_DIR.resolve())
    print("[INFO] STATES_BASE    =", STATES_BASE_DIR.resolve())
    print("[INFO] STATES_VAR     =", STATES_VAR_DIR.resolve())
    print("[INFO] allowed TFs    =", TIMEFRAMES_PREFERRED)
    print("[INFO] lookback       =", int(args.lookback_bars))
    print("[INFO] start_time_utc =", str(start_time_utc) if start_time_utc is not None else None)
    print("[INFO] run_base       =", bool(args.run_base))
    print("[INFO] variants_json  =", str(variants_path))
    print("[INFO] variants_exist =", bool(variants_exist))
    print("[INFO] run_variants   =", bool(args.run_variants))
    print("[INFO] n_variants     =", int(len(variants)))
    print("[INFO] mode           =", "loop" if args.loop else "once")

    def run_once() -> None:
        base_counts: Dict[str, int] = {}
        variants_counts: Dict[str, Dict[str, int]] = {}

        if args.run_base:
            base_counts = build_cycle_base(
                lookback_bars=int(args.lookback_bars),
                smooth_span=int(args.smooth_span),
                w_spread=float(args.w_spread),
                w_price=float(args.w_price),
                w_slope=float(args.w_slope),
                t_enter=float(args.t_enter),
                t_exit=float(args.t_exit),
                start_time_utc=start_time_utc,
            )
            print("[INFO] base_status_counts:", base_counts)

        if args.run_variants:
            if len(variants) == 0:
                variants_counts = {}
                print("[WARN] run_variants=True but n_variants=0 -> nothing written")
            else:
                variants_counts = build_cycle_variants(variants, start_time_utc=start_time_utc)
                ok_sum = sum(v.get("ok", 0) for v in variants_counts.values())
                err_sum = sum(v.get("error", 0) for v in variants_counts.values())
                print("[INFO] variants_written_ok =", int(ok_sum), "errors =", int(err_sum))

        base_params = {
            "lookback_bars": int(args.lookback_bars),
            "smooth_span": int(args.smooth_span),
            "w_spread": float(args.w_spread),
            "w_price": float(args.w_price),
            "w_slope": float(args.w_slope),
            "t_enter": float(args.t_enter),
            "t_exit": float(args.t_exit),
        }

        write_summary(
            run_base=bool(args.run_base),
            run_variants=bool(args.run_variants),
            base_counts=base_counts,
            variants_counts=variants_counts,
            variants=variants,
            variants_json=variants_path,
            base_params=base_params,
            start_time_utc=start_time_utc,
        )

        print("[DONE] Trend states written ->", STATES_BASE_DIR.resolve())
        print("[DONE] Summary ->", SUMMARY_PATH.resolve())

    if args.once:
        run_once()
        return

    import time
    try:
        while True:
            run_once()
            print("[LOOP] updated_at_utc=", utc_now_str(), "->", SUMMARY_PATH.resolve())
            time.sleep(float(args.sleep))
    except KeyboardInterrupt:
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()

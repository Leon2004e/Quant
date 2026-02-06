# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/State_Dynamics.py

Zweck:
  Zusätzliche Dynamics-Messung für Trend-States (bull/neutral/bear):
    - state_age: Bars seit letztem State-Wechsel (0,1,2,...)
    - time_since_switch: Zeitdifferenz seit letztem State-Wechsel (Sekunden)
    - phase_ratio: state_age / avg_state_duration (avg pro Symbol/TF über gesamte Historie)

Optional:
  - Visualisierung: Overlay-Chart mit Farbintensität ∝ phase_ratio (je "älter" der State, desto stärker)

Input:
  <ROOT>/1_Data_Center/Data/Regime/States/Trend/<TF>/<SYMBOL>.csv     (time, trend_state)
  <ROOT>/1_Data_Center/Data/Regime/Ohcl/<TF>/<SYMBOL>.csv            (time, close)

Output:
  <ROOT>/1_Data_Center/Data/Regime/Dynamics/Trend/<TF>/<SYMBOL>.csv
  <ROOT>/1_Data_Center/Data/Regime/Dynamics/Trend/summary.json

Hinweise:
  - Kein Filtern/Ranking, nur Messung.
  - phase_ratio nutzt avg_state_duration über die gesamte Serie (stabil, deterministisch).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

STATES_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "States" / "Trend"
OHLC_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Ohcl"

DYN_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Dynamics" / "Trend"
SUMMARY_PATH = DYN_DIR / "summary.json"

PLOTS_DIR = DYN_DIR / "Plots"


# =========================
# SETTINGS
# =========================

TIMEFRAMES_PREFERRED = ["M1", "M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]

DEFAULT_PHASE_CAP = 1.5     # phase_ratio clipped for visualization
DEFAULT_MAX_POINTS = 6000   # cap points per plot


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


def safe_symbol_from_path(p: Path) -> str:
    return p.stem


def out_path(tf: str, symbol: str) -> Path:
    return DYN_DIR / tf / f"{symbol}.csv"


def plot_path(tf: str, symbol: str) -> Path:
    return PLOTS_DIR / tf / f"{symbol}.png"


# =========================
# Discovery
# =========================

def load_timeframes() -> List[str]:
    if not STATES_DIR.exists():
        return []
    tfs = [p.name for p in STATES_DIR.iterdir() if p.is_dir()]
    return [x for x in TIMEFRAMES_PREFERRED if x in tfs] + sorted([x for x in tfs if x not in TIMEFRAMES_PREFERRED])


def list_symbols_for_tf(tf: str) -> List[Path]:
    tf_dir = STATES_DIR / tf
    if not tf_dir.exists():
        return []
    return sorted(tf_dir.glob("*.csv"))


# =========================
# Read helpers
# =========================

def read_states_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "time" not in df.columns or "trend_state" not in df.columns:
        raise ValueError(f"states missing required cols in: {path}")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["trend_state"] = pd.to_numeric(df["trend_state"], errors="coerce")
    df = (
        df.dropna(subset=["time", "trend_state"])
          .sort_values("time")
          .drop_duplicates(subset=["time"], keep="last")
          .reset_index(drop=True)
    )
    df["trend_state"] = df["trend_state"].astype(int).clip(-1, 1)
    return df


def read_ohlc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    if "time" not in df.columns or "close" not in df.columns:
        raise ValueError(f"ohlc missing required cols in: {path}")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = (
        df.dropna(subset=["time", "close"])
          .sort_values("time")
          .drop_duplicates(subset=["time"], keep="last")
          .reset_index(drop=True)
    )
    return df


def merge_states_ohlc(states: pd.DataFrame, ohlc: pd.DataFrame) -> pd.DataFrame:
    if states.empty or ohlc.empty:
        return pd.DataFrame(columns=["time", "trend_state", "close"])
    m = pd.merge(states[["time", "trend_state"]], ohlc[["time", "close"]], on="time", how="inner")
    m = m.dropna(subset=["time", "trend_state", "close"]).sort_values("time").reset_index(drop=True)
    m["trend_state"] = m["trend_state"].astype(int).clip(-1, 1)
    m["close"] = pd.to_numeric(m["close"], errors="coerce")
    m = m.dropna(subset=["close"]).reset_index(drop=True)
    return m


# =========================
# Dynamics computation
# =========================

def _run_lengths(x: np.ndarray) -> np.ndarray:
    if len(x) == 0:
        return np.array([], dtype=int)
    change = np.flatnonzero(np.diff(x) != 0) + 1
    boundaries = np.r_[0, change, len(x)]
    return np.diff(boundaries).astype(int)


def compute_state_age_and_switch_time(times: np.ndarray, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      state_age_bars: int array, bars since last switch (0..)
      time_since_switch_sec: float array, seconds since last switch
      run_id: int array, increments on each new run
    """
    n = len(states)
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.array([], dtype=int)

    # detect switch points
    switches = np.r_[True, states[1:] != states[:-1]]  # True at first and at each change
    run_id = np.cumsum(switches.astype(int)) - 1       # 0-based run id

    state_age = np.empty(n, dtype=int)
    time_since = np.empty(n, dtype=float)

    last_idx = 0
    last_t = times[0]
    cur_run = run_id[0]

    for i in range(n):
        if run_id[i] != cur_run:
            cur_run = run_id[i]
            last_idx = i
            last_t = times[i]
        state_age[i] = i - last_idx

        # seconds since switch
        dt = (times[i] - last_t)
        # numpy datetime64 -> timedelta64
        try:
            sec = dt / np.timedelta64(1, "s")
        except Exception:
            sec = np.nan
        time_since[i] = float(sec) if np.isfinite(sec) else np.nan

    return state_age, time_since, run_id


def compute_phase_ratio(state_age_bars: np.ndarray, avg_state_duration: float) -> np.ndarray:
    if not np.isfinite(avg_state_duration) or avg_state_duration <= 0:
        return np.full(len(state_age_bars), np.nan, dtype=float)
    return state_age_bars.astype(float) / float(avg_state_duration)


def compute_dynamics(merged: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    merged columns: time, trend_state, close
    Output df includes:
      time, trend_state, close, run_id, state_age, time_since_switch_sec, phase_ratio
    """
    if merged.empty:
        return pd.DataFrame(), {"avg_state_duration": np.nan, "n_rows": 0.0}

    t = merged["time"].to_numpy(dtype="datetime64[ns]")
    st = merged["trend_state"].to_numpy(dtype=int)

    run_lengths = _run_lengths(st)
    avg_dur = float(np.mean(run_lengths)) if len(run_lengths) else np.nan

    state_age, time_since_sec, run_id = compute_state_age_and_switch_time(t, st)
    phase_ratio = compute_phase_ratio(state_age, avg_dur)

    out = merged.copy()
    out["run_id"] = run_id.astype(int)
    out["state_age"] = state_age.astype(int)
    out["time_since_switch_sec"] = time_since_sec.astype(float)
    out["phase_ratio"] = phase_ratio.astype(float)

    meta = {
        "avg_state_duration": avg_dur,
        "n_rows": float(len(out)),
        "n_runs": float(len(run_lengths)),
        "min_run": float(np.min(run_lengths)) if len(run_lengths) else np.nan,
        "max_run": float(np.max(run_lengths)) if len(run_lengths) else np.nan,
    }
    return out, meta


# =========================
# Visualization (optional)
# =========================

def _state_color_hex(s: int) -> str:
    if s == 1:
        return "#2ca02c"  # bull
    if s == -1:
        return "#d62728"  # bear
    return "#7f7f7f"      # neutral


def plot_overlay_intensity(
    tf: str,
    symbol: str,
    df: pd.DataFrame,
    outp: Path,
    phase_cap: float = DEFAULT_PHASE_CAP,
    max_points: int = DEFAULT_MAX_POINTS,
    mark_switches: bool = True,
) -> bool:
    """
    Close-Linie + Hintergrund-Färbung nach State.
    Alpha ∝ phase_ratio (höher = intensiver), gecappt bei phase_cap.
    """
    if df.empty or "time" not in df.columns or "close" not in df.columns or "trend_state" not in df.columns:
        return False

    ensure_dir(outp.parent)

    d = df.copy()
    if len(d) > int(max_points):
        d = d.iloc[-int(max_points):].copy()

    t = d["time"].to_numpy()
    close = d["close"].to_numpy(dtype=float)
    st = d["trend_state"].to_numpy(dtype=int)

    pr = d["phase_ratio"].to_numpy(dtype=float) if "phase_ratio" in d.columns else np.full(len(d), np.nan)
    pr_clip = np.clip(pr, 0.0, float(phase_cap))
    pr_norm = pr_clip / float(phase_cap)  # 0..1

    plt.figure(figsize=(14, 6))
    plt.plot(t, close, linewidth=1.0)

    # runs
    changes = np.flatnonzero(np.diff(st) != 0) + 1
    boundaries = np.r_[0, changes, len(st)]

    # Base alpha + additional alpha from age
    base_alpha = 0.06
    add_alpha = 0.22

    for i in range(len(boundaries) - 1):
        a = int(boundaries[i])
        b = int(boundaries[i + 1])
        s = int(st[a])

        # use mean normalized phase within the segment for intensity
        seg = pr_norm[a:b]
        seg_val = float(np.nanmean(seg)) if np.any(np.isfinite(seg)) else 0.0
        alpha = float(np.clip(base_alpha + add_alpha * seg_val, 0.02, 0.35))

        plt.axvspan(t[a], t[b - 1], alpha=alpha, color=_state_color_hex(s), linewidth=0)

    if mark_switches and len(changes) > 0:
        sw_t = t[changes]
        sw_y = close[changes]
        plt.scatter(sw_t, sw_y, s=10)

    plt.title(f"Trend State Dynamics Overlay (intensity ~ phase_ratio): {symbol} | {tf}")
    plt.xlabel("Time")
    plt.ylabel("Close")
    plt.tight_layout()
    plt.savefig(outp, dpi=160)
    plt.close()
    return True


# =========================
# One symbol processing
# =========================

def compute_symbol(tf: str, symbol: str, do_plot: bool, phase_cap: float, max_points: int) -> Tuple[str, Dict[str, object]]:
    states_path = STATES_DIR / tf / f"{symbol}.csv"
    ohlc_path = OHLC_DIR / tf / f"{symbol}.csv"

    if not states_path.exists():
        return "missing_states", {"states_file": str(states_path)}
    if not ohlc_path.exists():
        return "missing_ohlc", {"ohlc_file": str(ohlc_path)}

    states = read_states_csv(states_path)
    if states.empty:
        return "empty_states", {"states_file": str(states_path)}

    ohlc = read_ohlc_csv(ohlc_path)
    if ohlc.empty:
        return "empty_ohlc", {"ohlc_file": str(ohlc_path)}

    merged = merge_states_ohlc(states, ohlc)
    if merged.empty:
        return "no_overlap", {"states_file": str(states_path), "ohlc_file": str(ohlc_path)}

    dyn_df, meta = compute_dynamics(merged)
    if dyn_df.empty:
        return "empty_dynamics", {}

    outp = out_path(tf, symbol)
    atomic_write_csv(dyn_df, outp)

    plot_ok = False
    plot_file = None
    if do_plot:
        pout = plot_path(tf, symbol)
        plot_ok = plot_overlay_intensity(tf, symbol, dyn_df, pout, phase_cap=phase_cap, max_points=max_points, mark_switches=True)
        plot_file = str(pout) if plot_ok else None

    info = {
        "file": str(outp),
        "rows_saved": int(len(dyn_df)),
        "from_utc": str(pd.to_datetime(dyn_df["time"], utc=True).min()),
        "to_utc": str(pd.to_datetime(dyn_df["time"], utc=True).max()),
        "avg_state_duration": float(meta.get("avg_state_duration", np.nan)),
        "n_runs": float(meta.get("n_runs", np.nan)),
        "min_run": float(meta.get("min_run", np.nan)),
        "max_run": float(meta.get("max_run", np.nan)),
        "plot_file": plot_file,
        "updated_at_utc": utc_now_str(),
    }
    return "ok", info


# =========================
# Build cycle
# =========================

def build_cycle(do_plot: bool, phase_cap: float, max_points: int, verbose_counts: bool = True) -> Dict[str, object]:
    ensure_dir(DYN_DIR)
    if do_plot:
        ensure_dir(PLOTS_DIR)

    tfs = load_timeframes()
    if not tfs:
        raise RuntimeError(f"Keine Timeframe-Ordner gefunden unter: {STATES_DIR}")

    summary: Dict[str, object] = {
        "meta": {
            "updated_at_utc": utc_now_str(),
            "root": str(ROOT.resolve()),
            "states_dir": str(STATES_DIR.resolve()),
            "ohlc_dir": str(OHLC_DIR.resolve()),
            "dyn_dir": str(DYN_DIR.resolve()),
            "plots_dir": str(PLOTS_DIR.resolve()),
            "do_plot": bool(do_plot),
            "phase_cap": float(phase_cap),
            "max_points": int(max_points),
        },
        "timeframes": {},
    }

    counters: Dict[str, int] = {}

    for tf in tfs:
        tf_block: Dict[str, object] = {"status": "ok", "symbols": {}}
        paths = list_symbols_for_tf(tf)
        if not paths:
            tf_block["status"] = "no_symbols"
            summary["timeframes"][tf] = tf_block
            continue

        for p in paths:
            symbol = safe_symbol_from_path(p)
            try:
                status, info = compute_symbol(tf, symbol, do_plot=do_plot, phase_cap=phase_cap, max_points=max_points)
                tf_block["symbols"][symbol] = {"status": status, **info}
                counters[status] = counters.get(status, 0) + 1
            except Exception as e:
                tf_block["symbols"][symbol] = {"status": "error", "error": str(e), "updated_at_utc": utc_now_str()}
                counters["error"] = counters.get("error", 0) + 1

        summary["timeframes"][tf] = tf_block

    summary["meta"]["status_counts"] = counters
    atomic_write_json(summary, SUMMARY_PATH)

    if verbose_counts:
        print("[INFO] status_counts:", counters)
        print("[INFO] summary ->", str(SUMMARY_PATH.resolve()))

    return summary


# =========================
# CLI
# =========================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=60.0)

    ap.add_argument("--plot", action="store_true", help="write overlay charts with intensity ~ phase_ratio")
    ap.add_argument("--phase-cap", type=float, default=DEFAULT_PHASE_CAP)
    ap.add_argument("--max-points", type=int, default=DEFAULT_MAX_POINTS)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.loop and args.once:
        raise ValueError("Use either --once or --loop (not both).")
    if not args.loop and not args.once:
        args.once = True

    if not STATES_DIR.exists():
        raise RuntimeError(f"STATES_DIR not found: {STATES_DIR.resolve()}")
    if not OHLC_DIR.exists():
        raise RuntimeError(f"OHLC_DIR not found: {OHLC_DIR.resolve()}")

    print("[INFO] ROOT      =", ROOT.resolve())
    print("[INFO] STATES    =", STATES_DIR.resolve())
    print("[INFO] OHLC      =", OHLC_DIR.resolve())
    print("[INFO] DYN_DIR   =", DYN_DIR.resolve())
    print("[INFO] plot      =", bool(args.plot))
    print("[INFO] phase_cap =", float(args.phase_cap))
    print("[INFO] max_points=", int(args.max_points))
    print("[INFO] mode      =", "loop" if args.loop else "once")

    if args.once:
        build_cycle(do_plot=bool(args.plot), phase_cap=float(args.phase_cap), max_points=int(args.max_points), verbose_counts=True)
        print("[DONE] Dynamics written ->", DYN_DIR.resolve())
        print("[DONE] Summary ->", SUMMARY_PATH.resolve())
        if args.plot:
            print("[DONE] Plots ->", PLOTS_DIR.resolve())
        return

    import time
    try:
        while True:
            build_cycle(do_plot=bool(args.plot), phase_cap=float(args.phase_cap), max_points=int(args.max_points), verbose_counts=True)
            print("[LOOP] updated_at_utc=", utc_now_str(), "->", SUMMARY_PATH.resolve())
            time.sleep(float(args.sleep))
    except KeyboardInterrupt:
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Market_Performance_Labeling.py (loop + inkrementell labeln)

Fix:
- Kein pandas Nullable dtype "Int64" mehr -> nutze np.int8
- Lookahead-Zeilen bleiben NaN für Returns; für Klassenlabels setzen wir NaN und casten robust

Pfad:
- SYSTEM_ROOT = parents[2] (QUANT_SYSTEM_V2)
- OHLC_ROOT   = SYSTEM_ROOT/1_Data_Center/Data/Regime/Ohcl
- OUT_ROOT    = SYSTEM_ROOT/1_Data_Center/Data/Regime/States/Market_Performance
"""

from __future__ import annotations

import os
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SYSTEM_ROOT = Path(__file__).resolve().parents[2]
OHLC_ROOT = SYSTEM_ROOT / "1_Data_Center" / "Data" / "Regime" / "Ohcl"
OUT_ROOT = SYSTEM_ROOT / "1_Data_Center" / "Data" / "Regime" / "States" / "Market_Performance"
CHECKPOINT_PATH = OUT_ROOT / "_checkpoint.json"


# ============================================================
# KONFIG
# ============================================================

AUTO_TF_DISCOVERY = True
BASE_TIMEFRAMES: List[str] = ["M1", "M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]

HORIZONS_BARS: List[int] = [1, 2, 4, 8]
VOL_WINDOW = 64
K_THRESHOLD = 0.5

UPDATE_EVERY_SECONDS = 30.0
OVERLAP_BARS = max(VOL_WINDOW, max(HORIZONS_BARS) + 5)

CSV_TIME_COL = "time"


# ============================================================
# IO Helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_df_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def atomic_write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def read_csv_time_sorted(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if CSV_TIME_COL in df.columns:
        df[CSV_TIME_COL] = pd.to_datetime(df[CSV_TIME_COL], utc=True, errors="coerce")
        df = df.dropna(subset=[CSV_TIME_COL]).sort_values(CSV_TIME_COL).reset_index(drop=True)
    return df


def out_path(tf: str, symbol_csv_name: str) -> Path:
    return OUT_ROOT / tf / symbol_csv_name


def load_checkpoint() -> Dict[str, str]:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            return {str(k): str(v) for k, v in d.items()}
    except Exception:
        pass
    return {}


def save_checkpoint(cp: Dict[str, str]) -> None:
    atomic_write_json(cp, CHECKPOINT_PATH)


# ============================================================
# Label Logic
# ============================================================

def _to_int8_with_nan(x: np.ndarray) -> np.ndarray:
    """
    Cast float array with possible NaN to int8 but keep NaN as np.nan by returning float array.
    CSV speichert das ok (int columns mit NaN werden float in pandas).
    Wenn du strikt int willst, musst du NaNs vorher imputen.
    """
    # Wir lassen NaNs drin -> float
    return x.astype(float)


def compute_market_labels(
    df_ohlc: pd.DataFrame,
    horizons: List[int],
    vol_window: int,
    k: float,
) -> pd.DataFrame:
    if df_ohlc is None or df_ohlc.empty:
        return pd.DataFrame()

    d = df_ohlc.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)

    close = d["close"].astype(float)
    d["r1"] = np.log(close).diff()
    d["sigma"] = d["r1"].rolling(int(vol_window)).std()

    for h in horizons:
        h = int(h)

        # forward log return
        d[f"y_ret_h{h}"] = np.log(close.shift(-h) / close)

        denom = d["sigma"] * np.sqrt(h)
        d[f"y_ret_voladj_h{h}"] = d[f"y_ret_h{h}"] / denom

        # direction: NaN wenn y_ret NaN
        dir_raw = np.where(d[f"y_ret_h{h}"].isna(), np.nan, (d[f"y_ret_h{h}"] > 0).astype(np.int8))
        d[f"y_dir_h{h}"] = _to_int8_with_nan(dir_raw)

        # 3-class: NaN wenn y_ret oder tau NaN
        tau = k * denom
        cls = np.select(
            [d[f"y_ret_h{h}"] > tau, d[f"y_ret_h{h}"] < -tau],
            [1, -1],
            default=0
        ).astype(np.int8)

        cls = np.where(d[f"y_ret_h{h}"].isna() | tau.isna(), np.nan, cls)
        d[f"y_3class_h{h}"] = _to_int8_with_nan(cls)

    keep = ["time", "sigma"] + [c for c in d.columns if c.startswith("y_")]
    return d[keep].copy()


def merge_labels(existing: pd.DataFrame, new_part: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        out = new_part.copy()
    elif new_part is None or new_part.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new_part], ignore_index=True)

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


# ============================================================
# Incremental Update
# ============================================================

def relabel_incremental(ohlc_file: Path, tf: str, cp: Dict[str, str]) -> Optional[Path]:
    symbol_csv_name = ohlc_file.name
    key = f"{tf}/{symbol_csv_name}"

    df_ohlc = read_csv_time_sorted(ohlc_file)
    if df_ohlc.empty:
        return None

    last_labeled_time = None
    if key in cp:
        t = pd.to_datetime(cp[key], utc=True, errors="coerce")
        if not pd.isna(t):
            last_labeled_time = t

    if last_labeled_time is None:
        df_slice = df_ohlc
    else:
        times = pd.to_datetime(df_ohlc["time"], utc=True, errors="coerce").values
        target = np.array(last_labeled_time.to_datetime64())
        pos = int(np.searchsorted(times, target, side="left"))
        start_pos = max(0, pos - int(OVERLAP_BARS))
        df_slice = df_ohlc.iloc[start_pos:].copy()

    new_labels = compute_market_labels(df_slice, HORIZONS_BARS, VOL_WINDOW, K_THRESHOLD)
    if new_labels.empty:
        return None

    outp = out_path(tf, symbol_csv_name)
    existing = read_csv_time_sorted(outp)
    merged = merge_labels(existing, new_labels)

    atomic_write_df_csv(merged, outp)

    cp[key] = str(pd.to_datetime(df_ohlc["time"].max(), utc=True))
    return outp


# ============================================================
# MAIN LOOP
# ============================================================

def discover_timeframes() -> List[str]:
    if not OHLC_ROOT.exists():
        return []
    tfs = [p.name for p in OHLC_ROOT.iterdir() if p.is_dir()]
    tfs.sort()
    return tfs


def main() -> None:
    ensure_dir(OUT_ROOT)

    if not OHLC_ROOT.exists():
        raise RuntimeError(f"OHLC_ROOT not found: {OHLC_ROOT}")

    tfs = discover_timeframes() if AUTO_TF_DISCOVERY else BASE_TIMEFRAMES
    if not tfs:
        raise RuntimeError(f"No timeframe directories found under: {OHLC_ROOT}")

    print(f"[INFO] SYSTEM_ROOT = {SYSTEM_ROOT}")
    print(f"[INFO] OHLC_ROOT   = {OHLC_ROOT}")
    print(f"[INFO] OUT_ROOT    = {OUT_ROOT}")
    print(f"[INFO] TFs         = {tfs}")

    cp = load_checkpoint()

    while True:
        loop_start = time.time()

        for tf in tfs:
            tf_dir = OHLC_ROOT / tf
            if not tf_dir.exists():
                continue

            for p in sorted(tf_dir.glob("*.csv")):
                try:
                    outp = relabel_incremental(p, tf, cp)
                    if outp is not None:
                        print(f"[OK] labeled {tf}/{p.name}")
                except Exception as e:
                    print(f"[WARN] label failed {tf}/{p.name}: {e}")

        save_checkpoint(cp)

        elapsed = time.time() - loop_start
        sleep_s = max(0.0, UPDATE_EVERY_SECONDS - elapsed)
        print(f"[LOOP] done in {elapsed:.2f}s, sleeping {sleep_s:.2f}s")
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()

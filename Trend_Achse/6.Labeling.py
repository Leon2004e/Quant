# -*- coding: utf-8 -*-
"""
4_Regime_Builder/Trend_Achse/6.Labeling.py

H1 labeling (hour + session) WITHOUT regime merge.

Input:
  <ROOT>/1_Data_Center/Data/Regime/Ohcl/H1/<SYMBOL>.csv

Output:
  <ROOT>/1_Data_Center/Data/Regime/Labeling/H1/<SYMBOL>.csv
  + summary.json

Adds:
  date_utc, dow, hour_utc,
  session_block, session_core, hour_in_session, session_date,
  ret_cc, ret_oc

Sessions (UTC):
  block:
    ASIA: 00-06
    EU  : 07-12
    US  : 13-20
    OFF : 21-23
  core-hours flag (your focus):
    ASIA: 02-05
    EU  : 08-11
    US  : 14-17
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import traceback
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

# NOTE: your folder name is "Ohcl" (not Ohlc)
OHLC_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Ohcl" / "H1"
LABEL_DIR = ROOT / "1_Data_Center" / "Data" / "Regime" / "Labeling" / "H1"
SUMMARY_PATH = LABEL_DIR.parent / "summary.json"


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


# =========================
# Time helpers
# =========================

def parse_time_utc_series(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, utc=True, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(s, utc=True, errors="coerce")


def parse_time_utc_str(s: str) -> Optional[pd.Timestamp]:
    ss = str(s or "").strip()
    if not ss:
        return None
    ts = pd.to_datetime(ss, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid --start-time-utc: {s}")
    return ts


# =========================
# Column normalization
# =========================

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make columns lowercase and map common names to:
      time, open, high, low, close
    """
    if df.empty:
        return df

    cols = {c: str(c).strip() for c in df.columns}
    df = df.rename(columns=cols)

    lower = {c: c.lower().strip() for c in df.columns}
    df = df.rename(columns=lower)

    # common aliases
    alias = {
        "datetime": "time",
        "date": "time",
        "timestamp": "time",
        "ts": "time",
        "t": "time",
        "open": "open",
        "o": "open",
        "high": "high",
        "h": "high",
        "low": "low",
        "l": "low",
        "close": "close",
        "c": "close",
        "adj_close": "close",
        "adjclose": "close",
    }

    # apply alias mapping if alias exists
    for c in list(df.columns):
        if c in alias:
            df = df.rename(columns={c: alias[c]})

    return df


# =========================
# Read / incremental
# =========================

def read_h1_ohlc(path: Path, start_time_utc: Optional[pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df

    df = _normalize_cols(df)

    if "time" not in df.columns:
        raise ValueError(f"missing time column in {path.name}. cols={list(df.columns)[:20]}")

    # parse time
    df["time"] = parse_time_utc_series(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # require ohlc
    need = ["open", "high", "low", "close"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"missing OHLC columns {missing} in {path.name}. cols={list(df.columns)[:20]}")

    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=need).reset_index(drop=True)

    # basic sanity
    df = df[(df["high"] >= df["low"])].reset_index(drop=True)

    if start_time_utc is not None and not df.empty:
        df = df.loc[df["time"] >= start_time_utc].reset_index(drop=True)

    # dedup time
    df = df.drop_duplicates("time", keep="last").reset_index(drop=True)
    return df


def read_existing_labels(path: Path, start_time_utc: Optional[pd.Timestamp]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame()

    df = _normalize_cols(df)
    if "time" not in df.columns:
        return pd.DataFrame()

    df["time"] = parse_time_utc_series(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time")
    df = df.drop_duplicates("time", keep="last").reset_index(drop=True)

    if start_time_utc is not None and not df.empty:
        df = df.loc[df["time"] >= start_time_utc].reset_index(drop=True)

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

    out = _normalize_cols(out)
    out["time"] = parse_time_utc_series(out["time"])
    out = out.dropna(subset=["time"])
    out = out.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
    return out


# =========================
# Labeling core (vectorized)
# =========================

def label_h1(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # returns
    out["ret_cc"] = np.log(out["close"] / out["close"].shift(1))
    out["ret_oc"] = np.log(out["close"] / out["open"])

    # time parts
    out["date_utc"] = out["time"].dt.date.astype(str)
    out["dow"] = out["time"].dt.dayofweek.astype(int)
    out["hour_utc"] = out["time"].dt.hour.astype(int)

    h = out["hour_utc"].astype(int)

    # session_block (broad)
    block = np.full(len(out), "OFF", dtype=object)
    block[(h >= 0) & (h <= 6)] = "ASIA"
    block[(h >= 7) & (h <= 12)] = "EU"
    block[(h >= 13) & (h <= 20)] = "US"
    out["session_block"] = block

    # session_core flag (your main hours)
    core = ((h >= 2) & (h <= 5)) | ((h >= 8) & (h <= 11)) | ((h >= 14) & (h <= 17))
    out["session_core"] = core.astype(int)

    # hour_in_session
    start_hour = np.full(len(out), 21, dtype=int)
    start_hour[out["session_block"] == "ASIA"] = 0
    start_hour[out["session_block"] == "EU"] = 7
    start_hour[out["session_block"] == "US"] = 13
    out["hour_in_session"] = (h.to_numpy() - start_hour).astype(int)

    # stable aggregation key for session daily buckets
    out["session_date"] = out["time"].dt.normalize().dt.date.astype(str)

    return out


def process_symbol(
    symbol: str,
    start_time_utc: Optional[pd.Timestamp],
    lookback_bars: int,
) -> Tuple[str, Dict[str, object]]:
    in_path = OHLC_DIR / f"{symbol}.csv"
    out_path = LABEL_DIR / f"{symbol}.csv"

    ohlc = read_h1_ohlc(in_path, start_time_utc=start_time_utc)
    if ohlc.empty:
        return "empty_ohlc", {"symbol": symbol, "in": str(in_path), "out": str(out_path)}

    existing = read_existing_labels(out_path, start_time_utc=start_time_utc)
    last_time = pd.to_datetime(existing["time"].max(), utc=True) if (not existing.empty and "time" in existing.columns) else None

    ohlc_slice = slice_for_incremental(ohlc, last_time=last_time, lookback_bars=int(lookback_bars))
    if ohlc_slice.empty:
        return "empty_slice", {"symbol": symbol, "out": str(out_path)}

    lab = label_h1(ohlc_slice)

    if start_time_utc is not None and not lab.empty:
        lab = lab.loc[lab["time"] >= start_time_utc].reset_index(drop=True)

    merged = merge_on_time(existing, lab)

    cols = [
        "time", "open", "high", "low", "close",
        "ret_cc", "ret_oc",
        "date_utc", "dow", "hour_utc",
        "session_block", "session_core", "hour_in_session", "session_date",
    ]
    cols = [c for c in cols if c in merged.columns] + [c for c in merged.columns if c not in cols]
    merged = merged[cols]

    atomic_write_csv(merged, out_path)

    info = {
        "symbol": symbol,
        "in": str(in_path),
        "out": str(out_path),
        "rows_saved": int(len(merged)),
        "from_utc": str(pd.to_datetime(merged["time"], utc=True).min()),
        "to_utc": str(pd.to_datetime(merged["time"], utc=True).max()),
        "updated_at_utc": utc_now_str(),
    }
    return "ok", info


# =========================
# CLI / main
# =========================

def list_symbols() -> List[str]:
    if not OHLC_DIR.exists():
        return []
    return sorted([p.stem for p in OHLC_DIR.glob("*.csv")])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=60.0)

    ap.add_argument("--start-time-utc", type=str, default="",
                    help='global cutoff, e.g. "2016-02-02 21:00:00+00:00"')

    ap.add_argument("--lookback-bars", type=int, default=400,
                    help="incremental lookback bars for stable re-label around last_time")

    ap.add_argument("--print-errors", type=int, default=5,
                    help="print first N errors to terminal")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.loop and args.once:
        raise ValueError("Use either --once or --loop (not both).")
    if not args.loop and not args.once:
        args.once = True

    if not OHLC_DIR.exists():
        raise RuntimeError(f"OHLC_DIR not found: {OHLC_DIR.resolve()}")

    start_time_utc = parse_time_utc_str(args.start_time_utc)
    ensure_dir(LABEL_DIR)

    def run_once() -> Dict[str, object]:
        syms = list_symbols()
        if not syms:
            raise RuntimeError(f"No symbols found in: {OHLC_DIR.resolve()}")

        counts: Dict[str, int] = {}
        details: Dict[str, object] = {}
        errors_printed = 0

        for sym in syms:
            try:
                status, info = process_symbol(
                    symbol=sym,
                    start_time_utc=start_time_utc,
                    lookback_bars=int(args.lookback_bars),
                )
                counts[status] = counts.get(status, 0) + 1
                details[sym] = {"status": status, **info}

            except Exception as e:
                tb = traceback.format_exc()
                counts["error"] = counts.get("error", 0) + 1
                details[sym] = {
                    "status": "error",
                    "symbol": sym,
                    "error": str(e),
                    "traceback": tb,
                }
                if errors_printed < int(args.print_errors):
                    errors_printed += 1
                    print(f"[ERROR] {sym}: {e}")
                    # print the top of traceback for quick hint
                    print(tb.splitlines()[-6:])

        summary = {
            "meta": {
                "updated_at_utc": utc_now_str(),
                "root": str(ROOT.resolve()),
                "ohlc_h1_dir": str(OHLC_DIR.resolve()),
                "label_h1_dir": str(LABEL_DIR.resolve()),
                "start_time_utc": (str(start_time_utc) if start_time_utc is not None else None),
                "lookback_bars": int(args.lookback_bars),
            },
            "status_counts": counts,
            "symbols": details,
        }
        atomic_write_json(summary, SUMMARY_PATH)
        print("[INFO] status_counts:", counts)
        print("[DONE] summary ->", SUMMARY_PATH.resolve())
        return summary

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

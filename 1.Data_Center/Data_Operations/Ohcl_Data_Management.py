# -*- coding: utf-8 -*-
"""
Ohcl_Data_Management.py (loop + sauber speichern)

Änderung:
- zusätzlich H4, H8, H12 in BASE_TIMEFRAMES aufgenommen
- TF_MAP enthält die Enums bereits (H4/H8/H12), sonst keine Logikänderung
"""

from __future__ import annotations

import os
import json
import time
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "Data" / "Regime" / "Ohcl"


# ============================================================
# KONFIG
# ============================================================

FIXED_MT5_DIR = Path(
    r"C:\Users\Leon\Desktop\MetaTrader 5 - Kopie - Kopie - Kopie (10) - Kopie - Kopie - Kopie - Kopie"
)

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "540130486"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "T4b*5J2si")
MT5_SERVER = os.getenv("MT5_SERVER", "FTMO-Server4")

PORTABLE = True
START_TERMINAL = True

INIT_TIMEOUT_MS = 20000
STARTUP_WAIT_SECONDS = 5
INIT_RETRIES = 8
RETRY_SLEEP_SECONDS = 2

SYMBOLS: List[str] = [
    "AUDJPY", "AUDUSD", "EURGBP", "EURUSD", "GBPUSD", "GBPJPY",
    "NZDUSD", "US500.cash", "USDCAD", "USDCHF", "USDJPY", "USOIL.cash", "XAUUSD"
]

# Direkte MT5-Timeframes (inkl. H4/H8/H12)
BASE_TIMEFRAMES: List[str] = ["M1", "M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1"]

# Abgeleitet aus MN1
DERIVED_TIMEFRAMES: List[str] = ["Q", "Y"]

CHUNK_DAYS = 180
SLEEP_BETWEEN_CALLS = 0.05

EARLIEST_SEARCH_HORIZONS_DAYS = [30, 90, 180, 365, 365 * 3, 365 * 10, 365 * 20, 365 * 40]

# LOOP
UPDATE_EVERY_SECONDS = 30.0
SAFETY_LOOKBACK_BARS = 5
CSV_TIME_COL = "time"

TF_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M2": mt5.TIMEFRAME_M2,
    "M3": mt5.TIMEFRAME_M3,
    "M4": mt5.TIMEFRAME_M4,
    "M5": mt5.TIMEFRAME_M5,
    "M6": mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H2": mt5.TIMEFRAME_H2,
    "H3": mt5.TIMEFRAME_H3,
    "H4": mt5.TIMEFRAME_H4,
    "H6": mt5.TIMEFRAME_H6,
    "H8": mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}


# ============================================================
# MT5 Terminal / Connection
# ============================================================

def find_terminal64(root: Path) -> Path:
    exe = root / "terminal64.exe"
    if exe.exists():
        return exe
    hits = list(root.rglob("terminal64.exe"))
    if not hits:
        raise RuntimeError(f"terminal64.exe nicht gefunden unter: {root}")
    hits.sort(key=lambda p: p.stat().st_size, reverse=True)
    return hits[0]


def start_terminal(exe: Path) -> None:
    args = [str(exe)]
    if PORTABLE:
        args.append("/portable")
    subprocess.Popen(
        args,
        cwd=str(exe.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def connect_mt5(exe: Path) -> int:
    if not MT5_PASSWORD:
        raise RuntimeError("MT5_PASSWORD fehlt (ENV setzen)")

    mt5.shutdown()

    if START_TERMINAL:
        start_terminal(exe)
        time.sleep(STARTUP_WAIT_SECONDS)

    last_err = None
    for i in range(1, INIT_RETRIES + 1):
        if mt5.initialize(path=str(exe), portable=PORTABLE, timeout=INIT_TIMEOUT_MS):
            break
        last_err = mt5.last_error()
        print(f"[WARN] initialize failed ({i}/{INIT_RETRIES}): {last_err}")
        time.sleep(RETRY_SLEEP_SECONDS)
    else:
        raise RuntimeError(f"initialize failed: {last_err}")

    for i in range(1, INIT_RETRIES + 1):
        if mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            break
        last_err = mt5.last_error()
        print(f"[WARN] login failed ({i}/{INIT_RETRIES}): {last_err}")
        time.sleep(RETRY_SLEEP_SECONDS)
    else:
        mt5.shutdown()
        raise RuntimeError(f"login failed: {last_err}")

    acc = mt5.account_info()
    if acc is None:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"account_info failed: {err}")

    print(f"[OK] Connected: {acc.login} | {acc.name} | {MT5_SERVER}")
    return int(acc.login)


# ============================================================
# Helpers / IO
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace("/", "_") + ".csv"


def ohlc_path(out_dir: Path, symbol: str, tf: str) -> Path:
    return out_dir / tf / safe_symbol_filename(symbol)


def atomic_write_df_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(tmp_fd)
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
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(tmp_fd)
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


def ensure_symbol_selected(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol nicht gefunden: {symbol}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select failed: {symbol} / {mt5.last_error()}")


# ============================================================
# Data fetch
# ============================================================

def chunked_ranges(dt_from: datetime, dt_to: datetime, chunk_days: int) -> List[Tuple[datetime, datetime]]:
    out = []
    cur = dt_from
    step = timedelta(days=int(chunk_days))
    while cur < dt_to:
        nxt = min(dt_to, cur + step)
        out.append((cur, nxt))
        cur = nxt
    return out


def fetch_rates_range(symbol: str, tf: str, dt_from: datetime, dt_to: datetime) -> pd.DataFrame:
    ensure_symbol_selected(symbol)
    tf_enum = TF_MAP.get(tf)
    if tf_enum is None:
        raise ValueError(f"Unbekannter Timeframe: {tf}")

    rates = mt5.copy_rates_range(symbol, tf_enum, dt_from, dt_to)
    if rates is None:
        err = mt5.last_error()
        raise RuntimeError(f"copy_rates_range failed: {symbol} {tf} | {err}")

    df = pd.DataFrame(rates)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    keep = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    df = df[keep].copy()
    df["symbol"] = symbol
    df["timeframe"] = tf
    return df


def earliest_bar_time(symbol: str, tf: str) -> Optional[pd.Timestamp]:
    ensure_symbol_selected(symbol)
    tf_enum = TF_MAP[tf]
    now = datetime.now(timezone.utc)

    earliest: Optional[pd.Timestamp] = None
    for days in EARLIEST_SEARCH_HORIZONS_DAYS:
        dt_from = now - timedelta(days=int(days))
        rates = mt5.copy_rates_range(symbol, tf_enum, dt_from, now)
        if rates is None or len(rates) == 0:
            continue
        t0 = pd.to_datetime(rates[0]["time"], unit="s", utc=True)
        earliest = t0
    return earliest


def dump_full_history(symbol: str, tf: str, dt_from: datetime, dt_to: datetime) -> pd.DataFrame:
    parts = []
    for a, b in chunked_ranges(dt_from, dt_to, CHUNK_DAYS):
        df = fetch_rates_range(symbol, tf, a, b)
        if not df.empty:
            parts.append(df)
        time.sleep(SLEEP_BETWEEN_CALLS)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
    out = (
        out.dropna(subset=["time"])
           .drop_duplicates(subset=["time"])
           .sort_values("time")
           .reset_index(drop=True)
    )
    return out


def read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    if CSV_TIME_COL in df.columns:
        df[CSV_TIME_COL] = pd.to_datetime(df[CSV_TIME_COL], utc=True, errors="coerce")
        df = df.dropna(subset=[CSV_TIME_COL]).sort_values(CSV_TIME_COL)
    return df


def fetch_last_n_bars(symbol: str, tf: str, n: int) -> pd.DataFrame:
    ensure_symbol_selected(symbol)
    tf_enum = TF_MAP[tf]
    rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, int(n))
    if rates is None:
        err = mt5.last_error()
        raise RuntimeError(f"copy_rates_from_pos failed: {symbol} {tf} | {err}")
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    keep = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = 0
    df = df[keep].copy()
    df["symbol"] = symbol
    df["timeframe"] = tf
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def merge_and_dedup(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
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


# ============================================================
# Resample (MN1 -> Q/Y)
# ============================================================

def resample_ohlc_from_mn1(mn1_df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if mn1_df is None or mn1_df.empty:
        return pd.DataFrame()

    d = mn1_df.copy()
    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time").set_index("time")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "real_volume": "sum",
        "spread": "mean",
    }
    out = d.resample(rule).agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def write_qy_from_mn1(symbol: str, summary_sym: Dict[str, object]) -> None:
    mn1_path = ohlc_path(OUT_DIR, symbol, "MN1")
    if not mn1_path.exists():
        summary_sym["Q"] = {"status": "skipped", "reason": "MN1_missing"}
        summary_sym["Y"] = {"status": "skipped", "reason": "MN1_missing"}
        return

    mn1 = read_existing_csv(mn1_path)
    if mn1.empty:
        summary_sym["Q"] = {"status": "empty"}
        summary_sym["Y"] = {"status": "empty"}
        return

    for c in ["tick_volume", "real_volume", "spread"]:
        if c not in mn1.columns:
            mn1[c] = 0

    q = resample_ohlc_from_mn1(mn1, "Q")
    y = resample_ohlc_from_mn1(mn1, "Y")

    if not q.empty:
        q["symbol"] = symbol
        q["timeframe"] = "Q"
        qp = ohlc_path(OUT_DIR, symbol, "Q")
        atomic_write_df_csv(q, qp)
        summary_sym["Q"] = {
            "status": "ok",
            "rows": int(len(q)),
            "from_utc": str(pd.to_datetime(q["time"], utc=True).min()),
            "to_utc": str(pd.to_datetime(q["time"], utc=True).max()),
            "file": str(qp),
        }
    else:
        summary_sym["Q"] = {"status": "empty"}

    if not y.empty:
        y["symbol"] = symbol
        y["timeframe"] = "Y"
        yp = ohlc_path(OUT_DIR, symbol, "Y")
        atomic_write_df_csv(y, yp)
        summary_sym["Y"] = {
            "status": "ok",
            "rows": int(len(y)),
            "from_utc": str(pd.to_datetime(y["time"], utc=True).min()),
            "to_utc": str(pd.to_datetime(y["time"], utc=True).max()),
            "file": str(yp),
        }
    else:
        summary_sym["Y"] = {"status": "empty"}


# ============================================================
# Update logic
# ============================================================

def initial_full_dump(symbol: str, tf: str, now: datetime) -> pd.DataFrame:
    t0 = earliest_bar_time(symbol, tf)
    if t0 is None:
        return pd.DataFrame()
    dt_from = t0.to_pydatetime().replace(tzinfo=timezone.utc)
    return dump_full_history(symbol, tf, dt_from, now)


def incremental_update(symbol: str, tf: str, now: datetime) -> pd.DataFrame:
    path = ohlc_path(OUT_DIR, symbol, tf)
    existing = read_existing_csv(path)

    last_time = None
    if not existing.empty and "time" in existing.columns:
        last_time = pd.to_datetime(existing["time"].max(), utc=True)

    tail = fetch_last_n_bars(symbol, tf, SAFETY_LOOKBACK_BARS)

    if last_time is None:
        return merge_and_dedup(existing, tail)

    dt_from = (last_time.to_pydatetime().replace(tzinfo=timezone.utc) - timedelta(days=3))
    rng = fetch_rates_range(symbol, tf, dt_from, now)

    merged = merge_and_dedup(existing, rng)
    merged = merge_and_dedup(merged, tail)
    return merged


# ============================================================
# MAIN LOOP
# ============================================================

def main() -> None:
    ensure_dir(OUT_DIR)

    exe = find_terminal64(FIXED_MT5_DIR)
    print(f"[INFO] terminal64.exe = {exe}")

    account_id = connect_mt5(exe)

    summary: Dict[str, Dict[str, object]] = {sym: {"meta": {"account_id": account_id}} for sym in SYMBOLS}
    sum_path = OUT_DIR / "summary.json"

    try:
        # 0) Initialer Full Dump (einmal)
        now = datetime.now(timezone.utc)
        for sym in SYMBOLS:
            for tf in BASE_TIMEFRAMES:
                try:
                    p = ohlc_path(OUT_DIR, sym, tf)
                    if p.exists():
                        summary[sym][tf] = {"status": "exists", "file": str(p)}
                        continue

                    df = initial_full_dump(sym, tf, now)
                    if df.empty:
                        summary[sym][tf] = {"status": "empty"}
                        continue

                    atomic_write_df_csv(df, p)
                    summary[sym][tf] = {
                        "status": "ok",
                        "rows": int(len(df)),
                        "from_utc": str(df["time"].min()),
                        "to_utc": str(df["time"].max()),
                        "file": str(p),
                    }
                    print(f"[OK] init {sym} {tf}: rows={len(df)} -> {p}")
                except Exception as e:
                    summary[sym][tf] = {"status": "error", "error": str(e)}
                    print(f"[WARN] init {sym} {tf} failed: {e}")

            # Q/Y initial
            try:
                write_qy_from_mn1(sym, summary[sym])
            except Exception as e:
                summary[sym]["Q"] = {"status": "error", "error": str(e)}
                summary[sym]["Y"] = {"status": "error", "error": str(e)}

        atomic_write_json(summary, sum_path)
        print(f"[DONE] Initial summary -> {sum_path.resolve()}")

        # 1) Loop: inkrementell updaten + sauber speichern
        while True:
            loop_now = datetime.now(timezone.utc)

            for sym in SYMBOLS:
                for tf in BASE_TIMEFRAMES:
                    try:
                        p = ohlc_path(OUT_DIR, sym, tf)
                        df_new = incremental_update(sym, tf, loop_now)

                        if df_new.empty:
                            summary[sym][tf] = {"status": "empty"}
                            continue

                        atomic_write_df_csv(df_new, p)
                        summary[sym][tf] = {
                            "status": "ok",
                            "rows": int(len(df_new)),
                            "from_utc": str(df_new["time"].min()),
                            "to_utc": str(df_new["time"].max()),
                            "file": str(p),
                            "updated_at_utc": loop_now.isoformat(timespec="seconds"),
                        }
                    except Exception as e:
                        summary[sym][tf] = {
                            "status": "error",
                            "error": str(e),
                            "updated_at_utc": loop_now.isoformat(timespec="seconds"),
                        }
                        print(f"[WARN] update {sym} {tf} failed: {e}")

                try:
                    write_qy_from_mn1(sym, summary[sym])
                except Exception as e:
                    summary[sym]["Q"] = {"status": "error", "error": str(e)}
                    summary[sym]["Y"] = {"status": "error", "error": str(e)}
                    print(f"[WARN] update {sym} Q/Y failed: {e}")

            atomic_write_json(summary, sum_path)
            print(f"[LOOP] updated_at_utc={loop_now.isoformat(timespec='seconds')} -> summary.json")

            time.sleep(UPDATE_EVERY_SECONDS)

    except KeyboardInterrupt:
        print("[INFO] Stopping...")
        atomic_write_json(summary, sum_path)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

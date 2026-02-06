# -*- coding: utf-8 -*-
# Spread_Data_Management.py
#
# Zweck:
# - Startet fixes MT5-Terminal (optional portable)
# - Loggt ein
# - Loggt live Spreads (bid/ask) kontinuierlich für definierte Symbole
# - Speichert append-only CSV pro Symbol
#
# Speichert in deiner Struktur:
#   <PROJECT_ROOT>/Data/Spreads/<symbol>.csv
#
# pip install MetaTrader5

import os
import time
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import MetaTrader5 as mt5


# ============================================================
# PROJECT PATHS (deine Struktur)
# Data_Operations/Spread_Data_Management.py -> Root ist 2 Ebenen hoch
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "Data" / "Spreads"


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

POLL_SECONDS = 1.0  # sampling frequency (1s = viel Daten)

SYMBOLS: List[str] = [
    "AUDJPY", "AUDUSD", "EURGBP", "EURUSD", "GBPUSD", "GBPJPY",
    "NZDUSD", "US500.cash", "USDCAD", "USDCHF", "USDJPY", "USOIL.cash", "XAUUSD"
]


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
# IO helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace("/", "_") + ".csv"


def spread_path(out_dir: Path, symbol: str) -> Path:
    # Speichert in Data/Spreads/<symbol>.csv
    return out_dir / safe_symbol_filename(symbol)


def csv_append_row(path: Path, row: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# ============================================================
# Spread sampling
# ============================================================

def ensure_symbol_selected(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol nicht gefunden: {symbol}")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select failed: {symbol} / {mt5.last_error()}")


def sample_spread(symbol: str) -> Optional[Dict[str, Any]]:
    ensure_symbol_selected(symbol)

    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None

    bid = float(getattr(tick, "bid", 0.0))
    ask = float(getattr(tick, "ask", 0.0))
    if bid <= 0.0 or ask <= 0.0:
        return None

    point = float(getattr(info, "point", 0.0)) or 0.0
    digits = int(getattr(info, "digits", 0))
    trade_mode = int(getattr(info, "trade_mode", 0))

    spread = ask - bid
    spread_points = (spread / point) if point > 0 else None

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "time_utc": ts,
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "spread_points": float(spread_points) if spread_points is not None else "",
        "digits": digits,
        "trade_mode": trade_mode,
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(OUT_DIR)

    exe = find_terminal64(FIXED_MT5_DIR)
    print(f"[INFO] terminal64.exe = {exe}")

    _account_id = connect_mt5(exe)

    print(f"[INFO] Spread logging started: symbols={len(SYMBOLS)} poll={POLL_SECONDS}s")
    print(f"[INFO] Output: {OUT_DIR.resolve()}")

    try:
        while True:
            for sym in SYMBOLS:
                try:
                    row = sample_spread(sym)
                    if row is None:
                        continue
                    p = spread_path(OUT_DIR, sym)
                    csv_append_row(p, row)
                except Exception as e:
                    print(f"[WARN] {sym} sample failed: {e}")

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

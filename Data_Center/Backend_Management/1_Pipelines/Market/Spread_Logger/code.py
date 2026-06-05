# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/1_Pipelines/Market/Spread_Logger/code.py

Zweck:
- Startet fixes MT5-Terminal optional portable
- Loggt in MetaTrader 5 ein
- Loggt live Spreads Bid/Ask kontinuierlich für definierte Symbole
- Speichert inkrementell in SQLite pro Symbol
- Speichert UTC-Zeit und lokale Zeit
- Führt bestehende SQLite-Dateien automatisch auf das aktuelle Schema hoch

Output:
QUANT/Data_Center/Data/1_Pipeline/Market/spreads/<SYMBOL>.db

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: market_spread_mt5_logger
# script_name: Market Spread MT5 Logger
# owner: Leon Everts
# status: active
# layer: 1_Pipeline
# domain: Market
# asset_type: Pipeline
# purpose: Logs live bid/ask spreads from MetaTrader 5 into SQLite files per symbol.
# inputs:
#   - MetaTrader 5 terminal64.exe
#   - MT5 login/server/password from environment variables
#   - SYMBOLS list from this script
# outputs:
#   - Data_Center/Data/1_Pipeline/Market/spreads/<SYMBOL>.db
# upstream_data:
#   - MT5 symbol_info
#   - MT5 symbol_info_tick
# downstream_data:
#   - Feature Engineering
#   - Spread Analytics
#   - Execution Cost Analysis
#   - Dashboards
# dependencies:
#   - MetaTrader5
#   - sqlite3
#   - pathlib
# schedule: realtime
# version: v1.0.0
# last_reviewed: 2026-06-01
# business_criticality: high
# environment: desktop
# registry_group: pipeline
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# tags:
#   - mt5
#   - spread
#   - market_data
#   - sqlite
#   - realtime
#   - data_pipeline
# notes:
#   - Output root is Data_Center/Data/1_Pipeline/Market/spreads.
#   - Existing SQLite files are migrated automatically if columns are missing.
#   - MT5 credentials must be provided via environment variables.
# ============================================================
"""

from __future__ import annotations

import os
import time
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import MetaTrader5 as mt5


# ============================================================
# CODE REGISTRY - Runtime Metadata
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "market_spread_mt5_logger",
    "script_name": "Market Spread MT5 Logger",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "1_Pipeline",
    "domain": "Market",
    "asset_type": "Pipeline",
    "purpose": "Logs live bid/ask spreads from MetaTrader 5 into SQLite files per symbol.",
    "inputs": [
        "MetaTrader 5 terminal64.exe",
        "MT5 login/server/password from environment variables",
        "Configured SYMBOLS list",
    ],
    "outputs": [
        "Data_Center/Data/1_Pipeline/Market/spreads/<SYMBOL>.db",
    ],
    "upstream_data": [
        "MT5 symbol_info",
        "MT5 symbol_info_tick",
    ],
    "downstream_data": [
        "Feature Engineering",
        "Spread Analytics",
        "Execution Cost Analysis",
        "Dashboards",
    ],
    "dependencies": [
        "MetaTrader5",
        "sqlite3",
        "pathlib",
    ],
    "schedule": "realtime",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-01",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "pipeline",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "tags": [
        "mt5",
        "spread",
        "market_data",
        "sqlite",
        "realtime",
        "data_pipeline",
    ],
    "notes": [
        "Output root is Data_Center/Data/1_Pipeline/Market/spreads.",
        "Existing SQLite files are migrated automatically if columns are missing.",
        "MT5 credentials must be provided via environment variables.",
    ],
}


# ============================================================
# PROJECT PATHS
# ============================================================

def find_quant_root(start: Path) -> Path:
    """
    Erwartet QUANT-Root:

        QUANT/
            Dashboard/
            Data_Center/
                Backend_Management/
                Data/

    Erkennung:
    - Root enthält Dashboard/
    - Root enthält Data_Center/
    - Data_Center enthält Backend_Management/
    - Data_Center enthält Data/
    """
    cur = start.resolve()

    for p in [cur] + list(cur.parents):
        dashboard_dir = p / "Dashboard"
        data_center_dir = p / "Data_Center"
        backend_dir = data_center_dir / "Backend_Management"
        data_dir = data_center_dir / "Data"

        if (
            dashboard_dir.exists()
            and data_center_dir.exists()
            and backend_dir.exists()
            and data_dir.exists()
        ):
            return p

    raise RuntimeError(
        "QUANT-Root nicht gefunden. Erwartet Root mit "
        "'Dashboard', 'Data_Center', 'Data_Center/Data' und "
        "'Data_Center/Backend_Management'. "
        f"Start: {start}"
    )


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
BACKEND_MANAGEMENT_DIR = DATA_CENTER_DIR / "Backend_Management"

PIPELINES_DIR = BACKEND_MANAGEMENT_DIR / "1_Pipelines"
PIPELINE_DATA_DIR = DATA_DIR / "1_Pipeline"
MARKET_DATA_DIR = PIPELINE_DATA_DIR / "Market"

OUT_DIR = MARKET_DATA_DIR / "spreads"


# ============================================================
# KONFIG
# ============================================================

TERMINAL_EXE_PATH = Path(
    os.getenv(
        "MT5_TERMINAL_EXE",
        r"C:\Users\Leon\Desktop\Terminals\FTMO\MetaTrader 5 - Kopie - Kopie - Kopie (10) - Kopie - Kopie - Kopie - Kopie\terminal64.exe",
    )
)

MT5_LOGIN_RAW = os.getenv("MT5_LOGIN", "")
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")

PORTABLE = os.getenv("MT5_PORTABLE", "1") == "1"
START_TERMINAL = os.getenv("MT5_START_TERMINAL", "1") == "1"

INIT_TIMEOUT_MS = int(os.getenv("MT5_INIT_TIMEOUT_MS", "20000"))
STARTUP_WAIT_SECONDS = float(os.getenv("MT5_STARTUP_WAIT_SECONDS", "5"))
INIT_RETRIES = int(os.getenv("MT5_INIT_RETRIES", "8"))
RETRY_SLEEP_SECONDS = float(os.getenv("MT5_RETRY_SLEEP_SECONDS", "2"))

POLL_SECONDS = float(os.getenv("SPREAD_LOGGER_POLL_SECONDS", "1.0"))

SYMBOLS: List[str] = [
    "AUDJPY",
    "AUDUSD",
    "EURGBP",
    "EURUSD",
    "GBPUSD",
    "GBPJPY",
    "NZDUSD",
    "US500.cash",
    "USDCAD",
    "USDCHF",
    "USDJPY",
    "USOIL.cash",
    "XAUUSD",
]


# ============================================================
# MT5 TERMINAL / CONNECTION
# ============================================================

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
    if not exe.exists():
        raise RuntimeError(f"terminal64.exe nicht gefunden: {exe}")

    if not MT5_LOGIN_RAW:
        raise RuntimeError("MT5_LOGIN fehlt. Bitte ENV Variable setzen.")

    if not MT5_PASSWORD:
        raise RuntimeError("MT5_PASSWORD fehlt. Bitte ENV Variable setzen.")

    if not MT5_SERVER:
        raise RuntimeError("MT5_SERVER fehlt. Bitte ENV Variable setzen.")

    try:
        login = int(MT5_LOGIN_RAW)
    except ValueError as exc:
        raise RuntimeError(f"MT5_LOGIN ist keine gültige Zahl: {MT5_LOGIN_RAW}") from exc

    mt5.shutdown()

    if START_TERMINAL:
        print(f"[INFO] Starting MT5 terminal: {exe}")
        start_terminal(exe)
        time.sleep(STARTUP_WAIT_SECONDS)

    last_err = None

    for i in range(1, INIT_RETRIES + 1):
        if mt5.initialize(path=str(exe), portable=PORTABLE, timeout=INIT_TIMEOUT_MS):
            print("[OK] MT5 initialized")
            break

        last_err = mt5.last_error()
        print(f"[WARN] initialize failed ({i}/{INIT_RETRIES}): {last_err}")
        time.sleep(RETRY_SLEEP_SECONDS)
    else:
        raise RuntimeError(f"MT5 initialize failed: {last_err}")

    for i in range(1, INIT_RETRIES + 1):
        if mt5.login(login, password=MT5_PASSWORD, server=MT5_SERVER):
            print("[OK] MT5 login successful")
            break

        last_err = mt5.last_error()
        print(f"[WARN] login failed ({i}/{INIT_RETRIES}): {last_err}")
        time.sleep(RETRY_SLEEP_SECONDS)
    else:
        mt5.shutdown()
        raise RuntimeError(f"MT5 login failed: {last_err}")

    acc = mt5.account_info()
    if acc is None:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"account_info failed: {err}")

    print(f"[OK] Connected: {acc.login} | {acc.name} | {MT5_SERVER}")
    return int(acc.login)


# ============================================================
# IO HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symbol_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_") + ".db"


def spread_db_path(out_dir: Path, symbol: str) -> Path:
    return out_dir / safe_symbol_filename(symbol)


def table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    rows = cur.fetchall()
    return [str(row[1]) for row in rows]


def ensure_spreads_table_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS spreads (
            time_utc TEXT NOT NULL,
            time_local TEXT,
            symbol TEXT NOT NULL,
            bid REAL NOT NULL,
            ask REAL NOT NULL,
            spread REAL NOT NULL,
            spread_points REAL,
            digits INTEGER NOT NULL,
            trade_mode INTEGER NOT NULL,
            account_id INTEGER,
            source TEXT,
            written_at_utc TEXT,
            PRIMARY KEY (time_utc, symbol)
        )
        """
    )

    conn.commit()

    cols = table_columns(conn, "spreads")

    migrations = {
        "time_local": "ALTER TABLE spreads ADD COLUMN time_local TEXT",
        "spread_points": "ALTER TABLE spreads ADD COLUMN spread_points REAL",
        "account_id": "ALTER TABLE spreads ADD COLUMN account_id INTEGER",
        "source": "ALTER TABLE spreads ADD COLUMN source TEXT",
        "written_at_utc": "ALTER TABLE spreads ADD COLUMN written_at_utc TEXT",
    }

    for col, sql in migrations.items():
        if col not in cols:
            cur.execute(sql)
            conn.commit()
            print(f"[INFO] SQLite migration applied: added column '{col}'")

    cur.execute(
        """
        UPDATE spreads
        SET time_local = time_utc
        WHERE time_local IS NULL
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_spreads_time_utc
        ON spreads(time_utc)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_spreads_time_local
        ON spreads(time_local)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_spreads_symbol_time
        ON spreads(symbol, time_utc)
        """
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_spreads_account_time
        ON spreads(account_id, time_utc)
        """
    )

    conn.commit()


def init_symbol_db(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    conn = sqlite3.connect(path)
    ensure_spreads_table_schema(conn)
    return conn


def db_insert_spread_row(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR IGNORE INTO spreads (
            time_utc,
            time_local,
            symbol,
            bid,
            ask,
            spread,
            spread_points,
            digits,
            trade_mode,
            account_id,
            source,
            written_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(row["time_utc"]),
            str(row["time_local"]),
            str(row["symbol"]),
            float(row["bid"]),
            float(row["ask"]),
            float(row["spread"]),
            float(row["spread_points"]) if row["spread_points"] is not None else None,
            int(row["digits"]),
            int(row["trade_mode"]),
            int(row["account_id"]) if row["account_id"] is not None else None,
            str(row["source"]),
            str(row["written_at_utc"]),
        ),
    )

    conn.commit()


# ============================================================
# SPREAD SAMPLING
# ============================================================

def ensure_symbol_selected(symbol: str) -> None:
    info = mt5.symbol_info(symbol)

    if info is None:
        raise RuntimeError(f"Symbol nicht gefunden: {symbol}")

    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"symbol_select failed: {symbol} / {mt5.last_error()}")


def sample_spread(symbol: str, account_id: Optional[int]) -> Optional[Dict[str, Any]]:
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
    spread_points = spread / point if point > 0 else None

    utc_now = datetime.now(timezone.utc)
    local_now = datetime.now().astimezone()

    return {
        "time_utc": utc_now.isoformat(timespec="seconds"),
        "time_local": local_now.isoformat(timespec="seconds"),
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "spread": spread,
        "spread_points": float(spread_points) if spread_points is not None else None,
        "digits": digits,
        "trade_mode": trade_mode,
        "account_id": account_id,
        "source": "MetaTrader5.symbol_info_tick",
        "written_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(OUT_DIR)

    print(f"[INFO] SCRIPT_PATH              = {SCRIPT_PATH}")
    print(f"[INFO] QUANT_ROOT               = {QUANT_ROOT.resolve()}")
    print(f"[INFO] DATA_CENTER_DIR          = {DATA_CENTER_DIR.resolve()}")
    print(f"[INFO] DATA_DIR                 = {DATA_DIR.resolve()}")
    print(f"[INFO] BACKEND_MANAGEMENT_DIR   = {BACKEND_MANAGEMENT_DIR.resolve()}")
    print(f"[INFO] PIPELINES_DIR            = {PIPELINES_DIR.resolve()}")
    print(f"[INFO] PIPELINE_DATA_DIR        = {PIPELINE_DATA_DIR.resolve()}")
    print(f"[INFO] MARKET_DATA_DIR          = {MARKET_DATA_DIR.resolve()}")
    print(f"[INFO] OUT_DIR                  = {OUT_DIR.resolve()}")
    print(f"[INFO] TERMINAL_EXE             = {TERMINAL_EXE_PATH.resolve()}")
    print(f"[INFO] POLL_SECONDS             = {POLL_SECONDS}")
    print(f"[INFO] SYMBOLS                  = {len(SYMBOLS)}")

    if not TERMINAL_EXE_PATH.exists():
        raise RuntimeError(f"terminal64.exe nicht gefunden: {TERMINAL_EXE_PATH}")

    account_id = connect_mt5(TERMINAL_EXE_PATH)
    print(f"[INFO] ACCOUNT_ID               = {account_id}")

    db_conns: Dict[str, sqlite3.Connection] = {}

    try:
        for symbol in SYMBOLS:
            db_path = spread_db_path(OUT_DIR, symbol)
            db_conns[symbol] = init_symbol_db(db_path)
            print(f"[OK] DB ready                   = {db_path}")

        print(
            f"[OK] Spread logging started | "
            f"symbols={len(SYMBOLS)} | poll={POLL_SECONDS}s"
        )

        while True:
            loop_started = datetime.now(timezone.utc)

            for symbol in SYMBOLS:
                try:
                    row = sample_spread(symbol, account_id=account_id)
                    if row is None:
                        continue

                    db_insert_spread_row(db_conns[symbol], row)

                except Exception as exc:
                    print(f"[WARN] {symbol} sample failed: {exc}")

            print(
                f"[INFO] loop_written_at_utc={loop_started.isoformat(timespec='seconds')}"
            )

            time.sleep(float(POLL_SECONDS))

    except KeyboardInterrupt:
        print("[INFO] Stopping spread logger...")

    except Exception as exc:
        print(f"[ERROR] Fatal error: {exc}")
        raise

    finally:
        mt5.shutdown()

        for symbol, conn in db_conns.items():
            try:
                conn.close()
                print(f"[OK] DB closed: {symbol}")
            except Exception as exc:
                print(f"[WARN] DB close failed for {symbol}: {exc}")


if __name__ == "__main__":
    main()
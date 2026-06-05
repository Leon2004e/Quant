# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/1_Pipelines/Trades/Trade_Logger/FTMO_LIVE_540136824.py

Zweck:
- Startet fixes MT5-Terminal optional portable
- Verbindet sich mit MetaTrader 5 und loggt ein
- Baut aus MT5 DEAL-Historie ein Closed-Trades-Ledger pro position_id
- Speichert nur geschlossene Trades mit Exit-Deal
- Speichert zusätzlich den aktuellen Live-State offener Positionen
- Übernimmt Live-Extremwerte offener Positionen beim Schließen in closed_trades
- Nutzt SQLite mit UPSERT statt INSERT OR IGNORE
- Polling im Loop mit robuster Rebuild-Logik

Output:
QUANT/Data_Center/Data/1_Pipeline/Trades/live/account_<login>_<account_type>/closed_trades.db
QUANT/Data_Center/Data/1_Pipeline/Trades/live/account_<login>_<account_type>/open_positions_current.db

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: trade_logger_live_540136824
# script_name: Trade Logger LIVE 540136824
# owner: Leon Everts
# status: active
# layer: 1_Pipeline
# domain: Trades
# asset_type: Pipeline
# purpose: Logs MT5 closed trades and current open-position state into SQLite for account 540136824.
# inputs:
#   - MetaTrader 5 terminal64.exe
#   - MT5 account login/server/password from this script defaults or environment variables
#   - MT5 DEAL history via history_deals_get
#   - MT5 historical orders via history_orders_get
#   - MT5 open positions via positions_get
# outputs:
#   - Data_Center/Data/1_Pipeline/Trades/live/account_<login>_<account_type>/closed_trades.db
#   - Data_Center/Data/1_Pipeline/Trades/live/account_<login>_<account_type>/open_positions_current.db
# upstream_data:
#   - MetaTrader 5 DEAL history
#   - MetaTrader 5 historical orders
#   - MetaTrader 5 open positions
# downstream_data:
#   - Trade Analytics
#   - Portfolio Analytics
#   - Risk Dashboards
#   - Performance Attribution
#   - Code Registry Dashboard
# dependencies:
#   - MetaTrader5
#   - pandas
#   - numpy
#   - sqlite3
#   - pathlib
# schedule: realtime
# version: v1.0.0
# last_reviewed: 2026-06-01
# business_criticality: critical
# environment: desktop
# registry_group: trade_pipeline
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# tags:
#   - mt5
#   - trades
#   - closed_trades
#   - open_positions
#   - sqlite
#   - realtime
#   - ftmo
# notes:
#   - Credentials and terminal path are preserved from the original script as defaults.
#   - Environment variables can still override MT5_LOGIN, MT5_PASSWORD and MT5_SERVER.
#   - Output root migrated from Data_Center/Data/raw/trades/live to Data_Center/Data/1_Pipeline/Trades/live.
# ============================================================
"""

from __future__ import annotations

import os
import time
import sqlite3
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import MetaTrader5 as mt5
import numpy as np
import pandas as pd



# ============================================================
# CODE REGISTRY - Runtime Metadata
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "trade_logger_live_540136824",
    "script_name": "Trade Logger LIVE 540136824",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "1_Pipeline",
    "domain": "Trades",
    "asset_type": "Pipeline",
    "purpose": "Logs MT5 closed trades and current open-position state into SQLite for account 540136824.",
    "inputs": [
        "MetaTrader 5 terminal64.exe",
        "MT5 account login/server/password from this script defaults or environment variables",
        "MT5 DEAL history via history_deals_get",
        "MT5 historical orders via history_orders_get",
        "MT5 open positions via positions_get",
    ],
    "outputs": [
        "Data_Center/Data/1_Pipeline/Trades/live/account_<login>_<account_type>/closed_trades.db",
        "Data_Center/Data/1_Pipeline/Trades/live/account_<login>_<account_type>/open_positions_current.db",
    ],
    "upstream_data": [
        "MetaTrader 5 DEAL history",
        "MetaTrader 5 historical orders",
        "MetaTrader 5 open positions",
    ],
    "downstream_data": [
        "Trade Analytics",
        "Portfolio Analytics",
        "Risk Dashboards",
        "Performance Attribution",
        "Code Registry Dashboard",
    ],
    "dependencies": [
        "MetaTrader5",
        "pandas",
        "numpy",
        "sqlite3",
        "pathlib",
    ],
    "schedule": "realtime",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-01",
    "business_criticality": "critical",
    "environment": "desktop",
    "registry_group": "trade_pipeline",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "account_login_default": "540136824",
    "account_type_expected": "LIVE",
    "mt5_server_default": "FTMO-Server4",
    "terminal_exe_default": r"C:\Users\Leon\Desktop\Terminals\MetaTrader 5 - Kopie - Kopie - Kopie (8) - Kopie - Kopie - Kopie - Kopie\terminal64.exe",
    "tags": [
        "mt5",
        "trades",
        "closed_trades",
        "open_positions",
        "sqlite",
        "realtime",
        "ftmo",
    ],
    "notes": [
        "Credentials and terminal path are preserved from the original script as defaults.",
        "Environment variables can still override MT5_LOGIN, MT5_PASSWORD and MT5_SERVER.",
        "Output root migrated from Data_Center/Data/raw/trades/live to Data_Center/Data/1_Pipeline/Trades/live.",
    ],
}

# ============================================================
# CONFIG
# ============================================================

FIXED_TERMINAL_EXE = Path(
    r"C:\Users\Leon\Desktop\Terminals\MetaTrader 5 - Kopie - Kopie - Kopie (8) - Kopie - Kopie - Kopie - Kopie\terminal64.exe"
)

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "540136824"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "bXRS1Jl@")
MT5_SERVER = os.getenv("MT5_SERVER", "FTMO-Server4")

PORTABLE = True
START_TERMINAL = True

INIT_TIMEOUT_MS = 20000
STARTUP_WAIT_SECONDS = 5
INIT_RETRIES = 8
RETRY_SLEEP_SECONDS = 2

POLL_SECONDS = 60
POLL_OVERLAP_SECONDS = 300
INITIAL_LOOKBACK_DAYS = 365
POSITION_REBUILD_LOOKBACK_DAYS = 365

DEBUG = True
DEBUG_SHOW_LAST_ROWS = 10

DB_FILENAME = "closed_trades.db"
OPEN_DB_FILENAME = "open_positions_current.db"


# ============================================================
# ROOT / PATHS
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
TRADES_DATA_DIR = PIPELINE_DATA_DIR / "Trades"
OUTPUT_BASE = TRADES_DATA_DIR / "live"

# Backward-compatible alias for older log/debug names if needed.
FTMO_ROOT = QUANT_ROOT


# ============================================================
# GENERIC HELPERS
# ============================================================

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def log_debug(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_iso_from_unix(sec: int) -> str:
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).isoformat(timespec="seconds")


def symbol_point_size(symbol: str) -> float:
    s = str(symbol).upper()

    if s.startswith("XAU"):
        return 0.01

    if s.startswith("US500") or s.startswith("USOIL"):
        return 0.01

    if "JPY" in s:
        return 0.01

    return 0.0001


def clean_float_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    if not np.isfinite(v):
        return None

    return v


# ============================================================
# MT5 TERMINAL / CONNECTION
# ============================================================

def get_terminal_exe() -> Path:
    if not FIXED_TERMINAL_EXE.exists():
        raise RuntimeError(f"terminal64.exe nicht gefunden: {FIXED_TERMINAL_EXE}")
    return FIXED_TERMINAL_EXE


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


def get_account_type(server: str) -> str:
    s = str(server).strip().lower()
    if "demo" in s:
        return "DEMO"
    return "LIVE"


def connect_mt5(exe: Path) -> Dict[str, Any]:
    if not exe.exists():
        raise RuntimeError(f"terminal64.exe nicht gefunden: {exe}")

    if not MT5_PASSWORD:
        raise RuntimeError("MT5_PASSWORD fehlt. Bitte als Environment-Variable setzen.")

    mt5.shutdown()

    if START_TERMINAL:
        start_terminal(exe)
        time.sleep(STARTUP_WAIT_SECONDS)

    last_err = None

    for i in range(1, INIT_RETRIES + 1):
        if mt5.initialize(path=str(exe), portable=PORTABLE, timeout=INIT_TIMEOUT_MS):
            break
        last_err = mt5.last_error()
        log_warn(f"initialize failed ({i}/{INIT_RETRIES}): {last_err}")
        time.sleep(RETRY_SLEEP_SECONDS)
    else:
        raise RuntimeError(f"initialize failed: {last_err}")

    for i in range(1, INIT_RETRIES + 1):
        if mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            break
        last_err = mt5.last_error()
        log_warn(f"login failed ({i}/{INIT_RETRIES}): {last_err}")
        time.sleep(RETRY_SLEEP_SECONDS)
    else:
        mt5.shutdown()
        raise RuntimeError(f"login failed: {last_err}")

    acc = mt5.account_info()
    if acc is None:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"account_info failed: {err}")

    server = str(getattr(acc, "server", MT5_SERVER) or MT5_SERVER)
    account_type = get_account_type(server)

    print(f"[OK] Connected: {acc.login} | {acc.name} | {server} | {account_type}")
    return {
        "login": int(acc.login),
        "name": str(acc.name),
        "server": server,
        "account_type": account_type,
    }


# ============================================================
# SQLITE HELPERS
# ============================================================

def ensure_column_exists(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = {row[1] for row in cur.fetchall()}
    if column not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()


def init_db(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS closed_trades (
        account_id INTEGER NOT NULL,
        position_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,

        open_time_utc TEXT NOT NULL,
        close_time_utc TEXT NOT NULL,

        entry_price REAL,
        exit_price REAL,
        price_delta REAL,
        sl REAL,
        tp REAL,

        best_floating_pnl REAL,
        worst_floating_pnl REAL,
        best_price_seen_live REAL,
        worst_price_seen_live REAL,
        max_favorable_points_live REAL,
        max_adverse_points_live REAL,

        volume_in REAL NOT NULL,
        volume_out REAL NOT NULL,

        profit_sum REAL NOT NULL,
        swap_sum REAL NOT NULL,
        commission_sum REAL NOT NULL,
        net_sum REAL NOT NULL,

        magic INTEGER NOT NULL,
        comment_last TEXT,
        account_type TEXT,
        server TEXT,

        close_ticket INTEGER PRIMARY KEY
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_closed_trades_account_close_time
    ON closed_trades(account_id, close_time_utc)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_closed_trades_symbol_close_time
    ON closed_trades(symbol, close_time_utc)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_closed_trades_position_id
    ON closed_trades(position_id)
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS logger_state (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    conn.commit()

    ensure_column_exists(conn, "closed_trades", "account_type", "TEXT")
    ensure_column_exists(conn, "closed_trades", "server", "TEXT")
    ensure_column_exists(conn, "closed_trades", "sl", "REAL")
    ensure_column_exists(conn, "closed_trades", "tp", "REAL")
    ensure_column_exists(conn, "closed_trades", "best_floating_pnl", "REAL")
    ensure_column_exists(conn, "closed_trades", "worst_floating_pnl", "REAL")
    ensure_column_exists(conn, "closed_trades", "best_price_seen_live", "REAL")
    ensure_column_exists(conn, "closed_trades", "worst_price_seen_live", "REAL")
    ensure_column_exists(conn, "closed_trades", "max_favorable_points_live", "REAL")
    ensure_column_exists(conn, "closed_trades", "max_adverse_points_live", "REAL")

    return conn


def state_get(conn: sqlite3.Connection, key: str) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT value FROM logger_state WHERE key = ?", (key,))
    row = cur.fetchone()
    return str(row[0]) if row and row[0] is not None else None


def state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO logger_state(key, value)
    VALUES(?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value
    """, (key, value))
    conn.commit()


def db_count_rows(conn: sqlite3.Connection) -> int:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM closed_trades")
    row = cur.fetchone()
    return int(row[0]) if row else 0


def db_max_close_time(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute("SELECT MAX(close_time_utc) FROM closed_trades")
    row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return str(row[0])


def db_existing_close_tickets(conn: sqlite3.Connection, tickets: List[int]) -> Set[int]:
    if not tickets:
        return set()

    placeholders = ",".join(["?"] * len(tickets))
    cur = conn.cursor()
    cur.execute(
        f"SELECT close_ticket FROM closed_trades WHERE close_ticket IN ({placeholders})",
        [int(t) for t in tickets]
    )
    rows = cur.fetchall()
    return {int(r[0]) for r in rows}


def _row_float_or_none(r: Dict[str, Any], key: str) -> Optional[float]:
    if key not in r:
        return None

    v = r.get(key)
    if v == "":
        return None

    return clean_float_or_none(v)


def db_upsert_closed_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    payload = []
    for r in rows:
        payload.append((
            int(r["account_id"]),
            int(r["position_id"]),
            str(r["symbol"]),
            str(r["direction"]),
            str(r["open_time_utc"]),
            str(r["close_time_utc"]),
            _row_float_or_none(r, "entry_price"),
            _row_float_or_none(r, "exit_price"),
            _row_float_or_none(r, "price_delta"),
            _row_float_or_none(r, "sl"),
            _row_float_or_none(r, "tp"),
            _row_float_or_none(r, "best_floating_pnl"),
            _row_float_or_none(r, "worst_floating_pnl"),
            _row_float_or_none(r, "best_price_seen_live"),
            _row_float_or_none(r, "worst_price_seen_live"),
            _row_float_or_none(r, "max_favorable_points_live"),
            _row_float_or_none(r, "max_adverse_points_live"),
            float(r["volume_in"]),
            float(r["volume_out"]),
            float(r["profit_sum"]),
            float(r["swap_sum"]),
            float(r["commission_sum"]),
            float(r["net_sum"]),
            int(r["magic"]),
            str(r["comment_last"]),
            str(r["account_type"]),
            str(r["server"]),
            int(r["close_ticket"]),
        ))

    cur = conn.cursor()
    before = conn.total_changes

    cur.executemany("""
    INSERT INTO closed_trades (
        account_id,
        position_id,
        symbol,
        direction,
        open_time_utc,
        close_time_utc,
        entry_price,
        exit_price,
        price_delta,
        sl,
        tp,
        best_floating_pnl,
        worst_floating_pnl,
        best_price_seen_live,
        worst_price_seen_live,
        max_favorable_points_live,
        max_adverse_points_live,
        volume_in,
        volume_out,
        profit_sum,
        swap_sum,
        commission_sum,
        net_sum,
        magic,
        comment_last,
        account_type,
        server,
        close_ticket
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(close_ticket) DO UPDATE SET
        account_id=excluded.account_id,
        position_id=excluded.position_id,
        symbol=excluded.symbol,
        direction=excluded.direction,
        open_time_utc=excluded.open_time_utc,
        close_time_utc=excluded.close_time_utc,
        entry_price=excluded.entry_price,
        exit_price=excluded.exit_price,
        price_delta=excluded.price_delta,
        sl=excluded.sl,
        tp=excluded.tp,
        best_floating_pnl=excluded.best_floating_pnl,
        worst_floating_pnl=excluded.worst_floating_pnl,
        best_price_seen_live=excluded.best_price_seen_live,
        worst_price_seen_live=excluded.worst_price_seen_live,
        max_favorable_points_live=excluded.max_favorable_points_live,
        max_adverse_points_live=excluded.max_adverse_points_live,
        volume_in=excluded.volume_in,
        volume_out=excluded.volume_out,
        profit_sum=excluded.profit_sum,
        swap_sum=excluded.swap_sum,
        commission_sum=excluded.commission_sum,
        net_sum=excluded.net_sum,
        magic=excluded.magic,
        comment_last=excluded.comment_last,
        account_type=excluded.account_type,
        server=excluded.server
    """, payload)

    conn.commit()
    changed = conn.total_changes - before
    return int(changed)

# ============================================================
# OPEN POSITION CURRENT-STATE HELPERS
# ============================================================

def init_open_positions_db(path: Path) -> sqlite3.Connection:
    ensure_dir(path.parent)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS open_positions_current (
        account_id INTEGER NOT NULL,
        position_id INTEGER PRIMARY KEY,
        ticket INTEGER,
        symbol TEXT NOT NULL,
        direction TEXT NOT NULL,

        open_time_utc TEXT,
        first_seen_utc TEXT NOT NULL,
        last_seen_utc TEXT NOT NULL,

        volume REAL,
        entry_price REAL,
        current_price REAL,

        floating_pnl REAL,
        floating_pnl_pct REAL,
        best_floating_pnl REAL,
        worst_floating_pnl REAL,

        best_price_seen REAL,
        worst_price_seen REAL,
        max_favorable_points REAL,
        max_adverse_points REAL,

        sl REAL,
        tp REAL,

        magic INTEGER,
        comment TEXT,
        account_type TEXT,
        server TEXT
    )
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_open_positions_account_symbol
    ON open_positions_current(account_id, symbol)
    """)

    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_open_positions_last_seen
    ON open_positions_current(last_seen_utc)
    """)

    conn.commit()
    return conn


def read_existing_open_positions(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM open_positions_current")
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        d = dict(zip(cols, row))
        try:
            out[int(d["position_id"])] = d
        except Exception:
            continue

    return out


def read_open_positions_for_ids(
    conn: sqlite3.Connection,
    position_ids: List[int],
) -> Dict[int, Dict[str, Any]]:
    ids = sorted({int(x) for x in position_ids if int(x) > 0})
    if not ids:
        return {}

    placeholders = ",".join(["?"] * len(ids))
    cur = conn.cursor()
    cur.execute(
        f"SELECT * FROM open_positions_current WHERE position_id IN ({placeholders})",
        ids,
    )
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()

    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        d = dict(zip(cols, row))
        out[int(d["position_id"])] = d

    return out


def positions_to_df(positions) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame()

    rows = []
    for p in positions:
        position_id = int(getattr(p, "identifier", 0) or getattr(p, "ticket", 0))
        ticket = int(getattr(p, "ticket", 0))
        symbol = str(getattr(p, "symbol", ""))
        pos_type = int(getattr(p, "type", -1))
        direction = "BUY" if pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0) else (
            "SELL" if pos_type == getattr(mt5, "POSITION_TYPE_SELL", 1) else str(pos_type)
        )

        rows.append({
            "position_id": position_id,
            "ticket": ticket,
            "symbol": symbol,
            "direction": direction,
            "open_time_utc": ts_iso_from_unix(int(getattr(p, "time", 0))) if int(getattr(p, "time", 0) or 0) > 0 else "",
            "volume": float(getattr(p, "volume", 0.0) or 0.0),
            "entry_price": float(getattr(p, "price_open", np.nan)),
            "current_price": float(getattr(p, "price_current", np.nan)),
            "floating_pnl": float(getattr(p, "profit", 0.0) or 0.0),
            "sl": _clean_sl_tp_value(getattr(p, "sl", np.nan)),
            "tp": _clean_sl_tp_value(getattr(p, "tp", np.nan)),
            "magic": int(getattr(p, "magic", 0) or 0),
            "comment": str(getattr(p, "comment", "") or ""),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df[df["position_id"] > 0].sort_values(["symbol", "position_id"]).reset_index(drop=True)


def compute_live_price_extremes(
    direction: str,
    entry_price: float,
    current_price: float,
    previous: Optional[Dict[str, Any]],
    symbol: str,
) -> Dict[str, Optional[float]]:
    point = symbol_point_size(symbol)

    prev_best_price = clean_float_or_none(previous.get("best_price_seen")) if previous else None
    prev_worst_price = clean_float_or_none(previous.get("worst_price_seen")) if previous else None

    cur = clean_float_or_none(current_price)
    entry = clean_float_or_none(entry_price)

    if cur is None:
        return {
            "best_price_seen": prev_best_price,
            "worst_price_seen": prev_worst_price,
            "max_favorable_points": clean_float_or_none(previous.get("max_favorable_points")) if previous else None,
            "max_adverse_points": clean_float_or_none(previous.get("max_adverse_points")) if previous else None,
        }

    if prev_best_price is None:
        best_price_seen = cur
    else:
        best_price_seen = max(prev_best_price, cur) if direction == "BUY" else min(prev_best_price, cur)

    if prev_worst_price is None:
        worst_price_seen = cur
    else:
        worst_price_seen = min(prev_worst_price, cur) if direction == "BUY" else max(prev_worst_price, cur)

    if entry is None or point <= 0:
        mfe_points = None
        mae_points = None
    elif direction == "BUY":
        mfe_points = max(0.0, (best_price_seen - entry) / point)
        mae_points = max(0.0, (entry - worst_price_seen) / point)
    elif direction == "SELL":
        mfe_points = max(0.0, (entry - best_price_seen) / point)
        mae_points = max(0.0, (worst_price_seen - entry) / point)
    else:
        mfe_points = None
        mae_points = None

    return {
        "best_price_seen": best_price_seen,
        "worst_price_seen": worst_price_seen,
        "max_favorable_points": mfe_points,
        "max_adverse_points": mae_points,
    }


def build_open_position_rows(
    df: pd.DataFrame,
    existing: Dict[int, Dict[str, Any]],
    account_id: int,
    account_type: str,
    server: str,
) -> List[Dict[str, Any]]:
    now_iso = to_utc_now().isoformat(timespec="seconds")
    rows: List[Dict[str, Any]] = []

    if df.empty:
        return rows

    for _, r in df.iterrows():
        pos_id = int(r["position_id"])
        prev = existing.get(pos_id)

        floating = float(r["floating_pnl"])
        prev_best_pnl = clean_float_or_none(prev.get("best_floating_pnl")) if prev else None
        prev_worst_pnl = clean_float_or_none(prev.get("worst_floating_pnl")) if prev else None

        best_floating_pnl = floating if prev_best_pnl is None else max(prev_best_pnl, floating)
        worst_floating_pnl = floating if prev_worst_pnl is None else min(prev_worst_pnl, floating)

        entry_price = clean_float_or_none(r["entry_price"])
        volume = clean_float_or_none(r["volume"]) or 0.0
        floating_pnl_pct = None
        if entry_price is not None and entry_price > 0 and volume > 0:
            # Approximation; exact percent depends on contract size and account currency.
            floating_pnl_pct = None

        extremes = compute_live_price_extremes(
            direction=str(r["direction"]),
            entry_price=float(r["entry_price"]),
            current_price=float(r["current_price"]),
            previous=prev,
            symbol=str(r["symbol"]),
        )

        rows.append({
            "account_id": int(account_id),
            "position_id": pos_id,
            "ticket": int(r["ticket"]),
            "symbol": str(r["symbol"]),
            "direction": str(r["direction"]),
            "open_time_utc": str(r["open_time_utc"]),
            "first_seen_utc": str(prev.get("first_seen_utc")) if prev and prev.get("first_seen_utc") else now_iso,
            "last_seen_utc": now_iso,
            "volume": clean_float_or_none(r["volume"]),
            "entry_price": clean_float_or_none(r["entry_price"]),
            "current_price": clean_float_or_none(r["current_price"]),
            "floating_pnl": floating,
            "floating_pnl_pct": floating_pnl_pct,
            "best_floating_pnl": best_floating_pnl,
            "worst_floating_pnl": worst_floating_pnl,
            "best_price_seen": extremes["best_price_seen"],
            "worst_price_seen": extremes["worst_price_seen"],
            "max_favorable_points": extremes["max_favorable_points"],
            "max_adverse_points": extremes["max_adverse_points"],
            "sl": clean_float_or_none(r["sl"]),
            "tp": clean_float_or_none(r["tp"]),
            "magic": int(r["magic"]),
            "comment": str(r["comment"]),
            "account_type": str(account_type),
            "server": str(server),
        })

    return rows


def upsert_open_position_rows(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    payload = []
    for r in rows:
        payload.append((
            int(r["account_id"]),
            int(r["position_id"]),
            int(r["ticket"]),
            str(r["symbol"]),
            str(r["direction"]),
            str(r["open_time_utc"]),
            str(r["first_seen_utc"]),
            str(r["last_seen_utc"]),
            r["volume"],
            r["entry_price"],
            r["current_price"],
            r["floating_pnl"],
            r["floating_pnl_pct"],
            r["best_floating_pnl"],
            r["worst_floating_pnl"],
            r["best_price_seen"],
            r["worst_price_seen"],
            r["max_favorable_points"],
            r["max_adverse_points"],
            r["sl"],
            r["tp"],
            int(r["magic"]),
            str(r["comment"]),
            str(r["account_type"]),
            str(r["server"]),
        ))

    cur = conn.cursor()
    before = conn.total_changes

    cur.executemany("""
    INSERT INTO open_positions_current (
        account_id,
        position_id,
        ticket,
        symbol,
        direction,
        open_time_utc,
        first_seen_utc,
        last_seen_utc,
        volume,
        entry_price,
        current_price,
        floating_pnl,
        floating_pnl_pct,
        best_floating_pnl,
        worst_floating_pnl,
        best_price_seen,
        worst_price_seen,
        max_favorable_points,
        max_adverse_points,
        sl,
        tp,
        magic,
        comment,
        account_type,
        server
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(position_id) DO UPDATE SET
        account_id=excluded.account_id,
        ticket=excluded.ticket,
        symbol=excluded.symbol,
        direction=excluded.direction,
        open_time_utc=excluded.open_time_utc,
        first_seen_utc=excluded.first_seen_utc,
        last_seen_utc=excluded.last_seen_utc,
        volume=excluded.volume,
        entry_price=excluded.entry_price,
        current_price=excluded.current_price,
        floating_pnl=excluded.floating_pnl,
        floating_pnl_pct=excluded.floating_pnl_pct,
        best_floating_pnl=excluded.best_floating_pnl,
        worst_floating_pnl=excluded.worst_floating_pnl,
        best_price_seen=excluded.best_price_seen,
        worst_price_seen=excluded.worst_price_seen,
        max_favorable_points=excluded.max_favorable_points,
        max_adverse_points=excluded.max_adverse_points,
        sl=excluded.sl,
        tp=excluded.tp,
        magic=excluded.magic,
        comment=excluded.comment,
        account_type=excluded.account_type,
        server=excluded.server
    """, payload)

    conn.commit()
    return int(conn.total_changes - before)


def delete_open_positions_not_seen(conn: sqlite3.Connection, active_position_ids: List[int]) -> int:
    cur = conn.cursor()
    before = conn.total_changes

    if active_position_ids:
        placeholders = ",".join(["?"] * len(active_position_ids))
        cur.execute(
            f"DELETE FROM open_positions_current WHERE position_id NOT IN ({placeholders})",
            [int(x) for x in active_position_ids],
        )
    else:
        cur.execute("DELETE FROM open_positions_current")

    conn.commit()
    return int(conn.total_changes - before)


def fetch_open_positions_current() -> pd.DataFrame:
    positions = mt5.positions_get() or []
    log_debug(f"positions_get open_positions={len(positions)}")
    return positions_to_df(positions)




# ============================================================
# DEAL HELPERS
# ============================================================

def deals_to_df(deals) -> pd.DataFrame:
    if not deals:
        return pd.DataFrame()

    rows = []
    for d in deals:
        rows.append({
            "ticket": int(d.ticket),
            "order": int(d.order),
            "position_id": int(d.position_id),
            "symbol": str(d.symbol),
            "type": int(d.type),
            "entry": int(d.entry),
            "volume": float(d.volume),
            "price": float(d.price),
            "profit": float(d.profit),
            "swap": float(d.swap),
            "commission": float(d.commission),
            "magic": int(d.magic),
            "comment": str(d.comment),
            "time": int(d.time),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    return df.sort_values(["position_id", "time", "ticket"]).reset_index(drop=True)


def _vwap(prices: np.ndarray, vols: np.ndarray) -> float:
    v = np.asarray(vols, dtype=float)
    p = np.asarray(prices, dtype=float)
    s = float(np.sum(v))
    if not np.isfinite(s) or s <= 0:
        return float(np.nan)
    return float(np.sum(p * v) / s)


def _clean_sl_tp_value(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float(np.nan)

    if not np.isfinite(v) or v == 0.0:
        return float(np.nan)

    return v


def fetch_order_sl_tp(order_ticket: int) -> tuple[float, float]:
    """
    Holt SL/TP aus der historischen MT5-Order.

    Hinweis:
    - Deals selbst enthalten SL/TP meist nicht zuverlässig.
    - SL/TP liegen typischerweise auf Order-/Positions-Ebene.
    - Wenn der Broker/Terminal keine historischen Orderdaten liefert, kommt NaN zurück.
    """
    try:
        if int(order_ticket) <= 0:
            return float(np.nan), float(np.nan)

        orders = mt5.history_orders_get(ticket=int(order_ticket))
        if not orders:
            return float(np.nan), float(np.nan)

        o = orders[0]
        sl = _clean_sl_tp_value(getattr(o, "sl", np.nan))
        tp = _clean_sl_tp_value(getattr(o, "tp", np.nan))
        return sl, tp

    except Exception as e:
        log_debug(f"fetch_order_sl_tp failed order_ticket={order_ticket}: {e}")
        return float(np.nan), float(np.nan)


def infer_position_initial_sl_tp(g_in: pd.DataFrame) -> tuple[float, float]:
    """
    Versucht SL/TP der ersten Entry-Order zu lesen.
    Bei Scale-ins wird bewusst die erste Entry-Order verwendet, damit ein initiales Risk-Bild entsteht.
    """
    if g_in.empty or "order" not in g_in.columns:
        return float(np.nan), float(np.nan)

    entry_orders = [int(x) for x in g_in["order"].dropna().tolist() if int(x) > 0]
    if not entry_orders:
        return float(np.nan), float(np.nan)

    for order_ticket in entry_orders:
        sl, tp = fetch_order_sl_tp(order_ticket)
        if pd.notna(sl) or pd.notna(tp):
            return sl, tp

    return float(np.nan), float(np.nan)


def build_closed_trade_from_position_deals(
    g: pd.DataFrame,
    account_id: int,
) -> Optional[Dict[str, Any]]:
    if g.empty:
        return None

    exit_entries = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_OUT_BY", 2),
    }
    in_entries = {getattr(mt5, "DEAL_ENTRY_IN", 0)}

    deal_buy = getattr(mt5, "DEAL_TYPE_BUY", 0)
    deal_sell = getattr(mt5, "DEAL_TYPE_SELL", 1)

    g = g.sort_values(["time", "ticket"]).reset_index(drop=True)

    g_in = g[g["entry"].isin(in_entries)]
    g_out = g[g["entry"].isin(exit_entries)]

    if g_out.empty:
        return None
    if g_in.empty:
        return None

    pos_id = int(g["position_id"].iloc[0])
    symbol = str(g["symbol"].iloc[-1])
    magic = int(g["magic"].iloc[-1]) if "magic" in g.columns else 0

    first_type = int(g_in["type"].iloc[0])
    direction = "BUY" if first_type == deal_buy else ("SELL" if first_type == deal_sell else str(first_type))

    open_time = int(g_in["time"].min())
    close_time = int(g_out["time"].max())
    close_ticket = int(g_out.sort_values(["time", "ticket"]).iloc[-1]["ticket"])

    entry_price = _vwap(g_in["price"].to_numpy(), g_in["volume"].to_numpy())
    exit_price = _vwap(g_out["price"].to_numpy(), g_out["volume"].to_numpy())

    vol_in = float(g_in["volume"].sum())
    vol_out = float(g_out["volume"].sum())

    profit = float(g["profit"].sum())
    swap = float(g["swap"].sum())
    commission = float(g["commission"].sum())
    net = profit + swap + commission

    price_delta = (
        float(exit_price - entry_price)
        if (np.isfinite(entry_price) and np.isfinite(exit_price))
        else np.nan
    )

    sl, tp = infer_position_initial_sl_tp(g_in)

    return {
        "account_id": int(account_id),
        "position_id": int(pos_id),
        "symbol": symbol,
        "direction": direction,
        "open_time_utc": ts_iso_from_unix(open_time),
        "close_time_utc": ts_iso_from_unix(close_time),
        "entry_price": float(entry_price) if np.isfinite(entry_price) else "",
        "exit_price": float(exit_price) if np.isfinite(exit_price) else "",
        "price_delta": float(price_delta) if np.isfinite(price_delta) else "",
        "sl": float(sl) if pd.notna(sl) and np.isfinite(sl) else "",
        "tp": float(tp) if pd.notna(tp) and np.isfinite(tp) else "",
        "volume_in": vol_in,
        "volume_out": vol_out,
        "profit_sum": profit,
        "swap_sum": swap,
        "commission_sum": commission,
        "net_sum": net,
        "magic": magic,
        "comment_last": str(g["comment"].iloc[-1]) if "comment" in g.columns else "",
        "close_ticket": int(close_ticket),
    }


def build_closed_trades_from_deals(df: pd.DataFrame, account_id: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for _, g in df.groupby("position_id", sort=False):
        row = build_closed_trade_from_position_deals(g, account_id)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["close_time_utc", "position_id"]).reset_index(drop=True)


def extract_recent_exit_position_ids(df: pd.DataFrame) -> List[int]:
    if df.empty:
        return []

    exit_entries = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_OUT_BY", 2),
    }
    g_out = df[df["entry"].isin(exit_entries)]
    if g_out.empty:
        return []

    ids = sorted({int(x) for x in g_out["position_id"].dropna().tolist() if int(x) > 0})
    return ids


def fetch_deals_range(dt_from: datetime, dt_to: datetime) -> pd.DataFrame:
    deals = mt5.history_deals_get(dt_from, dt_to) or []
    log_debug(f"history_deals_get dt_from={dt_from.isoformat()} dt_to={dt_to.isoformat()} deals={len(deals)}")
    return deals_to_df(deals)


# ============================================================
# LOGGER
# ============================================================

class MT5ClosedTradeLogger:
    def __init__(self, account_id: int, account_type: str, server: str):
        self.account_id = int(account_id)
        self.account_type = str(account_type).upper().strip()
        self.server = str(server)

        self.dir = OUTPUT_BASE / f"account_{self.account_id}_{self.account_type}"
        ensure_dir(self.dir)

        self.db_path = self.dir / DB_FILENAME
        self.conn = init_db(self.db_path)

        self.open_db_path = self.dir / OPEN_DB_FILENAME
        self.open_conn = init_open_positions_db(self.open_db_path)
        self.closed_live_extremes: Dict[int, Dict[str, Any]] = {}

        state_last_poll = state_get(self.conn, "last_poll_utc")
        fallback_start = to_utc_now() - timedelta(seconds=POLL_OVERLAP_SECONDS)

        if state_last_poll:
            try:
                self.last_poll_dt = datetime.fromisoformat(state_last_poll)
                if self.last_poll_dt.tzinfo is None:
                    self.last_poll_dt = self.last_poll_dt.replace(tzinfo=timezone.utc)
                else:
                    self.last_poll_dt = self.last_poll_dt.astimezone(timezone.utc)
            except Exception:
                self.last_poll_dt = fallback_start
        else:
            self.last_poll_dt = fallback_start

    def persist_state(self) -> None:
        state_set(self.conn, "last_poll_utc", self.last_poll_dt.isoformat())

    def update_open_positions_current(self) -> int:
        existing = read_existing_open_positions(self.open_conn)
        df_open = fetch_open_positions_current()
        active_position_ids = sorted({int(x) for x in df_open["position_id"].tolist()}) if not df_open.empty else []

        previously_open_ids = sorted(existing.keys())
        closing_candidate_ids = sorted(set(previously_open_ids) - set(active_position_ids))
        self.closed_live_extremes = read_open_positions_for_ids(self.open_conn, closing_candidate_ids)

        rows = build_open_position_rows(
            df=df_open,
            existing=existing,
            account_id=self.account_id,
            account_type=self.account_type,
            server=self.server,
        )

        changed_upsert = upsert_open_position_rows(self.open_conn, rows)
        changed_delete = delete_open_positions_not_seen(self.open_conn, active_position_ids)

        log_debug(
            f"open_positions_current active={len(active_position_ids)} "
            f"| upsert_changed={changed_upsert} | deleted_closed={changed_delete} "
            f"| db={self.open_db_path}"
        )

        return int(changed_upsert + changed_delete)

    def _attach_live_extremes_to_closed_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows or not self.closed_live_extremes:
            return

        for row in rows:
            pos_id = int(row["position_id"])
            live = self.closed_live_extremes.get(pos_id)
            if not live:
                continue

            mapping = {
                "best_floating_pnl": "best_floating_pnl",
                "worst_floating_pnl": "worst_floating_pnl",
                "best_price_seen_live": "best_price_seen",
                "worst_price_seen_live": "worst_price_seen",
                "max_favorable_points_live": "max_favorable_points",
                "max_adverse_points_live": "max_adverse_points",
            }

            for target_key, live_key in mapping.items():
                value = clean_float_or_none(live.get(live_key))
                row[target_key] = value if value is not None else ""

    def backfill(self, lookback_days: int) -> None:
        dt_to = to_utc_now()
        dt_from = dt_to - timedelta(days=int(lookback_days))

        df = fetch_deals_range(dt_from, dt_to)
        closed = build_closed_trades_from_deals(df, self.account_id)

        log_debug(f"backfill closed_trades_found={len(closed)}")
        if not closed.empty:
            cols = [c for c in ["position_id", "symbol", "open_time_utc", "close_time_utc", "sl", "tp", "close_ticket"] if c in closed.columns]
            log_debug(str(closed[cols].tail(DEBUG_SHOW_LAST_ROWS)))

        self._append_new_closed(closed)

        self.last_poll_dt = dt_to - timedelta(seconds=POLL_OVERLAP_SECONDS)
        self.persist_state()

    def poll_once(self) -> int:
        open_changed = self.update_open_positions_current()

        dt_to = to_utc_now()
        dt_from = self.last_poll_dt - timedelta(seconds=POLL_OVERLAP_SECONDS)

        df_recent = fetch_deals_range(dt_from, dt_to)
        recent_exit_pos_ids = extract_recent_exit_position_ids(df_recent)

        log_debug(f"recent_exit_position_ids={len(recent_exit_pos_ids)}")
        if recent_exit_pos_ids:
            log_debug(f"recent_exit_position_ids_last={recent_exit_pos_ids[-DEBUG_SHOW_LAST_ROWS:]}")

        rebuilt_rows: List[Dict[str, Any]] = []

        if recent_exit_pos_ids:
            hist_from = dt_to - timedelta(days=POSITION_REBUILD_LOOKBACK_DAYS)
            df_hist = fetch_deals_range(hist_from, dt_to)

            if not df_hist.empty:
                df_hist = df_hist[df_hist["position_id"].isin(recent_exit_pos_ids)].copy()
                log_debug(f"rebuild_history_rows_for_exit_positions={len(df_hist)}")

                for _, g in df_hist.groupby("position_id", sort=False):
                    row = build_closed_trade_from_position_deals(g, self.account_id)
                    if row is not None:
                        rebuilt_rows.append(row)

        rebuilt_df = pd.DataFrame(rebuilt_rows)
        log_debug(f"rebuilt_closed_trades={len(rebuilt_df)}")
        if not rebuilt_df.empty:
            cols = [c for c in ["position_id", "symbol", "open_time_utc", "close_time_utc", "sl", "tp", "close_ticket"] if c in rebuilt_df.columns]
            log_debug(str(rebuilt_df[cols].tail(DEBUG_SHOW_LAST_ROWS)))

        changed = self._append_new_closed(rebuilt_df)

        self.last_poll_dt = dt_to
        self.persist_state()
        return int(changed + open_changed)

    def _append_new_closed(self, closed_df: pd.DataFrame) -> int:
        if closed_df is None or closed_df.empty:
            log_debug("rows_to_upsert=0")
            return 0

        rows: List[Dict[str, Any]] = []
        for _, r in closed_df.iterrows():
            row = {k: r[k] for k in closed_df.columns}
            row["account_type"] = self.account_type
            row["server"] = self.server
            rows.append(row)

        self._attach_live_extremes_to_closed_rows(rows)

        log_debug(f"rows_to_upsert={len(rows)}")

        tickets = [int(r["close_ticket"]) for r in rows]
        existing_before = db_existing_close_tickets(self.conn, tickets)
        log_debug(f"existing_close_tickets_before={len(existing_before)}")

        changed = db_upsert_closed_rows(self.conn, rows)

        if changed > 0:
            print(f"[DB][{self.account_type}][{self.server}] changed={changed} -> {self.db_path}")
        else:
            log_debug("upsert produced no db changes")

        return changed

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

        try:
            self.open_conn.close()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    exe = get_terminal_exe()
    preview_account_type = get_account_type(MT5_SERVER)

    log_info(f"terminal64.exe = {exe}")
    log_info(f"SCRIPT_PATH = {SCRIPT_PATH}")
    log_info(f"QUANT_ROOT = {QUANT_ROOT.resolve()}")
    log_info(f"DATA_DIR = {DATA_DIR.resolve()}")
    log_info(f"BACKEND_MANAGEMENT_DIR = {BACKEND_MANAGEMENT_DIR.resolve()}")
    log_info(f"PIPELINES_DIR = {PIPELINES_DIR.resolve()}")
    log_info(f"PIPELINE_DATA_DIR = {PIPELINE_DATA_DIR.resolve()}")
    log_info(f"TRADES_DATA_DIR = {TRADES_DATA_DIR.resolve()}")
    log_info(
        "OUTPUT_DIR = "
        + str((OUTPUT_BASE / f"account_{MT5_LOGIN}_{preview_account_type}").resolve())
    )

    logger: Optional[MT5ClosedTradeLogger] = None

    try:
        acc_info = connect_mt5(exe)
        account_id = int(acc_info["login"])
        account_type = str(acc_info["account_type"])
        server = str(acc_info["server"])

        logger = MT5ClosedTradeLogger(
            account_id=account_id,
            account_type=account_type,
            server=server,
        )

        try:
            logger.backfill(INITIAL_LOOKBACK_DAYS)
            print(
                f"[OK] Backfill done. Folder: {logger.dir.resolve()} | "
                f"account_type={account_type} | server={server}"
            )
            print(f"[OK] DB rows: {db_count_rows(logger.conn)}")
            print(f"[OK] DB max close_time_utc: {db_max_close_time(logger.conn)}")
            print(f"[OK] Open positions DB: {logger.open_db_path}")
        except Exception as e:
            log_warn(f"Backfill failed: {e}")

        log_info(
            f"Polling every {POLL_SECONDS}s. "
            f"Account={account_id} | Type={account_type} | Server={server}. Stop with CTRL+C."
        )

        while True:
            loop_ts = to_utc_now().isoformat(timespec="seconds")
            try:
                changed = logger.poll_once()
                max_close = db_max_close_time(logger.conn)

                if changed == 0:
                    print(
                        f"[LOOP] {loop_ts} | no db changes | "
                        f"account_type={account_type} | server={server} | "
                        f"max_close_time_utc={max_close}"
                    )
                else:
                    print(
                        f"[LOOP] {loop_ts} | db_changed={changed} | "
                        f"account_type={account_type} | server={server} | "
                        f"max_close_time_utc={max_close}"
                    )
            except Exception as e:
                log_warn(f"poll failed: {e}")

            time.sleep(float(POLL_SECONDS))

    except KeyboardInterrupt:
        log_info("Stopping...")
    finally:
        mt5.shutdown()
        if logger is not None:
            logger.close()


if __name__ == "__main__":
    main()
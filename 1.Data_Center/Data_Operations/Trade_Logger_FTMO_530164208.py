# -*- coding: utf-8 -*-
"""
1_Data_Center/Data_Operations/Trade_Logger_FTMO_530164208.py

Zweck:
- Startet ein fixes MT5-Terminal (optional portable)
- Verbindet sich, loggt ein
- Baut aus MT5 DEAL-Historie ein "Closed Trades"-Ledger (nur geschlossene Trades!)
- Speichert inkrementell in:
    <ROOT>/1_Data_Center/Data/Strategy_Data/Strategy_Live_Performance/account_<login>/closed_trades.csv
- Polling im Loop (default alle 60 Sekunden)
- Dedup über close_ticket (Ticket des letzten Exit-Deals pro Position)

Wichtig:
- Es wird NUR gespeichert, wenn eine Position geschlossen ist (Exit-Deal vorhanden).
- Offene Positionen werden NICHT als Trades gespeichert.

Install:
  pip install MetaTrader5 pandas numpy

Start:
  python 1_Data_Center/Data_Operations/Trade_Logger_FTMO_530164208.py

Stop:
  STRG+C
"""

from __future__ import annotations

import os
import time
import csv
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

FIXED_MT5_DIR = Path(
    r"C:\Users\Leon\Desktop\MetaTrader 5 - Kopie - Kopie - Kopie (19) - Kopie - Kopie - Kopie - Kopie - Kopie - Kopie - Kopie"
)

MT5_LOGIN = int(os.getenv("MT5_LOGIN", "530164208"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "9G8z5i!lanq7")  # besser: ENV setzen
MT5_SERVER = os.getenv("MT5_SERVER", "FTMO-Server3")

PORTABLE = True
START_TERMINAL = True

INIT_TIMEOUT_MS = 20000
STARTUP_WAIT_SECONDS = 5
INIT_RETRIES = 8
RETRY_SLEEP_SECONDS = 2

# Polling
POLL_SECONDS = 60                 # 1 Minute
POLL_OVERLAP_SECONDS = 300        # 5 Minuten Overlap
INITIAL_LOOKBACK_DAYS = 365       # einmaliger Backfill beim Start

# Output: flach, ohne Execution/MT5
# <ROOT>/1_Data_Center/Data/Strategy_Data/Strategy_Live_Performance/account_<login>/closed_trades.csv
OUTPUT_SUBDIR = Path("1_Data_Center") / "Data" / "Strategy_Data" / "Strategy_Live_Performance"
CLOSED_TRADES_FILENAME = "closed_trades.csv"


# ============================================================
# ROOT
# ============================================================

def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "1_Data_Center").exists():
            return p
    return start.resolve().parents[1]


ROOT = find_project_root(Path(__file__))


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
# CSV helpers
# ============================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def csv_read_seen_close_tickets(path: Path) -> Set[int]:
    """
    Dedup über close_ticket (Exit-Deal-Ticket).
    """
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["close_ticket"])
        return set(int(x) for x in df["close_ticket"].dropna().astype(int).tolist())
    except Exception:
        return set()


def csv_append_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    file_exists = path.exists()
    fieldnames = list(rows[0].keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)


def ts_iso_from_unix(sec: int) -> str:
    return datetime.fromtimestamp(int(sec), tz=timezone.utc).isoformat(timespec="seconds")


# ============================================================
# Deal -> Closed Trades Ledger
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


def build_closed_trades_from_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: deals dataframe über ein Zeitfenster (kann IN+OUT enthalten)
    Output: 1 Zeile pro geschlossener position_id, inkl. close_ticket.
    """
    if df.empty:
        return pd.DataFrame()

    exit_entries = {
        getattr(mt5, "DEAL_ENTRY_OUT", 1),
        getattr(mt5, "DEAL_ENTRY_OUT_BY", 2),
    }
    in_entries = {getattr(mt5, "DEAL_ENTRY_IN", 0)}

    deal_buy = getattr(mt5, "DEAL_TYPE_BUY", 0)
    deal_sell = getattr(mt5, "DEAL_TYPE_SELL", 1)

    closed_rows: List[Dict[str, Any]] = []

    for pos_id, g in df.groupby("position_id", sort=False):
        g = g.sort_values(["time", "ticket"]).reset_index(drop=True)

        g_in = g[g["entry"].isin(in_entries)]
        g_out = g[g["entry"].isin(exit_entries)]

        # nur wenn Exit existiert => closed
        if g_out.empty:
            continue
        # wenn der IN-Deal außerhalb des Fensters liegt, skip (Backfill deckt das i.d.R. ab)
        if g_in.empty:
            continue

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

        price_delta = float(exit_price - entry_price) if (np.isfinite(entry_price) and np.isfinite(exit_price)) else np.nan

        closed_rows.append({
            "account_id": int(MT5_LOGIN),
            "position_id": int(pos_id),
            "symbol": symbol,
            "direction": direction,

            "open_time_utc": ts_iso_from_unix(open_time),
            "close_time_utc": ts_iso_from_unix(close_time),

            "entry_price": float(entry_price) if np.isfinite(entry_price) else "",
            "exit_price": float(exit_price) if np.isfinite(exit_price) else "",
            "price_delta": float(price_delta) if np.isfinite(price_delta) else "",

            "volume_in": vol_in,
            "volume_out": vol_out,

            "profit_sum": profit,
            "swap_sum": swap,
            "commission_sum": commission,
            "net_sum": net,

            "magic": magic,
            "comment_last": str(g["comment"].iloc[-1]) if "comment" in g.columns else "",

            # Dedup key:
            "close_ticket": int(close_ticket),
        })

    out = pd.DataFrame(closed_rows)
    if out.empty:
        return out
    return out.sort_values(["close_time_utc", "position_id"]).reset_index(drop=True)


# ============================================================
# Logger
# ============================================================

class MT5ClosedTradeLogger:
    """
    Speichert nur geschlossene Trades als Ledger (pro Position).
    """
    def __init__(self, root: Path, account_id: int):
        self.account_id = int(account_id)
        self.dir = root / OUTPUT_SUBDIR / f"account_{self.account_id}"
        ensure_dir(self.dir)

        self.closed_path = self.dir / CLOSED_TRADES_FILENAME
        self.seen_close_tickets: Set[int] = csv_read_seen_close_tickets(self.closed_path)

        self.last_poll_dt: datetime = datetime.now(timezone.utc) - timedelta(seconds=POLL_OVERLAP_SECONDS)

    def backfill(self, lookback_days: int) -> None:
        dt_to = datetime.now(timezone.utc)
        dt_from = dt_to - timedelta(days=int(lookback_days))

        deals = mt5.history_deals_get(dt_from, dt_to) or []
        df = deals_to_df(deals)
        closed = build_closed_trades_from_deals(df)
        self._append_new_closed(closed)

        self.last_poll_dt = dt_to - timedelta(seconds=POLL_OVERLAP_SECONDS)

    def poll_once(self) -> None:
        dt_to = datetime.now(timezone.utc)
        dt_from = self.last_poll_dt - timedelta(seconds=POLL_OVERLAP_SECONDS)

        deals = mt5.history_deals_get(dt_from, dt_to) or []
        df = deals_to_df(deals)
        closed = build_closed_trades_from_deals(df)
        self._append_new_closed(closed)

        self.last_poll_dt = dt_to

    def _append_new_closed(self, closed_df: pd.DataFrame) -> None:
        if closed_df is None or closed_df.empty:
            return

        rows: List[Dict[str, Any]] = []
        for _, r in closed_df.iterrows():
            ct = int(r["close_ticket"])
            if ct in self.seen_close_tickets:
                continue
            self.seen_close_tickets.add(ct)
            rows.append({k: r[k] for k in closed_df.columns})

        if rows:
            csv_append_rows(self.closed_path, rows)
            print(f"[CLOSED] +{len(rows)} -> {self.closed_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    exe = find_terminal64(FIXED_MT5_DIR)
    print(f"[INFO] terminal64.exe = {exe}")
    print(f"[INFO] ROOT = {ROOT.resolve()}")

    account_id = connect_mt5(exe)

    logger = MT5ClosedTradeLogger(ROOT, account_id)

    # einmalig: Backfill (dedup verhindert Duplikate)
    try:
        logger.backfill(INITIAL_LOOKBACK_DAYS)
        print(f"[OK] Backfill done. Folder: {logger.dir.resolve()}")
    except Exception as e:
        print(f"[WARN] Backfill failed: {e}")

    print(f"[INFO] Polling every {POLL_SECONDS}s. Stop with CTRL+C.")
    try:
        while True:
            try:
                logger.poll_once()
            except Exception as e:
                print(f"[WARN] poll failed: {e}")
            time.sleep(float(POLL_SECONDS))
    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()

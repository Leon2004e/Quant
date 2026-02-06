# -*- coding: utf-8 -*-
"""
1_Data_Center/Data_Operations/live_performance_by_strategy.py

Zweck:
- Liest pro Account closed_trades.csv (nur geschlossene Trades)
- Ordnet Trades in Strategy-Buckets: <strategy_id>_<symbol>_<magic>_<side>
- Side/Strategy-ID wird primär aus Kommentar extrahiert ("Strategy X.Y.Z"),
  aber FALLBACK über Backtest-Dateinamen (Strategy_Backtest_Performance):
    listOfTrades_<strategy_id>_<symbol>_<magic>_<side>.csv
- Exportiert pro Bucket:
    trades.csv
    kpis.csv
    weekly_performance.csv
    monthly_performance.csv
- Loop optional.

Erwartete Struktur (Input):
1_Data_Center/Data/Strategy_Data/Strategy_Live_Performance/
  account_<id>/
    closed_trades.csv

Backtest-Registry Quelle:
1_Data_Center/Data/Strategy_Data/Strategy_Backtest_Performance/
  listOfTrades_<strategy_id>_<symbol>_<magic>_<side>.csv
"""

from __future__ import annotations

import argparse
import os
import re
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ----------------------------
# ROOT finder
# ----------------------------
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "1_Data_Center").exists():
            return p
    return start.resolve().parents[1]


THIS_FILE = Path(__file__).resolve()
ROOT = find_project_root(THIS_FILE)

DATA_DIR = ROOT / "1_Data_Center" / "Data"
LIVE_ROOT_DEFAULT = DATA_DIR / "Strategy_Data" / "Strategy_Live_Performance"
BACKTEST_DIR_DEFAULT = DATA_DIR / "Strategy_Data" / "Strategy_Backtest_Performance"


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Parsing helpers
# ----------------------------
_STRAT_RE = re.compile(r"Strategy\s+([0-9]+(?:\.[0-9]+)*)", re.IGNORECASE)


def extract_strategy_from_comment(c: str) -> Optional[str]:
    m = _STRAT_RE.search(str(c))
    return m.group(1) if m else None


def normalize_side(x: object) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("BUY", "B", "LONG", "L"):
        return "BUY"
    if s in ("SELL", "S", "SHORT", "SH", "SR"):
        return "SELL"
    if "BUY" in s and "SELL" not in s:
        return "BUY"
    if "SELL" in s and "BUY" not in s:
        return "SELL"
    return None


def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:200] if len(s) > 200 else s


# ----------------------------
# Backtest registry builder
# ----------------------------
_BACKTEST_FILE_RE = re.compile(
    r"^listOfTrades_(?P<sid>[0-9]+(?:\.[0-9]+)*)_(?P<sym>[^_]+)_(?P<magic>[0-9]+)_(?P<side>BUY|SELL)\.csv$",
    re.IGNORECASE
)


def build_magic_registry(backtest_dir: Path) -> pd.DataFrame:
    """
    Build mapping from Backtest Performance files.
    Returns DataFrame columns:
      magic (int), strategy_id (str), symbol (str), side (BUY/SELL), source_file (str)
    """
    if not backtest_dir.exists():
        return pd.DataFrame(columns=["magic", "strategy_id", "symbol", "side", "source_file"])

    rows = []
    for p in backtest_dir.glob("listOfTrades_*.csv"):
        m = _BACKTEST_FILE_RE.match(p.name)
        if not m:
            # handle odd double underscore case: listOfTrades_6.27.190__USDJPY_39_BUY.csv
            name2 = re.sub(r"__+", "_", p.name)
            m = _BACKTEST_FILE_RE.match(name2)
            if not m:
                continue

        sid = m.group("sid").strip()
        sym = m.group("sym").strip()
        magic = int(m.group("magic"))
        side = normalize_side(m.group("side"))

        if not sid or not sym or side is None:
            continue

        rows.append({
            "magic": magic,
            "strategy_id": sid,
            "symbol": sym,
            "side": side,
            "source_file": p.name,
        })

    if not rows:
        return pd.DataFrame(columns=["magic", "strategy_id", "symbol", "side", "source_file"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["magic", "strategy_id", "symbol", "side"])
    df["magic"] = pd.to_numeric(df["magic"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["magic"])
    df["magic"] = df["magic"].astype(int)
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str)
    df["side"] = df["side"].astype(str)
    return df.sort_values(["magic", "symbol", "side", "strategy_id"]).reset_index(drop=True)


def resolve_from_registry(reg: pd.DataFrame, magic: int, symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve (strategy_id, side) using backtest registry.
    Priority:
      1) exact match on (magic, symbol)
      2) if none, and magic is unique across registry -> use that
      3) else None
    """
    if reg is None or reg.empty:
        return None, None
    try:
        magic = int(magic)
    except Exception:
        return None, None

    sym = str(symbol).strip()
    hit = reg[(reg["magic"] == magic) & (reg["symbol"] == sym)]
    if len(hit) >= 1:
        return str(hit.iloc[0]["strategy_id"]), normalize_side(hit.iloc[0]["side"])

    hit2 = reg[reg["magic"] == magic]
    if len(hit2) == 1:
        return str(hit2.iloc[0]["strategy_id"]), normalize_side(hit2.iloc[0]["side"])

    return None, None


# ----------------------------
# Input loader: closed_trades.csv
# ----------------------------
REQUIRED_CLOSED_COLS = [
    "account_id", "position_id", "symbol",
    "open_time_utc", "close_time_utc",
    "profit_sum", "swap_sum", "commission_sum", "net_sum",
    "magic", "comment_last"
]


def read_closed_trades(path: Path, account_id: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_CLOSED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"closed_trades.csv missing columns: {missing} | file={path}")

    out = df.copy()
    out["account_id"] = pd.to_numeric(out["account_id"], errors="coerce").astype("Int64")
    out["position_id"] = pd.to_numeric(out["position_id"], errors="coerce").astype("Int64")
    out["magic"] = pd.to_numeric(out["magic"], errors="coerce").astype("Int64")

    out["open_time_utc"] = pd.to_datetime(out["open_time_utc"], errors="coerce", utc=True)
    out["close_time_utc"] = pd.to_datetime(out["close_time_utc"], errors="coerce", utc=True)

    for c in ["profit_sum", "swap_sum", "commission_sum", "net_sum"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)

    out["symbol"] = out["symbol"].astype(str).fillna("")
    out["comment_last"] = out["comment_last"].astype(str).fillna("")

    out["net_pnl"] = out["net_sum"].astype(float)

    out["strategy_id_comment"] = out["comment_last"].apply(extract_strategy_from_comment)

    # side may exist or not
    if "side" in out.columns:
        out["side_raw"] = out["side"].apply(normalize_side)
    else:
        out["side_raw"] = None

    out.loc[out["account_id"].isna(), "account_id"] = int(account_id)

    out = out.dropna(subset=["account_id", "position_id", "close_time_utc", "symbol", "magic"])
    out["account_id"] = out["account_id"].astype(int)
    out["position_id"] = out["position_id"].astype(int)
    out["magic"] = out["magic"].astype(int)

    return out.sort_values(["close_time_utc", "position_id"]).reset_index(drop=True)


# ----------------------------
# KPI + performance
# ----------------------------
def kpis_from_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()

    g = trades.groupby(["strategy_id", "symbol", "magic", "side"], dropna=False)
    out = g.agg(
        net_pnl=("net_pnl", "sum"),
        n_trades=("position_id", "count"),
        avg_trade=("net_pnl", "mean"),
        win_rate=("net_pnl", lambda x: float((x > 0).mean()) if len(x) else np.nan),
        first_time=("open_time_utc", "min"),
        last_time=("close_time_utc", "max"),
    ).reset_index()
    return out.sort_values("net_pnl", ascending=False).reset_index(drop=True)


def perf_from_trades(trades_sub: pd.DataFrame, freq: str, start_equity: float) -> pd.DataFrame:
    """
    Perf aggregated by close_time_utc:
      pnl_money = sum(net_pnl) in period
      cum_pnl_money = cumsum
      nav = start_equity + cum_pnl_money
    """
    if trades_sub is None or trades_sub.empty:
        return pd.DataFrame()

    d = trades_sub.dropna(subset=["close_time_utc"]).copy()
    d = d.sort_values("close_time_utc").set_index("close_time_utc")
    pnl = d["net_pnl"].resample(freq).sum().astype(float)

    out = pd.DataFrame({"pnl_money": pnl})
    out["cum_pnl_money"] = out["pnl_money"].cumsum()
    out["nav"] = float(start_equity) + out["cum_pnl_money"]
    out = out.reset_index().rename(columns={"close_time_utc": "date"})
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out


# ----------------------------
# Export
# ----------------------------
def export_bucket(base_dir: Path, bucket_name: str, trades_sub: pd.DataFrame, start_equity: float) -> None:
    p = base_dir / bucket_name
    ensure_dir(p)

    trades_sub.sort_values("close_time_utc").to_csv(p / "trades.csv", index=False)

    k = kpis_from_trades(trades_sub)
    if not k.empty:
        k.to_csv(p / "kpis.csv", index=False)

    wk = perf_from_trades(trades_sub, freq="W-FRI", start_equity=start_equity)
    mo = perf_from_trades(trades_sub, freq="M", start_equity=start_equity)

    if not wk.empty:
        wk.to_csv(p / "weekly_performance.csv", index=False)
    if not mo.empty:
        mo.to_csv(p / "monthly_performance.csv", index=False)


# ----------------------------
# Cycle
# ----------------------------
def run_cycle(
    input_root: Path,
    backtest_dir: Path,
    cutoff: pd.Timestamp,
    start_equity: float,
) -> Dict[str, object]:
    reg = build_magic_registry(backtest_dir)

    # persist registry for debugging
    try:
        reg_path = input_root / "magic_registry.csv"
        reg.to_csv(reg_path, index=False)
    except Exception:
        pass

    accounts = sorted([p for p in input_root.iterdir() if p.is_dir() and p.name.lower().startswith("account_")])

    out = {
        "status": "ok",
        "updated_at_utc": utc_now_str(),
        "input_root": str(input_root),
        "backtest_dir": str(backtest_dir),
        "cutoff_utc": str(cutoff),
        "n_accounts": int(len(accounts)),
        "accounts": [],
    }

    for acc_dir in accounts:
        try:
            account_id = int(acc_dir.name.split("_", 1)[1])
        except Exception:
            out["accounts"].append({"account_dir": str(acc_dir), "status": "bad_account_folder"})
            continue

        closed_path = acc_dir / "closed_trades.csv"
        if not closed_path.exists():
            out["accounts"].append({"account_dir": str(acc_dir), "status": "missing_closed_trades"})
            continue

        trades = read_closed_trades(closed_path, account_id=account_id)
        if trades.empty:
            out["accounts"].append({"account_dir": str(acc_dir), "status": "no_trades_rows"})
            continue

        trades = trades[trades["close_time_utc"] >= cutoff].copy()
        if trades.empty:
            out["accounts"].append({"account_dir": str(acc_dir), "status": "no_trades_after_cutoff"})
            continue

        # fill from comment
        trades["strategy_id"] = trades["strategy_id_comment"].astype("object")

        # fill strategy_id + side from backtest registry
        sids = []
        sides = []
        for _, r in trades.iterrows():
            sid = r["strategy_id"]
            side = r["side_raw"]

            if (sid is None) or (str(sid).strip() == "") or (str(sid).lower() == "nan"):
                sid2, side2 = resolve_from_registry(reg, magic=r["magic"], symbol=r["symbol"])
                if sid2 is not None:
                    sid = sid2
                if side is None and side2 is not None:
                    side = side2

            if side is None:
                _, side2 = resolve_from_registry(reg, magic=r["magic"], symbol=r["symbol"])
                if side2 is not None:
                    side = side2

            sids.append(sid)
            sides.append(side)

        trades["strategy_id"] = pd.Series(sids, index=trades.index).astype("object")
        trades["side"] = pd.Series(sides, index=trades.index).astype("object")

        trades["strategy_id"] = trades["strategy_id"].astype(str)
        trades["strategy_id"] = trades["strategy_id"].where(trades["strategy_id"].str.len() > 0)
        trades["side"] = trades["side"].apply(normalize_side)

        miss_sid = int(trades["strategy_id"].isna().sum())
        miss_side = int(trades["side"].isna().sum())

        valid = trades.dropna(subset=["strategy_id", "side"]).copy()
        if valid.empty:
            out["accounts"].append({
                "account_dir": str(acc_dir),
                "status": "no_valid_buckets",
                "debug": {
                    "n_trades_rows": int(len(trades)),
                    "missing_strategy_id": miss_sid,
                    "missing_side": miss_side,
                    "sum_net_pnl": float(trades["net_pnl"].sum()),
                    "registry_rows": int(len(reg)),
                }
            })
            continue

        strat_base = acc_dir / "strategies"
        ensure_dir(strat_base)

        n_buckets = 0
        for (sid, sym, mag, side), g in valid.groupby(["strategy_id", "symbol", "magic", "side"], sort=False):
            bucket_name = f"{sanitize_name(sid)}_{sanitize_name(sym)}_{int(mag)}_{sanitize_name(side)}"
            export_bucket(strat_base, bucket_name, g, start_equity=start_equity)
            n_buckets += 1

        out["accounts"].append({
            "account_dir": str(acc_dir),
            "status": "ok",
            "n_buckets": int(n_buckets),
            "debug": {
                "n_trades_rows": int(len(trades)),
                "missing_strategy_id": miss_sid,
                "missing_side": miss_side,
                "sum_net_pnl": float(trades["net_pnl"].sum()),
                "registry_rows": int(len(reg)),
            }
        })

    # write summary
    try:
        summary_path = input_root / "summary_live_performance.json"
        summary_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass

    return out


# ----------------------------
# CLI
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=str, default=str(LIVE_ROOT_DEFAULT))
    ap.add_argument("--backtest-dir", type=str, default=str(BACKTEST_DIR_DEFAULT))
    ap.add_argument("--cutoff-date", type=str, default="2026-01-11")
    ap.add_argument("--start-equity", type=float, default=100000.0)

    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=60.0)

    args = ap.parse_args()

    if args.loop and args.once:
        raise ValueError("Use either --once or --loop, not both.")
    if not args.loop and not args.once:
        args.once = True

    input_root = Path(args.input_root).resolve()
    backtest_dir = Path(args.backtest_dir).resolve()
    cutoff = pd.to_datetime(args.cutoff_date, utc=True)

    print("[INFO] ROOT       =", ROOT)
    print("[INFO] input_root =", input_root)
    print("[INFO] backtest   =", backtest_dir)
    print("[INFO] cutoff     =", cutoff)
    print("[INFO] mode       =", "loop" if args.loop else "once")

    if args.once:
        info = run_cycle(
            input_root=input_root,
            backtest_dir=backtest_dir,
            cutoff=cutoff,
            start_equity=float(args.start_equity),
        )
        print("[DONE]", info)
        return

    try:
        while True:
            info = run_cycle(
                input_root=input_root,
                backtest_dir=backtest_dir,
                cutoff=cutoff,
                start_equity=float(args.start_equity),
            )
            print("[LOOP]", utc_now_str(), "|", {"n_accounts": info.get("n_accounts"), "status": info.get("status")})
            time.sleep(float(args.sleep))
    except KeyboardInterrupt:
        print("[INFO] Stopped.")


if __name__ == "__main__":
    main()

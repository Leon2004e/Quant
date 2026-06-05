# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/3_Research/Trades/Backtest/code.py

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: research_backtest_lot_scaling_loader
# script_name: Backtest Lot Scaling Research Loader
# owner: Leon Everts
# status: active
# layer: 3_Research
# domain: Trades/Backtest
# asset_type: Research_Loader
# purpose: Load baseline backtest trades, simulate lot scaling per strategy, check risk limits, and save research outputs into Data_Center/Data/3_Research/Trades/Backtest.
# inputs:
#   - Data_Center/Data/2_Baseline/Trades/Backtest/DEMO_ACCOUNT/**/*.csv
#   - Data_Center/Data/2_Baseline/Trades/Backtest/LIVE_ACCOUNT/**/*.csv
#   - Data_Center/Data/2_Baseline/Strategy/**/*.csv
# outputs:
#   - Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Trades/<account>/<strategy_key>/lot_x_xx.csv
#   - Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Summary/lot_scaling_full_summary.csv
#   - Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Summary/lot_scaling_dashboard_table.csv
#   - Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Summary/best_valid_lot_per_strategy.csv
#   - Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Registry/code_registry.json
# dependencies:
#   - pathlib
#   - pandas
#   - numpy
#   - math
#   - re
#   - json
# schedule: manual
# version: v3.0.0_quant_research_structure
# last_reviewed: 2026-06-04
# business_criticality: high
# environment: desktop
# registry_group: research_backtest
# notes:
#   - This script reads from 2_Baseline and writes only to 3_Research.
#   - It does not write to feature_engineered or raw.
#   - It supports DEMO_ACCOUNT and LIVE_ACCOUNT.
#   - It keeps CODE_REGISTRY metadata inside the code and saves a registry JSON copy.
# ============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import math
import re

import numpy as np
import pandas as pd


# ============================================================
# CODE_REGISTRY OBJECT
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "research_backtest_lot_scaling_loader",
    "script_name": "Backtest Lot Scaling Research Loader",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "3_Research",
    "domain": "Trades/Backtest",
    "asset_type": "Research_Loader",
    "purpose": (
        "Load baseline backtest trades, simulate lot scaling per strategy, "
        "check risk limits, and save research outputs into "
        "Data_Center/Data/3_Research/Trades/Backtest."
    ),
    "inputs": [
        "Data_Center/Data/2_Baseline/Trades/Backtest/DEMO_ACCOUNT/**/*.csv",
        "Data_Center/Data/2_Baseline/Trades/Backtest/LIVE_ACCOUNT/**/*.csv",
        "Data_Center/Data/2_Baseline/Strategy/**/*.csv",
    ],
    "outputs": [
        "Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Trades/<account>/<strategy_key>/lot_x_xx.csv",
        "Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Summary/lot_scaling_full_summary.csv",
        "Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Summary/lot_scaling_dashboard_table.csv",
        "Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Summary/best_valid_lot_per_strategy.csv",
        "Data_Center/Data/3_Research/Trades/Backtest/Lot_Scaling/Registry/code_registry.json",
    ],
    "dependencies": ["pathlib", "pandas", "numpy", "math", "re", "json"],
    "schedule": "manual",
    "version": "v3.0.0_quant_research_structure",
    "last_reviewed": "2026-06-04",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "research_backtest",
    "notes": [
        "Reads from 2_Baseline and writes only to 3_Research.",
        "Does not write to feature_engineered or raw.",
        "Supports DEMO_ACCOUNT and LIVE_ACCOUNT.",
        "Keeps CODE_REGISTRY metadata inside the code and saves registry JSON copy.",
    ],
}


def get_code_registry() -> Dict[str, object]:
    return dict(CODE_REGISTRY)


# ============================================================
# ROOT / PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_quant_root(start: Path) -> Path:
    """
    Finds project root by searching for:
        QUANT/
        ├── Dashboard/
        └── Data_Center/
    """
    current = start.resolve()
    for p in [current] + list(current.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p.resolve()

    # fallback for older FTMO structure
    for p in [current] + list(current.parents):
        data_center = p / "Data_Center"
        data_dir = data_center / "Data"
        if data_center.exists() and data_dir.exists():
            return p.resolve()

    raise RuntimeError(
        "QUANT root not found. Expected folder with Dashboard/ and Data_Center/. "
        f"Start={start}"
    )


QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"

BASELINE_ROOT = DATA_DIR / "2_Baseline"
BASELINE_TRADES_BACKTEST_ROOT = BASELINE_ROOT / "Trades" / "Backtest"
BASELINE_STRATEGY_ROOT = BASELINE_ROOT / "Strategy"

RESEARCH_ROOT = DATA_DIR / "3_Research"
RESEARCH_TRADES_BACKTEST_ROOT = RESEARCH_ROOT / "Trades" / "Backtest"

OUTPUT_ROOT = RESEARCH_TRADES_BACKTEST_ROOT / "Lot_Scaling"
OUTPUT_TRADES_ROOT = OUTPUT_ROOT / "Trades"
OUTPUT_SUMMARY_ROOT = OUTPUT_ROOT / "Summary"
OUTPUT_REGISTRY_ROOT = OUTPUT_ROOT / "Registry"

DEMO_INPUT_ROOT = BASELINE_TRADES_BACKTEST_ROOT / "DEMO_ACCOUNT"
LIVE_INPUT_ROOT = BASELINE_TRADES_BACKTEST_ROOT / "LIVE_ACCOUNT"


# ============================================================
# CONFIG
# ============================================================

BASE_LOT = 0.10
MIN_LOT = 0.10
MAX_LOT = 10.00
LOT_STEP = 0.10

ACCOUNT_SIZE = 200_000.0

DAILY_LOSS_LIMIT_PCT = 0.05
WEEKLY_LOSS_LIMIT_PCT = 0.02
TOTAL_DD_LIMIT_PCT = 0.10

WRITE_EACH_LOT_TRADE_FILE = True
WRITE_CSV = True
WRITE_PARQUET = True

SUPPORTED_INPUT_EXTENSIONS = {".csv", ".parquet"}


# ============================================================
# HELPERS
# ============================================================

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_ok(msg: str) -> None:
    print(f"[OK] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def to_utc(x) -> pd.Series:
    return pd.to_datetime(x, errors="coerce", utc=True)


def sanitize_name(x: object, max_len: int = 220) -> str:
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "nat", "<na>"}:
        return "UNKNOWN"
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = re.sub(r"\s+", "_", s)
    while "__" in s:
        s = s.replace("__", "_")
    return (s.strip("._") or "UNKNOWN")[:max_len]


def norm_dir(x: object) -> str:
    s = str(x).strip().upper()
    if s in {"BUY", "LONG"}:
        return "BUY"
    if s in {"SELL", "SHORT"}:
        return "SELL"
    if s == "BOTH":
        return "BOTH"
    return s


def lot_grid() -> List[float]:
    n = int(round((MAX_LOT - MIN_LOT) / LOT_STEP)) + 1
    return [round(MIN_LOT + i * LOT_STEP, 2) for i in range(n)]


def write_table(df: pd.DataFrame, path_base: Path, index: bool = False) -> None:
    ensure_dir(path_base.parent)

    if WRITE_PARQUET:
        try:
            df.to_parquet(path_base.with_suffix(".parquet"), index=index)
        except Exception as e:
            log_warn(f"parquet write failed: {path_base} | {e}")

    if WRITE_CSV:
        df.to_csv(path_base.with_suffix(".csv"), index=index)


def write_registry_copy() -> None:
    ensure_dir(OUTPUT_REGISTRY_ROOT)
    out = OUTPUT_REGISTRY_ROOT / "code_registry.json"
    out.write_text(json.dumps(CODE_REGISTRY, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================
# DISCOVERY / STRATEGY META
# ============================================================

def discover_backtest_files() -> List[Path]:
    """
    Reads only from:
        Data_Center/Data/2_Baseline/Trades/Backtest/DEMO_ACCOUNT
        Data_Center/Data/2_Baseline/Trades/Backtest/LIVE_ACCOUNT
    """
    files: List[Path] = []

    for root in [DEMO_INPUT_ROOT, LIVE_INPUT_ROOT]:
        if not root.exists():
            continue

        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
                continue

            name = f.name.lower()
            if any(x in name for x in ["summary", "failed", "enriched", "scaled", "dashboard_table"]):
                continue

            files.append(f)

    return sorted(files)


def detect_account_group(path: Path) -> str:
    parts = [p.upper() for p in path.parts]
    if "DEMO_ACCOUNT" in parts:
        return "DEMO_ACCOUNT"
    if "LIVE_ACCOUNT" in parts:
        return "LIVE_ACCOUNT"

    text = str(path).upper()
    if "DEMO" in text:
        return "DEMO_ACCOUNT"
    if "LIVE" in text:
        return "LIVE_ACCOUNT"

    return "UNKNOWN_ACCOUNT"


def infer_strategy_from_filename(path: Path) -> Dict[str, str]:
    """
    Supports:
        listOfTrades_AUDJPY_1_3.14.146_BUY_M15.csv
        listOfTrades_AUDJPY_1_3.14.146_BOTH_M15.csv
        AUDJPY_1_3.14.146_BUY_M15.csv
        AUDJPY_3.14.146_BUY.csv
    """
    stem = path.stem
    parts = stem.split("_")

    out = {
        "file_stem": stem,
        "symbol": "",
        "internal_id": "",
        "magic": "",
        "strategy_id": "",
        "direction": "",
        "tf": "",
    }

    if len(parts) >= 6 and parts[0].lower() == "listoftrades":
        out["symbol"] = parts[1]
        out["internal_id"] = parts[2]
        out["magic"] = parts[2]
        out["strategy_id"] = parts[3]
        out["direction"] = norm_dir(parts[4])
        out["tf"] = parts[5]
        return out

    m = re.search(
        r"(?P<symbol>[A-Z0-9.]+)_(?P<magic>\d+)_(?P<sid>\d+(?:\.\d+)+)_(?P<side>BUY|SELL|BOTH)_(?P<tf>[A-Za-z0-9]+)",
        stem,
        re.IGNORECASE,
    )
    if m:
        out["symbol"] = m.group("symbol")
        out["magic"] = m.group("magic")
        out["internal_id"] = m.group("magic")
        out["strategy_id"] = m.group("sid")
        out["direction"] = norm_dir(m.group("side"))
        out["tf"] = m.group("tf")
        return out

    m = re.search(
        r"(?P<symbol>[A-Z0-9.]+)_(?P<sid>\d+(?:\.\d+)+)_(?P<side>BUY|SELL|BOTH)",
        stem,
        re.IGNORECASE,
    )
    if m:
        out["symbol"] = m.group("symbol")
        out["strategy_id"] = m.group("sid")
        out["direction"] = norm_dir(m.group("side"))
        return out

    sid = re.search(r"(\d+(?:\.\d+)+)", stem)
    if sid:
        out["strategy_id"] = sid.group(1)

    side = re.search(r"(BUY|SELL|BOTH)", stem, re.IGNORECASE)
    if side:
        out["direction"] = norm_dir(side.group(1))

    return out


def build_strategy_key(path: Path, df: Optional[pd.DataFrame] = None) -> str:
    meta = infer_strategy_from_filename(path)
    account = detect_account_group(path)

    symbol = meta["symbol"]
    strategy_id = meta["strategy_id"]
    direction = meta["direction"]

    if df is not None and not df.empty:
        if not symbol and "symbol" in df.columns:
            symbol = str(df["symbol"].dropna().astype(str).iloc[0]) if len(df["symbol"].dropna()) else ""
        if not direction and "direction" in df.columns:
            uniq = sorted(set(df["direction"].dropna().astype(str).map(norm_dir)))
            direction = "BOTH" if len([x for x in uniq if x in {"BUY", "SELL"}]) > 1 else (uniq[0] if uniq else "")

    parts = [account]
    if symbol:
        parts.append(symbol)
    if strategy_id:
        parts.append(strategy_id)
    if direction:
        parts.append(direction)

    return sanitize_name("_".join(parts))


def build_strategy_name(path: Path, df: Optional[pd.DataFrame] = None) -> str:
    meta = infer_strategy_from_filename(path)

    symbol = meta["symbol"]
    strategy_id = meta["strategy_id"]
    direction = meta["direction"]

    if df is not None and not df.empty:
        if not symbol and "symbol" in df.columns:
            symbol = str(df["symbol"].dropna().astype(str).iloc[0]) if len(df["symbol"].dropna()) else ""
        if not direction and "direction" in df.columns:
            uniq = sorted(set(df["direction"].dropna().astype(str).map(norm_dir)))
            direction = "BOTH" if len([x for x in uniq if x in {"BUY", "SELL"}]) > 1 else (uniq[0] if uniq else "")

    parts = []
    if symbol:
        parts.append(symbol)
    if strategy_id:
        parts.append(strategy_id)
    if direction:
        parts.append(direction)

    return "_".join(parts) if parts else path.stem


# ============================================================
# LOAD BASELINE TRADES
# ============================================================

def read_input_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_baseline_trades(path: Path) -> pd.DataFrame:
    raw = read_input_table(path)

    rename_map = {
        "Ticket": "position_id",
        "Symbol": "symbol",
        "Type": "direction",
        "Open time": "open_time_utc",
        "Open price": "entry_price",
        "Size": "volume_in",
        "Close time": "close_time_utc",
        "Close price": "exit_price",
        "Time in trade": "time_in_trade",
        "Profit/Loss": "profit_sum",
        "Cummulative P/L": "cumulative_profit_loss",
        "Comm/Swap": "commission_sum",
        "Commission": "commission_sum",
        "Swap": "swap_sum",
        "P/L in money": "net_sum",
        "Cummulative money P/L": "cumulative_money_pnl",
        "P/L in pips": "pnl_pips",
        "Cummulative pips P/L": "cumulative_pips_pnl",
        "P/L in %": "pnl_pct",
        "Cummulative % P/L": "cumulative_pct_pnl",
        "Comment": "comment",
        "Sample type": "sample_type",
    }

    d = raw.rename(columns=rename_map).copy()

    # Support already normalized baseline schemas.
    if "net_sum" not in d.columns:
        if "profit_sum" in d.columns:
            d["net_sum"] = d["profit_sum"]
        elif "Profit/Loss" in raw.columns:
            d["net_sum"] = raw["Profit/Loss"]
        else:
            d["net_sum"] = 0.0

    if "profit_sum" not in d.columns:
        d["profit_sum"] = d["net_sum"]

    if "commission_sum" not in d.columns:
        d["commission_sum"] = 0.0

    if "volume_in" not in d.columns:
        d["volume_in"] = BASE_LOT

    if "volume_out" not in d.columns:
        d["volume_out"] = d["volume_in"]

    if "position_id" not in d.columns:
        d["position_id"] = range(1, len(d) + 1)

    required = [
        "position_id",
        "symbol",
        "direction",
        "open_time_utc",
        "entry_price",
        "volume_in",
        "close_time_utc",
        "exit_price",
        "net_sum",
    ]

    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"Pflichtspalten fehlen {missing} | Datei={path}")

    d["open_time_utc"] = to_utc(d["open_time_utc"])
    d["close_time_utc"] = to_utc(d["close_time_utc"])

    # If strict UTC parsing fails because input uses German date format, try dayfirst.
    if d["open_time_utc"].isna().all() and "Open time" in raw.columns:
        d["open_time_utc"] = pd.to_datetime(raw["Open time"], format="%d.%m.%Y %H:%M:%S", errors="coerce", utc=True)
    if d["close_time_utc"].isna().all() and "Close time" in raw.columns:
        d["close_time_utc"] = pd.to_datetime(raw["Close time"], format="%d.%m.%Y %H:%M:%S", errors="coerce", utc=True)

    numeric_cols = [
        "entry_price",
        "exit_price",
        "volume_in",
        "volume_out",
        "profit_sum",
        "commission_sum",
        "swap_sum",
        "net_sum",
        "cumulative_profit_loss",
        "cumulative_money_pnl",
        "pnl_pips",
        "cumulative_pips_pnl",
        "pnl_pct",
        "cumulative_pct_pnl",
    ]

    for c in numeric_cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    meta = infer_strategy_from_filename(path)

    d["position_id"] = d["position_id"].astype(str)
    d["symbol"] = d["symbol"].astype(str).str.strip()
    d["direction"] = d["direction"].map(norm_dir)

    if "strategy_id" not in d.columns or d["strategy_id"].astype(str).str.strip().eq("").all():
        d["strategy_id"] = meta["strategy_id"]
    else:
        d["strategy_id"] = d["strategy_id"].astype(str).str.strip()

    if "sample_type" not in d.columns:
        d["sample_type"] = ""
    d["sample_type"] = d["sample_type"].astype(str).str.upper().str.strip()

    account_group = detect_account_group(path)
    d["account_group"] = account_group
    d["account_type"] = "DEMO" if account_group == "DEMO_ACCOUNT" else "LIVE" if account_group == "LIVE_ACCOUNT" else ""
    d["base_lot_detected"] = d["volume_in"]
    d["source_file"] = str(path)

    d = d.dropna(
        subset=[
            "open_time_utc",
            "close_time_utc",
            "entry_price",
            "exit_price",
            "net_sum",
        ]
    ).copy()

    d = d.sort_values(
        ["open_time_utc", "close_time_utc", "position_id"]
    ).reset_index(drop=True)

    return d


# ============================================================
# SCALING
# ============================================================

def scale_trade_list(df: pd.DataFrame, lot: float) -> pd.DataFrame:
    d = df.copy()

    scale = float(lot) / float(BASE_LOT)

    d["base_lot"] = BASE_LOT
    d["target_lot"] = float(lot)
    d["lot_scale_factor"] = scale

    for c in ["profit_sum", "commission_sum", "swap_sum", "net_sum"]:
        if c in d.columns:
            d[f"scaled_{c}"] = safe_num(d[c]).fillna(0.0) * scale

    d["scaled_volume_in"] = safe_num(d["volume_in"]).fillna(BASE_LOT) * scale
    d["scaled_volume_out"] = safe_num(d["volume_out"]).fillna(BASE_LOT) * scale

    d["scaled_net_sum"] = safe_num(d["net_sum"]).fillna(0.0) * scale
    d["scaled_cumulative_net_sum"] = d["scaled_net_sum"].cumsum()

    if "pnl_pips" in d.columns:
        d["scaled_pnl_pips"] = d["pnl_pips"]

    if "pnl_pct" in d.columns:
        d["scaled_pnl_pct_raw"] = d["pnl_pct"]

    return d


# ============================================================
# METRICS
# ============================================================

def max_total_drawdown(pnl: pd.Series) -> float:
    x = safe_num(pnl).fillna(0.0).astype(float)

    if x.empty:
        return 0.0

    curve = x.cumsum()
    peak = curve.cummax()
    dd = curve - peak

    return float(dd.min())


def resample_period(df: pd.DataFrame, pnl_col: str, time_col: str, freq: str) -> pd.Series:
    d = df[[time_col, pnl_col]].copy()

    d[time_col] = to_utc(d[time_col])
    d[pnl_col] = safe_num(d[pnl_col]).fillna(0.0)

    d = d.dropna(subset=[time_col]).sort_values(time_col)

    if d.empty:
        return pd.Series(dtype=float)

    return d.set_index(time_col)[pnl_col].resample(freq).sum().astype(float)


def period_metrics(s: pd.Series, prefix: str) -> Dict[str, float]:
    if s.empty:
        return {
            f"avg_{prefix}": np.nan,
            f"avg_good_{prefix}": np.nan,
            f"avg_bad_{prefix}": np.nan,
            f"best_{prefix}": np.nan,
            f"worst_{prefix}": np.nan,
            f"{prefix}_vola": np.nan,
            f"{prefix}_downside_vola": np.nan,
            f"positive_{prefix}_ratio": np.nan,
            f"{prefix}_count": 0,
        }

    good = s[s > 0]
    bad = s[s < 0]

    return {
        f"avg_{prefix}": float(s.mean()),
        f"avg_good_{prefix}": float(good.mean()) if len(good) else 0.0,
        f"avg_bad_{prefix}": float(bad.mean()) if len(bad) else 0.0,
        f"best_{prefix}": float(s.max()),
        f"worst_{prefix}": float(s.min()),
        f"{prefix}_vola": float(s.std()) if len(s) > 1 else 0.0,
        f"{prefix}_downside_vola": float(bad.std()) if len(bad) > 1 else 0.0,
        f"positive_{prefix}_ratio": float((s > 0).mean()),
        f"{prefix}_count": int(len(s)),
    }


def profit_factor(pnl: pd.Series) -> float:
    x = safe_num(pnl).fillna(0.0)

    gross_profit = float(x[x > 0].sum())
    gross_loss = float(x[x < 0].sum())

    if gross_loss >= 0:
        return np.nan

    return gross_profit / abs(gross_loss)


def losing_streak(pnl: pd.Series) -> int:
    cur = 0
    best = 0

    for v in safe_num(pnl).fillna(0.0):
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0

    return int(best)


def calc_metrics(
    df: pd.DataFrame,
    strategy_key: str,
    strategy_name: str,
    source_file: str,
    lot: float,
) -> Dict[str, object]:

    time_col = "close_time_utc"
    pnl_col = "scaled_net_sum"

    d = df.copy()
    d[time_col] = to_utc(d[time_col])
    d[pnl_col] = safe_num(d[pnl_col]).fillna(0.0)

    d = d.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    pnl = d[pnl_col]
    n = int(len(d))

    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())

    total_pnl = float(pnl.sum()) if n else 0.0
    avg_trade = float(pnl.mean()) if n else np.nan
    std_trade = float(pnl.std()) if n > 1 else 0.0

    max_dd = max_total_drawdown(pnl)

    day = resample_period(d, pnl_col, time_col, "1D")
    week = resample_period(d, pnl_col, time_col, "1W")
    month = resample_period(d, pnl_col, time_col, "1M")

    daily_limit = ACCOUNT_SIZE * DAILY_LOSS_LIMIT_PCT
    weekly_limit = ACCOUNT_SIZE * WEEKLY_LOSS_LIMIT_PCT
    total_dd_limit = ACCOUNT_SIZE * TOTAL_DD_LIMIT_PCT

    worst_day = float(day.min()) if len(day) else 0.0
    worst_week = float(week.min()) if len(week) else 0.0
    worst_month = float(month.min()) if len(month) else 0.0

    daily_usage = abs(worst_day) / daily_limit if daily_limit else np.nan
    weekly_usage = abs(worst_week) / weekly_limit if weekly_limit else np.nan
    total_dd_usage = abs(max_dd) / total_dd_limit if total_dd_limit else np.nan

    usages = {
        "DAILY": daily_usage,
        "WEEKLY": weekly_usage,
        "TOTAL_DD": total_dd_usage,
    }

    valid_usages = {k: v for k, v in usages.items() if pd.notna(v)}
    limit_bottleneck = max(valid_usages, key=valid_usages.get) if valid_usages else ""

    is_valid = (
        abs(worst_day) <= daily_limit
        and abs(worst_week) <= weekly_limit
        and abs(max_dd) <= total_dd_limit
    )

    pf = profit_factor(pnl)
    wr = wins / n if n else np.nan

    sqn = avg_trade / std_trade * math.sqrt(n) if n > 1 and std_trade > 0 else np.nan
    ret_dd = total_pnl / abs(max_dd) if max_dd < 0 else np.nan

    out = {
        "strategy_key": strategy_key,
        "strategy_name": strategy_name,
        "account_group": str(d["account_group"].iloc[0]) if "account_group" in d.columns and len(d) else "",
        "account_type": str(d["account_type"].iloc[0]) if "account_type" in d.columns and len(d) else "",
        "symbol": str(d["symbol"].iloc[0]) if "symbol" in d.columns and len(d) else "",
        "strategy_id": str(d["strategy_id"].iloc[0]) if "strategy_id" in d.columns and len(d) else "",
        "direction": str(d["direction"].iloc[0]) if "direction" in d.columns and len(d) else "",
        "sample_types": "+".join(sorted(set(d["sample_type"].dropna().astype(str)))) if "sample_type" in d.columns else "",
        "source_file": source_file,

        "base_lot": BASE_LOT,
        "target_lot": lot,
        "scale_factor": lot / BASE_LOT,

        "trade_count": n,
        "wins": wins,
        "losses": losses,
        "winrate": wr,
        "profit_factor": pf,

        "total_pnl": total_pnl,
        "avg_trade": avg_trade,
        "std_trade": std_trade,
        "sqn": sqn,
        "ret_over_dd": ret_dd,

        "max_total_dd": max_dd,
        "max_losing_streak": losing_streak(pnl),

        "account_size": ACCOUNT_SIZE,
        "daily_limit": daily_limit,
        "weekly_limit": weekly_limit,
        "total_dd_limit": total_dd_limit,

        "worst_day": worst_day,
        "worst_week": worst_week,
        "worst_month": worst_month,

        "daily_limit_usage": daily_usage,
        "weekly_limit_usage": weekly_usage,
        "total_dd_limit_usage": total_dd_usage,
        "max_limit_usage": max(valid_usages.values()) if valid_usages else np.nan,
        "limit_bottleneck": limit_bottleneck,

        "violates_daily_limit": int(abs(worst_day) > daily_limit),
        "violates_weekly_limit": int(abs(worst_week) > weekly_limit),
        "violates_total_dd_limit": int(abs(max_dd) > total_dd_limit),
        "is_valid_lot": int(is_valid),
    }

    out.update(period_metrics(day, "day"))
    out.update(period_metrics(week, "week"))
    out.update(period_metrics(month, "month"))

    return out


# ============================================================
# STRATEGY LEVEL BEST FIELDS
# ============================================================

def add_strategy_level_best_fields(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary

    d = summary.copy()

    extra_cols = {
        "best_valid_lot": np.nan,
        "max_pnl_valid_lot": np.nan,
        "first_invalid_lot": np.nan,
        "best_valid_total_pnl": np.nan,
        "best_valid_max_total_dd": np.nan,
        "best_valid_worst_day": np.nan,
        "best_valid_worst_week": np.nan,
        "best_valid_worst_month": np.nan,
        "best_valid_limit_bottleneck": "",
    }

    for c, v in extra_cols.items():
        d[c] = v

    for key, g in d.groupby("strategy_key"):
        g = g.sort_values("target_lot")

        valid = g[g["is_valid_lot"] == 1]
        invalid = g[g["is_valid_lot"] == 0]

        idx = d["strategy_key"] == key

        if not valid.empty:
            best_lot_row = valid.sort_values("target_lot", ascending=False).iloc[0]
            max_pnl_row = valid.sort_values("total_pnl", ascending=False).iloc[0]

            d.loc[idx, "best_valid_lot"] = float(best_lot_row["target_lot"])
            d.loc[idx, "max_pnl_valid_lot"] = float(max_pnl_row["target_lot"])
            d.loc[idx, "best_valid_total_pnl"] = float(best_lot_row["total_pnl"])
            d.loc[idx, "best_valid_max_total_dd"] = float(best_lot_row["max_total_dd"])
            d.loc[idx, "best_valid_worst_day"] = float(best_lot_row["worst_day"])
            d.loc[idx, "best_valid_worst_week"] = float(best_lot_row["worst_week"])
            d.loc[idx, "best_valid_worst_month"] = float(best_lot_row["worst_month"])
            d.loc[idx, "best_valid_limit_bottleneck"] = str(best_lot_row["limit_bottleneck"])

        if not invalid.empty:
            d.loc[idx, "first_invalid_lot"] = float(invalid["target_lot"].min())

    return d


# ============================================================
# OUTPUT BUILDERS
# ============================================================

def output_base_for_strategy(account_group: str, strategy_key: str) -> Path:
    return OUTPUT_TRADES_ROOT / sanitize_name(account_group) / sanitize_name(strategy_key)


def process_file(path: Path) -> pd.DataFrame:
    raw = load_baseline_trades(path)

    if raw.empty:
        return pd.DataFrame()

    strategy_key = build_strategy_key(path, raw)
    strategy_name = build_strategy_name(path, raw)
    account_group = str(raw["account_group"].iloc[0]) if "account_group" in raw.columns and len(raw) else detect_account_group(path)

    rows = []

    for lot in lot_grid():
        scaled = scale_trade_list(raw, lot)

        rows.append(
            calc_metrics(
                df=scaled,
                strategy_key=strategy_key,
                strategy_name=strategy_name,
                source_file=str(path),
                lot=lot,
            )
        )

        if WRITE_EACH_LOT_TRADE_FILE:
            lot_name = f"lot_{lot:.2f}".replace(".", "_")
            write_table(scaled, output_base_for_strategy(account_group, strategy_key) / lot_name, index=False)

    return pd.DataFrame(rows)


def build_best_valid_lot(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    valid = summary[summary["is_valid_lot"] == 1].copy()

    if valid.empty:
        return pd.DataFrame()

    idx = valid.groupby("strategy_key")["target_lot"].idxmax()

    return (
        valid.loc[idx]
        .sort_values(["target_lot", "ret_over_dd"], ascending=[False, False])
        .reset_index(drop=True)
    )


def build_dashboard_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strategy_key",
        "strategy_name",
        "account_group",
        "account_type",
        "symbol",
        "strategy_id",
        "direction",
        "sample_types",
        "target_lot",
        "scale_factor",
        "trade_count",

        "total_pnl",
        "avg_trade",
        "std_trade",
        "profit_factor",
        "winrate",
        "ret_over_dd",
        "sqn",

        "max_total_dd",
        "max_losing_streak",

        "avg_day",
        "avg_good_day",
        "avg_bad_day",
        "best_day",
        "worst_day",
        "day_vola",
        "day_downside_vola",
        "positive_day_ratio",

        "avg_week",
        "avg_good_week",
        "avg_bad_week",
        "best_week",
        "worst_week",
        "week_vola",
        "week_downside_vola",
        "positive_week_ratio",

        "avg_month",
        "avg_good_month",
        "avg_bad_month",
        "best_month",
        "worst_month",
        "month_vola",
        "month_downside_vola",
        "positive_month_ratio",

        "daily_limit",
        "weekly_limit",
        "total_dd_limit",
        "daily_limit_usage",
        "weekly_limit_usage",
        "total_dd_limit_usage",
        "max_limit_usage",
        "limit_bottleneck",

        "violates_daily_limit",
        "violates_weekly_limit",
        "violates_total_dd_limit",
        "is_valid_lot",

        "best_valid_lot",
        "max_pnl_valid_lot",
        "first_invalid_lot",
        "best_valid_total_pnl",
        "best_valid_max_total_dd",
        "best_valid_worst_day",
        "best_valid_worst_week",
        "best_valid_worst_month",
        "best_valid_limit_bottleneck",

        "source_file",
    ]

    return summary[[c for c in cols if c in summary.columns]].copy()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(OUTPUT_TRADES_ROOT)
    ensure_dir(OUTPUT_SUMMARY_ROOT)
    ensure_dir(OUTPUT_REGISTRY_ROOT)
    write_registry_copy()

    print("=" * 100)
    print("QUANT RESEARCH BACKTEST LOT SCALING LOADER")
    print("=" * 100)
    print(f"SCRIPT_PATH         = {SCRIPT_PATH}")
    print(f"QUANT_ROOT          = {QUANT_ROOT}")
    print(f"BASELINE_TRADES     = {BASELINE_TRADES_BACKTEST_ROOT}")
    print(f"DEMO_INPUT_ROOT     = {DEMO_INPUT_ROOT}")
    print(f"LIVE_INPUT_ROOT     = {LIVE_INPUT_ROOT}")
    print(f"RESEARCH_OUTPUT     = {OUTPUT_ROOT}")
    print(f"SUMMARY_ROOT        = {OUTPUT_SUMMARY_ROOT}")
    print(f"TRADES_ROOT         = {OUTPUT_TRADES_ROOT}")
    print(f"REGISTRY_ROOT       = {OUTPUT_REGISTRY_ROOT}")
    print(f"BASE_LOT            = {BASE_LOT}")
    print(f"LOT RANGE           = {MIN_LOT} -> {MAX_LOT} step {LOT_STEP}")
    print(f"ACCOUNT_SIZE        = {ACCOUNT_SIZE:,.2f}")
    print(f"DAILY LIMIT         = {ACCOUNT_SIZE * DAILY_LOSS_LIMIT_PCT:,.2f}")
    print(f"WEEKLY LIMIT        = {ACCOUNT_SIZE * WEEKLY_LOSS_LIMIT_PCT:,.2f}")
    print(f"TOTAL DD LIMIT      = {ACCOUNT_SIZE * TOTAL_DD_LIMIT_PCT:,.2f}")
    print(f"CODE_REGISTRY       = {CODE_REGISTRY['script_id']} | {CODE_REGISTRY['version']}")
    print("=" * 100)

    files = discover_backtest_files()
    print(f"Files found: {len(files)}")

    if not files:
        log_warn("No files found.")
        log_warn(f"Expected files in: {DEMO_INPUT_ROOT}")
        log_warn(f"Expected files in: {LIVE_INPUT_ROOT}")
        return

    frames = []
    ok = 0
    fail = 0

    for i, path in enumerate(files, start=1):
        try:
            try:
                rel = path.relative_to(BASELINE_TRADES_BACKTEST_ROOT)
            except Exception:
                rel = path

            print(f"[{i}/{len(files)}] {rel}")

            res = process_file(path)

            if not res.empty:
                frames.append(res)
                ok += 1

        except Exception as e:
            print(f"[FAIL] {path.name} | {e}")
            fail += 1

    if not frames:
        print("No output generated.")
        return

    summary = pd.concat(frames, ignore_index=True)
    summary = add_strategy_level_best_fields(summary)

    dashboard = build_dashboard_table(summary)
    best = build_best_valid_lot(summary)

    write_table(summary, OUTPUT_SUMMARY_ROOT / "lot_scaling_full_summary", index=False)
    write_table(dashboard, OUTPUT_SUMMARY_ROOT / "lot_scaling_dashboard_table", index=False)
    write_table(best, OUTPUT_SUMMARY_ROOT / "best_valid_lot_per_strategy", index=False)

    print("=" * 100)
    print("DONE")
    print(f"OK:                 {ok}")
    print(f"FAIL:               {fail}")
    print(f"FULL SUMMARY:       {OUTPUT_SUMMARY_ROOT / 'lot_scaling_full_summary.csv'}")
    print(f"DASHBOARD TABLE:    {OUTPUT_SUMMARY_ROOT / 'lot_scaling_dashboard_table.csv'}")
    print(f"BEST VALID LOT:     {OUTPUT_SUMMARY_ROOT / 'best_valid_lot_per_strategy.csv'}")
    print(f"SCALED TRADES ROOT: {OUTPUT_TRADES_ROOT}")
    print(f"REGISTRY JSON:      {OUTPUT_REGISTRY_ROOT / 'code_registry.json'}")
    print("=" * 100)


if __name__ == "__main__":
    main()

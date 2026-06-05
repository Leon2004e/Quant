# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/2_Baseline/Trades/live_strategy_processor.py

Live Strategy Processor v2.3

Ziel:
- Liest alle Live-Trades aus:
    Data_Center/Data/1_Pipeline/Trades/Live/account_*/closed_trades.db

- Exportiert pro echter Strategie eine eigene SQLite-Datenbank:
    Data_Center/Data/2_Baseline/Trades/Live/account_*/<strategy_bucket>/trades.db

Jede Output-DB enthält:
- trades
- kpis
- daily_performance
- weekly_performance
- monthly_performance
- mapping_diagnostics
- inference_master_diagnostics

Zusätzliche Debug-Outputs pro Account:
- mapping_diagnostics.db
- unmapped_trades.csv
- unmapped_summary.csv
- not_exportable_trades.csv
- not_exportable_summary.csv

Kernproblem:
- Viele MT5-Closed-Trades haben comment_last wie:
    [sl 158.874]
    [tp 112.316]
    leer
- Diese Trades enthalten keine direkte Strategy-ID.
- Sie müssen über bereits eindeutig bekannte Strategie-Trades innerhalb desselben Accounts geerbt werden.

Strategie-Erkennung v2.2:
1) EA-Datei, wenn vorhanden:
   symbol_norm + magic + direction

2) comment_last, wenn valide Strategy-ID:
   Strategy_2_36_141       -> 2.36.141
   WF_Matrix_Strategy_1_30_143 -> 1.30.143
   Strategy 5.20.109       -> 5.20.109

3) Registry, wenn Magic eindeutig ist.

4) Account-weite Master-Inference aus den bereits gemappten Trades:
   - position_id
   - symbol_norm + magic
   - symbol_norm + magic + direction
   - symbol_norm + direction
   - symbol_norm only optional disabled

5) [sl]/[tp]/leere Kommentare werden nur zugeordnet, wenn der jeweilige Key eindeutig
   auf genau eine Strategy-ID zeigt.

Wichtige Regeln:
- Strategy-ID muss exakt x.y.z sein.
- Erste Zahl darf nicht 0 sein.
- Ungültig:
    0.47617
    0.18430
    11111
- Magic 0 und 11111 werden nie als finale Bucket-Magic exportiert.
- Wenn eine Strategiegruppe nur Magic 0/11111 hat, wird sie nicht exportiert.
- BOTH wird automatisch gebildet, wenn BUY und SELL derselben symbol_norm + strategy_id zugeordnet sind.
- Alte falsche Output-Buckets werden vor Export gelöscht.
- Datumsfilter aktiv:
    Es werden nur Trades mit close_time_utc >= 2026-02-22 00:00:00 UTC verarbeitet.

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: live_strategy_processor
# script_name: Live Strategy Processor
# owner: Leon Everts
# status: active
# layer: 2_Baseline
# domain: Trades
# asset_type: Strategy_Trade_Normalizer
# purpose: Normalize live closed trades into strategy-level Baseline SQLite databases using direct strategy comments, EA mappings, registry mappings, account-wide inference for sl/tp comments, and full unmapped diagnostics.
# inputs:
#   - Data_Center/Data/1_Pipeline/Trades/Live/*/closed_trades.db
#   - Data_Center/Data/1_Pipeline/Strategy/Strategy_EA/**/*.mq5
#   - Data_Center/Data/1_Pipeline/Strategy/Strategy_Profile/strategy_registry.csv
# outputs:
#   - Data_Center/Data/2_Baseline/Trades/Live/*/*/trades.db
#   - Data_Center/Data/2_Baseline/Trades/Live/*/unmapped_trades.csv
#   - Data_Center/Data/2_Baseline/Trades/Live/*/unmapped_summary.csv
#   - Data_Center/Data/2_Baseline/Trades/Live/*/not_exportable_trades.csv
#   - Data_Center/Data/2_Baseline/Trades/Live/*/not_exportable_summary.csv
# upstream_data:
#   - Data_Center/Data/1_Pipeline/Trades/Live
# downstream_data:
#   - Dashboard/Building_Blocks/Trades/Live
#   - Data_Center/Data/3_Research
# dependencies:
#   - pathlib
#   - sqlite3
#   - pandas
#   - datetime
#   - re
#   - json
#   - shutil
# schedule: manual_or_loop
# version: v2.3.0
# last_reviewed: 2026-06-02
# business_criticality: high
# environment: desktop
# registry_group: baseline_trades
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-02
# tags:
#   - live_trades
#   - strategy_mapping
#   - master_inference
#   - sl_tp_inference
#   - unmapped_diagnostics
#   - both_side_strategies
#   - canonical_magic
#   - sqlite
# notes:
#   - Direct Strategy comments are HIGH confidence.
#   - EA mappings are HIGH confidence.
#   - [sl]/[tp] comments are inferred only when account-wide mapping keys are unique.
#   - Invalid Strategy IDs like 0.47617 are ignored.
#   - Bad magic values 0 and 11111 are never used as final bucket magic.
#   - All unmapped clusters are printed in terminal and exported to CSV.
# ============================================================
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


# ============================================================
# CODE REGISTRY
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "live_strategy_processor",
    "script_name": "Live Strategy Processor",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "2_Baseline",
    "domain": "Trades",
    "asset_type": "Strategy_Trade_Normalizer",
    "purpose": (
        "Normalize live closed trades into strategy-level Baseline SQLite databases "
        "using direct strategy comments, EA mappings, registry mappings, account-wide inference for sl/tp comments, "
        "and full unmapped diagnostics."
    ),
    "inputs": [
        "Data_Center/Data/1_Pipeline/Trades/Live/*/closed_trades.db",
        "Data_Center/Data/1_Pipeline/Strategy/Strategy_EA/**/*.mq5",
        "Data_Center/Data/1_Pipeline/Strategy/Strategy_Profile/strategy_registry.csv",
    ],
    "outputs": [
        "Data_Center/Data/2_Baseline/Trades/Live/*/*/trades.db",
        "Data_Center/Data/2_Baseline/Trades/Live/*/unmapped_trades.csv",
        "Data_Center/Data/2_Baseline/Trades/Live/*/unmapped_summary.csv",
        "Data_Center/Data/2_Baseline/Trades/Live/*/not_exportable_trades.csv",
        "Data_Center/Data/2_Baseline/Trades/Live/*/not_exportable_summary.csv",
    ],
    "upstream_data": [
        "Data_Center/Data/1_Pipeline/Trades/Live",
    ],
    "downstream_data": [
        "Dashboard/Building_Blocks/Trades/Live",
        "Data_Center/Data/3_Research",
    ],
    "dependencies": [
        "pathlib",
        "sqlite3",
        "pandas",
        "datetime",
        "re",
        "json",
        "shutil",
    ],
    "schedule": "manual_or_loop",
    "version": "v2.3.0",
    "last_reviewed": "2026-06-02",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "baseline_trades",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-02",
    "tags": [
        "live_trades",
        "strategy_mapping",
        "master_inference",
        "sl_tp_inference",
        "unmapped_diagnostics",
        "both_side_strategies",
        "canonical_magic",
        "sqlite",
    ],
    "notes": [
        "Direct Strategy comments are HIGH confidence.",
        "EA mappings are HIGH confidence.",
        "[sl]/[tp] comments are inferred only when account-wide mapping keys are unique.",
        "Invalid Strategy IDs like 0.47617 are ignored.",
        "Bad magic values 0 and 11111 are never used as final bucket magic.",
        "All unmapped clusters are printed in terminal and exported to CSV.",
    ],
}


# ============================================================
# ROOT / PATHS
# ============================================================

def find_quant_root(start: Path) -> Path:
    cur = start.resolve()

    for p in [cur] + list(cur.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            dc = p / "Data_Center"
            if (dc / "Data").exists() and (dc / "Backend_Management").exists():
                return p

    raise RuntimeError(
        "QUANT Root nicht gefunden. Erwartet Ordner mit Dashboard und Data_Center. "
        f"Start={start}"
    )


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
BACKEND_MANAGEMENT_DIR = DATA_CENTER_DIR / "Backend_Management"

PIPELINE_ROOT = DATA_DIR / "1_Pipeline"
INPUT_LIVE_ROOT = PIPELINE_ROOT / "Trades" / "Live"
OUTPUT_LIVE_ROOT = DATA_DIR / "2_Baseline" / "Trades" / "Live"

STRATEGY_ROOT = PIPELINE_ROOT / "Strategy"
STRATEGY_EA_ROOT_PRIMARY = STRATEGY_ROOT / "Strategy_EA"
STRATEGY_EA_ROOT_ALT = PIPELINE_ROOT / "Strategy_EA"
REGISTRY_FILE = STRATEGY_ROOT / "Strategy_Profile" / "strategy_registry.csv"

PROCESSOR_ROOT = BACKEND_MANAGEMENT_DIR / "2_Baseline" / "Trades"
RUNTIME_DIR = PROCESSOR_ROOT / "runtime"
STATE_FILE = RUNTIME_DIR / "live_strategy_processor_state.json"


# ============================================================
# CONFIG
# ============================================================

RUN_FOREVER = False
POLL_SECONDS = 30.0

START_EQUITY = 100000.0
# Trades before this close_time_utc are ignored.
# Same logic as the older processor:
#   close_time_utc < 2026-02-22 00:00:00 UTC  -> ignored
#   close_time_utc >= 2026-02-22 00:00:00 UTC -> processed
GLOBAL_CUTOFF_DATE_UTC: Optional[pd.Timestamp] = pd.Timestamp("2026-02-22 00:00:00", tz="UTC")

INPUT_DB_FILENAME = "closed_trades.db"
INPUT_TABLE_NAME = "closed_trades"
OUTPUT_DB_FILENAME = "trades.db"

CLEAN_OUTPUT_ACCOUNT_DIR_BEFORE_EXPORT = True

BAD_MAGIC_VALUES = {0, 11111}

MIN_DIRECT_TRADES_FOR_KEY = 1
ALLOW_POSITION_ID_INFERENCE = True
ALLOW_SYMBOL_MAGIC_INFERENCE = True
ALLOW_SYMBOL_MAGIC_DIRECTION_INFERENCE = True
ALLOW_SYMBOL_DIRECTION_INFERENCE = True
ALLOW_SYMBOL_ONLY_INFERENCE = False
MIN_DIRECT_TRADES_FOR_SYMBOL_DIRECTION = 1

PRINT_UNMAPPED_FULL = True
PRINT_UNMAPPED_EXAMPLES_PER_CLUSTER = 20
PRINT_NOT_EXPORTABLE_FULL = True
PRINT_NOT_EXPORTABLE_EXAMPLES_PER_CLUSTER = 10

TRADES_REQUIRED_COLS = [
    "account_id",
    "position_id",
    "symbol",
    "direction",
    "open_time_utc",
    "close_time_utc",
    "entry_price",
    "exit_price",
    "price_delta",
    "volume_in",
    "volume_out",
    "profit_sum",
    "swap_sum",
    "commission_sum",
    "net_sum",
    "magic",
    "comment_last",
    "close_ticket",
]

TRADES_OPTIONAL_FINAL_COLS = [
    "sl",
    "tp",
    "best_floating_pnl",
    "worst_floating_pnl",
    "best_price_seen_live",
    "worst_price_seen_live",
    "max_favorable_points_live",
    "max_adverse_points_live",
    "account_type",
    "server",
]

TRADES_OUTPUT_COLS = TRADES_REQUIRED_COLS + TRADES_OPTIONAL_FINAL_COLS

_STRAT_DOT_RE = re.compile(r"\bStrategy\s+([0-9]+)\.([0-9]+)\.([0-9]+)\b", re.IGNORECASE)
_STRAT_UNDERSCORE_RE = re.compile(r"(?:WF_Matrix_)?Strategy_([0-9]+)_([0-9]+)_([0-9]+)\b", re.IGNORECASE)

_EA_FILENAME_RE = re.compile(
    r"^(?P<symbol>.+?)_(?P<magic>\d+)_(?P<strategy_id>.+?)_(?P<side>BUY|SELL|BOTH)_(?P<timeframe>[A-Z0-9]+)\.mq5$",
    re.IGNORECASE,
)

KNOWN_BASE_SYMBOLS = [
    "AUDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDJPY", "USDJPY",
    "EURGBP", "AUDNZD",
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    "XAUUSD", "XAGUSD",
    "USOIL", "UKOIL",
    "US500", "US100", "US30", "GER40", "DAX", "EU50", "US2000",
    "BTCUSD", "ETHUSD",
]

BROKER_SUFFIXES = [
    ".PRO", ".CASH", ".RAW", ".ECN", ".R", ".M", ".I", ".A", ".B",
    "PRO", "CASH", "RAW", "ECN",
]


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: object) -> str:
    s = str(value).strip()
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._ ")
    return (s[:200] if len(s) > 200 else s) or "UNKNOWN"


def atomic_write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(prefix=path.stem + "_", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def is_missing_text(x: Any) -> bool:
    return str(x).strip() in {"", "nan", "None", "NaN", "<NA>"}


def is_valid_strategy_id(strategy_id: object) -> bool:
    s = str(strategy_id).strip()
    m = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)", s)
    if not m:
        return False
    if int(m.group(1)) == 0:
        return False
    return True


def parse_strategy_from_comment(comment: object) -> Optional[str]:
    text = str(comment).strip()

    m = _STRAT_DOT_RE.search(text)
    if m:
        sid = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        return sid if is_valid_strategy_id(sid) else None

    m = _STRAT_UNDERSCORE_RE.search(text)
    if m:
        sid = f"{m.group(1)}.{m.group(2)}.{m.group(3)}"
        return sid if is_valid_strategy_id(sid) else None

    return None


def normalize_direction(value: object) -> str:
    x = str(value).strip().upper()
    if x in {"BUY", "LONG", "B", "0"}:
        return "BUY"
    if x in {"SELL", "SHORT", "S", "1"}:
        return "SELL"
    return x


def normalize_symbol(value: object) -> str:
    raw = str(value).strip().upper()
    if not raw:
        return ""

    s = raw.replace("/", "").replace("-", "").replace("_", "")

    for suffix in BROKER_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]

    s_compact = re.sub(r"[^A-Z0-9]", "", s)

    for base in sorted(KNOWN_BASE_SYMBOLS, key=len, reverse=True):
        base_compact = re.sub(r"[^A-Z0-9]", "", base.upper())
        if s_compact == base_compact:
            return base_compact
        if s_compact.startswith(base_compact):
            return base_compact

    letters = re.sub(r"[^A-Z]", "", s_compact)
    if len(letters) >= 6:
        return letters[:6]
    return letters


def parse_time_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def safe_optional_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def safe_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def ts_to_str(ts: Optional[pd.Timestamp]) -> str:
    if ts is None or pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def apply_global_cutoff_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)

    if GLOBAL_CUTOFF_DATE_UTC is None:
        print("[DATE_FILTER] disabled | processing full history")
        return df.reset_index(drop=True)

    d = df.copy()
    before = len(d)
    d["close_time_utc"] = pd.to_datetime(d["close_time_utc"], errors="coerce", utc=True)
    d = d.dropna(subset=["close_time_utc"])
    after_dropna = len(d)
    d = d[d["close_time_utc"] >= GLOBAL_CUTOFF_DATE_UTC].copy()
    after_filter = len(d)

    print(
        "[DATE_FILTER] "
        f"cutoff={GLOBAL_CUTOFF_DATE_UTC.isoformat()} "
        f"before={before} "
        f"after_dropna={after_dropna} "
        f"after_filter={after_filter} "
        f"removed={before - after_filter}"
    )

    return d.reset_index(drop=True)


def get_measurement_day_utc() -> pd.Timestamp:
    now_utc = datetime.now(timezone.utc)
    return pd.Timestamp(now_utc.date(), tz="UTC")


def get_strategy_effective_window(trades: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if trades.empty:
        return None, None
    close_times = pd.to_datetime(trades["close_time_utc"], errors="coerce", utc=True).dropna()
    if close_times.empty:
        return None, None
    return close_times.min(), close_times.max()


def build_bucket_date_range_for_measurement(
    trades: pd.DataFrame,
    measurement_day_utc: pd.Timestamp,
) -> Tuple[str, str]:
    start_ts, end_ts = get_strategy_effective_window(trades)
    if start_ts is None or pd.isna(start_ts):
        return "unknown", "unknown"
    if end_ts is None or pd.isna(end_ts):
        end_ts = measurement_day_utc

    start_day = pd.Timestamp(start_ts).tz_convert("UTC").normalize()
    end_day = pd.Timestamp(end_ts).tz_convert("UTC").normalize()

    return start_day.strftime("%Y-%m-%d"), end_day.strftime("%Y-%m-%d")


def bucket_folder_name(
    symbol_norm: str,
    canonical_magic: object,
    strategy_id: str,
    bucket_side: str,
    start_date: str,
    end_date: str,
) -> str:
    return (
        f"{sanitize_name(symbol_norm)}_"
        f"{sanitize_name(canonical_magic)}_"
        f"{sanitize_name(strategy_id)}_"
        f"{sanitize_name(bucket_side)}_"
        f"{sanitize_name(start_date)}_to_{sanitize_name(end_date)}"
    )


def tuple_key(row: pd.Series, cols: List[str]) -> Tuple[Any, ...]:
    return tuple(row.get(c) for c in cols)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================
# STATE
# ============================================================

@dataclass
class AccountState:
    account_id: int
    account_folder: str
    last_max_close_ticket: Optional[int] = None
    last_rows: int = 0
    last_mapped_rows: int = 0
    last_exportable_rows: int = 0
    last_unmapped_rows: int = 0
    last_not_exportable_rows: int = 0
    last_exported_buckets: int = 0
    last_updated_utc: Optional[str] = None


def load_state() -> Dict[str, AccountState]:
    ensure_dir(RUNTIME_DIR)
    if not STATE_FILE.exists():
        return {}

    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, AccountState] = {}
    for key, payload in raw.items():
        try:
            out[key] = AccountState(**payload)
        except Exception:
            pass
    return out


def save_state(state: Dict[str, AccountState]) -> None:
    atomic_write_json({k: asdict(v) for k, v in state.items()}, STATE_FILE)


# ============================================================
# EA / REGISTRY
# ============================================================

def discover_ea_files() -> List[Path]:
    roots = [STRATEGY_EA_ROOT_PRIMARY, STRATEGY_EA_ROOT_ALT, PIPELINE_ROOT]
    seen: Set[Path] = set()
    files: List[Path] = []

    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.mq5"):
            try:
                rp = p.resolve()
            except Exception:
                rp = p
            if rp in seen:
                continue
            seen.add(rp)
            files.append(p)

    files.sort(key=lambda x: str(x).lower())
    return files


def build_strategy_ea_mapping() -> pd.DataFrame:
    columns = [
        "strategy_id",
        "symbol",
        "symbol_norm",
        "magic",
        "match_direction",
        "strategy_side",
        "bucket_side",
        "timeframe",
        "ea_filename",
        "ea_path",
    ]

    rows: List[Dict[str, object]] = []

    for p in discover_ea_files():
        m = _EA_FILENAME_RE.match(p.name)
        if not m:
            continue

        symbol = str(m.group("symbol")).strip()
        magic = int(m.group("magic"))
        strategy_id = str(m.group("strategy_id")).strip()
        side = str(m.group("side")).strip().upper()
        timeframe = str(m.group("timeframe")).strip().upper()

        if not is_valid_strategy_id(strategy_id):
            continue

        symbol_norm = normalize_symbol(symbol)

        if side == "BOTH":
            for match_direction in ["BUY", "SELL"]:
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "symbol": symbol,
                        "symbol_norm": symbol_norm,
                        "magic": magic,
                        "match_direction": match_direction,
                        "strategy_side": "BOTH",
                        "bucket_side": "BOTH",
                        "timeframe": timeframe,
                        "ea_filename": p.name,
                        "ea_path": str(p),
                    }
                )
        else:
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "symbol_norm": symbol_norm,
                    "magic": magic,
                    "match_direction": side,
                    "strategy_side": side,
                    "bucket_side": side,
                    "timeframe": timeframe,
                    "ea_filename": p.name,
                    "ea_path": str(p),
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df["magic"] = pd.to_numeric(df["magic"], errors="coerce").astype("Int64")

    for c in ["strategy_id", "symbol", "symbol_norm", "match_direction", "strategy_side", "bucket_side", "timeframe", "ea_filename", "ea_path"]:
        df[c] = df[c].astype(str).fillna("").str.strip()

    df["symbol_norm"] = df["symbol_norm"].str.upper()
    df["match_direction"] = df["match_direction"].str.upper()
    df["strategy_side"] = df["strategy_side"].str.upper()
    df["bucket_side"] = df["bucket_side"].str.upper()

    return df[columns].reset_index(drop=True)


def load_registry(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["magic", "strategy_id"])

    try:
        reg = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["magic", "strategy_id"])

    if "magic" not in reg.columns or "strategy_id" not in reg.columns:
        return pd.DataFrame(columns=["magic", "strategy_id"])

    reg = reg.copy()
    reg["magic"] = pd.to_numeric(reg["magic"], errors="coerce").astype("Int64")
    reg["strategy_id"] = reg["strategy_id"].astype(str).fillna("").str.strip()
    reg = reg[reg["strategy_id"].apply(is_valid_strategy_id)].copy()

    return reg[["magic", "strategy_id"]].reset_index(drop=True)


def build_registry_unique_lookup(reg: pd.DataFrame) -> Dict[int, str]:
    if reg.empty:
        return {}

    tmp = reg.dropna(subset=["magic"]).copy()
    tmp["magic_int"] = tmp["magic"].astype(int)

    out: Dict[int, str] = {}
    for magic, g in tmp.groupby("magic_int"):
        ids = sorted(set(g["strategy_id"].astype(str).tolist()))
        if len(ids) == 1:
            out[int(magic)] = ids[0]
    return out


# ============================================================
# SQLITE INPUT
# ============================================================

def read_closed_trades_db(path: Path, table_name: str = INPUT_TABLE_NAME) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=TRADES_OUTPUT_COLS)

    conn = sqlite3.connect(path)
    try:
        table_info = pd.read_sql_query(f"PRAGMA table_info({table_name})", conn)
        existing_cols = set(table_info["name"].astype(str).tolist()) if not table_info.empty else set()

        missing_required = [c for c in TRADES_REQUIRED_COLS if c not in existing_cols]
        if missing_required:
            raise ValueError(f"Missing required columns in {path}:{table_name}: {missing_required}")

        selected_cols = TRADES_REQUIRED_COLS + [c for c in TRADES_OPTIONAL_FINAL_COLS if c in existing_cols]
        select_sql = ", ".join([f'"{c}"' for c in selected_cols])
        df = pd.read_sql_query(f'SELECT {select_sql} FROM "{table_name}"', conn)

    finally:
        conn.close()

    for c in TRADES_OPTIONAL_FINAL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    df = df[TRADES_OUTPUT_COLS].copy()

    for c in ["account_id", "position_id", "magic", "close_ticket"]:
        df[c] = safe_int_series(df[c])

    for c in [
        "entry_price", "exit_price", "price_delta", "volume_in", "volume_out",
        "profit_sum", "swap_sum", "commission_sum", "net_sum",
    ]:
        df[c] = safe_float_series(df[c])

    for c in [
        "sl", "tp", "best_floating_pnl", "worst_floating_pnl", "best_price_seen_live",
        "worst_price_seen_live", "max_favorable_points_live", "max_adverse_points_live",
    ]:
        if c in df.columns:
            df[c] = safe_optional_float_series(df[c])

    df["open_time_utc"] = parse_time_utc(df["open_time_utc"])
    df["close_time_utc"] = parse_time_utc(df["close_time_utc"])

    for c in ["symbol", "direction", "comment_last", "account_type", "server"]:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("").str.strip()

    df["direction"] = df["direction"].apply(normalize_direction)
    df["symbol_norm"] = df["symbol"].apply(normalize_symbol)
    df["strategy_id_from_comment"] = df["comment_last"].apply(parse_strategy_from_comment)
    df["raw_magic_is_bad"] = pd.to_numeric(df["magic"], errors="coerce").fillna(-999999).astype(int).isin(BAD_MAGIC_VALUES)

    df = df.dropna(subset=["account_id", "position_id", "close_ticket", "close_time_utc"]).copy()
    df = df.sort_values(["close_time_utc", "position_id", "close_ticket"]).reset_index(drop=True)

    return df


# ============================================================
# STRATEGY DIRECT MAPPING
# ============================================================

def initialize_strategy_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in [
        "strategy_id", "strategy_side", "bucket_side", "timeframe", "ea_filename",
        "strategy_mapping_source", "strategy_mapping_confidence", "strategy_mapping_reason",
    ]:
        d[c] = ""
    d["inference_sample_trades"] = 0
    d["inference_key"] = ""
    return d


def _choose_ea_candidate(candidates: pd.DataFrame, comment_strategy_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if candidates.empty:
        return None

    c = candidates.copy()

    if comment_strategy_id and is_valid_strategy_id(comment_strategy_id):
        by_comment = c[c["strategy_id"].astype(str) == comment_strategy_id].copy()
        if not by_comment.empty:
            c = by_comment

    both = c[c["bucket_side"].astype(str).str.upper() == "BOTH"].copy()
    if not both.empty:
        uniq = both[["strategy_id", "magic", "bucket_side", "strategy_side", "timeframe", "ea_filename"]].drop_duplicates()
        core = uniq[["strategy_id", "magic", "bucket_side"]].drop_duplicates()
        if len(core) == 1:
            return uniq.iloc[0].to_dict()
        return None

    uniq = c[["strategy_id", "magic", "bucket_side", "strategy_side", "timeframe", "ea_filename"]].drop_duplicates()
    core = uniq[["strategy_id", "magic", "bucket_side"]].drop_duplicates()
    if len(core) == 1:
        return uniq.iloc[0].to_dict()

    return None


def direct_ea_mapping_for_trade(row: pd.Series, ea_map: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if ea_map.empty:
        return None

    try:
        magic = int(row["magic"])
    except Exception:
        return None

    symbol_norm = str(row.get("symbol_norm", "")).upper()
    direction = str(row.get("direction", "")).upper()
    comment_sid = row.get("strategy_id_from_comment")

    candidates = ea_map[
        (ea_map["symbol_norm"].astype(str).str.upper() == symbol_norm)
        & (ea_map["magic"].astype("Int64") == magic)
        & (ea_map["match_direction"].astype(str).str.upper() == direction)
    ].copy()

    chosen = _choose_ea_candidate(candidates, comment_sid)
    if chosen is not None:
        return {
            "strategy_id": str(chosen["strategy_id"]),
            "strategy_side": str(chosen.get("strategy_side", "")),
            "bucket_side": str(chosen.get("bucket_side", "")),
            "timeframe": str(chosen.get("timeframe", "")),
            "ea_filename": str(chosen.get("ea_filename", "")),
            "strategy_mapping_source": "ea_symbol_magic_direction",
            "strategy_mapping_confidence": "HIGH",
            "strategy_mapping_reason": "exact_ea_match",
        }

    candidates = ea_map[
        (ea_map["symbol_norm"].astype(str).str.upper() == symbol_norm)
        & (ea_map["magic"].astype("Int64") == magic)
    ].copy()
    both = candidates[candidates["bucket_side"].astype(str).str.upper() == "BOTH"].copy()

    chosen = _choose_ea_candidate(both, comment_sid)
    if chosen is not None:
        return {
            "strategy_id": str(chosen["strategy_id"]),
            "strategy_side": "BOTH",
            "bucket_side": "BOTH",
            "timeframe": str(chosen.get("timeframe", "")),
            "ea_filename": str(chosen.get("ea_filename", "")),
            "strategy_mapping_source": "ea_symbol_magic_both",
            "strategy_mapping_confidence": "HIGH",
            "strategy_mapping_reason": "ea_both_match",
        }

    return None


def apply_direct_mappings(df: pd.DataFrame, ea_map: pd.DataFrame, registry_lookup: Dict[int, str]) -> pd.DataFrame:
    d = initialize_strategy_columns(df)

    for idx, row in d.iterrows():
        mapped = direct_ea_mapping_for_trade(row, ea_map)
        if mapped is not None:
            for k, v in mapped.items():
                d.at[idx, k] = v
            continue

        sid = row.get("strategy_id_from_comment")
        if sid and is_valid_strategy_id(sid):
            direction = str(row.get("direction", "")).upper()
            d.at[idx, "strategy_id"] = sid
            d.at[idx, "strategy_side"] = direction
            d.at[idx, "bucket_side"] = direction
            d.at[idx, "strategy_mapping_source"] = "comment_last"
            d.at[idx, "strategy_mapping_confidence"] = "HIGH"
            d.at[idx, "strategy_mapping_reason"] = "valid_strategy_id_in_comment"
            continue

        try:
            magic = int(row.get("magic"))
        except Exception:
            magic = None

        if magic is not None and magic in registry_lookup:
            sid = registry_lookup[magic]
            if is_valid_strategy_id(sid):
                direction = str(row.get("direction", "")).upper()
                d.at[idx, "strategy_id"] = sid
                d.at[idx, "strategy_side"] = direction
                d.at[idx, "bucket_side"] = direction
                d.at[idx, "strategy_mapping_source"] = "strategy_registry_unique"
                d.at[idx, "strategy_mapping_confidence"] = "MEDIUM"
                d.at[idx, "strategy_mapping_reason"] = "unique_registry_magic_lookup"
                continue

    return d


# ============================================================
# MASTER INFERENCE
# ============================================================

def _valid_mapped_base(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["strategy_id"].apply(is_valid_strategy_id)].copy()


def _mode_valid_magic(g: pd.DataFrame) -> Optional[int]:
    magics = pd.to_numeric(g["magic"], errors="coerce").dropna().astype(int)
    valid = magics[~magics.isin(BAD_MAGIC_VALUES)]
    if valid.empty:
        return None
    counts = valid.value_counts()
    return int(counts.index[0])


def _unique_strategy_from_group(g: pd.DataFrame) -> Optional[str]:
    ids = sorted(set(x for x in g["strategy_id"].astype(str).tolist() if is_valid_strategy_id(x)))
    if len(ids) == 1:
        return ids[0]
    return None


def _side_from_group(g: pd.DataFrame) -> str:
    sides = set(g["bucket_side"].astype(str).str.upper().tolist())
    directions = set(g["direction"].astype(str).str.upper().tolist())
    if "BOTH" in sides:
        return "BOTH"
    if {"BUY", "SELL"}.issubset(directions):
        return "BOTH"
    if len(directions) == 1:
        return next(iter(directions))
    return "BOTH"


def build_unique_lookup(
    base: pd.DataFrame,
    key_cols: List[str],
    source_name: str,
    confidence: str,
    min_samples: int,
    require_valid_magic_sample: bool,
) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    out: Dict[Tuple[Any, ...], Dict[str, Any]] = {}

    if base.empty:
        return out

    existing_key_cols = [c for c in key_cols if c in base.columns]
    if len(existing_key_cols) != len(key_cols):
        return out

    for key, g in base.groupby(key_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)

        if len(g) < min_samples:
            continue

        strategy_id = _unique_strategy_from_group(g)
        if strategy_id is None:
            continue

        canonical_magic = _mode_valid_magic(g)

        if require_valid_magic_sample and canonical_magic is None:
            continue

        side = _side_from_group(g)

        timeframes = sorted(set(x for x in g.get("timeframe", pd.Series(dtype=str)).astype(str).tolist() if x))
        ea_files = sorted(set(x for x in g.get("ea_filename", pd.Series(dtype=str)).astype(str).tolist() if x))

        out[key] = {
            "strategy_id": strategy_id,
            "strategy_side": side,
            "bucket_side": side,
            "timeframe": ",".join(timeframes),
            "ea_filename": ",".join(ea_files),
            "strategy_mapping_source": source_name,
            "strategy_mapping_confidence": confidence,
            "strategy_mapping_reason": f"unique_master_lookup_{'_'.join(key_cols)}",
            "inference_sample_trades": int(len(g)),
            "inference_key": "|".join(str(x) for x in key),
            "inference_preferred_magic": canonical_magic,
        }

    return out


def build_master_inference_lookups(df: pd.DataFrame) -> List[Tuple[List[str], Dict[Tuple[Any, ...], Dict[str, Any]]]]:
    base = _valid_mapped_base(df)
    lookups: List[Tuple[List[str], Dict[Tuple[Any, ...], Dict[str, Any]]]] = []

    if base.empty:
        return lookups

    if ALLOW_POSITION_ID_INFERENCE:
        lookups.append((
            ["position_id"],
            build_unique_lookup(
                base=base,
                key_cols=["position_id"],
                source_name="inference_position_id",
                confidence="HIGH",
                min_samples=1,
                require_valid_magic_sample=False,
            ),
        ))

    if ALLOW_SYMBOL_MAGIC_INFERENCE:
        lookups.append((
            ["symbol_norm", "magic"],
            build_unique_lookup(
                base=base,
                key_cols=["symbol_norm", "magic"],
                source_name="inference_symbol_magic",
                confidence="HIGH",
                min_samples=MIN_DIRECT_TRADES_FOR_KEY,
                require_valid_magic_sample=True,
            ),
        ))

    if ALLOW_SYMBOL_MAGIC_DIRECTION_INFERENCE:
        lookups.append((
            ["symbol_norm", "magic", "direction"],
            build_unique_lookup(
                base=base,
                key_cols=["symbol_norm", "magic", "direction"],
                source_name="inference_symbol_magic_direction",
                confidence="MEDIUM_HIGH",
                min_samples=MIN_DIRECT_TRADES_FOR_KEY,
                require_valid_magic_sample=True,
            ),
        ))

    if ALLOW_SYMBOL_DIRECTION_INFERENCE:
        lookups.append((
            ["symbol_norm", "direction"],
            build_unique_lookup(
                base=base,
                key_cols=["symbol_norm", "direction"],
                source_name="inference_symbol_direction",
                confidence="LOW_MEDIUM",
                min_samples=MIN_DIRECT_TRADES_FOR_SYMBOL_DIRECTION,
                require_valid_magic_sample=True,
            ),
        ))

    if ALLOW_SYMBOL_ONLY_INFERENCE:
        lookups.append((
            ["symbol_norm"],
            build_unique_lookup(
                base=base,
                key_cols=["symbol_norm"],
                source_name="inference_symbol_only",
                confidence="LOW",
                min_samples=MIN_DIRECT_TRADES_FOR_KEY,
                require_valid_magic_sample=True,
            ),
        ))

    return lookups


def apply_master_inference_once(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    d = df.copy()
    lookups = build_master_inference_lookups(d)
    newly_mapped = 0

    for idx, row in d.iterrows():
        if is_valid_strategy_id(row.get("strategy_id", "")):
            continue

        for key_cols, lookup in lookups:
            key = tuple_key(row, key_cols)
            mapping = lookup.get(key)
            if mapping is None:
                continue

            for col in [
                "strategy_id",
                "strategy_side",
                "bucket_side",
                "timeframe",
                "ea_filename",
                "strategy_mapping_source",
                "strategy_mapping_confidence",
                "strategy_mapping_reason",
            ]:
                d.at[idx, col] = mapping.get(col, "")

            d.at[idx, "inference_sample_trades"] = int(mapping.get("inference_sample_trades", 0))
            d.at[idx, "inference_key"] = mapping.get("inference_key", "")
            newly_mapped += 1
            break

    return d, newly_mapped


def apply_master_inference_until_stable(df: pd.DataFrame, max_passes: int = 3) -> pd.DataFrame:
    d = df.copy()

    for pass_no in range(1, max_passes + 1):
        d, n = apply_master_inference_once(d)
        print(f"[INFERENCE_PASS] pass={pass_no} newly_mapped={n}")
        if n == 0:
            break

    return d


def apply_canonical_magic_and_auto_both(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["canonical_magic"] = pd.NA
    d["canonical_bucket"] = ""
    d["bucket_side_auto_reason"] = ""
    d["is_exportable"] = False
    d["not_exportable_reason"] = ""

    mapped_mask = d["strategy_id"].apply(is_valid_strategy_id)

    if not mapped_mask.any():
        d.loc[:, "not_exportable_reason"] = "no_valid_strategy_id"
        return d

    for (symbol_norm, strategy_id), idx in d[mapped_mask].groupby(["symbol_norm", "strategy_id"], dropna=False).groups.items():
        g = d.loc[idx].copy()

        if not is_valid_strategy_id(strategy_id):
            d.loc[idx, "not_exportable_reason"] = "invalid_strategy_id"
            continue

        canonical_magic = _mode_valid_magic(g)
        if canonical_magic is None or int(canonical_magic) in BAD_MAGIC_VALUES:
            d.loc[idx, "not_exportable_reason"] = "no_valid_canonical_magic"
            continue

        directions = set(g["direction"].astype(str).str.upper().tolist())
        existing_sides = set(g["bucket_side"].astype(str).str.upper().tolist())

        if "BOTH" in existing_sides or {"BUY", "SELL"}.issubset(directions):
            final_side = "BOTH"
        elif len(directions) == 1:
            final_side = next(iter(directions))
        else:
            final_side = "BOTH"

        d.loc[idx, "canonical_magic"] = int(canonical_magic)
        d.loc[idx, "bucket_side"] = final_side
        d.loc[idx, "canonical_bucket"] = f"{symbol_norm}_{int(canonical_magic)}_{strategy_id}_{final_side}"
        d.loc[idx, "is_exportable"] = True

        if "BOTH" in existing_sides:
            d.loc[idx, "bucket_side_auto_reason"] = "ea_or_existing_both"
        elif final_side == "BOTH":
            d.loc[idx, "bucket_side_auto_reason"] = "auto_both_buy_sell_same_symbol_strategy"
        else:
            d.loc[idx, "bucket_side_auto_reason"] = "single_direction"

    d.loc[~mapped_mask, "not_exportable_reason"] = "no_valid_strategy_id"

    d["canonical_magic"] = pd.to_numeric(d["canonical_magic"], errors="coerce").astype("Int64")
    d["is_exportable"] = d["is_exportable"].astype(bool)

    return d


def enrich_strategy_ids(df: pd.DataFrame, reg: pd.DataFrame, ea_map: pd.DataFrame) -> pd.DataFrame:
    registry_lookup = build_registry_unique_lookup(reg)

    d = apply_direct_mappings(df, ea_map, registry_lookup)

    before = int(d["strategy_id"].apply(is_valid_strategy_id).sum())
    print(f"[DIRECT_MAPPING] mapped_rows={before}/{len(d)}")

    d = apply_master_inference_until_stable(d, max_passes=3)

    after = int(d["strategy_id"].apply(is_valid_strategy_id).sum())
    print(f"[FINAL_MAPPING_BEFORE_EXPORT_RULES] mapped_rows={after}/{len(d)} inferred_added={after - before}")

    d = apply_canonical_magic_and_auto_both(d)

    for c in [
        "strategy_id", "strategy_side", "bucket_side", "timeframe", "ea_filename",
        "strategy_mapping_source", "strategy_mapping_confidence", "strategy_mapping_reason",
        "canonical_bucket", "bucket_side_auto_reason", "not_exportable_reason", "inference_key",
    ]:
        if c not in d.columns:
            d[c] = ""
        d[c] = d[c].astype(str).replace(["nan", "None", "<NA>"], "").str.strip()

    return d


# ============================================================
# PERFORMANCE
# ============================================================

def bucket_trades(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["close_time_utc", "position_id", "close_ticket"]).reset_index(drop=True)


def max_drawdown_from_pnl(pnl: pd.Series, start_equity: float) -> Tuple[float, float]:
    if pnl.empty:
        return 0.0, 0.0
    equity = float(start_equity) + pnl.astype(float).cumsum()
    peak = equity.cummax()
    dd_money = equity - peak
    dd_pct = (equity / peak - 1.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    return float(dd_money.min()), float(dd_pct.min() * 100.0)


def kpis_from_trades(trades: pd.DataFrame, measurement_day_utc: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    net = trades["net_sum"].astype(float)
    wins = int((net > 0).sum())
    losses = int((net < 0).sum())
    n = int(len(trades))

    gross_profit = float(net[net > 0].sum()) if wins > 0 else 0.0
    gross_loss = float(net[net < 0].sum()) if losses > 0 else 0.0
    total_net = float(net.sum())
    avg_trade = float(net.mean()) if n > 0 else 0.0
    median_trade = float(net.median()) if n > 0 else 0.0
    win_rate = float(wins / n) if n > 0 else 0.0
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 else None
    payoff_ratio = (
        float(net[net > 0].mean() / abs(net[net < 0].mean()))
        if wins > 0 and losses > 0 and abs(net[net < 0].mean()) > 0
        else None
    )
    max_dd_money, max_dd_pct = max_drawdown_from_pnl(net, START_EQUITY)

    first_open_time = pd.to_datetime(trades["open_time_utc"], utc=True, errors="coerce").min()
    first_close_time = pd.to_datetime(trades["close_time_utc"], utc=True, errors="coerce").min()
    last_close_time = pd.to_datetime(trades["close_time_utc"], utc=True, errors="coerce").max()
    strategy_live_start_utc, strategy_last_trade_close_utc = get_strategy_effective_window(trades)

    direction_counts = trades["direction"].value_counts().to_dict()
    mapping_source = ",".join(sorted(set(trades["strategy_mapping_source"].astype(str).tolist())))
    mapping_confidence = ",".join(sorted(set(trades["strategy_mapping_confidence"].astype(str).tolist())))

    return pd.DataFrame(
        [
            {
                "base_cutoff_utc": ts_to_str(GLOBAL_CUTOFF_DATE_UTC),
                "measurement_day_utc": ts_to_str(measurement_day_utc),
                "strategy_live_start_utc": ts_to_str(strategy_live_start_utc if strategy_live_start_utc is not None else first_close_time),
                "strategy_live_end_utc": ts_to_str(strategy_live_start_utc if False else strategy_last_trade_close_utc),
                "strategy_last_trade_close_utc": ts_to_str(strategy_last_trade_close_utc if strategy_last_trade_close_utc is not None else last_close_time),
                "first_open_time_utc": ts_to_str(first_open_time) if pd.notna(first_open_time) else "",
                "first_close_time_utc": ts_to_str(first_close_time) if pd.notna(first_close_time) else "",
                "last_close_time_utc": ts_to_str(last_close_time) if pd.notna(last_close_time) else "",
                "account_id": str(trades["account_id"].iloc[0]) if "account_id" in trades.columns else "",
                "symbol_norm": str(trades["symbol_norm"].iloc[0]),
                "strategy_id": str(trades["strategy_id"].iloc[0]),
                "canonical_magic": str(trades["canonical_magic"].iloc[0]),
                "bucket_side": str(trades["bucket_side"].iloc[0]),
                "canonical_bucket": str(trades["canonical_bucket"].iloc[0]),
                "mapping_source": mapping_source,
                "mapping_confidence": mapping_confidence,
                "buy_trades": int(direction_counts.get("BUY", 0)),
                "sell_trades": int(direction_counts.get("SELL", 0)),
                "n_trades": n,
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "net_pnl": total_net,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": profit_factor if profit_factor is not None else "",
                "avg_trade": avg_trade,
                "median_trade": median_trade,
                "payoff_ratio": payoff_ratio if payoff_ratio is not None else "",
                "max_drawdown_closed_money": max_dd_money,
                "max_drawdown_closed_pct": max_dd_pct,
                "swap_sum": float(trades["swap_sum"].sum()),
                "commission_sum": float(trades["commission_sum"].sum()),
                "volume_in_sum": float(trades["volume_in"].sum()),
                "volume_out_sum": float(trades["volume_out"].sum()),
                "raw_bad_magic_rows": int(trades["raw_magic_is_bad"].sum()) if "raw_magic_is_bad" in trades.columns else 0,
                "inferred_rows": int(trades["strategy_mapping_source"].astype(str).str.startswith("inference").sum()),
                "direct_comment_rows": int((trades["strategy_mapping_source"].astype(str) == "comment_last").sum()),
                "ea_rows": int(trades["strategy_mapping_source"].astype(str).str.startswith("ea_").sum()),
            }
        ]
    )


def perf_from_trades(trades: pd.DataFrame, freq: str, start_equity: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    d = trades.copy()
    d["close_time_utc"] = pd.to_datetime(d["close_time_utc"], utc=True, errors="coerce")
    d = d.dropna(subset=["close_time_utc"]).sort_values("close_time_utc")
    d = d.set_index("close_time_utc")

    grouped = d.resample(freq).agg(
        pnl_money=("net_sum", "sum"),
        n_trades=("net_sum", "size"),
        wins=("net_sum", lambda x: int((x > 0).sum())),
        losses=("net_sum", lambda x: int((x < 0).sum())),
        gross_profit=("net_sum", lambda x: float(x[x > 0].sum())),
        gross_loss=("net_sum", lambda x: float(x[x < 0].sum())),
    )

    grouped["cum_pnl_money"] = grouped["pnl_money"].cumsum()
    grouped["nav"] = float(start_equity) + grouped["cum_pnl_money"]
    grouped["win_rate"] = grouped["wins"] / grouped["n_trades"].replace(0, pd.NA)
    grouped["profit_factor"] = grouped.apply(
        lambda r: float(r["gross_profit"] / abs(r["gross_loss"])) if r["gross_loss"] < 0 else pd.NA,
        axis=1,
    )

    out = grouped.reset_index().rename(columns={"close_time_utc": "date"})
    out["date"] = pd.to_datetime(out["date"], utc=True)
    return out


def mapping_diagnostics_from_trades(all_trades: pd.DataFrame) -> pd.DataFrame:
    if all_trades.empty:
        return pd.DataFrame()

    group_cols = [
        "symbol", "symbol_norm", "magic", "direction", "strategy_id",
        "canonical_magic", "bucket_side", "strategy_mapping_source",
        "strategy_mapping_confidence", "is_exportable", "not_exportable_reason",
    ]
    existing = [c for c in group_cols if c in all_trades.columns]

    return (
        all_trades.groupby(existing, dropna=False)
        .size()
        .reset_index(name="n_trades")
        .sort_values("n_trades", ascending=False)
        .reset_index(drop=True)
    )


def inference_master_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    base = _valid_mapped_base(df)
    rows: List[Dict[str, Any]] = []

    configs = [
        ("position_id", ["position_id"]),
        ("symbol_magic", ["symbol_norm", "magic"]),
        ("symbol_magic_direction", ["symbol_norm", "magic", "direction"]),
        ("symbol_direction", ["symbol_norm", "direction"]),
        ("symbol_only", ["symbol_norm"]),
    ]

    for name, cols in configs:
        if any(c not in base.columns for c in cols):
            continue

        for key, g in base.groupby(cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            ids = sorted(set(x for x in g["strategy_id"].astype(str).tolist() if is_valid_strategy_id(x)))
            rows.append(
                {
                    "lookup_name": name,
                    "key": "|".join(str(x) for x in key),
                    "n_trades": int(len(g)),
                    "n_strategy_ids": int(len(ids)),
                    "strategy_ids": ",".join(ids),
                    "unique": bool(len(ids) == 1),
                    "canonical_magic": _mode_valid_magic(g),
                    "first_close_time_utc": g["close_time_utc"].min() if "close_time_utc" in g.columns else "",
                    "last_close_time_utc": g["close_time_utc"].max() if "close_time_utc" in g.columns else "",
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["lookup_name", "n_trades"], ascending=[True, False]).reset_index(drop=True)


# ============================================================
# UNMAPPED / NOT EXPORTABLE DIAGNOSTICS
# ============================================================

def make_unmapped_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    return df[~df["strategy_id"].apply(is_valid_strategy_id)].copy()


def make_unmapped_summary(unmapped: pd.DataFrame) -> pd.DataFrame:
    if unmapped.empty:
        return pd.DataFrame()

    group_cols = ["symbol", "symbol_norm", "magic", "direction", "comment_last"]

    summary = (
        unmapped.groupby(group_cols, dropna=False)
        .agg(
            n=("close_ticket", "count"),
            first_close=("close_time_utc", "min"),
            last_close=("close_time_utc", "max"),
            position_ids=("position_id", "nunique"),
            tickets=("close_ticket", "nunique"),
            pnl_sum=("net_sum", "sum"),
            avg_pnl=("net_sum", "mean"),
            raw_bad_magic=("raw_magic_is_bad", "max"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    return summary


def make_not_exportable_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "is_exportable" not in df.columns:
        return pd.DataFrame()

    return df[df["strategy_id"].apply(is_valid_strategy_id) & (~df["is_exportable"].astype(bool))].copy()


def make_not_exportable_summary(not_exp: pd.DataFrame) -> pd.DataFrame:
    if not_exp.empty:
        return pd.DataFrame()

    group_cols = ["symbol", "symbol_norm", "magic", "direction", "strategy_id", "not_exportable_reason", "comment_last"]

    summary = (
        not_exp.groupby(group_cols, dropna=False)
        .agg(
            n=("close_ticket", "count"),
            first_close=("close_time_utc", "min"),
            last_close=("close_time_utc", "max"),
            position_ids=("position_id", "nunique"),
            tickets=("close_ticket", "nunique"),
            pnl_sum=("net_sum", "sum"),
            avg_pnl=("net_sum", "mean"),
            raw_bad_magic=("raw_magic_is_bad", "max"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    return summary


def print_unmapped_terminal(df: pd.DataFrame, account_name: str) -> None:
    unmapped = make_unmapped_trades(df)
    summary = make_unmapped_summary(unmapped)

    if unmapped.empty:
        print(f"[UNMAPPED_DETAIL] {account_name}: none")
        return

    print(f"[UNMAPPED_DETAIL] {account_name}: rows={len(unmapped)} clusters={len(summary)}")

    rows_iter = summary.iterrows() if PRINT_UNMAPPED_FULL else summary.head(50).iterrows()

    for cluster_idx, row in rows_iter:
        print(
            "[UNMAPPED_CLUSTER] "
            f"account={account_name} "
            f"cluster={cluster_idx + 1} "
            f"symbol={row['symbol']} "
            f"symbol_norm={row['symbol_norm']} "
            f"magic={row['magic']} "
            f"direction={row['direction']} "
            f"n={row['n']} "
            f"positions={row['position_ids']} "
            f"tickets={row['tickets']} "
            f"first_close={row['first_close']} "
            f"last_close={row['last_close']} "
            f"pnl_sum={row['pnl_sum']:.2f} "
            f"raw_bad_magic={row['raw_bad_magic']} "
            f"comment_last={row['comment_last']}"
        )

        examples = unmapped[
            (unmapped["symbol"].astype(str) == str(row["symbol"]))
            & (unmapped["symbol_norm"].astype(str) == str(row["symbol_norm"]))
            & (unmapped["magic"].astype(str) == str(row["magic"]))
            & (unmapped["direction"].astype(str) == str(row["direction"]))
            & (unmapped["comment_last"].astype(str) == str(row["comment_last"]))
        ].sort_values("close_time_utc").head(PRINT_UNMAPPED_EXAMPLES_PER_CLUSTER)

        for _, ex in examples.iterrows():
            print(
                "[UNMAPPED_EXAMPLE] "
                f"account={account_name} "
                f"ticket={ex.get('close_ticket')} "
                f"position={ex.get('position_id')} "
                f"symbol={ex.get('symbol')} "
                f"symbol_norm={ex.get('symbol_norm')} "
                f"magic={ex.get('magic')} "
                f"direction={ex.get('direction')} "
                f"close={ex.get('close_time_utc')} "
                f"net={float(ex.get('net_sum', 0.0)):.2f} "
                f"comment={ex.get('comment_last')}"
            )


def print_not_exportable_terminal(df: pd.DataFrame, account_name: str) -> None:
    not_exp = make_not_exportable_trades(df)
    summary = make_not_exportable_summary(not_exp)

    if not_exp.empty:
        print(f"[NOT_EXPORTABLE_DETAIL] {account_name}: none")
        return

    print(f"[NOT_EXPORTABLE_DETAIL] {account_name}: rows={len(not_exp)} clusters={len(summary)}")

    rows_iter = summary.iterrows() if PRINT_NOT_EXPORTABLE_FULL else summary.head(50).iterrows()

    for cluster_idx, row in rows_iter:
        print(
            "[NOT_EXPORTABLE_CLUSTER] "
            f"account={account_name} "
            f"cluster={cluster_idx + 1} "
            f"symbol={row['symbol']} "
            f"symbol_norm={row['symbol_norm']} "
            f"magic={row['magic']} "
            f"direction={row['direction']} "
            f"strategy_id={row['strategy_id']} "
            f"reason={row['not_exportable_reason']} "
            f"n={row['n']} "
            f"positions={row['position_ids']} "
            f"tickets={row['tickets']} "
            f"first_close={row['first_close']} "
            f"last_close={row['last_close']} "
            f"pnl_sum={row['pnl_sum']:.2f} "
            f"comment_last={row['comment_last']}"
        )

        examples = not_exp[
            (not_exp["symbol"].astype(str) == str(row["symbol"]))
            & (not_exp["symbol_norm"].astype(str) == str(row["symbol_norm"]))
            & (not_exp["magic"].astype(str) == str(row["magic"]))
            & (not_exp["direction"].astype(str) == str(row["direction"]))
            & (not_exp["strategy_id"].astype(str) == str(row["strategy_id"]))
            & (not_exp["not_exportable_reason"].astype(str) == str(row["not_exportable_reason"]))
            & (not_exp["comment_last"].astype(str) == str(row["comment_last"]))
        ].sort_values("close_time_utc").head(PRINT_NOT_EXPORTABLE_EXAMPLES_PER_CLUSTER)

        for _, ex in examples.iterrows():
            print(
                "[NOT_EXPORTABLE_EXAMPLE] "
                f"account={account_name} "
                f"ticket={ex.get('close_ticket')} "
                f"position={ex.get('position_id')} "
                f"symbol={ex.get('symbol')} "
                f"symbol_norm={ex.get('symbol_norm')} "
                f"magic={ex.get('magic')} "
                f"direction={ex.get('direction')} "
                f"strategy_id={ex.get('strategy_id')} "
                f"close={ex.get('close_time_utc')} "
                f"net={float(ex.get('net_sum', 0.0)):.2f} "
                f"reason={ex.get('not_exportable_reason')} "
                f"comment={ex.get('comment_last')}"
            )


def export_debug_csvs(output_account_dir: Path, df: pd.DataFrame) -> Tuple[int, int]:
    ensure_dir(output_account_dir)

    unmapped = make_unmapped_trades(df)
    unmapped_summary = make_unmapped_summary(unmapped)
    not_exp = make_not_exportable_trades(df)
    not_exp_summary = make_not_exportable_summary(not_exp)

    safe_to_csv(unmapped, output_account_dir / "unmapped_trades.csv")
    safe_to_csv(unmapped_summary, output_account_dir / "unmapped_summary.csv")
    safe_to_csv(not_exp, output_account_dir / "not_exportable_trades.csv")
    safe_to_csv(not_exp_summary, output_account_dir / "not_exportable_summary.csv")

    return int(len(unmapped)), int(len(not_exp))


# ============================================================
# SQLITE OUTPUT
# ============================================================

def sqlite_write_table_replace(db_path: Path, table_name: str, df: pd.DataFrame) -> None:
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.commit()
    finally:
        conn.close()


def sqlite_create_indexes_for_strategy_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trades(close_time_utc)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_close_ticket ON trades(close_ticket)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_norm ON trades(symbol_norm)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_magic ON trades(magic)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_canonical_magic ON trades(canonical_magic)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_position_id ON trades(position_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_direction ON trades(direction)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy_id ON trades(strategy_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_bucket_side ON trades(bucket_side)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_performance(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_weekly_date ON weekly_performance(date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_monthly_date ON monthly_performance(date)")
        conn.commit()
    finally:
        conn.close()


def export_strategy_db(
    bucket_dir: Path,
    trades: pd.DataFrame,
    start_equity: float,
    measurement_day_utc: pd.Timestamp,
) -> None:
    ensure_dir(bucket_dir)
    db_path = bucket_dir / OUTPUT_DB_FILENAME

    trades_out = trades.copy()
    kpis = kpis_from_trades(trades_out, measurement_day_utc=measurement_day_utc)
    daily = perf_from_trades(trades_out, freq="D", start_equity=start_equity)
    weekly = perf_from_trades(trades_out, freq="W-FRI", start_equity=start_equity)
    monthly = perf_from_trades(trades_out, freq="ME", start_equity=start_equity)
    diag = mapping_diagnostics_from_trades(trades_out)
    master_diag = inference_master_diagnostics(trades_out)

    sqlite_write_table_replace(db_path, "trades", trades_out)
    sqlite_write_table_replace(db_path, "kpis", kpis)
    sqlite_write_table_replace(db_path, "daily_performance", daily)
    sqlite_write_table_replace(db_path, "weekly_performance", weekly)
    sqlite_write_table_replace(db_path, "monthly_performance", monthly)
    sqlite_write_table_replace(db_path, "mapping_diagnostics", diag)
    sqlite_write_table_replace(db_path, "inference_master_diagnostics", master_diag)

    sqlite_create_indexes_for_strategy_db(db_path)


def clean_output_account_dir(output_account_dir: Path) -> None:
    if not CLEAN_OUTPUT_ACCOUNT_DIR_BEFORE_EXPORT:
        return
    if not output_account_dir.exists():
        return

    for child in output_account_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
        except Exception as e:
            print(f"[WARN] Could not clean old output item: {child} | {e}")


def export_account_strategies(
    output_account_dir: Path,
    df: pd.DataFrame,
    start_equity: float,
    measurement_day_utc: pd.Timestamp,
) -> Tuple[int, int, int, int]:
    if df.empty:
        return 0, 0, 0, 0

    ensure_dir(output_account_dir)
    clean_output_account_dir(output_account_dir)

    unmapped_rows, not_exportable_rows = export_debug_csvs(output_account_dir, df)

    valid = df.copy()
    canonical_magic_num = pd.to_numeric(valid["canonical_magic"], errors="coerce")

    mask_valid = (
        valid["is_exportable"].astype(bool)
        & valid["strategy_id"].apply(is_valid_strategy_id)
        & canonical_magic_num.notna()
        & (~canonical_magic_num.astype("Int64").isin(list(BAD_MAGIC_VALUES)))
    )

    skipped_count = int((~mask_valid).sum())
    valid = valid[mask_valid].copy()

    if valid.empty:
        sqlite_write_table_replace(output_account_dir / "mapping_diagnostics.db", "mapping_diagnostics", mapping_diagnostics_from_trades(df))
        sqlite_write_table_replace(output_account_dir / "mapping_diagnostics.db", "inference_master_diagnostics", inference_master_diagnostics(df))
        return 0, skipped_count, unmapped_rows, not_exportable_rows

    bucket_count = 0

    grouped = valid.groupby(["symbol_norm", "strategy_id", "canonical_magic", "bucket_side"], dropna=False, sort=True)

    for (symbol_norm, strategy_id, canonical_magic, bucket_side), g in grouped:
        trades = bucket_trades(g)

        start_date, end_date = build_bucket_date_range_for_measurement(
            trades=trades,
            measurement_day_utc=measurement_day_utc,
        )

        bucket_dir = output_account_dir / bucket_folder_name(
            symbol_norm=symbol_norm,
            canonical_magic=canonical_magic,
            strategy_id=strategy_id,
            bucket_side=bucket_side,
            start_date=start_date,
            end_date=end_date,
        )

        export_strategy_db(
            bucket_dir=bucket_dir,
            trades=trades,
            start_equity=start_equity,
            measurement_day_utc=measurement_day_utc,
        )

        bucket_count += 1

    sqlite_write_table_replace(output_account_dir / "mapping_diagnostics.db", "mapping_diagnostics", mapping_diagnostics_from_trades(df))
    sqlite_write_table_replace(output_account_dir / "mapping_diagnostics.db", "inference_master_diagnostics", inference_master_diagnostics(df))

    return bucket_count, skipped_count, unmapped_rows, not_exportable_rows


# ============================================================
# DISCOVERY / DIAGNOSTICS
# ============================================================

def find_account_dirs(root: Path) -> List[Path]:
    if not root.exists():
        return []

    out: List[Path] = []
    for p in root.iterdir():
        if p.is_dir() and re.match(r"^account_\d+_(LIVE|DEMO)$", p.name, flags=re.IGNORECASE):
            out.append(p)

    out.sort(key=lambda x: x.name)
    return out


def print_ea_diagnostics(ea_map: pd.DataFrame) -> None:
    all_mq5 = discover_ea_files()
    print(f"[INFO] Total .mq5 files discovered = {len(all_mq5)}")

    if ea_map.empty:
        print("[WARN] Kein parsebares EA Mapping gefunden. Kommentare + Inference werden genutzt.")
        return

    ea_files = ea_map["ea_filename"].drop_duplicates().sort_values().tolist()
    both_files = (
        ea_map[ea_map["strategy_side"].astype(str).str.upper() == "BOTH"]["ea_filename"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(f"[INFO] Parseable EA expanded mapping rows = {len(ea_map)}")
    print(f"[INFO] Parseable EA files found          = {len(ea_files)}")
    print(f"[INFO] BOTH EA files found               = {len(both_files)}")

    eurgbp_map = ea_map[ea_map["symbol_norm"].astype(str).str.upper() == "EURGBP"].copy()
    print(f"[INFO] EURGBP EA map rows                = {len(eurgbp_map)}")
    for _, row in eurgbp_map.iterrows():
        print(
            "[EURGBP_MAP] "
            f"symbol={row['symbol']} symbol_norm={row['symbol_norm']} "
            f"magic={row['magic']} match_direction={row['match_direction']} "
            f"strategy_id={row['strategy_id']} bucket_side={row['bucket_side']} "
            f"file={row['ea_filename']}"
        )


def print_lookup_diagnostics(df: pd.DataFrame, account_name: str) -> None:
    base = _valid_mapped_base(df)
    if base.empty:
        print(f"[LOOKUP] {account_name}: no mapped base rows")
        return

    configs = [
        ("symbol_magic", ["symbol_norm", "magic"]),
        ("symbol_magic_direction", ["symbol_norm", "magic", "direction"]),
        ("symbol_direction", ["symbol_norm", "direction"]),
    ]

    for name, cols in configs:
        if any(c not in base.columns for c in cols):
            continue

        unique = 0
        ambiguous = 0
        total = 0
        for _, g in base.groupby(cols, dropna=False):
            total += 1
            sid = _unique_strategy_from_group(g)
            if sid is None:
                ambiguous += 1
            else:
                unique += 1
        print(f"[LOOKUP] {account_name}: {name} total_keys={total} unique={unique} ambiguous={ambiguous}")


def print_account_match_diagnostics(df: pd.DataFrame, account_name: str) -> None:
    if df.empty:
        return

    total = int(len(df))
    mapped = int(df["strategy_id"].apply(is_valid_strategy_id).sum())
    exportable = int(df["is_exportable"].astype(bool).sum()) if "is_exportable" in df.columns else 0
    parsed_comments = int(df["strategy_id_from_comment"].notna().sum()) if "strategy_id_from_comment" in df.columns else 0
    inferred = int(df["strategy_mapping_source"].astype(str).str.startswith("inference").sum())

    print(f"[MATCH] {account_name}: mapped={mapped}/{total} exportable={exportable}/{total} parsed_comments={parsed_comments} inferred={inferred}")

    print_lookup_diagnostics(df, account_name)

    if "strategy_mapping_source" in df.columns:
        src = (
            df["strategy_mapping_source"]
            .replace("", "UNMAPPED")
            .value_counts(dropna=False)
            .reset_index()
        )
        src.columns = ["source", "n"]
        for _, row in src.iterrows():
            print(f"[MATCH_SOURCE] {account_name}: {row['source']}={row['n']}")

    both_rows = int((df["bucket_side"].astype(str).str.upper() == "BOTH").sum())
    both_buckets = 0
    if both_rows > 0:
        both_buckets = int(
            df[df["bucket_side"].astype(str).str.upper() == "BOTH"]
            .groupby(["symbol_norm", "strategy_id", "canonical_magic", "bucket_side"])
            .ngroups
        )

    print(f"[BOTH_STATS] {account_name}: both_rows={both_rows} both_buckets={both_buckets}")

    bad_magic_raw = int(df["raw_magic_is_bad"].sum()) if "raw_magic_is_bad" in df.columns else 0
    bad_magic_canonical = int(
        pd.to_numeric(df["canonical_magic"], errors="coerce").dropna().astype(int).isin(BAD_MAGIC_VALUES).sum()
    ) if "canonical_magic" in df.columns else 0
    print(f"[MAGIC_STATS] {account_name}: raw_bad_magic_rows={bad_magic_raw} canonical_bad_magic_rows={bad_magic_canonical}")

    eurgbp = df[df["symbol_norm"].astype(str).str.upper() == "EURGBP"].copy()
    if not eurgbp.empty:
        eurgbp_total = int(len(eurgbp))
        eurgbp_mapped = int(eurgbp["strategy_id"].apply(is_valid_strategy_id).sum())
        eurgbp_exportable = int(eurgbp["is_exportable"].astype(bool).sum()) if "is_exportable" in eurgbp.columns else 0
        print(f"[EURGBP_STATS] {account_name}: rows={eurgbp_total} mapped={eurgbp_mapped} exportable={eurgbp_exportable}")

        summary = (
            eurgbp.groupby(
                ["symbol", "symbol_norm", "magic", "canonical_magic", "direction", "strategy_id", "bucket_side", "strategy_mapping_source"],
                dropna=False,
            )
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .head(50)
        )
        for _, row in summary.iterrows():
            print(
                "[EURGBP_MATCH] "
                f"symbol={row['symbol']} symbol_norm={row['symbol_norm']} "
                f"magic={row['magic']} canonical_magic={row['canonical_magic']} "
                f"direction={row['direction']} strategy_id={row['strategy_id']} "
                f"bucket_side={row['bucket_side']} source={row['strategy_mapping_source']} n={row['n']}"
            )

    print_unmapped_terminal(df, account_name)
    print_not_exportable_terminal(df, account_name)


# ============================================================
# PROCESSING
# ============================================================

def process_account_dir(
    input_account_dir: Path,
    output_root: Path,
    reg: pd.DataFrame,
    ea_map: pd.DataFrame,
    state: Dict[str, AccountState],
    measurement_day_utc: pd.Timestamp,
) -> None:
    closed_path = input_account_dir / INPUT_DB_FILENAME

    if not closed_path.exists():
        print(f"[WARN] {input_account_dir.name}: {INPUT_DB_FILENAME} nicht gefunden")
        return

    df = read_closed_trades_db(closed_path)
    if df.empty:
        print(f"[INFO] {input_account_dir.name}: closed_trades leer")
        return

    rows_before = len(df)
    df = apply_global_cutoff_filter(df)
    rows_after = len(df)

    if df.empty:
        print(
            f"[INFO] {input_account_dir.name}: keine Trades nach Cutoff "
            f"{ts_to_str(GLOBAL_CUTOFF_DATE_UTC)} rows_before={rows_before} rows_after={rows_after}"
        )
        return

    df = enrich_strategy_ids(df, reg, ea_map)

    print_account_match_diagnostics(df, input_account_dir.name)

    account_id = int(df["account_id"].dropna().iloc[0])
    output_account_dir = output_root / input_account_dir.name
    ensure_dir(output_account_dir)

    bucket_count, skipped_count, unmapped_rows, not_exportable_rows = export_account_strategies(
        output_account_dir=output_account_dir,
        df=df,
        start_equity=START_EQUITY,
        measurement_day_utc=measurement_day_utc,
    )

    max_close_ticket = pd.to_numeric(df["close_ticket"], errors="coerce").dropna()
    max_ticket_val = int(max_close_ticket.max()) if not max_close_ticket.empty else None

    mapped = int(df["strategy_id"].apply(is_valid_strategy_id).sum())
    exportable = int(df["is_exportable"].astype(bool).sum())

    state[str(account_id)] = AccountState(
        account_id=account_id,
        account_folder=input_account_dir.name,
        last_max_close_ticket=max_ticket_val,
        last_rows=int(len(df)),
        last_mapped_rows=mapped,
        last_exportable_rows=exportable,
        last_unmapped_rows=unmapped_rows,
        last_not_exportable_rows=not_exportable_rows,
        last_exported_buckets=bucket_count,
        last_updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )

    print(
        f"[OK] account={account_id} folder={input_account_dir.name} "
        f"rows_before={rows_before} rows_after={rows_after} mapped={mapped}/{len(df)} "
        f"exportable={exportable}/{len(df)} unmapped={unmapped_rows} not_exportable={not_exportable_rows} "
        f"skipped={skipped_count} buckets={bucket_count} "
        f"max_close_ticket={max_ticket_val} measurement_day_utc={measurement_day_utc.strftime('%Y-%m-%d')} "
        f"output_dir={output_account_dir}"
    )
    print(f"[DEBUG_FILES] {output_account_dir / 'unmapped_trades.csv'}")
    print(f"[DEBUG_FILES] {output_account_dir / 'unmapped_summary.csv'}")
    print(f"[DEBUG_FILES] {output_account_dir / 'not_exportable_trades.csv'}")
    print(f"[DEBUG_FILES] {output_account_dir / 'not_exportable_summary.csv'}")


def run_once() -> None:
    ensure_dir(RUNTIME_DIR)
    ensure_dir(OUTPUT_LIVE_ROOT)

    print(f"[INFO] SCRIPT_PATH             = {SCRIPT_PATH}")
    print(f"[INFO] QUANT_ROOT              = {QUANT_ROOT}")
    print(f"[INFO] DATA_DIR                = {DATA_DIR}")
    print(f"[INFO] PIPELINE_ROOT           = {PIPELINE_ROOT}")
    print(f"[INFO] INPUT_LIVE_ROOT         = {INPUT_LIVE_ROOT}")
    print(f"[INFO] OUTPUT_LIVE_ROOT        = {OUTPUT_LIVE_ROOT}")
    print(f"[INFO] STRATEGY_EA_ROOT_PRIMARY= {STRATEGY_EA_ROOT_PRIMARY}")
    print(f"[INFO] STRATEGY_EA_ROOT_ALT    = {STRATEGY_EA_ROOT_ALT}")
    print(f"[INFO] REGISTRY_FILE           = {REGISTRY_FILE if REGISTRY_FILE.exists() else 'not found'}")
    print(f"[INFO] RUNTIME_DIR             = {RUNTIME_DIR}")
    print(f"[INFO] STATE_FILE              = {STATE_FILE}")
    print(f"[INFO] START_EQUITY            = {START_EQUITY}")
    print(f"[INFO] GLOBAL_CUTOFF_DATE_UTC  = {GLOBAL_CUTOFF_DATE_UTC}")
    print(f"[INFO] BAD_MAGIC_VALUES        = {sorted(BAD_MAGIC_VALUES)}")
    print(f"[INFO] CLEAN_OUTPUT_ACCOUNT_DIR= {CLEAN_OUTPUT_ACCOUNT_DIR_BEFORE_EXPORT}")
    print(f"[INFO] PRINT_UNMAPPED_FULL     = {PRINT_UNMAPPED_FULL}")
    print(f"[INFO] UNMAPPED_EXAMPLES       = {PRINT_UNMAPPED_EXAMPLES_PER_CLUSTER}")
    print(f"[INFO] CODE_REGISTRY.script_id = {CODE_REGISTRY['script_id']}")
    print(f"[INFO] CODE_REGISTRY.version   = {CODE_REGISTRY['version']}")

    state = load_state()
    measurement_day_utc = get_measurement_day_utc()

    reg = load_registry(REGISTRY_FILE)
    ea_map = build_strategy_ea_mapping()

    print_ea_diagnostics(ea_map)

    if reg.empty:
        print("[WARN] Keine strategy_registry.csv geladen oder leer.")
    else:
        unique_registry = build_registry_unique_lookup(reg)
        print(f"[INFO] Strategy registry rows loaded: {len(reg)}")
        print(f"[INFO] Strategy registry unique magic mappings: {len(unique_registry)}")

    account_dirs = find_account_dirs(INPUT_LIVE_ROOT)

    if not account_dirs:
        print(f"[WARN] Keine account_* Ordner gefunden unter: {INPUT_LIVE_ROOT}")
    else:
        print(f"[INFO] Account folders found: {len(account_dirs)}")
        for input_account_dir in account_dirs:
            try:
                process_account_dir(
                    input_account_dir=input_account_dir,
                    output_root=OUTPUT_LIVE_ROOT,
                    reg=reg,
                    ea_map=ea_map,
                    state=state,
                    measurement_day_utc=measurement_day_utc,
                )
            except Exception as e:
                print(f"[WARN] account processing failed: {input_account_dir.name} | {e}")

    save_state(state)

    print(
        f"[DONE] updated_at_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"measurement_day_utc={measurement_day_utc.strftime('%Y-%m-%d')}"
    )


def main() -> None:
    if RUN_FOREVER:
        try:
            while True:
                run_once()
                print(f"[LOOP] sleep_seconds={POLL_SECONDS}")
                time.sleep(POLL_SECONDS)
        except KeyboardInterrupt:
            print("[INFO] Stopping...")
    else:
        run_once()


if __name__ == "__main__":
    main()

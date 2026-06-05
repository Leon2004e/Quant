# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/2_Baseline/Trades/Live/Combiner/code.py

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: baseline_live_trade_combiner
# script_name: Baseline Live Trade Combiner
# owner: Leon Everts
# status: active
# layer: 2_Baseline
# domain: Trades/Live
# asset_type: Combiner
# purpose: Combine normalized live/demo strategy trade databases into stable baseline account/strategy databases with trades, kpis, weekly_performance, monthly_performance and meta tables. No runtime/summary files are written into Data/2_Baseline.
# inputs:
#   - Data_Center/Data/1_Pipeline/Trades/Live/account_*_LIVE/**/*.db
#   - Data_Center/Data/1_Pipeline/Trades/Live/account_*_DEMO/**/*.db
#   - Data_Center/Data/2_Baseline/Trades/Live/account_*_LIVE/**/*.db
#   - Data_Center/Data/2_Baseline/Trades/Live/account_*_DEMO/**/*.db
# outputs:
#   - Data_Center/Data/2_Baseline/Trades/Live/account_DEMO_COMBINED/strategies/<strategy_bucket>/trades.db
#   - Data_Center/Data/2_Baseline/Trades/Live/account_*_LIVE/strategies/<strategy_bucket>/trades.db
# dependencies:
#   - pathlib
#   - pandas
#   - sqlite3
#   - json
#   - re
#   - shutil
#   - tempfile
#   - dataclasses
# schedule: manual_or_loop
# version: v2.2.0_recursive_strategy_db_discovery
# last_reviewed: 2026-06-05
# business_criticality: high
# environment: desktop
# registry_group: baseline_live_trades
# notes:
#   - Designed for QUANT structure, not old FTMO feature_engineered structure.
#   - Reads from Pipeline/Trades/Live first.
#   - Writes only account/strategy trades.db outputs into Data_Center/Data/2_Baseline/Trades/Live.
#   - Demo accounts are combined into account_DEMO_COMBINED.
#   - Live accounts stay separated.
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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# ============================================================
# CODE_REGISTRY - Runtime Metadata
# ============================================================

CODE_REGISTRY = {
    "script_id": "baseline_live_trade_combiner",
    "script_name": "Baseline Live Trade Combiner",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "2_Baseline",
    "domain": "Trades/Live",
    "asset_type": "Combiner",
    "purpose": (
        "Combine normalized live/demo strategy trade databases into stable baseline "
        "account/strategy databases with trades, kpis, weekly_performance, "
        "monthly_performance and meta tables."
    ),
    "inputs": [
        "Data_Center/Data/1_Pipeline/Trades/Live/account_*_LIVE/**/*.db",
        "Data_Center/Data/1_Pipeline/Trades/Live/account_*_DEMO/**/*.db",
        "Data_Center/Data/2_Baseline/Trades/Live/account_*_LIVE/**/*.db",
        "Data_Center/Data/2_Baseline/Trades/Live/account_*_DEMO/**/*.db",
    ],
    "outputs": [
        "Data_Center/Data/2_Baseline/Trades/Live/account_DEMO_COMBINED/strategies/<strategy_bucket>/trades.db",
        "Data_Center/Data/2_Baseline/Trades/Live/account_*_LIVE/strategies/<strategy_bucket>/trades.db",
    ],
    "dependencies": [
        "pathlib",
        "pandas",
        "sqlite3",
        "json",
        "re",
        "shutil",
        "tempfile",
        "dataclasses",
    ],
    "schedule": "manual_or_loop",
    "version": "v2.2.0_recursive_strategy_db_discovery",
    "last_reviewed": "2026-06-05",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "baseline_live_trades",
    "notes": [
        "Designed for QUANT structure.",
        "Reads from Pipeline/Trades/Live first.",
        "Writes only account/strategy trades.db outputs into Data_Center/Data/2_Baseline/Trades/Live.",
        "Demo accounts are combined into account_DEMO_COMBINED.",
        "Live accounts stay separated.",
    ],
}


def get_code_registry() -> dict:
    return dict(CODE_REGISTRY)



# ============================================================
# CONFIG
# ============================================================

POLL_SECONDS = 30.0
START_EQUITY = 100000.0

INPUT_DB = "trades.db"
OUTPUT_DB = "trades.db"

COMBINED_DEMO_ACCOUNT = "account_DEMO_COMBINED"

TRADES_TABLE = "trades"
KPIS_TABLE = "kpis"
WEEKLY_TABLE = "weekly_performance"
MONTHLY_TABLE = "monthly_performance"
META_TABLE = "meta"


# ============================================================
# ROOT DETECTION
# ============================================================

def find_quant_root(start: Path) -> Path:
    cur = start.resolve()

    for p in [cur] + list(cur.parents):
        if (
            (p / "Dashboard").exists()
            and (p / "Data_Center").exists()
            and (p / "Data_Center" / "Data").exists()
        ):
            return p

        if (
            (p / "Data_Center").exists()
            and (p / "Data_Center" / "Data").exists()
        ):
            return p

    raise RuntimeError(f"QUANT root not found from: {start}")


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_DIR = QUANT_ROOT / "Data_Center" / "Data"

PIPELINE_LIVE_ROOT = DATA_DIR / "1_Pipeline" / "Trades" / "Live"
BASELINE_LIVE_ROOT = DATA_DIR / "2_Baseline" / "Trades" / "Live"
NORMALIZED_LIVE_ROOT = PIPELINE_LIVE_ROOT

FEATURE_LIVE_ROOT = BASELINE_LIVE_ROOT

DEMO_COMBINED_DIR = FEATURE_LIVE_ROOT / COMBINED_DEMO_ACCOUNT
DEMO_COMBINED_STRATEGIES_DIR = DEMO_COMBINED_DIR / "strategies"



# ============================================================
# REGEX
# ============================================================

BUCKET_RE = re.compile(
    r"^(?P<symbol>.+?)_"
    r"(?P<magic>\d+)_"
    r"(?P<strategy_id>.+?)_"
    r"(?P<side>BUY|SELL)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_to_"
    r"(?P<end>\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)

COMBINED_RE = re.compile(
    r"^(?P<canonical>.+?)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_to_"
    r"(?P<end>\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_name(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:200]


def atomic_write_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)

    try:
        tmp_path.write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def safe_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def safe_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_direction(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"LONG": "BUY", "SHORT": "SELL"})
    )


def ts_to_str(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return ""

    ts = pd.Timestamp(ts)

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.isoformat()


def today_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).date(), tz="UTC")




# ============================================================
# SQLITE
# ============================================================

def sqlite_table_exists(db_path: Path, table: str) -> bool:
    try:
        with sqlite3.connect(db_path) as conn:
            result = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table';",
                conn,
            )
            return table in result["name"].tolist()
    except Exception:
        return False


def sqlite_read(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    if not sqlite_table_exists(db_path, table):
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql(f"SELECT * FROM {table}", conn)
    except Exception:
        return pd.DataFrame()


def sqlite_write(db_path: Path, table: str, df: pd.DataFrame) -> None:
    ensure_dir(db_path.parent)

    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.commit()


def sqlite_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    return col in {r[1] for r in cur.fetchall()}


def sqlite_create_indexes(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_open_time "
            f"ON {TRADES_TABLE}(open_time_utc)"
        )

        cur.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_close_time "
            f"ON {TRADES_TABLE}(close_time_utc)"
        )

        if sqlite_has_column(conn, TRADES_TABLE, "position_id"):
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_position_id "
                f"ON {TRADES_TABLE}(position_id)"
            )

        if sqlite_has_column(conn, TRADES_TABLE, "symbol"):
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_symbol "
                f"ON {TRADES_TABLE}(symbol)"
            )

        if sqlite_has_column(conn, TRADES_TABLE, "magic"):
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_magic "
                f"ON {TRADES_TABLE}(magic)"
            )

        if sqlite_has_column(conn, TRADES_TABLE, "strategy_id"):
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_strategy_id "
                f"ON {TRADES_TABLE}(strategy_id)"
            )

        if sqlite_has_column(conn, TRADES_TABLE, "trade_signature"):
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{TRADES_TABLE}_trade_signature "
                f"ON {TRADES_TABLE}(trade_signature)"
            )

        conn.commit()


# ============================================================
# DISCOVERY
# ============================================================

def find_demo_accounts() -> list[Path]:
    if not NORMALIZED_LIVE_ROOT.exists():
        return []

    accounts = []

    for path in NORMALIZED_LIVE_ROOT.iterdir():
        if not path.is_dir():
            continue

        if path.name == COMBINED_DEMO_ACCOUNT:
            continue

        if path.name.startswith("account_") and path.name.endswith("_DEMO"):
            accounts.append(path)

    return sorted(accounts, key=lambda p: p.name)


def find_live_accounts() -> list[Path]:
    if not NORMALIZED_LIVE_ROOT.exists():
        return []

    accounts = []

    for path in NORMALIZED_LIVE_ROOT.iterdir():
        if not path.is_dir():
            continue

        if path.name.startswith("account_") and path.name.endswith("_LIVE"):
            accounts.append(path)

    return sorted(accounts, key=lambda p: p.name)


def find_strategy_dbs(account_dir: Path) -> list[tuple[str, Path]]:
    """
    Robust strategy discovery.

    Supported structures:
        account_xxx_DEMO/<strategy_bucket>/trades.db
        account_xxx_DEMO/strategies/<strategy_bucket>/trades.db
        account_xxx_DEMO/.../<strategy_bucket>/trades.db

    The previous version only checked one folder level. Your structure has
    strategy folders directly under the account and may contain many buckets.
    This version recursively searches every trades.db below the account folder.
    """
    results: list[tuple[str, Path]] = []

    if not account_dir.exists():
        return results

    seen: set[str] = set()

    for db_path in sorted(account_dir.rglob(INPUT_DB)):
        if not db_path.is_file():
            continue

        # Do not read outputs from a previously combined account inside the same root.
        lower_parts = [p.lower() for p in db_path.parts]
        if "runtime" in lower_parts:
            continue

        try:
            key = str(db_path.resolve()).lower()
        except Exception:
            key = str(db_path).lower()

        if key in seen:
            continue
        seen.add(key)

        bucket_dir = db_path.parent
        bucket_name = bucket_dir.name

        # If path is account/strategies/<bucket>/trades.db => bucket is parent
        # If path is account/<bucket>/trades.db => bucket is parent
        # If path is deeper, parent folder still identifies the strategy bucket.
        if bucket_name.lower() in {"strategies", "strategy", "live", "backtest"}:
            bucket_name = bucket_dir.parent.name

        results.append((bucket_name, db_path))

    return sorted(results, key=lambda x: x[0].lower())


# ============================================================
# NAMING
# ============================================================

def canonical_bucket(bucket_name: str) -> str:
    match = BUCKET_RE.match(bucket_name)

    if not match:
        return sanitize_name(bucket_name)

    symbol = match.group("symbol")
    magic = match.group("magic")
    strategy_id = match.group("strategy_id")
    side = match.group("side").upper()

    return sanitize_name(f"{symbol}_{magic}_{strategy_id}_{side}")


def output_bucket_name(canonical: str, trades: pd.DataFrame) -> str:
    if trades.empty:
        return f"{canonical}_unknown_to_unknown"

    open_min = pd.NaT
    close_max = pd.NaT

    if "open_time_utc" in trades.columns:
        open_min = parse_time(trades["open_time_utc"]).min()

    if "close_time_utc" in trades.columns:
        close_max = parse_time(trades["close_time_utc"]).max()

    if pd.isna(open_min) or pd.isna(close_max):
        return f"{canonical}_unknown_to_unknown"

    start = open_min.strftime("%Y-%m-%d")
    end = close_max.strftime("%Y-%m-%d")

    return sanitize_name(f"{canonical}_{start}_to_{end}")


def cleanup_old_strategy_dirs(strategies_dir: Path, canonical: str, keep_dir: Path) -> None:
    if not strategies_dir.exists():
        return

    for path in strategies_dir.iterdir():
        if not path.is_dir():
            continue

        match = COMBINED_RE.match(path.name)

        if not match:
            continue

        if match.group("canonical") != canonical:
            continue

        if path.resolve() == keep_dir.resolve():
            continue

        shutil.rmtree(path, ignore_errors=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_trades(
    df: pd.DataFrame,
    source_account: str,
    source_bucket: str,
    canonical: str,
) -> pd.DataFrame:
    d = df.copy()

    if d.empty:
        return d

    for col in ["open_time_utc", "close_time_utc"]:
        if col in d.columns:
            d[col] = parse_time(d[col])

    for col in ["account_id", "position_id", "magic", "close_ticket"]:
        if col in d.columns:
            d[col] = safe_int(d[col])

    # Finale numerische Trade-Spalten.
    # Wichtig:
    # - Diese Werte kommen aus normalized/.../trades.db.
    # - Reine Live-Snapshot-Felder wie floating_pnl/current_price werden hier nicht erzeugt.
    # - Wenn finale Live-Extremwerte vom Logger/Loader vorhanden sind, bleiben sie erhalten
    #   und werden sauber als float normalisiert.
    for col in [
        "entry_price",
        "exit_price",
        "price_delta",
        "sl",
        "tp",
        "volume_in",
        "volume_out",
        "profit_sum",
        "swap_sum",
        "commission_sum",
        "net_sum",
        "best_floating_pnl",
        "worst_floating_pnl",
        "best_price_seen_live",
        "worst_price_seen_live",
        "max_favorable_points_live",
        "max_adverse_points_live",
    ]:
        if col in d.columns:
            d[col] = safe_float(d[col])

    for col in ["symbol", "direction", "strategy_id", "comment_last"]:
        if col in d.columns:
            d[col] = d[col].astype(str).fillna("").str.strip()

    if "direction" in d.columns:
        d["direction"] = normalize_direction(d["direction"])

    d["source_account"] = source_account
    d["source_bucket"] = source_bucket
    d["canonical_bucket"] = canonical

    sig = pd.DataFrame(index=d.index)

    sig["position_id"] = d["position_id"].astype(str) if "position_id" in d.columns else ""
    sig["close_ticket"] = d["close_ticket"].astype(str) if "close_ticket" in d.columns else ""
    sig["symbol"] = d["symbol"].astype(str) if "symbol" in d.columns else ""
    sig["direction"] = d["direction"].astype(str) if "direction" in d.columns else ""
    sig["open_time_utc"] = d["open_time_utc"].astype(str) if "open_time_utc" in d.columns else ""
    sig["close_time_utc"] = d["close_time_utc"].astype(str) if "close_time_utc" in d.columns else ""
    sig["entry_price"] = d["entry_price"].round(8).astype(str) if "entry_price" in d.columns else ""
    sig["exit_price"] = d["exit_price"].round(8).astype(str) if "exit_price" in d.columns else ""
    sig["net_sum"] = d["net_sum"].round(8).astype(str) if "net_sum" in d.columns else ""
    sig["magic"] = d["magic"].astype(str) if "magic" in d.columns else ""
    sig["strategy_id"] = d["strategy_id"].astype(str) if "strategy_id" in d.columns else ""

    d["trade_signature"] = sig.agg("|".join, axis=1)

    sort_cols = [
        c for c in
        ["close_time_utc", "open_time_utc", "position_id", "close_ticket"]
        if c in d.columns
    ]

    if sort_cols:
        d = d.sort_values(sort_cols).reset_index(drop=True)

    d = d.drop_duplicates(subset=["trade_signature"], keep="first")
    d = d.reset_index(drop=True)

    return d


def combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()

    d = pd.concat(frames, ignore_index=True)

    for col in ["open_time_utc", "close_time_utc"]:
        if col in d.columns:
            d[col] = parse_time(d[col])

    sort_cols = [
        c for c in
        ["close_time_utc", "open_time_utc", "position_id", "close_ticket"]
        if c in d.columns
    ]

    if sort_cols:
        d = d.sort_values(sort_cols).reset_index(drop=True)

    if "trade_signature" in d.columns:
        d = d.drop_duplicates(subset=["trade_signature"], keep="first")

    return d.reset_index(drop=True)


# ============================================================
# KPI / PERFORMANCE / META
# ============================================================

def net_series(trades: pd.DataFrame) -> pd.Series:
    if "net_sum" in trades.columns:
        return safe_float(trades["net_sum"])
    if "net_profit" in trades.columns:
        return safe_float(trades["net_profit"])
    if "profit" in trades.columns:
        return safe_float(trades["profit"])
    return pd.Series([0.0] * len(trades))


def build_kpis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    d = trades.copy()
    net = net_series(d)

    n = int(len(d))
    wins = int((net > 0).sum())
    losses = int((net < 0).sum())

    gross_profit = float(net[net > 0].sum())
    gross_loss = float(net[net < 0].sum())
    net_pnl = float(net.sum())

    profit_factor = gross_profit / abs(gross_loss) if gross_loss < 0 else None

    equity = START_EQUITY + net.cumsum()
    drawdown = equity - equity.cummax()
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    first_open = (
        parse_time(d["open_time_utc"]).min()
        if "open_time_utc" in d.columns
        else pd.NaT
    )

    last_close = (
        parse_time(d["close_time_utc"]).max()
        if "close_time_utc" in d.columns
        else pd.NaT
    )

    return pd.DataFrame(
        [
            {
                "measurement_day_utc": ts_to_str(today_utc()),
                "first_open_time_utc": ts_to_str(first_open),
                "last_close_time_utc": ts_to_str(last_close),
                "net_pnl": net_pnl,
                "gross_profit": gross_profit,
                "gross_loss": gross_loss,
                "profit_factor": profit_factor if profit_factor is not None else "",
                "n_trades": n,
                "wins": wins,
                "losses": losses,
                "win_rate": wins / n if n > 0 else 0.0,
                "avg_trade": float(net.mean()) if n > 0 else 0.0,
                "max_drawdown_closed": max_drawdown,
                "unique_source_accounts": (
                    int(d["source_account"].nunique())
                    if "source_account" in d.columns
                    else 0
                ),
            }
        ]
    )


def build_period_performance(trades: pd.DataFrame, freq: str) -> pd.DataFrame:
    if trades.empty or "close_time_utc" not in trades.columns:
        return pd.DataFrame()

    d = trades.copy()
    d["close_time_utc"] = parse_time(d["close_time_utc"])
    d = d.dropna(subset=["close_time_utc"])

    if d.empty:
        return pd.DataFrame()

    d["pnl"] = net_series(d)
    d = d.set_index("close_time_utc")

    out = pd.DataFrame()
    out["pnl_money"] = d["pnl"].resample(freq).sum()
    out["cum_pnl_money"] = out["pnl_money"].cumsum()
    out["nav"] = START_EQUITY + out["cum_pnl_money"]

    out = out.reset_index().rename(columns={"close_time_utc": "date"})
    return out


def build_meta(
    account_output: str,
    canonical: str,
    bucket_name: str,
    trades: pd.DataFrame,
    source_accounts: list[str],
    source_buckets: list[str],
    mode: str,
) -> pd.DataFrame:
    start = (
        ts_to_str(parse_time(trades["open_time_utc"]).min())
        if not trades.empty and "open_time_utc" in trades.columns
        else ""
    )

    end = (
        ts_to_str(parse_time(trades["close_time_utc"]).max())
        if not trades.empty and "close_time_utc" in trades.columns
        else ""
    )

    return pd.DataFrame(
        [
            {
                "mode": mode,
                "account_output": account_output,
                "canonical_bucket": canonical,
                "combined_bucket_name": bucket_name,
                "from_open_utc": start,
                "to_close_utc": end,
                "rows": int(len(trades)),
                "source_accounts_count": len(set(source_accounts)),
                "source_buckets_count": len(set(source_buckets)),
                "source_accounts": ",".join(sorted(set(source_accounts))),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    )


# ============================================================
# EXPORT
# ============================================================

def export_strategy_db(
    output_dir: Path,
    account_output: str,
    canonical: str,
    bucket_name: str,
    trades: pd.DataFrame,
    source_accounts: list[str],
    source_buckets: list[str],
    mode: str,
) -> None:
    ensure_dir(output_dir)

    db_path = output_dir / OUTPUT_DB

    sqlite_write(db_path, TRADES_TABLE, trades)
    sqlite_write(db_path, KPIS_TABLE, build_kpis(trades))
    sqlite_write(db_path, WEEKLY_TABLE, build_period_performance(trades, "W-FRI"))
    sqlite_write(db_path, MONTHLY_TABLE, build_period_performance(trades, "ME"))
    sqlite_write(
        db_path,
        META_TABLE,
        build_meta(
            account_output=account_output,
            canonical=canonical,
            bucket_name=bucket_name,
            trades=trades,
            source_accounts=source_accounts,
            source_buckets=source_buckets,
            mode=mode,
        ),
    )

    sqlite_create_indexes(db_path)


# ============================================================
# DEMO COMBINED
# ============================================================

def collect_demo_groups() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}

    demo_accounts = find_demo_accounts()
    print(f"[DIAG] demo_accounts_found={len(demo_accounts)}")
    for account_dir in demo_accounts:
        strategy_dbs = find_strategy_dbs(account_dir)
        print(f"[DIAG] {account_dir.name}: strategy_dbs_found={len(strategy_dbs)}")
        for source_bucket, db_path in strategy_dbs:
            canonical = canonical_bucket(source_bucket)

            groups.setdefault(canonical, []).append(
                {
                    "account": account_dir.name,
                    "bucket": source_bucket,
                    "db_path": db_path,
                }
            )

    return groups


def process_demo_combined() -> tuple[int, int, dict]:
    ensure_dir(DEMO_COMBINED_STRATEGIES_DIR)

    groups = collect_demo_groups()
    print(f"[DIAG] demo_groups_found={len(groups)}")
    summary = {}
    total_trades = 0

    for canonical, items in sorted(groups.items(), key=lambda x: x[0]):
        frames = []
        source_accounts = []
        source_buckets = []
        errors = []

        for item in items:
            account = item["account"]
            bucket = item["bucket"]
            db_path = Path(item["db_path"])

            try:
                raw = sqlite_read(db_path, TRADES_TABLE)

                if raw.empty:
                    continue

                normalized = normalize_trades(
                    raw,
                    source_account=account,
                    source_bucket=bucket,
                    canonical=canonical,
                )

                frames.append(normalized)
                source_accounts.append(account)
                source_buckets.append(bucket)

            except Exception as exc:
                errors.append(f"{db_path}: {exc}")

        combined = combine_frames(frames)

        if combined.empty:
            summary[canonical] = {
                "mode": "demo_combined",
                "status": "empty",
                "errors": errors,
            }
            continue

        bucket_name = output_bucket_name(canonical, combined)
        output_dir = DEMO_COMBINED_STRATEGIES_DIR / bucket_name

        cleanup_old_strategy_dirs(DEMO_COMBINED_STRATEGIES_DIR, canonical, output_dir)

        export_strategy_db(
            output_dir=output_dir,
            account_output=COMBINED_DEMO_ACCOUNT,
            canonical=canonical,
            bucket_name=bucket_name,
            trades=combined,
            source_accounts=source_accounts,
            source_buckets=source_buckets,
            mode="demo_combined",
        )

        total_trades += int(len(combined))

        summary[bucket_name] = {
            "mode": "demo_combined",
            "status": "ok" if not errors else "ok_with_errors",
            "canonical_bucket": canonical,
            "rows": int(len(combined)),
            "source_accounts": sorted(set(source_accounts)),
            "source_buckets": sorted(set(source_buckets)),
            "output_db": str(output_dir / OUTPUT_DB),
            "errors": errors,
        }

    return len(groups), total_trades, summary


# ============================================================
# LIVE SEPARATE
# ============================================================

def process_single_live_account(account_dir: Path) -> tuple[int, int, dict]:
    account_output_dir = FEATURE_LIVE_ROOT / account_dir.name
    strategies_output_dir = account_output_dir / "strategies"
    ensure_dir(strategies_output_dir)

    summary = {}
    strategy_count = 0
    trade_count = 0

    strategy_dbs = find_strategy_dbs(account_dir)
    print(f"[DIAG] {account_dir.name}: strategy_dbs_found={len(strategy_dbs)}")

    for source_bucket, db_path in strategy_dbs:
        canonical = canonical_bucket(source_bucket)

        try:
            raw = sqlite_read(db_path, TRADES_TABLE)

            if raw.empty:
                summary[source_bucket] = {
                    "mode": "live_separate",
                    "account": account_dir.name,
                    "status": "empty",
                    "input_db": str(db_path),
                }
                continue

            normalized = normalize_trades(
                raw,
                source_account=account_dir.name,
                source_bucket=source_bucket,
                canonical=canonical,
            )

            combined = combine_frames([normalized])

            if combined.empty:
                summary[source_bucket] = {
                    "mode": "live_separate",
                    "account": account_dir.name,
                    "status": "empty_after_normalization",
                    "input_db": str(db_path),
                }
                continue

            bucket_name = output_bucket_name(canonical, combined)
            output_dir = strategies_output_dir / bucket_name

            cleanup_old_strategy_dirs(strategies_output_dir, canonical, output_dir)

            export_strategy_db(
                output_dir=output_dir,
                account_output=account_dir.name,
                canonical=canonical,
                bucket_name=bucket_name,
                trades=combined,
                source_accounts=[account_dir.name],
                source_buckets=[source_bucket],
                mode="live_separate",
            )

            strategy_count += 1
            trade_count += int(len(combined))

            summary[bucket_name] = {
                "mode": "live_separate",
                "account": account_dir.name,
                "status": "ok",
                "canonical_bucket": canonical,
                "rows": int(len(combined)),
                "input_db": str(db_path),
                "output_db": str(output_dir / OUTPUT_DB),
            }

        except Exception as exc:
            summary[source_bucket] = {
                "mode": "live_separate",
                "account": account_dir.name,
                "status": "error",
                "input_db": str(db_path),
                "error": str(exc),
            }

    return strategy_count, trade_count, summary


def process_live_separate() -> tuple[int, int, int, dict]:
    accounts = find_live_accounts()

    total_strategy_count = 0
    total_trade_count = 0
    summary = {}

    for account_dir in accounts:
        strategy_count, trade_count, account_summary = process_single_live_account(account_dir)

        total_strategy_count += strategy_count
        total_trade_count += trade_count

        summary[account_dir.name] = {
            "strategies": strategy_count,
            "trades": trade_count,
            "summary": account_summary,
        }

    return len(accounts), total_strategy_count, total_trade_count, summary


# ============================================================
# RUN
# ============================================================

def process_all() -> dict:
    ensure_dir(FEATURE_LIVE_ROOT)

    demo_strategy_count, demo_trade_count, demo_summary = process_demo_combined()

    (
        live_account_count,
        live_strategy_count,
        live_trade_count,
        live_summary,
    ) = process_live_separate()

    result = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_registry": CODE_REGISTRY,
        "input_root": str(NORMALIZED_LIVE_ROOT),
        "output_root": str(FEATURE_LIVE_ROOT),
        "demo": {
            "output_account": COMBINED_DEMO_ACCOUNT,
            "strategies": demo_strategy_count,
            "trades": demo_trade_count,
            "summary": demo_summary,
        },
        "live": {
            "accounts": live_account_count,
            "strategies": live_strategy_count,
            "trades": live_trade_count,
            "summary": live_summary,
        },
    }


    return result


def run_once() -> dict:
    return process_all()


def run_loop() -> None:
    print(f"[INFO] QUANT_ROOT       = {QUANT_ROOT}")
    print(f"[INFO] INPUT_ROOT       = {NORMALIZED_LIVE_ROOT}")
    print(f"[INFO] OUTPUT_ROOT      = {FEATURE_LIVE_ROOT}")
    print(f"[INFO] POLL_SECONDS     = {POLL_SECONDS}")

    try:
        while True:
            result = run_once()

            print(
                f"[LOOP] updated={result['updated_at_utc']} "
                f"demo_strategies={result['demo']['strategies']} "
                f"demo_trades={result['demo']['trades']} "
                f"live_accounts={result['live']['accounts']} "
                f"live_strategies={result['live']['strategies']} "
                f"live_trades={result['live']['trades']}"
            )

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("[INFO] stopped")


if __name__ == "__main__":
    run_loop()
# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/3_Research/Market/code.py
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: market_daily_regime_research
# script_name: code.py
# owner: Leon Everts
# status: active
# layer: 3_Research
# domain: Market
# asset_type: Research
# purpose: Calculates daily market regime research features from M15 OHLC pipeline data.
# inputs:
#   - Data_Center/Data/1_Pipeline/Market/ohcl/M15/*.parquet
# outputs:
#   - Data_Center/Data/3_Research/Market/daily_regime_features.csv
#   - Data_Center/Data/3_Research/Market/daily_regime_features.parquet
# upstream_data:
#   - 1_Pipeline
# downstream_data:
#   - Analytics
# dependencies:
#   - pandas
#   - numpy
#   - pathlib
# schedule: manual
# version: v1.0.0
# last_reviewed: 2026-06-05
# business_criticality: high
# environment: desktop
# registry_group: research
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-05
# tags:
#   - market
#   - regime
#   - research
#   - ohlc
#   - features
# notes:
#   - Reads directly from 1_Pipeline Market OHLC data.
#   - Stores daily market behavior research features in 3_Research.
# ============================================================


CODE_REGISTRY: Dict[str, object] = {
    "script_id": "market_daily_regime_research",
    "script_name": "code.py",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "3_Research",
    "domain": "Market",
    "asset_type": "Research",
    "purpose": "Calculates daily market regime research features from M15 OHLC pipeline data.",
    "inputs": ["Data_Center/Data/1_Pipeline/Market/ohcl/M15/*.parquet"],
    "outputs": [
        "Data_Center/Data/3_Research/Market/daily_regime_features.csv",
        "Data_Center/Data/3_Research/Market/daily_regime_features.parquet",
    ],
    "upstream_data": ["1_Pipeline"],
    "downstream_data": ["Analytics"],
    "dependencies": ["pandas", "numpy", "pathlib"],
    "schedule": "manual",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-05",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "research",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-05",
    "tags": ["market", "regime", "research", "ohlc", "features"],
}


TARGET_TIMEFRAME = "M15"

INPUT_COLUMNS = [
    "time", "open", "high", "low", "close",
    "tick_volume", "spread", "real_volume",
    "symbol", "timeframe",
]

OUTPUT_REQUIRED_COLUMNS = [
    "date", "symbol", "timeframe", "bars",
    "open", "high", "low", "close",
    "net_move", "daily_range", "path_length",
    "trend_efficiency", "path_efficiency", "noise_ratio",
    "up_run_count", "down_run_count", "total_run_count",
    "avg_up_run_length", "max_up_run_length",
    "avg_down_run_length", "max_down_run_length",
    "pullback_count", "pullback_ratio",
    "avg_body_size", "avg_upper_wick", "avg_lower_wick",
    "max_upper_wick", "max_lower_wick", "wick_ratio",
    "new_high_count", "new_low_count",
    "high_break_acceptance", "low_break_acceptance",
    "failed_high_breakouts", "failed_low_breakouts",
    "false_breakout_ratio", "direction_changes",
    "largest_candle", "largest_candle_pct_of_day",
    "top_3_candles_share", "atr_spike_count",
]


def find_quant_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p
    raise RuntimeError(f"QUANT root not found from: {start}")


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER = QUANT_ROOT / "Data_Center"
DATA_ROOT = DATA_CENTER / "Data"

INPUT_ROOT = DATA_ROOT / "1_Pipeline" / "Market" / "ohcl"
OUTPUT_ROOT = DATA_ROOT / "3_Research" / "Market"
OUTPUT_BY_SYMBOL = OUTPUT_ROOT / "by_symbol"
LOG_DIR = OUTPUT_ROOT / "_logs"
META_DIR = OUTPUT_ROOT / "_metadata"

REGISTRY_DB = DATA_CENTER / "Backend_Management" / "code_registry.db"


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "market_research.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def ensure_dirs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_BY_SYMBOL.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def normalize_symbol(symbol: str) -> str:
    return (
        str(symbol)
        .replace(".cash", "")
        .replace("/", "_")
        .upper()
        .strip()
    )


def validate_input_schema(df: pd.DataFrame, file: Path) -> None:
    missing = [c for c in ["time", "open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file}: {missing}")


def validate_output_schema(df: pd.DataFrame) -> None:
    missing = [c for c in OUTPUT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Output schema validation failed. Missing columns: {missing}")


def discover_input_files() -> List[Path]:
    tf_dir = INPUT_ROOT / TARGET_TIMEFRAME

    if not tf_dir.exists():
        raise FileNotFoundError(f"Input timeframe folder not found: {tf_dir}")

    files = sorted(tf_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in: {tf_dir}")

    return files


def clean_ohlc(df: pd.DataFrame, file: Path) -> pd.DataFrame:
    validate_input_schema(df, file)

    d = df.copy()

    for col in INPUT_COLUMNS:
        if col not in d.columns:
            if col in ["tick_volume", "spread", "real_volume"]:
                d[col] = 0
            elif col == "symbol":
                d[col] = file.stem
            elif col == "timeframe":
                d[col] = TARGET_TIMEFRAME
            else:
                d[col] = np.nan

    d = d[INPUT_COLUMNS].copy()

    d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")

    for col in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["symbol"] = d["symbol"].apply(normalize_symbol)
    d["timeframe"] = d["timeframe"].astype(str).str.upper()

    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    d = d[d["timeframe"] == TARGET_TIMEFRAME]

    d = d[
        (d["high"] >= d["low"])
        & (d["high"] >= d["open"])
        & (d["high"] >= d["close"])
        & (d["low"] <= d["open"])
        & (d["low"] <= d["close"])
    ]

    d = d.drop_duplicates(subset=["symbol", "timeframe", "time"], keep="last")
    d = d.sort_values(["symbol", "timeframe", "time"]).reset_index(drop=True)

    return d


def add_candle_base_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["range"] = d["high"] - d["low"]
    d["body"] = (d["close"] - d["open"]).abs()

    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    d["direction"] = np.where(
        d["close"] > d["open"],
        1,
        np.where(d["close"] < d["open"], -1, 0),
    )

    d["close_direction"] = np.sign(d["close"].diff()).fillna(0)

    d["range_safe"] = d["range"].replace(0, np.nan)
    d["wick_ratio_row"] = (d["upper_wick"] + d["lower_wick"]) / d["range_safe"]

    return d


def run_lengths(series: pd.Series, value: int) -> List[int]:
    runs: List[int] = []
    count = 0

    for x in series:
        if x == value:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0

    if count > 0:
        runs.append(count)

    return runs


def count_direction_changes(series: pd.Series) -> int:
    s = series.replace(0, np.nan).dropna()
    if len(s) <= 1:
        return 0
    return int((s != s.shift(1)).sum() - 1)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def calculate_daily_features(day: pd.DataFrame) -> Dict[str, object]:
    d = add_candle_base_features(day)

    first_open = safe_float(d["open"].iloc[0])
    last_close = safe_float(d["close"].iloc[-1])
    day_high = safe_float(d["high"].max())
    day_low = safe_float(d["low"].min())

    daily_range = day_high - day_low
    net_move = abs(last_close - first_open)

    close_diff_abs = d["close"].diff().abs()
    path_length = safe_float(close_diff_abs.sum())

    trend_efficiency = net_move / path_length if path_length > 0 else 0.0
    path_efficiency = net_move / daily_range if daily_range > 0 else 0.0
    noise_ratio = 1.0 - trend_efficiency if 0 <= trend_efficiency <= 1 else 0.0

    up_runs = run_lengths(d["close_direction"], 1)
    down_runs = run_lengths(d["close_direction"], -1)

    prev_cum_high = d["high"].shift(1).cummax()
    prev_cum_low = d["low"].shift(1).cummin()

    high_breaks = d["high"] > prev_cum_high
    low_breaks = d["low"] < prev_cum_low

    new_high_count = int(high_breaks.sum())
    new_low_count = int(low_breaks.sum())

    high_break_acceptance = (
        safe_float((d.loc[high_breaks, "close"] > prev_cum_high.loc[high_breaks]).mean())
        if high_breaks.sum() > 0 else 0.0
    )

    low_break_acceptance = (
        safe_float((d.loc[low_breaks, "close"] < prev_cum_low.loc[low_breaks]).mean())
        if low_breaks.sum() > 0 else 0.0
    )

    failed_high_breakouts = int((high_breaks & (d["close"] < d["open"])).sum())
    failed_low_breakouts = int((low_breaks & (d["close"] > d["open"])).sum())

    total_breakouts = new_high_count + new_low_count
    false_breakout_ratio = (
        (failed_high_breakouts + failed_low_breakouts) / total_breakouts
        if total_breakouts > 0 else 0.0
    )

    direction_changes = count_direction_changes(d["close_direction"])
    pullback_count = direction_changes

    largest_candle = safe_float(d["range"].max())
    largest_candle_pct_of_day = largest_candle / daily_range if daily_range > 0 else 0.0

    top_3_candles_share = (
        safe_float(d["range"].nlargest(3).sum()) / daily_range
        if daily_range > 0 else 0.0
    )

    median_range = safe_float(d["range"].median())
    atr_spike_count = int((d["range"] > 2.0 * median_range).sum()) if median_range > 0 else 0

    return {
        "date": str(d["time"].dt.date.iloc[0]),
        "symbol": d["symbol"].iloc[0],
        "timeframe": d["timeframe"].iloc[0],
        "bars": int(len(d)),

        "open": first_open,
        "high": day_high,
        "low": day_low,
        "close": last_close,

        "net_move": float(net_move),
        "daily_range": float(daily_range),
        "path_length": float(path_length),

        "trend_efficiency": float(trend_efficiency),
        "path_efficiency": float(path_efficiency),
        "noise_ratio": float(noise_ratio),

        "up_run_count": int(len(up_runs)),
        "down_run_count": int(len(down_runs)),
        "total_run_count": int(len(up_runs) + len(down_runs)),

        "avg_up_run_length": float(np.mean(up_runs)) if up_runs else 0.0,
        "max_up_run_length": float(np.max(up_runs)) if up_runs else 0.0,
        "avg_down_run_length": float(np.mean(down_runs)) if down_runs else 0.0,
        "max_down_run_length": float(np.max(down_runs)) if down_runs else 0.0,

        "pullback_count": int(pullback_count),
        "pullback_ratio": float(pullback_count / len(d)) if len(d) > 0 else 0.0,

        "avg_body_size": safe_float(d["body"].mean()),
        "avg_upper_wick": safe_float(d["upper_wick"].mean()),
        "avg_lower_wick": safe_float(d["lower_wick"].mean()),
        "max_upper_wick": safe_float(d["upper_wick"].max()),
        "max_lower_wick": safe_float(d["lower_wick"].max()),
        "wick_ratio": safe_float(d["wick_ratio_row"].mean()),

        "new_high_count": int(new_high_count),
        "new_low_count": int(new_low_count),
        "high_break_acceptance": float(high_break_acceptance),
        "low_break_acceptance": float(low_break_acceptance),

        "failed_high_breakouts": int(failed_high_breakouts),
        "failed_low_breakouts": int(failed_low_breakouts),
        "false_breakout_ratio": float(false_breakout_ratio),

        "direction_changes": int(direction_changes),

        "largest_candle": float(largest_candle),
        "largest_candle_pct_of_day": float(largest_candle_pct_of_day),
        "top_3_candles_share": float(top_3_candles_share),
        "atr_spike_count": int(atr_spike_count),
    }


def process_file(file: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    raw = pd.read_parquet(file)
    clean = clean_ohlc(raw, file)

    if clean.empty:
        raise ValueError(f"No valid rows after cleaning: {file}")

    clean["date"] = clean["time"].dt.date

    rows: List[Dict[str, object]] = []

    for _, day in clean.groupby("date"):
        day = day.sort_values("time")
        if len(day) > 0:
            rows.append(calculate_daily_features(day))

    features = pd.DataFrame(rows)

    meta = {
        "file": str(file),
        "symbol": clean["symbol"].iloc[0],
        "timeframe": TARGET_TIMEFRAME,
        "input_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "output_rows": int(len(features)),
        "from_utc": str(clean["time"].min()),
        "to_utc": str(clean["time"].max()),
    }

    return features, meta


def save_output(features: pd.DataFrame) -> None:
    validate_output_schema(features)

    full_csv = OUTPUT_ROOT / "daily_regime_features.csv"
    full_parquet = OUTPUT_ROOT / "daily_regime_features.parquet"

    features = features.sort_values(["symbol", "timeframe", "date"]).reset_index(drop=True)

    features.to_csv(full_csv, index=False)
    features.to_parquet(full_parquet, index=False)

    for symbol, group in features.groupby("symbol"):
        safe_symbol = normalize_symbol(symbol)
        group.to_csv(OUTPUT_BY_SYMBOL / f"{safe_symbol}_daily_regime_features.csv", index=False)
        group.to_parquet(OUTPUT_BY_SYMBOL / f"{safe_symbol}_daily_regime_features.parquet", index=False)


def save_summary(features: pd.DataFrame, file_metadata: List[Dict[str, object]], errors: List[Dict[str, str]]) -> None:
    summary = {
        "script": CODE_REGISTRY,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_root": str(INPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "target_timeframe": TARGET_TIMEFRAME,
        "rows": int(len(features)),
        "symbols": sorted(features["symbol"].unique().tolist()) if not features.empty else [],
        "from_date": str(features["date"].min()) if not features.empty else "",
        "to_date": str(features["date"].max()) if not features.empty else "",
        "files_processed": int(len(file_metadata)),
        "errors": int(len(errors)),
    }

    with open(OUTPUT_ROOT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(META_DIR / "file_metadata.json", "w", encoding="utf-8") as f:
        json.dump(file_metadata, f, indent=2, ensure_ascii=False)

    with open(LOG_DIR / "errors.json", "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)


def update_code_registry_db() -> None:
    try:
        REGISTRY_DB.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(REGISTRY_DB)
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS code_registry (
                script_id TEXT PRIMARY KEY,
                script_name TEXT,
                owner TEXT,
                status TEXT,
                layer TEXT,
                domain TEXT,
                asset_type TEXT,
                purpose TEXT,
                version TEXT,
                last_reviewed TEXT,
                business_criticality TEXT,
                environment TEXT,
                registry_group_name TEXT,
                author TEXT,
                reviewer TEXT,
                created_date TEXT,
                updated_at_utc TEXT,
                metadata_json TEXT
            )
            """
        )

        cur.execute(
            """
            INSERT OR REPLACE INTO code_registry (
                script_id,
                script_name,
                owner,
                status,
                layer,
                domain,
                asset_type,
                purpose,
                version,
                last_reviewed,
                business_criticality,
                environment,
                registry_group_name,
                author,
                reviewer,
                created_date,
                updated_at_utc,
                metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                CODE_REGISTRY["script_id"],
                CODE_REGISTRY["script_name"],
                CODE_REGISTRY["owner"],
                CODE_REGISTRY["status"],
                CODE_REGISTRY["layer"],
                CODE_REGISTRY["domain"],
                CODE_REGISTRY["asset_type"],
                CODE_REGISTRY["purpose"],
                CODE_REGISTRY["version"],
                CODE_REGISTRY["last_reviewed"],
                CODE_REGISTRY["business_criticality"],
                CODE_REGISTRY["environment"],
                CODE_REGISTRY["registry_group"],
                CODE_REGISTRY["author"],
                CODE_REGISTRY["reviewer"],
                CODE_REGISTRY["created_date"],
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                json.dumps(CODE_REGISTRY, ensure_ascii=False),
            ),
        )

        conn.commit()
        conn.close()

    except Exception as e:
        logging.warning(f"Code registry DB update failed: {e}")


def main() -> None:
    setup_logging()
    ensure_dirs()

    logging.info(f"QUANT_ROOT       = {QUANT_ROOT}")
    logging.info(f"INPUT_ROOT       = {INPUT_ROOT}")
    logging.info(f"OUTPUT_ROOT      = {OUTPUT_ROOT}")
    logging.info(f"TARGET_TIMEFRAME = {TARGET_TIMEFRAME}")

    update_code_registry_db()

    files = discover_input_files()

    all_features: List[pd.DataFrame] = []
    file_metadata: List[Dict[str, object]] = []
    errors: List[Dict[str, str]] = []

    for file in files:
        try:
            logging.info(f"Processing: {file}")
            features, meta = process_file(file)

            if not features.empty:
                all_features.append(features)
                file_metadata.append(meta)
                logging.info(
                    f"OK {meta['symbol']} rows_in={meta['clean_rows']} rows_out={meta['output_rows']}"
                )

        except Exception as e:
            err = {
                "file": str(file),
                "error": str(e),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            errors.append(err)
            logging.error(f"FAILED {file}: {e}")

    if not all_features:
        save_summary(pd.DataFrame(), file_metadata, errors)
        raise RuntimeError("No research features generated.")

    final = pd.concat(all_features, ignore_index=True)
    final = final.drop_duplicates(subset=["symbol", "timeframe", "date"], keep="last")

    validate_output_schema(final)
    save_output(final)
    save_summary(final, file_metadata, errors)

    logging.info(f"DONE rows={len(final):,} symbols={final['symbol'].nunique()}")
    logging.info(f"Saved: {OUTPUT_ROOT / 'daily_regime_features.csv'}")
    logging.info(f"Saved: {OUTPUT_ROOT / 'daily_regime_features.parquet'}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/2_Baseline/Trades/Backtest_IS_OOS_Splitter/code.py

Zweck:
- Liest rekursiv Backtest-Trade-CSV-Dateien aus Data_Center/Data/1_Pipeline/Trades/Backtest
- Normalisiert StrategyQuant/Backtest-Trades in ein einheitliches Baseline-Trade-Schema
- Teilt Trades in IS und OOS auf
- Speichert die Ergebnisse in Data_Center/Data/2_Baseline/Trades/Backtest
- Erzeugt Summary- und Error-Reports für Kontrolle und Debugging

IS/OOS Regel:
- IS  = bis einschließlich 2024-07-01 23:59:59 UTC
- OOS = ab 2024-07-02 00:00:00 UTC

Input:
QUANT/Data_Center/Data/1_Pipeline/Trades/Backtest/**/*.csv

Output:
QUANT/Data_Center/Data/2_Baseline/Trades/Backtest/<ACCOUNT>/IS/*.csv
QUANT/Data_Center/Data/2_Baseline/Trades/Backtest/<ACCOUNT>/OOS/*.csv
QUANT/Data_Center/Data/2_Baseline/Trades/Backtest/_reports/baseline_backtest_is_oos_summary.csv
QUANT/Data_Center/Data/2_Baseline/Trades/Backtest/_reports/baseline_backtest_is_oos_errors.csv

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: baseline_backtest_is_oos_splitter
# script_name: Baseline Backtest IS OOS Splitter
# owner: Leon Everts
# status: active
# layer: 2_Baseline
# domain: Trades
# asset_type: Baseline
# purpose: Normalize raw backtest trade CSV files from the pipeline layer and split them into standardized IS/OOS baseline datasets.
# inputs:
#   - Data_Center/Data/1_Pipeline/Trades/Backtest/**/*.csv
# outputs:
#   - Data_Center/Data/2_Baseline/Trades/Backtest/<ACCOUNT>/IS/*.csv
#   - Data_Center/Data/2_Baseline/Trades/Backtest/<ACCOUNT>/OOS/*.csv
#   - Data_Center/Data/2_Baseline/Trades/Backtest/_reports/baseline_backtest_is_oos_summary.csv
#   - Data_Center/Data/2_Baseline/Trades/Backtest/_reports/baseline_backtest_is_oos_errors.csv
# upstream_data:
#   - 1_Pipeline/Trades/Backtest
# downstream_data:
#   - 3_Features/Trades
#   - 4_Analytics/Trades
#   - Dashboards
# dependencies:
#   - pandas
#   - pathlib
#   - re
# schedule: manual
# version: v1.0.0
# last_reviewed: 2026-06-01
# business_criticality: high
# environment: desktop
# registry_group: baseline
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# tags:
#   - backtest
#   - baseline
#   - trades
#   - is_oos_split
#   - normalization
#   - strategyquant
# notes:
#   - IS cutoff is 2024-07-01 23:59:59 UTC.
#   - Split is based on open_time_utc.
#   - Output is account-separated and sample-separated.
# ============================================================
"""

from __future__ import annotations

import re
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


# ============================================================
# CODE REGISTRY - Runtime Metadata
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "baseline_backtest_is_oos_splitter",
    "script_name": "Baseline Backtest IS OOS Splitter",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "2_Baseline",
    "domain": "Trades",
    "asset_type": "Baseline",
    "purpose": (
        "Normalize raw backtest trade CSV files from the pipeline layer and split them "
        "into standardized IS/OOS baseline datasets."
    ),
    "inputs": [
        "Data_Center/Data/1_Pipeline/Trades/Backtest/**/*.csv",
    ],
    "outputs": [
        "Data_Center/Data/2_Baseline/Trades/Backtest/<ACCOUNT>/IS/*.csv",
        "Data_Center/Data/2_Baseline/Trades/Backtest/<ACCOUNT>/OOS/*.csv",
        "Data_Center/Data/2_Baseline/Trades/Backtest/_reports/baseline_backtest_is_oos_summary.csv",
        "Data_Center/Data/2_Baseline/Trades/Backtest/_reports/baseline_backtest_is_oos_errors.csv",
    ],
    "upstream_data": [
        "1_Pipeline/Trades/Backtest",
    ],
    "downstream_data": [
        "3_Features/Trades",
        "4_Analytics/Trades",
        "Dashboards",
    ],
    "dependencies": [
        "pandas",
        "pathlib",
        "re",
    ],
    "schedule": "manual",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-01",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "baseline",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "tags": [
        "backtest",
        "baseline",
        "trades",
        "is_oos_split",
        "normalization",
        "strategyquant",
    ],
    "notes": [
        "IS cutoff is 2024-07-01 23:59:59 UTC.",
        "Split is based on open_time_utc.",
        "Output is account-separated and sample-separated.",
    ],
}


# ============================================================
# CONFIG
# ============================================================

IS_CUTOFF = pd.Timestamp("2024-07-01 23:59:59", tz="UTC")

REQUIRED_INPUT_COLUMNS: List[str] = [
    "Ticket",
    "Symbol",
    "Type",
    "Open time",
    "Close time",
    "Open price",
    "Close price",
    "Profit/Loss",
]

OUTPUT_COLUMNS: List[str] = [
    "position_id",
    "symbol",
    "direction",
    "open_time_utc",
    "close_time_utc",
    "entry_price",
    "exit_price",
    "price_delta",
    "profit_sum",
    "net_sum",
    "strategy_id",
    "account_type",
    "sample_type",
    "source_file",
    "source_rel_path",
    "processed_at_utc",
]


# ============================================================
# LOGGING
# ============================================================

def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")


def log_ok(msg: str) -> None:
    print(f"[OK] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")


# ============================================================
# ROOT FINDER / PATHS
# ============================================================

def find_quant_root(start: Path) -> Path:
    """
    QUANT Root wird erkannt über:

        QUANT/
            Dashboard/
            Data_Center/
                Data/
                Backend_Management/

    Minimal erforderlich:
    - Dashboard/
    - Data_Center/

    Wenn Data_Center/Data und Data_Center/Backend_Management existieren,
    wird die Erkennung zusätzlich validiert.
    """
    cur = start.resolve()

    for p in [cur] + list(cur.parents):
        dashboard_dir = p / "Dashboard"
        data_center_dir = p / "Data_Center"

        if dashboard_dir.exists() and data_center_dir.exists():
            data_dir = data_center_dir / "Data"
            backend_dir = data_center_dir / "Backend_Management"

            if data_dir.exists() or backend_dir.exists():
                return p

            return p

    raise RuntimeError(
        "QUANT Root nicht gefunden. Erwartet Ordner mit Dashboard/ und Data_Center/. "
        f"Start={start}"
    )


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
BACKEND_MANAGEMENT_DIR = DATA_CENTER_DIR / "Backend_Management"

INPUT_ROOT = DATA_DIR / "1_Pipeline" / "Trades" / "Backtest"
OUTPUT_ROOT = DATA_DIR / "2_Baseline" / "Trades" / "Backtest"
REPORTS_DIR = OUTPUT_ROOT / "_reports"

SUMMARY_REPORT_PATH = REPORTS_DIR / "baseline_backtest_is_oos_summary.csv"
ERROR_REPORT_PATH = REPORTS_DIR / "baseline_backtest_is_oos_errors.csv"


# ============================================================
# GENERIC HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dirs() -> None:
    """
    Erstellt die Basisstruktur.
    Account-spezifische Ordner werden zusätzlich dynamisch beim Speichern erstellt.
    """
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for account in ["DEMO_ACCOUNT", "LIVE_ACCOUNT", "UNKNOWN_ACCOUNT"]:
        (OUTPUT_ROOT / account / "IS").mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / account / "OOS").mkdir(parents=True, exist_ok=True)


def safe_filename_part(value: str) -> str:
    s = str(value).strip()
    s = re.sub(r"[^\w\-.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s)
    return s.strip("._") or "UNKNOWN"


def discover_input_files() -> List[Path]:
    if not INPUT_ROOT.exists():
        log_warn(f"INPUT_ROOT existiert nicht: {INPUT_ROOT}")
        return []

    files = sorted([p for p in INPUT_ROOT.rglob("*.csv") if p.is_file()])
    return files


def read_csv_robust(file: Path) -> pd.DataFrame:
    """
    Liest CSV robust mit mehreren Encodings/Separator-Varianten.
    Standardmäßig wird zuerst pandas Auto-Detection verwendet.
    """
    attempts = [
        {"encoding": "utf-8-sig", "sep": None, "engine": "python"},
        {"encoding": "utf-8", "sep": None, "engine": "python"},
        {"encoding": "cp1252", "sep": None, "engine": "python"},
        {"encoding": "latin1", "sep": None, "engine": "python"},
    ]

    last_error: Exception | None = None

    for kwargs in attempts:
        try:
            return pd.read_csv(file, **kwargs)
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"CSV konnte nicht gelesen werden: {file} | Fehler={last_error}")


# ============================================================
# ACCOUNT / STRATEGY DETECTION
# ============================================================

def detect_account(file: Path) -> str:
    """
    Account wird primär aus dem ersten Unterordner unter INPUT_ROOT erkannt.

    Beispiel:
        INPUT_ROOT/DEMO_ACCOUNT/file.csv -> DEMO_ACCOUNT
        INPUT_ROOT/LIVE_ACCOUNT/file.csv -> LIVE_ACCOUNT

    Fallback:
        Dateiname/Path enthält DEMO -> DEMO_ACCOUNT
        Dateiname/Path enthält LIVE -> LIVE_ACCOUNT
    """
    try:
        parts = file.relative_to(INPUT_ROOT).parts
        if parts:
            candidate = str(parts[0]).strip()
            if candidate:
                return safe_filename_part(candidate.upper())
    except Exception:
        pass

    path_text = str(file).upper()
    if "DEMO" in path_text:
        return "DEMO_ACCOUNT"
    if "LIVE" in path_text:
        return "LIVE_ACCOUNT"

    return "UNKNOWN_ACCOUNT"


def strategy_id(file: Path) -> str:
    """
    Extrahiert Strategy-ID aus StrategyQuant/listOfTrades-Dateinamen.

    Unterstützte Beispiele:
    - listOfTrades_XXX_123_5.20.109_BUY_AUDJPY
    - listOfTrades_5.20.109_AUDJPY_BUY
    """
    stem = file.stem

    patterns = [
        r"listOfTrades_[^_]+_\d+_([0-9]+(?:\.[0-9]+)+)_(BUY|SELL|BOTH)_[A-Za-z0-9.]+$",
        r"listOfTrades_([0-9]+(?:\.[0-9]+)+)",
    ]

    for pattern in patterns:
        m = re.search(pattern, stem, re.IGNORECASE)
        if m:
            return str(m.group(1))

    return "UNKNOWN"


# ============================================================
# VALIDATION / NORMALIZATION
# ============================================================

def validate_input_schema(df: pd.DataFrame, file: Path) -> None:
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Pflichtspalten fehlen: {missing} | Datei={file}")


def validate_output_schema(df: pd.DataFrame, file: Path) -> None:
    missing = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Output-Spalten fehlen: {missing} | Datei={file}")


def parse_dt(series: pd.Series) -> pd.Series:
    """
    Standardformat der Backtestdaten:
        DD.MM.YYYY HH:MM:SS

    Fallback:
        pandas flexible parsing
    """
    parsed = pd.to_datetime(
        series,
        format="%d.%m.%Y %H:%M:%S",
        errors="coerce",
    )

    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce")

    return parsed


def to_utc_str(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")

    # Naive Timestamps werden als UTC interpretiert.
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize("UTC")
    else:
        dt = dt.dt.tz_convert("UTC")

    return dt.astype(str)


def normalize_direction(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "LONG": "BUY",
                "SHORT": "SELL",
                "BUY LIMIT": "BUY",
                "SELL LIMIT": "SELL",
                "BUY STOP": "BUY",
                "SELL STOP": "SELL",
            }
        )
    )


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def convert_to_baseline_schema(df: pd.DataFrame, file: Path) -> pd.DataFrame:
    validate_input_schema(df, file)

    d = df.copy()

    d["Open time"] = parse_dt(d["Open time"])
    d["Close time"] = parse_dt(d["Close time"])

    processed_at = utc_now_iso()
    account = detect_account(file)

    try:
        rel_path = str(file.relative_to(INPUT_ROOT))
    except Exception:
        rel_path = str(file)

    out = pd.DataFrame()
    out["position_id"] = d["Ticket"].astype(str).str.strip()
    out["symbol"] = d["Symbol"].astype(str).str.strip()
    out["direction"] = normalize_direction(d["Type"])

    out["open_time_utc"] = to_utc_str(d["Open time"])
    out["close_time_utc"] = to_utc_str(d["Close time"])

    out["entry_price"] = safe_num(d["Open price"])
    out["exit_price"] = safe_num(d["Close price"])
    out["price_delta"] = out["exit_price"] - out["entry_price"]

    out["profit_sum"] = safe_num(d["Profit/Loss"])
    out["net_sum"] = out["profit_sum"]

    out["strategy_id"] = strategy_id(file)
    out["account_type"] = account
    out["sample_type"] = ""
    out["source_file"] = file.name
    out["source_rel_path"] = rel_path
    out["processed_at_utc"] = processed_at

    open_dt = pd.to_datetime(out["open_time_utc"], utc=True, errors="coerce")
    close_dt = pd.to_datetime(out["close_time_utc"], utc=True, errors="coerce")

    valid_mask = (
        open_dt.notna()
        & close_dt.notna()
        & out["position_id"].astype(str).str.len().gt(0)
        & out["symbol"].astype(str).str.len().gt(0)
    )

    out = out[valid_mask].copy()

    out = out.sort_values(
        ["open_time_utc", "close_time_utc", "position_id"]
    ).reset_index(drop=True)

    validate_output_schema(out, file)
    return out[OUTPUT_COLUMNS].copy()


# ============================================================
# SPLIT IS / OOS
# ============================================================

def split_is_oos(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dt = pd.to_datetime(df["open_time_utc"], utc=True, errors="coerce")

    is_df = df[dt <= IS_CUTOFF].copy()
    oos_df = df[dt > IS_CUTOFF].copy()

    is_df["sample_type"] = "IS"
    oos_df["sample_type"] = "OOS"

    return is_df, oos_df


# ============================================================
# SAVE / REPORTS
# ============================================================

def save_output(df: pd.DataFrame, source_file: Path, account: str, sample_type: str) -> Path:
    out_dir = OUTPUT_ROOT / safe_filename_part(account) / sample_type
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{safe_filename_part(source_file.stem)}_{sample_type}.csv"
    df.to_csv(out_file, index=False)

    log_ok(f"SAVED {sample_type}: {out_file}")
    return out_file


def build_summary_row(
    file: Path,
    converted: pd.DataFrame,
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    account: str,
    is_path: Path | None,
    oos_path: Path | None,
) -> Dict[str, Any]:
    open_times = pd.to_datetime(converted["open_time_utc"], utc=True, errors="coerce") if not converted.empty else pd.Series(dtype="datetime64[ns, UTC]")
    close_times = pd.to_datetime(converted["close_time_utc"], utc=True, errors="coerce") if not converted.empty else pd.Series(dtype="datetime64[ns, UTC]")

    return {
        "processed_at_utc": utc_now_iso(),
        "source_file": file.name,
        "source_path": str(file),
        "account": account,
        "strategy_id": strategy_id(file),
        "rows_total": int(len(converted)),
        "rows_is": int(len(is_df)),
        "rows_oos": int(len(oos_df)),
        "sum_net_total": float(converted["net_sum"].sum()) if not converted.empty else 0.0,
        "sum_net_is": float(is_df["net_sum"].sum()) if not is_df.empty else 0.0,
        "sum_net_oos": float(oos_df["net_sum"].sum()) if not oos_df.empty else 0.0,
        "first_open_time_utc": str(open_times.min()) if len(open_times) and pd.notna(open_times.min()) else "",
        "last_close_time_utc": str(close_times.max()) if len(close_times) and pd.notna(close_times.max()) else "",
        "is_output_path": str(is_path) if is_path else "",
        "oos_output_path": str(oos_path) if oos_path else "",
        "status": "ok",
    }


def write_report(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log_ok(f"REPORT: {path}")


def build_error_row(file: Path, exc: Exception) -> Dict[str, Any]:
    return {
        "processed_at_utc": utc_now_iso(),
        "source_file": file.name,
        "source_path": str(file),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


# ============================================================
# PROCESS
# ============================================================

def process_file(file: Path) -> Dict[str, Any]:
    log_info(f"PROCESS: {file.name}")

    raw = read_csv_robust(file)
    converted = convert_to_baseline_schema(raw, file)

    if converted.empty:
        log_warn(f"SKIP EMPTY AFTER CONVERT: {file.name}")
        account = detect_account(file)
        return {
            "processed_at_utc": utc_now_iso(),
            "source_file": file.name,
            "source_path": str(file),
            "account": account,
            "strategy_id": strategy_id(file),
            "rows_total": 0,
            "rows_is": 0,
            "rows_oos": 0,
            "sum_net_total": 0.0,
            "sum_net_is": 0.0,
            "sum_net_oos": 0.0,
            "first_open_time_utc": "",
            "last_close_time_utc": "",
            "is_output_path": "",
            "oos_output_path": "",
            "status": "empty_after_convert",
        }

    is_df, oos_df = split_is_oos(converted)
    account = detect_account(file)

    is_path: Path | None = None
    oos_path: Path | None = None

    if not is_df.empty:
        is_path = save_output(is_df, file, account, "IS")
    else:
        log_warn(f"NO IS DATA: {file.name}")

    if not oos_df.empty:
        oos_path = save_output(oos_df, file, account, "OOS")
    else:
        log_warn(f"NO OOS DATA: {file.name}")

    return build_summary_row(
        file=file,
        converted=converted,
        is_df=is_df,
        oos_df=oos_df,
        account=account,
        is_path=is_path,
        oos_path=oos_path,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log_info("RUN BASELINE BACKTEST IS/OOS SPLITTER")
    log_info(f"SCRIPT_PATH = {SCRIPT_PATH}")
    log_info(f"QUANT_ROOT  = {QUANT_ROOT}")
    log_info(f"INPUT_ROOT  = {INPUT_ROOT}")
    log_info(f"OUTPUT_ROOT = {OUTPUT_ROOT}")
    log_info(f"IS_CUTOFF   = {IS_CUTOFF}")

    ensure_dirs()

    files = discover_input_files()
    log_info(f"FILES FOUND: {len(files)}")

    summary_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    ok_count = 0
    fail_count = 0

    for file in files:
        try:
            summary_rows.append(process_file(file))
            ok_count += 1
        except Exception as exc:
            fail_count += 1
            error_rows.append(build_error_row(file, exc))
            log_error(f"FAIL: {file.name} | {exc}")

    write_report(summary_rows, SUMMARY_REPORT_PATH)

    if error_rows:
        write_report(error_rows, ERROR_REPORT_PATH)
    else:
        if ERROR_REPORT_PATH.exists():
            ERROR_REPORT_PATH.unlink()
        log_ok("NO ERRORS")

    log_info("DONE")
    log_info(f"OK   = {ok_count}")
    log_info(f"FAIL = {fail_count}")


if __name__ == "__main__":
    main()

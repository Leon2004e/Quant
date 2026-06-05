# -*- coding: utf-8 -*-
"""
QUANT/Data_Center/Backend_Management/5_Catalog/Data_Catalog_Scanner/code.py

Zweck:
- Scannt den gesamten QUANT/Data_Center/Data Ordner nach Datenassets.
- Klassifiziert Dateien nach Stage, Domain, Asset Type, Symbol, Strategy ID, Account ID,
  Account Type und Sample Type.
- Erkennt Live-Trading-Daten aus:
    Data_Center/Data/1_Pipeline/Trades/Live/account_<id>_<LIVE|DEMO>/
- Erkennt SQLite-Datenbanken wie:
    closed_trades.db
    open_positions_current.db
    trades.db
    catalog.db
    code_registry.db
- Liest Preview-Metadaten für CSV/JSON/XLSX/Parquet/PKL.
- Liest SQLite-Metadaten für .db:
    - Tabellen
    - Zeilen je Tabelle
    - Spalten je Tabelle
    - Datumsbereich aus date/time/open_time/close_time/utc-Spalten
- Berechnet Qualitätsstatus, Zeilen/Spalten, Datumsbereich, Dateigröße, Checksum.
- Speichert alles in:
    Data_Center/Data/5_Catalog/catalog.db
- Erstellt pro Scan einen Eintrag in scan_runs.
- Nutzt UPSERT, damit bestehende Assets aktualisiert werden.

Wichtig:
- Dieser Code ist ein DATA CATALOG SCANNER.
- Er normalisiert keine Trades nach 2_Baseline.
- Er macht Live-Daten im Catalog sichtbar.
- Für strukturiertes Speichern nach 2_Baseline/Trades/Live brauchst du zusätzlich
  den Trade Loader / Normalizer.

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: data_catalog_scanner
# script_name: Data Catalog Scanner
# owner: Leon Everts
# status: active
# layer: 5_Catalog
# domain: Catalog
# asset_type: Data_Catalog
# purpose: Scan QUANT data folders, register data assets, classify files, compute metadata and quality scores, and store results in catalog.db.
# inputs:
#   - Data_Center/Data
# outputs:
#   - Data_Center/Data/5_Catalog/catalog.db
# upstream_data:
#   - Data_Center/Data/1_Pipeline
#   - Data_Center/Data/2_Baseline
#   - Data_Center/Data/3_Research
#   - Data_Center/Data/4_Production
#   - Data_Center/Data/5_Catalog
#   - Data_Center/Data/6_Code_Registry
# downstream_data:
#   - Data Catalog Dashboard
#   - Code Registry Dashboard
#   - Data Quality Monitoring
#   - Research and Reporting
# dependencies:
#   - pathlib
#   - sqlite3
#   - hashlib
#   - pandas
#   - datetime
#   - re
#   - json
# schedule: manual
# version: v1.4.0_strategy_ea_scan
# last_reviewed: 2026-06-01
# business_criticality: high
# environment: desktop
# registry_group: data_catalog
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# tags:
#   - data_catalog
#   - metadata
#   - data_quality
#   - sqlite
#   - quant_data
#   - live_trades
#   - account_discovery
# notes:
#   - Scans supported file extensions including SQLite .db files.
#   - Detects account_<id>_LIVE and account_<id>_DEMO folders.
#   - Reads SQLite metadata from closed_trades.db and open_positions_current.db.
#   - Keeps the original CODE_REGISTRY concept inside the scanner.
#   - Uses schema migration so existing catalog.db files can be upgraded.
# ============================================================
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import re
import sqlite3

import pandas as pd


# ============================================================
# CODE REGISTRY - Runtime Metadata
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "data_catalog_scanner",
    "script_name": "Data Catalog Scanner",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "5_Catalog",
    "domain": "Catalog",
    "asset_type": "Data_Catalog",
    "purpose": (
        "Scan QUANT data folders, register data assets, classify files, compute "
        "metadata and quality scores, and store results in catalog.db."
    ),
    "inputs": ["Data_Center/Data"],
    "outputs": ["Data_Center/Data/5_Catalog/catalog.db"],
    "upstream_data": [
        "Data_Center/Data/1_Pipeline",
        "Data_Center/Data/2_Baseline",
        "Data_Center/Data/3_Research",
        "Data_Center/Data/4_Production",
        "Data_Center/Data/5_Catalog",
        "Data_Center/Data/6_Code_Registry",
    ],
    "downstream_data": [
        "Data Catalog Dashboard",
        "Code Registry Dashboard",
        "Data Quality Monitoring",
        "Research and Reporting",
    ],
    "dependencies": ["pathlib", "sqlite3", "hashlib", "pandas", "datetime", "re", "json"],
    "schedule": "manual",
    "version": "v1.4.0_strategy_ea_scan",
    "last_reviewed": "2026-06-01",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "data_catalog",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "tags": [
        "data_catalog",
        "metadata",
        "data_quality",
        "sqlite",
        "quant_data",
        "live_trades",
        "account_discovery",
    ],
    "notes": [
        "Scans supported file extensions including SQLite .db files.",
        "Detects account_<id>_LIVE and account_<id>_DEMO folders.",
        "Reads SQLite metadata from closed_trades.db, open_positions_current.db and trades.db.",
        "Does not move or normalize trades into Baseline; this scanner only catalogs data assets.",
        "Uses schema migration so existing catalog.db files can be upgraded.",
    ],
}


# ============================================================
# ROOT FINDER
# ============================================================

def find_quant_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p
    raise RuntimeError(
        "QUANT Root nicht gefunden. Erwartet Ordner mit Dashboard und Data_Center "
        f"| Start={start}"
    )


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
BACKEND_MANAGEMENT_DIR = DATA_CENTER_DIR / "Backend_Management"

CATALOG_DIR = DATA_DIR / "5_Catalog"
CATALOG_DB = CATALOG_DIR / "catalog.db"


# ============================================================
# CONFIG
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".txt",
    ".xlsx",
    ".parquet",
    ".pkl",
    ".db",
    ".mq5",
    ".mq4",
    ".ex5",
    ".ex4",
    ".set",
    ".ini",
}

IGNORE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}

DATE_COLUMN_KEYWORDS = ["date", "time", "open_time", "close_time", "timestamp", "utc"]

KNOWN_SYMBOLS = [
    "AUDJPY", "CADJPY", "CHFJPY", "EURJPY", "GBPJPY", "NZDJPY", "USDJPY",
    "EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCAD", "USDCHF",
    "XAUUSD", "XAGUSD",
    "USOIL", "UKOIL", "OIL", "CRUDE",
    "US500", "US100", "US30", "GER40", "DAX", "EU50", "US2000",
    "BTCUSD", "ETHUSD",
]

STRATEGY_EA_DIR_CANDIDATES = [
    DATA_DIR / "1_Pipeline" / "Strategy" / "Strategy_EA",
    DATA_DIR / "1_Pipeline" / "Strategies" / "Strategy_EA",
    DATA_DIR / "1_Pipeline" / "Strategy_EA",
]

STRATEGY_EA_EXTENSIONS = {".mq5", ".mq4", ".ex5", ".ex4", ".set", ".ini"}

EA_FILENAME_PATTERN = re.compile(
    r"^(?P<symbol>[A-Z0-9.]+)_(?P<magic>\d+)_(?P<strategy_id>\d+(?:\.\d+)+)_(?P<side>BUY|SELL|BOTH)_(?P<timeframe>[A-Za-z0-9]+)$",
    re.IGNORECASE,
)


# ============================================================
# LOGGING
# ============================================================

def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_ok(message: str) -> None:
    print(f"[OK] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


# ============================================================
# DATABASE SCHEMA
# ============================================================

DATA_ASSETS_COLUMNS: Dict[str, str] = {
    "asset_name": "TEXT",
    "file_name": "TEXT",
    "file_extension": "TEXT",
    "file_path": "TEXT UNIQUE",
    "relative_path": "TEXT",
    "stage": "TEXT",
    "domain": "TEXT",
    "asset_type": "TEXT",
    "symbol": "TEXT",
    "strategy_id": "TEXT",
    "account_id": "TEXT",
    "account_type": "TEXT",
    "sample_type": "TEXT",
    "ea_magic": "TEXT",
    "ea_side": "TEXT",
    "ea_timeframe": "TEXT",
    "sqlite_tables": "TEXT",
    "sqlite_table_count": "INTEGER",
    "primary_table": "TEXT",
    "rows_count": "INTEGER",
    "columns_count": "INTEGER",
    "columns_list": "TEXT",
    "date_start": "TEXT",
    "date_end": "TEXT",
    "file_size_mb": "REAL",
    "last_modified_utc": "TEXT",
    "checksum": "TEXT",
    "quality_status": "TEXT",
    "quality_score": "REAL",
    "quality_message": "TEXT",
    "scanned_at_utc": "TEXT",
}

SCAN_RUNS_COLUMNS: Dict[str, str] = {
    "scanned_at_utc": "TEXT",
    "quant_root": "TEXT",
    "data_root": "TEXT",
    "files_found": "INTEGER",
    "files_ok": "INTEGER",
    "files_failed": "INTEGER",
    "code_registry_json": "TEXT",
}


def connect_db() -> sqlite3.Connection:
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CATALOG_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return [str(row[1]) for row in cur.fetchall()]


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, definition: str) -> None:
    cols = set(table_columns(conn, table_name))
    if column_name not in cols:
        cur = conn.cursor()
        cur.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
        conn.commit()
        log_info(f"SQLite migration: added {table_name}.{column_name}")


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS data_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            file_name TEXT,
            file_extension TEXT,
            file_path TEXT UNIQUE,
            relative_path TEXT,
            stage TEXT,
            domain TEXT,
            asset_type TEXT,
            symbol TEXT,
            strategy_id TEXT,
            account_id TEXT,
            account_type TEXT,
            sample_type TEXT,
            ea_magic TEXT,
            ea_side TEXT,
            ea_timeframe TEXT,
            sqlite_tables TEXT,
            sqlite_table_count INTEGER,
            primary_table TEXT,
            rows_count INTEGER,
            columns_count INTEGER,
            columns_list TEXT,
            date_start TEXT,
            date_end TEXT,
            file_size_mb REAL,
            last_modified_utc TEXT,
            checksum TEXT,
            quality_status TEXT,
            quality_score REAL,
            quality_message TEXT,
            scanned_at_utc TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at_utc TEXT,
            quant_root TEXT,
            data_root TEXT,
            files_found INTEGER,
            files_ok INTEGER,
            files_failed INTEGER,
            code_registry_json TEXT
        )
    """)
    conn.commit()

    for col, definition in DATA_ASSETS_COLUMNS.items():
        ensure_column(conn, "data_assets", col, definition)
    for col, definition in SCAN_RUNS_COLUMNS.items():
        ensure_column(conn, "scan_runs", col, definition)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_stage_domain ON data_assets(stage, domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_symbol_strategy ON data_assets(symbol, strategy_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_account ON data_assets(account_id, account_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_asset_type ON data_assets(asset_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_ea_magic ON data_assets(ea_magic, ea_side, ea_timeframe)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_quality ON data_assets(quality_status, quality_score)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_data_assets_relative_path ON data_assets(relative_path)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scan_runs_scanned_at ON scan_runs(scanned_at_utc)")
    conn.commit()


# ============================================================
# BASIC HELPERS
# ============================================================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def file_checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIR_NAMES for part in path.parts)


def discover_files() -> List[Path]:
    """
    Scans all supported files inside Data_Center/Data.

    Strategy EA support:
    - .mq5 / .mq4 source files
    - .ex5 / .ex4 compiled files
    - .set / .ini parameter/config files

    Expected EA path:
        Data_Center/Data/1_Pipeline/Strategy/Strategy_EA/**/*.mq5
    """
    if not DATA_DIR.exists():
        return []

    files: List[Path] = []
    seen: set[str] = set()

    def add_file(p: Path) -> None:
        if not p.is_file():
            return
        if should_ignore(p):
            return
        if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return
        try:
            if p.resolve() == CATALOG_DB.resolve():
                return
        except Exception:
            pass
        key = str(p.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        files.append(p)

    for p in DATA_DIR.rglob("*"):
        add_file(p)

    for ea_root in STRATEGY_EA_DIR_CANDIDATES:
        if not ea_root.exists():
            continue
        for p in ea_root.rglob("*"):
            add_file(p)

    return sorted(files)


def is_date_like_column(column_name: str) -> bool:
    c = str(column_name).lower()
    return any(k in c for k in DATE_COLUMN_KEYWORDS)


# ============================================================
# CLASSIFICATION
# ============================================================

def detect_stage(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if "1_Pipeline" in parts:
        return "Pipeline"
    if "2_Baseline" in parts:
        return "Baseline"
    if "3_Research" in parts:
        return "Research"
    if "4_Production" in parts:
        return "Production"
    if "5_Catalog" in parts:
        return "Catalog"
    if "6_Code_Registry" in parts:
        return "Code_Registry"
    if "3_Features" in parts:
        return "Features"
    if "4_Analytics" in parts:
        return "Analytics"
    if "5_Reports" in parts:
        return "Reports"
    return "Unknown"


def detect_domain(relative_path: str) -> str:
    p = relative_path.lower()
    if "trades" in p:
        return "Trades"
    if "market" in p:
        return "Market"
    if "portfolio" in p:
        return "Portfolio"
    if "strategy" in p:
        return "Strategy"
    if "risk" in p:
        return "Risk"
    if "analytics" in p:
        return "Analytics"
    if "catalog" in p:
        return "Catalog"
    if "code_registry" in p or "code registry" in p:
        return "Code_Registry"
    return "Unknown"


def detect_asset_type(relative_path: str, file_name: str) -> str:
    p = relative_path.lower()
    f = file_name.lower()

    if f.endswith((".mq5", ".mq4", ".ex5", ".ex4")):
        return "Strategy_EA"
    if f.endswith(".set"):
        return "EA_Settings"
    if f.endswith(".ini"):
        return "EA_Config"
    if "strategy_ea" in p:
        return "Strategy_EA"

    if f == "closed_trades.db":
        return "Closed_Trades"
    if f == "open_positions_current.db":
        return "Open_Positions"
    if f == "trades.db":
        return "Normalized_Trades"
    if f == "catalog.db":
        return "Catalog_DB"
    if f == "code_registry.db":
        return "Code_Registry_DB"
    if "backtest" in p:
        return "Backtest"
    if "live" in p:
        return "Live"
    if "quant_analyzer" in p:
        return "Quant_Analyzer"
    if "closed_trades" in f:
        return "Closed_Trades"
    if "open_positions" in f:
        return "Open_Positions"
    if "spread" in p or "spread" in f:
        return "Spread_Data"
    if "ohlc" in p or "ohcl" in p:
        return "OHLC"
    if "daily" in f or "d1" in f:
        return "Market_Daily"
    if "h1" in f or "hour" in f:
        return "Market_Hourly"
    if "m15" in f or "m5" in f or "m1" in f:
        return "Market_Intraday"
    if "registry" in p or "registry" in f:
        return "Registry"
    return "Unknown"


def detect_account(relative_path: str) -> Dict[str, str]:
    m = re.search(r"account_(\d+)_(LIVE|DEMO)", relative_path, re.IGNORECASE)
    if not m:
        return {"account_id": "", "account_type": ""}
    return {"account_id": m.group(1), "account_type": m.group(2).upper()}


def detect_account_type(relative_path: str) -> str:
    meta = detect_account(relative_path)
    if meta["account_type"]:
        return meta["account_type"]
    p = relative_path.upper()
    if "DEMO_ACCOUNT" in p or "_DEMO" in p or "DEMO" in p:
        return "DEMO"
    if "LIVE_ACCOUNT" in p or "_LIVE" in p or "LIVE" in p:
        return "LIVE"
    return ""


def detect_sample_type(file_name: str, relative_path: str) -> str:
    text = f"{file_name} {relative_path}".upper()
    if "_IS" in text or "/IS/" in text or "\\IS\\" in text:
        return "IS"
    if "_OOS" in text or "/OOS/" in text or "\\OOS\\" in text:
        return "OOS"
    return ""


def detect_strategy_id(file_name: str) -> str:
    ea = parse_strategy_ea_filename(file_name)
    if ea.get("strategy_id"):
        return ea["strategy_id"]

    m = re.search(r"([0-9]+(?:\.[0-9]+)+)", file_name)
    return m.group(1) if m else ""


def parse_strategy_ea_filename(file_name: str) -> Dict[str, str]:
    """
    Parses Strategy EA filename format:
        <SYMBOL>_<MAGIC>_<STRATEGY_ID>_<SIDE>_<TIMEFRAME>.mq5

    Example:
        EURGBP_123456_5.20.109_BOTH_M1.mq5
    """
    stem = Path(str(file_name)).stem.strip()
    m = EA_FILENAME_PATTERN.match(stem)
    if not m:
        return {
            "symbol": "",
            "magic": "",
            "strategy_id": "",
            "side": "",
            "timeframe": "",
        }

    return {
        "symbol": m.group("symbol").upper(),
        "magic": m.group("magic"),
        "strategy_id": m.group("strategy_id"),
        "side": m.group("side").upper(),
        "timeframe": m.group("timeframe").upper(),
    }


def detect_symbol(file_name: str, relative_path: str) -> str:
    ea = parse_strategy_ea_filename(file_name)
    if ea.get("symbol"):
        return ea["symbol"]

    text = f"{file_name} {relative_path}".upper()
    for sym in KNOWN_SYMBOLS:
        if sym in text:
            return sym
    return ""


# ============================================================
# FILE READ / METADATA
# ============================================================

def _safe_read_dataframe(path: Path) -> Optional[pd.DataFrame]:
    ext = path.suffix.lower()
    if ext == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin1")
    if ext == ".json":
        return pd.read_json(path)
    if ext == ".xlsx":
        return pd.read_excel(path)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".pkl":
        obj = pd.read_pickle(path)
        if isinstance(obj, pd.DataFrame):
            return obj
        return None
    return None


def base_metadata_result() -> Dict[str, Any]:
    return {
        "rows_count": None,
        "columns_count": None,
        "columns_list": "",
        "date_start": "",
        "date_end": "",
        "sqlite_tables": "",
        "sqlite_table_count": 0,
        "primary_table": "",
        "quality_status": "passed",
        "quality_score": 100.0,
        "quality_message": "OK",
    }


def read_sqlite_metadata(path: Path) -> Dict[str, Any]:
    result = base_metadata_result()
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(path)
        tables_df = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """,
            conn,
        )
        tables = tables_df["name"].astype(str).tolist()
        result["sqlite_table_count"] = int(len(tables))
        if not tables:
            result["quality_status"] = "warning"
            result["quality_score"] = 60.0
            result["quality_message"] = "SQLite DB enthält keine Tabellen"
            return result

        table_summaries: List[str] = []
        total_rows = 0
        max_cols = 0
        all_cols: List[str] = []
        detected_dates: List[pd.Timestamp] = []
        primary_table = ""

        for table in tables:
            safe_table = table.replace('"', '""')
            try:
                row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone()[0])
            except Exception:
                row_count = 0
            try:
                info_df = pd.read_sql_query(f'PRAGMA table_info("{safe_table}")', conn)
                cols = info_df["name"].astype(str).tolist() if not info_df.empty else []
            except Exception:
                cols = []

            table_summaries.append(f"{table}:{row_count}")
            total_rows += row_count
            max_cols = max(max_cols, len(cols))
            all_cols.extend([f"{table}.{c}" for c in cols])
            if not primary_table or row_count > 0:
                primary_table = table

            date_cols = [c for c in cols if is_date_like_column(c)]
            for col in date_cols:
                safe_col = col.replace('"', '""')
                try:
                    query = (
                        f'SELECT "{safe_col}" AS dt '
                        f'FROM "{safe_table}" '
                        f'WHERE "{safe_col}" IS NOT NULL '
                        f'LIMIT 10000'
                    )
                    sample = pd.read_sql_query(query, conn)
                    if sample.empty:
                        continue
                    s = pd.to_datetime(sample["dt"], errors="coerce", utc=True)
                    s = s.dropna()
                    if not s.empty:
                        detected_dates.append(s.min())
                        detected_dates.append(s.max())
                except Exception:
                    continue

        result["sqlite_tables"] = ",".join(table_summaries)
        result["primary_table"] = primary_table
        result["rows_count"] = int(total_rows)
        result["columns_count"] = int(max_cols)
        result["columns_list"] = ",".join(all_cols)
        if detected_dates:
            result["date_start"] = str(min(detected_dates))
            result["date_end"] = str(max(detected_dates))
        if total_rows == 0:
            result["quality_status"] = "warning"
            result["quality_score"] = 70.0
            result["quality_message"] = "SQLite DB enthält Tabellen, aber keine Zeilen"
    except Exception as exc:
        result["quality_status"] = "failed"
        result["quality_score"] = 0.0
        result["quality_message"] = f"SQLite read error: {exc}"
    finally:
        if conn is not None:
            conn.close()
    return result


def read_preview_metadata(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".db":
        return read_sqlite_metadata(path)

    if path.suffix.lower() in STRATEGY_EA_EXTENSIONS:
        result = base_metadata_result()
        try:
            if path.suffix.lower() in {".mq5", ".mq4", ".set", ".ini"}:
                raw = path.read_text(encoding="utf-8", errors="ignore")
                lines = raw.splitlines()
                result["rows_count"] = len(lines)
                result["columns_count"] = 1
                result["columns_list"] = "source_text"
                result["quality_status"] = "passed" if path.stat().st_size > 0 else "warning"
                result["quality_score"] = 100.0 if path.stat().st_size > 0 else 70.0
                result["quality_message"] = "Strategy EA file scanned"
            else:
                result["rows_count"] = 1
                result["columns_count"] = 1
                result["columns_list"] = "compiled_binary"
                result["quality_status"] = "passed"
                result["quality_score"] = 100.0
                result["quality_message"] = "Compiled Strategy EA file scanned"
        except Exception as exc:
            result["quality_status"] = "warning"
            result["quality_score"] = 60.0
            result["quality_message"] = f"Strategy EA metadata read warning: {exc}"
        return result

    result = base_metadata_result()
    try:
        df = _safe_read_dataframe(path)
        if df is None:
            return result
        result["rows_count"] = int(len(df))
        result["columns_count"] = int(len(df.columns))
        result["columns_list"] = ",".join([str(c) for c in df.columns])

        quality_messages: List[str] = []
        score = 100.0
        if df.empty:
            quality_messages.append("Datei ist leer")
            score -= 70
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count > 0:
            quality_messages.append(f"Doppelte Zeilen: {duplicate_count}")
            score -= 10
        missing_cells = int(df.isna().sum().sum())
        if missing_cells > 0:
            quality_messages.append(f"Fehlende Werte: {missing_cells}")
            score -= 10

        date_cols = [c for c in df.columns if is_date_like_column(str(c))]
        detected_dates: List[pd.Timestamp] = []
        for col in date_cols:
            s = pd.to_datetime(df[col], errors="coerce", dayfirst=True, utc=True)
            s = s.dropna()
            if not s.empty:
                detected_dates.append(s.min())
                detected_dates.append(s.max())
        if detected_dates:
            result["date_start"] = str(min(detected_dates))
            result["date_end"] = str(max(detected_dates))

        if score >= 90:
            status = "passed"
        elif score >= 60:
            status = "warning"
        else:
            status = "failed"
        result["quality_status"] = status
        result["quality_score"] = max(score, 0.0)
        result["quality_message"] = " | ".join(quality_messages) if quality_messages else "OK"
    except Exception as exc:
        result["quality_status"] = "failed"
        result["quality_score"] = 0.0
        result["quality_message"] = f"Read error: {exc}"
    return result


def build_asset_record(path: Path) -> Dict[str, Any]:
    stat = path.stat()

    try:
        relative_path = str(path.relative_to(DATA_DIR))
    except ValueError:
        relative_path = str(path.relative_to(QUANT_ROOT))

    file_name = path.name
    asset_name = path.stem
    read_meta = read_preview_metadata(path)
    account_meta = detect_account(relative_path)
    ea_meta = parse_strategy_ea_filename(file_name)

    symbol = ea_meta.get("symbol") or detect_symbol(file_name, relative_path)
    strategy = ea_meta.get("strategy_id") or detect_strategy_id(file_name)

    return {
        "asset_name": asset_name,
        "file_name": file_name,
        "file_extension": path.suffix.lower(),
        "file_path": str(path),
        "relative_path": relative_path,
        "stage": detect_stage(relative_path),
        "domain": detect_domain(relative_path),
        "asset_type": detect_asset_type(relative_path, file_name),
        "symbol": symbol,
        "strategy_id": strategy,
        "account_id": account_meta["account_id"],
        "account_type": account_meta["account_type"] or detect_account_type(relative_path),
        "sample_type": detect_sample_type(file_name, relative_path),
        "ea_magic": ea_meta.get("magic", ""),
        "ea_side": ea_meta.get("side", ""),
        "ea_timeframe": ea_meta.get("timeframe", ""),
        "sqlite_tables": read_meta.get("sqlite_tables", ""),
        "sqlite_table_count": read_meta.get("sqlite_table_count", 0),
        "primary_table": read_meta.get("primary_table", ""),
        "rows_count": read_meta["rows_count"],
        "columns_count": read_meta["columns_count"],
        "columns_list": read_meta["columns_list"],
        "date_start": read_meta["date_start"],
        "date_end": read_meta["date_end"],
        "file_size_mb": round(stat.st_size / (1024 * 1024), 4),
        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "checksum": file_checksum(path),
        "quality_status": read_meta["quality_status"],
        "quality_score": read_meta["quality_score"],
        "quality_message": read_meta["quality_message"],
        "scanned_at_utc": utc_now_str(),
    }


# ============================================================
# UPSERT
# ============================================================

def upsert_asset(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO data_assets (
            asset_name,
            file_name,
            file_extension,
            file_path,
            relative_path,
            stage,
            domain,
            asset_type,
            symbol,
            strategy_id,
            account_id,
            account_type,
            sample_type,
            ea_magic,
            ea_side,
            ea_timeframe,
            sqlite_tables,
            sqlite_table_count,
            primary_table,
            rows_count,
            columns_count,
            columns_list,
            date_start,
            date_end,
            file_size_mb,
            last_modified_utc,
            checksum,
            quality_status,
            quality_score,
            quality_message,
            scanned_at_utc
        )
        VALUES (
            :asset_name,
            :file_name,
            :file_extension,
            :file_path,
            :relative_path,
            :stage,
            :domain,
            :asset_type,
            :symbol,
            :strategy_id,
            :account_id,
            :account_type,
            :sample_type,
            :ea_magic,
            :ea_side,
            :ea_timeframe,
            :sqlite_tables,
            :sqlite_table_count,
            :primary_table,
            :rows_count,
            :columns_count,
            :columns_list,
            :date_start,
            :date_end,
            :file_size_mb,
            :last_modified_utc,
            :checksum,
            :quality_status,
            :quality_score,
            :quality_message,
            :scanned_at_utc
        )
        ON CONFLICT(file_path) DO UPDATE SET
            asset_name=excluded.asset_name,
            file_name=excluded.file_name,
            file_extension=excluded.file_extension,
            relative_path=excluded.relative_path,
            stage=excluded.stage,
            domain=excluded.domain,
            asset_type=excluded.asset_type,
            symbol=excluded.symbol,
            strategy_id=excluded.strategy_id,
            account_id=excluded.account_id,
            account_type=excluded.account_type,
            sample_type=excluded.sample_type,
            ea_magic=excluded.ea_magic,
            ea_side=excluded.ea_side,
            ea_timeframe=excluded.ea_timeframe,
            sqlite_tables=excluded.sqlite_tables,
            sqlite_table_count=excluded.sqlite_table_count,
            primary_table=excluded.primary_table,
            rows_count=excluded.rows_count,
            columns_count=excluded.columns_count,
            columns_list=excluded.columns_list,
            date_start=excluded.date_start,
            date_end=excluded.date_end,
            file_size_mb=excluded.file_size_mb,
            last_modified_utc=excluded.last_modified_utc,
            checksum=excluded.checksum,
            quality_status=excluded.quality_status,
            quality_score=excluded.quality_score,
            quality_message=excluded.quality_message,
            scanned_at_utc=excluded.scanned_at_utc
    """, record)
    conn.commit()


def insert_scan_run(conn: sqlite3.Connection, files_found: int, files_ok: int, files_failed: int) -> None:
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO scan_runs (
            scanned_at_utc,
            quant_root,
            data_root,
            files_found,
            files_ok,
            files_failed,
            code_registry_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        utc_now_str(),
        str(QUANT_ROOT),
        str(DATA_DIR),
        int(files_found),
        int(files_ok),
        int(files_failed),
        json.dumps(CODE_REGISTRY, ensure_ascii=False),
    ))
    conn.commit()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log_info("RUN DATA CATALOG SCANNER")
    log_info(f"SCRIPT_PATH = {SCRIPT_PATH}")
    log_info(f"QUANT_ROOT  = {QUANT_ROOT}")
    log_info(f"DATA_DIR    = {DATA_DIR}")
    log_info(f"CATALOG_DB  = {CATALOG_DB}")
    log_info(f"CODE_REGISTRY.script_id = {CODE_REGISTRY['script_id']}")
    log_info(f"CODE_REGISTRY.version   = {CODE_REGISTRY['version']}")

    conn = connect_db()
    ok = 0
    failed = 0

    try:
        init_db(conn)
        files = discover_files()
        log_info(f"FILES FOUND = {len(files)}")
        for file in files:
            try:
                rel = file.relative_to(DATA_DIR)
                log_info(f"SCAN: {rel}")
                record = build_asset_record(file)
                upsert_asset(conn, record)
                ok += 1
            except Exception as exc:
                failed += 1
                log_warn(f"FAIL: {file} | {exc}")
        insert_scan_run(conn, len(files), ok, failed)

        try:
            ea_count = conn.execute(
                "SELECT COUNT(*) FROM data_assets WHERE asset_type IN ('Strategy_EA', 'EA_Settings', 'EA_Config')"
            ).fetchone()[0]
            log_ok(f"STRATEGY EA ASSETS = {ea_count}")
        except Exception:
            pass

        log_ok("DONE")
        log_ok(f"OK     = {ok}")
        log_ok(f"FAILED = {failed}")
        log_ok(f"DB     = {CATALOG_DB}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
'''
QUANT/Data_Center/Backend_Management/6_Code_Registry/Code_Registry_Scanner/code.py

Zweck:
- Scannt Python-Dateien im QUANT-Projekt
- Liest CODE_REGISTRY Header aus Python-Dateien
- Liest optional Runtime CODE_REGISTRY Dictionaries, wenn im Header Felder fehlen
- Speichert Code-Metadaten in SQLite
- Bewertet Registry-Qualität
- Erstellt Scan-Run-Historie
- Unterstützt Schema-Migrationen für bestehende code_registry.db

Output:
QUANT/Data_Center/Data/6_Code_Registry/code_registry.db

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: code_registry_scanner
# script_name: Code Registry Scanner
# owner: Leon Everts
# status: active
# layer: 6_Code_Registry
# domain: Catalog
# asset_type: Scanner
# purpose: Scan Python files, read CODE_REGISTRY headers and runtime metadata, validate registry quality, and store code metadata in SQLite.
# inputs:
#   - Data_Center/Backend_Management
#   - Dashboard
# outputs:
#   - Data_Center/Data/6_Code_Registry/code_registry.db
# upstream_data:
#   - Python files with CODE_REGISTRY headers
# downstream_data:
#   - Code Registry Dashboard
#   - System Documentation
#   - Dependency Mapping
#   - AI Code Navigation
# dependencies:
#   - pathlib
#   - sqlite3
#   - hashlib
#   - ast
# schedule: manual
# version: v1.1.0
# last_reviewed: 2026-06-01
# business_criticality: high
# environment: desktop
# registry_group: code_registry
# author: Leon Everts
# reviewer: ChatGPT
# created_date: 2026-06-01
# change_log:
#   - v1.0.0 - Initial code registry scanner
#   - v1.1.0 - Added runtime CODE_REGISTRY parsing, schema migrations, stronger quality checks, and QUANT path standardization
# tags:
#   - code_registry
#   - scanner
#   - metadata
#   - sqlite
#   - documentation
# notes:
#   - Scans Backend_Management and Dashboard by default.
#   - Files without CODE_REGISTRY are still inserted with quality_status=missing.
#   - Runtime CODE_REGISTRY dictionary is used as fallback when header parsing misses fields.
# ============================================================
'''

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# CODE REGISTRY - Runtime Metadata
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "code_registry_scanner",
    "script_name": "Code Registry Scanner",
    "owner": "Leon Everts",
    "status": "active",
    "layer": "6_Code_Registry",
    "domain": "Catalog",
    "asset_type": "Scanner",
    "purpose": (
        "Scan Python files, read CODE_REGISTRY headers and runtime metadata, "
        "validate registry quality, and store code metadata in SQLite."
    ),
    "inputs": ["Data_Center/Backend_Management", "Dashboard"],
    "outputs": ["Data_Center/Data/6_Code_Registry/code_registry.db"],
    "upstream_data": ["Python files with CODE_REGISTRY headers"],
    "downstream_data": [
        "Code Registry Dashboard",
        "System Documentation",
        "Dependency Mapping",
        "AI Code Navigation",
    ],
    "dependencies": ["pathlib", "sqlite3", "hashlib", "ast"],
    "schedule": "manual",
    "version": "v1.1.0",
    "last_reviewed": "2026-06-01",
    "business_criticality": "high",
    "environment": "desktop",
    "registry_group": "code_registry",
    "author": "Leon Everts",
    "reviewer": "ChatGPT",
    "created_date": "2026-06-01",
    "change_log": [
        "v1.0.0 - Initial code registry scanner",
        "v1.1.0 - Added runtime CODE_REGISTRY parsing, schema migrations, stronger quality checks, and QUANT path standardization",
    ],
    "tags": ["code_registry", "scanner", "metadata", "sqlite", "documentation"],
    "notes": [
        "Scans Backend_Management and Dashboard by default.",
        "Files without CODE_REGISTRY are still inserted with quality_status=missing.",
        "Runtime CODE_REGISTRY dictionary is used as fallback when header parsing misses fields.",
    ],
}


# ============================================================
# ROOT / PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_quant_root(start: Path) -> Path:
    cur = start.resolve()

    for p in [cur] + list(cur.parents):
        dashboard_dir = p / "Dashboard"
        data_center_dir = p / "Data_Center"
        data_dir = data_center_dir / "Data"
        backend_dir = data_center_dir / "Backend_Management"

        if dashboard_dir.exists() and data_center_dir.exists() and data_dir.exists() and backend_dir.exists():
            return p.resolve()

        if dashboard_dir.exists() and data_center_dir.exists():
            return p.resolve()

    raise RuntimeError(
        "QUANT Root nicht gefunden. Erwartet Ordner mit Dashboard und Data_Center. "
        f"Start={start}"
    )


QUANT_ROOT = find_quant_root(SCRIPT_PATH)

DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
BACKEND_DIR = DATA_CENTER_DIR / "Backend_Management"
DASHBOARD_DIR = QUANT_ROOT / "Dashboard"

OUTPUT_DIR = DATA_DIR / "6_Code_Registry"
DB_PATH = OUTPUT_DIR / "code_registry.db"


# ============================================================
# CONFIG
# ============================================================

SCAN_ROOTS: List[Path] = [BACKEND_DIR, DASHBOARD_DIR]

IGNORE_DIRS = {
    "__pycache__", ".git", ".venv", "venv", "env", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", "output", "outputs", "artifacts",
}

REQUIRED_FIELDS = [
    "script_id", "script_name", "status", "layer", "domain", "asset_type",
    "purpose", "inputs", "outputs", "dependencies", "version",
]

REGISTRY_FIELDS = [
    "script_id", "script_name", "owner", "status", "layer", "domain", "asset_type",
    "purpose", "inputs", "outputs", "upstream_data", "downstream_data", "dependencies",
    "schedule", "version", "last_reviewed", "business_criticality", "environment",
    "registry_group", "author", "reviewer", "created_date", "change_log", "tags", "notes",
]


# ============================================================
# DATABASE
# ============================================================

CODE_ASSET_COLUMNS: Dict[str, str] = {
    "script_id": "TEXT",
    "script_name": "TEXT",
    "owner": "TEXT",
    "status": "TEXT",
    "layer": "TEXT",
    "domain": "TEXT",
    "asset_type": "TEXT",
    "purpose": "TEXT",
    "inputs": "TEXT",
    "outputs": "TEXT",
    "upstream_data": "TEXT",
    "downstream_data": "TEXT",
    "dependencies": "TEXT",
    "schedule": "TEXT",
    "version": "TEXT",
    "last_reviewed": "TEXT",
    "business_criticality": "TEXT",
    "environment": "TEXT",
    "registry_group": "TEXT",
    "author": "TEXT",
    "reviewer": "TEXT",
    "created_date": "TEXT",
    "change_log": "TEXT",
    "tags": "TEXT",
    "notes": "TEXT",
    "file_name": "TEXT",
    "file_path": "TEXT UNIQUE",
    "relative_path": "TEXT",
    "file_size_kb": "REAL",
    "last_modified_utc": "TEXT",
    "checksum": "TEXT",
    "has_registry": "INTEGER",
    "has_runtime_registry": "INTEGER",
    "registry_quality_status": "TEXT",
    "registry_quality_score": "REAL",
    "registry_quality_message": "TEXT",
    "scanned_at_utc": "TEXT",
}


def connect_db() -> sqlite3.Connection:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
        conn.commit()
        print(f"[INFO] Migration applied: {table_name}.{column_name}")


def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS code_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            script_id TEXT,
            script_name TEXT,
            owner TEXT,
            status TEXT,
            layer TEXT,
            domain TEXT,
            asset_type TEXT,
            purpose TEXT,
            inputs TEXT,
            outputs TEXT,
            upstream_data TEXT,
            downstream_data TEXT,
            dependencies TEXT,
            schedule TEXT,
            version TEXT,
            last_reviewed TEXT,
            business_criticality TEXT,
            environment TEXT,
            registry_group TEXT,
            author TEXT,
            reviewer TEXT,
            created_date TEXT,
            change_log TEXT,
            tags TEXT,
            notes TEXT,
            file_name TEXT,
            file_path TEXT UNIQUE,
            relative_path TEXT,
            file_size_kb REAL,
            last_modified_utc TEXT,
            checksum TEXT,
            has_registry INTEGER,
            has_runtime_registry INTEGER,
            registry_quality_status TEXT,
            registry_quality_score REAL,
            registry_quality_message TEXT,
            scanned_at_utc TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS scan_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_at_utc TEXT,
            quant_root TEXT,
            backend_root TEXT,
            dashboard_root TEXT,
            db_path TEXT,
            files_found INTEGER,
            files_registered INTEGER,
            files_missing_registry INTEGER,
            files_warning INTEGER,
            files_failed INTEGER
        )
    """)

    conn.commit()

    for column, definition in CODE_ASSET_COLUMNS.items():
        ensure_column(conn, "code_assets", column, definition)

    scan_run_columns = {
        "scanned_at_utc": "TEXT",
        "quant_root": "TEXT",
        "backend_root": "TEXT",
        "dashboard_root": "TEXT",
        "db_path": "TEXT",
        "files_found": "INTEGER",
        "files_registered": "INTEGER",
        "files_missing_registry": "INTEGER",
        "files_warning": "INTEGER",
        "files_failed": "INTEGER",
    }

    for column, definition in scan_run_columns.items():
        ensure_column(conn, "scan_runs", column, definition)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_code_assets_script_id ON code_assets(script_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_code_assets_layer_domain ON code_assets(layer, domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_code_assets_quality ON code_assets(registry_quality_status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_code_assets_status ON code_assets(status)")

    conn.commit()


# ============================================================
# HELPERS
# ============================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def checksum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def should_ignore(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)


def discover_python_files() -> List[Path]:
    files: List[Path] = []

    for root in SCAN_ROOTS:
        if not root.exists():
            print(f"[WARN] Scan root missing: {root}")
            continue

        for p in root.rglob("*.py"):
            if not p.is_file():
                continue
            if should_ignore(p):
                continue
            files.append(p.resolve())

    return sorted(set(files))


def safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(QUANT_ROOT))
    except Exception:
        return str(path.resolve())


def list_to_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return "\n".join(str(x) for x in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    return str(value)


def normalize_record_text(value: Any) -> str:
    return list_to_text(value).strip()


# ============================================================
# REGISTRY PARSER
# ============================================================


def read_file_text(path: Path, max_chars: Optional[int] = None) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(max_chars) if max_chars else f.read()
        return text
    except Exception:
        return ""


def read_header_text(path: Path, max_lines: int = 220) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    except Exception:
        return ""


def extract_registry_block(text: str) -> str:
    if "CODE_REGISTRY" not in text:
        return ""

    lines = text.splitlines()
    start_idx = None

    for i, line in enumerate(lines):
        if "CODE_REGISTRY" in line:
            start_idx = max(0, i - 2)
            break

    if start_idx is None:
        return ""

    block_lines: List[str] = []
    separator_count_after_start = 0
    seen_registry = False

    for line in lines[start_idx:]:
        stripped = line.strip()
        block_lines.append(line)

        if "CODE_REGISTRY" in stripped:
            seen_registry = True

        if seen_registry and stripped.startswith("# ==="):
            separator_count_after_start += 1
            if separator_count_after_start >= 3:
                break

        if seen_registry and (stripped.startswith("from ") or stripped.startswith("import ")):
            break

        if len(block_lines) > 180:
            break

    return "\n".join(block_lines)


def clean_comment_line(line: str) -> str:
    line = line.strip()

    if line.startswith('\"\"\"') or line.startswith("'''"):
        line = line[3:].strip()

    if line.endswith('\"\"\"') or line.endswith("'''"):
        line = line[:-3].strip()

    if line.startswith("#"):
        line = line[1:].strip()

    return line


def empty_registry_fields() -> Dict[str, str]:
    return {field: "" for field in REGISTRY_FIELDS}


def parse_registry_block(block: str) -> Dict[str, str]:
    fields = empty_registry_fields()

    if not block:
        return fields

    current_key: Optional[str] = None
    multi_values: Dict[str, List[str]] = {k: [] for k in fields.keys()}

    for raw in block.splitlines():
        line = clean_comment_line(raw)

        if not line:
            continue

        if line.startswith("="):
            continue

        if line == "CODE_REGISTRY":
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key in fields:
                current_key = key

                if value:
                    multi_values[key].append(value)

                continue

        if current_key:
            item = line.strip()
            if item.startswith("-"):
                item = item[1:].strip()

            if item and not item.startswith("="):
                multi_values[current_key].append(item)

    for key, values in multi_values.items():
        fields[key] = "\n".join(values).strip()

    return fields


def parse_runtime_code_registry(path: Path) -> Dict[str, Any]:
    text = read_file_text(path)
    if not text or "CODE_REGISTRY" not in text:
        return {}

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "CODE_REGISTRY":
                try:
                    value = ast.literal_eval(node.value)
                except Exception:
                    return {}

                if isinstance(value, dict):
                    return value

    return {}


def merge_header_and_runtime_fields(header_fields: Dict[str, str], runtime_fields: Dict[str, Any]) -> Dict[str, str]:
    merged = dict(header_fields)

    for key in REGISTRY_FIELDS:
        if str(merged.get(key, "")).strip():
            continue

        if key in runtime_fields:
            merged[key] = normalize_record_text(runtime_fields.get(key))

    return merged


def detect_registry_quality(fields: Dict[str, str], has_registry: bool, has_runtime_registry: bool) -> Tuple[str, float, str]:
    if not has_registry and not has_runtime_registry:
        return "missing", 0.0, "No CODE_REGISTRY header or runtime CODE_REGISTRY dictionary found"

    missing = [k for k in REQUIRED_FIELDS if not str(fields.get(k, "")).strip()]

    total = len(REQUIRED_FIELDS)
    filled = total - len(missing)
    score = round((filled / total) * 100.0, 2)

    if missing:
        return "warning", score, "Missing required fields: " + ", ".join(missing)

    return "passed", 100.0, "OK"


# ============================================================
# RECORD BUILDING
# ============================================================


def build_record(path: Path) -> Dict[str, Any]:
    header_text = read_header_text(path)
    block = extract_registry_block(header_text)
    has_registry = bool(block)

    header_fields = parse_registry_block(block)
    runtime_fields = parse_runtime_code_registry(path)
    has_runtime_registry = bool(runtime_fields)

    fields = merge_header_and_runtime_fields(header_fields, runtime_fields)

    quality_status, quality_score, quality_message = detect_registry_quality(
        fields=fields,
        has_registry=has_registry,
        has_runtime_registry=has_runtime_registry,
    )

    stat = path.stat()

    record: Dict[str, Any] = {field: normalize_record_text(fields.get(field, "")) for field in REGISTRY_FIELDS}

    record.update({
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "relative_path": safe_relative(path),
        "file_size_kb": round(stat.st_size / 1024, 3),
        "last_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "checksum": checksum(path),
        "has_registry": 1 if has_registry else 0,
        "has_runtime_registry": 1 if has_runtime_registry else 0,
        "registry_quality_status": quality_status,
        "registry_quality_score": quality_score,
        "registry_quality_message": quality_message,
        "scanned_at_utc": utc_now(),
    })

    if not record["script_id"]:
        record["script_id"] = path.stem

    if not record["script_name"]:
        record["script_name"] = path.stem

    return record


# ============================================================
# UPSERT
# ============================================================


def upsert_code_asset(conn: sqlite3.Connection, record: Dict[str, Any]) -> None:
    columns = list(CODE_ASSET_COLUMNS.keys())
    placeholders = ", ".join([f":{c}" for c in columns])
    col_sql = ", ".join(columns)

    update_cols = [c for c in columns if c != "file_path"]
    update_sql = ",\n            ".join([f"{c}=excluded.{c}" for c in update_cols])

    sql = f"""
        INSERT INTO code_assets (
            {col_sql}
        )
        VALUES (
            {placeholders}
        )
        ON CONFLICT(file_path) DO UPDATE SET
            {update_sql}
    """

    safe_record = {c: record.get(c, None) for c in columns}
    conn.execute(sql, safe_record)
    conn.commit()


def insert_scan_run(
    conn: sqlite3.Connection,
    files_found: int,
    files_registered: int,
    files_missing_registry: int,
    files_warning: int,
    files_failed: int,
) -> None:
    conn.execute("""
        INSERT INTO scan_runs (
            scanned_at_utc,
            quant_root,
            backend_root,
            dashboard_root,
            db_path,
            files_found,
            files_registered,
            files_missing_registry,
            files_warning,
            files_failed
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        utc_now(),
        str(QUANT_ROOT),
        str(BACKEND_DIR),
        str(DASHBOARD_DIR),
        str(DB_PATH),
        int(files_found),
        int(files_registered),
        int(files_missing_registry),
        int(files_warning),
        int(files_failed),
    ))

    conn.commit()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("RUN CODE REGISTRY SCANNER")
    print(f"QUANT_ROOT = {QUANT_ROOT}")
    print(f"BACKEND    = {BACKEND_DIR}")
    print(f"DASHBOARD  = {DASHBOARD_DIR}")
    print(f"DB         = {DB_PATH}")

    conn = connect_db()

    try:
        init_db(conn)

        files = discover_python_files()
        print(f"FILES FOUND: {len(files)}")

        registered = 0
        missing = 0
        warning = 0
        failed = 0

        for file in files:
            try:
                rel = safe_relative(file)
                print(f"SCAN: {rel}")

                record = build_record(file)
                upsert_code_asset(conn, record)

                quality = str(record.get("registry_quality_status", ""))

                if quality == "passed":
                    registered += 1
                elif quality == "warning":
                    warning += 1
                elif quality == "missing":
                    missing += 1
                else:
                    warning += 1

            except Exception as exc:
                failed += 1
                print(f"[WARN] FAIL: {file} | {exc}")

        insert_scan_run(
            conn=conn,
            files_found=len(files),
            files_registered=registered,
            files_missing_registry=missing,
            files_warning=warning,
            files_failed=failed,
        )

        print("")
        print("DONE")
        print(f"REGISTERED = {registered}")
        print(f"WARNING    = {warning}")
        print(f"MISSING    = {missing}")
        print(f"FAILED     = {failed}")
        print(f"DB         = {DB_PATH}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()

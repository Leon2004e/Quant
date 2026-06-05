# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: pipeline_management_dashboard
# script_name: Pipeline Management Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Pipeline Management
# asset_type: Dashboard
# purpose: Control, monitor and inspect QUANT pipeline scripts from one dashboard building block with QUANT Terminal/Bloomberg-inspired UI.
# inputs:
#   - Data_Center/Backend_Management/1_Pipelines/
#   - Data_Center/Data/6_Code_Registry/code_registry.db
# outputs:
#   - Dashboard UI
#   - Data_Center/Data/4_Production/System_Runtime/Pipeline_Management/pipeline_state.json
#   - Data_Center/Data/4_Production/System_Runtime/Pipeline_Management/logs/
# dependencies:
#   - tkinter
#   - pathlib
#   - sqlite3
# schedule: manual
# version: v1.1.0_terminal_ui
# last_reviewed: 2026-06-01
# ============================================================

# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk, messagebox


# ============================================================
# ROOT DETECTION
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_quant_root(start: Path) -> Path:
    """
    Findet den QUANT Root.
    Der QUANT Root enthält:
    - Dashboard/
    - Data_Center/
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p.resolve()
    raise RuntimeError(
        f"QUANT Root nicht gefunden. Erwartet Ordner mit Dashboard/ und Data_Center/. Start={start}"
    )


QUANT_ROOT = find_quant_root(SCRIPT_PATH)
DASHBOARD_DIR = QUANT_ROOT / "Dashboard"
DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
BACKEND_DIR = DATA_CENTER_DIR / "Backend_Management"
DATA_DIR = DATA_CENTER_DIR / "Data"

PIPELINE_BACKEND_DIR = BACKEND_DIR / "1_Pipelines"
REGISTRY_DB = DATA_DIR / "6_Code_Registry" / "code_registry.db"

RUNTIME_DIR = DATA_DIR / "4_Production" / "System_Runtime" / "Pipeline_Management"
STATE_FILE = RUNTIME_DIR / "pipeline_state.json"
LOG_DIR = RUNTIME_DIR / "logs"


# ============================================================
# THEME - QUANT TERMINAL / BLOOMBERG-INSPIRED STYLE
# ============================================================
# Ziel: institutioneller Terminal-Look.
# Kein Logo, keine Marken-Assets, nur Bloomberg-inspirierte Terminal-Optik.

BG = "#000000"
BG_2 = "#050505"

PANEL = "#0A0A0A"
PANEL_2 = "#111111"
PANEL_3 = "#171717"
CARD = "#0D0D0D"

TABLE_BG = "#000000"
DETAIL_BG = "#050505"

BORDER = "#2A2A2A"
BORDER_2 = "#3A3A3A"

FG = "#FFFFFF"
MUTED = "#B8B8B8"
SUBTLE = "#7A7A7A"
WHITE = "#FFFFFF"

ORANGE = "#FF9900"
YELLOW = "#FFD400"
GREEN = "#00FF66"
RED = "#FF4444"
BLUE = "#00AEEF"
PURPLE = "#FF9900"
CYAN = "#00AEEF"

RUN_BG = "#06220F"
STOP_BG = "#2A1800"
MISS_BG = "#2A0000"
SELECT_BG = "#332000"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_H2 = ("Consolas", 10, "bold")
FONT_TEXT = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_MONO = ("Consolas", 9)

COMPACT_WIDTH = 1050
DETAILS_BREAKPOINT = 1220
MIN_LEFT_WIDTH = 260
MIN_MIDDLE_WIDTH = 560
MIN_RIGHT_WIDTH = 360




# ============================================================
# CODE_REGISTRY OBJECT
# ============================================================

CODE_REGISTRY: Dict[str, Any] = {
    "script_id": "pipeline_management_dashboard",
    "script_name": "Pipeline Management Dashboard",
    "owner": "Leon",
    "status": "active",
    "layer": "Dashboard",
    "domain": "Pipeline Management",
    "asset_type": "Dashboard",
    "purpose": (
        "Control, monitor and inspect QUANT pipeline scripts from one dashboard building block "
        "with QUANT Terminal/Bloomberg-inspired UI."
    ),
    "inputs": [
        "Data_Center/Backend_Management/1_Pipelines/",
        "Data_Center/Data/6_Code_Registry/code_registry.db",
    ],
    "outputs": [
        "Dashboard UI",
        "Data_Center/Data/4_Production/System_Runtime/Pipeline_Management/pipeline_state.json",
        "Data_Center/Data/4_Production/System_Runtime/Pipeline_Management/logs/",
    ],
    "dependencies": ["tkinter", "pathlib", "sqlite3"],
    "schedule": "manual",
    "version": "v1.1.0_terminal_ui",
    "last_reviewed": "2026-06-04",
}


def get_code_registry() -> Dict[str, Any]:
    return dict(CODE_REGISTRY)


# ============================================================
# PIPELINE DEFINITIONS
# ============================================================

PIPELINE_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "market_ohcl_logger",
        "label": "OHCL Logger",
        "category": "Market",
        "layer": "1_Pipelines",
        "domain": "Market Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Market/Ohcl_Logger/code.py",
            "Data_Center/Backend_Management/1_Pipelines/Market/OHLC_Logger/code.py",
            "Data_Center/Backend_Management/1_Pipelines/Market/Ohcl_Logger/Loader.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "market_spread_logger",
        "label": "Spread Logger",
        "category": "Market",
        "layer": "1_Pipelines",
        "domain": "Market Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Market/Spread_Logger/code.py",
            "Data_Center/Backend_Management/1_Pipelines/Market/Spread_Logger/Loader.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "ftmo_demo_1",
        "label": "FTMO DEMO 1",
        "category": "Trades",
        "layer": "1_Pipelines",
        "domain": "Trade Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Trades/FTMO_DEMO_1.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "ftmo_demo_2",
        "label": "FTMO DEMO 2",
        "category": "Trades",
        "layer": "1_Pipelines",
        "domain": "Trade Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Trades/FTMO_DEMO_2.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "ftmo_live_530164208",
        "label": "FTMO LIVE 530164208",
        "category": "Trades",
        "layer": "1_Pipelines",
        "domain": "Trade Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Trades/FTMO_LIVE_530164208.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "ftmo_live_540130486",
        "label": "FTMO LIVE 540130486",
        "category": "Trades",
        "layer": "1_Pipelines",
        "domain": "Trade Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Trades/FTMO_LIVE_540130486.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "ftmo_live_540136817",
        "label": "FTMO LIVE 540136817",
        "category": "Trades",
        "layer": "1_Pipelines",
        "domain": "Trade Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Trades/FTMO_LIVE_540136817.py",
        ],
        "auto_restart": False,
        "env": {},
    },
    {
        "id": "ftmo_live_540136824",
        "label": "FTMO LIVE 540136824",
        "category": "Trades",
        "layer": "1_Pipelines",
        "domain": "Trade Data",
        "asset_type": "Pipeline",
        "script_candidates": [
            "Data_Center/Backend_Management/1_Pipelines/Trades/FTMO_LIVE_540136824.py",
        ],
        "auto_restart": False,
        "env": {},
    },
]


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def shorten(text: str, max_len: int = 80) -> str:
    text = safe_text(text)
    if len(text) <= max_len:
        return text
    return "..." + text[-(max_len - 3):]


def split_lines(value: Any) -> List[str]:
    text = safe_text(value).strip()
    if not text:
        return []
    out = []
    for raw in text.replace("\r", "\n").split("\n"):
        item = raw.strip()
        if item.startswith("-"):
            item = item[1:].strip()
        if item:
            out.append(item)
    return out


def format_epoch(ts: Optional[float]) -> str:
    if not ts:
        return ""
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:
        return ""


def format_uptime(started_at: Optional[float], running: bool) -> str:
    if not started_at or not running:
        return ""
    sec = max(0, int(time.time() - started_at))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def resolve_script_path(cfg: Dict[str, Any]) -> Path:
    for candidate in cfg.get("script_candidates", []) or []:
        p = Path(str(candidate))
        if not p.is_absolute():
            p = QUANT_ROOT / p
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Script nicht gefunden für pipeline_id={cfg.get('id')} | candidates={cfg.get('script_candidates')}"
    )


def is_pid_running(pid: Optional[int]) -> bool:
    if pid is None or int(pid) <= 0:
        return False

    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                return code.value == 259
            finally:
                kernel32.CloseHandle(handle)
        else:
            os.kill(int(pid), 0)
            return True
    except Exception:
        return False


def stop_pid(pid: int, timeout_sec: float = 8.0) -> bool:
    if not is_pid_running(pid):
        return True

    if os.name == "nt":
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        if not is_pid_running(pid):
            return True
        time.sleep(0.25)

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

    time.sleep(0.5)
    return not is_pid_running(pid)


def build_log_file(pipeline_id: str) -> Path:
    ensure_dir(LOG_DIR)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{pipeline_id}_{ts}.log"


# ============================================================
# STATE
# ============================================================

@dataclass
class PipelineState:
    pipeline_id: str
    pid: Optional[int] = None
    running: bool = False
    started_at_epoch: Optional[float] = None
    last_exit_code: Optional[int] = None
    auto_restart: bool = False
    log_file: Optional[str] = None
    script_path: Optional[str] = None
    last_error: Optional[str] = None


def load_state_file() -> Dict[str, PipelineState]:
    ensure_dir(RUNTIME_DIR)
    ensure_dir(LOG_DIR)

    if not STATE_FILE.exists():
        return {}

    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[str, PipelineState] = {}
    for pipeline_id, payload in raw.items():
        try:
            out[pipeline_id] = PipelineState(**payload)
        except Exception:
            out[pipeline_id] = PipelineState(pipeline_id=pipeline_id, last_error="Invalid state payload")
    return out


def save_state_file(state: Dict[str, PipelineState]) -> None:
    ensure_dir(RUNTIME_DIR)
    tmp = STATE_FILE.with_suffix(".tmp")
    payload = {k: asdict(v) for k, v in state.items()}
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


# ============================================================
# REPOSITORY
# ============================================================

class PipelineRepository:
    def __init__(self, registry_db: Path = REGISTRY_DB):
        self.registry_db = registry_db
        self.pipeline_definitions = PIPELINE_DEFINITIONS
        self.state: Dict[str, PipelineState] = load_state_file()
        self._bootstrap_state()
        self.refresh_runtime_state(save=True)

    def _bootstrap_state(self) -> None:
        for cfg in self.pipeline_definitions:
            pipeline_id = str(cfg["id"])
            if pipeline_id not in self.state:
                script_path = ""
                err = None
                try:
                    script_path = str(resolve_script_path(cfg))
                except Exception as e:
                    script_path = str((cfg.get("script_candidates") or [""])[0])
                    err = str(e)
                self.state[pipeline_id] = PipelineState(
                    pipeline_id=pipeline_id,
                    auto_restart=bool(cfg.get("auto_restart", False)),
                    script_path=script_path,
                    last_error=err,
                )

    def refresh_runtime_state(self, save: bool = False) -> None:
        changed = False

        for pipeline_id, st in list(self.state.items()):
            alive = is_pid_running(st.pid)

            if st.running and not alive:
                st.running = False
                st.pid = None
                if st.last_exit_code is None:
                    st.last_exit_code = 0
                changed = True

                if st.auto_restart:
                    try:
                        self.start_pipeline(pipeline_id)
                    except Exception as e:
                        st.last_error = f"Auto-Restart failed: {e}"
                        changed = True

            elif alive and not st.running:
                st.running = True
                changed = True

        if save or changed:
            save_state_file(self.state)

    def load_registry_metadata(self) -> Dict[str, Dict[str, Any]]:
        if not self.registry_db.exists():
            return {}

        metadata: Dict[str, Dict[str, Any]] = {}

        try:
            conn = sqlite3.connect(self.registry_db)
            conn.row_factory = sqlite3.Row

            table_names = {
                row["name"]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "scripts" in table_names:
                rows = conn.execute("SELECT * FROM scripts").fetchall()
                for r in rows:
                    d = dict(r)
                    key_candidates = [
                        safe_text(d.get("relative_path")),
                        safe_text(d.get("file_path")),
                        safe_text(d.get("script_name")),
                        safe_text(d.get("script_id")),
                    ]
                    for key in key_candidates:
                        if key:
                            metadata[key.replace("\\", "/").lower()] = d

            if "code_assets" in table_names:
                rows = conn.execute("SELECT * FROM code_assets").fetchall()
                for r in rows:
                    d = dict(r)
                    key_candidates = [
                        safe_text(d.get("relative_path")),
                        safe_text(d.get("file_path")),
                        safe_text(d.get("script_name")),
                        safe_text(d.get("script_id")),
                    ]
                    for key in key_candidates:
                        if key:
                            metadata[key.replace("\\", "/").lower()] = d

            conn.close()
        except Exception:
            return {}

        return metadata

    def _find_registry_row(self, script_path: str, registry: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not script_path:
            return {}

        p = Path(script_path)
        keys = [
            str(p).replace("\\", "/").lower(),
            p.name.lower(),
        ]

        try:
            rel = str(p.relative_to(QUANT_ROOT)).replace("\\", "/").lower()
            keys.append(rel)
        except Exception:
            pass

        for key in keys:
            if key in registry:
                return registry[key]

        for key, row in registry.items():
            if key and (key.endswith(p.name.lower()) or p.name.lower() in key):
                return row

        return {}

    def load_data(self) -> List[Dict[str, Any]]:
        self.refresh_runtime_state(save=False)
        registry = self.load_registry_metadata()
        rows: List[Dict[str, Any]] = []

        for cfg in self.pipeline_definitions:
            pipeline_id = str(cfg["id"])
            st = self.state[pipeline_id]

            script_exists = False
            resolved_script = safe_text(st.script_path)
            missing_error = ""

            try:
                resolved_path = resolve_script_path(cfg)
                resolved_script = str(resolved_path)
                script_exists = True
            except Exception as e:
                missing_error = str(e)

            reg = self._find_registry_row(resolved_script, registry)

            row = {
                "id": pipeline_id,
                "label": safe_text(cfg.get("label", pipeline_id)),
                "category": safe_text(cfg.get("category", "")),
                "layer": safe_text(reg.get("layer") or cfg.get("layer", "")),
                "domain": safe_text(reg.get("domain") or cfg.get("domain", "")),
                "asset_type": safe_text(reg.get("asset_type") or cfg.get("asset_type", "")),
                "script_name": safe_text(reg.get("script_name") or Path(resolved_script).name),
                "script_id": safe_text(reg.get("script_id") or pipeline_id),
                "status": "RUNNING" if st.running else "STOPPED",
                "pid": safe_text(st.pid or ""),
                "uptime": format_uptime(st.started_at_epoch, st.running),
                "started_at": format_epoch(st.started_at_epoch),
                "auto_restart": "ON" if st.auto_restart else "OFF",
                "script_path": resolved_script,
                "script_exists": "YES" if script_exists else "NO",
                "log_file": safe_text(st.log_file or ""),
                "last_exit_code": safe_text(st.last_exit_code if st.last_exit_code is not None else ""),
                "last_error": safe_text(st.last_error or missing_error),
                "purpose": safe_text(reg.get("purpose") or "Pipeline process controlled by Pipeline Management Dashboard."),
                "inputs": safe_text(reg.get("inputs") or ""),
                "outputs": safe_text(reg.get("outputs") or ""),
                "dependencies": safe_text(reg.get("dependencies") or ""),
                "version": safe_text(reg.get("version") or ""),
                "last_reviewed": safe_text(reg.get("last_reviewed") or ""),
            }
            rows.append(row)

        return rows

    def start_pipeline(self, pipeline_id: str) -> None:
        cfg = next(x for x in self.pipeline_definitions if str(x["id"]) == pipeline_id)
        st = self.state[pipeline_id]

        if st.pid and is_pid_running(st.pid):
            raise RuntimeError(f"Pipeline läuft bereits: {pipeline_id} | PID={st.pid}")

        script_path = resolve_script_path(cfg)
        log_file = build_log_file(pipeline_id)

        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in dict(cfg.get("env", {})).items()})

        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

        log_handle = open(log_file, "a", encoding="utf-8", buffering=1)
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(script_path.parent),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
        finally:
            try:
                log_handle.close()
            except Exception:
                pass

        self.state[pipeline_id] = PipelineState(
            pipeline_id=pipeline_id,
            pid=int(proc.pid),
            running=True,
            started_at_epoch=time.time(),
            last_exit_code=None,
            auto_restart=bool(st.auto_restart),
            log_file=str(log_file),
            script_path=str(script_path),
            last_error=None,
        )
        save_state_file(self.state)

    def stop_pipeline(self, pipeline_id: str) -> None:
        st = self.state[pipeline_id]

        if st.pid:
            ok = stop_pid(int(st.pid))
            if not ok:
                st.last_error = f"Stop failed for PID={st.pid}"

        st.pid = None
        st.running = False
        st.last_exit_code = 0
        save_state_file(self.state)

    def restart_pipeline(self, pipeline_id: str) -> None:
        self.stop_pipeline(pipeline_id)
        time.sleep(0.5)
        self.start_pipeline(pipeline_id)

    def start_all(self) -> None:
        for cfg in self.pipeline_definitions:
            pipeline_id = str(cfg["id"])
            self.refresh_runtime_state(save=False)
            if not self.state[pipeline_id].running:
                try:
                    self.start_pipeline(pipeline_id)
                except Exception as e:
                    self.state[pipeline_id].last_error = str(e)
        save_state_file(self.state)

    def stop_all(self) -> None:
        for cfg in self.pipeline_definitions:
            pipeline_id = str(cfg["id"])
            if self.state[pipeline_id].running:
                try:
                    self.stop_pipeline(pipeline_id)
                except Exception as e:
                    self.state[pipeline_id].last_error = str(e)
        save_state_file(self.state)

    def set_auto_restart(self, pipeline_id: str, value: bool) -> None:
        self.state[pipeline_id].auto_restart = bool(value)
        save_state_file(self.state)

    def run_scanner(self) -> bool:
        return False


# ============================================================
# DASHBOARD
# ============================================================

class PipelineManagementDashboard(tk.Frame):
    TABLE_COLUMNS = [
        "label",
        "category",
        "status",
        "pid",
        "uptime",
        "auto_restart",
        "script_exists",
        "domain",
        "layer",
        "asset_type",
        "version",
    ]

    TABLE_HEADINGS = {
        "label": "Pipeline",
        "category": "Category",
        "status": "Status",
        "pid": "PID",
        "uptime": "Uptime",
        "auto_restart": "Auto Restart",
        "script_exists": "Script",
        "domain": "Domain",
        "layer": "Layer",
        "asset_type": "Asset Type",
        "version": "Version",
    }

    def __init__(self, parent, repository: Optional[PipelineRepository] = None, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)

        self.repository = repository or PipelineRepository()
        self.rows_all: List[Dict[str, Any]] = []
        self.rows_current: List[Dict[str, Any]] = []
        self.selected_id: Optional[str] = None
        self.selected_category = "ALL"

        self.search_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="ALL")
        self.summary_var = tk.StringVar(value="No data")
        self.status_var = tk.StringVar(value="Ready")
        self.auto_restart_var = tk.BooleanVar(value=False)

        self._responsive_mode: Optional[Tuple[str, str]] = None
        self._resize_after_id: Optional[str] = None
        self._refresh_after_id: Optional[str] = None

        self._setup_style()
        self._build_ui()
        self.refresh_data()
        self._schedule_refresh()

    # ========================================================
    # REQUIRED FUNCTIONS
    # ========================================================

    def load_data(self) -> List[Dict[str, Any]]:
        try:
            return self.repository.load_data()
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return []

    def refresh_data(self) -> None:
        self.rows_all = self.load_data()
        self._update_kpis()
        self._update_navigation()
        self._apply_filters()
        self._draw_flow()

    def update_table(self) -> None:
        self.table.delete(*self.table.get_children())

        self.table.tag_configure("running", background=RUN_BG, foreground=FG)
        self.table.tag_configure("stopped", background=STOP_BG, foreground=FG)
        self.table.tag_configure("missing", background=MISS_BG, foreground=FG)

        for row in self.rows_current:
            tag = "running" if row["status"] == "RUNNING" else "stopped"
            if row["script_exists"] != "YES":
                tag = "missing"

            values = [safe_text(row.get(col, "")) for col in self.TABLE_COLUMNS]
            self.table.insert("", "end", iid=row["id"], values=values, tags=(tag,))

        if self.selected_id and self.selected_id in self.table.get_children(""):
            self.table.selection_set(self.selected_id)
            self.table.focus(self.selected_id)
            self.table.see(self.selected_id)

    def update_details(self, row: Optional[Dict[str, Any]] = None) -> None:
        if row is None:
            row = self._get_selected_row()

        if row is None:
            self._set_details("Select a pipeline.")
            self.auto_restart_var.set(False)
            self._draw_flow()
            return

        pipeline_id = safe_text(row.get("id"))
        state = self.repository.state.get(pipeline_id)
        self.auto_restart_var.set(bool(state.auto_restart) if state else False)

        lines = [
            "PIPELINE",
            "-" * 70,
            f"pipeline_id   : {row.get('id', '')}",
            f"label         : {row.get('label', '')}",
            f"category      : {row.get('category', '')}",
            f"status        : {row.get('status', '')}",
            f"pid           : {row.get('pid', '')}",
            f"uptime        : {row.get('uptime', '')}",
            f"started_at    : {row.get('started_at', '')}",
            f"auto_restart  : {row.get('auto_restart', '')}",
            f"last_exit     : {row.get('last_exit_code', '')}",
            "",
            "REGISTRY / METADATA",
            "-" * 70,
            f"script_id     : {row.get('script_id', '')}",
            f"script_name   : {row.get('script_name', '')}",
            f"layer         : {row.get('layer', '')}",
            f"domain        : {row.get('domain', '')}",
            f"asset_type    : {row.get('asset_type', '')}",
            f"version       : {row.get('version', '')}",
            f"last_reviewed : {row.get('last_reviewed', '')}",
            "",
            "PURPOSE",
            "-" * 70,
            safe_text(row.get("purpose", "")),
            "",
            "INPUTS",
            "-" * 70,
            safe_text(row.get("inputs", "")) or "No inputs registered.",
            "",
            "OUTPUTS",
            "-" * 70,
            safe_text(row.get("outputs", "")) or "No outputs registered.",
            "",
            "DEPENDENCIES",
            "-" * 70,
            safe_text(row.get("dependencies", "")) or "No dependencies registered.",
            "",
            "LOCATION",
            "-" * 70,
            f"script_exists : {row.get('script_exists', '')}",
            f"script_path   : {row.get('script_path', '')}",
            f"log_file      : {row.get('log_file', '')}",
            "",
            "ERROR",
            "-" * 70,
            safe_text(row.get("last_error", "")) or "No error.",
        ]

        self._set_details("\n".join(lines))
        self._draw_flow(row)

    # ========================================================
    # UI BUILD
    # ========================================================

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            ".",
            background=BG,
            foreground=FG,
            fieldbackground=TABLE_BG,
            font=FONT_TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
        )

        style.configure(
            "Pipeline.Treeview",
            background=TABLE_BG,
            fieldbackground=TABLE_BG,
            foreground=FG,
            rowheight=21,
            borderwidth=1,
            relief="solid",
            font=FONT_SMALL,
        )
        style.configure(
            "Pipeline.Treeview.Heading",
            background=PANEL_3,
            foreground=ORANGE,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 8, "bold"),
        )
        style.map(
            "Pipeline.Treeview",
            background=[("selected", SELECT_BG)],
            foreground=[("selected", YELLOW)],
        )

        style.configure(
            "Pipeline.Vertical.TScrollbar",
            background=PANEL_3,
            troughcolor=TABLE_BG,
            bordercolor=BORDER,
            arrowcolor=ORANGE,
        )
        style.configure(
            "Pipeline.Horizontal.TScrollbar",
            background=PANEL_3,
            troughcolor=TABLE_BG,
            bordercolor=BORDER,
            arrowcolor=ORANGE,
        )

        style.configure(
            "Pipeline.TCombobox",
            fieldbackground=BG_2,
            background=PANEL_3,
            foreground=FG,
            arrowcolor=ORANGE,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            font=FONT_SMALL,
        )

    def _button(self, parent, text: str, command, bg: str = PANEL_3, fg: str = FG, width: Optional[int] = None):
        btn = tk.Button(
            parent,
            text=text.upper(),
            command=command,
            bg=bg,
            fg=fg,
            activebackground=SELECT_BG if bg == PANEL_3 else bg,
            activeforeground=YELLOW,
            relief="solid",
            bd=1,
            padx=10,
            pady=5,
            cursor="hand2",
            font=FONT_SMALL,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        if width:
            btn.configure(width=width)
        return btn

    def _build_ui(self) -> None:
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_header()
        self._build_kpis()
        self._build_search()
        self._build_main()
        self._build_statusbar()

        self.bind("<Configure>", self._on_resize, add="+")

    def _build_header(self) -> None:
        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        header.grid_columnconfigure(0, weight=1)

        title_box = tk.Frame(header, bg=BG)
        title_box.grid(row=0, column=0, sticky="w")

        tk.Label(
            title_box,
            text="QUANT TERMINAL · PIPELINE MANAGEMENT",
            bg=BG,
            fg=FG,
            font=FONT_TITLE,
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="PIPELINE CONTROL | PROCESS MONITOR | REGISTRY METADATA | RUNTIME LOGS",
            bg=BG,
            fg=SUBTLE,
            font=FONT_TINY,
        ).pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(header, bg=BG)
        actions.grid(row=0, column=1, sticky="e")

        self.scanner_button = self._button(actions, "Run Scanner", self.run_scanner, bg=PURPLE, fg=WHITE)
        self.scanner_button.pack(side="right", padx=(8, 0))

        self._button(actions, "Refresh", self.refresh_data, bg=PANEL_3, fg=FG).pack(side="right", padx=(8, 0))
        self._button(actions, "Start All", self.start_all, bg=GREEN, fg=WHITE).pack(side="right", padx=(8, 0))
        self._button(actions, "Stop All", self.stop_all, bg=RED, fg=WHITE).pack(side="right", padx=(8, 0))
        self._button(actions, "Logs", self.open_logs, bg=BLUE, fg=WHITE).pack(side="right", padx=(8, 0))

    def _build_kpis(self) -> None:
        self.kpi_frame = tk.Frame(self, bg=BG)
        self.kpi_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))

        self.kpi_total = self._kpi_card("TOTAL", "0", BLUE)
        self.kpi_running = self._kpi_card("RUNNING", "0", GREEN)
        self.kpi_stopped = self._kpi_card("STOPPED", "0", YELLOW)
        self.kpi_missing = self._kpi_card("MISSING", "0", RED)
        self.kpi_auto = self._kpi_card("AUTO RESTART", "0", PURPLE)
        self.kpi_registry = self._kpi_card("REGISTRY DB", "NO", CYAN)

        self.kpi_cards = [
            self.kpi_total.master,
            self.kpi_running.master,
            self.kpi_stopped.master,
            self.kpi_missing.master,
            self.kpi_auto.master,
            self.kpi_registry.master,
        ]

    def _kpi_card(self, title: str, value: str, color: str):
        card = tk.Frame(self.kpi_frame, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        card.pack(side="left", fill="x", expand=True, padx=(0, 6))

        tk.Label(card, text=title.upper(), bg=CARD, fg=ORANGE, font=FONT_TINY).pack(anchor="w", padx=9, pady=(6, 0))
        label = tk.Label(card, text=value, bg=CARD, fg=color, font=("Consolas", 14, "bold"))
        label.pack(anchor="w", padx=9, pady=(0, 6))
        return label

    def _build_search(self) -> None:
        bar = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        bar.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        bar.grid_columnconfigure(1, weight=1)

        tk.Label(bar, text="SEARCH", bg=PANEL, fg=MUTED, font=FONT_TINY).grid(row=0, column=0, sticky="w", padx=(12, 6), pady=9)

        entry = tk.Entry(
            bar,
            textvariable=self.search_var,
            bg=BG_2,
            fg=FG,
            insertbackground=FG,
            relief="solid",
            bd=1,
            font=FONT_SMALL,
        )
        entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=9)
        entry.bind("<KeyRelease>", lambda _e: self._apply_filters())

        tk.Label(bar, text="FILTER", bg=PANEL, fg=MUTED, font=FONT_TINY).grid(row=0, column=2, sticky="e", padx=(0, 6))

        self.filter_combo = ttk.Combobox(
            bar,
            textvariable=self.filter_var,
            values=["ALL", "RUNNING", "STOPPED", "MISSING", "Market", "Trades"],
            state="readonly",
            width=14,
            style="Pipeline.TCombobox",
            font=FONT_SMALL,
        )
        self.filter_combo.grid(row=0, column=3, sticky="e", padx=(0, 12), pady=9)
        self.filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_filters())

        tk.Label(bar, textvariable=self.summary_var, bg=PANEL, fg=MUTED, font=FONT_TINY).grid(row=0, column=4, sticky="e", padx=(0, 12))

    def _build_main(self) -> None:
        self.main = tk.PanedWindow(self, orient="horizontal", bg=BG, sashwidth=7, bd=0)
        self.main.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 8))

        self.left = tk.Frame(self.main, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        self.middle = tk.Frame(self.main, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        self.right = tk.Frame(self.main, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)

        self.main.add(self.left, minsize=MIN_LEFT_WIDTH, stretch="always")
        self.main.add(self.middle, minsize=MIN_MIDDLE_WIDTH, stretch="always")
        self.main.add(self.right, minsize=MIN_RIGHT_WIDTH, stretch="always")

        self._build_left()
        self._build_middle()
        self._build_right()

    def _build_left(self) -> None:
        self.left.grid_rowconfigure(1, weight=1)
        self.left.grid_columnconfigure(0, weight=1)

        tk.Label(self.left, text="NAVIGATION", bg=PANEL, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w", padx=12, pady=(10, 8))

        self.nav_canvas = tk.Canvas(self.left, bg=PANEL, highlightthickness=0, bd=0)
        self.nav_canvas.grid(row=1, column=0, sticky="nsew", padx=(12, 0), pady=(0, 12))

        sy = tk.Scrollbar(self.left, orient="vertical", command=self.nav_canvas.yview)
        sy.grid(row=1, column=1, sticky="ns", padx=(0, 8), pady=(0, 12))
        self.nav_canvas.configure(yscrollcommand=sy.set)

        self.nav_frame = tk.Frame(self.nav_canvas, bg=PANEL)
        self.nav_window = self.nav_canvas.create_window((0, 0), window=self.nav_frame, anchor="nw")

        self.nav_frame.bind("<Configure>", lambda _e: self.nav_canvas.configure(scrollregion=self.nav_canvas.bbox("all")))
        self.nav_canvas.bind("<Configure>", lambda e: self.nav_canvas.itemconfigure(self.nav_window, width=e.width))

    def _build_middle(self) -> None:
        self.middle.grid_rowconfigure(2, weight=1)
        self.middle.grid_columnconfigure(0, weight=1)

        top = tk.Frame(self.middle, bg=PANEL)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        tk.Label(top, text="PIPELINES", bg=PANEL, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w")

        self._button(top, "Start", self.start_selected, bg=GREEN, fg=WHITE, width=8).grid(row=0, column=1, padx=(6, 0))
        self._button(top, "Stop", self.stop_selected, bg=RED, fg=WHITE, width=8).grid(row=0, column=2, padx=(6, 0))
        self._button(top, "Restart", self.restart_selected, bg=YELLOW, fg="#101010", width=8).grid(row=0, column=3, padx=(6, 0))

        self.auto_chk = tk.Checkbutton(
            top,
            text="Auto Restart",
            variable=self.auto_restart_var,
            command=self.toggle_auto_restart,
            bg=PANEL,
            fg=MUTED,
            selectcolor=BG_2,
            activebackground=PANEL,
            activeforeground=FG,
            font=FONT_SMALL,
        )
        self.auto_chk.grid(row=0, column=4, padx=(10, 0))

        self.table_summary = tk.Label(
            self.middle,
            text="SELECT A PIPELINE.",
            bg=CARD,
            fg=MUTED,
            font=FONT_SMALL,
            anchor="w",
            padx=10,
            pady=7,
        )
        self.table_summary.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))

        shell = tk.Frame(self.middle, bg=PANEL)
        shell.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        sy = ttk.Scrollbar(shell, orient="vertical", style="Pipeline.Vertical.TScrollbar")
        sx = ttk.Scrollbar(shell, orient="horizontal", style="Pipeline.Horizontal.TScrollbar")

        self.table = ttk.Treeview(
            shell,
            columns=self.TABLE_COLUMNS,
            show="headings",
            style="Pipeline.Treeview",
            yscrollcommand=sy.set,
            xscrollcommand=sx.set,
        )

        sy.configure(command=self.table.yview)
        sx.configure(command=self.table.xview)

        self.table.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")

        widths = {
            "label": 230,
            "category": 100,
            "status": 100,
            "pid": 90,
            "uptime": 100,
            "auto_restart": 110,
            "script_exists": 80,
            "domain": 130,
            "layer": 120,
            "asset_type": 100,
            "version": 90,
        }

        for col in self.TABLE_COLUMNS:
            self.table.heading(col, text=self.TABLE_HEADINGS[col], command=lambda c=col: self._sort_by(c))
            self.table.column(col, width=widths.get(col, 100), minwidth=70, anchor="w", stretch=True)

        self.table.bind("<<TreeviewSelect>>", self._on_table_select)
        self.table.bind("<Double-1>", lambda _e: self.open_selected_script())

    def _build_right(self) -> None:
        self.right.grid_rowconfigure(3, weight=1)
        self.right.grid_columnconfigure(0, weight=1)

        top = tk.Frame(self.right, bg=PANEL)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 8))
        top.grid_columnconfigure(0, weight=1)

        tk.Label(top, text="DETAILS", bg=PANEL, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w")
        self._button(top, "Open File", self.open_selected_script, bg=PANEL_3, fg=FG).grid(row=0, column=1, padx=(6, 0))
        self._button(top, "Open Folder", self.open_selected_folder, bg=PANEL_3, fg=FG).grid(row=0, column=2, padx=(6, 0))

        flow_header = tk.Frame(self.right, bg=PANEL)
        flow_header.grid(row=1, column=0, sticky="ew", padx=12)
        tk.Label(flow_header, text="VISUAL FLOW", bg=PANEL, fg=SUBTLE, font=FONT_TINY).pack(anchor="w")

        self.flow_canvas = tk.Canvas(self.right, bg=BG_2, height=160, highlightthickness=1, highlightbackground=BORDER)
        self.flow_canvas.grid(row=2, column=0, sticky="ew", padx=12, pady=(4, 8))
        self.flow_canvas.bind("<Configure>", lambda _e: self._draw_flow())

        detail_shell = tk.Frame(self.right, bg=PANEL)
        detail_shell.grid(row=3, column=0, sticky="nsew", padx=12, pady=(0, 12))
        detail_shell.grid_rowconfigure(0, weight=1)
        detail_shell.grid_columnconfigure(0, weight=1)

        self.details = tk.Text(
            detail_shell,
            bg=DETAIL_BG,
            fg=FG,
            insertbackground=FG,
            relief="solid",
            bd=1,
            wrap="word",
            font=FONT_MONO,
            padx=12,
            pady=12,
        )
        self.details.grid(row=0, column=0, sticky="nsew")

        sy = tk.Scrollbar(detail_shell, orient="vertical", command=self.details.yview)
        sy.grid(row=0, column=1, sticky="ns")
        self.details.configure(yscrollcommand=sy.set)

    def _build_statusbar(self) -> None:
        footer = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=BORDER, height=30)
        footer.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 12))
        footer.grid_propagate(False)

        tk.Label(footer, textvariable=self.status_var, bg=PANEL, fg=MUTED, font=FONT_TINY).pack(side="left", padx=10)
        tk.Label(footer, text=shorten(str(QUANT_ROOT), 100), bg=PANEL, fg=SUBTLE, font=FONT_TINY).pack(side="right", padx=10)

    # ========================================================
    # DATA / FILTER / SORT
    # ========================================================

    def _apply_filters(self) -> None:
        q = self.search_var.get().strip().lower()
        f = self.filter_var.get().strip()

        rows = list(self.rows_all)

        if self.selected_category != "ALL":
            rows = [r for r in rows if safe_text(r.get("category")) == self.selected_category]

        if f == "RUNNING":
            rows = [r for r in rows if r.get("status") == "RUNNING"]
        elif f == "STOPPED":
            rows = [r for r in rows if r.get("status") == "STOPPED"]
        elif f == "MISSING":
            rows = [r for r in rows if r.get("script_exists") != "YES"]
        elif f in {"Market", "Trades"}:
            rows = [r for r in rows if r.get("category") == f]

        if q:
            searchable_cols = [
                "id", "label", "category", "status", "script_name", "script_id",
                "domain", "layer", "asset_type", "script_path", "inputs", "outputs",
                "dependencies", "purpose", "last_error",
            ]

            def match(row: Dict[str, Any]) -> bool:
                return any(q in safe_text(row.get(c, "")).lower() for c in searchable_cols)

            rows = [r for r in rows if match(r)]

        self.rows_current = rows
        self.update_table()
        self.summary_var.set(f"Showing {len(self.rows_current)} / {len(self.rows_all)}")
        self.table_summary.config(text=f"Selection: {len(self.rows_current)} pipelines")
        self.update_details()

    def _sort_by(self, col: str) -> None:
        self.rows_current = sorted(self.rows_current, key=lambda r: safe_text(r.get(col, "")).lower())
        self.update_table()

    def _get_selected_row(self) -> Optional[Dict[str, Any]]:
        if not self.selected_id:
            return None
        for row in self.rows_all:
            if row.get("id") == self.selected_id:
                return row
        return None

    def _update_kpis(self) -> None:
        total = len(self.rows_all)
        running = sum(1 for r in self.rows_all if r.get("status") == "RUNNING")
        stopped = total - running
        missing = sum(1 for r in self.rows_all if r.get("script_exists") != "YES")
        auto = sum(1 for r in self.rows_all if r.get("auto_restart") == "ON")
        reg = "YES" if REGISTRY_DB.exists() else "NO"

        self.kpi_total.config(text=str(total))
        self.kpi_running.config(text=str(running))
        self.kpi_stopped.config(text=str(stopped))
        self.kpi_missing.config(text=str(missing))
        self.kpi_auto.config(text=str(auto))
        self.kpi_registry.config(text=reg)

    def _update_navigation(self) -> None:
        for widget in self.nav_frame.winfo_children():
            widget.destroy()

        categories = ["ALL"] + sorted({safe_text(r.get("category", "")) for r in self.rows_all if r.get("category")})
        for cat in categories:
            count = len(self.rows_all) if cat == "ALL" else sum(1 for r in self.rows_all if r.get("category") == cat)
            running = sum(1 for r in self.rows_all if (cat == "ALL" or r.get("category") == cat) and r.get("status") == "RUNNING")
            self._nav_item(cat, count, running)

        self._nav_section("Folders")
        self._folder_item("Backend Pipelines", PIPELINE_BACKEND_DIR)
        self._folder_item("Runtime", RUNTIME_DIR)
        self._folder_item("Logs", LOG_DIR)

    def _nav_section(self, text: str) -> None:
        tk.Label(self.nav_frame, text=text.upper(), bg=PANEL, fg=SUBTLE, font=FONT_TINY).pack(anchor="w", padx=10, pady=(14, 5))

    def _nav_item(self, category: str, count: int, running: int) -> None:
        selected = category == self.selected_category
        bg = SELECT_BG if selected else CARD
        fg = WHITE if selected else FG

        frame = tk.Frame(self.nav_frame, bg=bg, highlightthickness=1, highlightbackground=ORANGE if selected else BORDER)
        frame.pack(fill="x", padx=(0, 8), pady=(0, 6))

        btn = tk.Button(
            frame,
            text=f"{category}  ·  {count} total  ·  {running} running",
            command=lambda c=category: self._select_category(c),
            bg=bg,
            fg=fg,
            activebackground=SELECT_BG,
            activeforeground=YELLOW,
            relief="flat",
            bd=0,
            anchor="w",
            padx=10,
            pady=8,
            font=FONT_SMALL,
            cursor="hand2",
        )
        btn.pack(fill="x")

    def _folder_item(self, label: str, path: Path) -> None:
        exists = path.exists()
        color = GREEN if exists else RED
        text = f"{label}  ·  {'OK' if exists else 'MISSING'}"

        frame = tk.Frame(self.nav_frame, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        frame.pack(fill="x", padx=(0, 8), pady=(0, 6))

        btn = tk.Button(
            frame,
            text=text,
            command=lambda p=path: self._open_folder_safe(p),
            bg=CARD,
            fg=color,
            activebackground=SELECT_BG,
            activeforeground=YELLOW,
            relief="flat",
            bd=0,
            anchor="w",
            padx=10,
            pady=7,
            font=FONT_TINY,
            cursor="hand2",
        )
        btn.pack(fill="x")

    def _select_category(self, category: str) -> None:
        self.selected_category = category
        self._update_navigation()
        self._apply_filters()

    # ========================================================
    # ACTIONS
    # ========================================================

    def _run_action(self, action, success_msg: str) -> None:
        def worker():
            try:
                action()
                self.after(0, self.refresh_data)
                self.after(0, lambda: self.status_var.set(success_msg))
            except Exception as e:
                self.after(0, lambda err=str(e): messagebox.showerror("Fehler", err))
                self.after(0, lambda err=str(e): self.status_var.set(err))

        threading.Thread(target=worker, daemon=True).start()

    def _require_selected(self) -> str:
        if not self.selected_id:
            raise RuntimeError("Keine Pipeline ausgewählt.")
        return self.selected_id

    def start_selected(self) -> None:
        pipeline_id = self._require_selected()
        self._run_action(lambda: self.repository.start_pipeline(pipeline_id), f"Started: {pipeline_id}")

    def stop_selected(self) -> None:
        pipeline_id = self._require_selected()
        self._run_action(lambda: self.repository.stop_pipeline(pipeline_id), f"Stopped: {pipeline_id}")

    def restart_selected(self) -> None:
        pipeline_id = self._require_selected()
        self._run_action(lambda: self.repository.restart_pipeline(pipeline_id), f"Restarted: {pipeline_id}")

    def start_all(self) -> None:
        self._run_action(self.repository.start_all, "All pipelines started.")

    def stop_all(self) -> None:
        self._run_action(self.repository.stop_all, "All pipelines stopped.")

    def toggle_auto_restart(self) -> None:
        pipeline_id = self._require_selected()
        value = bool(self.auto_restart_var.get())
        self.repository.set_auto_restart(pipeline_id, value)
        self.refresh_data()
        self.status_var.set(f"Auto Restart {'ON' if value else 'OFF'}: {pipeline_id}")

    def run_scanner(self) -> None:
        messagebox.showinfo("Scanner", "Kein Scanner für Pipeline Management definiert.")

    def open_logs(self) -> None:
        ensure_dir(LOG_DIR)
        self._open_folder_safe(LOG_DIR)

    def open_selected_script(self) -> None:
        row = self._get_selected_row()
        if not row:
            messagebox.showwarning("Keine Auswahl", "Keine Pipeline ausgewählt.")
            return
        p = Path(safe_text(row.get("script_path", "")))
        if not p.exists():
            messagebox.showwarning("Datei fehlt", safe_text(row.get("script_path", "")))
            return
        self._open_folder_safe(p)

    def open_selected_folder(self) -> None:
        row = self._get_selected_row()
        if not row:
            messagebox.showwarning("Keine Auswahl", "Keine Pipeline ausgewählt.")
            return
        p = Path(safe_text(row.get("script_path", "")))
        if not p.exists():
            messagebox.showwarning("Datei fehlt", safe_text(row.get("script_path", "")))
            return
        self._open_folder_safe(p.parent)

    def _open_folder_safe(self, path: Path) -> None:
        try:
            if not path.exists():
                messagebox.showwarning("Pfad fehlt", str(path))
                return
            open_path(path)
        except Exception as e:
            messagebox.showerror("Open Error", str(e))

    # ========================================================
    # EVENTS / DETAILS / FLOW
    # ========================================================

    def _on_table_select(self, _event=None) -> None:
        selected = self.table.selection()
        if not selected:
            self.selected_id = None
            self.update_details(None)
            return
        self.selected_id = str(selected[0])
        self.update_details()

    def _set_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def _draw_flow(self, row: Optional[Dict[str, Any]] = None) -> None:
        if not hasattr(self, "flow_canvas"):
            return

        if row is None:
            row = self._get_selected_row()

        c = self.flow_canvas
        c.delete("all")

        w = max(c.winfo_width(), 320)
        h = max(c.winfo_height(), 150)

        c.create_rectangle(0, 0, w, h, fill=BG_2, outline=BG_2)

        for x in range(0, w, 40):
            c.create_line(x, 0, x, h, fill="#161616")
        for y in range(0, h, 40):
            c.create_line(0, y, w, y, fill="#161616")

        if row is None:
            c.create_text(w / 2, h / 2, text="Select pipeline for INPUTS → SCRIPT → OUTPUTS", fill=SUBTLE, font=FONT_SMALL)
            return

        inputs = split_lines(row.get("inputs")) or ["No input registered"]
        outputs = split_lines(row.get("outputs")) or ["No output registered"]
        script = row.get("script_name") or row.get("label") or "Script"

        box_w = max(90, min(160, int(w * 0.25)))
        box_h = 76
        y = int(h / 2 - box_h / 2)

        x1 = int(w * 0.06)
        x2 = int(w * 0.38)
        x3 = int(w * 0.70)

        self._flow_box(x1, y, box_w, box_h, "INPUTS", inputs[:2], BLUE)
        self._flow_box(x2, y, box_w, box_h, "SCRIPT", [safe_text(script)], PURPLE)
        self._flow_box(x3, y, box_w, box_h, "OUTPUTS", outputs[:2], GREEN)

        self._flow_arrow(x1 + box_w, h / 2, x2, h / 2)
        self._flow_arrow(x2 + box_w, h / 2, x3, h / 2)

    def _flow_box(self, x: int, y: int, w: int, h: int, title: str, items: List[str], color: str) -> None:
        c = self.flow_canvas
        c.create_rectangle(x + 4, y + 4, x + w + 4, y + h + 4, fill="#000000", outline="")
        c.create_rectangle(x, y, x + w, y + h, fill=PANEL, outline=color, width=2)
        c.create_text(x + 10, y + 14, text=title, anchor="w", fill=color, font=("Consolas", 8, "bold"))
        c.create_line(x + 8, y + 28, x + w - 8, y + 28, fill=BORDER)

        for i, item in enumerate(items[:2]):
            c.create_text(
                x + 10,
                y + 44 + i * 16,
                text=shorten(safe_text(item), 22),
                anchor="w",
                fill=FG if i == 0 else MUTED,
                font=FONT_TINY,
            )

    def _flow_arrow(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.flow_canvas.create_line(x1, y1, x2, y2, fill=MUTED, width=2, arrow=tk.LAST)

    # ========================================================
    # RESPONSIVE
    # ========================================================

    def _on_resize(self, _event=None) -> None:
        if self._resize_after_id:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        self._resize_after_id = self.after(120, self._apply_responsive_layout)

    def _apply_responsive_layout(self) -> None:
        self._resize_after_id = None

        width = max(self.winfo_width(), 1)
        compact = width < COMPACT_WIDTH
        hide_details = width < DETAILS_BREAKPOINT

        mode = ("compact" if compact else "normal", "hide_details" if hide_details else "show_details")
        if mode == self._responsive_mode:
            return

        self._responsive_mode = mode
        self._layout_kpis(compact)
        self._layout_panes(compact=compact, hide_details=hide_details)

    def _layout_kpis(self, compact: bool) -> None:
        for card in self.kpi_cards:
            card.pack_forget()
            card.grid_forget()

        if compact:
            for i, card in enumerate(self.kpi_cards):
                card.grid(row=i // 3, column=i % 3, sticky="ew", padx=4, pady=4)
            for col in range(3):
                self.kpi_frame.grid_columnconfigure(col, weight=1)
        else:
            for card in self.kpi_cards:
                card.pack(side="left", fill="x", expand=True, padx=(0, 8))

    def _layout_panes(self, compact: bool, hide_details: bool) -> None:
        for pane in [self.left, self.middle, self.right]:
            try:
                self.main.forget(pane)
            except Exception:
                pass

        if compact:
            self.main.configure(orient="vertical")
            self.main.add(self.left, minsize=180, stretch="always")
            self.main.add(self.middle, minsize=320, stretch="always")
            if not hide_details:
                self.main.add(self.right, minsize=240, stretch="always")
        else:
            self.main.configure(orient="horizontal")
            self.main.add(self.left, minsize=MIN_LEFT_WIDTH, stretch="always")
            self.main.add(self.middle, minsize=MIN_MIDDLE_WIDTH, stretch="always")
            if not hide_details:
                self.main.add(self.right, minsize=MIN_RIGHT_WIDTH, stretch="always")

        if hide_details:
            self.table_summary.config(text="Details panel hidden at this width. Widen dashboard to inspect metadata.")

    def _schedule_refresh(self) -> None:
        if self._refresh_after_id:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        self._refresh_after_id = self.after(1500, self._periodic_refresh)

    def _periodic_refresh(self) -> None:
        try:
            self.refresh_data()
        finally:
            self._schedule_refresh()

    def destroy(self) -> None:
        if self._refresh_after_id:
            try:
                self.after_cancel(self._refresh_after_id)
            except Exception:
                pass
        if self._resize_after_id:
            try:
                self.after_cancel(self._resize_after_id)
            except Exception:
                pass
        super().destroy()


# ============================================================
# REQUIRED PANEL API
# ============================================================

def build_panel(parent, repository=None, **kwargs):
    return PipelineManagementDashboard(parent, repository=repository, **kwargs)


# ============================================================
# STANDALONE
# ============================================================

class PipelineManagementApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("QUANT TERMINAL - Pipeline Management")
        self.configure(bg=BG)
        self.minsize(1100, 720)

        panel = PipelineManagementDashboard(self)
        panel.pack(fill="both", expand=True)


def main() -> None:
    ensure_dir(RUNTIME_DIR)
    ensure_dir(LOG_DIR)

    print("QUANT_ROOT:", QUANT_ROOT)
    print("PIPELINE_BACKEND_DIR:", PIPELINE_BACKEND_DIR)
    print("REGISTRY_DB:", REGISTRY_DB)
    print("RUNTIME_DIR:", RUNTIME_DIR)

    app = PipelineManagementApp()
    app.mainloop()


if __name__ == "__main__":
    main()

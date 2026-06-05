# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: code_registry_dashboard
# script_name: Code Registry Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Catalog
# asset_type: Dashboard
# purpose: Terminal/Bloomberg-inspired dashboard for browsing registered Python scripts with a visual folder map, registry quality, metadata, inputs/outputs and dependencies.
# inputs:
#   - Data_Center/Data/6_Code_Registry/code_registry.db
# outputs:
#   - Dashboard UI
# dependencies:
#   - tkinter
#   - pandas
#   - sqlite3
#   - pathlib
# schedule: manual
# version: v2.2.0_terminal_bloomberg_ui
# last_reviewed: 2026-06-04
# ============================================================

from __future__ import annotations

import os
import sys
import sqlite3
import subprocess
import textwrap
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog, simpledialog, filedialog
from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# THEME - QUANT TERMINAL / BLOOMBERG-INSPIRED STYLE
# ============================================================
# Ziel: institutioneller Terminal-Look.
# Keine Logos, keine Marken-Assets, nur Terminal-Optik.

APP_BG = "#000000"
SURFACE = "#0A0A0A"
SURFACE_2 = "#050505"

SIDEBAR_1 = "#000000"
SIDEBAR_2 = "#0A0A0A"
SIDEBAR_CARD = "#111111"
SIDEBAR_HOVER = "#332000"
SIDEBAR_SELECTED = "#332000"

TEXT = "#FFFFFF"
TEXT_2 = "#D6D6D6"
MUTED = "#B8B8B8"
FAINT = "#6F6F6F"

BORDER = "#2A2A2A"
BORDER_2 = "#3A3A3A"

BLUE = "#FF9900"
GREEN = "#00FF66"
YELLOW = "#FFD400"
RED = "#FF4444"
WHITE = "#FFFFFF"

ROW_ALT = "#050505"
ROW_SELECTED = "#332000"
MARK_FILL = "#332000"
MARK_OUTLINE = "#FFD400"
PURPLE = "#00AEEF"

ORANGE = "#FF9900"
CYAN = "#00AEEF"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_H1 = ("Consolas", 13, "bold")
FONT_H2 = ("Consolas", 10, "bold")
FONT_BODY = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_MONO = ("Consolas", 9)

VISUAL_TREE_W = 1200
INPUT_BOX_W = 300
SCRIPT_BOX_W = 260
OUTPUT_BOX_W = 300
VISUAL_GAP = 18
VISUAL_LINE_H = 16
VISUAL_BOX_PAD_X = 12
VISUAL_BOX_PAD_Y = 10
VISUAL_TEXT_CHARS_PER_LINE = 34


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_quant_root(start: Path) -> Path:
    for p in [start.resolve()] + list(start.resolve().parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p.resolve()
    # fallback for testing outside the real project
    return start.resolve().parent


QUANT_ROOT = find_quant_root(SCRIPT_PATH)
DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
REGISTRY_DB = DATA_DIR / "6_Code_Registry" / "code_registry.db"
SCANNER_CODE = DATA_CENTER_DIR / "Backend_Management" / "6_Code_Registry" / "code.py"


# ============================================================
# HELPERS
# ============================================================


def safe_str(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def split_block(value: object) -> list[str]:
    text = safe_str(value).strip()
    if not text:
        return []
    out: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        item = raw.strip()
        if not item:
            continue
        if item.startswith("-"):
            item = item[1:].strip()
        if item:
            out.append(item)
    return out


def compact_io(value: object, max_items: int = 2, max_len: int = 18) -> str:
    items = split_block(value)
    if not items:
        return "-"
    shown = [ellipsize(x, max_len) for x in items[:max_items]]
    if len(items) > max_items:
        shown.append(f"+{len(items) - max_items}")
    return " | ".join(shown)


def ellipsize(text: object, max_len: int = 36) -> str:
    s = safe_str(text)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def normalize_rel_path(path_text: str) -> str:
    return path_text.replace("\\", "/")


# ============================================================
# REPOSITORY
# ============================================================


class CodeRegistryRepository:
    def __init__(self, db_path: Path = REGISTRY_DB):
        self.db_path = db_path

    def load_assets(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM code_assets ORDER BY relative_path", conn)
            return self._clean_for_dashboard(df)
        except Exception as exc:
            messagebox.showerror("DB Fehler", str(exc))
            return pd.DataFrame()

    def _clean_for_dashboard(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        d = df.copy()
        required = [
            "id", "file_path", "relative_path", "file_name", "script_name", "script_id",
            "layer", "domain", "asset_type", "status", "version", "has_registry",
            "registry_quality_status", "registry_quality_message", "scanned_at_utc",
            "purpose", "inputs", "outputs", "dependencies", "last_reviewed", "last_modified_utc",
            "file_size_bytes", "created_date", "author", "owner", "schedule",
        ]
        for col in required:
            if col not in d.columns:
                d[col] = ""

        d["file_path"] = d["file_path"].fillna("").astype(str)
        d["relative_path"] = d["relative_path"].fillna("").astype(str)

        def exists(path_text: str) -> bool:
            try:
                return Path(path_text).exists()
            except Exception:
                return False

        d["file_exists"] = d["file_path"].apply(exists)
        d = d[d["file_exists"]].copy()
        if d.empty:
            return d

        def rel(row: pd.Series) -> str:
            current = safe_str(row.get("relative_path")).strip()
            if current:
                return current
            try:
                return str(Path(safe_str(row.get("file_path"))).relative_to(QUANT_ROOT))
            except Exception:
                return safe_str(row.get("file_path"))

        d["relative_path"] = d.apply(rel, axis=1)
        d["_scanned_dt"] = pd.to_datetime(d["scanned_at_utc"], errors="coerce", utc=True)
        d["_id_sort"] = pd.to_numeric(d["id"], errors="coerce").fillna(0)
        d = d.sort_values(["relative_path", "_scanned_dt", "_id_sort"], ascending=[True, False, False])
        d = d.drop_duplicates(subset=["relative_path"], keep="first").copy()

        d["has_registry"] = pd.to_numeric(d["has_registry"], errors="coerce").fillna(0).astype(int)
        d["registry_quality_status"] = d["registry_quality_status"].fillna("").astype(str).str.strip()
        d.loc[(d["has_registry"] == 1) & (d["registry_quality_status"] == ""), "registry_quality_status"] = "passed"
        d["registry_quality_status"] = d["registry_quality_status"].replace({"passed": "OK", "warning": "Warning", "failed": "Failed"})
        d["status"] = d["status"].fillna("").astype(str).str.strip()

        d = d.sort_values("relative_path").reset_index(drop=True)
        return d

    def cleanup_stale_records(self) -> bool:
        if not self.db_path.exists():
            return False
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT id, file_path FROM code_assets", conn)
                stale_ids: list[int] = []
                for _, row in df.iterrows():
                    try:
                        if not Path(str(row["file_path"])).exists():
                            stale_ids.append(int(row["id"]))
                    except Exception:
                        stale_ids.append(int(row["id"]))
                if stale_ids:
                    placeholders = ",".join(["?"] * len(stale_ids))
                    conn.execute(f"DELETE FROM code_assets WHERE id IN ({placeholders})", stale_ids)
                    conn.commit()
            messagebox.showinfo("Cleanup", f"Stale registry rows removed: {len(stale_ids)}")
            return True
        except Exception as exc:
            messagebox.showerror("Cleanup Fehler", str(exc))
            return False

    def run_scanner(self) -> bool:
        if not SCANNER_CODE.exists():
            messagebox.showerror("Scanner fehlt", str(SCANNER_CODE))
            return False
        try:
            subprocess.run([sys.executable, str(SCANNER_CODE)], cwd=str(QUANT_ROOT), check=True)
            return True
        except Exception as exc:
            messagebox.showerror("Scanner Fehler", str(exc))
            return False


# ============================================================
# UI WIDGETS
# ============================================================


class RoundedCard(tk.Frame):
    """Simple white card with border. Tkinter has no native rounded rectangle for frames."""

    def __init__(self, parent, bg=SURFACE, border=BORDER, **kwargs):
        super().__init__(parent, bg=border, **kwargs)
        self.inner = tk.Frame(self, bg=bg)
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)


# ============================================================
# DASHBOARD
# ============================================================


class CodeRegistryDashboardBlock(tk.Frame):
    TABLE_COLUMNS = ["file_name", "script_name", "layer", "domain", "asset_type", "status", "version"]
    TABLE_LABELS = {
        "file_name": "File Name",
        "script_name": "Script Name",
        "layer": "Layer",
        "domain": "Domain",
        "asset_type": "Asset Type",
        "status": "Status",
        "version": "Version",
    }

    def __init__(self, parent, repository: Optional[CodeRegistryRepository] = None, **kwargs):
        super().__init__(parent, bg=APP_BG, **kwargs)
        self.repo = repository or CodeRegistryRepository()
        self.df_all = pd.DataFrame()
        self.df_current = pd.DataFrame()
        self.selected_code_id: Optional[int] = None
        self.selected_folder_path = ""
        self.expanded_keys: set[str] = set()
        self.marked_folders: set[str] = set()
        self.marked_scripts: set[int] = set()
        self.marked_virtual_scripts: set[str] = set()
        self.virtual_nodes: dict[str, dict[str, str]] = {}
        self.selected_virtual_path = ""
        self.search_var = tk.StringVar()
        self.breadcrumb_var = tk.StringVar(value="Dashboard  ›  Building_Blocks  ›  Code_Registry  ›  code.py")
        self._build_ui()
        self.reload()

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def _build_ui(self):
        self._setup_style()
        self.grid_columnconfigure(0, minsize=VISUAL_TREE_W)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # The visual folder map is the only left navigation.
        # The former blue sidebar was intentionally removed to avoid duplicate trees.
        self.visual_tree_panel = tk.Frame(
            self,
            bg=SURFACE_2,
            width=VISUAL_TREE_W,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.visual_tree_panel.grid(row=0, column=0, sticky="nsew")
        self.visual_tree_panel.grid_propagate(False)

        self.content = tk.Frame(self, bg=APP_BG)
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(2, weight=1)

        self._build_visual_tree_panel()
        self._build_content()

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            ".",
            background=APP_BG,
            foreground=TEXT,
            fieldbackground=SURFACE,
            font=FONT_BODY,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
        )
        style.configure(
            "Registry.Treeview",
            background="#000000",
            foreground=TEXT,
            fieldbackground="#000000",
            rowheight=22,
            borderwidth=1,
            relief="solid",
            font=FONT_SMALL,
        )
        style.configure(
            "Registry.Treeview.Heading",
            background="#111111",
            foreground=ORANGE,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 8, "bold"),
            padding=(6, 7),
        )
        style.map(
            "Registry.Treeview",
            background=[("selected", ROW_SELECTED)],
            foreground=[("selected", YELLOW)],
        )

    def _button(self, parent, text: str, command, bg=SURFACE, fg=TEXT, border=BORDER):
        outer = tk.Frame(parent, bg=border)
        btn = tk.Button(
            outer,
            text=text.upper(),
            command=command,
            bg=bg,
            fg=fg if fg != TEXT else ORANGE,
            activebackground=ROW_SELECTED,
            activeforeground=YELLOW,
            relief="solid",
            bd=1,
            padx=10,
            pady=6,
            cursor="hand2",
            font=FONT_SMALL,
            highlightthickness=1,
            highlightbackground=border,
        )
        btn.pack(fill="both", expand=True, padx=0, pady=0)
        return outer

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Visual folder mirror
    # --------------------------------------------------------

    def _build_visual_tree_panel(self):
        header = tk.Frame(self.visual_tree_panel, bg=SURFACE_2)
        header.pack(fill="x", padx=14, pady=(18, 10))

        tk.Label(
            header,
            text="CODE TREE",
            bg=SURFACE_2,
            fg=TEXT,
            font=FONT_H2,
        ).pack(anchor="w")

        tk.Label(
            header,
            text="INPUT → SCRIPT → OUTPUT terminal map",
            bg=SURFACE_2,
            fg=MUTED,
            font=FONT_TINY,
        ).pack(anchor="w", pady=(2, 0))

        search_wrap = tk.Frame(
            self.visual_tree_panel,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        search_wrap.pack(fill="x", padx=14, pady=(0, 10))

        search_entry = tk.Entry(
            search_wrap,
            textvariable=self.search_var,
            relief="solid",
            bd=1,
            bg=SURFACE,
            fg=TEXT,
            insertbackground=TEXT,
            font=FONT_SMALL,
        )
        search_entry.pack(side="left", fill="x", expand=True, padx=10, pady=8)
        search_entry.bind("<KeyRelease>", lambda _e: self.apply_search())

        tk.Button(
            search_wrap,
            text="Clear",
            command=self.clear_search,
            bg=SURFACE,
            fg=BLUE,
            activebackground="#332000",
            activeforeground=BLUE,
            relief="flat",
            bd=0,
            padx=8,
            cursor="hand2",
            font=FONT_TINY,
        ).pack(side="right", padx=(0, 6))

        toolbar = tk.Frame(self.visual_tree_panel, bg=SURFACE_2)
        toolbar.pack(fill="x", padx=14, pady=(0, 10))

        self._small_visual_button(toolbar, "Expand", self.expand_visual_all).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Collapse", self.collapse_visual_all).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Refresh", self.reload).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "New Folder", self.add_physical_folder).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "New Code", self.add_code_file).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Mark Selected", self.toggle_mark_selected).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Clear Marks", self.clear_marks).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Copy Tree", self.copy_expanded_tree).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Copy Code", self.copy_selected_file_content).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Copy Folder", self.copy_selected_folder_contents).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Copy For ChatGPT", self.copy_selected_context_for_chatgpt).pack(side="left")

        shell = tk.Frame(self.visual_tree_panel, bg=SURFACE_2)
        shell.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        shell.grid_rowconfigure(0, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        self.visual_canvas = tk.Canvas(shell, bg=SURFACE_2, highlightthickness=0, bd=0)
        self.visual_canvas.grid(row=0, column=0, sticky="nsew")

        self.visual_y = tk.Scrollbar(shell, orient="vertical", command=self.visual_canvas.yview)
        self.visual_y.grid(row=0, column=1, sticky="ns")

        self.visual_x = tk.Scrollbar(shell, orient="horizontal", command=self.visual_canvas.xview)
        self.visual_x.grid(row=1, column=0, sticky="ew")

        self.visual_canvas.configure(
            yscrollcommand=self.visual_y.set,
            xscrollcommand=self.visual_x.set,
        )
        self.visual_canvas.bind("<Configure>", lambda _e: self.draw_visual_folder_tree())
        self.visual_canvas.bind("<MouseWheel>", self._visual_mousewheel)

    def _small_visual_button(self, parent, text: str, command):
        return tk.Button(
            parent,
            text=text.upper(),
            command=command,
            bg=SURFACE,
            fg=ORANGE,
            activebackground=ROW_SELECTED,
            activeforeground=YELLOW,
            relief="solid",
            bd=1,
            padx=7,
            pady=4,
            cursor="hand2",
            font=FONT_TINY,
            highlightthickness=1,
            highlightbackground=BORDER,
        )

    def _visual_mousewheel(self, event):
        try:
            self.visual_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def clear_search(self):
        self.search_var.set("")
        self.apply_search()


    def _clean_virtual_path(self, path_text: str) -> str:
        p = normalize_rel_path(path_text).strip().strip("/")
        while "//" in p:
            p = p.replace("//", "/")
        return p

    def _selected_base_folder(self) -> str:
        if self.selected_virtual_path:
            parent = str(Path(self.selected_virtual_path).parent).replace("\\", "/")
            return "" if parent == "." else parent
        if self.selected_folder_path:
            return normalize_rel_path(self.selected_folder_path)
        return ""

    def _join_virtual_path(self, name_or_path: str, default_ext: str = "") -> str:
        raw = self._clean_virtual_path(name_or_path)
        if not raw:
            return ""
        has_folder = "/" in raw or "\\" in raw
        if not has_folder:
            base = self._selected_base_folder()
            raw = f"{base}/{raw}" if base else raw
        if default_ext and "." not in Path(raw).name:
            raw = f"{raw}.{default_ext}"
        return self._clean_virtual_path(raw)

    def _ensure_parent_expanded(self, rel_path: str):
        parts = list(Path(rel_path).parts)
        if not parts:
            return
        if "." in Path(rel_path).name:
            parts = parts[:-1]
        for i in range(1, len(parts) + 1):
            folder = str(Path(*parts[:i])).replace("\\", "/")
            self.expanded_keys.add(f"folder::{folder}")

    def add_physical_folder(self):
        name = simpledialog.askstring(
            "New Folder",
            "Folder path relative to QUANT root or relative to selected folder:",
            parent=self,
        )
        if not name:
            return

        rel_path = self._join_virtual_path(name)
        if not rel_path:
            return

        target_dir = QUANT_ROOT / rel_path

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Ordner konnte nicht erstellt werden", str(exc))
            return

        self.virtual_nodes[rel_path] = {
            "kind": "folder",
            "file_type": "folder",
            "physical_path": str(target_dir),
            "created_on_disk": "yes",
        }

        self.selected_code_id = None
        self.selected_virtual_path = ""
        self.selected_folder_path = rel_path
        self._ensure_parent_expanded(rel_path)
        self.expanded_keys.add(f"folder::{rel_path}")

        self.reload()
        self.selected_folder_path = rel_path
        self.expanded_keys.add(f"folder::{rel_path}")
        self.draw_visual_folder_tree()

    def add_code_file(self):
        name = simpledialog.askstring(
            "New Code",
            "Python file path relative to QUANT root or relative to selected folder:",
            parent=self,
        )
        if not name:
            return

        rel_path = self._join_virtual_path(name, default_ext="py")
        if not rel_path:
            return

        target_file = QUANT_ROOT / rel_path

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if not target_file.exists():
                target_file.write_text(
                    "# ============================================================\n"
                    "# PLANNED CODE FILE\n"
                    "# ============================================================\n"
                    "# This file was created from the Code Registry visual tree.\n"
                    "# Inputs, outputs, dependencies and details are not defined yet.\n"
                    "# ============================================================\n\n",
                    encoding="utf-8",
                )
        except Exception as exc:
            messagebox.showerror("Code-Datei konnte nicht erstellt werden", str(exc))
            return

        self.virtual_nodes[rel_path] = {
            "kind": "file",
            "file_type": "python",
            "physical_path": str(target_file),
            "created_on_disk": "yes",
            "registry_status": "planned",
        }

        self.selected_code_id = None
        self.selected_virtual_path = rel_path
        parent = str(Path(rel_path).parent).replace("\\", "/")
        self.selected_folder_path = "" if parent == "." else parent
        self._ensure_parent_expanded(rel_path)
        self.show_virtual_code_details(rel_path)
        self.draw_visual_folder_tree()

    def get_direct_virtual_scripts(self, folder_path: str) -> list[str]:
        folder_norm = normalize_rel_path(folder_path)
        out: list[str] = []
        for path, meta in self.virtual_nodes.items():
            if meta.get("kind") != "file":
                continue
            parent = str(Path(path).parent).replace("\\", "/")
            if parent == ".":
                parent = ""
            if parent == folder_norm:
                out.append(path)
        return sorted(out, key=lambda x: x.lower())

    def select_virtual_script(self, rel_path: str):
        rel_path = self._clean_virtual_path(rel_path)
        self.selected_code_id = None
        self.selected_virtual_path = rel_path
        parent = str(Path(rel_path).parent).replace("\\", "/")
        self.selected_folder_path = "" if parent == "." else parent
        self.breadcrumb_var.set("Dashboard  ›  " + rel_path.replace("\\", "  ›  ").replace("/", "  ›  "))
        self.show_virtual_code_details(rel_path)
        self.draw_visual_folder_tree()

    def show_virtual_code_details(self, rel_path: str):
        meta = self.virtual_nodes.get(rel_path, {})
        file_name = Path(rel_path).name
        folder = str(Path(rel_path).parent).replace("\\", "/")
        if folder == ".":
            folder = ""

        self.sel_title.config(text=f"PLANNED: {file_name}")
        self.sel_sub.config(text=f"{rel_path} | new code file | no registry details yet")

        self._set_kv(self.sel_kv, [
            ("Script ID", "-"),
            ("Layer", "PLANNED"),
            ("Domain", "PLANNED"),
            ("Asset Type", "Code"),
        ])

        self._set_kv(self.sel_kv2, [
            ("Status", "● planned"),
            ("Version", "-"),
            ("Registry Status", "Missing / not scanned yet"),
            ("Letzte Änderung", "-"),
        ])

        self._set_text(self.description_box, "Planned code file. No purpose/details registered yet.")
        self._set_text(self.io_box, "Inputs\n• -\n\nOutputs\n• -")
        self._set_text(self.dep_box, "• -")

        self.meta_labels[0].config(text=folder or "-")
        self.meta_labels[1].config(text="0 KB" if meta.get("created_on_disk") == "yes" else "-")
        self.meta_labels[2].config(text="-")
        self.meta_labels[3].config(text="-")

    def toggle_folder_mark(self, folder_path: str):
        if folder_path in self.marked_folders:
            self.marked_folders.remove(folder_path)
        else:
            self.marked_folders.add(folder_path)
        self.selected_folder_path = folder_path
        self.draw_visual_folder_tree()

    def toggle_script_mark(self, code_id: int):
        code_id = int(code_id)
        if code_id in self.marked_scripts:
            self.marked_scripts.remove(code_id)
        else:
            self.marked_scripts.add(code_id)
        self.selected_code_id = code_id
        self.selected_virtual_path = ""
        self.draw_visual_folder_tree()

    def toggle_virtual_script_mark(self, rel_path: str):
        rel_path = self._clean_virtual_path(rel_path)
        if rel_path in self.marked_virtual_scripts:
            self.marked_virtual_scripts.remove(rel_path)
        else:
            self.marked_virtual_scripts.add(rel_path)
        self.selected_virtual_path = rel_path
        self.selected_code_id = None
        self.draw_visual_folder_tree()

    def toggle_mark_selected(self):
        if self.selected_virtual_path:
            self.toggle_virtual_script_mark(self.selected_virtual_path)
            return
        if self.selected_code_id is not None:
            self.toggle_script_mark(int(self.selected_code_id))
            return
        if self.selected_folder_path:
            self.toggle_folder_mark(self.selected_folder_path)
            return
        messagebox.showwarning("Keine Auswahl", "Bitte zuerst einen Ordner oder eine Code-Datei auswählen.")

    def clear_marks(self):
        self.marked_folders.clear()
        self.marked_scripts.clear()
        self.marked_virtual_scripts.clear()
        self.draw_visual_folder_tree()

    def copy_expanded_tree(self):
        if self.df_all.empty and not self.virtual_nodes:
            return
        self.copy_to_clipboard(self.build_expanded_ascii_tree())

    def build_expanded_ascii_tree(self) -> str:
        virtual_file_count = sum(1 for m in self.virtual_nodes.values() if m.get("kind") == "file")
        lines: list[str] = [f"CODE TREE ({len(self.df_all)} registered scripts + {virtual_file_count} planned scripts)"]
        root_folders = self.get_child_folders("")
        for idx, folder in enumerate(root_folders):
            self._append_tree_folder(lines, folder, prefix="", is_last=(idx == len(root_folders) - 1))

        if self.marked_folders or self.marked_scripts or self.marked_virtual_scripts:
            lines.append("")
            lines.append("MARKED ITEMS")
            lines.append("------------")
            for folder in sorted(self.marked_folders, key=lambda x: x.lower()):
                lines.append(f"[MARKED FOLDER] {folder}")
            if self.marked_scripts:
                marked = self.df_all[self.df_all["id"].astype(int).isin(self.marked_scripts)].copy()
                for _, row in marked.sort_values("relative_path").iterrows():
                    lines.append(f"[MARKED CODE] {safe_str(row.get('relative_path'))}")
            for rel_path in sorted(self.marked_virtual_scripts, key=lambda x: x.lower()):
                lines.append(f"[MARKED PLANNED CODE] {rel_path}")

        return "\n".join(lines)

    def _append_tree_folder(self, lines: list[str], folder_path: str, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        marker = "⭐ [MARKED] " if folder_path in self.marked_folders else ""
        open_flag = "[open]" if f"folder::{folder_path}" in self.expanded_keys else "[closed]"
        folder_meta = self.virtual_nodes.get(folder_path, {})
        planned_folder = " [CREATED FOLDER]" if folder_meta.get("created_on_disk") == "yes" else (" [PLANNED]" if folder_meta.get("kind") == "folder" else "")
        lines.append(f"{prefix}{connector}{marker}📁 {Path(folder_path).name} ({self.count_folder_scripts(folder_path)}) {open_flag}{planned_folder}")

        if f"folder::{folder_path}" not in self.expanded_keys:
            return

        child_prefix = prefix + ("    " if is_last else "│   ")
        direct = self.get_direct_scripts(folder_path)
        virtual_direct = self.get_direct_virtual_scripts(folder_path)
        children = self.get_child_folders(folder_path)

        entries = [("file", row) for _, row in direct.iterrows()]
        entries += [("virtual_file", rel_path) for rel_path in virtual_direct]
        entries += [("folder", child) for child in children]

        for i, (kind, obj) in enumerate(entries):
            last = i == len(entries) - 1
            sub_connector = "└── " if last else "├── "
            if kind == "file":
                row = obj
                try:
                    cid = int(row.get("id"))
                except Exception:
                    cid = -1
                marker = "⭐ [MARKED] " if cid in self.marked_scripts else ""
                rel = safe_str(row.get("relative_path"))
                status = safe_str(row.get("registry_quality_status")) or "-"
                inputs = len(split_block(row.get("inputs")))
                outputs = len(split_block(row.get("outputs")))
                lines.append(f"{child_prefix}{sub_connector}{marker}🐍 {safe_str(row.get('file_name'))} | registry={status} | inputs={inputs} | outputs={outputs} | {rel}")
            elif kind == "virtual_file":
                rel_path = str(obj)
                marker = "⭐ [MARKED] " if rel_path in self.marked_virtual_scripts else ""
                lines.append(f"{child_prefix}{sub_connector}{marker}🐍 {Path(rel_path).name} [PLANNED CODE] | inputs=0 | outputs=0 | details=missing | {rel_path}")
            else:
                self._append_tree_folder(lines, str(obj), child_prefix, last)


    def expand_visual_all(self):
        if self.df_all.empty and not self.virtual_nodes:
            return
        self.expanded_keys = {f"folder::{folder}" for folder in self._all_folder_paths()}
        self.draw_visual_folder_tree()

    def collapse_visual_all(self):
        self.expanded_keys.clear()
        self.draw_visual_folder_tree()

    def _all_folder_paths(self) -> list[str]:
        folders = set()
        if not self.df_all.empty and "folder_parts" in self.df_all.columns:
            for parts in self.df_all["folder_parts"].tolist():
                for i in range(1, len(parts) + 1):
                    folders.add(str(Path(*parts[:i])).replace("\\", "/"))

        for rel_path, meta in self.virtual_nodes.items():
            parts = list(Path(rel_path).parts)
            if meta.get("kind") == "file":
                parts = parts[:-1]
            for i in range(1, len(parts) + 1):
                folders.add(str(Path(*parts[:i])).replace("\\", "/"))

        return sorted(folders, key=lambda x: x.lower())

    def draw_visual_folder_tree(self):
        if not hasattr(self, "visual_canvas"):
            return

        c = self.visual_canvas
        c.delete("all")
        w = max(c.winfo_width(), VISUAL_TREE_W)
        y = 26

        if self.df_all.empty and not self.virtual_nodes:
            c.create_text(20, y, text="No registry data", anchor="w", fill=MUTED, font=FONT_SMALL)
            c.configure(scrollregion=c.bbox("all"))
            return

        virtual_file_count = sum(1 for m in self.virtual_nodes.values() if m.get("kind") == "file")
        c.create_text(18, y, text="QUANT", anchor="w", fill=TEXT, font=FONT_H2)
        c.create_text(w - 22, y, text=f"{len(self.df_all)} + {virtual_file_count}", anchor="e", fill=BLUE, font=("Consolas", 8, "bold"))
        y += 24
        c.create_text(18, y, text="Code nodes show INPUT → SCRIPT → OUTPUT with full wrapped paths", anchor="w", fill=MUTED, font=FONT_TINY)
        y += 24

        for folder in self.get_child_folders(""):
            y = self._draw_visual_folder_node(folder, level=0, y=y)

        bbox = c.bbox("all")
        if bbox:
            c.configure(scrollregion=(0, 0, max(w, bbox[2] + 30), bbox[3] + 30))

    def _draw_visual_folder_node(self, folder_path: str, level: int, y: int) -> int:
        c = self.visual_canvas
        key = f"folder::{folder_path}"
        is_open = key in self.expanded_keys
        is_selected = folder_path == self.selected_folder_path
        is_marked = folder_path in self.marked_folders
        name = Path(folder_path).name if folder_path else "ROOT"
        count = self.count_folder_scripts(folder_path)

        x = 22 + level * 34
        box_w = max(420, min(720, VISUAL_TREE_W - x - 34))
        box_h = 30
        fill = MARK_FILL if is_marked else ("#332000" if is_selected else SURFACE)
        outline = MARK_OUTLINE if is_marked else (BLUE if is_selected else BORDER)
        text_color = MARK_OUTLINE if is_marked else (BLUE if is_selected else TEXT)

        # vertical connector from parent level
        if level > 0:
            c.create_line(x - 18, y - 10, x - 18, y + box_h // 2, fill=BORDER, width=1)
            c.create_line(x - 18, y + box_h // 2, x - 5, y + box_h // 2, fill=BORDER, width=1)

        node_id = c.create_rectangle(x, y, x + box_w, y + box_h, fill=fill, outline=outline, width=1)
        toggle_id = c.create_text(x + 10, y + box_h / 2, text="▾" if is_open else "▸", anchor="w", fill=BLUE, font=("Consolas", 8, "bold"))
        folder_label = f"⭐ 📁 {ellipsize(name, 24)}" if is_marked else f"📁 {ellipsize(name, 24)}"
        label_id = c.create_text(x + 28, y + box_h / 2, text=folder_label, anchor="w", fill=text_color, font=("Consolas", 8, "bold") if level <= 1 else FONT_SMALL)
        count_id = c.create_text(x + box_w - 10, y + box_h / 2, text=str(count), anchor="e", fill=MUTED, font=FONT_TINY)

        for item in (node_id, toggle_id, label_id, count_id):
            c.tag_bind(item, "<Button-1>", lambda _e, k=key, fp=folder_path: self.toggle_folder(k, fp))
            c.tag_bind(item, "<Button-3>", lambda _e, fp=folder_path: self.toggle_folder_mark(fp))
            c.tag_bind(item, "<Enter>", lambda _e: c.configure(cursor="hand2"))
            c.tag_bind(item, "<Leave>", lambda _e: c.configure(cursor=""))

        y += box_h + 8

        if is_open:
            direct = self.get_direct_scripts(folder_path)
            max_files = 5
            for i, (_, script_row) in enumerate(direct.iterrows()):
                if i >= max_files:
                    hidden = len(direct) - max_files
                    if hidden > 0:
                        y = self._draw_visual_more_node(level + 1, y, hidden)
                    break
                y = self._draw_visual_file_node(script_row, level + 1, y)

            for virtual_path in self.get_direct_virtual_scripts(folder_path):
                y = self._draw_virtual_script_node(virtual_path, level + 1, y)

            for child in self.get_child_folders(folder_path):
                y = self._draw_visual_folder_node(child, level + 1, y)

        return y

    def _wrap_canvas_lines(self, text: str, chars_per_line: int = VISUAL_TEXT_CHARS_PER_LINE) -> list[str]:
        clean = safe_str(text).strip()
        if not clean:
            return ["-"]
        lines: list[str] = []
        for raw in clean.replace("\\", "/").split("\n"):
            raw = raw.strip()
            if not raw:
                continue
            wrapped = textwrap.wrap(
                raw,
                width=chars_per_line,
                break_long_words=True,
                break_on_hyphens=False,
            )
            lines.extend(wrapped if wrapped else [raw])
        return lines or ["-"]

    def _io_lines(self, value: object) -> list[str]:
        items = split_block(value)
        if not items:
            return ["-"]
        out: list[str] = []
        for item in items:
            wrapped = self._wrap_canvas_lines(item)
            for i, line in enumerate(wrapped):
                out.append(("• " if i == 0 else "  ") + line)
        return out or ["-"]

    def _draw_visual_io_box(self, x: int, y: int, w: int, title: str, lines: list[str], title_color: str, outline: str) -> int:
        c = self.visual_canvas
        line_count = max(1, len(lines))
        box_h = VISUAL_BOX_PAD_Y * 2 + 18 + line_count * VISUAL_LINE_H

        c.create_rectangle(x, y, x + w, y + box_h, fill=SURFACE, outline=outline, width=1)
        c.create_text(
            x + VISUAL_BOX_PAD_X,
            y + VISUAL_BOX_PAD_Y,
            text=title,
            anchor="nw",
            fill=title_color,
            font=("Consolas", 8, "bold"),
        )

        ty = y + VISUAL_BOX_PAD_Y + 20
        for line in lines:
            c.create_text(
                x + VISUAL_BOX_PAD_X,
                ty,
                text=line,
                anchor="nw",
                fill=TEXT_2,
                font=FONT_TINY,
                width=w - VISUAL_BOX_PAD_X * 2,
            )
            ty += VISUAL_LINE_H
        return box_h

    def _draw_visual_file_node(self, row: pd.Series, level: int, y: int) -> int:
        c = self.visual_canvas
        try:
            code_id = int(row.get("id"))
        except Exception:
            return y

        is_selected = self.selected_code_id == code_id
        is_marked = code_id in self.marked_scripts
        q = safe_str(row.get("registry_quality_status")).lower()
        has_registry = int(row.get("has_registry", 0) or 0)
        status_color = GREEN if has_registry == 1 and q in {"ok", "passed"} else (YELLOW if "warn" in q else RED)

        input_lines = self._io_lines(row.get("inputs"))
        output_lines = self._io_lines(row.get("outputs"))

        x = 22 + level * 34
        input_x = x
        script_x = input_x + INPUT_BOX_W + VISUAL_GAP
        output_x = script_x + SCRIPT_BOX_W + VISUAL_GAP
        total_w = INPUT_BOX_W + SCRIPT_BOX_W + OUTPUT_BOX_W + VISUAL_GAP * 2

        input_h = VISUAL_BOX_PAD_Y * 2 + 18 + max(1, len(input_lines)) * VISUAL_LINE_H
        output_h = VISUAL_BOX_PAD_Y * 2 + 18 + max(1, len(output_lines)) * VISUAL_LINE_H
        script_h = 72
        total_h = max(input_h, script_h, output_h)

        fill = MARK_FILL if is_marked else ("#332000" if is_selected else "#050505")
        outline = MARK_OUTLINE if is_marked else (BLUE if is_selected else BORDER_2)

        # branch connector from folder tree into INPUT -> SCRIPT -> OUTPUT row
        c.create_line(x - 18, y - 9, x - 18, y + total_h // 2, fill=BORDER, width=1)
        c.create_line(x - 18, y + total_h // 2, input_x - 5, y + total_h // 2, fill=BORDER, width=1)

        self._draw_visual_io_box(input_x, y, INPUT_BOX_W, "INPUT", input_lines, BLUE, outline)

        script_y = y + max(0, (total_h - script_h) // 2)
        script_rect = c.create_rectangle(
            script_x,
            script_y,
            script_x + SCRIPT_BOX_W,
            script_y + script_h,
            fill=fill,
            outline=outline,
            width=1,
        )
        dot_id = c.create_oval(script_x + 12, script_y + 14, script_x + 22, script_y + 24, fill=status_color, outline=status_color)
        label_id = c.create_text(
            script_x + 32,
            script_y + 11,
            text=(f"⭐ 🐍 {safe_str(row.get('file_name'))}" if is_marked else f"🐍 {safe_str(row.get('file_name'))}"),
            anchor="nw",
            fill=TEXT,
            font=("Consolas", 8, "bold"),
            width=SCRIPT_BOX_W - 44,
        )
        script_name_id = c.create_text(
            script_x + 12,
            script_y + 39,
            text=safe_str(row.get("script_name")) or safe_str(row.get("script_id")) or "-",
            anchor="nw",
            fill=MUTED,
            font=FONT_TINY,
            width=SCRIPT_BOX_W - 24,
        )
        ver_id = c.create_text(
            script_x + SCRIPT_BOX_W - 12,
            script_y + script_h - 11,
            text=safe_str(row.get("version")),
            anchor="se",
            fill=MUTED,
            font=FONT_TINY,
        )

        self._draw_visual_io_box(output_x, y, OUTPUT_BOX_W, "OUTPUT", output_lines, GREEN, outline)

        mid_y = y + total_h // 2
        c.create_line(input_x + INPUT_BOX_W, mid_y, script_x, mid_y, fill=BORDER, width=1)
        c.create_text(input_x + INPUT_BOX_W + VISUAL_GAP / 2, mid_y - 1, text="→", fill=BLUE, font=("Consolas", 8, "bold"))
        c.create_line(script_x + SCRIPT_BOX_W, mid_y, output_x, mid_y, fill=BORDER, width=1)
        c.create_text(script_x + SCRIPT_BOX_W + VISUAL_GAP / 2, mid_y - 1, text="→", fill=GREEN, font=("Consolas", 8, "bold"))

        click_items = c.find_overlapping(input_x, y, output_x + OUTPUT_BOX_W, y + total_h)
        for item in click_items:
            c.tag_bind(item, "<Button-1>", lambda _e, cid=code_id: self.select_script_by_id(cid))
            c.tag_bind(item, "<Button-3>", lambda _e, cid=code_id: self.toggle_script_mark(cid))
            c.tag_bind(item, "<Enter>", lambda _e: c.configure(cursor="hand2"))
            c.tag_bind(item, "<Leave>", lambda _e: c.configure(cursor=""))

        return y + total_h + 12


    def _draw_virtual_script_node(self, rel_path: str, level: int, y: int) -> int:
        c = self.visual_canvas
        is_selected = self.selected_virtual_path == rel_path
        is_marked = rel_path in self.marked_virtual_scripts

        x = 22 + level * 34
        input_x = x
        script_x = input_x + INPUT_BOX_W + VISUAL_GAP
        output_x = script_x + SCRIPT_BOX_W + VISUAL_GAP

        input_lines = ["• -"]
        output_lines = ["• -"]

        input_h = VISUAL_BOX_PAD_Y * 2 + 18 + max(1, len(input_lines)) * VISUAL_LINE_H
        output_h = VISUAL_BOX_PAD_Y * 2 + 18 + max(1, len(output_lines)) * VISUAL_LINE_H
        script_h = 90
        total_h = max(input_h, script_h, output_h)

        fill = MARK_FILL if is_marked else ("#111111" if is_selected else SURFACE)
        outline = MARK_OUTLINE if is_marked else (BLUE if is_selected else BORDER_2)

        c.create_line(x - 18, y - 9, x - 18, y + total_h // 2, fill=BORDER, width=1)
        c.create_line(x - 18, y + total_h // 2, input_x - 5, y + total_h // 2, fill=BORDER, width=1)

        self._draw_visual_io_box(input_x, y, INPUT_BOX_W, "INPUT", input_lines, BLUE, outline)

        script_y = y + max(0, (total_h - script_h) // 2)
        c.create_rectangle(
            script_x,
            script_y,
            script_x + SCRIPT_BOX_W,
            script_y + script_h,
            fill=fill,
            outline=outline,
            width=1,
        )

        c.create_oval(script_x + 12, script_y + 14, script_x + 22, script_y + 24, fill=PURPLE, outline=PURPLE)
        c.create_text(
            script_x + 32,
            script_y + 11,
            text=(f"⭐ 🐍 {Path(rel_path).name}" if is_marked else f"🐍 {Path(rel_path).name}"),
            anchor="nw",
            fill=TEXT,
            font=("Consolas", 8, "bold"),
            width=SCRIPT_BOX_W - 44,
        )
        c.create_text(
            script_x + 12,
            script_y + 39,
            text="PLANNED CODE | no registry details yet",
            anchor="nw",
            fill=MUTED,
            font=FONT_TINY,
            width=SCRIPT_BOX_W - 24,
        )
        c.create_text(
            script_x + 12,
            script_y + 58,
            text=rel_path,
            anchor="nw",
            fill=MUTED,
            font=FONT_TINY,
            width=SCRIPT_BOX_W - 24,
        )

        self._draw_visual_io_box(output_x, y, OUTPUT_BOX_W, "OUTPUT", output_lines, GREEN, outline)

        mid_y = y + total_h // 2
        c.create_line(input_x + INPUT_BOX_W, mid_y, script_x, mid_y, fill=BORDER, width=1)
        c.create_text(input_x + INPUT_BOX_W + VISUAL_GAP / 2, mid_y - 1, text="→", fill=BLUE, font=("Consolas", 8, "bold"))
        c.create_line(script_x + SCRIPT_BOX_W, mid_y, output_x, mid_y, fill=BORDER, width=1)
        c.create_text(script_x + SCRIPT_BOX_W + VISUAL_GAP / 2, mid_y - 1, text="→", fill=GREEN, font=("Consolas", 8, "bold"))

        click_items = c.find_overlapping(input_x, y, output_x + OUTPUT_BOX_W, y + total_h)
        for item in click_items:
            c.tag_bind(item, "<Button-1>", lambda _e, p=rel_path: self.select_virtual_script(p))
            c.tag_bind(item, "<Button-3>", lambda _e, p=rel_path: self.toggle_virtual_script_mark(p))
            c.tag_bind(item, "<Enter>", lambda _e: c.configure(cursor="hand2"))
            c.tag_bind(item, "<Leave>", lambda _e: c.configure(cursor=""))

        return y + total_h + 12


    def _draw_visual_more_node(self, level: int, y: int, hidden: int) -> int:
        c = self.visual_canvas
        x = 22 + level * 34
        c.create_text(x, y + 10, text=f"+ {hidden} more files", anchor="w", fill=MUTED, font=FONT_TINY)
        return y + 24

    # --------------------------------------------------------
    # Main content
    # --------------------------------------------------------

    def _build_content(self):
        top = tk.Frame(self.content, bg=APP_BG)
        top.grid(row=0, column=0, sticky="ew", padx=26, pady=(24, 16))
        top.grid_columnconfigure(0, weight=1)

        tk.Label(top, textvariable=self.breadcrumb_var, bg=APP_BG, fg=TEXT, font=FONT_SMALL).grid(row=0, column=0, sticky="w")
        actions = tk.Frame(top, bg=APP_BG)
        actions.grid(row=0, column=1, sticky="e")
        self._button(actions, "↻  Rescan", self.run_scanner).pack(side="left", padx=(0, 12))
        self._button(actions, "📄  Copy Path", self.copy_selected_path).pack(side="left", padx=(0, 12))
        self._button(actions, "📋  Copy Code", self.copy_selected_file_content).pack(side="left", padx=(0, 12))
        self._button(actions, "📁  Copy Folder", self.copy_selected_folder_contents).pack(side="left", padx=(0, 12))
        self._button(actions, "💬  Copy For ChatGPT", self.copy_selected_context_for_chatgpt).pack(side="left")

        self.kpi_area = tk.Frame(self.content, bg=APP_BG)
        self.kpi_area.grid(row=1, column=0, sticky="ew", padx=26, pady=(0, 26))
        for i in range(4):
            self.kpi_area.grid_columnconfigure(i, weight=1)

        self.kpi_total = self._kpi_card(self.kpi_area, 0, "📄", "0", "Code-Dateien", "registriert", BLUE)
        self.kpi_ok = self._kpi_card(self.kpi_area, 1, "✓", "0", "OK", "keine Fehler", GREEN)
        self.kpi_warn = self._kpi_card(self.kpi_area, 2, "⚠", "0", "Warnings", "prüfen", YELLOW)
        self.kpi_missing = self._kpi_card(self.kpi_area, 3, "✕", "0", "Missing", "nicht gefunden", RED)

        body = tk.Frame(self.content, bg=APP_BG)
        body.grid(row=2, column=0, sticky="nsew", padx=26, pady=(0, 24))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.table_card = RoundedCard(body, bg=SURFACE)
        self.table_card.grid(row=0, column=0, sticky="nsew")
        self._build_table(self.table_card.inner)

        self.selected_card = RoundedCard(body, bg=SURFACE)
        self.selected_card.grid(row=1, column=0, sticky="ew", pady=(26, 0))
        self._build_selected_file(self.selected_card.inner)

        details_grid = tk.Frame(body, bg=APP_BG)
        details_grid.grid(row=2, column=0, sticky="ew", pady=(26, 0))
        details_grid.grid_columnconfigure(0, weight=1)
        details_grid.grid_columnconfigure(1, weight=1)
        details_grid.grid_columnconfigure(2, weight=1)
        self.description_box = self._info_box(details_grid, 0, "DESCRIPTION")
        self.io_box = self._info_box(details_grid, 1, "INPUTS / OUTPUTS")
        self.dep_box = self._info_box(details_grid, 2, "DEPENDENCIES")

        self.meta_card = RoundedCard(body, bg=SURFACE)
        self.meta_card.grid(row=3, column=0, sticky="ew", pady=(26, 0))
        self._build_meta_bar(self.meta_card.inner)

        tk.Label(body, text="QUANT TERMINAL · Code Registry v2.2.0", bg=APP_BG, fg=MUTED, font=FONT_TINY).grid(row=4, column=0, sticky="e", pady=(12, 0))

    def _kpi_card(self, parent, col: int, icon: str, value: str, title: str, subtitle: str, color: str):
        card = RoundedCard(parent, bg=SURFACE)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 10, 0 if col == 3 else 10))
        inner = card.inner
        inner.grid_columnconfigure(1, weight=1)
        tk.Label(inner, text=icon, bg=SURFACE, fg=color, font=("Consolas", 18, "bold"), width=3).grid(row=0, column=0, rowspan=2, padx=(18, 10), pady=22)
        val = tk.Label(inner, text=value, bg=SURFACE, fg=TEXT, font=("Consolas", 15, "bold"))
        val.grid(row=0, column=1, sticky="sw", pady=(22, 0))
        tk.Label(inner, text=title, bg=SURFACE, fg=TEXT, font=("Consolas", 8, "bold")).grid(row=1, column=1, sticky="nw")
        tk.Label(inner, text=subtitle, bg=SURFACE, fg=MUTED, font=FONT_SMALL).grid(row=2, column=1, sticky="nw", pady=(0, 18))
        return val

    def _build_table(self, parent):
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(fill="both", expand=True)
        sy = tk.Scrollbar(frame, orient="vertical")
        self.table = ttk.Treeview(frame, columns=self.TABLE_COLUMNS, show="headings", style="Registry.Treeview", yscrollcommand=sy.set)
        sy.config(command=self.table.yview)
        self.table.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        widths = {
            "file_name": 170, "script_name": 260, "layer": 140, "domain": 130,
            "asset_type": 125, "status": 100, "version": 95,
        }
        for col in self.TABLE_COLUMNS:
            self.table.heading(col, text=self.TABLE_LABELS[col])
            self.table.column(col, width=widths[col], minwidth=70, anchor="w", stretch=True)

        self.table.tag_configure("ok", background=SURFACE, foreground=TEXT)
        self.table.tag_configure("alt", background=ROW_ALT, foreground=TEXT)
        self.table.tag_configure("warn", background="#332000", foreground=TEXT)
        self.table.tag_configure("missing", background="#300A0A", foreground=TEXT)
        self.table.bind("<<TreeviewSelect>>", self.on_select)
        self.table.bind("<Double-1>", lambda _e: self.open_file())

    def _build_selected_file(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        tk.Label(parent, text="SELECTED CODE FILE", bg=SURFACE, fg=BLUE, font=FONT_H2).grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(18, 12))
        tk.Label(parent, text="📄", bg="#332000", fg=BLUE, font=("Consolas", 18, "bold"), width=3).grid(row=1, column=0, rowspan=4, padx=(20, 14), pady=(0, 20))
        self.sel_title = tk.Label(parent, text="NO FILE SELECTED", bg=SURFACE, fg=TEXT, font=FONT_TITLE)
        self.sel_title.grid(row=1, column=1, sticky="w")
        self.sel_sub = tk.Label(parent, text="", bg=SURFACE, fg=TEXT_2, font=FONT_SMALL)
        self.sel_sub.grid(row=2, column=1, sticky="w")

        self.sel_kv = tk.Frame(parent, bg=SURFACE)
        self.sel_kv.grid(row=1, column=2, rowspan=4, sticky="ew", padx=(20, 20), pady=(0, 20))
        self.sel_kv2 = tk.Frame(parent, bg=SURFACE)
        self.sel_kv2.grid(row=1, column=3, rowspan=4, sticky="ew", padx=(20, 20), pady=(0, 20))

    def _info_box(self, parent, col: int, title: str) -> tk.Text:
        card = RoundedCard(parent, bg=SURFACE)
        card.grid(row=0, column=col, sticky="nsew", padx=(0 if col == 0 else 10, 0 if col == 2 else 10))
        tk.Label(card.inner, text=title, bg=SURFACE, fg=BLUE, font=FONT_H2).pack(anchor="w", padx=20, pady=(18, 10))
        box = tk.Text(card.inner, bg=SURFACE, fg=TEXT, relief="solid", bd=1, wrap="word", height=7, font=FONT_SMALL)
        box.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        box.configure(state="disabled")
        return box

    def _build_meta_bar(self, parent):
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_columnconfigure(2, weight=1)
        parent.grid_columnconfigure(3, weight=1)
        self.meta_labels: list[tk.Label] = []
        for i, title in enumerate(["FILE PATH", "FILE SIZE", "CREATED", "SCANNED"]):
            frame = tk.Frame(parent, bg=SURFACE)
            frame.grid(row=0, column=i, sticky="ew", padx=20, pady=18)
            tk.Label(frame, text=title, bg=SURFACE, fg=MUTED, font=FONT_TINY).pack(anchor="w")
            label = tk.Label(frame, text="-", bg=SURFACE, fg=TEXT, font=FONT_SMALL, wraplength=260, justify="left")
            label.pack(anchor="w")
            self.meta_labels.append(label)

    # --------------------------------------------------------
    # Data/load
    # --------------------------------------------------------

    def reload(self):
        self.df_all = self.repo.load_assets()
        if self.df_all.empty:
            self.update_kpis()
            self.populate_table(pd.DataFrame())
            self.show_empty_details()
            self.draw_visual_folder_tree()
            return
        self.normalize_df()
        self.df_current = self.df_all.copy()
        self.populate_table(self.df_current)
        self.update_kpis()
        self.show_empty_details()
        self.draw_visual_folder_tree()

    def normalize_df(self):
        for col in ["relative_path", "file_name", "script_name", "script_id", "layer", "domain", "asset_type", "status", "version", "purpose", "inputs", "outputs", "dependencies"]:
            if col not in self.df_all.columns:
                self.df_all[col] = ""
        self.df_all["folder_parts"] = self.df_all["relative_path"].apply(lambda x: list(Path(safe_str(x)).parts[:-1]))
        self.df_all["folder_path"] = self.df_all["folder_parts"].apply(lambda parts: str(Path(*parts)) if parts else "")

    def update_kpis(self):
        if self.df_all.empty:
            for label in [self.kpi_total, self.kpi_ok, self.kpi_warn, self.kpi_missing]:
                label.config(text="0")
            return
        total = len(self.df_all)
        q = self.df_all["registry_quality_status"].astype(str).str.lower()
        ok = int(((self.df_all["has_registry"].astype(int) == 1) & (q.isin(["ok", "passed"]))).sum())
        warn = int(q.str.contains("warn", na=False).sum())
        missing = int((self.df_all["has_registry"].astype(int) == 0).sum())
        self.kpi_total.config(text=str(total))
        self.kpi_ok.config(text=str(ok))
        self.kpi_warn.config(text=str(warn))
        self.kpi_missing.config(text=str(missing))

    # --------------------------------------------------------
    # Navigation logic
    # --------------------------------------------------------

    def get_child_folders(self, parent_path: str) -> list[str]:
        folders = set()
        parent_parts = list(Path(parent_path).parts) if parent_path else []

        if not self.df_all.empty and "folder_parts" in self.df_all.columns:
            for parts in self.df_all["folder_parts"].tolist():
                if len(parts) <= len(parent_parts):
                    continue
                if parts[: len(parent_parts)] == parent_parts:
                    folders.add(str(Path(*parts[: len(parent_parts) + 1])).replace("\\", "/"))

        for rel_path, meta in self.virtual_nodes.items():
            parts = list(Path(rel_path).parts)
            if meta.get("kind") == "file":
                parts = parts[:-1]
            if len(parts) <= len(parent_parts):
                continue
            if parts[: len(parent_parts)] == parent_parts:
                folders.add(str(Path(*parts[: len(parent_parts) + 1])).replace("\\", "/"))

        return sorted(folders, key=lambda x: x.lower())

    def count_folder_scripts(self, folder_path: str) -> int:
        prefix = normalize_rel_path(folder_path)
        real_count = 0
        if not self.df_all.empty:
            if not prefix:
                real_count = len(self.df_all)
            else:
                real_count = int(self.df_all["relative_path"].astype(str).str.replace("\\", "/", regex=False).str.startswith(prefix + "/").sum())

        virtual_count = 0
        for rel_path, meta in self.virtual_nodes.items():
            if meta.get("kind") != "file":
                continue
            if not prefix or rel_path.startswith(prefix + "/"):
                virtual_count += 1

        return int(real_count + virtual_count)

    def get_direct_scripts(self, folder_path: str) -> pd.DataFrame:
        if self.df_all.empty:
            return pd.DataFrame()
        d = self.df_all[self.df_all["folder_path"].astype(str).str.replace("\\", "/", regex=False) == normalize_rel_path(folder_path)].copy()
        if d.empty:
            return d
        return d.sort_values(["file_name", "script_name"]).reset_index(drop=True)

    def toggle_folder(self, key: str, folder_path: str):
        if key in self.expanded_keys:
            self.expanded_keys.remove(key)
        else:
            self.expanded_keys.add(key)
        self.select_folder(folder_path)

    def select_folder(self, folder_path: str):
        self.selected_code_id = None
        self.selected_virtual_path = ""
        self.selected_folder_path = folder_path
        prefix = normalize_rel_path(folder_path)
        self.df_current = self.df_all[
            self.df_all["relative_path"].astype(str).str.replace("\\", "/", regex=False).str.startswith(prefix + "/")
        ].copy()
        self.breadcrumb_var.set("Dashboard  ›  " + folder_path.replace("\\", "  ›  "))
        self.apply_search()

    def select_script_by_id(self, code_id: int):
        self.selected_virtual_path = ""
        self.selected_code_id = int(code_id)
        row = self.get_selected_row()
        if row is None:
            return
        self.selected_folder_path = safe_str(row.get("folder_path"))
        self.breadcrumb_var.set("Dashboard  ›  " + safe_str(row.get("relative_path")).replace("\\", "  ›  "))
        self.populate_table(self.df_current if not self.df_current.empty else self.df_all)
        if str(code_id) in self.table.get_children(""):
            self.table.selection_set(str(code_id))
            self.table.focus(str(code_id))
            self.table.see(str(code_id))
        self.show_details(row)

    # --------------------------------------------------------
    # Table/search/details
    # --------------------------------------------------------

    def apply_search(self):
        base = self.df_current if not self.df_current.empty else self.df_all
        d = base.copy()
        q = self.search_var.get().strip().lower()
        if q:
            mask = pd.Series(False, index=d.index)
            for col in ["file_name", "script_name", "script_id", "relative_path", "layer", "domain", "asset_type", "purpose", "inputs", "outputs", "dependencies", "status", "version"]:
                if col in d.columns:
                    mask = mask | d[col].astype(str).str.lower().str.contains(q, na=False)
            d = d[mask]
        self.populate_table(d.reset_index(drop=True))
        self.draw_visual_folder_tree()

    def populate_table(self, df: pd.DataFrame):
        self.table.delete(*self.table.get_children())
        if df.empty:
            return
        for idx, (_, row) in enumerate(df.iterrows()):
            values = []
            for col in self.TABLE_COLUMNS:
                v = safe_str(row.get(col))
                if col == "status" and v:
                    v = f"●  {v}"
                values.append(v)
            quality = safe_str(row.get("registry_quality_status")).lower()
            has_registry = int(row.get("has_registry", 0) or 0)
            tag = "alt" if idx % 2 else "ok"
            if has_registry == 0:
                tag = "missing"
            elif "warn" in quality:
                tag = "warn"
            self.table.insert("", "end", iid=str(row.get("id")), values=values, tags=(tag,))

    def on_select(self, _event=None):
        self.selected_virtual_path = ""
        selected = self.table.selection()
        if not selected:
            return
        self.selected_code_id = int(selected[0])
        row = self.get_selected_row()
        if row is not None:
            self.selected_folder_path = safe_str(row.get("folder_path"))
            self.show_details(row)
            self.draw_visual_folder_tree()

    def get_selected_row(self) -> Optional[pd.Series]:
        if self.selected_code_id is None or self.df_all.empty:
            return None
        row = self.df_all[self.df_all["id"].astype(int) == int(self.selected_code_id)]
        if row.empty:
            return None
        return row.iloc[0]

    def show_empty_details(self):
        self.sel_title.config(text="NO FILE SELECTED")
        self.sel_sub.config(text="SELECT A CODE FILE FROM TABLE OR CODE TREE.")
        self._set_kv(self.sel_kv, [])
        self._set_kv(self.sel_kv2, [])
        self._set_text(self.description_box, "")
        self._set_text(self.io_box, "")
        self._set_text(self.dep_box, "")
        for label in self.meta_labels:
            label.config(text="-")

    def show_details(self, row: pd.Series):
        file_name = safe_str(row.get("file_name"))
        folder = Path(safe_str(row.get("relative_path"))).parent
        self.sel_title.config(text=file_name or "code.py")
        self.sel_sub.config(text=f"({ellipsize(row.get('script_name'), 42)})")
        self._set_kv(self.sel_kv, [
            ("Script ID", row.get("script_id")),
            ("Layer", row.get("layer")),
            ("Domain", row.get("domain")),
            ("Asset Type", row.get("asset_type")),
        ])
        self._set_kv(self.sel_kv2, [
            ("Status", f"● {safe_str(row.get('status'))}"),
            ("Version", row.get("version")),
            ("Registry Status", safe_str(row.get("registry_quality_status")) or "OK"),
            ("Letzte Änderung", row.get("last_modified_utc")),
        ])

        self._set_text(self.description_box, safe_str(row.get("purpose")) or "Keine DESCRIPTION registriert.")
        io_lines = ["Inputs"]
        io_lines += [f"• {x}" for x in split_block(row.get("inputs"))] or ["• -"]
        io_lines += ["", "Outputs"]
        io_lines += [f"• {x}" for x in split_block(row.get("outputs"))] or ["• -"]
        self._set_text(self.io_box, "\n".join(io_lines))
        deps = split_block(row.get("dependencies"))
        self._set_text(self.dep_box, "\n".join([f"• {d}" for d in deps]) if deps else "• -")

        path = Path(safe_str(row.get("file_path")))
        size = "-"
        try:
            size = f"{path.stat().st_size / 1024:.1f} KB"
        except Exception:
            if safe_str(row.get("file_size_bytes")):
                try:
                    size = f"{float(row.get('file_size_bytes')) / 1024:.1f} KB"
                except Exception:
                    size = safe_str(row.get("file_size_bytes"))
        self.meta_labels[0].config(text=str(folder))
        self.meta_labels[1].config(text=size)
        self.meta_labels[2].config(text=safe_str(row.get("created_date")) or safe_str(row.get("last_reviewed")) or "-")
        self.meta_labels[3].config(text=safe_str(row.get("scanned_at_utc")) or "-")

    def _set_kv(self, parent: tk.Frame, pairs: list[tuple[str, object]]):
        for w in parent.winfo_children():
            w.destroy()
        for i, (k, v) in enumerate(pairs):
            tk.Label(parent, text=k, bg=SURFACE, fg=MUTED, font=FONT_TINY).grid(row=i, column=0, sticky="w", padx=(0, 22), pady=4)
            color = GREEN if k in {"Status", "Version", "Registry Status"} else TEXT
            tk.Label(parent, text=safe_str(v), bg=SURFACE, fg=color, font=("Consolas", 8, "bold") if k in {"Status", "Version", "Registry Status"} else FONT_SMALL).grid(row=i, column=1, sticky="w", pady=4)

    def _set_text(self, box: tk.Text, text: str):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("end", text)
        box.configure(state="disabled")

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------

    def selected_path(self) -> Optional[Path]:
        row = self.get_selected_row()
        if row is None:
            return None
        p = Path(safe_str(row.get("file_path")))
        return p if p.exists() else None

    def copy_to_clipboard(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    def copy_selected_path(self):
        if self.selected_virtual_path:
            self.copy_to_clipboard(str(QUANT_ROOT / self.selected_virtual_path))
            return
        row = self.get_selected_row()
        if row is None:
            messagebox.showwarning("Keine Datei", "Kein Script ausgewählt.")
            return
        self.copy_to_clipboard(safe_str(row.get("file_path")))

    def open_file(self):
        if self.selected_virtual_path:
            p = QUANT_ROOT / self.selected_virtual_path
            if p.exists():
                os.startfile(str(p))
                return
        p = self.selected_path()
        if p is None:
            messagebox.showwarning("Keine Datei", "Keine gültige Datei ausgewählt.")
            return
        os.startfile(str(p))

    def run_scanner(self):
        if self.repo.run_scanner():
            self.reload()


    def _safe_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(QUANT_ROOT)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    def _selected_real_file_path(self) -> Optional[Path]:
        if getattr(self, "selected_virtual_path", ""):
            p = QUANT_ROOT / self.selected_virtual_path
            return p if p.exists() and p.is_file() else None

        row = self.get_selected_row()
        if row is None:
            return None

        p = Path(safe_str(row.get("file_path")))
        return p if p.exists() and p.is_file() else None

    def _selected_real_folder_path(self) -> Optional[Path]:
        if self.selected_folder_path:
            p = QUANT_ROOT / self.selected_folder_path
            if p.exists() and p.is_dir():
                return p

        if getattr(self, "selected_virtual_path", ""):
            p = (QUANT_ROOT / self.selected_virtual_path).parent
            return p if p.exists() and p.is_dir() else None

        row = self.get_selected_row()
        if row is not None:
            p = Path(safe_str(row.get("file_path")))
            if p.exists():
                return p.parent if p.is_file() else p

        return None

    def _read_text_file_for_chatgpt(self, path: Path) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            content = f"[READ_ERROR] {exc}"

        rel = self._safe_relative(path)
        return (
            f"# FILE: {rel}\n"
            f"# FULL_PATH: {path}\n"
            f"# COPY_FOR_CHATGPT\n\n"
            f"{content}"
        )

    def copy_selected_file_content(self):
        p = self._selected_real_file_path()
        if p is None:
            messagebox.showwarning("Keine Datei", "Keine echte Code-Datei ausgewählt.")
            return

        self.copy_to_clipboard(self._read_text_file_for_chatgpt(p))
        messagebox.showinfo("Copied", "Kompletter Dateiinhalt wurde direkt in die Zwischenablage kopiert.")

    def copy_selected_folder_contents(self):
        folder = self._selected_real_folder_path()
        if folder is None:
            messagebox.showwarning("Kein Ordner", "Kein echter Ordner ausgewählt.")
            return

        allowed_suffixes = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv"}
        excluded_parts = {"__pycache__", ".git", ".venv", "venv", "env"}
        max_files = 100
        max_chars = 350_000

        files = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in allowed_suffixes
            and not any(part in excluded_parts for part in p.parts)
        ]
        files = sorted(files, key=lambda x: str(x).lower())[:max_files]

        folder_rel = self._safe_relative(folder)
        chunks: list[str] = [
            f"# FOLDER: {folder_rel}\n",
            f"# FULL_PATH: {folder}\n",
            "# COPY_FOR_CHATGPT\n",
            f"# FILE_COUNT_INCLUDED: {len(files)}\n\n",
        ]

        if hasattr(self, "build_expanded_ascii_tree"):
            chunks.append("# TREE\n")
            chunks.append(self.build_expanded_ascii_tree())
            chunks.append("\n\n")

        total_chars = sum(len(x) for x in chunks)

        for p in files:
            block = (
                "\n\n# ============================================================\n"
                f"# FILE: {self._safe_relative(p)}\n"
                "# ============================================================\n\n"
            )

            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                content = f"[READ_ERROR] {exc}"

            block += content

            if total_chars + len(block) > max_chars:
                chunks.append("\n\n# EXPORT_TRUNCATED: clipboard size limit reached")
                break

            chunks.append(block)
            total_chars += len(block)

        self.copy_to_clipboard("".join(chunks))
        messagebox.showinfo("Copied", "Ordnerinhalt wurde direkt in die Zwischenablage kopiert.")

    def copy_selected_context_for_chatgpt(self):
        p = self._selected_real_file_path()
        if p is not None:
            self.copy_selected_file_content()
            return

        folder = self._selected_real_folder_path()
        if folder is not None:
            self.copy_selected_folder_contents()
            return

        if hasattr(self, "build_expanded_ascii_tree"):
            self.copy_to_clipboard(self.build_expanded_ascii_tree())
            messagebox.showinfo("Copied", "Tree wurde direkt in die Zwischenablage kopiert.")
            return

        messagebox.showwarning("Keine Auswahl", "Keine Datei oder Ordner ausgewählt.")


    def clean_stale_records(self):
        if self.repo.cleanup_stale_records():
            self.reload()


# ============================================================
# PANEL API / STANDALONE
# ============================================================


def build_panel(parent, repository: Optional[CodeRegistryRepository] = None, **kwargs):
    return CodeRegistryDashboardBlock(parent, repository=repository, **kwargs)


def main():
    app = tk.Tk()
    app.title("CODE REGISTRY Dashboard")
    app.geometry("1900x1000")
    app.minsize(1500, 780)
    app.configure(bg=APP_BG)
    block = CodeRegistryDashboardBlock(app)
    block.pack(fill="both", expand=True)
    print("QUANT_ROOT:", QUANT_ROOT)
    print("REGISTRY_DB:", REGISTRY_DB)
    print("SCANNER_CODE:", SCANNER_CODE)
    app.mainloop()


if __name__ == "__main__":
    main()

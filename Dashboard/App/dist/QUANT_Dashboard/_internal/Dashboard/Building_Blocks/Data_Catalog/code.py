# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: data_catalog_dashboard
# script_name: Data Catalog Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Catalog
# asset_type: Dashboard
# purpose: Terminal/Bloomberg-inspired interface for browsing all assets inside the Quant Data Center with a wide visual folder map, searchable asset table, metadata inspection and quality overview.
# inputs:
#   - Data/5_Catalog/catalog.db
# outputs:
#   - Dashboard UI
#   - File navigation
#   - Metadata inspection
# dependencies:
#   - tkinter
#   - pandas
#   - sqlite3
#   - pathlib
#   - os
#   - subprocess
# schedule: manual
# version: v1.2.0_terminal_bloomberg_ui
# last_reviewed: 2026-06-04
# ============================================================

from __future__ import annotations

import os
import sys
import sqlite3
import subprocess
import textwrap
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
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

TEXT = "#FFFFFF"
TEXT_2 = "#D6D6D6"
MUTED = "#B8B8B8"

BORDER = "#2A2A2A"
BORDER_2 = "#3A3A3A"

BLUE = "#FF9900"
GREEN = "#00FF66"
YELLOW = "#FFD400"
RED = "#FF4444"
PURPLE = "#00AEEF"
WHITE = "#FFFFFF"

ORANGE = "#FF9900"
CYAN = "#00AEEF"

ROW_ALT = "#050505"
ROW_SELECTED = "#332000"
MARK_FILL = "#332000"
MARK_OUTLINE = "#FFD400"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_H1 = ("Consolas", 13, "bold")
FONT_H2 = ("Consolas", 10, "bold")
FONT_BODY = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_MONO = ("Consolas", 9)

VISUAL_TREE_W = 1500
VISUAL_CANVAS_MIN_W = 2600
DATA_BOX_W = 440
ASSET_BOX_W = 420
META_BOX_W = 420
LINE_H = 15

MAX_CLIPBOARD_CHARS = 900_000
FOLDER_COPY_MAX_FILES = 80


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_quant_root(start: Path) -> Path:
    for p in [start.resolve()] + list(start.resolve().parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p.resolve()
    return start.resolve().parent


QUANT_ROOT = find_quant_root(SCRIPT_PATH)
DATA_CENTER_DIR = QUANT_ROOT / "Data_Center"
DATA_DIR = DATA_CENTER_DIR / "Data"
CATALOG_DB = DATA_DIR / "5_Catalog" / "catalog.db"
SCANNER_CODE = DATA_CENTER_DIR / "Backend_Management" / "5_Data_Catalog" / "code.py"


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


def normalize_rel_path(path_text: str) -> str:
    return safe_str(path_text).replace("\\", "/")


def wrap_text(value: object, width: int = 44) -> list[str]:
    s = safe_str(value).strip()
    if not s:
        return ["-"]
    result: list[str] = []
    for raw in s.replace("\\", "/").split("\n"):
        raw = raw.strip()
        if not raw:
            continue
        wrapped = textwrap.wrap(raw, width=width, break_long_words=True, break_on_hyphens=False)
        result.extend(wrapped if wrapped else [raw])
    return result or ["-"]


def quality_color(value: object) -> str:
    q = safe_str(value).strip().lower()
    if q in {"passed", "ok", "valid", "good"}:
        return GREEN
    if q in {"warning", "warn", "partial"}:
        return YELLOW
    if q in {"failed", "fail", "error", "missing"}:
        return RED
    return MUTED


# ============================================================
# REPOSITORY
# ============================================================


class DataCatalogRepository:
    def __init__(self, db_path: Path = CATALOG_DB):
        self.db_path = db_path

    def load_assets(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM data_assets ORDER BY relative_path", conn)
            return self._clean_for_dashboard(df)
        except Exception as exc:
            messagebox.showerror("DB Fehler", str(exc))
            return pd.DataFrame()

    def _clean_for_dashboard(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        d = df.copy()
        required = [
            "id", "file_path", "relative_path", "file_name", "stage", "domain", "asset_type",
            "account_type", "symbol", "strategy_id", "sample_type", "rows_count", "columns_count",
            "quality_status", "quality_score", "quality_message", "file_size_mb", "date_start",
            "date_end", "last_modified_utc", "scanned_at_utc", "checksum", "columns_list",
        ]
        for col in required:
            if col not in d.columns:
                d[col] = ""

        d["file_path"] = d["file_path"].fillna("").astype(str)
        d["relative_path"] = d["relative_path"].fillna("").astype(str)

        def exists(path_text: str) -> bool:
            if not path_text:
                return True
            try:
                return Path(path_text).exists()
            except Exception:
                return True

        d["file_exists"] = d["file_path"].apply(exists)

        def rel(row: pd.Series) -> str:
            current = safe_str(row.get("relative_path")).strip()
            if current:
                return current
            try:
                return str(Path(safe_str(row.get("file_path"))).relative_to(QUANT_ROOT))
            except Exception:
                return safe_str(row.get("file_path"))

        d["relative_path"] = d.apply(rel, axis=1)
        d["file_name"] = d["file_name"].fillna("").astype(str)
        d.loc[d["file_name"].str.strip() == "", "file_name"] = d["relative_path"].apply(lambda x: Path(safe_str(x)).name)

        for col in ["stage", "domain", "asset_type", "account_type", "symbol", "strategy_id", "sample_type", "quality_status"]:
            d[col] = d[col].fillna("").astype(str)

        d["_id_sort"] = pd.to_numeric(d["id"], errors="coerce").fillna(0)
        d = d.sort_values(["relative_path", "_id_sort"], ascending=[True, False])
        d = d.drop_duplicates(subset=["relative_path"], keep="first").copy()
        d = d.sort_values("relative_path").reset_index(drop=True)
        return d

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
    def __init__(self, parent, bg=SURFACE, border=BORDER, **kwargs):
        super().__init__(parent, bg=border, **kwargs)
        self.inner = tk.Frame(self, bg=bg)
        self.inner.pack(fill="both", expand=True, padx=1, pady=1)


# ============================================================
# DASHBOARD
# ============================================================


class DataCatalogDashboardBlock(tk.Frame):
    FILE_COLUMNS = [
        "file_name", "stage", "domain", "asset_type", "symbol", "strategy_id",
        "account_type", "sample_type", "rows_count", "columns_count", "quality_status", "file_size_mb",
    ]

    FILE_LABELS = {
        "file_name": "File Name",
        "stage": "Stage",
        "domain": "Domain",
        "asset_type": "Asset Type",
        "symbol": "Symbol",
        "strategy_id": "Strategy ID",
        "account_type": "Account",
        "sample_type": "Sample",
        "rows_count": "Rows",
        "columns_count": "Columns",
        "quality_status": "Quality",
        "file_size_mb": "Size MB",
    }

    def __init__(self, parent, repository: Optional[DataCatalogRepository] = None, **kwargs):
        super().__init__(parent, bg=APP_BG, **kwargs)
        self.repo = repository or DataCatalogRepository()
        self.df_all = pd.DataFrame()
        self.df_current = pd.DataFrame()
        self.selected_file_id: Optional[int] = None
        self.selected_folder_path = ""
        self.expanded_keys: set[str] = set()
        self.marked_folders: set[str] = set()
        self.marked_assets: set[int] = set()
        self.marked_virtual_assets: set[str] = set()
        self.virtual_nodes: dict[str, dict[str, str]] = {}
        self.selected_virtual_path = ""
        self.search_var = tk.StringVar()
        self.breadcrumb_var = tk.StringVar(value="Dashboard  ›  Data_Center  ›  Data_Catalog")
        self._build_ui()
        self.reload()

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def _build_ui(self):
        self._setup_style()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.main_pane = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=APP_BG,
            sashwidth=8,
            bd=0,
            showhandle=False,
        )
        self.main_pane.grid(row=0, column=0, sticky="nsew")

        self.visual_tree_panel = tk.Frame(
            self.main_pane,
            bg=SURFACE_2,
            width=VISUAL_TREE_W,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        self.visual_tree_panel.grid_propagate(False)

        self.content = tk.Frame(self.main_pane, bg=APP_BG)
        self.content.grid_columnconfigure(0, weight=1)
        self.content.grid_rowconfigure(2, weight=1)

        self.main_pane.add(self.visual_tree_panel, minsize=900, stretch="always")
        self.main_pane.add(self.content, minsize=500, stretch="always")

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
            "Catalog.Treeview",
            background="#000000",
            foreground=TEXT,
            fieldbackground="#000000",
            rowheight=22,
            borderwidth=1,
            relief="solid",
            font=FONT_SMALL,
        )
        style.configure(
            "Catalog.Treeview.Heading",
            background="#111111",
            foreground=ORANGE,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 8, "bold"),
            padding=(6, 7),
        )
        style.map(
            "Catalog.Treeview",
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
    # Visual folder map
    # --------------------------------------------------------

    def _build_visual_tree_panel(self):
        header = tk.Frame(self.visual_tree_panel, bg=SURFACE_2)
        header.pack(fill="x", padx=14, pady=(18, 10))

        tk.Label(header, text="DATA CENTER MAP", bg=SURFACE_2, fg=TEXT, font=FONT_H2).pack(anchor="w")
        tk.Label(
            header,
            text="DATA → ASSET → META terminal map",
            bg=SURFACE_2,
            fg=MUTED,
            font=FONT_TINY,
        ).pack(anchor="w", pady=(2, 0))

        search_wrap = tk.Frame(self.visual_tree_panel, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER)
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
            activebackground=ROW_SELECTED,
            activeforeground=BLUE,
            relief="solid",
            bd=1,
                        padx=8,
            cursor="hand2",
            font=FONT_TINY,
        ).pack(side="right", padx=(0, 6))

        toolbar = tk.Frame(self.visual_tree_panel, bg=SURFACE_2)
        toolbar.pack(fill="x", padx=14, pady=(0, 10))
        self._small_visual_button(toolbar, "Expand", self.expand_visual_all).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Collapse", self.collapse_visual_all).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Refresh", self.reload).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "New Folder", self.add_virtual_folder).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "New CSV", lambda: self.add_virtual_file("csv")).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "New Parquet", lambda: self.add_virtual_file("parquet")).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Mark Selected", self.toggle_mark_selected).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Clear Marks", self.clear_marks).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Copy Tree", self.copy_expanded_tree).pack(side="left", padx=(0, 6))
        self._small_visual_button(toolbar, "Copy Data", self.copy_selected_data_content).pack(side="left", padx=(0, 6))

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

        self.visual_canvas.configure(yscrollcommand=self.visual_y.set, xscrollcommand=self.visual_x.set)
        self.visual_canvas.bind("<Configure>", lambda _e: self.draw_visual_folder_tree())
        self.visual_canvas.bind("<MouseWheel>", self._visual_mousewheel)
        self.visual_canvas.bind("<Shift-MouseWheel>", self._visual_shift_mousewheel)

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

    def _visual_shift_mousewheel(self, event):
        try:
            self.visual_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
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

    def add_virtual_folder(self):
        name = simpledialog.askstring(
            "New Folder",
            "Folder path relative to Data_Center/Data or relative to selected folder:",
            parent=self,
        )
        if not name:
            return

        rel_path = self._join_virtual_path(name)
        if not rel_path:
            return

        target_dir = DATA_DIR / rel_path

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

        self.selected_file_id = None
        self.selected_virtual_path = ""
        self.selected_folder_path = rel_path
        self._ensure_parent_expanded(rel_path)
        self.expanded_keys.add(f"folder::{rel_path}")

        self.reload()
        self.selected_folder_path = rel_path
        self.expanded_keys.add(f"folder::{rel_path}")
        self.draw_visual_folder_tree()

    def add_virtual_file(self, file_type: str):
        ext = "csv" if file_type.lower() == "csv" else "parquet"
        name = simpledialog.askstring(
            f"New {ext.upper()}",
            f"{ext.upper()} placeholder path. The file is NOT created; only shown in the tree:",
            parent=self,
        )
        if not name:
            return
        rel_path = self._join_virtual_path(name, default_ext=ext)
        if not rel_path:
            return
        self.virtual_nodes[rel_path] = {"kind": "file", "file_type": ext, "created_on_disk": "no", "physical_file": "no"}
        self.selected_file_id = None
        self.selected_virtual_path = rel_path
        parent = str(Path(rel_path).parent).replace("\\", "/")
        self.selected_folder_path = "" if parent == "." else parent
        self._ensure_parent_expanded(rel_path)
        self.show_virtual_details(rel_path)
        self.draw_visual_folder_tree()

    def get_direct_virtual_assets(self, folder_path: str) -> list[str]:
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

    def select_virtual_asset(self, rel_path: str):
        rel_path = self._clean_virtual_path(rel_path)
        self.selected_file_id = None
        self.selected_virtual_path = rel_path
        parent = str(Path(rel_path).parent).replace("\\", "/")
        self.selected_folder_path = "" if parent == "." else parent
        self.breadcrumb_var.set("Dashboard  ›  " + rel_path.replace("\\", "  ›  ").replace("/", "  ›  "))
        self.show_virtual_details(rel_path)
        self.draw_visual_folder_tree()

    def show_virtual_details(self, rel_path: str):
        meta = self.virtual_nodes.get(rel_path, {})
        file_type = meta.get("file_type", Path(rel_path).suffix.replace(".", "").upper() or "virtual")
        file_name = Path(rel_path).name
        folder = str(Path(rel_path).parent).replace("\\", "/")
        if folder == ".":
            folder = ""
        self.sel_title.config(text=f"PLANNED: {file_name}")
        self.sel_sub.config(text=f"{rel_path} | virtual {file_type.upper()} | not created on disk")
        self._set_kv(self.sel_kv, [
            ("Stage", self._stage_from_path(rel_path)),
            ("Domain", "PLANNED"),
            ("Asset Type", file_type.upper()),
            ("Symbol", "-"),
        ])
        self._set_kv(self.sel_kv2, [
            ("Quality", "planned"),
            ("Rows", "0"),
            ("Columns", "0"),
            ("Size MB", "0"),
        ])
        self._set_text(self.profile_box, "\n".join([
            f"File Name: {file_name}",
            f"Virtual Path: {rel_path}",
            f"Type: {file_type.upper()}",
            "Status: PLANNED / COMMUNICATION ONLY",
            "Physical File: not created by Data Catalog",
            "Creation Rule: file will be created later by the responsible pipeline/code",
        ]))
        self._set_text(self.quality_box, "Quality Status: planned\nQuality Score: -\nQuality Message: virtual placeholder for planning")
        self._set_text(self.columns_box, "• planned schema not defined")
        self.meta_labels[0].config(text=folder or "-")
        self.meta_labels[1].config(text="-")
        self.meta_labels[2].config(text="-")
        self.meta_labels[3].config(text="-")

    def _stage_from_path(self, rel_path: str) -> str:
        parts = Path(rel_path).parts
        first = parts[0] if parts else ""
        mapping = {
            "1_Pipeline": "Pipeline",
            "2_Baseline": "Baseline",
            "3_Research": "Research",
            "4_Production": "Production",
            "5_Catalog": "Catalog",
            "6_Code_Registry": "Code Registry",
        }
        return mapping.get(first, first or "-")

    def toggle_virtual_asset_mark(self, rel_path: str):
        rel_path = self._clean_virtual_path(rel_path)
        if rel_path in self.marked_virtual_assets:
            self.marked_virtual_assets.remove(rel_path)
        else:
            self.marked_virtual_assets.add(rel_path)
        self.selected_virtual_path = rel_path
        self.selected_file_id = None
        self.draw_visual_folder_tree()


    def toggle_folder_mark(self, folder_path: str):
        if folder_path in self.marked_folders:
            self.marked_folders.remove(folder_path)
        else:
            self.marked_folders.add(folder_path)
        self.selected_folder_path = folder_path
        self.draw_visual_folder_tree()

    def toggle_asset_mark(self, file_id: int):
        file_id = int(file_id)
        if file_id in self.marked_assets:
            self.marked_assets.remove(file_id)
        else:
            self.marked_assets.add(file_id)
        self.selected_file_id = file_id
        self.draw_visual_folder_tree()

    def toggle_mark_selected(self):
        if self.selected_virtual_path:
            self.toggle_virtual_asset_mark(self.selected_virtual_path)
            return
        if self.selected_file_id is not None:
            self.toggle_asset_mark(int(self.selected_file_id))
            return
        if self.selected_folder_path:
            self.toggle_folder_mark(self.selected_folder_path)
            return
        messagebox.showwarning("Keine Auswahl", "Bitte zuerst einen Ordner oder eine CSV/Datei auswählen.")

    def clear_marks(self):
        self.marked_folders.clear()
        self.marked_assets.clear()
        self.marked_virtual_assets.clear()
        self.draw_visual_folder_tree()

    def copy_expanded_tree(self):
        if self.df_all.empty and not self.virtual_nodes:
            return
        self.copy_to_clipboard(self.build_expanded_ascii_tree())

    def build_expanded_ascii_tree(self) -> str:
        virtual_file_count = sum(1 for m in self.virtual_nodes.values() if m.get("kind") == "file")
        lines: list[str] = [f"DATA CENTER ({len(self.df_all)} real assets + {virtual_file_count} planned assets)"]
        root_folders = self.get_child_folders("")
        for idx, folder in enumerate(root_folders):
            self._append_tree_folder(lines, folder, prefix="", is_last=(idx == len(root_folders) - 1))
        if self.marked_folders or self.marked_assets or self.marked_virtual_assets:
            lines.append("")
            lines.append("MARKED ITEMS")
            lines.append("------------")
            for folder in sorted(self.marked_folders, key=lambda x: x.lower()):
                lines.append(f"[MARKED FOLDER] {folder}")
            if self.marked_assets:
                marked = self.df_all[self.df_all["id"].astype(int).isin(self.marked_assets)].copy()
                for _, row in marked.sort_values("relative_path").iterrows():
                    lines.append(f"[MARKED FILE] {safe_str(row.get('relative_path'))}")
            for rel_path in sorted(self.marked_virtual_assets, key=lambda x: x.lower()):
                lines.append(f"[MARKED PLANNED FILE] {rel_path}")
        return "\n".join(lines)

    def _append_tree_folder(self, lines: list[str], folder_path: str, prefix: str, is_last: bool):
        connector = "└── " if is_last else "├── "
        marker = "⭐ [MARKED] " if folder_path in self.marked_folders else ""
        open_flag = "[open]" if f"folder::{folder_path}" in self.expanded_keys else "[closed]"
        folder_meta = self.virtual_nodes.get(folder_path, {})
        planned_folder = " [CREATED FOLDER]" if folder_meta.get("created_on_disk") == "yes" else (" [PLANNED]" if folder_meta.get("kind") == "folder" else "")
        lines.append(f"{prefix}{connector}{marker}📁 {Path(folder_path).name} ({self.count_folder_assets(folder_path)}) {open_flag}{planned_folder}")

        if f"folder::{folder_path}" not in self.expanded_keys:
            return

        child_prefix = prefix + ("    " if is_last else "│   ")
        direct = self.get_direct_assets(folder_path)
        virtual_direct = self.get_direct_virtual_assets(folder_path)
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
                    fid = int(row.get("id"))
                except Exception:
                    fid = -1
                marker = "⭐ [MARKED] " if fid in self.marked_assets else ""
                rel = safe_str(row.get("relative_path"))
                quality = safe_str(row.get("quality_status")) or "-"
                rows = safe_str(row.get("rows_count")) or "-"
                cols = safe_str(row.get("columns_count")) or "-"
                lines.append(f"{child_prefix}{sub_connector}{marker}📄 {safe_str(row.get('file_name'))} | quality={quality} | rows={rows} | cols={cols} | {rel}")
            elif kind == "virtual_file":
                rel_path = str(obj)
                meta = self.virtual_nodes.get(rel_path, {})
                file_type = meta.get("file_type", Path(rel_path).suffix.replace(".", "") or "file").upper()
                marker = "⭐ [MARKED] " if rel_path in self.marked_virtual_assets else ""
                lines.append(f"{child_prefix}{sub_connector}{marker}📄 {Path(rel_path).name} [PLANNED {file_type}] | rows=0 | cols=0 | {rel_path}")
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
        canvas_w = max(c.winfo_width(), VISUAL_CANVAS_MIN_W)
        y = 26

        if self.df_all.empty and not self.virtual_nodes:
            c.create_text(20, y, text="No catalog data", anchor="w", fill=MUTED, font=FONT_SMALL)
            c.configure(scrollregion=c.bbox("all"))
            return

        c.create_text(18, y, text="DATA CENTER", anchor="w", fill=TEXT, font=FONT_H2)
        c.create_text(canvas_w - 22, y, text=str(len(self.df_all)), anchor="e", fill=BLUE, font=("Consolas", 8, "bold"))
        y += 24
        c.create_text(
            18,
            y,
            text="Folders keep the same tree structure. Asset cards show DATA → ASSET → META.",
            anchor="w",
            fill=MUTED,
            font=FONT_TINY,
        )
        y += 24

        for folder in self.get_child_folders(""):
            y = self._draw_visual_folder_node(folder, level=0, y=y)

        bbox = c.bbox("all")
        if bbox:
            c.configure(scrollregion=(0, 0, max(canvas_w, bbox[2] + 30), bbox[3] + 30))

    def _draw_visual_folder_node(self, folder_path: str, level: int, y: int) -> int:
        c = self.visual_canvas
        key = f"folder::{folder_path}"
        is_open = key in self.expanded_keys
        is_selected = folder_path == self.selected_folder_path
        is_marked = folder_path in self.marked_folders
        name = Path(folder_path).name if folder_path else "ROOT"
        count = self.count_folder_assets(folder_path)
        x = 22 + level * 34
        box_w = max(520, min(900, VISUAL_CANVAS_MIN_W - x - 60))
        box_h = 32
        fill = MARK_FILL if is_marked else (ROW_SELECTED if is_selected else SURFACE)
        outline = MARK_OUTLINE if is_marked else (BLUE if is_selected else BORDER)
        text_color = MARK_OUTLINE if is_marked else (BLUE if is_selected else TEXT)

        if level > 0:
            c.create_line(x - 18, y - 10, x - 18, y + box_h // 2, fill=BORDER, width=1)
            c.create_line(x - 18, y + box_h // 2, x - 5, y + box_h // 2, fill=BORDER, width=1)

        node_id = c.create_rectangle(x, y, x + box_w, y + box_h, fill=fill, outline=outline, width=1)
        toggle_id = c.create_text(x + 10, y + box_h / 2, text="▾" if is_open else "▸", anchor="w", fill=BLUE, font=("Consolas", 8, "bold"))
        folder_label = f"⭐ 📁 {name}" if is_marked else f"📁 {name}"
        label_id = c.create_text(x + 30, y + box_h / 2, text=folder_label, anchor="w", fill=text_color, font=("Consolas", 8, "bold") if level <= 1 else FONT_SMALL)
        count_id = c.create_text(x + box_w - 10, y + box_h / 2, text=str(count), anchor="e", fill=MUTED, font=FONT_TINY)

        for item in (node_id, toggle_id, label_id, count_id):
            c.tag_bind(item, "<Button-1>", lambda _e, k=key, fp=folder_path: self.toggle_folder(k, fp))
            c.tag_bind(item, "<Button-3>", lambda _e, fp=folder_path: self.toggle_folder_mark(fp))
            c.tag_bind(item, "<Enter>", lambda _e: c.configure(cursor="hand2"))
            c.tag_bind(item, "<Leave>", lambda _e: c.configure(cursor=""))

        y += box_h + 8

        if is_open:
            direct = self.get_direct_assets(folder_path)
            max_files = 8
            for i, (_, asset_row) in enumerate(direct.iterrows()):
                if i >= max_files:
                    hidden = len(direct) - max_files
                    if hidden > 0:
                        y = self._draw_visual_more_node(level + 1, y, hidden)
                    break
                y = self._draw_visual_asset_node(asset_row, level + 1, y)

            for virtual_path in self.get_direct_virtual_assets(folder_path):
                y = self._draw_virtual_asset_node(virtual_path, level + 1, y)

            for child in self.get_child_folders(folder_path):
                y = self._draw_visual_folder_node(child, level + 1, y)

        return y

    def _draw_visual_asset_node(self, row: pd.Series, level: int, y: int) -> int:
        c = self.visual_canvas
        try:
            asset_id = int(row.get("id"))
        except Exception:
            return y

        is_selected = self.selected_file_id == asset_id
        is_marked = asset_id in self.marked_assets
        q_color = quality_color(row.get("quality_status"))
        x = 22 + level * 34
        gap = 18
        data_x = x
        asset_x = data_x + DATA_BOX_W + gap
        meta_x = asset_x + ASSET_BOX_W + gap

        data_lines = []
        data_lines += ["STAGE: " + (safe_str(row.get("stage")) or "-")]
        data_lines += ["DOMAIN: " + (safe_str(row.get("domain")) or "-")]
        data_lines += ["TYPE: " + (safe_str(row.get("asset_type")) or "-")]
        data_lines += ["PATH:"] + wrap_text(row.get("relative_path"), 58)

        asset_lines = []
        asset_lines += wrap_text(row.get("file_name"), 54)
        if safe_str(row.get("symbol")):
            asset_lines += ["SYMBOL: " + safe_str(row.get("symbol"))]
        if safe_str(row.get("strategy_id")):
            asset_lines += ["STRATEGY: " + safe_str(row.get("strategy_id"))]
        if safe_str(row.get("account_type")):
            asset_lines += ["ACCOUNT: " + safe_str(row.get("account_type"))]
        if safe_str(row.get("sample_type")):
            asset_lines += ["SAMPLE: " + safe_str(row.get("sample_type"))]

        meta_lines = []
        meta_lines += ["QUALITY: " + (safe_str(row.get("quality_status")) or "-")]
        meta_lines += ["ROWS: " + (safe_str(row.get("rows_count")) or "-")]
        meta_lines += ["COLS: " + (safe_str(row.get("columns_count")) or "-")]
        meta_lines += ["SIZE MB: " + (safe_str(row.get("file_size_mb")) or "-")]
        if safe_str(row.get("date_start")) or safe_str(row.get("date_end")):
            meta_lines += ["DATE:"] + wrap_text(f"{safe_str(row.get('date_start'))} → {safe_str(row.get('date_end'))}", 54)

        line_count = max(len(data_lines), len(asset_lines), len(meta_lines))
        box_h = max(92, 38 + line_count * LINE_H)
        fill = MARK_FILL if is_marked else ("#111111" if is_selected else SURFACE)
        outline = MARK_OUTLINE if is_marked else (BLUE if is_selected else BORDER_2)

        c.create_line(x - 18, y - 9, x - 18, y + box_h // 2, fill=BORDER, width=1)
        c.create_line(x - 18, y + box_h // 2, x - 5, y + box_h // 2, fill=BORDER, width=1)
        c.create_line(data_x + DATA_BOX_W, y + box_h / 2, asset_x, y + box_h / 2, fill=BORDER, width=1, arrow=tk.LAST)
        c.create_line(asset_x + ASSET_BOX_W, y + box_h / 2, meta_x, y + box_h / 2, fill=BORDER, width=1, arrow=tk.LAST)

        items: list[int] = []
        items += self._draw_labeled_box(data_x, y, DATA_BOX_W, box_h, "DATA BOX", BLUE, data_lines, outline, fill)
        items += self._draw_labeled_box(asset_x, y, ASSET_BOX_W, box_h, "★ ASSET BOX" if is_marked else "ASSET BOX", MARK_OUTLINE if is_marked else q_color, asset_lines, outline, fill)
        items += self._draw_labeled_box(meta_x, y, META_BOX_W, box_h, "META / QUALITY BOX", GREEN, meta_lines, outline, fill)

        for item in items:
            c.tag_bind(item, "<Button-1>", lambda _e, fid=asset_id: self.select_asset_by_id(fid))
            c.tag_bind(item, "<Button-3>", lambda _e, fid=asset_id: self.toggle_asset_mark(fid))
            c.tag_bind(item, "<Enter>", lambda _e: c.configure(cursor="hand2"))
            c.tag_bind(item, "<Leave>", lambda _e: c.configure(cursor=""))

        return y + box_h + 14


    def _draw_virtual_asset_node(self, rel_path: str, level: int, y: int) -> int:
        c = self.visual_canvas
        meta = self.virtual_nodes.get(rel_path, {})
        file_type = meta.get("file_type", Path(rel_path).suffix.replace(".", "") or "file").upper()
        is_selected = self.selected_virtual_path == rel_path
        is_marked = rel_path in self.marked_virtual_assets

        x = 22 + level * 34
        gap = 18
        data_x = x
        asset_x = data_x + DATA_BOX_W + gap
        meta_x = asset_x + ASSET_BOX_W + gap

        data_lines = [
            "STAGE: " + self._stage_from_path(rel_path),
            "DOMAIN: PLANNED",
            "TYPE: " + file_type,
            "PATH:",
        ] + wrap_text(rel_path, 44)

        asset_lines = [
            "PLANNED FILE",
            Path(rel_path).name,
            "STATUS: virtual only",
            "PURPOSE: communication / design",
        ]

        meta_lines = [
            "QUALITY: planned",
            "ROWS: 0",
            "COLS: 0",
            "SIZE MB: 0",
            "PHYSICAL FILE: no",
            "CREATED BY: pipeline/code later",
        ]

        line_count = max(len(data_lines), len(asset_lines), len(meta_lines))
        box_h = max(92, 38 + line_count * LINE_H)
        fill = MARK_FILL if is_marked else ("#111111" if is_selected else SURFACE)
        outline = MARK_OUTLINE if is_marked else (BLUE if is_selected else BORDER_2)

        c.create_line(x - 18, y - 9, x - 18, y + box_h // 2, fill=BORDER, width=1)
        c.create_line(x - 18, y + box_h // 2, x - 5, y + box_h // 2, fill=BORDER, width=1)
        c.create_line(data_x + DATA_BOX_W, y + box_h / 2, asset_x, y + box_h / 2, fill=BORDER, width=1, arrow=tk.LAST)
        c.create_line(asset_x + ASSET_BOX_W, y + box_h / 2, meta_x, y + box_h / 2, fill=BORDER, width=1, arrow=tk.LAST)

        items: list[int] = []
        items += self._draw_labeled_box(data_x, y, DATA_BOX_W, box_h, "DATA BOX", BLUE, data_lines, outline, fill)
        items += self._draw_labeled_box(asset_x, y, ASSET_BOX_W, box_h, "★ PLANNED ASSET" if is_marked else "PLANNED ASSET", MARK_OUTLINE if is_marked else PURPLE, asset_lines, outline, fill)
        items += self._draw_labeled_box(meta_x, y, META_BOX_W, box_h, "META / EMPTY FILE", GREEN, meta_lines, outline, fill)

        for item in items:
            c.tag_bind(item, "<Button-1>", lambda _e, p=rel_path: self.select_virtual_asset(p))
            c.tag_bind(item, "<Button-3>", lambda _e, p=rel_path: self.toggle_virtual_asset_mark(p))
            c.tag_bind(item, "<Enter>", lambda _e: c.configure(cursor="hand2"))
            c.tag_bind(item, "<Leave>", lambda _e: c.configure(cursor=""))

        return y + box_h + 14


    def _draw_labeled_box(self, x: int, y: int, w: int, h: int, title: str, color: str, lines: list[str], outline: str, fill: str) -> list[int]:
        c = self.visual_canvas
        ids: list[int] = []
        rect = c.create_rectangle(x, y, x + w, y + h, fill=fill, outline=outline, width=1)
        ids.append(rect)
        band = c.create_rectangle(x, y, x + w, y + 28, fill=fill, outline=outline, width=1)
        ids.append(band)
        dot = c.create_oval(x + 10, y + 10, x + 18, y + 18, fill=color, outline=color)
        ids.append(dot)
        title_id = c.create_text(x + 26, y + 14, text=title, anchor="w", fill=color, font=("Segoe UI", 7, "bold"))
        ids.append(title_id)
        ty = y + 38
        for line in lines:
            txt = c.create_text(x + 12, ty, text=line, anchor="w", fill=TEXT_2, font=FONT_TINY)
            ids.append(txt)
            ty += LINE_H
        return ids

    def _draw_visual_more_node(self, level: int, y: int, hidden: int) -> int:
        c = self.visual_canvas
        x = 22 + level * 34
        c.create_text(x, y + 10, text=f"+ {hidden} more assets", anchor="w", fill=MUTED, font=FONT_TINY)
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
        self._button(actions, "↻  Run Scanner", self.run_scanner).pack(side="left", padx=(0, 12))
        self._button(actions, "📄  Copy Path", self.copy_selected_path).pack(side="left", padx=(0, 12))
        self._button(actions, "📋  Copy Data", self.copy_selected_data_content).pack(side="left", padx=(0, 12))
        self._button(actions, "📂  Open Folder", self.open_folder).pack(side="left")

        self.kpi_area = tk.Frame(self.content, bg=APP_BG)
        self.kpi_area.grid(row=1, column=0, sticky="ew", padx=26, pady=(0, 26))
        for i in range(6):
            self.kpi_area.grid_columnconfigure(i, weight=1)

        self.kpi_total = self._kpi_card(self.kpi_area, 0, "📦", "0", "Assets", "cataloged", BLUE)
        self.kpi_pipeline = self._kpi_card(self.kpi_area, 1, "⚙", "0", "Pipeline", "data flow", PURPLE)
        self.kpi_baseline = self._kpi_card(self.kpi_area, 2, "✓", "0", "Baseline", "validated", GREEN)
        self.kpi_research = self._kpi_card(self.kpi_area, 3, "⌕", "0", "Research", "analysis", YELLOW)
        self.kpi_production = self._kpi_card(self.kpi_area, 4, "◆", "0", "Production", "live assets", BLUE)
        self.kpi_quality = self._kpi_card(self.kpi_area, 5, "✓", "0%", "Quality OK", "passed", GREEN)

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
        self.profile_box = self._info_box(details_grid, 0, "Data Profile")
        self.quality_box = self._info_box(details_grid, 1, "Quality")
        self.columns_box = self._info_box(details_grid, 2, "Columns")

        self.meta_card = RoundedCard(body, bg=SURFACE)
        self.meta_card.grid(row=3, column=0, sticky="ew", pady=(26, 0))
        self._build_meta_bar(self.meta_card.inner)

        tk.Label(body, text="QUANT TERMINAL · Data Catalog v1.2.0", bg=APP_BG, fg=MUTED, font=FONT_TINY).grid(row=4, column=0, sticky="e", pady=(12, 0))

    def _kpi_card(self, parent, col: int, icon: str, value: str, title: str, subtitle: str, color: str):
        card = RoundedCard(parent, bg=SURFACE)
        card.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 7, 0 if col == 5 else 7))
        inner = card.inner
        inner.grid_columnconfigure(1, weight=1)
        tk.Label(inner, text=icon, bg=SURFACE, fg=color, font=("Consolas", 16, "bold"), width=3).grid(row=0, column=0, rowspan=2, padx=(12, 8), pady=18)
        val = tk.Label(inner, text=value, bg=SURFACE, fg=TEXT, font=("Consolas", 14, "bold"))
        val.grid(row=0, column=1, sticky="sw", pady=(16, 0))
        tk.Label(inner, text=title, bg=SURFACE, fg=TEXT, font=("Consolas", 8, "bold")).grid(row=1, column=1, sticky="nw")
        tk.Label(inner, text=subtitle, bg=SURFACE, fg=MUTED, font=FONT_TINY).grid(row=2, column=1, sticky="nw", pady=(0, 14))
        return val

    def _build_table(self, parent):
        frame = tk.Frame(parent, bg=SURFACE)
        frame.pack(fill="both", expand=True)
        sy = tk.Scrollbar(frame, orient="vertical")
        sx = tk.Scrollbar(frame, orient="horizontal")
        self.file_table = ttk.Treeview(
            frame,
            columns=self.FILE_COLUMNS,
            show="headings",
            style="Catalog.Treeview",
            yscrollcommand=sy.set,
            xscrollcommand=sx.set,
        )
        sy.config(command=self.file_table.yview)
        sx.config(command=self.file_table.xview)
        self.file_table.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        widths = {
            "file_name": 260,
            "stage": 100,
            "domain": 130,
            "asset_type": 140,
            "symbol": 80,
            "strategy_id": 110,
            "account_type": 110,
            "sample_type": 90,
            "rows_count": 90,
            "columns_count": 90,
            "quality_status": 110,
            "file_size_mb": 90,
        }
        for col in self.FILE_COLUMNS:
            self.file_table.heading(col, text=self.FILE_LABELS[col])
            self.file_table.column(col, width=widths.get(col, 100), minwidth=70, anchor="w", stretch=True)

        self.file_table.tag_configure("ok", background=SURFACE, foreground=TEXT)
        self.file_table.tag_configure("alt", background=ROW_ALT, foreground=TEXT)
        self.file_table.tag_configure("warn", background="#332000", foreground=TEXT)
        self.file_table.tag_configure("failed", background="#300A0A", foreground=TEXT)
        self.file_table.bind("<<TreeviewSelect>>", self.on_file_select)
        self.file_table.bind("<Double-1>", lambda _e: self.open_file())

    def _build_selected_file(self, parent):
        parent.grid_columnconfigure(1, weight=1)
        tk.Label(parent, text="SELECTED DATA ASSET", bg=SURFACE, fg=BLUE, font=FONT_H2).grid(row=0, column=0, columnspan=4, sticky="w", padx=20, pady=(18, 12))
        tk.Label(parent, text="📦", bg=ROW_SELECTED, fg=BLUE, font=("Consolas", 18, "bold"), width=3).grid(row=1, column=0, rowspan=4, padx=(20, 14), pady=(0, 20))
        self.sel_title = tk.Label(parent, text="NO ASSET SELECTED", bg=SURFACE, fg=TEXT, font=FONT_TITLE)
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
        for i, title in enumerate(["FILE PATH", "Date Range", "Modified UTC", "Scanned UTC"]):
            frame = tk.Frame(parent, bg=SURFACE)
            frame.grid(row=0, column=i, sticky="ew", padx=20, pady=18)
            tk.Label(frame, text=title, bg=SURFACE, fg=MUTED, font=FONT_TINY).pack(anchor="w")
            label = tk.Label(frame, text="-", bg=SURFACE, fg=TEXT, font=FONT_SMALL, wraplength=300, justify="left")
            label.pack(anchor="w")
            self.meta_labels.append(label)

    # --------------------------------------------------------
    # Data/load
    # --------------------------------------------------------

    def reload(self):
        self.df_all = self.repo.load_assets()
        if self.df_all.empty:
            self.update_kpis()
            self.populate_files(pd.DataFrame())
            self.show_empty_details()
            self.draw_visual_folder_tree()
            return
        self.normalize_df()
        self.df_current = self.df_all.copy()
        self.populate_files(self.df_current)
        self.update_kpis()
        self.show_empty_details()
        self.draw_visual_folder_tree()

    def normalize_df(self):
        for col in self.FILE_COLUMNS + ["relative_path", "file_path", "quality_message", "quality_score", "date_start", "date_end", "columns_list"]:
            if col not in self.df_all.columns:
                self.df_all[col] = ""
        self.df_all["folder_parts"] = self.df_all["relative_path"].apply(lambda x: list(Path(safe_str(x)).parts[:-1]))
        self.df_all["folder_path"] = self.df_all["folder_parts"].apply(lambda parts: str(Path(*parts)) if parts else "")

    def update_kpis(self):
        if self.df_all.empty:
            for label in [self.kpi_total, self.kpi_pipeline, self.kpi_baseline, self.kpi_research, self.kpi_production, self.kpi_quality]:
                label.config(text="0")
            self.kpi_quality.config(text="0%")
            return
        total = len(self.df_all)
        stage = self.df_all["stage"].astype(str)
        pipeline = int((stage == "Pipeline").sum())
        baseline = int((stage == "Baseline").sum())
        research = int((stage == "Research").sum())
        production = int((stage == "Production").sum())
        passed = int((self.df_all["quality_status"].astype(str).str.lower().isin(["passed", "ok"])).sum())
        q_rate = 100.0 * passed / total if total else 0.0
        self.kpi_total.config(text=f"{total:,}")
        self.kpi_pipeline.config(text=f"{pipeline:,}")
        self.kpi_baseline.config(text=f"{baseline:,}")
        self.kpi_research.config(text=f"{research:,}")
        self.kpi_production.config(text=f"{production:,}")
        self.kpi_quality.config(text=f"{q_rate:.1f}%")

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

    def count_folder_assets(self, folder_path: str) -> int:
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

    def get_direct_assets(self, folder_path: str) -> pd.DataFrame:
        if self.df_all.empty:
            return pd.DataFrame()
        d = self.df_all[self.df_all["folder_path"].astype(str).str.replace("\\", "/", regex=False) == normalize_rel_path(folder_path)].copy()
        if d.empty:
            return d
        return d.sort_values(["file_name", "relative_path"]).reset_index(drop=True)

    def toggle_folder(self, key: str, folder_path: str):
        if key in self.expanded_keys:
            self.expanded_keys.remove(key)
        else:
            self.expanded_keys.add(key)
        self.select_folder(folder_path)

    def select_folder(self, folder_path: str):
        self.selected_file_id = None
        self.selected_virtual_path = ""
        self.selected_folder_path = folder_path
        prefix = normalize_rel_path(folder_path)
        if not prefix:
            self.df_current = self.df_all.copy()
        else:
            self.df_current = self.df_all[
                self.df_all["relative_path"].astype(str).str.replace("\\", "/", regex=False).str.startswith(prefix + "/")
            ].copy()
        self.breadcrumb_var.set("Dashboard  ›  " + folder_path.replace("\\", "  ›  ").replace("/", "  ›  "))
        self.apply_search()

    def select_asset_by_id(self, file_id: int):
        self.selected_virtual_path = ""
        self.selected_file_id = int(file_id)
        row = self.get_selected_row()
        if row is None:
            return
        self.selected_folder_path = safe_str(row.get("folder_path"))
        self.breadcrumb_var.set("Dashboard  ›  " + safe_str(row.get("relative_path")).replace("\\", "  ›  ").replace("/", "  ›  "))
        self.populate_files(self.df_current if not self.df_current.empty else self.df_all)
        if str(file_id) in self.file_table.get_children(""):
            self.file_table.selection_set(str(file_id))
            self.file_table.focus(str(file_id))
            self.file_table.see(str(file_id))
        self.show_file_details(row)
        self.draw_visual_folder_tree()

    # --------------------------------------------------------
    # Table/search/details
    # --------------------------------------------------------

    def apply_search(self):
        base = self.df_current if not self.df_current.empty else self.df_all
        d = base.copy()
        q = self.search_var.get().strip().lower()
        if q and not d.empty:
            mask = pd.Series(False, index=d.index)
            for col in [
                "file_name", "relative_path", "file_path", "stage", "domain", "asset_type", "symbol",
                "strategy_id", "sample_type", "account_type", "quality_status", "columns_list", "quality_message",
            ]:
                if col in d.columns:
                    mask = mask | d[col].astype(str).str.lower().str.contains(q, na=False)
            d = d[mask]
        self.populate_files(d.reset_index(drop=True))
        self.update_folder_summary(d.reset_index(drop=True))
        self.draw_visual_folder_tree()

    def populate_files(self, df: pd.DataFrame):
        self.file_table.delete(*self.file_table.get_children())
        if df.empty:
            return
        for idx, (_, row) in enumerate(df.iterrows()):
            values = []
            for col in self.FILE_COLUMNS:
                v = safe_str(row.get(col))
                values.append(v)
            q = safe_str(row.get("quality_status")).lower()
            tag = "alt" if idx % 2 else "ok"
            if q in {"warning", "warn"}:
                tag = "warn"
            elif q in {"failed", "fail", "error"}:
                tag = "failed"
            self.file_table.insert("", "end", iid=str(row.get("id")), values=values, tags=(tag,))

    def update_folder_summary(self, df: pd.DataFrame):
        if df.empty:
            self.sel_sub.config(text="No assets in current selection")
            return
        assets = len(df)
        rows_total = 0
        if "rows_count" in df.columns:
            rows_total = pd.to_numeric(df["rows_count"], errors="coerce").fillna(0).sum()
        passed = int((df["quality_status"].astype(str).str.lower().isin(["passed", "ok"])).sum())
        warning = int((df["quality_status"].astype(str).str.lower().isin(["warning", "warn"])).sum())
        failed = int((df["quality_status"].astype(str).str.lower().isin(["failed", "fail", "error"])).sum())
        self.sel_sub.config(text=f"Selection: {assets:,} assets | Rows: {int(rows_total):,} | Passed: {passed} | Warning: {warning} | Failed: {failed}")

    def on_file_select(self, _event=None):
        self.selected_virtual_path = ""
        selected = self.file_table.selection()
        if not selected:
            return
        self.selected_file_id = int(selected[0])
        row = self.get_selected_row()
        if row is not None:
            self.selected_folder_path = safe_str(row.get("folder_path"))
            self.show_file_details(row)
            self.draw_visual_folder_tree()

    def get_selected_row(self) -> Optional[pd.Series]:
        if self.selected_file_id is None or self.df_all.empty:
            return None
        row = self.df_all[pd.to_numeric(self.df_all["id"], errors="coerce").astype("Int64") == int(self.selected_file_id)]
        if row.empty:
            return None
        return row.iloc[0]

    def show_empty_details(self):
        self.sel_title.config(text="NO ASSET SELECTED")
        self.sel_sub.config(text="SELECT AN ASSET FROM DATA MAP OR TABLE.")
        self._set_kv(self.sel_kv, [])
        self._set_kv(self.sel_kv2, [])
        self._set_text(self.profile_box, "")
        self._set_text(self.quality_box, "")
        self._set_text(self.columns_box, "")
        for label in self.meta_labels:
            label.config(text="-")

    def show_file_details(self, row: pd.Series):
        file_name = safe_str(row.get("file_name")) or Path(safe_str(row.get("relative_path"))).name
        folder = Path(safe_str(row.get("relative_path"))).parent
        self.sel_title.config(text=file_name or "Asset")
        self.sel_sub.config(text=safe_str(row.get("relative_path")))

        self._set_kv(self.sel_kv, [
            ("Stage", row.get("stage")),
            ("Domain", row.get("domain")),
            ("Asset Type", row.get("asset_type")),
            ("Symbol", row.get("symbol")),
        ])
        self._set_kv(self.sel_kv2, [
            ("Quality", row.get("quality_status")),
            ("Rows", row.get("rows_count")),
            ("Columns", row.get("columns_count")),
            ("Size MB", row.get("file_size_mb")),
        ])

        profile = [
            f"File Name: {file_name}",
            f"Stage: {safe_str(row.get('stage'))}",
            f"Domain: {safe_str(row.get('domain'))}",
            f"Asset Type: {safe_str(row.get('asset_type'))}",
            f"Symbol: {safe_str(row.get('symbol'))}",
            f"Strategy ID: {safe_str(row.get('strategy_id'))}",
            f"Account Type: {safe_str(row.get('account_type'))}",
            f"Sample Type: {safe_str(row.get('sample_type'))}",
            f"Rows: {safe_str(row.get('rows_count'))}",
            f"Columns: {safe_str(row.get('columns_count'))}",
            f"File Size MB: {safe_str(row.get('file_size_mb'))}",
            f"Date Start: {safe_str(row.get('date_start'))}",
            f"Date End: {safe_str(row.get('date_end'))}",
        ]
        self._set_text(self.profile_box, "\n".join(profile))

        quality = [
            f"Quality Status: {safe_str(row.get('quality_status'))}",
            f"Quality Score: {safe_str(row.get('quality_score'))}",
            f"Quality Message: {safe_str(row.get('quality_message'))}",
            f"Checksum: {safe_str(row.get('checksum'))}",
        ]
        self._set_text(self.quality_box, "\n".join(quality))

        columns_raw = safe_str(row.get("columns_list"))
        columns = [c.strip() for c in columns_raw.split(",") if c.strip()]
        self._set_text(self.columns_box, "\n".join([f"• {c}" for c in columns]) if columns else "• -")

        self.meta_labels[0].config(text=str(folder))
        self.meta_labels[1].config(text=f"{safe_str(row.get('date_start'))} → {safe_str(row.get('date_end'))}")
        self.meta_labels[2].config(text=safe_str(row.get("last_modified_utc")) or "-")
        self.meta_labels[3].config(text=safe_str(row.get("scanned_at_utc")) or "-")

    def _set_kv(self, parent: tk.Frame, pairs: list[tuple[str, object]]):
        for w in parent.winfo_children():
            w.destroy()
        for i, (k, v) in enumerate(pairs):
            tk.Label(parent, text=k, bg=SURFACE, fg=MUTED, font=FONT_TINY).grid(row=i, column=0, sticky="w", padx=(0, 22), pady=4)
            color = quality_color(v) if k == "Quality" else TEXT
            tk.Label(parent, text=safe_str(v) or "-", bg=SURFACE, fg=color, font=("Consolas", 8, "bold") if k in {"Quality", "Rows", "Columns", "Size MB"} else FONT_SMALL).grid(row=i, column=1, sticky="w", pady=4)

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
        return p if str(p) and p.exists() else None

    def copy_to_clipboard(self, text: str):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    def copy_selected_path(self):
        if self.selected_virtual_path:
            self.copy_to_clipboard(str(DATA_DIR / self.selected_virtual_path))
            return
        row = self.get_selected_row()
        if row is None:
            messagebox.showwarning("Keine Datei", "NO ASSET SELECTED.")
            return
        self.copy_to_clipboard(safe_str(row.get("file_path")) or safe_str(row.get("relative_path")))

    def copy_selected_details(self):
        row = self.get_selected_row()
        if row is None:
            messagebox.showwarning("Keine Datei", "NO ASSET SELECTED.")
            return
        self.copy_to_clipboard(self.build_details_text(row))

    def build_details_text(self, row: pd.Series) -> str:
        lines = []

        def add(k, v):
            lines.append(f"{k}: {safe_str(v)}")

        lines.append("ASSET")
        lines.append("-" * 60)
        for k, col in [
            ("File Name", "file_name"), ("Stage", "stage"), ("Domain", "domain"), ("Asset Type", "asset_type"),
            ("Symbol", "symbol"), ("Strategy ID", "strategy_id"), ("Account Type", "account_type"), ("Sample Type", "sample_type"),
        ]:
            add(k, row.get(col, ""))
        lines.append("")
        lines.append("DATA PROFILE")
        lines.append("-" * 60)
        for k, col in [("Rows", "rows_count"), ("Columns", "columns_count"), ("Date Start", "date_start"), ("Date End", "date_end"), ("File Size MB", "file_size_mb")]:
            add(k, row.get(col, ""))
        lines.append("")
        lines.append("QUALITY")
        lines.append("-" * 60)
        for k, col in [("Quality Status", "quality_status"), ("Quality Score", "quality_score"), ("Quality Message", "quality_message")]:
            add(k, row.get(col, ""))
        lines.append("")
        lines.append("LOCATION")
        lines.append("-" * 60)
        for k, col in [("Relative Path", "relative_path"), ("Full Path", "file_path"), ("Last Modified UTC", "last_modified_utc"), ("Scanned UTC", "scanned_at_utc"), ("Checksum", "checksum")]:
            add(k, row.get(col, ""))
        lines.append("")
        lines.append("COLUMNS")
        lines.append("-" * 60)
        columns = [c.strip() for c in safe_str(row.get("columns_list", "")).split(",") if c.strip()]
        lines += [f"- {c}" for c in columns] if columns else ["-"]
        return "\n".join(lines)


    def _safe_relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(QUANT_ROOT)).replace("\\", "/")
        except Exception:
            return str(path).replace("\\", "/")

    def _selected_real_data_path(self) -> Optional[Path]:
        if self.selected_virtual_path:
            p = DATA_DIR / self.selected_virtual_path
            return p if p.exists() and p.is_file() else None

        row = self.get_selected_row()
        if row is None:
            return None

        p = Path(safe_str(row.get("file_path")))
        return p if p.exists() and p.is_file() else None

    def _selected_real_folder_path(self) -> Optional[Path]:
        if self.selected_folder_path:
            p = DATA_DIR / self.selected_folder_path
            if p.exists() and p.is_dir():
                return p

        if self.selected_virtual_path:
            p = (DATA_DIR / self.selected_virtual_path).parent
            return p if p.exists() and p.is_dir() else None

        row = self.get_selected_row()
        if row is not None:
            p = Path(safe_str(row.get("file_path")))
            if p.exists():
                return p.parent if p.is_file() else p

        return None

    def _truncate_for_clipboard(self, payload: str) -> str:
        if len(payload) <= MAX_CLIPBOARD_CHARS:
            return payload
        return (
            payload[:MAX_CLIPBOARD_CHARS]
            + "\n\n# TRUNCATED_FOR_CLIPBOARD\n"
            + f"# ORIGINAL_CHARS_APPROX: {len(payload)}\n"
        )

    def _dataframe_to_copy_text(self, df: pd.DataFrame, source_path: Path, source_kind: str) -> str:
        rel = self._safe_relative(source_path)
        header = (
            f"# FILE: {rel}\n"
            f"# FULL_PATH: {source_path}\n"
            f"# COPY_FOR_CHATGPT\n"
            f"# SOURCE_KIND: {source_kind}\n"
            f"# ROWS: {len(df)}\n"
            f"# COLUMNS: {len(df.columns)}\n"
            f"# COLUMN_NAMES: {', '.join(map(str, df.columns.tolist()))}\n\n"
        )

        csv_text = df.to_csv(index=False)
        return self._truncate_for_clipboard(header + csv_text)

    def _read_data_file_for_chatgpt(self, path: Path) -> str:
        suffix = path.suffix.lower()
        rel = self._safe_relative(path)

        text_suffixes = {".csv", ".txt", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".log", ".py"}

        if suffix in text_suffixes:
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                content = f"[READ_ERROR] {exc}"

            payload = (
                f"# FILE: {rel}\n"
                f"# FULL_PATH: {path}\n"
                f"# COPY_FOR_CHATGPT\n"
                f"# SOURCE_KIND: TEXT_OR_CSV\n\n"
                f"{content}"
            )
            return self._truncate_for_clipboard(payload)

        if suffix == ".parquet":
            try:
                df = pd.read_parquet(path)
                return self._dataframe_to_copy_text(df, path, "PARQUET_CONVERTED_TO_CSV_TEXT")
            except Exception as exc:
                return (
                    f"# FILE: {rel}\n"
                    f"# FULL_PATH: {path}\n"
                    f"# COPY_FOR_CHATGPT\n"
                    f"# SOURCE_KIND: PARQUET\n"
                    f"# READ_ERROR: {exc}\n"
                    "# NOTE: Parquet is binary. Install pyarrow/fastparquet locally or export to CSV for full text copy.\n"
                )

        if suffix in {".xlsx", ".xls"}:
            try:
                df = pd.read_excel(path)
                return self._dataframe_to_copy_text(df, path, "EXCEL_CONVERTED_TO_CSV_TEXT")
            except Exception as exc:
                return (
                    f"# FILE: {rel}\n"
                    f"# FULL_PATH: {path}\n"
                    f"# COPY_FOR_CHATGPT\n"
                    f"# SOURCE_KIND: EXCEL\n"
                    f"# READ_ERROR: {exc}\n"
                )

        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
            payload = (
                f"# FILE: {rel}\n"
                f"# FULL_PATH: {path}\n"
                f"# COPY_FOR_CHATGPT\n"
                f"# SOURCE_KIND: UNKNOWN_TEXT_ATTEMPT\n\n"
                f"{raw}"
            )
            return self._truncate_for_clipboard(payload)
        except Exception:
            return (
                f"# FILE: {rel}\n"
                f"# FULL_PATH: {path}\n"
                f"# COPY_FOR_CHATGPT\n"
                f"# SOURCE_KIND: UNSUPPORTED_BINARY\n"
                "# NOTE: This file type is not directly pasteable as text.\n"
            )

    def copy_selected_data_content(self):
        path = self._selected_real_data_path()
        if path is None:
            messagebox.showwarning("Keine Datei", "Keine echte Daten-Datei ausgewählt.")
            return

        self.copy_to_clipboard(self._read_data_file_for_chatgpt(path))
        messagebox.showinfo("Copied", "Daten-Datei wurde direkt in die Zwischenablage kopiert.")

    def copy_selected_folder_data(self):
        folder = self._selected_real_folder_path()
        if folder is None:
            messagebox.showwarning("Kein Ordner", "Kein echter Daten-Ordner ausgewählt.")
            return

        allowed_suffixes = {
            ".csv", ".parquet", ".txt", ".md", ".json", ".jsonl",
            ".yaml", ".yml", ".toml", ".log", ".xlsx", ".xls"
        }
        excluded_parts = {"__pycache__", ".git", ".venv", "venv", "env"}

        files = [
            p for p in folder.rglob("*")
            if p.is_file()
            and p.suffix.lower() in allowed_suffixes
            and not any(part in excluded_parts for part in p.parts)
        ]
        files = sorted(files, key=lambda x: str(x).lower())[:FOLDER_COPY_MAX_FILES]

        folder_rel = self._safe_relative(folder)
        chunks: list[str] = [
            f"# DATA_FOLDER: {folder_rel}\n",
            f"# FULL_PATH: {folder}\n",
            "# COPY_FOR_CHATGPT\n",
            f"# FILE_COUNT_INCLUDED: {len(files)}\n\n",
            "# TREE\n",
            self.build_expanded_ascii_tree(),
            "\n\n",
        ]

        total_chars = sum(len(x) for x in chunks)

        for p in files:
            block = (
                "\n\n# ============================================================\n"
                f"# FILE_BLOCK: {self._safe_relative(p)}\n"
                "# ============================================================\n\n"
            )
            block += self._read_data_file_for_chatgpt(p)

            if total_chars + len(block) > MAX_CLIPBOARD_CHARS:
                chunks.append("\n\n# FOLDER_COPY_TRUNCATED: clipboard size limit reached")
                break

            chunks.append(block)
            total_chars += len(block)

        self.copy_to_clipboard("".join(chunks))
        messagebox.showinfo("Copied", "Daten-Ordner wurde direkt in die Zwischenablage kopiert.")

    def copy_selected_context_for_chatgpt(self):
        path = self._selected_real_data_path()
        if path is not None:
            self.copy_selected_data_content()
            return

        folder = self._selected_real_folder_path()
        if folder is not None:
            self.copy_selected_folder_data()
            return

        if hasattr(self, "build_expanded_ascii_tree"):
            self.copy_to_clipboard(self.build_expanded_ascii_tree())
            messagebox.showinfo("Copied", "Tree wurde direkt in die Zwischenablage kopiert.")
            return

        messagebox.showwarning("Keine Auswahl", "Keine Datei oder Ordner ausgewählt.")


    def open_file(self):
        p = self.selected_path()
        if p is None:
            messagebox.showwarning("Keine Datei", "Keine gültige Datei ausgewählt.")
            return
        os.startfile(str(p))

    def open_folder(self):
        p = self.selected_path()
        if p is None:
            row = self.get_selected_row()
            if row is None:
                messagebox.showwarning("Kein Ordner", "NO ASSET SELECTED.")
                return
            raw = safe_str(row.get("file_path"))
            folder = Path(raw).parent if raw else QUANT_ROOT
            if folder.exists():
                os.startfile(str(folder))
                return
            messagebox.showwarning("Kein Ordner", "Keine gültige Datei ausgewählt.")
            return
        os.startfile(str(p.parent))

    def run_scanner(self):
        if self.repo.run_scanner():
            self.reload()


# ============================================================
# PANEL API / STANDALONE
# ============================================================


def build_panel(parent, repository: Optional[DataCatalogRepository] = None, **kwargs):
    return DataCatalogDashboardBlock(parent, repository=repository, **kwargs)


def main():
    app = tk.Tk()
    app.title("DATA CATALOG Dashboard")
    app.geometry("2200x1100")
    app.minsize(1600, 850)
    app.configure(bg=APP_BG)
    block = DataCatalogDashboardBlock(app)
    block.pack(fill="both", expand=True)
    print("QUANT_ROOT:", QUANT_ROOT)
    print("CATALOG_DB:", CATALOG_DB)
    print("SCANNER_CODE:", SCANNER_CODE)
    app.mainloop()


if __name__ == "__main__":
    main()

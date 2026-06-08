# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: code_registry_tree_selector
# script_name: Code Registry Tree Selector
# owner: Leon
# status: active
# layer: Dashboard
# domain: Code Registry
# asset_type: Dashboard Building Block
# purpose: Minimal Code Registry tree/actions panel. Shows selectable code tree, actions, output preview, inputs/outputs, and can create real folders/code files inside the project.
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
# version: v1.2.2_show_created_folders_and_files
# last_reviewed: 2026-06-08
# required_api:
#   - build_panel(parent, repository=None, **kwargs)
# ============================================================

from __future__ import annotations

import os
import sys
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import pandas as pd


# ============================================================
# THEME - MAINBOARD VISUAL STYLE
# ============================================================

APP_BG = "#080D12"
SURFACE = "#0E141B"
SURFACE_2 = "#111923"
SURFACE_3 = "#151E29"
SURFACE_HOVER = "#172434"
SURFACE_SELECTED = "#123456"

BORDER = "#26313D"
BORDER_SOFT = "#1B2530"
TEXT = "#F4F7FA"
TEXT_2 = "#B8C2CC"
MUTED = "#768390"

ACCENT = "#55A7FF"
ACCENT_SOFT = "#17375C"
GREEN = "#2FD36B"
YELLOW = "#F1B84B"
RED = "#EF5B5B"

FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_SECTION = ("Segoe UI", 9, "bold")
FONT_BODY = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_MONO = ("Cascadia Mono", 9)

TREE_ROW_H = 26
IO_ROW_H = 18
INDENT = 22
CHECK_SIZE = 15


# ============================================================
# PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()


def find_quant_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur] + list(cur.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p.resolve()
    return cur


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


def norm_path(value: object) -> str:
    return safe_str(value).replace("\\", "/").strip("/")


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
        if item.startswith("•"):
            item = item[1:].strip()
        if item:
            out.append(item)
    return out


def ellipsize(text: object, max_len: int = 72) -> str:
    s = safe_str(text).replace("\n", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def compact_io(value: object, max_items: int = 2, max_len: int = 44) -> str:
    items = split_block(value)
    if not items:
        return "-"
    shown = [ellipsize(x, max_len) for x in items[:max_items]]
    if len(items) > max_items:
        shown.append(f"+{len(items) - max_items}")
    return " | ".join(shown)


def file_icon(name: str) -> str:
    ext = Path(name).suffix.lower()
    if ext == ".py":
        return "⌘"
    if ext == ".md":
        return "▤"
    if ext in {".json", ".yaml", ".yml"}:
        return "▧"
    if ext in {".db", ".sqlite"}:
        return "◉"
    return "□"


def quality_color(value: object) -> str:
    q = safe_str(value).lower().strip()
    if q in {"ok", "passed", "valid", "good"}:
        return GREEN
    if "warn" in q or q in {"partial"}:
        return YELLOW
    if "fail" in q or "missing" in q or "error" in q:
        return RED
    return MUTED


def safe_destroy(widget: Any) -> None:
    try:
        if isinstance(widget, (list, tuple, set)):
            for item in widget:
                safe_destroy(item)
            return
        if widget is not None and hasattr(widget, "destroy"):
            widget.destroy()
    except Exception:
        pass


# ============================================================
# REPOSITORY
# ============================================================


class CodeRegistryTreeRepository:
    def __init__(self, db_path: Path = REGISTRY_DB):
        self.db_path = Path(db_path)

    def load_assets(self) -> pd.DataFrame:
        """
        Load code assets from code_registry.db and merge with real filesystem .py files.

        Important:
        - Newly created code files should appear immediately even if the scanner did not update the DB yet.
        - Empty folders are handled later in the UI index builder.
        """
        fs_df = self._fallback_from_filesystem()

        if self.db_path.exists():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    db_df = pd.read_sql_query("SELECT * FROM code_assets ORDER BY relative_path", conn)

                db_clean = self._clean(db_df)

                if db_clean.empty:
                    return fs_df

                if fs_df.empty:
                    return db_clean

                merged = pd.concat([db_clean, fs_df], ignore_index=True)
                merged["relative_path"] = merged["relative_path"].fillna("").astype(str).map(norm_path)

                # Keep DB metadata where available, but add filesystem-only new files immediately.
                merged["_source_priority"] = merged["has_registry"].astype(str).map(lambda x: 0 if x not in {"0", "", "nan"} else 1)
                merged = merged.sort_values(["relative_path", "_source_priority"])
                merged = merged.drop_duplicates(subset=["relative_path"], keep="first")
                merged = merged.drop(columns=["_source_priority"], errors="ignore")
                return self._clean(merged)

            except Exception as exc:
                messagebox.showerror("Code Registry DB Error", str(exc))

        return fs_df

    def _fallback_from_filesystem(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        roots = [QUANT_ROOT / "Dashboard", QUANT_ROOT / "Data_Center"]
        idx = 1
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*.py"):
                try:
                    rel = p.relative_to(QUANT_ROOT).as_posix()
                except Exception:
                    rel = p.as_posix()
                rows.append({
                    "id": idx,
                    "file_path": str(p),
                    "relative_path": rel,
                    "file_name": p.name,
                    "script_name": p.stem,
                    "script_id": p.stem,
                    "layer": rel.split("/")[0] if "/" in rel else "",
                    "domain": "",
                    "asset_type": "Python Script",
                    "status": "unknown",
                    "version": "",
                    "has_registry": "0",
                    "registry_quality_status": "unknown",
                    "purpose": "",
                    "inputs": "",
                    "outputs": "",
                    "dependencies": "",
                    "scanned_at_utc": "",
                    "last_modified_utc": "",
                })
                idx += 1
        return self._clean(pd.DataFrame(rows))

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        d = df.copy()
        required = [
            "id", "file_path", "relative_path", "file_name", "script_name", "script_id",
            "layer", "domain", "asset_type", "status", "version", "has_registry",
            "registry_quality_status", "purpose", "inputs", "outputs", "dependencies",
            "scanned_at_utc", "last_modified_utc",
        ]
        for col in required:
            if col not in d.columns:
                d[col] = ""

        d["file_path"] = d["file_path"].fillna("").astype(str)
        d["relative_path"] = d["relative_path"].fillna("").astype(str).map(norm_path)
        d["file_name"] = d["file_name"].fillna("").astype(str)
        d.loc[d["file_name"].str.strip() == "", "file_name"] = d["relative_path"].map(lambda x: Path(x).name)

        def exists_or_keep(path_text: str) -> bool:
            if not path_text:
                return True
            try:
                return Path(path_text).exists()
            except Exception:
                return True

        d = d[d["file_path"].map(exists_or_keep)].copy()
        if d.empty:
            return d

        for col in required:
            d[col] = d[col].fillna("").astype(str)

        d["_id_num"] = pd.to_numeric(d["id"], errors="coerce").fillna(0).astype(int)
        d = d.sort_values(["relative_path", "_id_num"])
        d = d.drop_duplicates(subset=["relative_path"], keep="last").reset_index(drop=True)
        d["id"] = range(1, len(d) + 1)
        d["folder_path"] = d["relative_path"].map(lambda x: str(Path(x).parent).replace("\\", "/") if "/" in x else "")
        d["folder_path"] = d["folder_path"].replace(".", "")
        d["folder_parts"] = d["folder_path"].map(lambda x: [p for p in norm_path(x).split("/") if p])
        return d

    def run_scanner(self) -> bool:
        if not SCANNER_CODE.exists():
            messagebox.showerror("Scanner not found", str(SCANNER_CODE))
            return False
        try:
            subprocess.run([sys.executable, str(SCANNER_CODE)], cwd=str(QUANT_ROOT), check=True)
            return True
        except Exception as exc:
            messagebox.showerror("Scanner Error", str(exc))
            return False

    def load_real_folders(self) -> list[str]:
        """
        Return real project folders relative to QUANT_ROOT.
        This makes newly created empty folders visible in the tree immediately.
        """
        roots = [QUANT_ROOT / "Dashboard", QUANT_ROOT / "Data_Center", QUANT_ROOT / "System_Info", QUANT_ROOT / "tools"]
        skip_names = {"__pycache__", ".git", ".venv", "venv", "env", "build", "dist", ".idea", ".vscode"}

        folders: set[str] = set()

        for root in roots:
            if not root.exists() or not root.is_dir():
                continue

            try:
                folders.add(root.relative_to(QUANT_ROOT).as_posix())
            except Exception:
                pass

            for p in root.rglob("*"):
                if not p.is_dir():
                    continue
                if any(part in skip_names for part in p.parts):
                    continue
                try:
                    rel = p.relative_to(QUANT_ROOT).as_posix()
                    folders.add(norm_path(rel))
                except Exception:
                    pass

        return sorted(folders, key=lambda x: (len(x.split("/")), x.lower()))


# ============================================================
# UI PANEL
# ============================================================


class CodeRegistryTreeSelector(tk.Frame):
    def __init__(self, parent, repository: Optional[CodeRegistryTreeRepository] = None, **kwargs):
        super().__init__(parent, bg=APP_BG, **kwargs)
        self.repo = repository or CodeRegistryTreeRepository()
        self.df_all = pd.DataFrame()
        self.df_current = pd.DataFrame()

        self.search_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self.selected_folder = ""
        self.selected_script_id: Optional[int] = None
        self.expanded_folders: set[str] = set()
        self.marked_folders: set[str] = set()
        self.marked_scripts: set[int] = set()

        self._folders_cache: list[str] = []
        self._children_cache: dict[str, list[str]] = {"": []}
        self._direct_scripts_cache: dict[str, pd.DataFrame] = {}
        self._subtree_count_cache: dict[str, int] = {}
        self._tree_y_to_item: list[tuple[int, int, str, str, Optional[int]]] = []

        self._build_ui()
        self.reload()
        self._bind_close_keys()

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_search()
        self._build_tree()
        self._build_selection_bar()
        self._build_actions()
        self._build_output()
        self._build_footer()

    def _panel(self, parent, bg=SURFACE, border=BORDER_SOFT) -> tk.Frame:
        return tk.Frame(parent, bg=bg, highlightbackground=border, highlightthickness=1)

    def _build_header(self):
        header = tk.Frame(self, bg=APP_BG)
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)

        tk.Label(header, text="CODE REGISTRY", bg=APP_BG, fg=TEXT, font=FONT_TITLE).grid(row=0, column=0, sticky="w")

        right = tk.Frame(header, bg=APP_BG)
        right.grid(row=0, column=1, sticky="e")
        self._small_button(right, "?", self.show_help).pack(side="left", padx=(0, 8))
        self._small_button(right, "⋮", self.show_menu).pack(side="left")

    def _build_search(self):
        wrap = tk.Frame(self, bg=APP_BG)
        wrap.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 14))
        wrap.grid_columnconfigure(0, weight=1)

        search = tk.Frame(wrap, bg=SURFACE, highlightbackground=BORDER, highlightthickness=1)
        search.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        search.grid_columnconfigure(1, weight=1)
        tk.Label(search, text="⌕", bg=SURFACE, fg=TEXT_2, font=FONT_BODY).grid(row=0, column=0, padx=(14, 8), pady=10)
        entry = tk.Entry(search, textvariable=self.search_var, bg=SURFACE, fg=TEXT, insertbackground=TEXT, relief="flat", font=FONT_BODY)
        entry.grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=10)
        entry.bind("<KeyRelease>", lambda _e: self.apply_search())

        self._button(wrap, "Filter", self.apply_search).grid(row=0, column=1, sticky="e")

    def _build_tree(self):
        tree_panel = self._panel(self)
        tree_panel.grid(row=2, column=0, sticky="nsew", padx=22, pady=(0, 14))
        tree_panel.grid_columnconfigure(0, weight=1)
        tree_panel.grid_rowconfigure(0, weight=1)

        self.tree_canvas = tk.Canvas(tree_panel, bg=SURFACE, bd=0, highlightthickness=0)
        self.tree_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.tree_scroll = tk.Scrollbar(tree_panel, orient="vertical", command=self.tree_canvas.yview)
        self.tree_scroll.grid(row=0, column=1, sticky="ns")
        self.tree_canvas.configure(yscrollcommand=self.tree_scroll.set)

        self.tree_canvas.bind("<Configure>", lambda _e: self.draw_tree())
        self.tree_canvas.bind("<MouseWheel>", self._tree_mousewheel)
        self.tree_canvas.bind("<Button-3>", self.show_tree_context_menu)

    def _build_selection_bar(self):
        bar = self._panel(self, bg=SURFACE)
        bar.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 14))
        bar.grid_columnconfigure(0, weight=1)
        self.selected_label = tk.Label(bar, text="Selected: 0 Items", bg=SURFACE, fg=TEXT_2, font=FONT_BODY)
        self.selected_label.grid(row=0, column=0, sticky="w", padx=18, pady=13)
        self._button(bar, "Clear Selection  ×", self.clear_selection).grid(row=0, column=1, sticky="e", padx=10, pady=8)

    def _build_actions(self):
        outer = tk.Frame(self, bg=APP_BG)
        outer.grid(row=4, column=0, sticky="ew", padx=22, pady=(0, 14))
        for i in range(3):
            outer.grid_columnconfigure(i, weight=1)

        tk.Label(outer, text="ACTIONS", bg=APP_BG, fg=TEXT_2, font=FONT_SECTION).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 8))

        actions = [
            ("Copy Tree", self.copy_tree),
            ("Copy Selected", self.copy_selected),
            ("Copy Branch", self.copy_branch),
            ("Copy Path", self.copy_path),
            ("Copy Relative Path", self.copy_relative_path),
            ("Copy File Content", self.copy_file_content),
            ("Copy IO", self.copy_io),
            ("New Folder", self.new_folder),
            ("New Code", self.new_code),
            ("Export Tree", self.export_tree),
            ("Select All", self.select_all),
            ("Select Folder", self.select_current_folder),
            ("Expand All", self.expand_all),
            ("Collapse All", self.collapse_all),
            ("Refresh", self.reload),
        ]
        for idx, (text, cmd) in enumerate(actions):
            r = 1 + idx // 3
            c = idx % 3
            self._button(outer, text, cmd).grid(row=r, column=c, sticky="ew", padx=(0 if c == 0 else 6, 0 if c == 2 else 6), pady=4)

    def _build_output(self):
        panel = self._panel(self, bg=SURFACE)
        panel.grid(row=5, column=0, sticky="nsew", padx=22, pady=(0, 14))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        head = tk.Frame(panel, bg=SURFACE)
        head.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        head.grid_columnconfigure(0, weight=1)
        tk.Label(head, text="TREE OUTPUT  (will be copied)", bg=SURFACE, fg=TEXT_2, font=FONT_SECTION).grid(row=0, column=0, sticky="w")
        self._button(head, "Copy to Clipboard", self.copy_output_to_clipboard).grid(row=0, column=1, sticky="e")

        self.output = tk.Text(
            panel,
            bg="#090F15",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            bd=0,
            height=8,
            wrap="none",
            font=FONT_MONO,
        )
        self.output.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self.output.configure(state="disabled")

    def _build_footer(self):
        footer = tk.Frame(self, bg=APP_BG)
        footer.grid(row=6, column=0, sticky="ew", padx=22, pady=(0, 18))
        footer.grid_columnconfigure(0, weight=1)
        tk.Label(
            footer,
            text="Tip: Checkbox = Mark/Select   |   Double-click folder = Open/Close",
            bg=APP_BG,
            fg=TEXT_2,
            font=FONT_SMALL,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(footer, textvariable=self.status_var, bg=APP_BG, fg=GREEN, font=FONT_SMALL).grid(row=0, column=1, sticky="e")

    def _button(self, parent, text: str, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=SURFACE_2,
            fg=TEXT,
            activebackground=SURFACE_HOVER,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=9,
            cursor="hand2",
            font=FONT_BODY,
            highlightthickness=1,
            highlightbackground=BORDER_SOFT,
        )

    def _small_button(self, parent, text: str, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=SURFACE_2,
            fg=TEXT,
            activebackground=SURFACE_HOVER,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            width=3,
            height=1,
            cursor="hand2",
            font=FONT_BODY,
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    def reload(self):
        self.df_all = self.repo.load_assets()
        self.df_current = self.df_all.copy()
        self.selected_folder = ""
        self.selected_script_id = None
        self.marked_folders.clear()
        self.marked_scripts.clear()
        self.rebuild_indexes()

        # Root always visible/open for this tree tool.
        self.expanded_folders = set(self.child_folders(""))
        self.apply_search(redraw=False)
        self.draw_tree()
        self.update_selection_state()
        self.status_var.set(f"Loaded {len(self.df_all):,} scripts")

    def rebuild_indexes(self):
        self._folders_cache = []
        self._children_cache = {"": []}
        self._direct_scripts_cache = {}
        self._subtree_count_cache = {"": len(self.df_all) if not self.df_all.empty else 0}

        folder_set: set[str] = set()

        # 1) Add all real filesystem folders so empty newly created folders are visible immediately.
        try:
            for folder in self.repo.load_real_folders():
                folder = norm_path(folder)
                if not folder:
                    continue
                parts = [p for p in folder.split("/") if p]
                for i in range(1, len(parts) + 1):
                    folder_set.add("/".join(parts[:i]))
        except Exception:
            pass

        if not self.df_all.empty:
            # 2) Add folders from registered / filesystem Python files.
            for folder, group in self.df_all.groupby(self.df_all["folder_path"].astype(str).map(norm_path), dropna=False):
                folder = norm_path(folder)
                self._direct_scripts_cache[folder] = group.copy()
                if folder:
                    parts = [p for p in folder.split("/") if p]
                    for i in range(1, len(parts) + 1):
                        folder_set.add("/".join(parts[:i]))

            # 3) Count scripts in every subtree.
            for rel in self.df_all["relative_path"].astype(str).map(norm_path):
                parts = [p for p in rel.split("/")[:-1] if p]
                for i in range(1, len(parts) + 1):
                    f = "/".join(parts[:i])
                    self._subtree_count_cache[f] = self._subtree_count_cache.get(f, 0) + 1

        # 4) Build child index from all folders, including empty folders.
        self._folders_cache = sorted(folder_set, key=lambda x: (len(x.split("/")), x.lower()))
        self._children_cache = {f: [] for f in [""] + self._folders_cache}

        for folder in self._folders_cache:
            parent = "/".join(folder.split("/")[:-1])
            self._children_cache.setdefault(parent, []).append(folder)

        for parent in list(self._children_cache.keys()):
            self._children_cache[parent] = sorted(set(self._children_cache[parent]), key=lambda x: x.lower())

    def apply_search(self, redraw: bool = True):
        q = self.search_var.get().strip().lower()
        d = self.df_all.copy()
        if q and not d.empty:
            mask = pd.Series(False, index=d.index)
            for col in ["file_name", "relative_path", "script_name", "script_id", "layer", "domain", "asset_type", "status", "purpose"]:
                if col in d.columns:
                    mask = mask | d[col].astype(str).str.lower().str.contains(q, na=False)
            d = d[mask].copy()
            for fp in d["folder_path"].astype(str).map(norm_path).head(300):
                parts = [p for p in fp.split("/") if p]
                for i in range(1, len(parts) + 1):
                    self.expanded_folders.add("/".join(parts[:i]))
        self.df_current = d
        if redraw:
            self.draw_tree()
            self.update_output()

    def child_folders(self, folder: str) -> list[str]:
        return self._children_cache.get(norm_path(folder), [])

    def direct_scripts(self, folder: str) -> pd.DataFrame:
        return self._direct_scripts_cache.get(norm_path(folder), pd.DataFrame()).copy()

    def subtree_count(self, folder: str) -> int:
        return int(self._subtree_count_cache.get(norm_path(folder), 0))

    # --------------------------------------------------------
    # Tree drawing
    # --------------------------------------------------------

    def draw_tree(self):
        c = self.tree_canvas
        c.delete("all")
        self._tree_y_to_item.clear()
        y = 10
        y = self._draw_folder_row("", "Code", y, level=0, root=True)
        for folder in self.child_folders(""):
            y = self._draw_folder(folder, y, level=1)
        c.configure(scrollregion=(0, 0, max(c.winfo_width(), 500), y + 20))

    def _draw_folder(self, folder: str, y: int, level: int) -> int:
        name = folder.split("/")[-1]
        y = self._draw_folder_row(folder, name, y, level=level)
        if folder in self.expanded_folders:
            for child in self.child_folders(folder):
                y = self._draw_folder(child, y, level + 1)
            direct = self.direct_scripts(folder)
            for _, row in direct.head(300).iterrows():
                y = self._draw_file_row(row, y, level + 1)
            hidden = len(direct) - 300
            if hidden > 0:
                c = self.tree_canvas
                x = 44 + (level + 1) * INDENT
                c.create_text(x, y + TREE_ROW_H / 2, text=f"+ {hidden:,} more scripts", anchor="w", fill=MUTED, font=FONT_SMALL)
                y += TREE_ROW_H
        return y

    def _draw_folder_row(self, folder: str, name: str, y: int, level: int, root: bool = False) -> int:
        c = self.tree_canvas
        w = max(c.winfo_width() - 20, 480)
        h = TREE_ROW_H
        x = 26 + level * INDENT
        selected = self.selected_script_id is None and self.selected_folder == folder
        checked = folder in self.marked_folders
        has_children = bool(self.child_folders(folder) or len(self.direct_scripts(folder)) > 0 or root)

        if selected:
            c.create_rectangle(8, y, w, y + h, fill=SURFACE_SELECTED, outline="")
        elif y // h % 2 == 0:
            c.create_rectangle(8, y, w, y + h, fill=SURFACE, outline="")

        self._draw_checkbox(c, x, y + (h - CHECK_SIZE) / 2, checked)
        folder_color = ACCENT if checked or selected else TEXT_2
        c.create_text(x + 28, y + h / 2, text="▰" if root else "▱", anchor="w", fill=folder_color, font=FONT_SMALL)
        c.create_text(x + 52, y + h / 2, text=name, anchor="w", fill=TEXT, font=FONT_BODY)
        c.create_text(w - 16, y + h / 2, text=f"{self.subtree_count(folder):,}", anchor="e", fill=MUTED, font=FONT_SMALL)

        self._bind_area(8, y, w, y + h, kind="folder", folder=folder, file_id=None)
        self._tree_y_to_item.append((y, y + h, "folder", folder, None))
        return y + h + 1

    def _draw_file_row(self, row: pd.Series, y: int, level: int) -> int:
        c = self.tree_canvas
        w = max(c.winfo_width() - 20, 480)
        h = TREE_ROW_H
        fid = int(row.get("id"))
        x = 26 + level * INDENT
        selected = self.selected_script_id == fid
        checked = fid in self.marked_scripts

        input_text = compact_io(row.get("inputs"))
        output_text = compact_io(row.get("outputs"))
        has_io = input_text != "-" or output_text != "-"
        total_h = h + (IO_ROW_H * 2 if has_io else 0)

        if selected:
            c.create_rectangle(8, y, w, y + total_h, fill=SURFACE_SELECTED, outline="")
        elif y // h % 2 == 0:
            c.create_rectangle(8, y, w, y + total_h, fill=SURFACE, outline="")

        self._draw_checkbox(c, x, y + (h - CHECK_SIZE) / 2, checked)
        name = safe_str(row.get("file_name"))
        qc = quality_color(row.get("registry_quality_status"))
        c.create_text(x + 28, y + h / 2, text=file_icon(name), anchor="w", fill=qc, font=FONT_SMALL)
        c.create_text(x + 52, y + h / 2, text=name, anchor="w", fill=TEXT, font=FONT_BODY)

        if has_io:
            io_x = x + 52
            iy = y + h + 1
            c.create_text(io_x, iy + IO_ROW_H / 2, text="IN", anchor="w", fill=ACCENT, font=("Segoe UI", 7, "bold"))
            c.create_text(io_x + 28, iy + IO_ROW_H / 2, text=input_text, anchor="w", fill=TEXT_2, font=FONT_SMALL)
            oy = iy + IO_ROW_H
            c.create_text(io_x, oy + IO_ROW_H / 2, text="OUT", anchor="w", fill=GREEN, font=("Segoe UI", 7, "bold"))
            c.create_text(io_x + 28, oy + IO_ROW_H / 2, text=output_text, anchor="w", fill=TEXT_2, font=FONT_SMALL)

        self._bind_area(8, y, w, y + total_h, kind="file", folder=norm_path(row.get("folder_path")), file_id=fid)
        self._tree_y_to_item.append((y, y + total_h, "file", norm_path(row.get("folder_path")), fid))
        return y + total_h + 1

    def _draw_checkbox(self, c: tk.Canvas, x: float, y: float, checked: bool):
        fill = ACCENT if checked else SURFACE
        outline = ACCENT if checked else MUTED
        c.create_rectangle(x, y, x + CHECK_SIZE, y + CHECK_SIZE, fill=fill, outline=outline, width=1)
        if checked:
            c.create_text(x + CHECK_SIZE / 2, y + CHECK_SIZE / 2, text="✓", fill="#03111F", font=("Segoe UI", 9, "bold"))

    def _bind_area(self, x1: int, y1: int, x2: int, y2: int, kind: str, folder: str, file_id: Optional[int]):
        c = self.tree_canvas
        hit = c.create_rectangle(x1, y1, x2, y2, fill="", outline="", width=0)
        if kind == "folder":
            c.tag_bind(hit, "<Button-1>", lambda e, f=folder: self._folder_click(e, f))
            c.tag_bind(hit, "<Double-1>", lambda e, f=folder: self.toggle_folder(f))
        else:
            c.tag_bind(hit, "<Button-1>", lambda e, fid=file_id: self._file_click(e, fid))
            c.tag_bind(hit, "<Double-1>", lambda e, fid=file_id: self.open_file(fid))
        c.tag_bind(hit, "<Button-3>", self.show_tree_context_menu)
        c.tag_bind(hit, "<Enter>", lambda _e: c.configure(cursor="hand2"))
        c.tag_bind(hit, "<Leave>", lambda _e: c.configure(cursor=""))

    def _folder_click(self, event, folder: str):
        x = self.tree_canvas.canvasx(event.x)
        checkbox_x = 26 + (0 if folder == "" else len(folder.split("/"))) * INDENT
        if checkbox_x - 3 <= x <= checkbox_x + CHECK_SIZE + 3:
            self.toggle_folder_mark(folder)
        else:
            self.select_folder(folder)

    def _file_click(self, event, file_id: Optional[int]):
        if file_id is None:
            return
        row = self.row_by_id(file_id)
        if row is None:
            return
        folder = norm_path(row.get("folder_path"))
        level = len([p for p in folder.split("/") if p]) + 1
        checkbox_x = 26 + level * INDENT
        x = self.tree_canvas.canvasx(event.x)
        if checkbox_x - 3 <= x <= checkbox_x + CHECK_SIZE + 3:
            self.toggle_file_mark(file_id)
        else:
            self.select_file(file_id)

    # --------------------------------------------------------
    # Selection/mark logic
    # --------------------------------------------------------

    def select_folder(self, folder: str):
        self.selected_folder = norm_path(folder)
        self.selected_script_id = None
        self.status_var.set(f"Selected folder: {self.selected_folder or 'Code'}")
        self.draw_tree()

    def select_file(self, file_id: int):
        self.selected_script_id = int(file_id)
        row = self.row_by_id(file_id)
        self.selected_folder = norm_path(row.get("folder_path")) if row is not None else ""
        self.status_var.set(f"Selected file: {safe_str(row.get('relative_path')) if row is not None else file_id}")
        self.draw_tree()

    def toggle_folder(self, folder: str):
        folder = norm_path(folder)
        if folder == "":
            return
        if folder in self.expanded_folders:
            self.expanded_folders.remove(folder)
        else:
            self.expanded_folders.add(folder)
        self.select_folder(folder)

    def toggle_folder_mark(self, folder: str):
        folder = norm_path(folder)
        if folder in self.marked_folders:
            self._unmark_subtree(folder)
        else:
            self._mark_subtree(folder)
            self._expand_to_folder(folder)
            self.expanded_folders.add(folder)
        self.selected_folder = folder
        self.selected_script_id = None
        self.update_selection_state()
        self.draw_tree()
        self.update_output()

    def toggle_file_mark(self, file_id: int):
        file_id = int(file_id)
        if file_id in self.marked_scripts:
            self.marked_scripts.remove(file_id)
        else:
            self.marked_scripts.add(file_id)
        self.selected_script_id = file_id
        self.update_selection_state()
        self.draw_tree()
        self.update_output()

    def _mark_subtree(self, folder: str):
        folder = norm_path(folder)
        self.marked_folders.add(folder)
        for child in self.child_folders(folder):
            self._mark_subtree(child)
        for _, row in self.direct_scripts(folder).iterrows():
            self.marked_scripts.add(int(row.get("id")))

    def _unmark_subtree(self, folder: str):
        folder = norm_path(folder)
        self.marked_folders.discard(folder)
        for child in self.child_folders(folder):
            self._unmark_subtree(child)
        for _, row in self.direct_scripts(folder).iterrows():
            self.marked_scripts.discard(int(row.get("id")))

    def _expand_to_folder(self, folder: str):
        parts = [p for p in norm_path(folder).split("/") if p]
        for i in range(1, len(parts) + 1):
            self.expanded_folders.add("/".join(parts[:i]))

    def update_selection_state(self):
        total = len(self.marked_folders) + len(self.marked_scripts)
        self.selected_label.config(text=f"Selected: {total:,} Items")

    def clear_selection(self):
        self.marked_folders.clear()
        self.marked_scripts.clear()
        self.update_selection_state()
        self.draw_tree()
        self.update_output()

    def select_all(self):
        for folder in self._folders_cache:
            self.marked_folders.add(folder)
        if not self.df_all.empty:
            self.marked_scripts = set(pd.to_numeric(self.df_all["id"], errors="coerce").dropna().astype(int).tolist())
        self.expanded_folders = set(self._folders_cache)
        self.update_selection_state()
        self.draw_tree()
        self.update_output()

    def select_current_folder(self):
        self.toggle_folder_mark(self.selected_folder)

    def expand_all(self):
        self.expanded_folders = set(self._folders_cache)
        self.draw_tree()

    def collapse_all(self):
        self.expanded_folders = set(self.child_folders(""))
        self.draw_tree()

    # --------------------------------------------------------
    # Output/copy/export
    # --------------------------------------------------------

    def update_output(self):
        text = self.format_selected_tree() if (self.marked_folders or self.marked_scripts) else self.format_tree(max_depth=2)
        self._set_output(text)

    def _set_output(self, text: str):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("end", text)
        self.output.configure(state="disabled")

    def format_tree(self, root: str = "", max_depth: Optional[int] = None) -> str:
        lines = ["Code"]

        def walk(folder: str, prefix: str, depth: int):
            if max_depth is not None and depth > max_depth:
                return
            entries: list[tuple[str, Any]] = []
            for child in self.child_folders(folder):
                entries.append(("folder", child))
            for _, row in self.direct_scripts(folder).iterrows():
                entries.append(("file", row))
            for i, (kind, obj) in enumerate(entries):
                last = i == len(entries) - 1
                con = "└─ " if last else "├─ "
                next_prefix = prefix + ("   " if last else "│  ")
                if kind == "folder":
                    name = str(obj).split("/")[-1]
                    lines.append(f"{prefix}{con}{name}")
                    walk(str(obj), next_prefix, depth + 1)
                else:
                    inputs = split_block(obj.get("inputs"))
                    outputs = split_block(obj.get("outputs"))
                    lines.append(f"{prefix}{con}{safe_str(obj.get('file_name'))} | inputs={len(inputs)} | outputs={len(outputs)}")
                    io_prefix = next_prefix
                    if inputs:
                        lines.append(f"{io_prefix}├─ INPUTS")
                        for j, item in enumerate(inputs[:8]):
                            icon = "└─ " if j == len(inputs[:8]) - 1 and not outputs else "├─ "
                            lines.append(f"{io_prefix}│  {icon}{item}")
                    if outputs:
                        lines.append(f"{io_prefix}└─ OUTPUTS")
                        for j, item in enumerate(outputs[:8]):
                            icon = "└─ " if j == len(outputs[:8]) - 1 else "├─ "
                            lines.append(f"{io_prefix}   {icon}{item}")

        walk(root, "", 0)
        return "\n".join(lines)

    def format_selected_tree(self) -> str:
        selected_files = set(self.marked_scripts)
        selected_folders = set(self.marked_folders)
        lines = ["Code"]

        def subtree_has_selected(folder: str) -> bool:
            if folder in selected_folders:
                return True
            for _, row in self.direct_scripts(folder).iterrows():
                if int(row.get("id")) in selected_files:
                    return True
            return any(subtree_has_selected(child) for child in self.child_folders(folder))

        def walk(folder: str, prefix: str):
            entries: list[tuple[str, Any]] = []
            for child in self.child_folders(folder):
                if subtree_has_selected(child):
                    entries.append(("folder", child))
            for _, row in self.direct_scripts(folder).iterrows():
                if int(row.get("id")) in selected_files:
                    entries.append(("file", row))
            for i, (kind, obj) in enumerate(entries):
                last = i == len(entries) - 1
                con = "└─ " if last else "├─ "
                next_prefix = prefix + ("   " if last else "│  ")
                if kind == "folder":
                    name = str(obj).split("/")[-1]
                    lines.append(f"{prefix}{con}{name}")
                    walk(str(obj), next_prefix)
                else:
                    inputs = split_block(obj.get("inputs"))
                    outputs = split_block(obj.get("outputs"))
                    lines.append(f"{prefix}{con}{safe_str(obj.get('file_name'))} | inputs={len(inputs)} | outputs={len(outputs)}")
                    io_prefix = next_prefix
                    if inputs:
                        lines.append(f"{io_prefix}├─ INPUTS")
                        for j, item in enumerate(inputs[:8]):
                            icon = "└─ " if j == len(inputs[:8]) - 1 and not outputs else "├─ "
                            lines.append(f"{io_prefix}│  {icon}{item}")
                    if outputs:
                        lines.append(f"{io_prefix}└─ OUTPUTS")
                        for j, item in enumerate(outputs[:8]):
                            icon = "└─ " if j == len(outputs[:8]) - 1 else "├─ "
                            lines.append(f"{io_prefix}   {icon}{item}")

        walk("", "")
        return "\n".join(lines)

    def copy_to_clipboard(self, text: str, status: str = "Copied"):
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self.status_var.set(status)

    def copy_output_to_clipboard(self):
        self.copy_to_clipboard(self.output.get("1.0", "end").strip(), "Copied output")

    def copy_tree(self):
        text = self.format_tree()
        self._set_output(text)
        self.copy_to_clipboard(text, "Copied full tree")

    def copy_selected(self):
        text = self.format_selected_tree()
        self._set_output(text)
        self.copy_to_clipboard(text, "Copied selected tree")

    def copy_branch(self):
        root = self.selected_folder or ""
        text = self.format_tree(root=root)
        self._set_output(text)
        self.copy_to_clipboard(text, "Copied branch")

    def copy_path(self):
        p = self.selected_path()
        if p is None:
            messagebox.showwarning("No selection", "Select a folder or script first.")
            return
        self.copy_to_clipboard(str(p), "Copied path")

    def copy_relative_path(self):
        row = self.selected_row()
        if row is not None:
            rel = safe_str(row.get("relative_path"))
        else:
            rel = self.selected_folder
        self.copy_to_clipboard(rel or "Code", "Copied relative path")

    def copy_file_content(self):
        row = self.selected_row()
        if row is None:
            messagebox.showwarning("No script", "Select a script first.")
            return
        p = Path(safe_str(row.get("file_path")))
        if not p.exists():
            messagebox.showwarning("File missing", str(p))
            return
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            messagebox.showerror("Read error", str(exc))
            return
        self.copy_to_clipboard(text, "Copied file content")

    def copy_io(self):
        row = self.selected_row()
        if row is None:
            messagebox.showwarning("No script", "Select a script first.")
            return
        inputs = split_block(row.get("inputs"))
        outputs = split_block(row.get("outputs"))
        text_lines = [safe_str(row.get("file_name")), "", "INPUTS"]
        text_lines += [f"- {x}" for x in inputs] if inputs else ["- -"]
        text_lines += ["", "OUTPUTS"]
        text_lines += [f"- {x}" for x in outputs] if outputs else ["- -"]
        text = "\n".join(text_lines)
        self._set_output(text)
        self.copy_to_clipboard(text, "Copied IO")

    def export_tree(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="code_registry_tree.txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not p:
            return
        text = self.output.get("1.0", "end").strip() or self.format_tree()
        Path(p).write_text(text, encoding="utf-8")
        self.status_var.set(f"Exported tree: {p}")



    # --------------------------------------------------------
    # Create folders / code files
    # --------------------------------------------------------

    def _selected_base_folder(self) -> Path:
        row = self.selected_row()
        if row is not None:
            file_path = Path(safe_str(row.get("file_path")))
            if file_path.exists() and file_path.is_file():
                return file_path.parent
        if self.selected_folder:
            return (QUANT_ROOT / self.selected_folder).resolve()
        return QUANT_ROOT.resolve()

    def _safe_child_path(self, base_folder: Path, user_text: str, default_suffix: str = "") -> Optional[Path]:
        raw = norm_path(user_text)
        if not raw:
            return None
        if default_suffix and not Path(raw).suffix:
            raw = raw + default_suffix
        target = (base_folder / raw).resolve()
        try:
            target.relative_to(QUANT_ROOT.resolve())
        except Exception:
            messagebox.showerror("Invalid Path", "Path must stay inside the QUANT project folder.")
            return None
        return target

    def _expand_and_select_path(self, target: Path):
        try:
            rel = target.relative_to(QUANT_ROOT).as_posix()
        except Exception:
            rel = target.as_posix()

        folder_rel = rel if target.is_dir() else str(Path(rel).parent).replace("\\", "/")
        if folder_rel == ".":
            folder_rel = ""

        parts = [p for p in norm_path(folder_rel).split("/") if p]
        for i in range(1, len(parts) + 1):
            self.expanded_folders.add("/".join(parts[:i]))

        self.selected_folder = folder_rel
        self.selected_script_id = None

    def new_folder(self):
        base_folder = self._selected_base_folder()
        name = simpledialog.askstring(
            "New Folder",
            f"Create folder inside:\n{base_folder}\n\nFolder name or relative path:",
            parent=self,
        )
        if not name:
            return

        target = self._safe_child_path(base_folder, name)
        if target is None:
            return

        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            messagebox.showerror("Create Folder Error", str(exc))
            return

        # Reload first, because reload rebuilds indexes and clears temporary state.
        self.reload()

        # Then select and expand the newly created real folder.
        self._expand_and_select_path(target)
        self.draw_tree()
        self.update_output()
        self.status_var.set(f"Created folder: {target}")

    def _default_code_template(self, file_path: Path) -> str:
        file_name = file_path.name
        script_id = file_path.stem.lower().replace(" ", "_").replace("-", "_")
        try:
            rel = file_path.relative_to(QUANT_ROOT).as_posix()
        except Exception:
            rel = file_path.as_posix()

        return (
            "# ============================================================\n"
            "# CODE_REGISTRY\n"
            "# ============================================================\n"
            f"# script_id: {script_id}\n"
            f"# script_name: {file_path.stem}\n"
            "# owner: Leon\n"
            "# status: planned\n"
            "# layer: Dashboard\n"
            "# domain: Undefined\n"
            "# asset_type: Python Script\n"
            "# purpose: TODO\n"
            "# inputs:\n"
            "#   - TODO\n"
            "# outputs:\n"
            "#   - TODO\n"
            "# dependencies:\n"
            "#   - TODO\n"
            "# schedule: manual\n"
            "# version: v0.1.0\n"
            "# last_reviewed: 2026-06-08\n"
            "# file_path:\n"
            f"#   - {rel}\n"
            "# ============================================================\n\n"
            "from __future__ import annotations\n\n\n"
            "def main():\n"
            f"    print({file_name!r} + ' created')\n\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

    def new_code(self):
        base_folder = self._selected_base_folder()
        name = simpledialog.askstring(
            "New Code",
            f"Create Python code file inside:\n{base_folder}\n\nFile name or relative path:",
            parent=self,
        )
        if not name:
            return

        target = self._safe_child_path(base_folder, name, default_suffix=".py")
        if target is None:
            return

        if target.exists():
            keep = messagebox.askyesno(
                "File exists",
                f"File already exists:\n{target}\n\nSelect existing file?",
                parent=self,
            )
            if not keep:
                return
        else:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(self._default_code_template(target), encoding="utf-8")
            except Exception as exc:
                messagebox.showerror("Create Code Error", str(exc))
                return

        # Scanner is optional. The filesystem merge makes the file visible even if scanner fails.
        try:
            if SCANNER_CODE.exists():
                subprocess.run([sys.executable, str(SCANNER_CODE)], cwd=str(QUANT_ROOT), check=False)
        except Exception:
            pass

        self.reload()

        # Expand and select after reload.
        self._expand_and_select_path(target)

        try:
            rel = target.relative_to(QUANT_ROOT).as_posix()
            match = pd.DataFrame()
            if not self.df_all.empty and "relative_path" in self.df_all.columns:
                match = self.df_all[self.df_all["relative_path"].astype(str).map(norm_path) == norm_path(rel)]

            if not match.empty:
                self.select_file(int(match.iloc[0]["id"]))
            else:
                self.selected_folder = str(Path(rel).parent).replace("\\", "/")
                if self.selected_folder == ".":
                    self.selected_folder = ""
                self.draw_tree()

        except Exception:
            self.draw_tree()

        self.update_output()
        self.status_var.set(f"Created code file: {target}")


    # --------------------------------------------------------
    # Utility/actions
    # --------------------------------------------------------

    def row_by_id(self, file_id: int) -> Optional[pd.Series]:
        if self.df_all.empty:
            return None
        ids = pd.to_numeric(self.df_all["id"], errors="coerce").fillna(-1).astype(int)
        d = self.df_all[ids == int(file_id)]
        return None if d.empty else d.iloc[0]

    def selected_row(self) -> Optional[pd.Series]:
        if self.selected_script_id is None:
            return None
        return self.row_by_id(self.selected_script_id)

    def selected_path(self) -> Optional[Path]:
        row = self.selected_row()
        if row is not None:
            p = Path(safe_str(row.get("file_path")))
            if str(p):
                return p
        if self.selected_folder:
            return QUANT_ROOT / self.selected_folder
        return QUANT_ROOT

    def open_file(self, file_id: Optional[int] = None):
        if file_id is not None:
            self.select_file(int(file_id))
        p = self.selected_path()
        if p is None or not p.exists():
            return
        target = p if p.is_file() else p
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(target)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            messagebox.showerror("Open error", str(exc))

    def show_help(self):
        messagebox.showinfo(
            "Code Registry Tree",
            "Checkbox = mark/select\nDouble-click folder = open/close\nRight-click = context menu\nOutput shows what will be copied.",
        )

    def show_menu(self):
        menu = tk.Menu(self, tearoff=0, bg=SURFACE_2, fg=TEXT, activebackground=SURFACE_HOVER, activeforeground=TEXT)
        menu.add_command(label="Run Scanner", command=self.run_scanner)
        menu.add_separator()
        menu.add_command(label="New Folder", command=self.new_folder)
        menu.add_command(label="New Code", command=self.new_code)
        menu.add_separator()
        menu.add_command(label="Expand All", command=self.expand_all)
        menu.add_command(label="Collapse All", command=self.collapse_all)
        menu.add_separator()
        menu.add_command(label="Copy Tree", command=self.copy_tree)
        menu.add_command(label="Export Tree", command=self.export_tree)
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def show_tree_context_menu(self, event):
        item = self._hit_item(event.y)
        if item is not None:
            _y1, _y2, kind, folder, fid = item
            if kind == "folder":
                self.select_folder(folder)
            elif fid is not None:
                self.select_file(fid)
        menu = tk.Menu(self, tearoff=0, bg=SURFACE_2, fg=TEXT, activebackground=SURFACE_HOVER, activeforeground=TEXT)
        menu.add_command(label="Copy Path", command=self.copy_path)
        menu.add_command(label="Copy Relative Path", command=self.copy_relative_path)
        menu.add_command(label="Copy Branch", command=self.copy_branch)
        menu.add_command(label="Copy Selected", command=self.copy_selected)
        menu.add_separator()
        menu.add_command(label="New Folder Here", command=self.new_folder)
        menu.add_command(label="New Code Here", command=self.new_code)
        menu.add_separator()
        menu.add_command(label="Open", command=self.open_file)
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _hit_item(self, y: int):
        cy = int(self.tree_canvas.canvasy(y))
        for item in self._tree_y_to_item:
            y1, y2, *_ = item
            if y1 <= cy <= y2:
                return item
        return None

    def run_scanner(self):
        if self.repo.run_scanner():
            self.reload()

    def _tree_mousewheel(self, event):
        try:
            self.tree_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _bind_close_keys(self):
        root = self.winfo_toplevel()
        try:
            root.protocol("WM_DELETE_WINDOW", self.close_window)
            root.bind("<Escape>", lambda _e: self.close_window())
            root.bind("<Control-q>", lambda _e: self.close_window())
            root.bind("<Alt-F4>", lambda _e: self.close_window())
        except Exception:
            pass

    def close_window(self):
        root = self.winfo_toplevel()
        safe_destroy(root)

    def destroy(self):
        try:
            self.tree_canvas.unbind("<Configure>")
            self.tree_canvas.unbind("<MouseWheel>")
        except Exception:
            pass
        super().destroy()


# ============================================================
# Building Block API
# ============================================================


def build_panel(parent, repository: Optional[CodeRegistryTreeRepository] = None, **kwargs):
    return CodeRegistryTreeSelector(parent, repository=repository, **kwargs)


def create_panel(parent, repository: Optional[CodeRegistryTreeRepository] = None, **kwargs):
    return build_panel(parent, repository=repository, **kwargs)


DashboardPanel = CodeRegistryTreeSelector


if __name__ == "__main__":
    root = tk.Tk()
    root.title("QUANT WORKSPACE - Code Registry Tree")
    root.configure(bg=APP_BG)
    root.geometry("760x1220")
    root.minsize(620, 760)
    app = build_panel(root)
    app.pack(fill="both", expand=True)
    root.mainloop()

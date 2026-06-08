"""
QUANT/Dashboard/Building_Blocks/Market/Correlation_Matrix/code.py

Market Correlation Matrix Dashboard Building Block

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: market_correlation_matrix
# script_name: Market Correlation Matrix
# owner: Leon
# status: active
# layer: Dashboard
# domain: Market Analytics
# asset_type: Dashboard
# purpose: Account-style Bloomberg/OMS market correlation dashboard for OHLC parquet data with heatmap, matrix table, KPI cards, relationship tables, clusters, data health and exports.
# inputs:
#   - Data_Center/Data/1_Pipeline/Market/ohcl/<TIMEFRAME>/*.parquet
# outputs:
#   - Dashboard UI
#   - CSV exports
# dependencies:
#   - tkinter
#   - ttk
#   - pathlib
#   - pandas
#   - numpy
#   - matplotlib
# schedule: manual
# version: v1.1.0_ui_style_fixed
# last_reviewed: 2026-06-07
# required_api:
#   - build_panel(parent, repository=None, **kwargs)
# expected_location:
#   - QUANT/Dashboard/Building_Blocks/Market/Correlation_Matrix/code.py
# data_source:
#   - Data_Center/Data/1_Pipeline/Market/ohcl
# scanner:
#   - kein Scanner
# ============================================================
"""

from __future__ import annotations

import math
import os
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:  # pragma: no cover
    matplotlib = None
    Figure = None
    FigureCanvasTkAgg = None


# ============================================================
# PATHS
# ============================================================

def _is_quant_root(path: Path) -> bool:
    try:
        return (path / "Dashboard").exists() and (path / "Data_Center").exists()
    except Exception:
        return False


def find_quant_root(start: Path) -> Path:
    """
    Robust QUANT root detection.

    Valid root must contain:
        Dashboard/
        Data_Center/

    Also supports:
        QUANT_ROOT environment variable
        direct execution from Downloads / Desktop / VS Code cwd
        common Business_Code/QUANT structure
    """
    env = os.environ.get("QUANT_ROOT", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if _is_quant_root(p):
            return p

    start = Path(start).expanduser().resolve()
    if start.is_file():
        start = start.parent

    candidates: List[Path] = []
    candidates.extend([start] + list(start.parents))

    try:
        cwd = Path.cwd().expanduser().resolve()
        candidates.extend([cwd] + list(cwd.parents))
    except Exception:
        pass

    home = Path.home().expanduser().resolve()
    common = [
        home / "Desktop" / "Business_Code" / "QUANT",
        home / "Desktop" / "Business_Code",
        home / "Desktop" / "QUANT",
        home / "Desktop",
        home / "Documents" / "QUANT",
        home / "Downloads" / "QUANT",
        home / "OneDrive" / "Desktop" / "QUANT",
        home / "OneDrive" / "Documents" / "QUANT",
    ]
    candidates.extend(common)

    seen = set()
    for c in candidates:
        try:
            c = c.resolve()
        except Exception:
            continue
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)

        for candidate in [c, c / "QUANT"]:
            if _is_quant_root(candidate):
                return candidate.resolve()

    # bounded fallback scan
    for base in [home / "Desktop" / "Business_Code", home / "Desktop"]:
        if not base.exists():
            continue
        try:
            for p in base.rglob("QUANT"):
                if p.is_dir() and _is_quant_root(p):
                    return p.resolve()
        except Exception:
            continue

    raise RuntimeError(
        "QUANT root not found. Save this file under QUANT/Dashboard/Building_Blocks/Market/Correlation_Matrix/code.py "
        "or set QUANT_ROOT to the project root."
    )


# ============================================================
# CODE REGISTRY
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "market_correlation_matrix",
    "script_name": "Market Correlation Matrix",
    "owner": "Leon",
    "status": "active",
    "layer": "Dashboard",
    "domain": "Market Analytics",
    "asset_type": "Dashboard",
    "purpose": "Account-style Bloomberg/OMS market correlation dashboard for OHLC parquet data.",
    "inputs": ["Data_Center/Data/1_Pipeline/Market/ohcl/<TIMEFRAME>/*.parquet"],
    "outputs": ["Dashboard UI", "CSV exports"],
    "dependencies": ["tkinter", "ttk", "pathlib", "pandas", "numpy", "matplotlib"],
    "schedule": "manual",
    "version": "v1.1.0_ui_style_fixed",
    "last_reviewed": "2026-06-07",
    "required_api": "build_panel(parent, repository=None, **kwargs)",
    "expected_location": "QUANT/Dashboard/Building_Blocks/Market/Correlation_Matrix/code.py",
    "data_source": "Data_Center/Data/1_Pipeline/Market/ohcl",
    "scanner": "kein Scanner",
}


def get_code_registry() -> Dict[str, object]:
    return dict(CODE_REGISTRY)


# ============================================================
# THEME - same visual direction as Live Trades dashboard
# ============================================================

BG = "#000000"
CARD = "#0A0A0A"
CARD_DARK = "#050505"
CARD_SOFT = "#332000"
BORDER = "#2A2A2A"
GRID = "#263746"
FG = "#FFFFFF"
MUTED = "#B8B8B8"
SUBTLE = "#687789"
GREEN = "#00FF66"
RED = "#FF4444"
ORANGE = "#FF9900"
CYAN = "#35FFE2"
YELLOW = "#FFD400"
BLUE = "#00AEEF"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_SUB = ("Consolas", 8)
FONT_NAV = ("Consolas", 8, "bold")
FONT_H2 = ("Consolas", 10, "bold")
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_KPI = ("Consolas", 16, "bold")

DEFAULT_TIMEFRAMES = ["M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]
CORR_METHODS = ["pearson", "spearman", "kendall"]
STRONG_POS = 0.70
STRONG_NEG = -0.70


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class OhlcFileInfo:
    symbol: str
    timeframe: str
    path: Path
    rows: int = 0
    cols: int = 0
    start: str = "-"
    end: str = "-"
    close_column: str = "-"
    date_column: str = "-"
    quality: str = "UNKNOWN"
    error: str = ""


# ============================================================
# HELPERS
# ============================================================

def clear_children(widget: tk.Widget) -> None:
    for child in widget.winfo_children():
        child.destroy()


def safe_float(x: object, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if pd is not None and pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def fmt_num(x: object, decimals: int = 2) -> str:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "-"
        return f"{v:,.{decimals}f}"
    except Exception:
        return "-"


def fmt_pct(x: object, decimals: int = 1) -> str:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "-"
        return f"{v:.{decimals}f}%"
    except Exception:
        return "-"


def normalize_symbol(name: str) -> str:
    s = str(name).replace(".parquet", "").strip()
    return s.replace(".cash", "")


def corr_color(v: object) -> str:
    x = safe_float(v)
    if x >= STRONG_POS:
        return GREEN
    if x <= STRONG_NEG:
        return RED
    if abs(x) >= 0.40:
        return YELLOW
    return MUTED


def pick_date_column(df: "pd.DataFrame") -> Optional[str]:
    candidates = ["time", "timestamp", "date", "datetime", "Date", "Time", "Datetime", "open_time", "close_time"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = str(col).lower()
        if "time" in lower or "date" in lower:
            return col
    return None


def pick_close_column(df: "pd.DataFrame") -> Optional[str]:
    candidates = ["close", "Close", "CLOSE", "bid_close", "ask_close", "mid_close", "close_price"]
    for col in candidates:
        if col in df.columns:
            return col
    for col in df.columns:
        lower = str(col).lower()
        if lower == "c" or lower.endswith("close") or "close" in lower:
            return col
    return None


def style_ax(ax) -> None:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color(CARD)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


# ============================================================
# UI COMPONENTS
# ============================================================

class Card(tk.Frame):
    def __init__(self, parent, bg: str = CARD, padx: int = 12, pady: int = 10):
        super().__init__(parent, bg=bg, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self.inner = tk.Frame(self, bg=bg)
        self.inner.pack(fill="both", expand=True, padx=padx, pady=pady)


class NavButton(tk.Label):
    def __init__(self, parent, text: str, active: bool = False):
        super().__init__(
            parent,
            text=text,
            bg=BG,
            fg=ORANGE if active else MUTED,
            font=FONT_NAV,
            padx=8,
            pady=7,
            cursor="hand2",
        )


class KpiCard(Card):
    def __init__(self, parent, title: str):
        super().__init__(parent, bg=CARD, padx=12, pady=9)
        self.value_var = tk.StringVar(value="-")
        self.sub_var = tk.StringVar(value="")
        tk.Label(self.inner, text=title, bg=CARD, fg=MUTED, font=FONT_SMALL).pack(anchor="w")
        self.value_label = tk.Label(self.inner, textvariable=self.value_var, bg=CARD, fg=FG, font=FONT_KPI)
        self.value_label.pack(anchor="w", pady=(2, 0))
        self.sub_label = tk.Label(self.inner, textvariable=self.sub_var, bg=CARD, fg=SUBTLE, font=FONT_TINY)
        self.sub_label.pack(anchor="w")

    def set(self, value: str, sub: str = "", value_color: str = FG) -> None:
        self.value_var.set(value)
        self.sub_var.set(sub)
        self.value_label.configure(fg=value_color)


class ChartCard(Card):
    def __init__(self, parent, title: str, height: float = 3.0, header_builder=None):
        super().__init__(parent, bg=CARD, padx=12, pady=9)
        self.head = tk.Frame(self.inner, bg=CARD)
        self.head.pack(fill="x")
        tk.Label(self.head, text=title, bg=CARD, fg=FG, font=FONT_H2).pack(side="left")
        if header_builder:
            header_builder(self.head)

        if Figure is None or FigureCanvasTkAgg is None:
            self.fig = None
            self.ax = None
            self.canvas = None
            self.fallback = tk.Label(self.inner, text="matplotlib not available", bg=CARD, fg=RED, font=FONT_SMALL)
            self.fallback.pack(fill="both", expand=True)
        else:
            self.fig = Figure(figsize=(5.2, height), dpi=100)
            self.fig.patch.set_facecolor(CARD)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.inner)
            self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))

    def reset(self) -> None:
        if self.ax is None or self.fig is None:
            return
        self.ax.clear()
        self.fig.patch.set_facecolor(CARD)
        self.ax.set_facecolor(CARD)


# ============================================================
# DASHBOARD
# ============================================================

class CorrelationMatrixDashboard(tk.Frame):
    def __init__(self, parent, repository=None, **kwargs):
        super().__init__(parent, bg=BG)
        self.repository = repository
        self.kwargs = kwargs

        self.quant_root = self._extract_root(repository) or find_quant_root(Path(__file__))
        self.data_root = self.quant_root / "Data_Center" / "Data" / "1_Pipeline" / "Market" / "ohcl"
        self.data_root = self._resolve_ohlc_root()

        self.selected_timeframe = tk.StringVar(value="H1")
        self.selected_method = tk.StringVar(value="pearson")
        self.search_var = tk.StringVar(value="")
        self.symbol_var = tk.StringVar(value="All Symbols")
        self.min_abs_var = tk.DoubleVar(value=0.00)
        self.summary_var = tk.StringVar(value="Relationships: 0")
        self.status_var = tk.StringVar(value="INIT")

        self.files: Dict[str, OhlcFileInfo] = {}
        self.price_data: Dict[str, "pd.Series"] = {}
        self.returns_df: Optional["pd.DataFrame"] = None
        self.corr_df: Optional["pd.DataFrame"] = None
        self.relationships: List[Tuple[str, str, float]] = []
        self.clusters: List[List[str]] = []
        self.last_error = ""
        self.last_selected_pair: Optional[Tuple[str, str, float]] = None

        self.kpis: Dict[str, KpiCard] = {}

        self._configure_style()
        self._build_ui()
        self.refresh_data()
        self.after(1000, self._tick_clock)

    # --------------------------------------------------------
    # ROOT

    def _extract_root(self, repository) -> Optional[Path]:
        candidates: List[Any] = []
        if repository is not None:
            candidates.append(repository)
            for attr in ("quant_root", "root", "root_path", "project_root", "base_path"):
                try:
                    val = getattr(repository, attr, None)
                    if val:
                        candidates.append(val)
                except Exception:
                    pass
            if isinstance(repository, dict):
                for key in ("quant_root", "root", "root_path", "project_root", "base_path"):
                    if repository.get(key):
                        candidates.append(repository.get(key))
        for item in candidates:
            try:
                p = Path(str(item)).expanduser().resolve()
            except Exception:
                continue
            for c in [p] + list(p.parents):
                if _is_quant_root(c):
                    return c
                if _is_quant_root(c / "QUANT"):
                    return c / "QUANT"
        return None

    def _resolve_ohlc_root(self) -> Path:
        env = os.environ.get("QUANT_OHLC_ROOT", "").strip()
        if env:
            p = Path(env).expanduser().resolve()
            if p.exists():
                return p

        if self.data_root.exists():
            return self.data_root

        rel = Path("Data_Center") / "Data" / "1_Pipeline" / "Market" / "ohcl"
        home = Path.home().expanduser().resolve()
        for base in [self.quant_root, home / "Desktop" / "Business_Code", home / "Desktop", home / "Documents", home / "Downloads"]:
            try:
                base = Path(base).resolve()
            except Exception:
                continue
            candidates = [base / rel, base / "QUANT" / rel, base / "Business_Code" / "QUANT" / rel]
            try:
                if base.exists():
                    candidates.extend([p / rel for p in base.glob("*/QUANT") if p.is_dir()])
            except Exception:
                pass
            for c in candidates:
                if c.exists():
                    return c
        return self.data_root

    # --------------------------------------------------------
    # STYLE

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("Dark.Treeview", background=CARD, foreground=FG, fieldbackground=CARD, borderwidth=0, rowheight=24, font=FONT_SMALL)
        style.configure("Dark.Treeview.Heading", background=CARD_DARK, foreground=MUTED, borderwidth=0, font=FONT_TINY)
        style.map("Dark.Treeview", background=[("selected", CARD_SOFT)], foreground=[("selected", FG)])
        style.configure("Dark.TCombobox", fieldbackground=CARD_DARK, background=CARD, foreground=FG, arrowcolor=ORANGE, bordercolor=BORDER)
        style.configure("Dark.Horizontal.TScale", background=CARD, troughcolor=CARD_DARK)

    # --------------------------------------------------------
    # BUILD

    def _build_ui(self) -> None:
        self.pack(fill="both", expand=True)
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=20, pady=16)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        self._build_header(root)
        self._build_nav(root)
        self._build_kpis(root)
        self._build_filters(root)
        self._build_dashboard_grid(root)

    def _build_header(self, parent) -> None:
        header = tk.Frame(parent, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        left = tk.Frame(header, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Market Correlation Matrix", bg=BG, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(left, text=f"Source: {self.data_root}", bg=BG, fg=MUTED, font=FONT_SUB).pack(anchor="w", pady=(1, 0))

        right = tk.Frame(header, bg=BG)
        right.grid(row=0, column=1, sticky="e")
        self.clock_label = tk.Label(right, text="-", bg=BG, fg=MUTED, font=FONT_SMALL)
        self.clock_label.pack(side="right", padx=(10, 0))
        self.status_label = tk.Label(right, textvariable=self.status_var, bg=BG, fg=ORANGE, font=FONT_SMALL)
        self.status_label.pack(side="right")

    def _build_nav(self, parent) -> None:
        nav = tk.Frame(parent, bg=BG)
        nav.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        labels = ["⌂  Overview", "▦  Portfolio", "♙  Strategies", "⚚  Trades", "◈  Research", "▤  Market", "▣  Registry", "⚙  Settings"]
        for label in labels:
            NavButton(nav, label, active=("Market" in label)).pack(side="left", padx=(0, 18))
        tk.Frame(parent, bg=BORDER, height=1).grid(row=2, column=0, sticky="ew", pady=(0, 14))

    def _build_kpis(self, parent) -> None:
        row = tk.Frame(parent, bg=BG)
        row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for i in range(8):
            row.columnconfigure(i, weight=1, uniform="kpi")

        names = ["Assets", "Avg Abs Corr", "Strong Pos", "Strong Neg", "Clusters", "Diversify", "Quality", "Updated"]
        for i, name in enumerate(names):
            card = KpiCard(row, name)
            card.grid(row=0, column=i, sticky="nsew", padx=5)
            self.kpis[name] = card
        row.bind("<Configure>", lambda e: self._responsive_kpis(row, e.width))

    def _responsive_kpis(self, row: tk.Frame, width: int) -> None:
        children = row.winfo_children()
        cols = 8
        if width < 800:
            cols = 2
        elif width < 1100:
            cols = 4
        elif width < 1450:
            cols = 6
        for i, child in enumerate(children):
            child.grid_forget()
            child.grid(row=i // cols, column=i % cols, sticky="nsew", padx=5, pady=5)
        for c in range(8):
            row.columnconfigure(c, weight=1 if c < cols else 0, uniform="kpi" if c < cols else "")

    def _build_filters(self, parent) -> None:
        card = Card(parent, bg=CARD, padx=12, pady=8)
        card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        inner = card.inner
        inner.columnconfigure(1, weight=1)

        tk.Label(inner, text="Search", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=0, padx=(0, 6), sticky="w")
        search = tk.Entry(inner, textvariable=self.search_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL)
        search.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        search.bind("<KeyRelease>", lambda _e: self.update_table())

        tk.Label(inner, text="TF", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=2, padx=(4, 6))
        self.tf_combo = ttk.Combobox(inner, textvariable=self.selected_timeframe, values=DEFAULT_TIMEFRAMES, state="readonly", style="Dark.TCombobox", width=8)
        self.tf_combo.grid(row=0, column=3, padx=(0, 10))
        self.tf_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_data())

        tk.Label(inner, text="Method", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=4, padx=(4, 6))
        self.method_combo = ttk.Combobox(inner, textvariable=self.selected_method, values=CORR_METHODS, state="readonly", style="Dark.TCombobox", width=10)
        self.method_combo.grid(row=0, column=5, padx=(0, 10))
        self.method_combo.bind("<<ComboboxSelected>>", lambda _e: self.recalculate())

        tk.Label(inner, text="Min |Corr|", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=6, padx=(4, 6))
        self.min_scale = tk.Scale(inner, from_=0.0, to=0.95, resolution=0.05, orient="horizontal", showvalue=True, length=120,
                                  variable=self.min_abs_var, bg=CARD, fg=MUTED, troughcolor=CARD_DARK,
                                  activebackground=ORANGE, highlightthickness=0, bd=0, command=lambda _v: self.update_table())
        self.min_scale.grid(row=0, column=7, padx=(0, 10))

        tk.Label(inner, textvariable=self.summary_var, bg=CARD, fg=ORANGE, font=FONT_SMALL).grid(row=0, column=8, padx=(0, 12))

        self._button(inner, "Refresh", self.refresh_data).grid(row=0, column=9, padx=(0, 6))
        self._button(inner, "Export", self.export_matrix).grid(row=0, column=10, padx=(0, 6))
        self._button(inner, "Reset", self.reset_filters).grid(row=0, column=11)

    def _button(self, parent, text: str, command) -> tk.Button:
        return tk.Button(parent, text=text, command=command, bg=CARD_DARK, fg=ORANGE, activebackground=CARD_SOFT,
                         activeforeground=FG, relief="flat", font=FONT_SMALL, padx=14, pady=7)

    def _build_dashboard_grid(self, parent) -> None:
        grid = tk.Frame(parent, bg=BG)
        grid.grid(row=5, column=0, sticky="nsew")
        for c in range(12):
            grid.columnconfigure(c, weight=1, uniform="dash")
        for r in range(8):
            grid.rowconfigure(r, weight=1, uniform="dash")

        self.market_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.market_card.grid(row=0, column=0, rowspan=3, columnspan=2, sticky="nsew", padx=5, pady=5)
        self._build_market_shell()

        self.heatmap_card = ChartCard(grid, "Correlation Heatmap", height=3.7, header_builder=self._heatmap_header)
        self.heatmap_card.grid(row=0, column=2, rowspan=4, columnspan=6, sticky="nsew", padx=5, pady=5)

        self.health_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.health_card.grid(row=0, column=8, rowspan=2, columnspan=4, sticky="nsew", padx=5, pady=5)
        self._build_health_shell()

        self.details_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.details_card.grid(row=2, column=8, rowspan=2, columnspan=4, sticky="nsew", padx=5, pady=5)
        self._build_details_shell()

        self.matrix_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.matrix_card.grid(row=4, column=0, rowspan=2, columnspan=8, sticky="nsew", padx=5, pady=5)
        self._build_matrix_shell()

        self.pos_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.pos_card.grid(row=6, column=0, rowspan=2, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.pos_tree = self._build_relation_shell(self.pos_card, "Strongest Positive")

        self.neg_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.neg_card.grid(row=6, column=3, rowspan=2, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.neg_tree = self._build_relation_shell(self.neg_card, "Strongest Negative")

        self.cluster_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.cluster_card.grid(row=4, column=8, rowspan=4, columnspan=4, sticky="nsew", padx=5, pady=5)
        self._build_cluster_shell()

        self.flow_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.flow_card.grid(row=6, column=6, rowspan=2, columnspan=2, sticky="nsew", padx=5, pady=5)
        self._build_flow_shell()

    def _heatmap_header(self, parent) -> None:
        tk.Label(parent, text="Click matrix rows for pair details", bg=CARD, fg=SUBTLE, font=FONT_TINY).pack(side="right")

    def _build_market_shell(self) -> None:
        tk.Label(self.market_card.inner, text="Market Watchlist", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.asset_listbox = tk.Listbox(self.market_card.inner, bg=CARD_DARK, fg=FG, selectbackground=CARD_SOFT, selectforeground=FG,
                                        relief="flat", font=FONT_SMALL, exportselection=False)
        self.asset_listbox.pack(fill="both", expand=True, pady=(8, 8))
        self.asset_listbox.bind("<<ListboxSelect>>", self._on_asset_select)

        tk.Label(self.market_card.inner, text="Timeframes", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w", pady=(6, 0))
        tf_frame = tk.Frame(self.market_card.inner, bg=CARD)
        tf_frame.pack(fill="x", pady=(6, 0))
        self.tf_buttons: Dict[str, tk.Label] = {}
        for i, tf in enumerate(DEFAULT_TIMEFRAMES):
            lbl = tk.Label(tf_frame, text=tf, bg=CARD_DARK, fg=MUTED, font=FONT_TINY, padx=8, pady=4, cursor="hand2")
            lbl.grid(row=i // 4, column=i % 4, sticky="ew", padx=2, pady=2)
            lbl.bind("<Button-1>", lambda _e, t=tf: self._select_timeframe(t))
            self.tf_buttons[tf] = lbl
        for c in range(4):
            tf_frame.columnconfigure(c, weight=1)

    def _build_health_shell(self) -> None:
        tk.Label(self.health_card.inner, text="Data Health", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.health_text = tk.Text(self.health_card.inner, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, wrap="word", height=8)
        self.health_text.pack(fill="both", expand=True, pady=(8, 0))
        self.health_text.configure(state="disabled")

    def _build_details_shell(self) -> None:
        tk.Label(self.details_card.inner, text="Correlation Details", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.details_text = tk.Text(self.details_card.inner, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, wrap="word", height=8)
        self.details_text.pack(fill="both", expand=True, pady=(8, 0))
        self.details_text.configure(state="disabled")

    def _build_matrix_shell(self) -> None:
        head = tk.Frame(self.matrix_card.inner, bg=CARD)
        head.pack(fill="x")
        tk.Label(head, text="Correlation Matrix", bg=CARD, fg=FG, font=FONT_H2).pack(side="left")
        self.matrix_period_var = tk.StringVar(value="-")
        tk.Label(head, textvariable=self.matrix_period_var, bg=CARD, fg=MUTED, font=FONT_TINY).pack(side="right")

        table_wrap = tk.Frame(self.matrix_card.inner, bg=CARD)
        table_wrap.pack(fill="both", expand=True, pady=(10, 0))
        self.matrix_tree = ttk.Treeview(table_wrap, show="headings", style="Dark.Treeview")
        self.matrix_v = tk.Scrollbar(table_wrap, orient="vertical", command=self.matrix_tree.yview)
        self.matrix_h = tk.Scrollbar(table_wrap, orient="horizontal", command=self.matrix_tree.xview)
        self.matrix_tree.configure(yscrollcommand=self.matrix_v.set, xscrollcommand=self.matrix_h.set)
        self.matrix_tree.pack(side="left", fill="both", expand=True)
        self.matrix_v.pack(side="right", fill="y")
        self.matrix_h.pack(side="bottom", fill="x")
        self.matrix_tree.bind("<<TreeviewSelect>>", self._on_matrix_select)
        self.matrix_tree.tag_configure("high_pos", foreground=GREEN)
        self.matrix_tree.tag_configure("high_neg", foreground=RED)
        self.matrix_tree.tag_configure("neutral", foreground=FG)

    def _build_relation_shell(self, card: Card, title: str) -> ttk.Treeview:
        tk.Label(card.inner, text=title, bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        wrap = tk.Frame(card.inner, bg=CARD)
        wrap.pack(fill="both", expand=True, pady=(8, 0))
        cols = ("a", "b", "corr")
        tree = ttk.Treeview(wrap, columns=cols, show="headings", style="Dark.Treeview", height=8)
        for col, text, width, anchor in [("a", "Asset A", 80, "w"), ("b", "Asset B", 80, "w"), ("corr", "Corr", 70, "e")]:
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor=anchor, stretch=True)
        ysb = tk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=ysb.set)
        tree.pack(side="left", fill="both", expand=True)
        ysb.pack(side="right", fill="y")
        tree.bind("<<TreeviewSelect>>", self._on_relation_select)
        return tree

    def _build_cluster_shell(self) -> None:
        tk.Label(self.cluster_card.inner, text="Cluster Analysis", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.cluster_text = tk.Text(self.cluster_card.inner, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, wrap="word")
        self.cluster_text.pack(fill="both", expand=True, pady=(8, 0))
        self.cluster_text.configure(state="disabled")

    def _build_flow_shell(self) -> None:
        tk.Label(self.flow_card.inner, text="Visual Flow", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        flow = "OHLC\n ↓\nReturns\n ↓\nCorrelation\n ↓\nMatrix\n ↓\nClusters"
        tk.Label(self.flow_card.inner, text=flow, bg=CARD, fg=MUTED, font=FONT_SMALL, justify="center").pack(fill="both", expand=True, pady=(8, 0))

    # --------------------------------------------------------
    # DATA

    def refresh_data(self) -> None:
        self.status_var.set("LOADING")
        self.last_error = ""
        try:
            self.load_data()
            self.recalculate()
            self.status_var.set("ONLINE")
        except Exception as exc:
            self.last_error = traceback.format_exc()
            self.status_var.set("ERROR")
            self.price_data = {}
            self.returns_df = None
            self.corr_df = None
            self.relationships = []
            self.clusters = []
            self._update_error_state(str(exc))

    def load_data(self) -> None:
        if pd is None:
            raise ImportError("pandas is required for this dashboard.")
        if np is None:
            raise ImportError("numpy is required for this dashboard.")

        tf = self.selected_timeframe.get().strip() or "H1"
        tf_dir = self.data_root / tf
        if not tf_dir.exists():
            raise FileNotFoundError(f"Timeframe folder not found: {tf_dir}")

        files = sorted(tf_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in: {tf_dir}")

        self.files = {}
        self.price_data = {}

        for path in files:
            symbol = normalize_symbol(path.stem)
            info = OhlcFileInfo(symbol=symbol, timeframe=tf, path=path)
            try:
                df = pd.read_parquet(path)
                info.rows = int(len(df))
                info.cols = int(len(df.columns))
                date_col = pick_date_column(df)
                close_col = pick_close_column(df)
                info.date_column = date_col or "-"
                info.close_column = close_col or "-"

                if close_col is None:
                    raise ValueError(f"No close column found. Columns={list(df.columns)}")

                close = pd.to_numeric(df[close_col], errors="coerce")
                if date_col is not None:
                    idx = pd.to_datetime(df[date_col], errors="coerce", utc=True)
                    close.index = idx
                    valid_idx = close.index.notna()
                    close = close[valid_idx]
                    if len(close) > 0:
                        info.start = str(close.index.min()).replace("+00:00", "")[:19]
                        info.end = str(close.index.max()).replace("+00:00", "")[:19]
                else:
                    close.index = pd.RangeIndex(len(close))
                    info.start = "row 0"
                    info.end = f"row {max(len(close)-1, 0)}"

                close = close.dropna()
                if len(close) < 5:
                    raise ValueError("Not enough valid close values")

                info.quality = "PASS"
                self.price_data[symbol] = close.astype(float)
            except Exception as exc:
                info.quality = "ERROR"
                info.error = str(exc)
            self.files[symbol] = info

        if not self.price_data:
            errors = "\n".join(f"{s}: {i.error}" for s, i in self.files.items() if i.error)
            raise RuntimeError(f"No valid OHLC price series loaded from {tf_dir}\n{errors}")

    def recalculate(self) -> None:
        if pd is None or np is None:
            return
        if not self.price_data:
            self.update_table()
            return

        aligned = pd.DataFrame(self.price_data)
        aligned = aligned.sort_index()
        returns = aligned.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(how="all")
        returns = returns.dropna(axis=1, how="all")

        method = self.selected_method.get() or "pearson"
        self.returns_df = returns
        self.corr_df = returns.corr(method=method)

        self.relationships = []
        assets = list(self.corr_df.columns)
        for i, a in enumerate(assets):
            for j in range(i + 1, len(assets)):
                b = assets[j]
                val = safe_float(self.corr_df.loc[a, b], 0.0)
                if not math.isnan(val):
                    self.relationships.append((a, b, val))
        self.relationships.sort(key=lambda x: abs(x[2]), reverse=True)

        self._detect_clusters()
        self._refresh_symbol_filter()
        self.update_table()
        self.update_details()

    def _detect_clusters(self) -> None:
        self.clusters = []
        if self.corr_df is None or self.corr_df.empty:
            return

        assets = list(self.corr_df.columns)
        visited = set()
        for a in assets:
            if a in visited:
                continue
            cluster = {a}
            queue = [a]
            visited.add(a)
            while queue:
                x = queue.pop(0)
                for b in assets:
                    if b in visited or b == x:
                        continue
                    if safe_float(self.corr_df.loc[x, b], 0.0) >= STRONG_POS:
                        visited.add(b)
                        cluster.add(b)
                        queue.append(b)
            self.clusters.append(sorted(cluster))

    # --------------------------------------------------------
    # UPDATE UI

    def update_table(self) -> None:
        self._update_kpis()
        self._update_asset_list()
        self._update_tf_buttons()
        self._plot_heatmap()
        self._update_matrix_table()
        self._update_relationship_tables()
        self._update_cluster_text()
        self._update_health_text()
        self.matrix_period_var.set(f"TF: {self.selected_timeframe.get()} | Method: {self.selected_method.get()} | Source: {self.data_root.name}")

    def _filtered_assets(self) -> List[str]:
        if self.corr_df is None or self.corr_df.empty:
            return []
        assets = list(self.corr_df.columns)
        q = self.search_var.get().strip().lower()
        selected = self.symbol_var.get()
        if selected and selected != "All Symbols":
            assets = [a for a in assets if a == selected]
        elif q:
            assets = [a for a in assets if q in a.lower()]
        return assets or list(self.corr_df.columns)

    def _filtered_relationships(self) -> List[Tuple[str, str, float]]:
        q = self.search_var.get().strip().lower()
        selected = self.symbol_var.get()
        min_abs = safe_float(self.min_abs_var.get(), 0.0)
        rels = []
        for a, b, v in self.relationships:
            if selected != "All Symbols" and selected not in {a, b}:
                continue
            if q and q not in a.lower() and q not in b.lower():
                continue
            if abs(v) < min_abs:
                continue
            rels.append((a, b, v))
        return rels

    def _update_kpis(self) -> None:
        valid_files = [i for i in self.files.values() if i.quality == "PASS"]
        failed_files = [i for i in self.files.values() if i.quality != "PASS"]

        if self.corr_df is not None and not self.corr_df.empty:
            vals = []
            for a, b, v in self.relationships:
                vals.append(v)
            avg_abs = float(np.mean(np.abs(vals))) if vals and np is not None else 0.0
            pos = sum(1 for _, _, v in self.relationships if v >= STRONG_POS)
            neg = sum(1 for _, _, v in self.relationships if v <= STRONG_NEG)
            div_score = max(0.0, 100.0 - avg_abs * 100.0)
        else:
            avg_abs = 0.0
            pos = 0
            neg = 0
            div_score = 0.0

        quality = "PASS" if valid_files and not failed_files else "WARN" if valid_files else "ERROR"
        q_color = GREEN if quality == "PASS" else YELLOW if quality == "WARN" else RED

        mapping = {
            "Assets": (str(len(valid_files)), f"failed {len(failed_files)}", FG),
            "Avg Abs Corr": (fmt_num(avg_abs, 2), self.selected_method.get(), ORANGE),
            "Strong Pos": (str(pos), f"> {STRONG_POS:.2f}", GREEN),
            "Strong Neg": (str(neg), f"< {STRONG_NEG:.2f}", RED),
            "Clusters": (str(len(self.clusters)), "positive groups", CYAN),
            "Diversify": (fmt_num(div_score, 1), "100 - avg abs corr", FG),
            "Quality": (quality, f"files {len(self.files)}", q_color),
            "Updated": (datetime.now().strftime("%H:%M"), self.selected_timeframe.get(), FG),
        }
        for name, (value, sub, color) in mapping.items():
            if name in self.kpis:
                self.kpis[name].set(value, sub, color)

        self.summary_var.set(f"Relationships: {len(self._filtered_relationships())}")

    def _refresh_symbol_filter(self) -> None:
        values = ["All Symbols"]
        if self.corr_df is not None and not self.corr_df.empty:
            values += list(self.corr_df.columns)
        current = self.symbol_var.get()
        if hasattr(self, "symbol_combo"):
            self.symbol_combo["values"] = values
        if current not in values:
            self.symbol_var.set("All Symbols")

    def _update_asset_list(self) -> None:
        if not hasattr(self, "asset_listbox"):
            return
        self.asset_listbox.delete(0, "end")
        assets = list(self.files.keys())
        for asset in sorted(assets):
            info = self.files[asset]
            suffix = "PASS" if info.quality == "PASS" else "ERR"
            self.asset_listbox.insert("end", f"{asset:<10} {suffix}")

    def _update_tf_buttons(self) -> None:
        current = self.selected_timeframe.get()
        for tf, lbl in self.tf_buttons.items():
            active = tf == current
            lbl.configure(bg=CARD_SOFT if active else CARD_DARK, fg=ORANGE if active else MUTED)

    def _plot_heatmap(self) -> None:
        card = self.heatmap_card
        if card.ax is None or card.canvas is None:
            return
        card.reset()
        ax = card.ax
        style_ax(ax)

        if self.corr_df is None or self.corr_df.empty:
            ax.text(0.5, 0.5, "No correlation data", color=RED, ha="center", va="center", transform=ax.transAxes)
            card.canvas.draw_idle()
            return

        df = self.corr_df.copy()
        assets = self._filtered_assets()
        if assets:
            df = df.loc[assets, assets]

        arr = df.values.astype(float)
        im = ax.imshow(arr, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(df.columns)))
        ax.set_yticks(range(len(df.index)))
        ax.set_xticklabels(df.columns, rotation=45, ha="right", color=MUTED, fontsize=7)
        ax.set_yticklabels(df.index, color=MUTED, fontsize=7)

        for i in range(len(df.index)):
            for j in range(len(df.columns)):
                v = arr[i, j]
                color = FG if abs(v) < 0.65 else BG
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

        try:
            cbar = card.fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.ax.tick_params(colors=MUTED, labelsize=7)
            cbar.outline.set_edgecolor(BORDER)
        except Exception:
            pass
        card.fig.tight_layout(pad=1.2)
        card.canvas.draw_idle()

    def _update_matrix_table(self) -> None:
        tree = self.matrix_tree
        tree.delete(*tree.get_children())
        if self.corr_df is None or self.corr_df.empty:
            tree["columns"] = []
            return

        df = self.corr_df.copy()
        assets = list(df.columns)
        cols = ["asset"] + assets
        tree["columns"] = cols
        for col in cols:
            text = "Asset" if col == "asset" else col
            width = 90 if col == "asset" else 76
            anchor = "w" if col == "asset" else "e"
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor=anchor, stretch=False)

        for asset in assets:
            row_vals = [asset]
            row = df.loc[asset]
            for col in assets:
                row_vals.append(fmt_num(row[col], 2))
            max_abs = max(abs(safe_float(x, 0.0)) for x in row.values if not pd.isna(x)) if pd is not None else 0.0
            tag = "high_pos" if max_abs >= STRONG_POS else "neutral"
            tree.insert("", "end", iid=asset, values=row_vals, tags=(tag,))

    def _update_relationship_tables(self) -> None:
        for tree in [self.pos_tree, self.neg_tree]:
            tree.delete(*tree.get_children())

        rels = self._filtered_relationships()
        pos = sorted([r for r in rels if r[2] > 0], key=lambda x: x[2], reverse=True)[:30]
        neg = sorted([r for r in rels if r[2] < 0], key=lambda x: x[2])[:30]

        for a, b, v in pos:
            self.pos_tree.insert("", "end", values=(a, b, fmt_num(v, 2)), tags=(a, b, str(v)))
        for a, b, v in neg:
            self.neg_tree.insert("", "end", values=(a, b, fmt_num(v, 2)), tags=(a, b, str(v)))

    def _update_cluster_text(self) -> None:
        self.cluster_text.configure(state="normal")
        self.cluster_text.delete("1.0", "end")
        if not self.clusters:
            self.cluster_text.insert("end", "No clusters detected.\n")
        else:
            for i, cluster in enumerate(sorted(self.clusters, key=lambda c: (-len(c), c)), start=1):
                label = f"CLUSTER {i:02d}"
                score = self._cluster_score(cluster)
                self.cluster_text.insert("end", f"{label:<12} score={score:>5.1f} | size={len(cluster)}\n")
                self.cluster_text.insert("end", "  " + "  ".join(cluster) + "\n\n")
        self.cluster_text.configure(state="disabled")

    def _cluster_score(self, cluster: List[str]) -> float:
        if self.corr_df is None or len(cluster) < 2:
            return 0.0
        vals = []
        for i, a in enumerate(cluster):
            for b in cluster[i+1:]:
                vals.append(safe_float(self.corr_df.loc[a, b], 0.0))
        return float(np.mean(vals) * 100.0) if vals and np is not None else 0.0

    def _update_health_text(self) -> None:
        self.health_text.configure(state="normal")
        self.health_text.delete("1.0", "end")
        valid = [i for i in self.files.values() if i.quality == "PASS"]
        failed = [i for i in self.files.values() if i.quality != "PASS"]
        ret_rows = 0 if self.returns_df is None else len(self.returns_df)
        lines = [
            f"Root      : {self.data_root}",
            f"Timeframe : {self.selected_timeframe.get()}",
            f"Method    : {self.selected_method.get()}",
            f"Files     : {len(self.files)}",
            f"Loaded    : {len(valid)}",
            f"Failed    : {len(failed)}",
            f"Return rows: {ret_rows}",
        ]
        if failed:
            lines.append("\nErrors:")
            for info in failed[:5]:
                lines.append(f"- {info.symbol}: {info.error[:80]}")
        self.health_text.insert("end", "\n".join(lines))
        self.health_text.configure(state="disabled")

    def update_details(self, pair: Optional[Tuple[str, str, float]] = None) -> None:
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")

        if self.last_error:
            self.details_text.insert("end", "LOAD ERROR\n\n", "head")
            self.details_text.insert("end", self.last_error)
            self.details_text.configure(state="disabled")
            return

        if pair is None:
            pair = self.last_selected_pair
        if pair is None and self.relationships:
            pair = self.relationships[0]
        if pair is None:
            self.details_text.insert("end", "No pair selected.")
            self.details_text.configure(state="disabled")
            return

        a, b, v = pair
        self.last_selected_pair = pair
        info_a = self.files.get(a)
        info_b = self.files.get(b)

        lines = [
            f"PAIR      : {a} ↔ {b}",
            f"CORR      : {v:.4f}",
            f"METHOD    : {self.selected_method.get()}",
            f"STRENGTH  : {self._classify_corr(v)}",
            "",
            f"{a} FILE",
            f"Rows      : {info_a.rows if info_a else '-'}",
            f"Start     : {info_a.start if info_a else '-'}",
            f"End       : {info_a.end if info_a else '-'}",
            "",
            f"{b} FILE",
            f"Rows      : {info_b.rows if info_b else '-'}",
            f"Start     : {info_b.start if info_b else '-'}",
            f"End       : {info_b.end if info_b else '-'}",
        ]
        self.details_text.insert("end", "\n".join(lines))
        self.details_text.configure(state="disabled")

    def _classify_corr(self, v: float) -> str:
        if v >= 0.85:
            return "VERY STRONG POSITIVE"
        if v >= 0.70:
            return "STRONG POSITIVE"
        if v >= 0.40:
            return "MODERATE POSITIVE"
        if v <= -0.85:
            return "VERY STRONG NEGATIVE"
        if v <= -0.70:
            return "STRONG NEGATIVE"
        if v <= -0.40:
            return "MODERATE NEGATIVE"
        return "LOW / NEUTRAL"

    def _update_error_state(self, message: str) -> None:
        for card in self.kpis.values():
            card.set("-", "load failed", RED)
        self.summary_var.set("Relationships: 0")
        self._update_health_text()
        self.update_details()
        if hasattr(self, "matrix_tree"):
            self.matrix_tree.delete(*self.matrix_tree.get_children())
        if hasattr(self, "pos_tree"):
            self.pos_tree.delete(*self.pos_tree.get_children())
        if hasattr(self, "neg_tree"):
            self.neg_tree.delete(*self.neg_tree.get_children())

    # --------------------------------------------------------
    # EVENTS

    def _tick_clock(self) -> None:
        try:
            self.clock_label.configure(text=datetime.now().strftime("%a %d %b %Y %H:%M:%S").upper())
        except Exception:
            pass
        self.after(1000, self._tick_clock)

    def _select_timeframe(self, tf: str) -> None:
        self.selected_timeframe.set(tf)
        self.refresh_data()

    def _on_asset_select(self, _event=None) -> None:
        sel = self.asset_listbox.curselection()
        if not sel:
            return
        raw = self.asset_listbox.get(sel[0])
        asset = raw.split()[0].strip()
        if asset:
            self.symbol_var.set(asset)
            self.update_table()

    def _on_matrix_select(self, _event=None) -> None:
        sel = self.matrix_tree.selection()
        if not sel or self.corr_df is None:
            return
        a = sel[0]
        best = None
        best_abs = -1.0
        for b in self.corr_df.columns:
            if b == a:
                continue
            v = safe_float(self.corr_df.loc[a, b], 0.0)
            if abs(v) > best_abs:
                best_abs = abs(v)
                best = (a, b, v)
        if best:
            self.update_details(best)

    def _on_relation_select(self, event=None) -> None:
        tree = event.widget if event is not None else None
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        if len(vals) >= 3:
            a, b, v = vals[0], vals[1], safe_float(vals[2], 0.0)
            self.update_details((a, b, v))

    def reset_filters(self) -> None:
        self.search_var.set("")
        self.symbol_var.set("All Symbols")
        self.min_abs_var.set(0.0)
        self.update_table()

    def export_matrix(self) -> None:
        if self.corr_df is None or self.corr_df.empty:
            messagebox.showinfo("Export", "No matrix available.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Correlation Matrix",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"correlation_matrix_{self.selected_timeframe.get()}_{self.selected_method.get()}.csv",
        )
        if not path:
            return
        self.corr_df.to_csv(path, encoding="utf-8-sig")


# ============================================================
# REQUIRED API
# ============================================================

def load_data(*args, **kwargs):
    """Module-level compatibility stub. Dashboard instance uses CorrelationMatrixDashboard.load_data()."""
    raise RuntimeError("Use build_panel(parent).load_data() on the dashboard instance.")


def refresh_data(*args, **kwargs):
    """Module-level compatibility stub. Dashboard instance uses CorrelationMatrixDashboard.refresh_data()."""
    raise RuntimeError("Use build_panel(parent).refresh_data() on the dashboard instance.")


def update_table(*args, **kwargs):
    """Module-level compatibility stub. Dashboard instance uses CorrelationMatrixDashboard.update_table()."""
    raise RuntimeError("Use build_panel(parent).update_table() on the dashboard instance.")


def update_details(*args, **kwargs):
    """Module-level compatibility stub. Dashboard instance uses CorrelationMatrixDashboard.update_details()."""
    raise RuntimeError("Use build_panel(parent).update_details() on the dashboard instance.")


def build_panel(parent, repository=None, **kwargs):
    return CorrelationMatrixDashboard(parent, repository=repository, **kwargs)


# ============================================================
# DIRECT RUN TEST
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.title("QUANT - Market Correlation Matrix")
    root.configure(bg=BG)
    root.geometry("1500x900")
    panel = build_panel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()

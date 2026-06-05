# -*- coding: utf-8 -*-
"""
QUANT/Dashboard/Building_Blocks/Market/code.py

Fast Market Research Dashboard
- Reads daily regime/research features once
- Sorts latest day at top
- Uses feature views: OVERVIEW, TREND, PULLBACK, BREAKOUT, CANDLE, SPIKE, CHAOS
- Does NOT hard-label days as choppy/spike/trend
- Shows efficient mini M15 charts in the first table column
- Mini charts are loaded lazily in small batches so the dashboard opens fast
- Double-click row opens full M15 context chart with 2 trading days before/after
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle


# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: dashboard_market_fast_research_panel
# script_name: code.py
# owner: Leon
# status: active
# layer: Dashboard
# domain: Market
# asset_type: Dashboard
# purpose: Fast visual dashboard for market research features and M15 chart context.
# inputs:
# - Data_Center/Data/3_Research/Market/daily_regime_features.csv
# - Data_Center/Data/1_Pipeline/Market/ohcl/M15/*.parquet
# outputs:
# - Dashboard UI
# dependencies:
# - tkinter
# - pathlib
# - pandas
# - numpy
# - matplotlib
# schedule: manual
# version: v1.2.0
# last_reviewed: 2026-06-05
# ============================================================


CODE_REGISTRY = {
    "script_id": "dashboard_market_fast_research_panel",
    "script_name": "code.py",
    "owner": "Leon",
    "status": "active",
    "layer": "Dashboard",
    "domain": "Market",
    "asset_type": "Dashboard",
    "purpose": "Fast visual dashboard for market research features and M15 chart context.",
    "inputs": [
        "Data_Center/Data/3_Research/Market/daily_regime_features.csv",
        "Data_Center/Data/1_Pipeline/Market/ohcl/M15/*.parquet",
    ],
    "outputs": ["Dashboard UI"],
    "dependencies": ["tkinter", "pathlib", "pandas", "numpy", "matplotlib"],
    "schedule": "manual",
    "version": "v1.2.0",
    "last_reviewed": "2026-06-05",
}


# ============================================================
# SETTINGS
# ============================================================

MAX_TABLE_ROWS_ALL = 120
MAX_TABLE_ROWS_SYMBOL = 260
MINI_CHART_WIDTH = 300
MINI_CHART_HEIGHT = 84
MINI_CHART_BATCH_SIZE = 8


# ============================================================
# TERMINAL THEME
# ============================================================

BG = "#000000"
PANEL_BG = "#0A0A0A"
PANEL_LIGHT = "#151515"
BORDER = "#2A2A2A"
ORANGE = "#FF9900"
YELLOW = "#FFD400"
GREEN = "#00FF66"
RED = "#FF4444"
WHITE = "#FFFFFF"
GREY = "#888888"

FONT_TITLE = ("Consolas", 15, "bold")
FONT_HEAD = ("Consolas", 12, "bold")
FONT_SMALL = ("Consolas", 11, "bold")
FONT_TINY = ("Consolas", 10, "bold")


# ============================================================
# FEATURE VIEWS
# ============================================================

VIEW_COLUMNS: Dict[str, List[str]] = {
    "OVERVIEW": [
        "date", "symbol",
        "trend_efficiency", "noise_ratio", "daily_range", "path_efficiency",
        "direction_changes", "false_breakout_ratio",
        "largest_candle_pct_of_day", "top_3_candles_share",
        "trend_intensity_score", "spike_intensity_score", "chaos_score",
    ],
    "TREND": [
        "date", "symbol",
        "trend_intensity_score",
        "trend_efficiency", "path_efficiency",
        "net_move", "path_length",
        "up_run_count", "down_run_count", "total_run_count",
        "avg_up_run_length", "max_up_run_length",
        "avg_down_run_length", "max_down_run_length",
    ],
    "PULLBACK": [
        "date", "symbol",
        "pullback_intensity_score",
        "pullback_count", "pullback_ratio",
        "direction_changes",
        "avg_up_run_length", "avg_down_run_length",
        "noise_ratio", "path_efficiency",
    ],
    "BREAKOUT": [
        "date", "symbol",
        "breakout_intensity_score",
        "new_high_count", "new_low_count",
        "high_break_acceptance", "low_break_acceptance",
        "failed_high_breakouts", "failed_low_breakouts",
        "false_breakout_ratio",
    ],
    "CANDLE": [
        "date", "symbol",
        "candle_intensity_score",
        "avg_body_size", "wick_ratio",
        "avg_upper_wick", "avg_lower_wick",
        "max_upper_wick", "max_lower_wick",
        "daily_range",
    ],
    "SPIKE": [
        "date", "symbol",
        "spike_intensity_score",
        "largest_candle", "largest_candle_pct_of_day",
        "top_3_candles_share", "atr_spike_count",
        "daily_range", "wick_ratio",
        "max_upper_wick", "max_lower_wick",
    ],
    "CHAOS": [
        "date", "symbol",
        "chaos_score",
        "noise_ratio", "direction_changes",
        "pullback_count", "path_length",
        "path_efficiency", "trend_efficiency",
        "false_breakout_ratio",
    ],
}


# ============================================================
# PATHS
# ============================================================

def find_quant_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p
    raise RuntimeError(f"QUANT root not found from: {start}")


# ============================================================
# DASHBOARD
# ============================================================

class MarketDashboard(tk.Frame):
    def __init__(self, parent, repository=None, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)
        self.repository = repository

        self.quant_root = find_quant_root(Path(__file__))
        self.data_path = (
            self.quant_root
            / "Data_Center"
            / "Data"
            / "3_Research"
            / "Market"
            / "daily_regime_features.csv"
        )
        self.ohlc_root = (
            self.quant_root
            / "Data_Center"
            / "Data"
            / "1_Pipeline"
            / "Market"
            / "ohcl"
            / "M15"
        )

        self.df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()
        self.visible_df = pd.DataFrame()

        self.search_var = tk.StringVar()
        self.symbol_var = tk.StringVar(value="ALL")
        self.view_var = tk.StringVar(value="OVERVIEW")
        self.status_var = tk.StringVar(value="NO DATA")

        self.kpi_labels: Dict[str, tk.Label] = {}
        self.view_buttons: Dict[str, tk.Button] = {}

        self._row_images: List[tk.PhotoImage] = []
        self._blank_image = tk.PhotoImage(width=MINI_CHART_WIDTH, height=MINI_CHART_HEIGHT)
        self._blank_image.put(BG, to=(0, 0, MINI_CHART_WIDTH, MINI_CHART_HEIGHT))

        self._symbol_ohlc_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self._mini_chart_cache: Dict[str, tk.PhotoImage] = {}
        self._pending_chart_items: List[Tuple[str, str, str]] = []

        self.build_ui()
        self.refresh_data()

    # ========================================================
    # UI BUILD
    # ========================================================

    def build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        self.build_header()
        self.build_view_buttons()
        self.build_kpis()
        self.build_filters()
        self.build_main_section()

    def panel(self, parent) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=PANEL_BG,
            highlightbackground=BORDER,
            highlightthickness=1,
            relief="solid",
            bd=0,
        )

    def build_header(self) -> None:
        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 2))
        header.columnconfigure(1, weight=1)

        tk.Label(
            header,
            text="QUANT TERMINAL  |  MARKET RESEARCH FAST",
            bg=BG,
            fg=ORANGE,
            font=FONT_TITLE,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            header,
            textvariable=self.status_var,
            bg=BG,
            fg=YELLOW,
            font=FONT_SMALL,
        ).grid(row=0, column=1, sticky="e", padx=10)

        tk.Button(
            header,
            text="REFRESH",
            command=self.refresh_data,
            bg=PANEL_LIGHT,
            fg=ORANGE,
            activebackground="#332000",
            activeforeground=YELLOW,
            font=FONT_SMALL,
            relief="solid",
            bd=1,
            padx=10,
        ).grid(row=0, column=2, sticky="e")

    def build_view_buttons(self) -> None:
        bar = tk.Frame(self, bg=BG)
        bar.grid(row=1, column=0, sticky="ew", padx=6, pady=2)

        for i, view in enumerate(VIEW_COLUMNS.keys()):
            btn = tk.Button(
                bar,
                text=view,
                command=lambda v=view: self.set_view(v),
                bg=PANEL_LIGHT,
                fg=ORANGE,
                activebackground="#332000",
                activeforeground=YELLOW,
                font=FONT_SMALL,
                relief="solid",
                bd=1,
                padx=12,
                pady=2,
            )
            btn.grid(row=0, column=i, padx=2, sticky="ew")
            bar.columnconfigure(i, weight=1)
            self.view_buttons[view] = btn

    def build_kpis(self) -> None:
        kpi = tk.Frame(self, bg=BG)
        kpi.grid(row=2, column=0, sticky="ew", padx=6, pady=2)

        names = [
            "ROWS",
            "VISIBLE",
            "SYMBOLS",
            "LAST DATE",
            "TREND SCORE",
            "SPIKE SCORE",
            "CHAOS SCORE",
            "MAX RANGE",
        ]

        for i, name in enumerate(names):
            card = self.panel(kpi)
            card.grid(row=0, column=i, sticky="ew", padx=2)
            kpi.columnconfigure(i, weight=1)

            tk.Label(card, text=name, bg=PANEL_BG, fg=ORANGE, font=FONT_TINY).pack(
                anchor="w", padx=6, pady=(4, 0)
            )
            val = tk.Label(card, text="--", bg=PANEL_BG, fg=WHITE, font=FONT_HEAD)
            val.pack(anchor="w", padx=6, pady=(0, 4))
            self.kpi_labels[name] = val

    def build_filters(self) -> None:
        filters = self.panel(self)
        filters.grid(row=3, column=0, sticky="ew", padx=6, pady=2)
        filters.columnconfigure(5, weight=1)

        tk.Label(filters, text="SEARCH", bg=PANEL_BG, fg=ORANGE, font=FONT_SMALL).grid(
            row=0, column=0, padx=6, pady=6
        )

        search = tk.Entry(
            filters,
            textvariable=self.search_var,
            bg=BG,
            fg=WHITE,
            insertbackground=WHITE,
            font=FONT_SMALL,
            relief="solid",
            bd=1,
            width=24,
        )
        search.grid(row=0, column=1, padx=4, pady=6)
        search.bind("<KeyRelease>", lambda e: self.apply_filters())

        tk.Label(filters, text="SYMBOL", bg=PANEL_BG, fg=ORANGE, font=FONT_SMALL).grid(
            row=0, column=2, padx=6
        )

        self.symbol_box = ttk.Combobox(
            filters,
            textvariable=self.symbol_var,
            state="readonly",
            width=12,
            font=FONT_SMALL,
        )
        self.symbol_box.grid(row=0, column=3, padx=4)
        self.symbol_box.bind("<<ComboboxSelected>>", lambda e: self.apply_filters())

        tk.Label(filters, text="VIEW", bg=PANEL_BG, fg=ORANGE, font=FONT_SMALL).grid(
            row=0, column=4, padx=6
        )

        self.view_box = ttk.Combobox(
            filters,
            textvariable=self.view_var,
            state="readonly",
            width=16,
            font=FONT_SMALL,
            values=list(VIEW_COLUMNS.keys()),
        )
        self.view_box.grid(row=0, column=5, sticky="w", padx=4)
        self.view_box.bind("<<ComboboxSelected>>", lambda e: self.set_view(self.view_var.get()))

    def build_main_section(self) -> None:
        main = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG, sashwidth=4)
        main.grid(row=4, column=0, sticky="nsew", padx=6, pady=(2, 6))

        left = self.panel(main)
        center = self.panel(main)
        right = self.panel(main)

        main.add(left, width=180)
        main.add(center, stretch="always")
        main.add(right, width=370)

        self.build_navigation(left)
        self.build_table(center)
        self.build_details(right)

    def build_navigation(self, parent) -> None:
        tk.Label(parent, text="FEATURE GROUPS", bg=PANEL_BG, fg=ORANGE, font=FONT_HEAD).pack(
            anchor="w", padx=8, pady=(8, 6)
        )

        for view in VIEW_COLUMNS:
            lab = tk.Label(
                parent,
                text=view,
                bg=BG,
                fg=WHITE,
                font=FONT_SMALL,
                anchor="w",
                padx=8,
            )
            lab.pack(fill="x", padx=6, pady=1)
            lab.bind("<Button-1>", lambda e, v=view: self.set_view(v))

        text = (
            "FAST MODE\n\n"
            "Table loads first.\n"
            "Mini charts load lazily.\n\n"
            "ALL mode shows limited\n"
            "latest rows for speed.\n\n"
            "Filter one symbol for\n"
            "deeper view."
        )
        tk.Label(
            parent,
            text=text,
            bg=PANEL_BG,
            fg=GREY,
            font=FONT_TINY,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=8, pady=12)

    def build_table(self, parent) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        self.table_title = tk.Label(
            parent,
            text="MARKET FEATURE VIEW: OVERVIEW",
            bg=PANEL_BG,
            fg=ORANGE,
            font=FONT_HEAD,
        )
        self.table_title.grid(row=0, column=0, sticky="w", padx=8, pady=6)

        table_frame = tk.Frame(parent, bg=BG)
        table_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_frame, columns=[], show="tree headings", height=18)

        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)

        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")

        self.tree.bind("<<TreeviewSelect>>", self.update_details)
        self.tree.bind("<Double-1>", self.open_chart_popup)

        self.style_tree()

    def style_tree(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=BG,
            foreground=WHITE,
            fieldbackground=BG,
            rowheight=96,
            font=("Consolas", 10, "bold"),
            bordercolor=BORDER,
            borderwidth=1,
            padding=(4, 10),
        )
        style.configure(
            "Treeview.Heading",
            background="#111111",
            foreground=ORANGE,
            font=("Consolas", 10, "bold"),
            relief="solid",
        )
        style.map(
            "Treeview",
            background=[("selected", "#332000")],
            foreground=[("selected", YELLOW)],
        )

    def build_details(self, parent) -> None:
        parent.rowconfigure(1, weight=1)
        parent.columnconfigure(0, weight=1)

        tk.Label(parent, text="FEATURE DETAILS", bg=PANEL_BG, fg=ORANGE, font=FONT_HEAD).grid(
            row=0, column=0, sticky="w", padx=8, pady=6
        )

        self.details = tk.Text(
            parent,
            bg=BG,
            fg=WHITE,
            insertbackground=WHITE,
            font=FONT_SMALL,
            relief="solid",
            bd=1,
            wrap="none",
            height=20,
        )
        self.details.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 6))

    # ========================================================
    # DATA LOAD
    # ========================================================

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Missing file: {self.data_path}")

        df = pd.read_csv(self.data_path)

        if df.empty:
            raise ValueError("daily_regime_features.csv is empty")

        required = ["date", "symbol", "trend_efficiency", "noise_ratio"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df["symbol"] = df["symbol"].astype(str).str.upper()
        df["date"] = df["date"].astype(str)

        df = self.add_intensity_scores(df)
        df["_date_sort"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values(["_date_sort", "symbol"], ascending=[False, True]).reset_index(drop=True)

        return df

    def add_intensity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        def col(name: str, default: float = 0.0) -> pd.Series:
            if name not in d.columns:
                return pd.Series(default, index=d.index)
            return pd.to_numeric(d[name], errors="coerce").fillna(default)

        def minmax(s: pd.Series) -> pd.Series:
            s = pd.to_numeric(s, errors="coerce").fillna(0.0)
            lo = float(s.min())
            hi = float(s.max())
            if hi - lo == 0:
                return pd.Series(0.0, index=s.index)
            return (s - lo) / (hi - lo)

        trend_eff = col("trend_efficiency").clip(0, 1)
        path_eff = col("path_efficiency").clip(0, 1)

        d["trend_intensity_score"] = (
            0.40 * trend_eff
            + 0.30 * path_eff
            + 0.15 * minmax(col("avg_up_run_length"))
            + 0.15 * minmax(col("avg_down_run_length"))
        ).clip(0, 1)

        pullback_count = minmax(col("pullback_count"))
        pullback_ratio = col("pullback_ratio").clip(0, 1)
        direction_changes = minmax(col("direction_changes"))
        noise = col("noise_ratio").clip(0, 1)

        d["pullback_intensity_score"] = (
            0.30 * pullback_count
            + 0.30 * pullback_ratio
            + 0.20 * direction_changes
            + 0.20 * noise
        ).clip(0, 1)

        d["breakout_intensity_score"] = (
            0.20 * minmax(col("new_high_count"))
            + 0.20 * minmax(col("new_low_count"))
            + 0.20 * col("high_break_acceptance").clip(0, 1)
            + 0.20 * col("low_break_acceptance").clip(0, 1)
            + 0.20 * col("false_breakout_ratio").clip(0, 1)
        ).clip(0, 1)

        d["candle_intensity_score"] = (
            0.35 * (col("wick_ratio").clip(0, 2) / 2)
            + 0.25 * minmax(col("avg_body_size"))
            + 0.20 * minmax(col("max_upper_wick"))
            + 0.20 * minmax(col("max_lower_wick"))
        ).clip(0, 1)

        d["spike_intensity_score"] = (
            0.40 * col("largest_candle_pct_of_day").clip(0, 1)
            + 0.30 * (col("top_3_candles_share").clip(0, 2) / 2)
            + 0.20 * minmax(col("atr_spike_count"))
            + 0.10 * minmax(col("daily_range"))
        ).clip(0, 1)

        d["chaos_score"] = (
            0.35 * noise
            + 0.25 * direction_changes
            + 0.20 * pullback_ratio
            + 0.20 * col("false_breakout_ratio").clip(0, 1)
        ).clip(0, 1)

        return d

    def refresh_data(self) -> None:
        try:
            self.status_var.set("LOADING DAILY FEATURES...")
            self.update_idletasks()

            self._symbol_ohlc_cache.clear()
            self._mini_chart_cache.clear()
            self._pending_chart_items.clear()

            self.df = self.load_data()
            self.filtered_df = self.df.copy()

            symbols = ["ALL"] + sorted(self.df["symbol"].dropna().unique().tolist())
            self.symbol_box["values"] = symbols
            self.symbol_var.set("ALL")

            self.set_view(self.view_var.get(), update_box=False)
            self.status_var.set(f"UPDATED {datetime.now().strftime('%H:%M:%S')}")

        except Exception as e:
            self.status_var.set("ERROR")
            self.details.delete("1.0", tk.END)
            self.details.insert(tk.END, f"ERROR LOADING MARKET DATA:\n\n{e}")

    # ========================================================
    # TABLE / VIEWS
    # ========================================================

    def set_view(self, view: str, update_box: bool = True) -> None:
        if view not in VIEW_COLUMNS:
            view = "OVERVIEW"

        self.view_var.set(view)
        if update_box:
            self.view_box.set(view)

        for k, btn in self.view_buttons.items():
            if k == view:
                btn.configure(bg="#332000", fg=YELLOW)
            else:
                btn.configure(bg=PANEL_LIGHT, fg=ORANGE)

        self.configure_tree_columns(VIEW_COLUMNS[view])
        self.table_title.configure(text=f"MARKET FEATURE VIEW: {view}")
        self.apply_filters()

    def configure_tree_columns(self, columns: List[str]) -> None:
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = columns

        self.tree.heading("#0", text="M15 MINI CHART")
        self.tree.column("#0", width=330, minwidth=310, stretch=False, anchor="center")

        for col_name in columns:
            self.tree.heading(col_name, text=col_name.upper(), command=lambda c=col_name: self.sort_by(c, False))

            width = 140
            if col_name == "date":
                width = 120
            elif col_name == "symbol":
                width = 105
            elif "score" in col_name:
                width = 175
            elif col_name in [
                "trend_efficiency",
                "noise_ratio",
                "path_efficiency",
                "daily_range",
                "direction_changes",
                "false_breakout_ratio",
                "largest_candle_pct_of_day",
                "top_3_candles_share",
            ]:
                width = 155
            elif col_name in ["path_length", "largest_candle", "net_move"]:
                width = 150

            self.tree.column(col_name, width=width, minwidth=95, anchor="center")

    def apply_filters(self) -> None:
        if self.df.empty:
            return

        df = self.df.copy()

        q = self.search_var.get().strip().lower()
        sym = self.symbol_var.get()

        if q:
            df = df[
                df.astype(str)
                .apply(lambda row: row.str.lower().str.contains(q, na=False).any(), axis=1)
            ]

        if sym and sym != "ALL":
            df = df[df["symbol"] == sym]
            max_rows = MAX_TABLE_ROWS_SYMBOL
        else:
            max_rows = MAX_TABLE_ROWS_ALL

        if "_date_sort" in df.columns:
            df = df.sort_values(["_date_sort", "symbol"], ascending=[False, True]).reset_index(drop=True)

        self.filtered_df = df
        self.visible_df = df.head(max_rows).copy()

        self.update_kpis(df)
        self.update_table(self.visible_df)

    def update_kpis(self, df: pd.DataFrame) -> None:
        def fmt(x):
            try:
                return f"{float(x):.3f}"
            except Exception:
                return "--"

        self.kpi_labels["ROWS"].config(text=f"{len(df):,}")
        self.kpi_labels["VISIBLE"].config(text=f"{len(self.visible_df):,}")
        self.kpi_labels["SYMBOLS"].config(text=str(df["symbol"].nunique()) if not df.empty else "0")
        self.kpi_labels["LAST DATE"].config(text=str(df["date"].max()) if not df.empty else "--")
        self.kpi_labels["TREND SCORE"].config(text=fmt(df["trend_intensity_score"].mean()) if "trend_intensity_score" in df else "--")
        self.kpi_labels["SPIKE SCORE"].config(text=fmt(df["spike_intensity_score"].mean()) if "spike_intensity_score" in df else "--")
        self.kpi_labels["CHAOS SCORE"].config(text=fmt(df["chaos_score"].mean()) if "chaos_score" in df else "--")
        self.kpi_labels["MAX RANGE"].config(text=fmt(df["daily_range"].max()) if "daily_range" in df else "--")

    def update_table(self, df: pd.DataFrame) -> None:
        self.tree.delete(*self.tree.get_children())
        self._row_images = []
        self._pending_chart_items = []

        columns = list(self.tree["columns"])
        if not columns:
            return

        view = self.view_var.get()

        for _, row in df.iterrows():
            values = []
            for c in columns:
                val = row.get(c, "")
                if isinstance(val, (float, np.floating)):
                    val = f"{float(val):.4f}"
                values.append(val)

            tag = self.row_tag_for_view(row, view)

            item_id = self.tree.insert(
                "",
                "end",
                text="",
                image=self._blank_image,
                values=values,
                tags=(tag,),
            )

            symbol = str(row.get("symbol", "")).upper()
            date = str(row.get("date", ""))
            self._pending_chart_items.append((item_id, symbol, date))

        self.tree.tag_configure("LOW", foreground=GREEN, font=("Consolas", 10, "bold"))
        self.tree.tag_configure("MID", foreground=YELLOW, font=("Consolas", 10, "bold"))
        self.tree.tag_configure("HIGH", foreground=RED, font=("Consolas", 10, "bold"))
        self.tree.tag_configure("BASE", foreground=WHITE, font=("Consolas", 10, "bold"))

        self.after(30, self.process_mini_chart_batch)

    def row_tag_for_view(self, row: pd.Series, view: str) -> str:
        score_col = {
            "TREND": "trend_intensity_score",
            "PULLBACK": "pullback_intensity_score",
            "BREAKOUT": "breakout_intensity_score",
            "CANDLE": "candle_intensity_score",
            "SPIKE": "spike_intensity_score",
            "CHAOS": "chaos_score",
            "OVERVIEW": "spike_intensity_score",
        }.get(view, "")

        try:
            score = float(row.get(score_col, 0))
        except Exception:
            score = 0

        if score >= 0.70:
            return "HIGH"
        if score >= 0.40:
            return "MID"
        return "LOW"

    def update_details(self, event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return

        item = self.tree.item(selected[0])
        values = item["values"]
        columns = self.tree["columns"]
        data = dict(zip(columns, values))
        view = self.view_var.get()

        self.details.delete("1.0", tk.END)

        lines = []
        lines.append(f"{view} FEATURE DETAIL")
        lines.append("=" * 46)
        lines.append("")
        for k, v in data.items():
            lines.append(f"{k:<32}: {v}")

        lines.append("")
        lines.append("ACTION")
        lines.append("=" * 46)
        lines.append("Double-click row to open full M15 chart with 2 trading days before/after.")

        self.details.insert(tk.END, "\n".join(lines))

    def sort_by(self, col_name: str, descending: bool) -> None:
        if self.filtered_df.empty or col_name not in self.filtered_df.columns:
            return

        try:
            df = self.filtered_df.sort_values(col_name, ascending=not descending)
        except Exception:
            df = self.filtered_df.sort_values(col_name, ascending=not descending, key=lambda s: s.astype(str))

        self.filtered_df = df.reset_index(drop=True)
        max_rows = MAX_TABLE_ROWS_SYMBOL if self.symbol_var.get() != "ALL" else MAX_TABLE_ROWS_ALL
        self.visible_df = self.filtered_df.head(max_rows).copy()

        self.update_table(self.visible_df)
        self.tree.heading(col_name, command=lambda: self.sort_by(col_name, not descending))

    # ========================================================
    # MINI CHARTS - FAST LAZY LOADING
    # ========================================================

    def symbol_to_file_symbol(self, symbol: str) -> str:
        mapping = {
            "US500": "US500.cash",
            "USOIL": "USOIL.cash",
        }
        return mapping.get(symbol.upper(), symbol.upper())

    def load_symbol_ohlc_cached(self, symbol: str) -> Optional[pd.DataFrame]:
        symbol = symbol.upper()
        if symbol in self._symbol_ohlc_cache:
            return self._symbol_ohlc_cache[symbol]

        file_symbol = self.symbol_to_file_symbol(symbol)
        path = self.ohlc_root / f"{file_symbol}.parquet"

        if not path.exists():
            alt = self.ohlc_root / f"{symbol}.parquet"
            if alt.exists():
                path = alt

        if not path.exists():
            self._symbol_ohlc_cache[symbol] = None
            return None

        try:
            df = pd.read_parquet(path, columns=["time", "open", "high", "low", "close"])
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            df = df.dropna(subset=["time", "open", "high", "low", "close"])
            df = df.sort_values("time").reset_index(drop=True)
            df["date"] = df["time"].dt.date.astype(str)

            self._symbol_ohlc_cache[symbol] = df
            return df

        except Exception:
            self._symbol_ohlc_cache[symbol] = None
            return None

    def process_mini_chart_batch(self) -> None:
        if not self._pending_chart_items:
            return

        batch = self._pending_chart_items[:MINI_CHART_BATCH_SIZE]
        self._pending_chart_items = self._pending_chart_items[MINI_CHART_BATCH_SIZE:]

        for item_id, symbol, date in batch:
            if not self.tree.exists(item_id):
                continue

            cache_key = f"{symbol}|{date}"
            if cache_key in self._mini_chart_cache:
                img = self._mini_chart_cache[cache_key]
            else:
                img = self.create_intraday_mini_chart(symbol, date)
                self._mini_chart_cache[cache_key] = img

            self._row_images.append(img)
            self.tree.item(item_id, image=img)

        if self._pending_chart_items:
            self.after(20, self.process_mini_chart_batch)
        else:
            self.status_var.set(f"MINI CHARTS READY {datetime.now().strftime('%H:%M:%S')}")

    def create_intraday_mini_chart(self, symbol: str, date: str) -> tk.PhotoImage:
        img = tk.PhotoImage(width=MINI_CHART_WIDTH, height=MINI_CHART_HEIGHT)
        img.put(BG, to=(0, 0, MINI_CHART_WIDTH, MINI_CHART_HEIGHT))

        df = self.load_symbol_ohlc_cached(symbol)
        if df is None or df.empty:
            return img

        day = df[df["date"] == str(date)].copy()
        if day.empty:
            return img

        day = day.sort_values("time").reset_index(drop=True)

        try:
            price_high = float(day["high"].max())
            price_low = float(day["low"].min())
            if price_high <= price_low:
                return img

            left = 5
            right = MINI_CHART_WIDTH - 5
            top = 4
            bottom = MINI_CHART_HEIGHT - 5

            n = len(day)
            if n <= 1:
                return img

            grid_color = "#101010"

            for gx in range(left, right + 1, max(1, (right - left) // 4)):
                for y in range(top, bottom + 1):
                    img.put(grid_color, (gx, y))

            for gy in range(top, bottom + 1, max(1, (bottom - top) // 3)):
                for x in range(left, right + 1):
                    img.put(grid_color, (x, gy))

            def y_pos(price: float) -> int:
                y = bottom - (price - price_low) / (price_high - price_low) * (bottom - top)
                return int(max(top, min(bottom, round(y))))

            x_step = (right - left) / max(n - 1, 1)
            body_w = max(1, int(x_step * 0.55))

            for i, candle in day.iterrows():
                o = float(candle["open"])
                h = float(candle["high"])
                l = float(candle["low"])
                c = float(candle["close"])

                x = int(round(left + i * x_step))
                y_o = y_pos(o)
                y_h = y_pos(h)
                y_l = y_pos(l)
                y_c = y_pos(c)

                color = GREEN if c >= o else RED

                for yy in range(min(y_h, y_l), max(y_h, y_l) + 1):
                    if 0 <= x < MINI_CHART_WIDTH and 0 <= yy < MINI_CHART_HEIGHT:
                        img.put(color, (x, yy))

                y1 = min(y_o, y_c)
                y2 = max(y_o, y_c)
                if y2 - y1 < 2:
                    y2 = y1 + 2

                x1 = max(left, x - body_w // 2)
                x2 = min(right, x + body_w // 2)

                for yy in range(y1, y2 + 1):
                    for xx in range(x1, x2 + 1):
                        if 0 <= xx < MINI_CHART_WIDTH and 0 <= yy < MINI_CHART_HEIGHT:
                            img.put(color, (xx, yy))

            return img

        except Exception:
            return img

    # ========================================================
    # FULL POPUP CHART
    # ========================================================

    def load_intraday_ohlc_context(self, symbol: str, date: str, days_before: int = 2, days_after: int = 2) -> pd.DataFrame:
        df = self.load_symbol_ohlc_cached(symbol)
        if df is None or df.empty:
            raise FileNotFoundError(f"OHLC file not found for symbol: {symbol}")

        target = pd.to_datetime(date).date()
        dates_as_date = pd.to_datetime(df["date"], errors="coerce").dt.date
        df2 = df.copy()
        df2["date_obj"] = dates_as_date

        available_dates = sorted(df2["date_obj"].dropna().unique().tolist())
        if target not in available_dates:
            raise ValueError(f"Selected date not found in M15 data: {symbol} {date}")

        target_idx = available_dates.index(target)
        start_idx = max(0, target_idx - days_before)
        end_idx = min(len(available_dates) - 1, target_idx + days_after)

        selected_dates = set(available_dates[start_idx:end_idx + 1])
        context = df2[df2["date_obj"].isin(selected_dates)].copy()

        if context.empty:
            raise ValueError(f"No M15 candles found for {symbol} around {date}")

        context["is_selected_day"] = context["date_obj"] == target
        context["date_str"] = context["date_obj"].astype(str)
        return context.reset_index(drop=True)

    def open_chart_popup(self, event=None) -> None:
        selected = self.tree.selection()
        if not selected:
            return

        item = self.tree.item(selected[0])
        values = item["values"]
        columns = self.tree["columns"]
        row = dict(zip(columns, values))

        symbol = str(row.get("symbol", "")).upper()
        date = str(row.get("date", ""))

        try:
            df = self.load_intraday_ohlc_context(symbol, date, days_before=2, days_after=2)
        except Exception as e:
            self.details.delete("1.0", tk.END)
            self.details.insert(tk.END, f"CHART ERROR:\n\n{e}")
            return

        popup = tk.Toplevel(self)
        popup.title(f"M15 CONTEXT CHART | {symbol} | {date}")
        popup.configure(bg=BG)
        popup.geometry("1450x800")

        tk.Label(
            popup,
            text=f"M15 CONTEXT CHART  |  {symbol}  |  SELECTED DAY: {date}  |  2 TRADING DAYS BEFORE/AFTER",
            bg=BG,
            fg=ORANGE,
            font=FONT_TITLE,
        ).pack(fill="x", padx=8, pady=6)

        fig = Figure(figsize=(14.5, 7.4), dpi=100)
        ax = fig.add_subplot(111)

        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        self.draw_candlestick_chart(ax, df, selected_date=date)

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

    def draw_candlestick_chart(self, ax, df: pd.DataFrame, selected_date: str) -> None:
        candle_width = 0.65

        unique_dates = df["date_str"].drop_duplicates().tolist() if "date_str" in df.columns else []
        for day in unique_dates:
            idxs = df.index[df["date_str"] == day].tolist()
            if not idxs:
                continue
            start_i = min(idxs) - 0.5
            end_i = max(idxs) + 0.5

            if str(day) == str(selected_date):
                ax.axvspan(start_i, end_i, color="#332000", alpha=0.55)
            else:
                ax.axvspan(start_i, end_i, color="#050505", alpha=0.35)

        for i, row in df.iterrows():
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])

            color = GREEN if c >= o else RED
            selected = bool(row.get("is_selected_day", False))

            line_width = 1.15 if selected else 0.75
            edge_width = 1.00 if selected else 0.60
            alpha = 1.0 if selected else 0.60

            ax.plot([i, i], [l, h], color=color, linewidth=line_width, alpha=alpha)

            body_low = min(o, c)
            body_height = abs(c - o)
            if body_height == 0:
                body_height = max((h - l) * 0.02, 1e-9)

            rect = Rectangle(
                (i - candle_width / 2, body_low),
                candle_width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=edge_width,
                alpha=alpha,
            )
            ax.add_patch(rect)

        ax.set_title(
            f"15M OHLC | highlighted day = {selected_date} | context = 2 trading days before/after",
            color=ORANGE,
            fontsize=10,
            fontname="Consolas",
        )

        ax.tick_params(axis="x", colors=WHITE, labelsize=7)
        ax.tick_params(axis="y", colors=WHITE, labelsize=8)

        for spine in ax.spines.values():
            spine.set_color(BORDER)

        ax.grid(True, color="#1A1A1A", linewidth=0.5)

        if "date_str" in df.columns:
            seen = []
            for day in df["date_str"].tolist():
                if day not in seen:
                    seen.append(day)

            y_top = df["high"].max()
            y_bottom = df["low"].min()
            y_label = y_top - (y_top - y_bottom) * 0.03

            for day in seen:
                idxs = df.index[df["date_str"] == day].tolist()
                if not idxs:
                    continue

                start_i = min(idxs)
                end_i = max(idxs)
                mid_i = (start_i + end_i) / 2

                ax.axvline(start_i - 0.5, color=BORDER, linewidth=0.8, linestyle="--")

                label_color = YELLOW if str(day) == str(selected_date) else GREY
                ax.text(
                    mid_i,
                    y_label,
                    str(day),
                    color=label_color,
                    fontsize=7,
                    fontname="Consolas",
                    ha="center",
                    va="top",
                )

        step = max(1, len(df) // 20)
        x_ticks = list(range(0, len(df), step))
        x_labels = [
            pd.to_datetime(df["time"].iloc[i]).strftime("%m-%d %H:%M")
            for i in x_ticks
        ]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlim(-1, len(df) + 1)


# ============================================================
# REQUIRED API
# ============================================================

def build_panel(parent, repository=None, **kwargs):
    return MarketDashboard(parent, repository=repository, **kwargs)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("QUANT TERMINAL | MARKET")
    root.configure(bg=BG)
    root.geometry("1600x900")

    app = build_panel(root)
    app.pack(fill="both", expand=True)

    root.mainloop()

"""
QUANT/Dashboard/Building_Blocks/Market/Seasonality/code.py

Market Seasonality Dashboard Building Block

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: dashboard_market_seasonality
# script_name: Market Seasonality Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Market Analytics
# asset_type: Dashboard
# purpose: OHLC-based seasonality dashboard for monthly, weekday, hour-of-day and session behavior analysis across market symbols and timeframes.
# inputs:
#   - Data_Center/Data/1_Pipeline/Market/ohcl/**/*.parquet
#   - Data_Center/Data/1_Pipeline/Market/ohcl/summary.json
# outputs:
#   - Dashboard UI
# dependencies:
#   - tkinter
#   - ttk
#   - pathlib
#   - pandas
#   - numpy
#   - matplotlib
#   - json
# schedule: manual
# version: v1.0.1_main_compatible
# last_reviewed: 2026-06-07
# required_api:
#   - build_panel(parent, repository=None, **kwargs)
# expected_location:
#   - QUANT/Dashboard/Building_Blocks/Market/Seasonality/code.py
# data_source:
#   - Data_Center/Data/1_Pipeline/Market/ohcl
# scanner:
#   - kein Scanner
# ============================================================
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


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
YELLOW = "#FFD400"
CYAN = "#35FFE2"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_SUB = ("Consolas", 8)
FONT_NAV = ("Consolas", 8, "bold")
FONT_H2 = ("Consolas", 10, "bold")
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_KPI = ("Consolas", 16, "bold")

TIMEFRAMES_ORDER = ["M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SESSION_ORDER = ["ASIA", "EU", "US", "OTHER"]


def find_quant_root(start: Path) -> Path:
    start = Path(start).resolve()

    def is_quant_root(p: Path) -> bool:
        return (p / "Dashboard").exists() and (p / "Data_Center").exists()

    candidates = []
    candidates.extend([start] + list(start.parents))
    try:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd] + list(cwd.parents))
    except Exception:
        pass

    home = Path.home()
    for base in [
        home / "Desktop" / "Business_Code" / "QUANT",
        home / "Desktop" / "Business_Code",
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
    ]:
        candidates.append(base)
        candidates.append(base / "QUANT")

    seen = set()
    for c in candidates:
        try:
            key = str(c.resolve()).lower()
        except Exception:
            key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if is_quant_root(c):
                return c.resolve()
        except Exception:
            pass

    for root in [home / "Desktop" / "Business_Code", home / "Desktop"]:
        if not root.exists():
            continue
        try:
            for p in root.rglob("QUANT"):
                if p.is_dir() and is_quant_root(p):
                    return p.resolve()
        except Exception:
            pass

    raise RuntimeError(f"QUANT root not found. Start={start} | CWD={Path.cwd()}")


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)
OHLC_ROOT = QUANT_ROOT / "Data_Center" / "Data" / "1_Pipeline" / "Market" / "ohcl"


CODE_REGISTRY: Dict[str, object] = {
    "script_id": "dashboard_market_seasonality",
    "script_name": "Market Seasonality Dashboard",
    "owner": "Leon",
    "status": "active",
    "layer": "Dashboard",
    "domain": "Market Analytics",
    "asset_type": "Dashboard",
    "purpose": "OHLC-based seasonality analysis for monthly, weekday, hour and session behavior.",
    "inputs": [
        "Data_Center/Data/1_Pipeline/Market/ohcl/**/*.parquet",
        "Data_Center/Data/1_Pipeline/Market/ohcl/summary.json",
    ],
    "outputs": ["Dashboard UI"],
    "dependencies": ["tkinter", "ttk", "pathlib", "pandas", "numpy", "matplotlib", "json"],
    "schedule": "manual",
    "version": "v1.0.1_main_compatible",
    "last_reviewed": "2026-06-07",
    "expected_location": "QUANT/Dashboard/Building_Blocks/Market/Seasonality/code.py",
    "data_source": "Data_Center/Data/1_Pipeline/Market/ohcl",
    "scanner": "kein Scanner",
    "required_api": "build_panel(parent, repository=None, **kwargs)",
}


def get_code_registry() -> Dict[str, object]:
    return dict(CODE_REGISTRY)


def classify_session(hour: int) -> str:
    try:
        h = int(hour)
    except Exception:
        return "OTHER"
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 13:
        return "EU"
    if 13 <= h < 21:
        return "US"
    return "OTHER"


class OHLCRepository:
    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root) if root else OHLC_ROOT

    def discover_timeframes(self) -> List[str]:
        if not self.root.exists():
            return []
        found = [p.name for p in self.root.iterdir() if p.is_dir()]
        return sorted(found, key=lambda x: TIMEFRAMES_ORDER.index(x) if x in TIMEFRAMES_ORDER else 999)

    def discover_symbols(self, timeframe: str) -> List[str]:
        folder = self.root / timeframe
        if not folder.exists():
            return []
        return [p.stem for p in sorted(folder.glob("*.parquet"))]

    def load_ohlc(self, timeframe: str, symbol: str) -> pd.DataFrame:
        path = self.root / timeframe / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"OHLC file not found: {path}")
        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"OHLC file is empty: {path}")
        return self.normalize_ohlc(df)

    @staticmethod
    def normalize_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        lower_map = {str(c).lower().strip(): c for c in d.columns}

        time_col = None
        for c in ["time", "timestamp", "datetime", "date", "open_time", "time_utc", "open_time_utc"]:
            if c in lower_map:
                time_col = lower_map[c]
                break

        if time_col is None:
            for c in d.columns:
                if "time" in str(c).lower() or "date" in str(c).lower():
                    time_col = c
                    break

        if time_col is None:
            d = d.reset_index()
            time_col = d.columns[0]

        d["timestamp"] = pd.to_datetime(d[time_col], errors="coerce", utc=True)
        d = d.dropna(subset=["timestamp"]).sort_values("timestamp")

        aliases = {
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c", "last"],
            "volume": ["volume", "vol", "tick_volume", "real_volume"],
        }

        for target, names in aliases.items():
            source = None
            for name in names:
                if name in lower_map:
                    source = lower_map[name]
                    break
            if source is None:
                if target == "volume":
                    d[target] = 0.0
                    continue
                raise ValueError(f"Required column missing: {target}. Available columns: {list(df.columns)}")
            d[target] = pd.to_numeric(d[source], errors="coerce")

        d = d.dropna(subset=["open", "high", "low", "close"])
        d["return"] = d["close"].pct_change()
        d["range_pct"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
        d["body_pct"] = (d["close"] - d["open"]) / d["open"].replace(0, np.nan)
        d["direction"] = np.where(d["close"] >= d["open"], "UP", "DOWN")
        d["year"] = d["timestamp"].dt.year
        d["month"] = d["timestamp"].dt.month
        d["month_name"] = d["month"].map(lambda x: MONTH_NAMES[int(x)-1] if pd.notna(x) else "")
        d["weekday"] = d["timestamp"].dt.weekday
        d["weekday_name"] = d["weekday"].map(lambda x: WEEKDAY_NAMES[int(x)] if pd.notna(x) else "")
        d["hour"] = d["timestamp"].dt.hour
        d["session"] = d["hour"].apply(classify_session)
        return d.reset_index(drop=True)


class SeasonalityEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().replace([np.inf, -np.inf], np.nan)

    def _agg(self, group_col: str, order: Optional[List[object]] = None) -> pd.DataFrame:
        d = self.df.copy()
        if d.empty:
            return pd.DataFrame()

        g = d.groupby(group_col, dropna=False).agg(
            bars=("close", "count"),
            avg_return=("return", "mean"),
            median_return=("return", "median"),
            total_return=("return", lambda x: (1.0 + x.dropna()).prod() - 1.0 if len(x.dropna()) else np.nan),
            winrate=("return", lambda x: float((x.dropna() > 0).mean()) if len(x.dropna()) else np.nan),
            avg_range=("range_pct", "mean"),
            avg_body=("body_pct", "mean"),
            up_bars=("direction", lambda x: int((x == "UP").sum())),
            down_bars=("direction", lambda x: int((x == "DOWN").sum())),
        ).reset_index()

        g["avg_return_pct"] = g["avg_return"] * 100.0
        g["median_return_pct"] = g["median_return"] * 100.0
        g["total_return_pct"] = g["total_return"] * 100.0
        g["winrate_pct"] = g["winrate"] * 100.0
        g["avg_range_pct"] = g["avg_range"] * 100.0
        g["avg_body_pct"] = g["avg_body"] * 100.0

        if order is not None:
            order_map = {v: i for i, v in enumerate(order)}
            g["_order"] = g[group_col].map(order_map).fillna(9999)
            g = g.sort_values("_order").drop(columns=["_order"])
        else:
            g = g.sort_values(group_col)

        return g.reset_index(drop=True)

    def monthly(self) -> pd.DataFrame:
        return self._agg("month_name", MONTH_NAMES)

    def weekdays(self) -> pd.DataFrame:
        return self._agg("weekday_name", WEEKDAY_NAMES)

    def hours(self) -> pd.DataFrame:
        return self._agg("hour", list(range(24)))

    def sessions(self) -> pd.DataFrame:
        return self._agg("session", SESSION_ORDER)

    def year_month_matrix(self) -> pd.DataFrame:
        d = self.df.dropna(subset=["return"]).copy()
        if d.empty:
            return pd.DataFrame()
        piv = d.groupby(["year", "month"])["return"].apply(lambda x: (1 + x).prod() - 1).reset_index()
        mat = piv.pivot(index="year", columns="month", values="return").sort_index()
        mat = mat.reindex(columns=list(range(1, 13)))
        mat.columns = MONTH_NAMES
        return mat * 100.0

    def kpis(self) -> Dict[str, object]:
        d = self.df.dropna(subset=["return"]).copy()
        monthly = self.monthly()
        weekdays = self.weekdays()
        sessions = self.sessions()

        def best_label(frame: pd.DataFrame, label_col: str) -> str:
            if frame.empty or "avg_return_pct" not in frame.columns:
                return "-"
            row = frame.sort_values("avg_return_pct", ascending=False).head(1)
            return str(row.iloc[0][label_col]) if not row.empty else "-"

        def worst_label(frame: pd.DataFrame, label_col: str) -> str:
            if frame.empty or "avg_return_pct" not in frame.columns:
                return "-"
            row = frame.sort_values("avg_return_pct", ascending=True).head(1)
            return str(row.iloc[0][label_col]) if not row.empty else "-"

        return {
            "bars": int(len(self.df)),
            "avg_return": float(d["return"].mean() * 100.0) if not d.empty else np.nan,
            "winrate": float((d["return"] > 0).mean() * 100.0) if not d.empty else np.nan,
            "avg_range": float(self.df["range_pct"].mean() * 100.0) if "range_pct" in self.df else np.nan,
            "best_month": best_label(monthly, "month_name"),
            "worst_month": worst_label(monthly, "month_name"),
            "best_weekday": best_label(weekdays, "weekday_name"),
            "best_session": best_label(sessions, "session"),
        }


class Card(tk.Frame):
    def __init__(self, parent, bg: str = CARD, padx: int = 12, pady: int = 10):
        super().__init__(parent, bg=bg, highlightthickness=1, highlightbackground=BORDER, bd=0)
        self.inner = tk.Frame(self, bg=bg)
        self.inner.pack(fill="both", expand=True, padx=padx, pady=pady)


class NavButton(tk.Label):
    def __init__(self, parent, text: str, active: bool = False):
        super().__init__(
            parent, text=text, bg=BG, fg=GREEN if active else MUTED,
            font=FONT_NAV, padx=8, pady=7, cursor="hand2"
        )


class KpiCard(Card):
    def __init__(self, parent, title: str):
        super().__init__(parent, bg=CARD, padx=12, pady=9)
        self.value_var = tk.StringVar(value="-")
        self.sub_var = tk.StringVar(value="")
        tk.Label(self.inner, text=title, bg=CARD, fg=MUTED, font=FONT_SMALL).pack(anchor="w")
        self.value_label = tk.Label(self.inner, textvariable=self.value_var, bg=CARD, fg=FG, font=FONT_KPI)
        self.value_label.pack(anchor="w", pady=(2, 0))
        tk.Label(self.inner, textvariable=self.sub_var, bg=CARD, fg=MUTED, font=FONT_TINY).pack(anchor="w")

    def set(self, value: str, sub: str = "", color: str = FG) -> None:
        self.value_var.set(value)
        self.sub_var.set(sub)
        self.value_label.configure(fg=color)


class ChartCard(Card):
    def __init__(self, parent, title: str, height: float = 2.4):
        super().__init__(parent, bg=CARD, padx=12, pady=9)
        tk.Label(self.inner, text=title, bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.fig = Figure(figsize=(5.2, height), dpi=100)
        self.fig.patch.set_facecolor(CARD)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.inner)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(8, 0))

    def reset(self) -> None:
        self.ax.clear()
        self.fig.patch.set_facecolor(CARD)
        self.ax.set_facecolor(CARD)


def style_ax(ax) -> None:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(CARD)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def fmt_pct(x: object, decimals: int = 2) -> str:
    try:
        if pd.isna(x):
            return "-"
        return f"{float(x):.{decimals}f}%"
    except Exception:
        return "-"


def value_color(x: object) -> str:
    try:
        v = float(x)
    except Exception:
        return FG
    if v > 0:
        return GREEN
    if v < 0:
        return RED
    return FG


def clear_children(widget: tk.Widget) -> None:
    for child in widget.winfo_children():
        child.destroy()


class SeasonalityDashboard(tk.Frame):
    def __init__(self, parent, repository: Optional[OHLCRepository] = None, **_kwargs):
        super().__init__(parent, bg=BG)
        self.repository = repository or OHLCRepository()

        self.timeframe_var = tk.StringVar(value="D1")
        self.symbol_var = tk.StringVar(value="")
        self.metric_var = tk.StringVar(value="Avg Return")

        self.df = pd.DataFrame()
        self.monthly_df = pd.DataFrame()
        self.weekday_df = pd.DataFrame()
        self.hour_df = pd.DataFrame()
        self.session_df = pd.DataFrame()
        self.matrix_df = pd.DataFrame()

        self.status_var = tk.StringVar(value="READY")
        self.source_var = tk.StringVar(value=str(self.repository.root))
        self.summary_var = tk.StringVar(value="")
        self.error_trace = ""

        self._build_ui()
        self.refresh_data()

    def refresh_data(self) -> None:
        self.refresh()

    def _build_ui(self) -> None:
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=20, pady=16)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(5, weight=1)

        self._build_header(root)
        self._build_nav(root)
        self._build_kpis(root)
        self._build_filters(root)
        self._build_grid(root)

    def _build_header(self, parent) -> None:
        header = tk.Frame(parent, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        left = tk.Frame(header, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Market Seasonality Dashboard", bg=BG, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(left, textvariable=self.source_var, bg=BG, fg=MUTED, font=FONT_SUB).pack(anchor="w", pady=(1, 0))

        right = tk.Frame(header, bg=BG)
        right.grid(row=0, column=1, sticky="e")

        tk.Button(right, text="Refresh", bg=CARD, fg=GREEN, activebackground=CARD_SOFT,
                  activeforeground=FG, relief="flat", font=FONT_SMALL, padx=14, pady=8,
                  command=self.refresh_data).pack(side="left", padx=(0, 8))

        tk.Button(right, text="Export CSV", bg=CARD, fg=FG, activebackground=CARD_SOFT,
                  activeforeground=FG, relief="flat", font=FONT_SMALL, padx=14, pady=8,
                  command=self.export_csv).pack(side="left")

    def _build_nav(self, parent) -> None:
        nav = tk.Frame(parent, bg=BG)
        nav.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        labels = ["⌂ Overview", "MKT Seasonality", "Monthly", "Weekday", "Sessions", "Matrix", "Registry"]
        for i, label in enumerate(labels):
            NavButton(nav, label, active=(i == 1)).pack(side="left", padx=(0, 18))
        tk.Frame(parent, bg=BORDER, height=1).grid(row=2, column=0, sticky="ew", pady=(0, 14))

    def _build_kpis(self, parent) -> None:
        row = tk.Frame(parent, bg=BG)
        row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for i in range(8):
            row.columnconfigure(i, weight=1, uniform="kpi")

        self.kpi_bars = KpiCard(row, "Bars")
        self.kpi_avg_ret = KpiCard(row, "Avg Return")
        self.kpi_winrate = KpiCard(row, "Win Rate")
        self.kpi_range = KpiCard(row, "Avg Range")
        self.kpi_best_month = KpiCard(row, "Best Month")
        self.kpi_worst_month = KpiCard(row, "Worst Month")
        self.kpi_best_day = KpiCard(row, "Best Day")
        self.kpi_best_session = KpiCard(row, "Best Session")

        for i, card in enumerate([
            self.kpi_bars, self.kpi_avg_ret, self.kpi_winrate, self.kpi_range,
            self.kpi_best_month, self.kpi_worst_month, self.kpi_best_day, self.kpi_best_session
        ]):
            card.grid(row=0, column=i, sticky="nsew", padx=5)

    def _build_filters(self, parent) -> None:
        filt = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        filt.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        filt.columnconfigure(7, weight=1)

        tk.Label(filt, text="Symbol", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=0, padx=(10, 5), pady=8)
        self.symbol_combo = ttk.Combobox(filt, textvariable=self.symbol_var, state="readonly", width=18)
        self.symbol_combo.grid(row=0, column=1, padx=(0, 12), pady=8)
        self.symbol_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_data())

        tk.Label(filt, text="Timeframe", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=2, padx=(0, 5), pady=8)
        self.timeframe_combo = ttk.Combobox(filt, textvariable=self.timeframe_var, state="readonly", width=10)
        self.timeframe_combo.grid(row=0, column=3, padx=(0, 12), pady=8)
        self.timeframe_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_timeframe_change())

        tk.Label(filt, text="Metric", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=4, padx=(0, 5), pady=8)
        self.metric_combo = ttk.Combobox(
            filt, textvariable=self.metric_var, state="readonly", width=14,
            values=["Avg Return", "Win Rate", "Avg Range", "Total Return"]
        )
        self.metric_combo.grid(row=0, column=5, padx=(0, 12), pady=8)
        self.metric_combo.bind("<<ComboboxSelected>>", lambda _e: self.update_table())

        tk.Label(filt, textvariable=self.summary_var, bg=CARD, fg=FG, font=FONT_SMALL).grid(row=0, column=6, padx=(0, 10), pady=8, sticky="w")
        tk.Label(filt, textvariable=self.status_var, bg=CARD, fg=GREEN, font=FONT_SMALL).grid(row=0, column=7, padx=(0, 10), pady=8, sticky="e")

    def _build_grid(self, parent) -> None:
        grid = tk.Frame(parent, bg=BG)
        grid.grid(row=5, column=0, sticky="nsew")
        for c in range(12):
            grid.columnconfigure(c, weight=1, uniform="dash")
        for r in range(8):
            grid.rowconfigure(r, weight=1, uniform="dash")

        self.month_card = ChartCard(grid, "Monthly Seasonality", height=2.4)
        self.month_card.grid(row=0, column=0, rowspan=3, columnspan=5, sticky="nsew", padx=5, pady=5)

        self.weekday_card = ChartCard(grid, "Day-of-Week Seasonality", height=2.4)
        self.weekday_card.grid(row=0, column=5, rowspan=3, columnspan=3, sticky="nsew", padx=5, pady=5)

        self.session_card = ChartCard(grid, "Session Behavior", height=2.4)
        self.session_card.grid(row=0, column=8, rowspan=3, columnspan=4, sticky="nsew", padx=5, pady=5)

        self.matrix_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.matrix_card.grid(row=3, column=0, rowspan=3, columnspan=6, sticky="nsew", padx=5, pady=5)
        self._build_matrix_shell()

        self.hour_card = ChartCard(grid, "Hour-of-Day Seasonality", height=2.5)
        self.hour_card.grid(row=3, column=6, rowspan=3, columnspan=6, sticky="nsew", padx=5, pady=5)

        self.table_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.table_card.grid(row=6, column=0, rowspan=2, columnspan=8, sticky="nsew", padx=5, pady=5)
        self._build_table_shell()

        self.detail_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.detail_card.grid(row=6, column=8, rowspan=2, columnspan=4, sticky="nsew", padx=5, pady=5)
        self._build_details_shell()

    def _build_matrix_shell(self) -> None:
        tk.Label(self.matrix_card.inner, text="Year × Month Return Matrix", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.matrix_canvas_frame = tk.Frame(self.matrix_card.inner, bg=CARD)
        self.matrix_canvas_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.matrix_canvas = tk.Canvas(self.matrix_canvas_frame, bg=CARD, highlightthickness=0, bd=0)
        self.matrix_canvas.pack(side="left", fill="both", expand=True)
        self.matrix_scroll = tk.Scrollbar(self.matrix_canvas_frame, orient="vertical", command=self.matrix_canvas.yview)
        self.matrix_scroll.pack(side="right", fill="y")
        self.matrix_canvas.configure(yscrollcommand=self.matrix_scroll.set)
        self.matrix_body = tk.Frame(self.matrix_canvas, bg=CARD)
        self.matrix_window = self.matrix_canvas.create_window((0, 0), window=self.matrix_body, anchor="nw")
        self.matrix_body.bind("<Configure>", lambda _e: self.matrix_canvas.configure(scrollregion=self.matrix_canvas.bbox("all")))
        self.matrix_canvas.bind("<Configure>", lambda e: self.matrix_canvas.itemconfigure(self.matrix_window, width=e.width))

    def _build_table_shell(self) -> None:
        tk.Label(self.table_card.inner, text="Seasonality Table", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        cols = ("bucket", "bars", "avg_return", "winrate", "avg_range", "total_return")
        self.table = ttk.Treeview(self.table_card.inner, columns=cols, show="headings", height=8, style="Dark.Treeview")
        headers = [
            ("bucket", "Bucket", 130, "w"),
            ("bars", "Bars", 80, "e"),
            ("avg_return", "Avg Return", 110, "e"),
            ("winrate", "Win Rate", 100, "e"),
            ("avg_range", "Avg Range", 100, "e"),
            ("total_return", "Total Return", 110, "e"),
        ]
        for col, text, width, anchor in headers:
            self.table.heading(col, text=text)
            self.table.column(col, width=width, anchor=anchor, stretch=True)
        wrap = tk.Frame(self.table_card.inner, bg=CARD)
        wrap.pack(fill="both", expand=True, pady=(8, 0))
        self.table.pack(in_=wrap, side="left", fill="both", expand=True)
        ysb = tk.Scrollbar(wrap, orient="vertical", command=self.table.yview)
        ysb.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=ysb.set)
        self.table.tag_configure("pos", foreground=GREEN)
        self.table.tag_configure("neg", foreground=RED)
        self.table.bind("<<TreeviewSelect>>", lambda _e: self.update_details())

    def _build_details_shell(self) -> None:
        tk.Label(self.detail_card.inner, text="Details / Diagnostics", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.details = tk.Text(self.detail_card.inner, bg=CARD_DARK, fg=FG, insertbackground=FG,
                               relief="flat", font=("Consolas", 8), wrap="word", height=8)
        self.details.pack(fill="both", expand=True, pady=(8, 0))
        self.details.configure(state="disabled")

    def _configure_ttk(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Dark.Treeview", background=CARD, foreground=FG, fieldbackground=CARD, borderwidth=0, rowheight=24, font=FONT_SMALL)
        style.configure("Dark.Treeview.Heading", background=CARD, foreground=MUTED, borderwidth=0, font=FONT_TINY)
        style.map("Dark.Treeview", background=[("selected", CARD_SOFT)], foreground=[("selected", FG)])

    def refresh(self) -> None:
        self._configure_ttk()
        try:
            if not self.repository.root.exists():
                raise FileNotFoundError(f"OHLC root not found: {self.repository.root}")

            timeframes = self.repository.discover_timeframes()
            if not timeframes:
                raise FileNotFoundError(f"No timeframe folders found in: {self.repository.root}")

            self.timeframe_combo["values"] = timeframes
            if self.timeframe_var.get() not in timeframes:
                self.timeframe_var.set("D1" if "D1" in timeframes else timeframes[0])

            symbols = self.repository.discover_symbols(self.timeframe_var.get())
            if not symbols:
                raise FileNotFoundError(f"No parquet files found for timeframe: {self.timeframe_var.get()}")

            self.symbol_combo["values"] = symbols
            if self.symbol_var.get() not in symbols:
                self.symbol_var.set(symbols[0])

            self.load_data()
            self.update_table()
            self.status_var.set("ONLINE")
            self.error_trace = ""
        except Exception as exc:
            self.status_var.set("ERROR")
            self.error_trace = traceback.format_exc()
            self._show_error(exc)

    def _on_timeframe_change(self) -> None:
        symbols = self.repository.discover_symbols(self.timeframe_var.get())
        self.symbol_combo["values"] = symbols
        if symbols:
            self.symbol_var.set(symbols[0])
        self.refresh_data()

    def load_data(self) -> pd.DataFrame:
        symbol = self.symbol_var.get().strip()
        timeframe = self.timeframe_var.get().strip()
        self.df = self.repository.load_ohlc(timeframe, symbol)

        engine = SeasonalityEngine(self.df)
        self.monthly_df = engine.monthly()
        self.weekday_df = engine.weekdays()
        self.hour_df = engine.hours()
        self.session_df = engine.sessions()
        self.matrix_df = engine.year_month_matrix()
        k = engine.kpis()

        self.kpi_bars.set(f"{k['bars']:,}", "loaded candles", FG)
        self.kpi_avg_ret.set(fmt_pct(k["avg_return"]), "mean bar return", value_color(k["avg_return"]))
        self.kpi_winrate.set(fmt_pct(k["winrate"], 1), "positive return bars", GREEN if pd.notna(k["winrate"]) and k["winrate"] >= 50 else RED)
        self.kpi_range.set(fmt_pct(k["avg_range"]), "mean candle range", FG)
        self.kpi_best_month.set(str(k["best_month"]), "highest avg return", GREEN)
        self.kpi_worst_month.set(str(k["worst_month"]), "lowest avg return", RED)
        self.kpi_best_day.set(str(k["best_weekday"]), "best weekday", GREEN)
        self.kpi_best_session.set(str(k["best_session"]), "best session", GREEN)

        start = self.df["timestamp"].min()
        end = self.df["timestamp"].max()
        self.summary_var.set(f"{symbol} | {timeframe} | {len(self.df):,} bars | {start.date()} → {end.date()}")

        self._plot_all()
        self._render_matrix()
        self._write_details(self._base_details())
        return self.df

    def _metric_col(self) -> str:
        metric = self.metric_var.get()
        if metric == "Win Rate":
            return "winrate_pct"
        if metric == "Avg Range":
            return "avg_range_pct"
        if metric == "Total Return":
            return "total_return_pct"
        return "avg_return_pct"

    def update_table(self) -> None:
        if not hasattr(self, "table"):
            return
        self.table.delete(*self.table.get_children())
        if self.monthly_df.empty:
            return

        for _, row in self.monthly_df.iterrows():
            avg_return = row.get("avg_return_pct", np.nan)
            tag = "pos" if pd.notna(avg_return) and avg_return >= 0 else "neg"
            values = (
                row.get("month_name", "-"),
                f"{int(row.get('bars', 0)):,}",
                fmt_pct(row.get("avg_return_pct")),
                fmt_pct(row.get("winrate_pct"), 1),
                fmt_pct(row.get("avg_range_pct")),
                fmt_pct(row.get("total_return_pct")),
            )
            self.table.insert("", "end", values=values, tags=(tag,))

        self._plot_all()
        self._render_matrix()

    def update_details(self) -> None:
        selection = self.table.selection()
        if not selection:
            self._write_details(self._base_details())
            return
        values = self.table.item(selection[0], "values")
        text = self._base_details() + "\n\nSELECTED BUCKET\n" + "-" * 40 + "\n"
        labels = ["Bucket", "Bars", "Avg Return", "Win Rate", "Avg Range", "Total Return"]
        for label, value in zip(labels, values):
            text += f"{label:18}: {value}\n"
        self._write_details(text)

    def _base_details(self) -> str:
        path = self.repository.root / self.timeframe_var.get() / f"{self.symbol_var.get()}.parquet"
        return (
            "MARKET SEASONALITY\n"
            + "-" * 50 + "\n"
            + f"Symbol            : {self.symbol_var.get()}\n"
            + f"Timeframe         : {self.timeframe_var.get()}\n"
            + f"Data Root         : {self.repository.root}\n"
            + f"File              : {path}\n"
            + f"Rows              : {len(self.df):,}\n"
            + f"Metric            : {self.metric_var.get()}\n"
            + "\nCALCULATION\n"
            + "-" * 50 + "\n"
            + "Return            : close.pct_change()\n"
            + "Range %           : (high-low)/close\n"
            + "Body %            : (close-open)/open\n"
            + "Session           : UTC hour buckets\n"
            + "\nFLOW\n"
            + "-" * 50 + "\n"
            + "OHLC → Returns → Month/Weekday/Hour/Session → Matrix → Edge Diagnostics\n"
        )

    def _write_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", text)
        self.details.configure(state="disabled")

    def _show_error(self, exc: Exception) -> None:
        for card_name in ["month_card", "weekday_card", "session_card", "hour_card"]:
            if hasattr(self, card_name):
                card = getattr(self, card_name)
                card.reset()
                card.ax.text(0.5, 0.5, "LOAD ERROR", color=RED, ha="center", va="center", transform=card.ax.transAxes)
                card.canvas.draw_idle()

        self.kpi_bars.set("-", "error", RED)
        self.summary_var.set(str(exc))
        self._write_details(
            "LOAD ERROR\n"
            + "-" * 80 + "\n"
            + str(exc)
            + "\n\nTRACEBACK\n"
            + "-" * 80 + "\n"
            + self.error_trace
        )

    def _plot_all(self) -> None:
        self._plot_bar(self.month_card, self.monthly_df, "month_name")
        self._plot_bar(self.weekday_card, self.weekday_df, "weekday_name")
        self._plot_bar(self.session_card, self.session_df, "session")
        self._plot_hour()

    def _plot_bar(self, card: ChartCard, frame: pd.DataFrame, label_col: str) -> None:
        card.reset()
        if frame.empty:
            card.ax.text(0.5, 0.5, "NO DATA", color=MUTED, ha="center", va="center", transform=card.ax.transAxes)
            card.canvas.draw_idle()
            return

        metric_col = self._metric_col()
        x = frame[label_col].astype(str).tolist()
        y = pd.to_numeric(frame[metric_col], errors="coerce").fillna(0.0).values
        colors = [GREEN if v >= 0 else RED for v in y]
        card.ax.bar(x, y, color=colors, alpha=0.85)
        card.ax.axhline(0, color=GRID, linewidth=1.0)
        card.ax.set_ylabel(self.metric_var.get(), fontsize=7)
        card.ax.tick_params(axis="x", rotation=45 if len(x) > 6 else 0)
        style_ax(card.ax)
        card.fig.tight_layout(pad=1.0)
        card.canvas.draw_idle()

    def _plot_hour(self) -> None:
        card = self.hour_card
        card.reset()
        if self.hour_df.empty:
            card.ax.text(0.5, 0.5, "NO INTRADAY DATA", color=MUTED, ha="center", va="center", transform=card.ax.transAxes)
            card.canvas.draw_idle()
            return

        metric_col = self._metric_col()
        x = pd.to_numeric(self.hour_df["hour"], errors="coerce").fillna(0).astype(int).values
        y = pd.to_numeric(self.hour_df[metric_col], errors="coerce").fillna(0.0).values
        colors = [GREEN if v >= 0 else RED for v in y]
        card.ax.bar(x, y, color=colors, alpha=0.85)
        card.ax.axhline(0, color=GRID, linewidth=1.0)
        card.ax.set_xticks(list(range(0, 24, 2)))
        card.ax.set_xlabel("UTC Hour", fontsize=7)
        card.ax.set_ylabel(self.metric_var.get(), fontsize=7)
        style_ax(card.ax)
        card.fig.tight_layout(pad=1.0)
        card.canvas.draw_idle()

    def _render_matrix(self) -> None:
        clear_children(self.matrix_body)

        if self.matrix_df.empty:
            tk.Label(self.matrix_body, text="NO MATRIX DATA", bg=CARD, fg=MUTED, font=FONT_SMALL).pack(anchor="center", pady=20)
            return

        header = tk.Frame(self.matrix_body, bg=CARD)
        header.pack(fill="x")
        tk.Label(header, text="Year", bg=CARD_DARK, fg=MUTED, font=FONT_TINY, width=8, anchor="center").pack(side="left", padx=1, pady=1)
        for m in MONTH_NAMES:
            tk.Label(header, text=m, bg=CARD_DARK, fg=MUTED, font=FONT_TINY, width=8, anchor="center").pack(side="left", padx=1, pady=1)

        for year, row in self.matrix_df.iterrows():
            line = tk.Frame(self.matrix_body, bg=CARD)
            line.pack(fill="x")
            tk.Label(line, text=str(year), bg=CARD_DARK, fg=FG, font=FONT_TINY, width=8, anchor="center").pack(side="left", padx=1, pady=1)
            for m in MONTH_NAMES:
                val = row.get(m, np.nan)
                try:
                    v = float(val)
                except Exception:
                    v = np.nan
                if pd.isna(v):
                    bg = CARD_DARK
                    fg = SUBTLE
                    txt = "-"
                else:
                    bg = "#063018" if v >= 0 else "#3A0808"
                    fg = GREEN if v >= 0 else RED
                    txt = f"{v:.1f}%"
                tk.Label(line, text=txt, bg=bg, fg=fg, font=FONT_TINY, width=8, anchor="center").pack(side="left", padx=1, pady=1)

    def export_csv(self) -> None:
        if self.monthly_df.empty:
            messagebox.showinfo("Export CSV", "No data loaded.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Seasonality",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"seasonality_{self.symbol_var.get()}_{self.timeframe_var.get()}.csv",
        )
        if not path:
            return

        out = self.monthly_df.copy()
        out.insert(0, "symbol", self.symbol_var.get())
        out.insert(1, "timeframe", self.timeframe_var.get())
        out.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Export CSV", f"Exported:\n{path}")


def load_data(*args, **kwargs):
    repo = kwargs.get("repository") or OHLCRepository()
    timeframe = kwargs.get("timeframe", "D1")
    symbol = kwargs.get("symbol")
    if symbol is None:
        symbols = repo.discover_symbols(timeframe)
        if not symbols:
            return pd.DataFrame()
        symbol = symbols[0]
    return repo.load_ohlc(timeframe, symbol)


def refresh_data(panel=None):
    if panel is not None and hasattr(panel, "refresh_data"):
        return panel.refresh_data()
    return None


def update_table(panel=None):
    if panel is not None and hasattr(panel, "update_table"):
        return panel.update_table()
    return None


def update_details(panel=None):
    if panel is not None and hasattr(panel, "update_details"):
        return panel.update_details()
    return None


def build_panel(parent, repository=None, **kwargs):
    return SeasonalityDashboard(parent, repository=repository, **kwargs)


def create_panel(parent, repository=None, **kwargs):
    return build_panel(parent, repository=repository, **kwargs)


DashboardPanel = SeasonalityDashboard


if __name__ == "__main__":
    root = tk.Tk()
    root.title("QUANT - Market Seasonality Dashboard")
    root.configure(bg=BG)
    root.geometry("1480x900")
    panel = build_panel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()

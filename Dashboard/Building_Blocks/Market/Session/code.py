"""
QUANT/Dashboard/Building_Blocks/Market/Session/code.py

Market Session Behavior Dashboard Building Block

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: dashboard_market_session_behavior
# script_name: Market Session Behavior Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Market Analytics
# asset_type: Dashboard
# purpose: OHLC-based session behavior dashboard for Asia, EU, US and overlap regime analysis across market symbols and timeframes.
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
# version: v1.0.0
# last_reviewed: 2026-06-07
# required_api:
#   - build_panel(parent, repository=None, **kwargs)
# expected_location:
#   - QUANT/Dashboard/Building_Blocks/Market/Session/code.py
# data_source:
#   - Data_Center/Data/1_Pipeline/Market/ohcl
# scanner:
#   - kein Scanner
# ============================================================
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
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


# ============================================================
# THEME
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
SESSION_ORDER = ["ASIA", "EU", "US", "EU_US_OVERLAP", "OTHER"]
WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ============================================================
# PATHS
# ============================================================

def find_quant_root(start: Path) -> Path:
    start = Path(start).resolve()

    def is_quant_root(p: Path) -> bool:
        try:
            return (p / "Dashboard").exists() and (p / "Data_Center").exists()
        except Exception:
            return False

    candidates: List[Path] = []
    candidates.extend([start] + list(start.parents))

    try:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd] + list(cwd.parents))
    except Exception:
        pass

    home = Path.home()
    known = [
        home / "Desktop" / "Business_Code" / "QUANT",
        home / "Desktop" / "Business_Code",
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
    ]

    for base in known:
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
        if is_quant_root(c):
            return c.resolve()

    for root in [home / "Desktop" / "Business_Code", home / "Desktop"]:
        if not root.exists():
            continue
        try:
            for p in root.rglob("QUANT"):
                if p.is_dir() and is_quant_root(p):
                    return p.resolve()
        except Exception:
            continue

    raise RuntimeError(
        f"QUANT root not found. Expected folder containing Dashboard/ and Data_Center/. "
        f"Start={start} | CWD={Path.cwd()}"
    )


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)
OHLC_ROOT = QUANT_ROOT / "Data_Center" / "Data" / "1_Pipeline" / "Market" / "ohcl"


# ============================================================
# REGISTRY
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "dashboard_market_session_behavior",
    "script_name": "Market Session Behavior Dashboard",
    "owner": "Leon",
    "status": "active",
    "layer": "Dashboard",
    "domain": "Market Analytics",
    "asset_type": "Dashboard",
    "purpose": "OHLC-based session behavior analysis for Asia, EU, US and EU/US overlap.",
    "inputs": [
        "Data_Center/Data/1_Pipeline/Market/ohcl/**/*.parquet",
        "Data_Center/Data/1_Pipeline/Market/ohcl/summary.json",
    ],
    "outputs": ["Dashboard UI"],
    "dependencies": ["tkinter", "ttk", "pathlib", "pandas", "numpy", "matplotlib", "json"],
    "schedule": "manual",
    "version": "v1.0.0",
    "last_reviewed": "2026-06-07",
    "expected_location": "QUANT/Dashboard/Building_Blocks/Market/Session/code.py",
    "data_source": "Data_Center/Data/1_Pipeline/Market/ohcl",
    "scanner": "kein Scanner",
    "required_api": "build_panel(parent, repository=None, **kwargs)",
}


def get_code_registry() -> Dict[str, object]:
    return dict(CODE_REGISTRY)


# ============================================================
# DATA
# ============================================================

@dataclass
class MarketFile:
    symbol: str
    timeframe: str
    path: Path


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

    def load_summary(self) -> Dict[str, object]:
        path = self.root / "summary.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def load_ohlc(self, timeframe: str, symbol: str) -> pd.DataFrame:
        path = self.root / timeframe / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"OHLC file not found: {path}")

        df = pd.read_parquet(path)
        if df.empty:
            raise ValueError(f"OHLC file is empty: {path}")

        return normalize_ohlc(df)


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
        src = None
        for name in names:
            if name in lower_map:
                src = lower_map[name]
                break
        if src is None:
            if target == "volume":
                d[target] = 0.0
                continue
            raise ValueError(f"Missing required OHLC column: {target}. Available={list(df.columns)}")
        d[target] = pd.to_numeric(d[src], errors="coerce")

    d = d.dropna(subset=["open", "high", "low", "close"])

    d["return"] = d["close"].pct_change()
    d["range_pct"] = (d["high"] - d["low"]) / d["close"].replace(0, np.nan)
    d["body_pct"] = (d["close"] - d["open"]) / d["open"].replace(0, np.nan)
    d["upper_wick_pct"] = (d["high"] - d[["open", "close"]].max(axis=1)) / d["close"].replace(0, np.nan)
    d["lower_wick_pct"] = (d[["open", "close"]].min(axis=1) - d["low"]) / d["close"].replace(0, np.nan)
    d["direction"] = np.where(d["close"] >= d["open"], "UP", "DOWN")
    d["hour"] = d["timestamp"].dt.hour
    d["weekday"] = d["timestamp"].dt.weekday
    d["weekday_name"] = d["weekday"].map(lambda x: WEEKDAY_NAMES[int(x)] if pd.notna(x) else "")
    d["date"] = d["timestamp"].dt.date
    d["session"] = d["hour"].apply(classify_session)

    return d.reset_index(drop=True)


def classify_session(hour: int) -> str:
    try:
        h = int(hour)
    except Exception:
        return "OTHER"

    # UTC session buckets. Adjust later if your broker data uses server time.
    if 0 <= h < 7:
        return "ASIA"
    if 7 <= h < 13:
        return "EU"
    if 13 <= h < 16:
        return "EU_US_OVERLAP"
    if 16 <= h < 21:
        return "US"
    return "OTHER"


# ============================================================
# ENGINE
# ============================================================

class SessionBehaviorEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().replace([np.inf, -np.inf], np.nan)

    def session_stats(self) -> pd.DataFrame:
        d = self.df.copy()
        if d.empty:
            return pd.DataFrame()

        g = d.groupby("session", dropna=False).agg(
            bars=("close", "count"),
            avg_return=("return", "mean"),
            median_return=("return", "median"),
            total_return=("return", lambda x: (1.0 + x.dropna()).prod() - 1.0 if len(x.dropna()) else np.nan),
            winrate=("return", lambda x: float((x.dropna() > 0).mean()) if len(x.dropna()) else np.nan),
            avg_range=("range_pct", "mean"),
            avg_body=("body_pct", "mean"),
            avg_upper_wick=("upper_wick_pct", "mean"),
            avg_lower_wick=("lower_wick_pct", "mean"),
            up_bars=("direction", lambda x: int((x == "UP").sum())),
            down_bars=("direction", lambda x: int((x == "DOWN").sum())),
        ).reset_index()

        g = self._add_pct_cols(g)
        order = {s: i for i, s in enumerate(SESSION_ORDER)}
        g["_order"] = g["session"].map(order).fillna(999)
        return g.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    def hour_stats(self) -> pd.DataFrame:
        d = self.df.copy()
        if d.empty:
            return pd.DataFrame()

        g = d.groupby("hour", dropna=False).agg(
            bars=("close", "count"),
            avg_return=("return", "mean"),
            winrate=("return", lambda x: float((x.dropna() > 0).mean()) if len(x.dropna()) else np.nan),
            avg_range=("range_pct", "mean"),
            avg_body=("body_pct", "mean"),
        ).reset_index()

        return self._add_pct_cols(g).sort_values("hour").reset_index(drop=True)

    def weekday_session_matrix(self, metric: str = "avg_return_pct") -> pd.DataFrame:
        d = self.df.copy()
        if d.empty:
            return pd.DataFrame()

        g = d.groupby(["weekday_name", "session"]).agg(
            avg_return=("return", "mean"),
            winrate=("return", lambda x: float((x.dropna() > 0).mean()) if len(x.dropna()) else np.nan),
            avg_range=("range_pct", "mean"),
            bars=("close", "count"),
        ).reset_index()

        g["avg_return_pct"] = g["avg_return"] * 100.0
        g["winrate_pct"] = g["winrate"] * 100.0
        g["avg_range_pct"] = g["avg_range"] * 100.0

        if metric not in g.columns:
            metric = "avg_return_pct"

        mat = g.pivot(index="weekday_name", columns="session", values=metric)
        mat = mat.reindex(index=WEEKDAY_NAMES, columns=SESSION_ORDER)
        return mat

    def daily_session_summary(self) -> pd.DataFrame:
        d = self.df.dropna(subset=["return"]).copy()
        if d.empty:
            return pd.DataFrame()

        daily = d.groupby(["date", "session"]).agg(
            session_return=("return", lambda x: (1.0 + x.dropna()).prod() - 1.0 if len(x.dropna()) else np.nan),
            session_range=("range_pct", "sum"),
            bars=("close", "count"),
        ).reset_index()

        daily["session_return_pct"] = daily["session_return"] * 100.0
        daily["session_range_pct"] = daily["session_range"] * 100.0
        return daily

    def open_close_bias(self) -> pd.DataFrame:
        d = self.df.copy()
        if d.empty:
            return pd.DataFrame()

        rows = []
        for sess in SESSION_ORDER:
            s = d[d["session"] == sess].copy()
            if s.empty:
                continue
            grouped = s.groupby("date")
            for date, x in grouped:
                x = x.sort_values("timestamp")
                if x.empty:
                    continue
                open_price = float(x["open"].iloc[0])
                close_price = float(x["close"].iloc[-1])
                high = float(x["high"].max())
                low = float(x["low"].min())
                ret = (close_price - open_price) / open_price if open_price else np.nan
                rng = (high - low) / close_price if close_price else np.nan
                rows.append({
                    "date": date,
                    "session": sess,
                    "open": open_price,
                    "close": close_price,
                    "return_pct": ret * 100.0,
                    "range_pct": rng * 100.0,
                    "direction": "UP" if close_price >= open_price else "DOWN",
                })
        return pd.DataFrame(rows)

    def kpis(self) -> Dict[str, object]:
        stats = self.session_stats()
        if stats.empty:
            return {
                "bars": 0, "best_session": "-", "worst_session": "-", "best_range": "-",
                "avg_range": np.nan, "winrate": np.nan, "active_hours": 0, "dominant_session": "-"
            }

        best = stats.sort_values("avg_return_pct", ascending=False).head(1)
        worst = stats.sort_values("avg_return_pct", ascending=True).head(1)
        best_range = stats.sort_values("avg_range_pct", ascending=False).head(1)
        dominant = stats.sort_values("bars", ascending=False).head(1)

        d = self.df.dropna(subset=["return"])

        return {
            "bars": int(len(self.df)),
            "best_session": str(best.iloc[0]["session"]) if not best.empty else "-",
            "worst_session": str(worst.iloc[0]["session"]) if not worst.empty else "-",
            "best_range": str(best_range.iloc[0]["session"]) if not best_range.empty else "-",
            "avg_range": float(self.df["range_pct"].mean() * 100.0) if "range_pct" in self.df else np.nan,
            "winrate": float((d["return"] > 0).mean() * 100.0) if not d.empty else np.nan,
            "active_hours": int(self.df["hour"].nunique()) if "hour" in self.df else 0,
            "dominant_session": str(dominant.iloc[0]["session"]) if not dominant.empty else "-",
        }

    @staticmethod
    def _add_pct_cols(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for source, target in [
            ("avg_return", "avg_return_pct"),
            ("median_return", "median_return_pct"),
            ("total_return", "total_return_pct"),
            ("winrate", "winrate_pct"),
            ("avg_range", "avg_range_pct"),
            ("avg_body", "avg_body_pct"),
            ("avg_upper_wick", "avg_upper_wick_pct"),
            ("avg_lower_wick", "avg_lower_wick_pct"),
        ]:
            if source in g.columns:
                g[target] = g[source] * 100.0
        return g


# ============================================================
# UI HELPERS
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
            fg=GREEN if active else MUTED,
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
        self.sub_label = tk.Label(self.inner, textvariable=self.sub_var, bg=CARD, fg=MUTED, font=FONT_TINY)
        self.sub_label.pack(anchor="w")

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


def clear_children(widget: tk.Widget) -> None:
    for child in widget.winfo_children():
        child.destroy()


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


# ============================================================
# DASHBOARD
# ============================================================

class SessionBehaviorDashboard(tk.Frame):
    def __init__(self, parent, repository: Optional[OHLCRepository] = None, **_kwargs):
        super().__init__(parent, bg=BG)
        self.repository = repository or OHLCRepository()

        self.symbol_var = tk.StringVar(value="")
        self.timeframe_var = tk.StringVar(value="H1")
        self.metric_var = tk.StringVar(value="Avg Return")

        self.status_var = tk.StringVar(value="READY")
        self.source_var = tk.StringVar(value=str(self.repository.root))
        self.summary_var = tk.StringVar(value="")
        self.error_trace = ""

        self.df = pd.DataFrame()
        self.session_df = pd.DataFrame()
        self.hour_df = pd.DataFrame()
        self.matrix_df = pd.DataFrame()
        self.open_close_df = pd.DataFrame()

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
        tk.Label(left, text="Market Session Behavior Dashboard", bg=BG, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(left, textvariable=self.source_var, bg=BG, fg=MUTED, font=FONT_SUB).pack(anchor="w", pady=(1, 0))

        right = tk.Frame(header, bg=BG)
        right.grid(row=0, column=1, sticky="e")

        tk.Button(
            right, text="Refresh", bg=CARD, fg=GREEN, activebackground=CARD_SOFT,
            activeforeground=FG, relief="flat", font=FONT_SMALL, padx=14, pady=8,
            command=self.refresh_data,
        ).pack(side="left", padx=(0, 8))

        tk.Button(
            right, text="Export CSV", bg=CARD, fg=FG, activebackground=CARD_SOFT,
            activeforeground=FG, relief="flat", font=FONT_SMALL, padx=14, pady=8,
            command=self.export_csv,
        ).pack(side="left")

    def _build_nav(self, parent) -> None:
        nav = tk.Frame(parent, bg=BG)
        nav.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        labels = ["⌂ Overview", "MKT Session", "Asia", "EU", "US", "Overlap", "Matrix", "Registry"]
        for i, label in enumerate(labels):
            btn = NavButton(nav, label, active=(i == 1))
            btn.pack(side="left", padx=(0, 18))
        tk.Frame(parent, bg=BORDER, height=1).grid(row=2, column=0, sticky="ew", pady=(0, 14))

    def _build_kpis(self, parent) -> None:
        row = tk.Frame(parent, bg=BG)
        row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for i in range(8):
            row.columnconfigure(i, weight=1, uniform="kpi")

        self.kpi_bars = KpiCard(row, "Bars")
        self.kpi_best_session = KpiCard(row, "Best Session")
        self.kpi_worst_session = KpiCard(row, "Worst Session")
        self.kpi_best_range = KpiCard(row, "Max Range")
        self.kpi_winrate = KpiCard(row, "Win Rate")
        self.kpi_avg_range = KpiCard(row, "Avg Range")
        self.kpi_active_hours = KpiCard(row, "Active Hours")
        self.kpi_dominant = KpiCard(row, "Dominant Sess.")

        for i, card in enumerate([
            self.kpi_bars, self.kpi_best_session, self.kpi_worst_session, self.kpi_best_range,
            self.kpi_winrate, self.kpi_avg_range, self.kpi_active_hours, self.kpi_dominant
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

        self.session_card = ChartCard(grid, "Session Performance", height=2.5)
        self.session_card.grid(row=0, column=0, rowspan=3, columnspan=4, sticky="nsew", padx=5, pady=5)

        self.hour_card = ChartCard(grid, "Hour-of-Day Behavior", height=2.5)
        self.hour_card.grid(row=0, column=4, rowspan=3, columnspan=4, sticky="nsew", padx=5, pady=5)

        self.open_close_card = ChartCard(grid, "Session Open → Close Bias", height=2.5)
        self.open_close_card.grid(row=0, column=8, rowspan=3, columnspan=4, sticky="nsew", padx=5, pady=5)

        self.matrix_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.matrix_card.grid(row=3, column=0, rowspan=3, columnspan=6, sticky="nsew", padx=5, pady=5)
        self._build_matrix_shell()

        self.table_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.table_card.grid(row=3, column=6, rowspan=3, columnspan=6, sticky="nsew", padx=5, pady=5)
        self._build_table_shell()

        self.detail_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.detail_card.grid(row=6, column=0, rowspan=2, columnspan=12, sticky="nsew", padx=5, pady=5)
        self._build_details_shell()

    def _build_matrix_shell(self) -> None:
        tk.Label(self.matrix_card.inner, text="Weekday × Session Matrix", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")

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
        tk.Label(self.table_card.inner, text="Session Statistics", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")

        cols = ("session", "bars", "avg_return", "winrate", "avg_range", "avg_body", "total_return")
        self.table = ttk.Treeview(self.table_card.inner, columns=cols, show="headings", height=8, style="Dark.Treeview")

        headers = [
            ("session", "Session", 120, "w"),
            ("bars", "Bars", 80, "e"),
            ("avg_return", "Avg Ret", 90, "e"),
            ("winrate", "Win %", 80, "e"),
            ("avg_range", "Range", 80, "e"),
            ("avg_body", "Body", 80, "e"),
            ("total_return", "Total", 90, "e"),
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
        self.details = tk.Text(
            self.detail_card.inner, bg=CARD_DARK, fg=FG, insertbackground=FG,
            relief="flat", font=("Consolas", 8), wrap="word", height=7
        )
        self.details.pack(fill="both", expand=True, pady=(8, 0))
        self.details.configure(state="disabled")

    # ========================================================
    # DATA + REFRESH
    # ========================================================

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
                self.timeframe_var.set("H1" if "H1" in timeframes else timeframes[0])

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
        engine = SessionBehaviorEngine(self.df)

        self.session_df = engine.session_stats()
        self.hour_df = engine.hour_stats()
        self.matrix_df = engine.weekday_session_matrix(metric=self._metric_col())
        self.open_close_df = engine.open_close_bias()

        k = engine.kpis()

        self.kpi_bars.set(f"{k['bars']:,}", "loaded candles", FG)
        self.kpi_best_session.set(str(k["best_session"]), "highest avg return", GREEN)
        self.kpi_worst_session.set(str(k["worst_session"]), "lowest avg return", RED)
        self.kpi_best_range.set(str(k["best_range"]), "largest avg range", YELLOW)
        self.kpi_winrate.set(fmt_pct(k["winrate"], 1), "positive return bars", GREEN if k["winrate"] >= 50 else RED)
        self.kpi_avg_range.set(fmt_pct(k["avg_range"]), "mean candle range", FG)
        self.kpi_active_hours.set(str(k["active_hours"]), "unique UTC hours", FG)
        self.kpi_dominant.set(str(k["dominant_session"]), "highest bars", CYAN)

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
        self.table.delete(*self.table.get_children())
        if self.session_df.empty:
            return

        for _, row in self.session_df.iterrows():
            avg_return = row.get("avg_return_pct", np.nan)
            tag = "pos" if pd.notna(avg_return) and avg_return >= 0 else "neg"
            values = (
                row.get("session", "-"),
                f"{int(row.get('bars', 0)):,}",
                fmt_pct(row.get("avg_return_pct")),
                fmt_pct(row.get("winrate_pct"), 1),
                fmt_pct(row.get("avg_range_pct")),
                fmt_pct(row.get("avg_body_pct")),
                fmt_pct(row.get("total_return_pct")),
            )
            self.table.insert("", "end", values=values, tags=(tag,))

        self.matrix_df = SessionBehaviorEngine(self.df).weekday_session_matrix(metric=self._metric_col())
        self._plot_all()
        self._render_matrix()

    def update_details(self) -> None:
        selection = self.table.selection()
        if not selection:
            self._write_details(self._base_details())
            return
        values = self.table.item(selection[0], "values")
        labels = ["Session", "Bars", "Avg Return", "Win Rate", "Avg Range", "Avg Body", "Total Return"]
        text = self._base_details() + "\n\nSELECTED SESSION\n" + "-" * 40 + "\n"
        for label, value in zip(labels, values):
            text += f"{label:18}: {value}\n"
        self._write_details(text)

    def _base_details(self) -> str:
        path = self.repository.root / self.timeframe_var.get() / f"{self.symbol_var.get()}.parquet"
        return (
            "MARKET SESSION BEHAVIOR\n"
            + "-" * 50 + "\n"
            + f"Symbol            : {self.symbol_var.get()}\n"
            + f"Timeframe         : {self.timeframe_var.get()}\n"
            + f"Data Root         : {self.repository.root}\n"
            + f"File              : {path}\n"
            + f"Rows              : {len(self.df):,}\n"
            + f"Metric            : {self.metric_var.get()}\n"
            + "\nSESSION LOGIC UTC\n"
            + "-" * 50 + "\n"
            + "ASIA             : 00:00-06:59\n"
            + "EU               : 07:00-12:59\n"
            + "EU_US_OVERLAP    : 13:00-15:59\n"
            + "US               : 16:00-20:59\n"
            + "OTHER            : 21:00-23:59\n"
            + "\nFLOW\n"
            + "-" * 50 + "\n"
            + "OHLC → Returns → Session Buckets → Session Stats → Hour Stats → Weekday/Session Matrix\n"
        )

    def _write_details(self, text: str) -> None:
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("end", text)
        self.details.configure(state="disabled")

    def _show_error(self, exc: Exception) -> None:
        for card in [self.session_card, self.hour_card, self.open_close_card]:
            card.reset()
            card.ax.text(0.5, 0.5, "LOAD ERROR", color=RED, ha="center", va="center", transform=card.ax.transAxes)
            card.canvas.draw_idle()

        self.summary_var.set(str(exc))
        self.kpi_bars.set("-", "error", RED)
        self._write_details(
            "LOAD ERROR\n"
            + "-" * 80 + "\n"
            + str(exc)
            + "\n\nTRACEBACK\n"
            + "-" * 80 + "\n"
            + self.error_trace
        )

    # ========================================================
    # PLOTS
    # ========================================================

    def _plot_all(self) -> None:
        self._plot_session()
        self._plot_hour()
        self._plot_open_close()

    def _plot_session(self) -> None:
        card = self.session_card
        card.reset()
        if self.session_df.empty:
            card.ax.text(0.5, 0.5, "NO DATA", color=MUTED, ha="center", va="center", transform=card.ax.transAxes)
            card.canvas.draw_idle()
            return

        metric_col = self._metric_col()
        x = self.session_df["session"].astype(str).tolist()
        y = pd.to_numeric(self.session_df[metric_col], errors="coerce").fillna(0.0).values
        colors = [GREEN if v >= 0 else RED for v in y]

        card.ax.bar(x, y, color=colors, alpha=0.85)
        card.ax.axhline(0, color=GRID, linewidth=1.0)
        card.ax.set_ylabel(self.metric_var.get(), fontsize=7)
        card.ax.tick_params(axis="x", rotation=25)
        style_ax(card.ax)
        card.fig.tight_layout(pad=1.0)
        card.canvas.draw_idle()

    def _plot_hour(self) -> None:
        card = self.hour_card
        card.reset()
        if self.hour_df.empty:
            card.ax.text(0.5, 0.5, "NO DATA", color=MUTED, ha="center", va="center", transform=card.ax.transAxes)
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

    def _plot_open_close(self) -> None:
        card = self.open_close_card
        card.reset()
        if self.open_close_df.empty:
            card.ax.text(0.5, 0.5, "NO SESSION OPEN/CLOSE DATA", color=MUTED, ha="center", va="center", transform=card.ax.transAxes)
            card.canvas.draw_idle()
            return

        g = self.open_close_df.groupby("session").agg(
            avg_return_pct=("return_pct", "mean"),
            winrate=("direction", lambda x: float((x == "UP").mean()) * 100.0 if len(x) else np.nan),
        ).reset_index()

        order = {s: i for i, s in enumerate(SESSION_ORDER)}
        g["_order"] = g["session"].map(order).fillna(999)
        g = g.sort_values("_order")

        x = g["session"].astype(str).tolist()
        y = g["avg_return_pct"].fillna(0.0).values
        colors = [GREEN if v >= 0 else RED for v in y]

        card.ax.bar(x, y, color=colors, alpha=0.85)
        card.ax.axhline(0, color=GRID, linewidth=1.0)
        card.ax.set_ylabel("Open→Close %", fontsize=7)
        card.ax.tick_params(axis="x", rotation=25)
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
        tk.Label(header, text="Day", bg=CARD_DARK, fg=MUTED, font=FONT_TINY, width=12, anchor="center").pack(side="left", padx=1, pady=1)
        for sess in SESSION_ORDER:
            tk.Label(header, text=sess, bg=CARD_DARK, fg=MUTED, font=FONT_TINY, width=15, anchor="center").pack(side="left", padx=1, pady=1)

        for day, row in self.matrix_df.iterrows():
            line = tk.Frame(self.matrix_body, bg=CARD)
            line.pack(fill="x")
            tk.Label(line, text=str(day), bg=CARD_DARK, fg=FG, font=FONT_TINY, width=12, anchor="center").pack(side="left", padx=1, pady=1)

            for sess in SESSION_ORDER:
                val = row.get(sess, np.nan)
                try:
                    v = float(val)
                except Exception:
                    v = np.nan

                if pd.isna(v):
                    bg = CARD_DARK
                    fg = SUBTLE
                    txt = "-"
                else:
                    metric = self.metric_var.get()
                    if metric == "Win Rate":
                        bg = "#063018" if v >= 50 else "#3A0808"
                        fg = GREEN if v >= 50 else RED
                    else:
                        bg = "#063018" if v >= 0 else "#3A0808"
                        fg = GREEN if v >= 0 else RED
                    txt = f"{v:.2f}%"

                tk.Label(line, text=txt, bg=bg, fg=fg, font=FONT_TINY, width=15, anchor="center").pack(side="left", padx=1, pady=1)

    # ========================================================
    # EXPORT
    # ========================================================

    def export_csv(self) -> None:
        if self.session_df.empty:
            messagebox.showinfo("Export CSV", "No session data loaded.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Session Behavior",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"session_behavior_{self.symbol_var.get()}_{self.timeframe_var.get()}.csv",
        )
        if not path:
            return

        out = self.session_df.copy()
        out.insert(0, "symbol", self.symbol_var.get())
        out.insert(1, "timeframe", self.timeframe_var.get())
        out.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("Export CSV", f"Exported:\n{path}")


# ============================================================
# REQUIRED API
# ============================================================

def load_data(*args, **kwargs):
    repo = kwargs.get("repository") or OHLCRepository()
    timeframe = kwargs.get("timeframe", "H1")
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
    return SessionBehaviorDashboard(parent, repository=repository, **kwargs)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":
    root = tk.Tk()
    root.title("QUANT - Market Session Behavior Dashboard")
    root.configure(bg=BG)
    root.geometry("1480x900")
    panel = build_panel(root)
    panel.pack(fill="both", expand=True)
    root.mainloop()

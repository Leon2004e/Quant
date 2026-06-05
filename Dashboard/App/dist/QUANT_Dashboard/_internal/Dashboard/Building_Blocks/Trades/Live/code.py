"""
QUANT/Dashboard/Building_Blocks/Trades/Live/code.py

Live Trades Performance Dashboard Building Block

# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: dashboard_trades_live_account_overview
# script_name: Live Trades Performance Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Trades
# asset_type: Dashboard
# purpose: Account-style live trades overview dashboard with equity curve, KPI cards, symbol donut, calendar, daily PnL, strategy ranking, heatmap, symbol breakdown, live portfolio builder, navigation-opened trade list, registry view and visual flow.
# inputs:
#   - Data_Center/Data/2_Baseline/Trades/Live/account_*/*/trades.db
#   - Data_Center/Data/2_Baseline/Trades/Live/account_*/mapping_diagnostics.db
#   - Data_Center/Data/6_Code_Registry/code_registry.db
#   - Data_Center/Backend_Management/6_Code_Registry/code_registry.db
# outputs:
#   - Dashboard UI
#   - Trade List window
#   - Registry window
#   - CSV exports
#   - Live portfolio JSON/CSV outputs
# dependencies:
#   - tkinter
#   - ttk
#   - pathlib
#   - sqlite3
#   - pandas
#   - matplotlib
#   - calendar
#   - json
#   - math
#   - re
# schedule: manual
# version: v1.3.0_overview_preserved_full_registry
# last_reviewed: 2026-06-03
# required_api:
#   - build_panel(parent, repository=None, **kwargs)
# expected_location:
#   - QUANT/Dashboard/Building_Blocks/Trades/Live/code.py
# data_source:
#   - Data_Center/Data/2_Baseline/Trades/Live
# scanner:
#   - Data_Center/Backend_Management/2_Baseline/Trades/live_strategy_processor.py
# ============================================================
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import calendar
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.dates as mdates
import pandas as pd
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk, filedialog, messagebox


# ============================================================
# PATHS
# ============================================================

def find_quant_root(start: Path) -> Path:
    """
    Findet den QUANT-Projektroot robust.

    Funktioniert auch, wenn diese Datei temporär aus Downloads gestartet wird.
    Gesucht wird ein Ordner, der direkt diese Struktur enthält:
        Data_Center/
        Dashboard/

    Typische gültige QUANT-Root-Beispiele:
        C:/Users/Leon/Desktop/Business_Code/Quant_Structure/FTMO
        .../Quant_Structure/FTMO
    """

    def is_ftmo_root(path: Path) -> bool:
        try:
            return (path / "Data_Center").exists() and (path / "Dashboard").exists()
        except Exception:
            return False

    start = start.resolve()

    candidates: list[Path] = []

    # 1) Normalfall: Datei liegt irgendwo innerhalb von Quant_Structure/FTMO
    candidates.extend([start] + list(start.parents))

    # 2) Aktuelles Arbeitsverzeichnis von VS Code / PowerShell
    try:
        cwd = Path.cwd().resolve()
        candidates.extend([cwd] + list(cwd.parents))
    except Exception:
        pass

    # 3) Bekannte Projektpfade auf deinem Windows-System
    home = Path.home()
    known_bases = [
        home / "Desktop" / "Business_Code",
        home / "Desktop" / "Business_Code" / "QUANT",
        home / "Desktop" / "Business_Code",
        home / "Desktop",
    ]

    for base in known_bases:
        candidates.append(base)
        candidates.append(base / "QUANT")
        candidates.append(base / "QUANT")

    # 4) Duplikate entfernen und direkte Kandidaten prüfen
    seen: set[str] = set()
    clean_candidates: list[Path] = []
    for c in candidates:
        try:
            key = str(c.resolve()).lower()
        except Exception:
            key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        clean_candidates.append(c)

    for c in clean_candidates:
        if is_ftmo_root(c):
            return c.resolve()

    # 5) Fallback-Suche unter Desktop/Business_Code und Desktop, begrenzt
    search_roots = [
        home / "Desktop" / "Business_Code",
        home / "Desktop",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("QUANT"):
                if p.is_dir() and is_ftmo_root(p):
                    return p.resolve()
        except Exception:
            continue

    raise RuntimeError(
        "QUANT root not found. Lege diese Datei in den QUANT-Ordner oder starte sie aus deinem Projekt. "
        f"Start={start} | CWD={Path.cwd()} | Erwartet: .../QUANT/Data_Center und .../QUANT/Dashboard"
    )


SCRIPT_PATH = Path(__file__).resolve()
QUANT_ROOT = find_quant_root(SCRIPT_PATH)
DATA_DIR = QUANT_ROOT / "Data_Center" / "Data"

# Neue zentrale Dashboard-Quelle:
# DEMO kombiniert + LIVE Accounts getrennt, erzeugt durch den neuen Combiner.
FEATURE_ENGINEERED_LIVE_ROOT = DATA_DIR / "2_Baseline" / "Trades" / "Live"

# Kompatibilitätsname für den restlichen bestehenden Dashboard-Code.
# Der Code nutzt intern weiterhin FEATURED_LIVE_ROOT, zeigt aber jetzt auf
# Data/2_Baseline/Trades/Live.
FEATURED_LIVE_ROOT = FEATURE_ENGINEERED_LIVE_ROOT
DEMO_COMBINED_DIR = FEATURED_LIVE_ROOT / "account_DEMO_COMBINED"


def count_trade_dbs(path: Path) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    try:
        return sum(1 for p in path.rglob("trades.db") if p.is_file())
    except Exception:
        return 0


def find_live_trade_roots(start: Path) -> List[Path]:
    """
    Dashboard-Datenquelle.

    Das Monitor Dashboard liest ab jetzt ausschließlich aus:
        Data_Center/Data/2_Baseline/Trades/Live

    Erwartete Output-Struktur des Combiners:
        2_Baseline/Trades/Live/account_*/<strategy_bucket>/trades.db
        2_Baseline/Trades/Live/account_*_LIVE/<strategy_bucket>/trades.db

    DEMO:
        account_DEMO_COMBINED

    LIVE:
        account_*_LIVE getrennt je Account
    """
    if FEATURE_ENGINEERED_LIVE_ROOT.exists():
        return [FEATURE_ENGINEERED_LIVE_ROOT.resolve()]

    # Fallback nur für UI-Diagnose, falls der Combiner noch keinen Output erzeugt hat.
    return [FEATURE_ENGINEERED_LIVE_ROOT]


def find_best_live_root() -> Path:
    return FEATURE_ENGINEERED_LIVE_ROOT


def find_default_account_dir(live_root: Path) -> Path:
    if not live_root.exists():
        return live_root / "account_DEMO_COMBINED"

    accounts = AccountTradeRepository.discover_accounts(live_root) if "AccountTradeRepository" in globals() else []
    if accounts:
        combined = [
            p for p in accounts
            if p.name == "account_DEMO_COMBINED" and count_trade_dbs(p) > 0
        ]
        if combined:
            return combined[0]

        live_accounts = [
            p for p in accounts
            if p.name.startswith("account_") and p.name.endswith("_LIVE")
        ]
        if live_accounts:
            return sorted(live_accounts, key=lambda p: (-count_trade_dbs(p), p.name.lower()))[0]

        return sorted(accounts, key=lambda p: (-count_trade_dbs(p), p.name.lower()))[0]

    return live_root / "account_DEMO_COMBINED"


LIVE_TRADE_ROOTS = find_live_trade_roots(SCRIPT_PATH)


# ============================================================
# CODE_REGISTRY
# ============================================================

CODE_REGISTRY: Dict[str, object] = {
    "script_id": "dashboard_trades_live_account_overview",
    "script_name": "Live Trades Performance Dashboard",
    "owner": "Leon",
    "status": "active",
    "layer": "Dashboard",
    "domain": "Trades",
    "asset_type": "Dashboard",
    "purpose": (
        "Account-style live trades overview dashboard with equity curve, KPI cards, "
        "symbol donut, calendar, daily PnL, strategy ranking, heatmap, symbol breakdown, "
        "live portfolio builder, navigation-opened trade list, registry view and visual flow."
    ),
    "inputs": [
        "Data_Center/Data/2_Baseline/Trades/Live/account_*/*/trades.db",
        "Data_Center/Data/2_Baseline/Trades/Live/account_*/mapping_diagnostics.db",
        "Data_Center/Data/6_Code_Registry/code_registry.db",
        "Data_Center/Backend_Management/6_Code_Registry/code_registry.db",
    ],
    "outputs": [
        "Dashboard UI",
        "Trade List window",
        "Registry window",
        "CSV exports",
        "Live portfolio JSON/CSV outputs",
    ],
    "dependencies": [
        "tkinter",
        "ttk",
        "pathlib",
        "sqlite3",
        "pandas",
        "matplotlib",
        "calendar",
        "json",
        "math",
        "re",
    ],
    "schedule": "manual",
    "version": "v1.3.0_overview_preserved_full_registry",
    "last_reviewed": "2026-06-03",
    "expected_location": "QUANT/Dashboard/Building_Blocks/Trades/Live/code.py",
    "data_source": "Data_Center/Data/2_Baseline/Trades/Live",
    "scanner": "Data_Center/Backend_Management/2_Baseline/Trades/live_strategy_processor.py",
    "required_api": "build_panel(parent, repository=None, **kwargs)",
    "main_class": "AccountMonitorPanel",
    "registry_window_class": "CodeRegistryWindow",
    "trade_list_window_class": "TradeListWindow",
    "notes": [
        "Original account-style overview dashboard is preserved.",
        "Navigation item ⚚ Trades opens a trade list window.",
        "Navigation item ▤ Registry opens a registry metadata window.",
        "Compatible with Dashboard/Main.py through build_panel().",
    ],
}

CODE_REGISTRY_DB_CANDIDATES_REL = [
    Path("Data_Center") / "Data" / "6_Code_Registry" / "code_registry.db",
    Path("Data_Center") / "Backend_Management" / "6_Code_Registry" / "code_registry.db",
    Path("Data_Center") / "Backend_Management" / "Code_Registry" / "code_registry.db",
]

def get_code_registry() -> Dict[str, object]:
    return dict(CODE_REGISTRY)


def find_code_registry_db() -> Optional[Path]:
    for rel in CODE_REGISTRY_DB_CANDIDATES_REL:
        path = QUANT_ROOT / rel
        if path.exists():
            return path
    return None


def read_registry_db_metadata(script_id: str = "dashboard_trades_live_account_overview") -> Dict[str, object]:
    meta: Dict[str, object] = {
        "db_path": str(find_code_registry_db()) if find_code_registry_db() else "not found",
        "script": {},
        "inputs": [],
        "outputs": [],
        "dependencies": [],
    }

    db_path = find_code_registry_db()
    if db_path is None or not db_path.exists():
        return meta

    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None

    def _read_records(conn: sqlite3.Connection, query: str) -> List[Dict[str, object]]:
        frame = pd.read_sql_query(query, conn)
        return frame.to_dict("records")

    try:
        with sqlite3.connect(db_path) as conn:
            if _table_exists(conn, "scripts"):
                rows = _read_records(conn, f"SELECT * FROM scripts WHERE script_id='{script_id}' LIMIT 1")
                if rows:
                    meta["script"] = rows[0]
            if _table_exists(conn, "script_inputs"):
                meta["inputs"] = _read_records(conn, f"SELECT * FROM script_inputs WHERE script_id='{script_id}'")
            if _table_exists(conn, "script_outputs"):
                meta["outputs"] = _read_records(conn, f"SELECT * FROM script_outputs WHERE script_id='{script_id}'")
            if _table_exists(conn, "script_dependencies"):
                meta["dependencies"] = _read_records(conn, f"SELECT * FROM script_dependencies WHERE script_id='{script_id}'")
    except Exception as exc:
        meta["error"] = str(exc)

    return meta

    def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None

    def _read_records(conn: sqlite3.Connection, query: str) -> List[Dict[str, object]]:
        frame = pd.read_sql_query(query, conn)
        return frame.to_dict("records")

    try:
        with sqlite3.connect(db_path) as conn:
            if _table_exists(conn, "scripts"):
                rows = _read_records(conn, f"SELECT * FROM scripts WHERE script_id='{script_id}' LIMIT 1")
                if rows:
                    meta["script"] = rows[0]
            if _table_exists(conn, "script_inputs"):
                meta["inputs"] = _read_records(conn, f"SELECT * FROM script_inputs WHERE script_id='{script_id}'")
            if _table_exists(conn, "script_outputs"):
                meta["outputs"] = _read_records(conn, f"SELECT * FROM script_outputs WHERE script_id='{script_id}'")
            if _table_exists(conn, "script_dependencies"):
                meta["dependencies"] = _read_records(conn, f"SELECT * FROM script_dependencies WHERE script_id='{script_id}'")
    except Exception as exc:
        meta["error"] = str(exc)

    return meta



# ============================================================
# CONFIG
# ============================================================

START_EQUITY = 200000.0
DB_FILENAME = "trades.db"
TRADES_TABLE = "trades"
REFRESH_MS = 300_000


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
BLUE = "#FF9900"
CYAN = "#35FFE2"
PURPLE = "#9B5DE5"
YELLOW = "#FFD400"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_SUB = ("Consolas", 8)
FONT_NAV = ("Consolas", 8, "bold")
FONT_H2 = ("Consolas", 10, "bold")
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_KPI = ("Consolas", 16, "bold")

SYMBOL_COLORS = [
    "#20D6B0", "#4F7DF3", "#FF6868", "#D9468F", "#6E8EDB",
    "#FFC267", "#FFB34D", "#50D7E9", "#9B5DE5", "#A3AAB8",
    "#2DD4BF", "#F97316", "#38BDF8", "#A855F7", "#84CC16",
]


# ============================================================
# DATA
# ============================================================

@dataclass
class DateWindow:
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]


class AccountTradeRepository:
    def __init__(self, root: Optional[Path] = None):
        if root is None:
            accounts = self.discover_accounts()
            if accounts:
                root = sorted(accounts, key=lambda p: (-count_trade_dbs(p), p.name.lower()))[0]
            else:
                root = DEMO_COMBINED_DIR
        self.root = Path(root)

    @staticmethod
    def discover_live_roots() -> List[Path]:
        return find_live_trade_roots(SCRIPT_PATH)

    @staticmethod
    def discover_accounts(live_root: Path = FEATURED_LIVE_ROOT) -> List[Path]:
        accounts: List[Path] = []
        roots = [live_root]

        # If caller passes the fallback/empty root, scan all discovered roots.
        for r in LIVE_TRADE_ROOTS:
            if r not in roots:
                roots.append(r)

        seen = set()
        for root in roots:
            if not root.exists():
                continue
            try:
                for p in root.iterdir():
                    if not p.is_dir() or not p.name.startswith("account_"):
                        continue
                    if count_trade_dbs(p) <= 0:
                        continue
                    key = str(p.resolve()).lower()
                    if key not in seen:
                        seen.add(key)
                        accounts.append(p.resolve())
            except Exception as e:
                print("Could not scan live root:", e)
                continue

        return sorted(
            accounts,
            key=lambda x: (
                0 if x.name == "account_DEMO_COMBINED" else 1,
                -count_trade_dbs(x),
                x.name.lower(),
            ),
        )

    def set_root(self, root: Path) -> None:
        self.root = Path(root)

    def find_trade_dbs(self) -> List[Tuple[str, Path]]:
        """
        New-structure compatible discovery.

        Supports:
            account_x/strategies/STRATEGY/trades.db
            account_x/STRATEGY/trades.db

        Also works if one more nesting layer is introduced later because it
        recursively searches trades.db inside the selected account folder.
        """
        out: List[Tuple[str, Path]] = []

        if not self.root.exists():
            return out

        try:
            for db in sorted(self.root.rglob(DB_FILENAME)):
                if not db.is_file():
                    continue
                parent = db.parent
                if parent.name.startswith("__"):
                    continue
                # bucket name should be the strategy folder, not 'strategies'
                bucket = parent.name
                out.append((bucket, db.resolve()))
        except Exception:
            return out

        seen = set()
        clean: List[Tuple[str, Path]] = []
        for bucket, db in sorted(out, key=lambda x: (x[0], str(x[1]).lower())):
            key = str(db).lower()
            if key not in seen:
                seen.add(key)
                clean.append((bucket, db))
        return clean

    @staticmethod
    def read_trades_db(db_path: Path) -> pd.DataFrame:
        with sqlite3.connect(db_path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type='table'",
                conn,
            )["name"].astype(str).tolist()

            table = TRADES_TABLE if TRADES_TABLE in tables else (tables[0] if tables else TRADES_TABLE)
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)

    def load_all_trades(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for bucket, db_path in self.find_trade_dbs():
            try:
                df = self.read_trades_db(db_path)
                if df.empty:
                    continue
                df["source_bucket"] = bucket
                df["source_db"] = str(db_path)
                frames.append(df)
            except Exception as e:
                print("Could not scan live root:", e)
                continue

        if not frames:
            return pd.DataFrame()

        d = pd.concat(frames, ignore_index=True)

        for col in ["open_time_utc", "close_time_utc"]:
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], errors="coerce", utc=True)

        for col in [
            "entry_price", "exit_price", "price_delta", "volume_in", "volume_out",
            "profit_sum", "swap_sum", "commission_sum", "net_sum",
        ]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0).astype(float)

        for col in ["account_id", "position_id", "magic", "close_ticket"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").astype("Int64")

        for col in ["symbol", "direction", "strategy_id", "canonical_bucket", "source_bucket"]:
            if col in d.columns:
                d[col] = d[col].astype(str).fillna("").str.strip()

        if "direction" in d.columns:
            d["direction"] = d["direction"].str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
        else:
            d["direction"] = "UNKNOWN"

        for col, default in [
            ("symbol", "UNKNOWN"),
            ("strategy_id", ""),
            ("canonical_bucket", ""),
        ]:
            if col not in d.columns:
                d[col] = default
        if "net_sum" not in d.columns:
            d["net_sum"] = 0.0

        if "trade_signature" in d.columns:
            d = d.drop_duplicates(subset=["trade_signature"], keep="first")
        elif all(c in d.columns for c in ["position_id", "close_ticket", "symbol", "close_time_utc"]):
            d = d.drop_duplicates(subset=["position_id", "close_ticket", "symbol", "close_time_utc"], keep="first")

        if "close_time_utc" in d.columns:
            d = d.dropna(subset=["close_time_utc"]).sort_values("close_time_utc")

        return d.reset_index(drop=True)


# ============================================================
# HELPERS
# ============================================================


def sanitize_portfolio_name(raw: str | None) -> str:
    """Return a Windows-safe portfolio filename stem."""
    if raw is None:
        return "portfolio_default"

    value = str(raw).strip()
    if not value:
        return "portfolio_default"

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    safe_name = re.sub(r"_+", "_", safe_name).strip("._")

    reserved = {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
        "LPT6", "LPT7", "LPT8", "LPT9",
    }

    if not safe_name:
        return "portfolio_default"

    if safe_name.upper() in reserved:
        safe_name = f"portfolio_{safe_name}"

    return safe_name[:150].rstrip("._") or "portfolio_default"

def fmt_money(x: object, decimals: int = 2) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "-"

    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.{decimals}f}"


def fmt_num(x: object, decimals: int = 2) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "-"

    if math.isinf(value):
        return "∞" if value > 0 else "-∞"

    if math.isnan(value):
        return "-"

    return f"{value:,.{decimals}f}"


def fmt_pct(x: object, decimals: int = 1) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "-"

    if math.isnan(value):
        return "-"

    return f"{value:.{decimals}f}%"


def pnl_color(x: object) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return MUTED

    return GREEN if value >= 0 else RED


def dd_color(x: object) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return MUTED

    return RED if value < 0 else GREEN


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b == 0 or pd.isna(b):
            return default
        return float(a) / float(b)
    except Exception:
        return default


def clear_children(widget: tk.Widget) -> None:
    for child in widget.winfo_children():
        child.destroy()


def style_ax(ax) -> None:
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color(CARD)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def apply_window(df: pd.DataFrame, window: DateWindow) -> pd.DataFrame:
    if df.empty or "close_time_utc" not in df.columns:
        return df.copy()
    d = df.copy()
    if window.start is not None:
        d = d[d["close_time_utc"] >= window.start]
    if window.end is not None:
        d = d[d["close_time_utc"] <= window.end]
    return d.reset_index(drop=True)


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
            fg=GREEN if active else MUTED,
            font=FONT_NAV,
            padx=8,
            pady=7,
            cursor="hand2",
        )


class KpiSparkCard(Card):
    def __init__(self, parent, title: str):
        super().__init__(parent, bg=CARD, padx=12, pady=9)
        self.value_var = tk.StringVar(value="-")
        self.sub_var = tk.StringVar(value="")

        tk.Label(self.inner, text=title, bg=CARD, fg=MUTED, font=FONT_SMALL).pack(anchor="w")
        self.value_label = tk.Label(self.inner, textvariable=self.value_var, bg=CARD, fg=FG, font=FONT_KPI)
        self.value_label.pack(anchor="w", pady=(2, 0))
        self.sub_label = tk.Label(self.inner, textvariable=self.sub_var, bg=CARD, fg=MUTED, font=FONT_TINY)
        self.sub_label.pack(anchor="w")

    def set(self, value: str, sub: str = "", value_color: str = FG) -> None:
        self.value_var.set(value)
        self.sub_var.set(sub)
        self.value_label.configure(fg=value_color)
        self.sub_label.configure(fg=GREEN if sub.startswith("▲") else RED if sub.startswith("▼") else MUTED)


class ChartCard(Card):
    def __init__(self, parent, title: str, height: float = 2.6, header_builder=None):
        super().__init__(parent, bg=CARD, padx=12, pady=9)

        self.head = tk.Frame(self.inner, bg=CARD)
        self.head.pack(fill="x")
        tk.Label(self.head, text=title, bg=CARD, fg=FG, font=FONT_H2).pack(side="left")
        if header_builder is not None:
            header_builder(self.head)

        self.fig = Figure(figsize=(5.2, height), dpi=100)
        self.fig.patch.set_facecolor(CARD)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.inner)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))

    def reset(self) -> None:
        self.ax.clear()
        self.fig.patch.set_facecolor(CARD)
        self.ax.set_facecolor(CARD)



# ============================================================
# TRADE LIST / REGISTRY WINDOWS
# ============================================================

class TradeListWindow(tk.Toplevel):
    def __init__(self, parent: tk.Widget, trades: pd.DataFrame, account_name: str = ""):
        super().__init__(parent)
        self.parent_panel = parent
        self.trades = trades.copy() if isinstance(trades, pd.DataFrame) else pd.DataFrame()
        self.filtered = self.trades.copy()
        self.account_name = account_name

        self.search_var = tk.StringVar(value="")
        self.symbol_var = tk.StringVar(value="All Symbols")
        self.strategy_var = tk.StringVar(value="All Strategies")
        self.direction_var = tk.StringVar(value="All Directions")
        self.summary_var = tk.StringVar(value="")

        self.sort_col: Optional[str] = None
        self.sort_reverse = False

        self.title(f"Trade List · {account_name or 'Live Trades'}")
        self.configure(bg=BG)
        self.minsize(1180, 720)
        self.geometry("1420x820")

        self._prepare_columns()
        self._build()
        self._refresh_filters()
        self.apply_filters()

    def _prepare_columns(self) -> None:
        preferred = [
            "close_time_utc", "open_time_utc", "symbol", "direction", "strategy_id",
            "canonical_bucket", "source_bucket", "volume_in", "entry_price", "exit_price",
            "net_sum", "profit_sum", "swap_sum", "commission_sum", "magic",
            "position_id", "close_ticket", "comment_last", "source_db",
        ]
        existing = [c for c in preferred if c in self.trades.columns]
        extra = [c for c in self.trades.columns if c not in existing]
        self.columns = existing + extra[:20]

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = tk.Frame(self, bg=BG)
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        header.columnconfigure(0, weight=1)

        tk.Label(
            header,
            text="Trade List",
            bg=BG,
            fg=FG,
            font=FONT_TITLE,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text=f"Account: {self.account_name or '-'} | Rows: {len(self.trades):,}",
            bg=BG,
            fg=MUTED,
            font=FONT_SUB,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        tk.Button(
            header,
            text="Export CSV",
            bg=CARD,
            fg=GREEN,
            activebackground=CARD_SOFT,
            activeforeground=FG,
            relief="flat",
            font=FONT_SMALL,
            padx=14,
            pady=8,
            command=self.export_csv,
        ).grid(row=0, column=1, rowspan=2, sticky="e", padx=(10, 0))

        filters = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        filters.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 10))
        filters.columnconfigure(1, weight=1)

        tk.Label(filters, text="Search", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=0, padx=(10, 6), pady=8, sticky="w")
        search = tk.Entry(filters, textvariable=self.search_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL)
        search.grid(row=0, column=1, sticky="ew", padx=(0, 8), pady=8)
        search.bind("<KeyRelease>", lambda _e: self.apply_filters())

        self.symbol_combo = ttk.Combobox(filters, textvariable=self.symbol_var, state="readonly", width=18)
        self.symbol_combo.grid(row=0, column=2, padx=6, pady=8)
        self.symbol_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        self.strategy_combo = ttk.Combobox(filters, textvariable=self.strategy_var, state="readonly", width=24)
        self.strategy_combo.grid(row=0, column=3, padx=6, pady=8)
        self.strategy_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        self.direction_combo = ttk.Combobox(filters, textvariable=self.direction_var, state="readonly", width=15)
        self.direction_combo.grid(row=0, column=4, padx=6, pady=8)
        self.direction_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        tk.Label(filters, textvariable=self.summary_var, bg=CARD, fg=FG, font=FONT_SMALL).grid(row=0, column=5, padx=10, pady=8)

        table_wrap = tk.Frame(self, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        table_wrap.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
        table_wrap.columnconfigure(0, weight=1)
        table_wrap.rowconfigure(0, weight=1)

        self.tree = ttk.Treeview(table_wrap, columns=self.columns, show="headings", style="Dark.Treeview")
        for col in self.columns:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_by(c))
            width = 170 if "time" in col else 250 if col in {"comment_last", "source_db", "canonical_bucket", "source_bucket"} else 115
            self.tree.column(col, width=width, anchor="w", stretch=True)

        ysb = tk.Scrollbar(table_wrap, orient="vertical", command=self.tree.yview)
        xsb = tk.Scrollbar(table_wrap, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")

        self.tree.bind("<Double-1>", self.copy_selected_to_clipboard)
        self.tree.tag_configure("pos", foreground=GREEN)
        self.tree.tag_configure("neg", foreground=RED)
        self.tree.tag_configure("flat", foreground=FG)

    def _refresh_filters(self) -> None:
        if self.trades.empty:
            symbols = ["All Symbols"]
            strategies = ["All Strategies"]
            directions = ["All Directions"]
        else:
            symbols = ["All Symbols"]
            if "symbol" in self.trades.columns:
                symbols += sorted([x for x in self.trades["symbol"].dropna().astype(str).unique() if x and x.lower() not in {"nan", "none"}])
            elif "symbol_norm" in self.trades.columns:
                symbols += sorted([x for x in self.trades["symbol_norm"].dropna().astype(str).unique() if x and x.lower() not in {"nan", "none"}])

            strategies = ["All Strategies"]
            if "strategy_id" in self.trades.columns:
                strategies += sorted([x for x in self.trades["strategy_id"].dropna().astype(str).unique() if x and x.lower() not in {"nan", "none"}])

            directions = ["All Directions"]
            if "direction" in self.trades.columns:
                directions += sorted([x for x in self.trades["direction"].dropna().astype(str).str.upper().unique() if x and x.lower() not in {"nan", "none"}])

        self.symbol_combo["values"] = symbols
        self.strategy_combo["values"] = strategies
        self.direction_combo["values"] = directions

    def apply_filters(self) -> None:
        d = self.trades.copy()
        q = self.search_var.get().strip().lower()

        if not d.empty:
            if self.symbol_var.get() != "All Symbols":
                sym_col = "symbol" if "symbol" in d.columns else "symbol_norm" if "symbol_norm" in d.columns else None
                if sym_col:
                    d = d[d[sym_col].astype(str) == self.symbol_var.get()]

            if self.strategy_var.get() != "All Strategies" and "strategy_id" in d.columns:
                d = d[d["strategy_id"].astype(str) == self.strategy_var.get()]

            if self.direction_var.get() != "All Directions" and "direction" in d.columns:
                d = d[d["direction"].astype(str).str.upper() == self.direction_var.get()]

            if q:
                mask = d.astype(str).apply(lambda row: q in " ".join(row.values).lower(), axis=1)
                d = d[mask]

        self.filtered = d.reset_index(drop=True)
        self.update_table()

    def update_table(self) -> None:
        self.tree.delete(*self.tree.get_children())

        d = self.filtered.copy()
        if self.sort_col and self.sort_col in d.columns:
            try:
                d["_sort_key"] = pd.to_numeric(d[self.sort_col], errors="coerce")
                if d["_sort_key"].isna().all():
                    d["_sort_key"] = d[self.sort_col].astype(str)
                d = d.sort_values("_sort_key", ascending=not self.sort_reverse).drop(columns=["_sort_key"])
            except Exception:
                d = d.sort_values(self.sort_col, ascending=not self.sort_reverse)

        for i, row in d.iterrows():
            values = []
            for col in self.columns:
                value = row.get(col, "")
                if col in {"net_sum", "profit_sum", "swap_sum", "commission_sum"}:
                    value = fmt_money(value)
                values.append(value)

            pnl = float(row.get("net_sum", 0.0)) if "net_sum" in row else 0.0
            tag = "pos" if pnl > 0 else "neg" if pnl < 0 else "flat"
            self.tree.insert("", "end", iid=str(i), values=values, tags=(tag,))

        self.summary_var.set(f"{len(d):,} / {len(self.trades):,} rows")

    def sort_by(self, col: str) -> None:
        if self.sort_col == col:
            self.sort_reverse = not self.sort_reverse
        else:
            self.sort_col = col
            self.sort_reverse = False
        self.update_table()

    def export_csv(self) -> None:
        if self.filtered.empty:
            messagebox.showinfo("Export CSV", "No rows to export.")
            return

        path = filedialog.asksaveasfilename(
            title="Export Trade List",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"trade_list_{self.account_name or 'live'}.csv",
        )
        if not path:
            return

        self.filtered.to_csv(path, index=False, encoding="utf-8-sig")

    def copy_selected_to_clipboard(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
            if idx < 0 or idx >= len(self.filtered):
                return
            row = self.filtered.iloc[idx].to_dict()
        except Exception:
            return

        text = "\n".join(f"{k}: {v}" for k, v in row.items())
        self.clipboard_clear()
        self.clipboard_append(text)


class CodeRegistryWindow(tk.Toplevel):
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.title("Code Registry · Live Trades Dashboard")
        self.configure(bg=BG)
        self.minsize(900, 650)
        self.geometry("980x720")

        meta = read_registry_db_metadata()

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=16)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        tk.Label(body, text="CODE REGISTRY", bg=BG, fg=FG, font=FONT_TITLE).grid(row=0, column=0, sticky="w", pady=(0, 10))

        txt = tk.Text(body, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=("Consolas", 9), wrap="word")
        ysb = tk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=ysb.set)

        txt.grid(row=1, column=0, sticky="nsew")
        ysb.grid(row=1, column=1, sticky="ns")

        txt.insert("end", "CODE_REGISTRY\n", "header")
        txt.insert("end", "-" * 90 + "\n")
        for key, value in CODE_REGISTRY.items():
            txt.insert("end", f"{key:28}: {value}\n")

        txt.insert("end", "\nREGISTRY DB\n", "header")
        txt.insert("end", "-" * 90 + "\n")
        txt.insert("end", f"path: {meta.get('db_path')}\n")

        if meta.get("script"):
            txt.insert("end", "\nSCRIPT TABLE\n", "header")
            txt.insert("end", "-" * 90 + "\n")
            for key, value in dict(meta["script"]).items():
                txt.insert("end", f"{key:28}: {value}\n")

        for section in ["inputs", "outputs", "dependencies"]:
            txt.insert("end", f"\n{section.upper()}\n", "header")
            txt.insert("end", "-" * 90 + "\n")
            items = meta.get(section, [])
            if not items:
                txt.insert("end", "-\n")
            else:
                for item in items:
                    txt.insert("end", f"{item}\n")

        if meta.get("error"):
            txt.insert("end", "\nERROR\n", "header")
            txt.insert("end", str(meta["error"]))

        txt.tag_configure("header", foreground=FG, font=("Consolas", 10, "bold"))
        txt.configure(state="disabled")



# ============================================================
# DASHBOARD
# ============================================================

class AccountMonitorPanel(tk.Frame):
    def __init__(self, parent, repo: Optional[AccountTradeRepository] = None, **_kwargs):
        super().__init__(parent, bg=BG)

        self.repo = repo or AccountTradeRepository()
        self.account_dirs: List[Path] = AccountTradeRepository.discover_accounts()
        self.account_name_to_path: Dict[str, Path] = {p.name: p for p in self.account_dirs}

        if self.account_dirs:
            resolved_accounts = {str(p.resolve()): p for p in self.account_dirs}
            current_key = str(self.repo.root.resolve()) if self.repo.root.exists() else str(self.repo.root)
            if current_key not in resolved_accounts:
                best = sorted(self.account_dirs, key=lambda p: (-count_trade_dbs(p), p.name.lower()))[0]
                self.repo.set_root(best)

        self.raw_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.symbol_colors: Dict[str, str] = {}

        now = pd.Timestamp.now(tz="UTC")
        self.window = DateWindow(start=None, end=None)
        self.selected_account = self.repo.root.name
        self.selected_symbol = "All Symbols"
        self.selected_strategy = "All Strategies"
        self.active_strategy_highlight: Optional[str] = None

        self.symbol_var = tk.StringVar(value="All Symbols")
        self.strategy_var = tk.StringVar(value="All Strategies")
        self.strategy_sort_mode = tk.StringVar(value="Net PnL")

        self.equity_range = "ALL"
        self.equity_custom_start: Optional[pd.Timestamp] = None
        self.equity_custom_end: Optional[pd.Timestamp] = None
        self.equity_date_points: List[pd.Timestamp] = []
        self.equity_available_var = tk.StringVar(value="Available: -")
        self.equity_selected_var = tk.StringVar(value="Selected: All")
        self.equity_start_var = tk.StringVar(value="")
        self.equity_end_var = tk.StringVar(value="")
        self.equity_stats_var = tk.StringVar(value="Equity Stats: -")

        self.calendar_year = int(now.year)
        self.calendar_month = int(now.month)

        self._build_ui()
        self.refresh()
        self.after(REFRESH_MS, self._auto_refresh)

    # --------------------------------------------------------
    # BUILD

    def _build_ui(self) -> None:
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=20, pady=16)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=1)

        self._build_header(root)
        self._build_nav(root)
        self._build_kpis(root)
        self._build_dashboard_grid(root)

    def _build_header(self, parent) -> None:
        header = tk.Frame(parent, bg=BG)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        left = tk.Frame(header, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Live Trades Performance Dashboard", bg=BG, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(left, text=f"Source: {FEATURED_LIVE_ROOT}  | DBs: {count_trade_dbs(FEATURED_LIVE_ROOT)}", bg=BG, fg=MUTED, font=FONT_SUB).pack(anchor="w", pady=(1, 0))

        right = tk.Frame(header, bg=BG)
        right.grid(row=0, column=1, sticky="e")

        self.date_btn = tk.Button(
            right,
            text="Latest Month  📅",
            bg=CARD,
            fg=FG,
            activebackground=CARD_SOFT,
            activeforeground=FG,
            relief="flat",
            font=FONT_SMALL,
            padx=14,
            pady=8,
            command=self.reset_to_latest_month,
        )
        self.date_btn.pack(side="left", padx=(0, 10))

        self.account_var = tk.StringVar(value=self.selected_account)
        self.account_combo = ttk.Combobox(right, textvariable=self.account_var, state="readonly", width=30)
        self.account_combo["values"] = list(self.account_name_to_path.keys()) or [self.selected_account]
        self.account_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_account_change())
        self.account_combo.pack(side="left", padx=(0, 10))

        self.symbol_combo = ttk.Combobox(right, textvariable=self.symbol_var, state="readonly", width=18)
        self.symbol_combo["values"] = ["All Symbols"]
        self.symbol_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_symbol_change())
        self.symbol_combo.pack(side="left", padx=(0, 10))

        self.strategy_combo = ttk.Combobox(right, textvariable=self.strategy_var, state="readonly", width=30)
        self.strategy_combo["values"] = ["All Strategies"]
        self.strategy_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_strategy_change())
        self.strategy_combo.pack(side="left")

        self.live_portfolio_btn = tk.Button(
            right,
            text="Live Portfolio",
            bg=CARD,
            fg=GREEN,
            activebackground=CARD_SOFT,
            activeforeground=FG,
            relief="flat",
            font=FONT_SMALL,
            padx=14,
            pady=8,
            command=self.open_live_portfolio_builder,
        )
        self.live_portfolio_btn.pack(side="left", padx=(10, 0))

    def _build_nav(self, parent) -> None:
        nav = tk.Frame(parent, bg=BG)
        nav.grid(row=1, column=0, sticky="ew", pady=(10, 12))
        labels = ["⌂  Overview", "♙  Strategies", "♢  Risk", "⚚  Trades", "▣  Calendar", "▧  Reports", "▤  Registry", "⚙  Settings"]
        for i, label in enumerate(labels):
            btn = NavButton(nav, label, active=(i == 0))
            btn.pack(side="left", padx=(0, 18))
            if "Trades" in label:
                btn.bind("<Button-1>", lambda _e: self.open_trade_list_window())
            elif "Registry" in label:
                btn.bind("<Button-1>", lambda _e: self.open_code_registry_window())
        tk.Frame(parent, bg=BORDER, height=1).grid(row=2, column=0, sticky="ew", pady=(0, 14))

    def _build_kpis(self, parent) -> None:
        row = tk.Frame(parent, bg=BG)
        row.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        for i in range(6):
            row.columnconfigure(i, weight=1, uniform="kpi")

        self.kpi_net = KpiSparkCard(row, "Net Profit")
        self.kpi_trades = KpiSparkCard(row, "Total Trades")
        self.kpi_winrate = KpiSparkCard(row, "Win Rate")
        self.kpi_pf = KpiSparkCard(row, "Profit Factor")
        self.kpi_dd = KpiSparkCard(row, "Max Drawdown")
        self.kpi_sharpe = KpiSparkCard(row, "Sharpe Ratio")

        for i, card in enumerate([self.kpi_net, self.kpi_trades, self.kpi_winrate, self.kpi_pf, self.kpi_dd, self.kpi_sharpe]):
            card.grid(row=0, column=i, sticky="nsew", padx=5)

    def _build_dashboard_grid(self, parent) -> None:
        grid = tk.Frame(parent, bg=BG)
        grid.grid(row=4, column=0, sticky="nsew")
        for c in range(12):
            grid.columnconfigure(c, weight=1, uniform="dash")
        for r in range(8):
            grid.rowconfigure(r, weight=1, uniform="dash")

        self.equity_card = ChartCard(grid, "Equity Curve", height=2.8, header_builder=self._build_equity_range_buttons)
        self.equity_card.grid(row=0, column=0, rowspan=3, columnspan=5, sticky="nsew", padx=5, pady=5)

        self.symbol_donut_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.symbol_donut_card.grid(row=0, column=5, rowspan=3, columnspan=3, sticky="nsew", padx=5, pady=5)
        self._build_symbol_donut_shell()

        self.calendar_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.calendar_card.grid(row=0, column=8, rowspan=4, columnspan=4, sticky="nsew", padx=5, pady=(5, 2))
        self._build_calendar_shell()

        self.daily_card = ChartCard(grid, "Daily PnL", height=2.7)
        self.daily_card.grid(row=3, column=0, rowspan=2, columnspan=8, sticky="nsew", padx=5, pady=5)

        self.strategy_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.strategy_card.grid(row=5, column=0, rowspan=3, columnspan=3, sticky="nsew", padx=5, pady=5)
        self._build_strategy_rank_shell()

        self.heatmap_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.heatmap_card.grid(row=5, column=3, rowspan=3, columnspan=5, sticky="nsew", padx=5, pady=5)
        self._build_heatmap_shell()

        self.top_card = Card(grid, bg=CARD, padx=12, pady=9)
        self.top_card.grid(row=4, column=8, rowspan=4, columnspan=4, sticky="nsew", padx=5, pady=(2, 5))
        self._build_top_table_shell()

    # --------------------------------------------------------
    # SHELLS

    def _build_equity_range_buttons(self, parent) -> None:
        """Visual equity stats: separate compact metric pills instead of one hard-to-read text line."""
        self.equity_stats_frame = tk.Frame(parent, bg=CARD)
        self.equity_stats_frame.pack(side="left", padx=(14, 8), fill="x", expand=True)

        self.equity_stat_vars: Dict[str, tk.StringVar] = {}
        self.equity_stat_value_labels: Dict[str, tk.Label] = {}

        for key, label in [
            ("high", "High"),
            ("low", "Low"),
            ("end", "End"),
            ("pnl", "PnL"),
            ("dd", "Max DD"),
        ]:
            pill = tk.Frame(
                self.equity_stats_frame,
                bg=CARD_DARK,
                highlightthickness=1,
                highlightbackground="#1D2A37",
            )
            pill.pack(side="left", padx=(0, 5), fill="x", expand=True)

            tk.Label(
                pill,
                text=label,
                bg=CARD_DARK,
                fg=SUBTLE,
                font=("Segoe UI", 6),
                anchor="w",
            ).pack(anchor="w", padx=6, pady=(2, 0))

            var = tk.StringVar(value="-")
            value = tk.Label(
                pill,
                textvariable=var,
                bg=CARD_DARK,
                fg=FG,
                font=("Segoe UI", 8, "bold"),
                anchor="w",
            )
            value.pack(anchor="w", padx=6, pady=(0, 3))

            self.equity_stat_vars[key] = var
            self.equity_stat_value_labels[key] = value

        wrap = tk.Frame(parent, bg=CARD)
        wrap.pack(side="right")

        info = tk.Frame(wrap, bg=CARD)
        info.pack(side="left", padx=(0, 8))
        tk.Label(info, textvariable=self.equity_available_var, bg=CARD, fg=SUBTLE, font=FONT_TINY).pack(anchor="e")
        tk.Label(info, textvariable=self.equity_selected_var, bg=CARD, fg=MUTED, font=FONT_TINY).pack(anchor="e")

        slider_wrap = tk.Frame(wrap, bg=CARD)
        slider_wrap.pack(side="left", padx=(0, 8))

        tk.Label(slider_wrap, text="Start", bg=CARD, fg=MUTED, font=FONT_TINY).grid(row=0, column=0, sticky="w")
        self.equity_start_scale = tk.Scale(
            slider_wrap,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            length=100,
            resolution=1,
            bg=CARD,
            fg=MUTED,
            troughcolor=CARD_DARK,
            activebackground=GREEN,
            highlightthickness=0,
            bd=0,
            command=lambda _v: self.on_equity_slider_change(),
        )
        self.equity_start_scale.grid(row=0, column=1, sticky="ew", padx=(4, 6))
        self.equity_start_entry = tk.Entry(slider_wrap, textvariable=self.equity_start_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_TINY, width=10)
        self.equity_start_entry.grid(row=0, column=2, sticky="ew")
        self.equity_start_entry.bind("<Return>", lambda _e: self.apply_equity_text_range())
        self.equity_start_entry.bind("<FocusOut>", lambda _e: self.apply_equity_text_range(silent=True))

        tk.Label(slider_wrap, text="End", bg=CARD, fg=MUTED, font=FONT_TINY).grid(row=1, column=0, sticky="w")
        self.equity_end_scale = tk.Scale(
            slider_wrap,
            from_=0,
            to=0,
            orient="horizontal",
            showvalue=False,
            length=100,
            resolution=1,
            bg=CARD,
            fg=MUTED,
            troughcolor=CARD_DARK,
            activebackground=GREEN,
            highlightthickness=0,
            bd=0,
            command=lambda _v: self.on_equity_slider_change(),
        )
        self.equity_end_scale.grid(row=1, column=1, sticky="ew", padx=(4, 6))
        self.equity_end_entry = tk.Entry(slider_wrap, textvariable=self.equity_end_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_TINY, width=10)
        self.equity_end_entry.grid(row=1, column=2, sticky="ew")
        self.equity_end_entry.bind("<Return>", lambda _e: self.apply_equity_text_range())
        self.equity_end_entry.bind("<FocusOut>", lambda _e: self.apply_equity_text_range(silent=True))

        self.equity_apply_btn = tk.Button(wrap, text="Apply", bg=CARD_DARK, fg=MUTED, activebackground=CARD_SOFT, activeforeground=FG, relief="flat", font=FONT_TINY, padx=8, pady=2, command=self.apply_equity_text_range)
        self.equity_apply_btn.pack(side="left", padx=(0, 4))

        self.equity_all_btn = tk.Button(wrap, text="All", bg=CARD_DARK, fg=MUTED, activebackground=CARD_SOFT, activeforeground=FG, relief="flat", font=FONT_TINY, padx=8, pady=2, command=self.reset_equity_range_all)
        self.equity_all_btn.pack(side="left")

    def _build_symbol_donut_shell(self) -> None:
        tk.Label(self.symbol_donut_card.inner, text="PnL by Symbol", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        body = tk.Frame(self.symbol_donut_card.inner, bg=CARD)
        body.pack(fill="both", expand=True, pady=(4, 0))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        self.donut_fig = Figure(figsize=(2.4, 2.2), dpi=100)
        self.donut_fig.patch.set_facecolor(CARD)
        self.donut_ax = self.donut_fig.add_subplot(111)
        self.donut_canvas = FigureCanvasTkAgg(self.donut_fig, master=body)
        self.donut_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.symbol_legend = tk.Frame(body, bg=CARD)
        self.symbol_legend.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    def _build_calendar_shell(self) -> None:
        top = tk.Frame(self.calendar_card.inner, bg=CARD)
        top.pack(fill="x")
        tk.Label(top, text="Live Performance Calendar", bg=CARD, fg=FG, font=FONT_H2).pack(side="left")
        tk.Button(top, text="Today", command=self.goto_today, bg=CARD_DARK, fg=FG, relief="flat", font=FONT_SMALL, padx=12).pack(side="right")

        self.period_title_var = tk.StringVar(value="Monthly Stats")
        self.period_pnl_var = tk.StringVar(value="$0.00")
        self.period_trades_var = tk.StringVar(value="Trades: 0")
        self.period_winrate_var = tk.StringVar(value="Win Rate: 0.0%")

        self.period_bar = tk.Frame(self.calendar_card.inner, bg=CARD_DARK, highlightthickness=1, highlightbackground="#1D2A37")
        self.period_bar.pack(fill="x", pady=(8, 0))
        period_inner = tk.Frame(self.period_bar, bg=CARD_DARK)
        period_inner.pack(fill="x", padx=10, pady=7)
        period_inner.columnconfigure(0, weight=1)

        tk.Label(period_inner, textvariable=self.period_title_var, bg=CARD_DARK, fg=MUTED, font=FONT_TINY).grid(row=0, column=0, sticky="w")
        self.period_pnl_label = tk.Label(period_inner, textvariable=self.period_pnl_var, bg=CARD_DARK, fg=FG, font=("Segoe UI", 13, "bold"))
        self.period_pnl_label.grid(row=1, column=0, sticky="w", pady=(1, 0))
        metric_row = tk.Frame(period_inner, bg=CARD_DARK)
        metric_row.grid(row=1, column=1, sticky="e")
        tk.Label(metric_row, textvariable=self.period_trades_var, bg=CARD_DARK, fg=MUTED, font=FONT_TINY).pack(side="left", padx=(0, 12))
        tk.Label(metric_row, textvariable=self.period_winrate_var, bg=CARD_DARK, fg=MUTED, font=FONT_TINY).pack(side="left")

        nav = tk.Frame(self.calendar_card.inner, bg=CARD)
        nav.pack(fill="x", pady=(10, 6))
        tk.Button(nav, text="‹", command=self.prev_month, bg=CARD_SOFT, fg=FG, relief="flat", width=3).pack(side="left")
        self.cal_title = tk.Label(nav, text="-", bg=CARD, fg=FG, font=FONT_H2)
        self.cal_title.pack(side="left", expand=True)
        tk.Button(nav, text="›", command=self.next_month, bg=CARD_SOFT, fg=FG, relief="flat", width=3).pack(side="right")

        self.calendar_grid = tk.Frame(self.calendar_card.inner, bg=CARD)
        self.calendar_grid.pack(fill="both", expand=True)
        tk.Label(self.calendar_card.inner, text="All times are UTC from 2_Baseline/Trades/Live trades.db", bg=CARD, fg=SUBTLE, font=FONT_TINY).pack(anchor="center", pady=(5, 0))

    def _build_strategy_rank_shell(self) -> None:
        title_row = tk.Frame(self.strategy_card.inner, bg=CARD)
        title_row.pack(fill="x")
        tk.Label(title_row, text="PnL by Strategy", bg=CARD, fg=FG, font=FONT_H2).pack(side="left")
        tk.Button(title_row, text="All", command=self.clear_strategy_selection, bg=CARD_DARK, fg=MUTED, activebackground=CARD_SOFT, activeforeground=FG, relief="flat", font=FONT_TINY, padx=8, pady=2).pack(side="right")
        self.strategy_sort_combo = ttk.Combobox(title_row, textvariable=self.strategy_sort_mode, values=["Net PnL", "Trades", "Win Rate"], state="readonly", width=10)
        self.strategy_sort_combo.pack(side="right", padx=(0, 6))
        self.strategy_sort_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_strategy_rank())

        header = tk.Frame(self.strategy_card.inner, bg=CARD)
        header.pack(fill="x", pady=(10, 4))
        tk.Label(header, text="Strategy", bg=CARD, fg=MUTED, font=FONT_TINY, width=18, anchor="w").pack(side="left")
        tk.Label(header, text="Net PnL", bg=CARD, fg=MUTED, font=FONT_TINY, width=10, anchor="e").pack(side="right")
        tk.Label(header, text="Trades", bg=CARD, fg=MUTED, font=FONT_TINY, width=7, anchor="e").pack(side="right")
        tk.Label(header, text="Win Rate", bg=CARD, fg=MUTED, font=FONT_TINY, width=8, anchor="e").pack(side="right")

        self.strategy_rank_scroll_wrap = tk.Frame(self.strategy_card.inner, bg=CARD)
        self.strategy_rank_scroll_wrap.pack(fill="both", expand=True)
        self.strategy_rank_canvas = tk.Canvas(self.strategy_rank_scroll_wrap, bg=CARD, highlightthickness=0, bd=0, relief="flat")
        self.strategy_rank_canvas.pack(side="left", fill="both", expand=True)
        self.strategy_rank_scrollbar = tk.Scrollbar(self.strategy_rank_scroll_wrap, orient="vertical", command=self.strategy_rank_canvas.yview)
        self.strategy_rank_scrollbar.pack(side="right", fill="y")
        self.strategy_rank_canvas.configure(yscrollcommand=self.strategy_rank_scrollbar.set)
        self.strategy_rank_body = tk.Frame(self.strategy_rank_canvas, bg=CARD)
        self.strategy_rank_window = self.strategy_rank_canvas.create_window((0, 0), window=self.strategy_rank_body, anchor="nw")
        self.strategy_rank_body.bind("<Configure>", lambda _e: self.strategy_rank_canvas.configure(scrollregion=self.strategy_rank_canvas.bbox("all")))
        self.strategy_rank_canvas.bind("<Configure>", lambda e: self.strategy_rank_canvas.itemconfigure(self.strategy_rank_window, width=e.width))

    def _build_heatmap_shell(self) -> None:
        tk.Label(self.heatmap_card.inner, text="Monthly Performance Heatmap", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        self.heatmap_body = tk.Frame(self.heatmap_card.inner, bg=CARD)
        self.heatmap_body.pack(fill="both", expand=True, pady=(12, 0))

    def _build_top_table_shell(self) -> None:
        title_row = tk.Frame(self.top_card.inner, bg=CARD)
        title_row.pack(fill="x")
        tk.Label(title_row, text="Calendar Month · Symbol Breakdown", bg=CARD, fg=FG, font=FONT_H2).pack(side="left")
        self.top_table_period_var = tk.StringVar(value="-")
        tk.Label(title_row, textvariable=self.top_table_period_var, bg=CARD, fg=MUTED, font=FONT_TINY).pack(side="right")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Dark.Treeview", background=CARD, foreground=FG, fieldbackground=CARD, borderwidth=0, rowheight=24, font=FONT_SMALL)
        style.configure("Dark.Treeview.Heading", background=CARD, foreground=MUTED, borderwidth=0, font=FONT_TINY)
        style.map("Dark.Treeview", background=[("selected", CARD_SOFT)], foreground=[("selected", FG)])

        cols = ("symbol", "net", "trades", "winrate")
        table_wrap = tk.Frame(self.top_card.inner, bg=CARD)
        table_wrap.pack(fill="both", expand=True, pady=(10, 0))
        self.top_table = ttk.Treeview(table_wrap, columns=cols, show="headings", height=8, style="Dark.Treeview")
        for col, text, width, anchor in [("symbol", "Symbol", 90, "w"), ("net", "Net PnL", 90, "e"), ("trades", "Trades", 70, "center"), ("winrate", "Win Rate", 80, "e")]:
            self.top_table.heading(col, text=text)
            self.top_table.column(col, width=width, anchor=anchor)
        self.top_table.pack(side="left", fill="both", expand=True)
        self.top_table_scrollbar = tk.Scrollbar(table_wrap, orient="vertical", command=self.top_table.yview)
        self.top_table_scrollbar.pack(side="right", fill="y")
        self.top_table.configure(yscrollcommand=self.top_table_scrollbar.set)

    # --------------------------------------------------------
    # REFRESH / FILTERS

    def refresh(self) -> None:
        self.raw_df = self.repo.load_all_trades()
        print(f"LiveTradesDashboard source root: {self.repo.root} | DBs: {len(self.repo.find_trade_dbs())} | Rows: {len(self.raw_df)}")
        self._init_month_window_from_data_if_needed()
        self._refresh_symbol_filter()
        self._refresh_strategy_filter()
        self._apply_filters()
        self._assign_symbol_colors()
        self._update_header_date_text()
        self._update_kpis()
        self._plot_equity()
        self._plot_donut()
        self._plot_daily_pnl()
        self._update_calendar()
        self._update_strategy_rank()
        self._update_heatmap()
        self._update_top_table()

    def _auto_refresh(self) -> None:
        try:
            self.refresh()
        finally:
            self.after(REFRESH_MS, self._auto_refresh)

    def _month_window(self, year: int, month: int) -> DateWindow:
        start = pd.Timestamp(year=int(year), month=int(month), day=1, tz="UTC")
        if month == 12:
            end = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz="UTC") - pd.Timedelta(microseconds=1)
        else:
            end = pd.Timestamp(year=int(year), month=int(month) + 1, day=1, tz="UTC") - pd.Timedelta(microseconds=1)
        return DateWindow(start=start, end=end)

    def _init_month_window_from_data_if_needed(self) -> None:
        if self.window.start is not None and self.window.end is not None:
            return
        if self.raw_df.empty or "close_time_utc" not in self.raw_df.columns:
            now = pd.Timestamp.now(tz="UTC")
            self.calendar_year = int(now.year)
            self.calendar_month = int(now.month)
            self.window = self._month_window(self.calendar_year, self.calendar_month)
            return
        last_ts = pd.to_datetime(self.raw_df["close_time_utc"], utc=True, errors="coerce").dropna().max()
        if pd.isna(last_ts):
            now = pd.Timestamp.now(tz="UTC")
            self.calendar_year = int(now.year)
            self.calendar_month = int(now.month)
        else:
            self.calendar_year = int(last_ts.year)
            self.calendar_month = int(last_ts.month)
        self.window = self._month_window(self.calendar_year, self.calendar_month)

    def _refresh_symbol_filter(self) -> None:
        if self.raw_df.empty or "symbol" not in self.raw_df.columns:
            values = ["All Symbols"]
        else:
            symbols = sorted([s for s in self.raw_df["symbol"].dropna().astype(str).str.strip().unique() if s and s.lower() not in {"nan", "none", "unknown"}])
            values = ["All Symbols"] + symbols
        current = self.symbol_var.get()
        self.symbol_combo["values"] = values
        if current not in values:
            self.symbol_var.set("All Symbols")
            self.selected_symbol = "All Symbols"

    def _symbol_filtered_raw_df(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        d = self.raw_df.copy() if df is None else df.copy()
        selected_symbol = self.symbol_var.get().strip()
        self.selected_symbol = selected_symbol
        if selected_symbol != "All Symbols" and not d.empty and "symbol" in d.columns:
            d = d[d["symbol"].astype(str).str.strip() == selected_symbol]
        return d.reset_index(drop=True)

    def _refresh_strategy_filter(self) -> None:
        base = self._symbol_filtered_raw_df()
        if base.empty:
            values = ["All Strategies"]
        else:
            label_df = self._build_strategy_label_column(base)
            strategies = sorted(label_df["strategy_label"].dropna().astype(str).unique())
            strategies = [s for s in strategies if s and s.lower() not in {"nan", "none", "unknown"}]
            values = ["All Strategies"] + strategies
        current = self.strategy_var.get()
        self.strategy_combo["values"] = values
        if current not in values:
            self.strategy_var.set("All Strategies")
            self.selected_strategy = "All Strategies"
            self.active_strategy_highlight = None

    def _apply_filters(self) -> None:
        d = apply_window(self.raw_df, self.window)
        d = self._symbol_filtered_raw_df(d)
        selected_strategy = self.strategy_var.get()
        self.selected_strategy = selected_strategy
        if selected_strategy != "All Strategies" and not d.empty:
            d = self._build_strategy_label_column(d)
            d = d[d["strategy_label"].astype(str) == selected_strategy]
        self.df = d.reset_index(drop=True)

    def _assign_symbol_colors(self) -> None:
        if self.raw_df.empty or "symbol" not in self.raw_df.columns:
            self.symbol_colors = {}
            return
        symbols = sorted([s for s in self.raw_df["symbol"].dropna().astype(str).unique() if s])
        self.symbol_colors = {s: SYMBOL_COLORS[i % len(SYMBOL_COLORS)] for i, s in enumerate(symbols)}

    def _refresh_views_from_current_filter(self) -> None:
        self._refresh_strategy_filter()
        self._apply_filters()
        self._update_header_date_text()
        self._update_kpis()
        self._plot_equity()
        self._plot_donut()
        self._plot_daily_pnl()
        self._update_calendar()
        self._update_strategy_rank()
        self._update_heatmap()
        self._update_top_table()

    def reset_equity_to_full_range(self) -> None:
        """Reset equity range to full available range.

        Rule: whenever account, symbol, or strategy changes, the dashboard first
        shows the complete equity curve for the newly selected data set. Manual
        Start/End shrinking is only applied afterwards by the user.
        """
        self.equity_range = "ALL"
        self.equity_custom_start = None
        self.equity_custom_end = None
        self.window = DateWindow(start=None, end=None)

        if hasattr(self, "equity_start_var"):
            self.equity_start_var.set("")
        if hasattr(self, "equity_end_var"):
            self.equity_end_var.set("")

        if hasattr(self, "equity_start_scale") and hasattr(self, "equity_end_scale"):
            try:
                self._sync_equity_sliders_to_all()
            except Exception:
                pass

        if hasattr(self, "equity_all_btn") and hasattr(self, "equity_apply_btn"):
            try:
                self._update_equity_range_button_styles()
            except Exception:
                pass

    def _on_account_change(self) -> None:
        account_name = self.account_var.get().strip()
        root = self.account_name_to_path.get(account_name)
        if root is None:
            return

        self.selected_account = account_name
        self.repo.set_root(root)

        self.symbol_var.set("All Symbols")
        self.strategy_var.set("All Strategies")
        self.active_strategy_highlight = None
        self.reset_equity_to_full_range()

        self.refresh()

    def _on_symbol_change(self) -> None:
        self.strategy_var.set("All Strategies")
        self.active_strategy_highlight = None
        self.reset_equity_to_full_range()
        self._refresh_views_from_current_filter()

    def _on_strategy_change(self) -> None:
        self.active_strategy_highlight = (
            None
            if self.strategy_var.get() == "All Strategies"
            else self.strategy_var.get()
        )
        self.reset_equity_to_full_range()
        self._refresh_views_from_current_filter()

    def select_strategy(self, strategy_name: str) -> None:
        if self.active_strategy_highlight == strategy_name:
            self.active_strategy_highlight = None
            self.strategy_var.set("All Strategies")
            self.selected_strategy = "All Strategies"
        else:
            self.active_strategy_highlight = strategy_name
            values = list(self.strategy_combo["values"])
            if strategy_name not in values:
                values.append(strategy_name)
                self.strategy_combo["values"] = values
            self.strategy_var.set(strategy_name)
            self.selected_strategy = strategy_name

        self.reset_equity_to_full_range()
        self._refresh_views_from_current_filter()

    def clear_strategy_selection(self) -> None:
        self.active_strategy_highlight = None
        self.strategy_var.set("All Strategies")
        self.selected_strategy = "All Strategies"
        self.reset_equity_to_full_range()
        self._refresh_views_from_current_filter()

    def open_live_portfolio_builder(self) -> None:
        """Open live portfolio builder for the currently selected monitor account."""
        try:
            if self.raw_df is None or self.raw_df.empty:
                self.refresh()
            LivePortfolioBuilder(
                parent=self,
                trades=self.raw_df.copy(),
                account_name=self.repo.root.name,
                account_root=self.repo.root,
            )
        except Exception as exc:
            win = tk.Toplevel(self)
            win.title("Live Portfolio Error")
            win.configure(bg=BG)
            tk.Label(win, text=str(exc), bg=BG, fg=RED, font=FONT_SMALL, padx=20, pady=20).pack(fill="both", expand=True)


    def open_trade_list_window(self) -> None:
        """Open full trade list from the currently selected account and current filters."""
        try:
            if self.raw_df is None or self.raw_df.empty:
                self.refresh()

            trades = self.df.copy() if isinstance(self.df, pd.DataFrame) and not self.df.empty else self.raw_df.copy()
            TradeListWindow(
                parent=self,
                trades=trades,
                account_name=self.repo.root.name,
            )
        except Exception as exc:
            win = tk.Toplevel(self)
            win.title("Trade List Error")
            win.configure(bg=BG)
            tk.Label(win, text=str(exc), bg=BG, fg=RED, font=FONT_SMALL, padx=20, pady=20).pack(fill="both", expand=True)

    def open_code_registry_window(self) -> None:
        """Open CODE_REGISTRY and optional code_registry.db metadata."""
        try:
            CodeRegistryWindow(self)
        except Exception as exc:
            win = tk.Toplevel(self)
            win.title("Code Registry Error")
            win.configure(bg=BG)
            tk.Label(win, text=str(exc), bg=BG, fg=RED, font=FONT_SMALL, padx=20, pady=20).pack(fill="both", expand=True)

    def get_registry_metadata(self) -> Dict[str, object]:
        """Expose metadata to Dashboard/Main.py or debug panels."""
        return {
            "code_registry": get_code_registry(),
            "code_registry_db": read_registry_db_metadata(),
        }

    # --------------------------------------------------------
    # NAV / RANGE

    def _format_day_label(self, ts: pd.Timestamp, include_year: bool = False) -> str:
        if ts is None or pd.isna(ts):
            return "-"
        ts = pd.Timestamp(ts)
        label = f"{ts.strftime('%b')} {ts.day}"
        if include_year:
            label += f", {ts.year}"
        return label

    def _update_header_date_text(self) -> None:
        if self.window.start is None or self.window.end is None:
            self.date_btn.configure(text="All Time  📅")
            return
        s = self._format_day_label(self.window.start, include_year=False)
        e = self._format_day_label(self.window.end, include_year=True)
        self.date_btn.configure(text=f"{s} - {e}  📅")

    def reset_to_latest_month(self) -> None:
        self.equity_range = "ALL"
        self.equity_custom_start = None
        self.equity_custom_end = None
        self.window = DateWindow(start=None, end=None)
        self.refresh()

    def _set_visible_month(self, year: int, month: int) -> None:
        self.calendar_year = int(year)
        self.calendar_month = int(month)
        self._update_calendar()
        self._update_top_table()

    def prev_month(self) -> None:
        month = self.calendar_month - 1
        year = self.calendar_year
        if month < 1:
            month = 12
            year -= 1
        self._set_visible_month(year, month)

    def next_month(self) -> None:
        month = self.calendar_month + 1
        year = self.calendar_year
        if month > 12:
            month = 1
            year += 1
        self._set_visible_month(year, month)

    def goto_today(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        self._set_visible_month(now.year, now.month)

    def reset_equity_range_all(self) -> None:
        self.equity_range = "ALL"
        self.equity_custom_start = None
        self.equity_custom_end = None
        self.equity_start_var.set("")
        self.equity_end_var.set("")
        self._sync_equity_sliders_to_all()
        self.window = DateWindow(start=None, end=None)
        self.refresh()

    def _sync_equity_sliders_to_all(self) -> None:
        if not hasattr(self, "equity_start_scale") or not self.equity_date_points:
            return
        last_idx = max(len(self.equity_date_points) - 1, 0)
        self.equity_start_scale.configure(from_=0, to=last_idx)
        self.equity_end_scale.configure(from_=0, to=last_idx)
        self.equity_start_scale.set(0)
        self.equity_end_scale.set(last_idx)

    def on_equity_slider_change(self) -> None:
        if not hasattr(self, "equity_start_scale") or not self.equity_date_points:
            return
        start_idx = int(float(self.equity_start_scale.get()))
        end_idx = int(float(self.equity_end_scale.get()))
        if start_idx > end_idx:
            self.equity_end_scale.set(start_idx)
            end_idx = start_idx
        start_idx = max(0, min(start_idx, len(self.equity_date_points) - 1))
        end_idx = max(0, min(end_idx, len(self.equity_date_points) - 1))
        self.equity_custom_start = self.equity_date_points[start_idx].normalize()
        self.equity_custom_end = self.equity_date_points[end_idx].normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        self.equity_range = "CUSTOM"
        self.window = DateWindow(start=self.equity_custom_start, end=self.equity_custom_end)
        self.calendar_year = int(self.equity_custom_start.year)
        self.calendar_month = int(self.equity_custom_start.month)
        self._sync_equity_entries_from_custom()
        self._update_equity_range_button_styles()
        self._refresh_views_from_current_filter()

    def _sync_equity_entries_from_custom(self) -> None:
        if self.equity_custom_start is not None:
            self.equity_start_var.set(pd.Timestamp(self.equity_custom_start).strftime("%Y-%m-%d"))
        if self.equity_custom_end is not None:
            end_day = pd.Timestamp(self.equity_custom_end).normalize()
            self.equity_end_var.set(end_day.strftime("%Y-%m-%d"))

    def apply_equity_text_range(self, silent: bool = False) -> None:
        if not self.equity_date_points:
            return
        start_raw = self.equity_start_var.get().strip()
        end_raw = self.equity_end_var.get().strip()
        if not start_raw and not end_raw:
            return
        start = pd.to_datetime(start_raw, errors="coerce", utc=True) if start_raw else self.equity_date_points[0]
        end = pd.to_datetime(end_raw, errors="coerce", utc=True) if end_raw else self.equity_date_points[-1]
        if pd.isna(start):
            if not silent:
                self.equity_selected_var.set("Invalid Start: YYYY-MM-DD")
            return
        if pd.isna(end):
            if not silent:
                self.equity_selected_var.set("Invalid End: YYYY-MM-DD")
            return
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        if end < start:
            if not silent:
                self.equity_selected_var.set("Invalid: End < Start")
            return
        self.equity_range = "CUSTOM"
        self.equity_custom_start = start
        self.equity_custom_end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        self.window = DateWindow(start=self.equity_custom_start, end=self.equity_custom_end)
        self.calendar_year = int(self.equity_custom_start.year)
        self.calendar_month = int(self.equity_custom_start.month)
        available = pd.Series(self.equity_date_points)
        start_idx = int((available - start).abs().idxmin())
        end_idx = int((available - end).abs().idxmin())
        self.equity_start_scale.set(start_idx)
        self.equity_end_scale.set(end_idx)
        self._sync_equity_entries_from_custom()
        self._update_equity_range_button_styles()
        self._refresh_views_from_current_filter()

    def _update_equity_range_button_styles(self) -> None:
        active_all = self.equity_range == "ALL"
        self.equity_all_btn.configure(bg=GREEN if active_all else CARD_DARK, fg="#06130F" if active_all else MUTED)
        self.equity_apply_btn.configure(bg=GREEN if self.equity_range == "CUSTOM" else CARD_DARK, fg="#06130F" if self.equity_range == "CUSTOM" else MUTED)
        self._update_equity_available_label()

    # --------------------------------------------------------
    # KPI / EQUITY

    def _equity_series(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        d = self.df if df is None else df
        if d.empty or "net_sum" not in d.columns:
            return pd.Series(dtype=float)

        net = pd.to_numeric(d["net_sum"], errors="coerce").fillna(0.0)
        return START_EQUITY + net.cumsum()

    def _daily_returns(self, df: Optional[pd.DataFrame] = None) -> pd.Series:
        d = self.df if df is None else df
        if d.empty or "close_time_utc" not in d.columns:
            return pd.Series(dtype=float)
        x = d.copy()
        x["date"] = pd.to_datetime(x["close_time_utc"], utc=True, errors="coerce").dt.floor("D")
        daily_pnl = x.groupby("date")["net_sum"].sum().sort_index().astype(float)
        nav = START_EQUITY + daily_pnl.cumsum()
        return nav.pct_change().dropna()

    def _update_kpis(self) -> None:
        d = self.df
        if d.empty:
            for card in [self.kpi_net, self.kpi_trades, self.kpi_winrate, self.kpi_pf, self.kpi_dd, self.kpi_sharpe]:
                card.set("-", "")
            return
        net = pd.to_numeric(d["net_sum"], errors="coerce").fillna(0.0)
        total = float(net.sum())
        n = int(len(d))
        wins = int((net > 0).sum())
        losses = int((net < 0).sum())
        winrate = 100.0 * safe_div(wins, n)
        gp = float(net[net > 0].sum())
        gl = float(net[net < 0].sum())
        pf = safe_div(gp, abs(gl), 0.0) if gl < 0 else 0.0
        eq = self._equity_series(d)
        peak = eq.cummax()
        dd_pct = (eq / peak - 1.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
        max_dd_pct = float(dd_pct.min() * 100.0) if len(dd_pct) else 0.0
        dd_money = eq - peak
        max_dd_money = float(dd_money.min()) if len(dd_money) else 0.0
        daily_ret = self._daily_returns(d)
        sharpe = 0.0
        if len(daily_ret) > 1 and float(daily_ret.std()) != 0.0:
            sharpe = float((daily_ret.mean() / daily_ret.std()) * math.sqrt(252))
        period = self.window.start.strftime("%b %Y") if self.window.start is not None else "Current period"
        self.kpi_net.set(fmt_money(total), f"▲ {period}" if total >= 0 else f"▼ {period}", GREEN if total >= 0 else RED)
        self.kpi_trades.set(f"{n:,}", f"Wins {wins} / Losses {losses}", FG)
        self.kpi_winrate.set(fmt_pct(winrate), "Hit ratio", FG)
        self.kpi_pf.set(f"{pf:.2f}", "Gross profit / gross loss", FG)
        self.kpi_dd.set(fmt_pct(max_dd_pct, 2), f"▼ {fmt_money(max_dd_money)}", RED if max_dd_pct < 0 else GREEN)
        self.kpi_sharpe.set(f"{sharpe:.2f}", "Daily NAV return", FG)

    def _base_equity_df(self) -> pd.DataFrame:
        if self.raw_df.empty:
            return pd.DataFrame()
        d = self._symbol_filtered_raw_df(self.raw_df)
        d = self._build_strategy_label_column(d)
        selected = self.strategy_var.get()
        if selected != "All Strategies":
            d = d[d["strategy_label"].astype(str) == selected]
        if d.empty or "close_time_utc" not in d.columns:
            return d
        return d.dropna(subset=["close_time_utc"]).sort_values("close_time_utc").reset_index(drop=True)

    def _update_equity_available_label(self) -> None:
        d = self._base_equity_df()
        if d.empty or "close_time_utc" not in d.columns:
            self.equity_date_points = []
            self.equity_available_var.set("Available: -")
            self.equity_selected_var.set("Selected: -")
            return
        dates = pd.to_datetime(d["close_time_utc"], utc=True, errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values()
        self.equity_date_points = list(dates)
        if not self.equity_date_points:
            self.equity_available_var.set("Available: -")
            self.equity_selected_var.set("Selected: -")
            return
        start = self.equity_date_points[0]
        end = self.equity_date_points[-1]
        self.equity_available_var.set(f"Available: {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
        last_idx = max(len(self.equity_date_points) - 1, 0)
        self.equity_start_scale.configure(from_=0, to=last_idx)
        self.equity_end_scale.configure(from_=0, to=last_idx)
        if self.equity_range == "ALL":
            self.equity_start_scale.set(0)
            self.equity_end_scale.set(last_idx)
        if self.equity_range == "CUSTOM" and self.equity_custom_start is not None and self.equity_custom_end is not None:
            self.equity_selected_var.set(f"Selected: {self.equity_custom_start.strftime('%Y-%m-%d')} → {self.equity_custom_end.strftime('%Y-%m-%d')}")
        else:
            self.equity_selected_var.set("Selected: All")

    def _equity_source_df(self) -> pd.DataFrame:
        d = self._base_equity_df()
        if d.empty or "close_time_utc" not in d.columns:
            return d
        if self.equity_range == "ALL":
            return d
        if self.equity_range == "CUSTOM":
            if self.equity_custom_start is not None:
                d = d[d["close_time_utc"] >= self.equity_custom_start]
            if self.equity_custom_end is not None:
                d = d[d["close_time_utc"] <= self.equity_custom_end]
        return d.reset_index(drop=True)

    def _update_equity_stats_label(self, d: pd.DataFrame) -> None:
        if d.empty or "net_sum" not in d.columns:
            self.equity_stats_var.set("Equity Stats: -")
            if hasattr(self, "equity_stat_vars"):
                for var in self.equity_stat_vars.values():
                    var.set("-")
            return

        net = pd.to_numeric(d["net_sum"], errors="coerce").fillna(0.0)
        eq = START_EQUITY + net.cumsum()

        if eq.empty:
            self.equity_stats_var.set("Equity Stats: -")
            if hasattr(self, "equity_stat_vars"):
                for var in self.equity_stat_vars.values():
                    var.set("-")
            return

        max_eq = float(eq.max())
        min_eq = float(eq.min())
        end_eq = float(eq.iloc[-1])
        pnl = end_eq - START_EQUITY
        peak = eq.cummax()
        dd_money = eq - peak
        max_dd_money = float(dd_money.min())
        max_dd_pct = float(((eq / peak) - 1.0).replace([float("inf"), float("-inf")], 0.0).fillna(0.0).min() * 100.0)

        self.equity_stats_var.set(
            f"High {fmt_money(max_eq)} | Low {fmt_money(min_eq)} | End {fmt_money(end_eq)} | "
            f"PnL {fmt_money(pnl)} | Max DD {fmt_money(max_dd_money)} ({fmt_pct(max_dd_pct, 2)})"
        )

        if hasattr(self, "equity_stat_vars"):
            self.equity_stat_vars["high"].set(fmt_money(max_eq))
            self.equity_stat_vars["low"].set(fmt_money(min_eq))
            self.equity_stat_vars["end"].set(fmt_money(end_eq))
            self.equity_stat_vars["pnl"].set(fmt_money(pnl))
            self.equity_stat_vars["dd"].set(f"{fmt_money(max_dd_money)} / {fmt_pct(max_dd_pct, 2)}")

            self.equity_stat_value_labels["high"].configure(fg=GREEN)
            self.equity_stat_value_labels["low"].configure(fg=RED)
            self.equity_stat_value_labels["end"].configure(fg=GREEN if end_eq >= START_EQUITY else RED)
            self.equity_stat_value_labels["pnl"].configure(fg=GREEN if pnl >= 0 else RED)
            self.equity_stat_value_labels["dd"].configure(fg=RED if max_dd_money < 0 else GREEN)

    def _plot_equity(self) -> None:
        card = self.equity_card
        card.reset()
        self._update_equity_range_button_styles()
        ax = card.ax
        d = self._equity_source_df()
        self._update_equity_stats_label(d)
        if d.empty or "close_time_utc" not in d.columns:
            ax.text(0.5, 0.5, "No data", color=MUTED, ha="center", va="center", transform=ax.transAxes)
            card.canvas.draw_idle()
            return
        d = d.sort_values("close_time_utc").copy()
        d["equity"] = self._equity_series(d).values
        ax.plot(d["close_time_utc"], d["equity"], color=GREEN, linewidth=1.8)
        ax.fill_between(d["close_time_utc"], d["equity"], START_EQUITY, color=GREEN, alpha=0.20)
        ax.yaxis.set_major_formatter(lambda x, _pos: f"${x / 1000:.0f}K")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        style_ax(ax)
        card.fig.tight_layout(pad=1.0)
        card.canvas.draw_idle()

    # --------------------------------------------------------
    # PLOTS / SECTIONS

    def _plot_donut(self) -> None:
        self.donut_ax.clear()
        self.donut_fig.patch.set_facecolor(CARD)
        clear_children(self.symbol_legend)
        if self.df.empty or "symbol" not in self.df.columns:
            self.donut_ax.text(0.5, 0.5, "No data", color=MUTED, ha="center", va="center")
            self.donut_ax.axis("off")
            self.donut_canvas.draw_idle()
            return
        g = self.df.groupby("symbol")["net_sum"].sum().reset_index()
        g["abs"] = g["net_sum"].abs()
        g = g[g["abs"] > 0].sort_values("abs", ascending=False).head(8)
        if g.empty:
            self.donut_ax.text(0.5, 0.5, "No PnL", color=MUTED, ha="center", va="center")
            self.donut_ax.axis("off")
            self.donut_canvas.draw_idle()
            return
        labels = g["symbol"].tolist()
        values = g["abs"].tolist()
        colors = [self.symbol_colors.get(s, GREEN) for s in labels]
        total_abs = sum(values)
        net_total = float(self.df["net_sum"].sum())
        self.donut_ax.pie(values, colors=colors, startangle=90, counterclock=False, wedgeprops={"width": 0.34, "edgecolor": CARD, "linewidth": 1})
        self.donut_ax.text(0, 0.10, "Total PnL", ha="center", va="center", color=MUTED, fontsize=8)
        self.donut_ax.text(0, -0.08, fmt_money(net_total), ha="center", va="center", color=FG, fontsize=11, fontweight="bold")
        self.donut_ax.axis("equal")
        self.donut_ax.axis("off")
        for _, r in g.iterrows():
            row = tk.Frame(self.symbol_legend, bg=CARD)
            row.pack(fill="x", pady=2)
            color = self.symbol_colors.get(str(r["symbol"]), GREEN)
            tk.Label(row, text="●", bg=CARD, fg=color, font=FONT_SMALL).pack(side="left")
            tk.Label(row, text=str(r["symbol"]), bg=CARD, fg=FG, font=FONT_TINY, width=9, anchor="w").pack(side="left", padx=(4, 2))
            tk.Label(row, text=fmt_money(float(r["net_sum"])), bg=CARD, fg=GREEN if r["net_sum"] >= 0 else RED, font=FONT_TINY, width=9, anchor="e").pack(side="left")
            pct = 100.0 * safe_div(float(r["abs"]), total_abs)
            tk.Label(row, text=f"{pct:.1f}%", bg=CARD, fg=MUTED, font=FONT_TINY, width=6, anchor="e").pack(side="right")
        self.donut_canvas.draw_idle()

    def _plot_daily_pnl(self) -> None:
        card = self.daily_card
        card.reset()
        ax = card.ax
        if self.df.empty or "close_time_utc" not in self.df.columns:
            ax.text(0.5, 0.5, "No data", color=MUTED, ha="center", va="center", transform=ax.transAxes)
            card.canvas.draw_idle()
            return
        d = self.df.copy()
        d["day"] = pd.to_datetime(d["close_time_utc"], utc=True).dt.floor("D")
        daily = d.groupby("day")["net_sum"].sum().sort_index()
        colors = [GREEN if v >= 0 else RED for v in daily.values]
        ax.bar(daily.index, daily.values, color=colors, width=0.65)
        ax.axhline(0, color=GRID, linewidth=0.9)
        ax.yaxis.set_major_formatter(lambda x, _pos: f"${x:.0f}")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        style_ax(ax)
        card.fig.tight_layout(pad=1.0)
        card.canvas.draw_idle()

    # --------------------------------------------------------
    # DAILY DETAILS

    def open_day_details(self, date_obj) -> None:
        d = self._calendar_source_df()
        if d.empty or "close_time_utc" not in d.columns:
            return

        day = pd.Timestamp(date_obj).date()
        x = d.copy()
        x["trade_day"] = pd.to_datetime(x["close_time_utc"], utc=True, errors="coerce").dt.date
        x = x[x["trade_day"] == day].copy()
        if x.empty:
            return

        x = self._build_strategy_label_column(x)
        x = x.sort_values("close_time_utc").reset_index(drop=True)
        net = pd.to_numeric(x["net_sum"], errors="coerce").fillna(0.0)

        total = float(net.sum())
        trades = int(len(x))
        wins = int((net > 0).sum())
        losses = int((net < 0).sum())
        winrate = 100.0 * safe_div(wins, trades)
        avg_trade = float(net.mean()) if trades else 0.0
        best_trade = float(net.max()) if trades else 0.0
        worst_trade = float(net.min()) if trades else 0.0

        win = tk.Toplevel(self)
        win.title(f"Daily Performance · {day}")
        win.geometry("1420x880")
        win.minsize(1150, 720)
        win.configure(bg=BG)

        root = tk.Frame(win, bg=BG)
        root.pack(fill="both", expand=True, padx=16, pady=14)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        tk.Label(root, text=f"Daily Performance · {day}", bg=BG, fg=FG, font=("Segoe UI", 17, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))

        kpi_row = tk.Frame(root, bg=BG)
        kpi_row.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for i in range(7):
            kpi_row.columnconfigure(i, weight=1, uniform="daily_kpi")

        metrics = [
            ("Net PnL", fmt_money(total), GREEN if total >= 0 else RED),
            ("Trades", f"{trades}", FG),
            ("Wins / Losses", f"{wins} / {losses}", FG),
            ("Win Rate", fmt_pct(winrate), FG),
            ("Avg Trade", fmt_money(avg_trade), GREEN if avg_trade >= 0 else RED),
            ("Best Trade", fmt_money(best_trade), GREEN),
            ("Worst Trade", fmt_money(worst_trade), RED),
        ]

        for i, (title, value, color) in enumerate(metrics):
            card = Card(kpi_row, bg=CARD, padx=12, pady=8)
            card.grid(row=0, column=i, sticky="nsew", padx=4)
            tk.Label(card.inner, text=title, bg=CARD, fg=MUTED, font=FONT_TINY).pack(anchor="w")
            tk.Label(card.inner, text=value, bg=CARD, fg=color, font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4, 0))

        chart_row = tk.Frame(root, bg=BG)
        chart_row.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        chart_row.columnconfigure(0, weight=2)
        chart_row.columnconfigure(1, weight=2)
        chart_row.columnconfigure(2, weight=2)

        self._daily_equity_chart(chart_row, x, net, total)
        self._daily_trade_sequence_chart(chart_row, net, trades)
        self._daily_symbol_chart(chart_row, x)
        self._daily_trades_table(root, x)

    def _daily_equity_chart(self, parent, x: pd.DataFrame, net: pd.Series, total: float) -> None:
        equity_card = Card(parent, bg=CARD, padx=10, pady=8)
        equity_card.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        tk.Label(equity_card.inner, text="Intraday Equity", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        fig = Figure(figsize=(4.8, 2.45), dpi=100)
        fig.patch.set_facecolor(CARD)
        ax = fig.add_subplot(111)
        x = x.copy()
        x["day_equity"] = net.cumsum()
        color = GREEN if total >= 0 else RED
        ax.plot(x["close_time_utc"], x["day_equity"], color=color, linewidth=1.8)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.fill_between(x["close_time_utc"], x["day_equity"], 0, color=color, alpha=0.18)
        ax.yaxis.set_major_formatter(lambda v, _p: f"${v:.0f}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        style_ax(ax)
        fig.tight_layout(pad=1.0)
        canvas = FigureCanvasTkAgg(fig, master=equity_card.inner)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
        canvas.draw_idle()

    def _daily_trade_sequence_chart(self, parent, net: pd.Series, trades: int) -> None:
        seq_card = Card(parent, bg=CARD, padx=10, pady=8)
        seq_card.grid(row=0, column=1, sticky="nsew", padx=5)
        tk.Label(seq_card.inner, text="Trade PnL Sequence", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        fig = Figure(figsize=(4.8, 2.45), dpi=100)
        fig.patch.set_facecolor(CARD)
        ax = fig.add_subplot(111)
        trade_idx = list(range(1, trades + 1))
        bar_colors = [GREEN if v >= 0 else RED for v in net.values]
        ax.bar(trade_idx, net.values, color=bar_colors, width=0.65)
        ax.axhline(0, color=GRID, linewidth=0.8)
        ax.set_xlabel("Trade #")
        ax.yaxis.set_major_formatter(lambda v, _p: f"${v:.0f}")
        style_ax(ax)
        fig.tight_layout(pad=1.0)
        canvas = FigureCanvasTkAgg(fig, master=seq_card.inner)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
        canvas.draw_idle()

    def _daily_symbol_chart(self, parent, x: pd.DataFrame) -> None:
        sym_card = Card(parent, bg=CARD, padx=10, pady=8)
        sym_card.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        tk.Label(sym_card.inner, text="PnL by Symbol", bg=CARD, fg=FG, font=FONT_H2).pack(anchor="w")
        fig = Figure(figsize=(4.8, 2.45), dpi=100)
        fig.patch.set_facecolor(CARD)
        ax = fig.add_subplot(111)
        sym = x.groupby("symbol")["net_sum"].sum().sort_values().tail(10)
        sym_colors = [GREEN if v >= 0 else RED for v in sym.values]
        ax.barh(sym.index.astype(str), sym.values, color=sym_colors)
        ax.axvline(0, color=GRID, linewidth=0.8)
        ax.xaxis.set_major_formatter(lambda v, _p: f"${v:.0f}")
        style_ax(ax)
        fig.tight_layout(pad=1.0)
        canvas = FigureCanvasTkAgg(fig, master=sym_card.inner)
        canvas.get_tk_widget().pack(fill="both", expand=True, pady=(6, 0))
        canvas.draw_idle()

    def _daily_trades_table(self, root, x: pd.DataFrame) -> None:
        table_card = Card(root, bg=CARD, padx=10, pady=8)
        table_card.grid(row=3, column=0, sticky="nsew")
        table_card.inner.rowconfigure(1, weight=1)
        table_card.inner.columnconfigure(0, weight=1)
        tk.Label(table_card.inner, text="Trades", bg=CARD, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w")
        table_wrap = tk.Frame(table_card.inner, bg=CARD)
        table_wrap.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        table_wrap.rowconfigure(0, weight=1)
        table_wrap.columnconfigure(0, weight=1)
        cols = ("close_time", "symbol", "strategy", "direction", "volume", "entry", "exit", "profit", "swap", "commission", "net", "position_id", "ticket")
        tree = ttk.Treeview(table_wrap, columns=cols, show="headings", style="Dark.Treeview")
        columns = [
            ("close_time", "Close Time UTC", 150, "w"),
            ("symbol", "Symbol", 90, "w"),
            ("strategy", "Strategy", 240, "w"),
            ("direction", "Side", 70, "center"),
            ("volume", "Volume", 80, "e"),
            ("entry", "Entry", 90, "e"),
            ("exit", "Exit", 90, "e"),
            ("profit", "Profit", 90, "e"),
            ("swap", "Swap", 80, "e"),
            ("commission", "Commission", 95, "e"),
            ("net", "Net", 100, "e"),
            ("position_id", "Position ID", 110, "e"),
            ("ticket", "Close Ticket", 110, "e"),
        ]
        for col, text, width, anchor in columns:
            tree.heading(col, text=text)
            tree.column(col, width=width, anchor=anchor)
        yscroll = tk.Scrollbar(table_wrap, orient="vertical", command=tree.yview)
        xscroll = tk.Scrollbar(table_wrap, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        tree.tag_configure("win", foreground=GREEN)
        tree.tag_configure("loss", foreground=RED)
        tree.tag_configure("flat", foreground=FG)
        for _, r in x.iterrows():
            close_ts = pd.Timestamp(r["close_time_utc"])
            close_time = close_ts.strftime("%Y-%m-%d %H:%M:%S")
            volume = r.get("volume_out", r.get("volume_in", 0.0))
            entry = float(r.get("entry_price", 0.0) or 0.0)
            exit_ = float(r.get("exit_price", 0.0) or 0.0)
            profit = float(r.get("profit_sum", 0.0) or 0.0)
            swap = float(r.get("swap_sum", 0.0) or 0.0)
            commission = float(r.get("commission_sum", 0.0) or 0.0)
            net_val = float(r.get("net_sum", 0.0) or 0.0)
            tag = "win" if net_val > 0 else "loss" if net_val < 0 else "flat"
            tree.insert("", "end", values=(close_time, str(r.get("symbol", "")), str(r.get("strategy_label", "")), str(r.get("direction", "")), f"{float(volume):.2f}", f"{entry:.5f}", f"{exit_:.5f}", fmt_money(profit), fmt_money(swap), fmt_money(commission), fmt_money(net_val), str(r.get("position_id", "")), str(r.get("close_ticket", ""))), tags=(tag,))

    # --------------------------------------------------------
    # CALENDAR

    def _update_calendar(self) -> None:
        clear_children(self.calendar_grid)
        self._update_daily_calendar()

    def _update_period_bar(self, title: str, pnl: float, trades: int, win_rate: float) -> None:
        self.period_title_var.set(title)
        self.period_pnl_var.set(fmt_money(pnl))
        self.period_trades_var.set(f"Trades: {int(trades)}")
        self.period_winrate_var.set(f"Win Rate: {fmt_pct(win_rate)}")
        self.period_pnl_label.configure(fg=GREEN if pnl >= 0 else RED)

    def _calendar_source_df(self) -> pd.DataFrame:
        if self.raw_df.empty or "close_time_utc" not in self.raw_df.columns:
            return pd.DataFrame()
        d = self._symbol_filtered_raw_df(self.raw_df)
        d = self._build_strategy_label_column(d)
        selected = self.strategy_var.get()
        if selected != "All Strategies":
            d = d[d["strategy_label"].astype(str) == selected]
        d = d.dropna(subset=["close_time_utc"]).copy()
        d = d[(d["close_time_utc"].dt.year == self.calendar_year) & (d["close_time_utc"].dt.month == self.calendar_month)].copy()
        return d.reset_index(drop=True)

    def _daily_calendar_map(self) -> Dict[object, Tuple[float, int]]:
        daily_map: Dict[object, Tuple[float, int]] = {}
        d = self._calendar_source_df()
        if not d.empty and "close_time_utc" in d.columns:
            d["date"] = pd.to_datetime(d["close_time_utc"], utc=True, errors="coerce").dt.date
            g = d.groupby("date").agg(pnl=("net_sum", "sum"), trades=("net_sum", "size")).reset_index()
            daily_map = {r["date"]: (float(r["pnl"]), int(r["trades"])) for _, r in g.iterrows()}
        return daily_map

    def _update_daily_calendar(self) -> None:
        year = self.calendar_year
        month = self.calendar_month
        self.cal_title.configure(text=f"{calendar.month_name[month]} {year}")
        cal_df = self._calendar_source_df()
        if not cal_df.empty:
            month_pnl = float(cal_df["net_sum"].sum())
            month_trades = int(len(cal_df))
            month_wins = int((cal_df["net_sum"] > 0).sum())
            month_wr = 100.0 * safe_div(month_wins, month_trades)
        else:
            month_pnl = 0.0
            month_trades = 0
            month_wr = 0.0
        self._update_period_bar(f"Calendar Month · {calendar.month_name[month]} {year}", month_pnl, month_trades, month_wr)

        headers = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN", "WEEK"]
        for c, label in enumerate(headers):
            self.calendar_grid.columnconfigure(c, weight=1, uniform="cal")
            tk.Label(self.calendar_grid, text=label, bg=CARD, fg=MUTED, font=FONT_TINY).grid(row=0, column=c, sticky="nsew", pady=(0, 4))

        daily_map = self._daily_calendar_map()
        weeks = calendar.Calendar(firstweekday=0).monthdatescalendar(year, month)
        for r, week in enumerate(weeks, start=1):
            self.calendar_grid.rowconfigure(r, weight=1, uniform="calrow")
            week_pnl = 0.0
            week_trades = 0
            for c, date_obj in enumerate(week):
                in_month = date_obj.month == month
                pnl, trades = daily_map.get(date_obj, (0.0, 0))
                if in_month:
                    week_pnl += pnl
                    week_trades += trades
                cell_bg = CARD_DARK
                if trades > 0:
                    cell_bg = "#12332E" if pnl >= 0 else "#351C24"
                cell = tk.Frame(self.calendar_grid, bg=cell_bg, highlightthickness=1, highlightbackground="#1D2A37")
                cell.grid(row=r, column=c, sticky="nsew", padx=1, pady=1)
                clickable = bool(in_month and trades > 0)
                if clickable:
                    cell.configure(cursor="hand2")
                    cell.bind("<Button-1>", lambda _e, d=date_obj: self.open_day_details(d))
                day_label = tk.Label(cell, text=str(date_obj.day), bg=cell_bg, fg=FG if in_month else SUBTLE, font=FONT_TINY, cursor="hand2" if clickable else "")
                day_label.pack(anchor="nw", padx=6, pady=(4, 0))
                if clickable:
                    day_label.bind("<Button-1>", lambda _e, d=date_obj: self.open_day_details(d))
                if trades > 0:
                    pnl_label = tk.Label(cell, text=fmt_money(pnl), bg=cell_bg, fg=GREEN if pnl >= 0 else RED, font=("Segoe UI", 8, "bold"), cursor="hand2" if clickable else "")
                    pnl_label.pack(anchor="w", padx=6, pady=(7, 0))
                    trades_label = tk.Label(cell, text=f"Trades: {trades}", bg=cell_bg, fg=MUTED, font=FONT_TINY, cursor="hand2" if clickable else "")
                    trades_label.pack(anchor="w", padx=6)
                    if clickable:
                        pnl_label.bind("<Button-1>", lambda _e, d=date_obj: self.open_day_details(d))
                        trades_label.bind("<Button-1>", lambda _e, d=date_obj: self.open_day_details(d))
            week_bg = "#12332E" if week_pnl >= 0 else "#351C24"
            if week_trades == 0:
                week_bg = CARD_DARK
            week_cell = tk.Frame(self.calendar_grid, bg=week_bg, highlightthickness=1, highlightbackground="#1D2A37")
            week_cell.grid(row=r, column=7, sticky="nsew", padx=(3, 1), pady=1)
            tk.Label(week_cell, text=f"W{pd.Timestamp(week[0]).isocalendar().week}", bg=week_bg, fg=MUTED, font=FONT_TINY).pack(anchor="nw", padx=6, pady=(4, 0))
            if week_trades > 0:
                tk.Label(week_cell, text=fmt_money(week_pnl), bg=week_bg, fg=GREEN if week_pnl >= 0 else RED, font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=6, pady=(7, 0))
                tk.Label(week_cell, text=f"Trades: {week_trades}", bg=week_bg, fg=MUTED, font=FONT_TINY).pack(anchor="w", padx=6)

    # --------------------------------------------------------
    # LOWER SECTIONS

    def _build_strategy_label_column(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        has_canonical = "canonical_bucket" in d.columns and d["canonical_bucket"].astype(str).str.strip().str.len().gt(0).any()
        if has_canonical:
            d["strategy_label"] = d["canonical_bucket"].astype(str).str.strip()
            return d
        symbol = d["symbol"].astype(str).str.strip() if "symbol" in d.columns else pd.Series("UNKNOWN", index=d.index)
        strategy_id = d["strategy_id"].astype(str).str.strip() if "strategy_id" in d.columns else pd.Series("", index=d.index)
        direction = d["direction"].astype(str).str.strip() if "direction" in d.columns else pd.Series("", index=d.index)
        strategy_id = strategy_id.replace({"nan": "", "None": "", "<NA>": ""})
        symbol = symbol.replace({"nan": "UNKNOWN", "None": "UNKNOWN", "<NA>": "UNKNOWN", "": "UNKNOWN"})
        direction = direction.replace({"nan": "", "None": "", "<NA>": ""})
        d["strategy_label"] = symbol
        mask = strategy_id.str.len() > 0
        d.loc[mask, "strategy_label"] = symbol[mask] + "_" + strategy_id[mask]
        side_mask = mask & direction.str.len().gt(0)
        d.loc[side_mask, "strategy_label"] = d.loc[side_mask, "strategy_label"].astype(str) + "_" + direction[side_mask]
        if "source_bucket" in d.columns:
            missing = d["strategy_label"].astype(str).str.strip().isin(["", "UNKNOWN"])
            d.loc[missing, "strategy_label"] = d.loc[missing, "source_bucket"].astype(str).str.strip()
        return d

    def _update_strategy_rank(self) -> None:
        clear_children(self.strategy_rank_body)
        rank_source = apply_window(self.raw_df, self.window)
        rank_source = self._symbol_filtered_raw_df(rank_source)
        if rank_source.empty:
            tk.Label(self.strategy_rank_body, text="No data", bg=CARD, fg=MUTED, font=FONT_SMALL).pack(anchor="w")
            return
        rank_df = self._build_strategy_label_column(rank_source)
        g = rank_df.groupby("strategy_label").agg(net=("net_sum", "sum"), trades=("net_sum", "size"), wins=("net_sum", lambda x: int((x > 0).sum()))).reset_index()
        g["winrate"] = 100.0 * g["wins"] / g["trades"].replace(0, pd.NA)
        sort_mode = self.strategy_sort_mode.get()
        if sort_mode == "Trades":
            g = g.sort_values(["trades", "net"], ascending=[False, False])
        elif sort_mode == "Win Rate":
            g = g.sort_values(["winrate", "trades", "net"], ascending=[False, False, False])
        else:
            g = g.sort_values(["net", "trades"], ascending=[False, False])
        max_abs = max(float(g["net"].abs().max()), 1.0)
        selected = self.active_strategy_highlight
        for _, r in g.iterrows():
            full_name = str(r["strategy_label"])
            is_selected = full_name == selected
            has_active = selected is not None
            row_bg = CARD_SOFT if is_selected else CARD
            row = tk.Frame(self.strategy_rank_body, bg=row_bg, cursor="hand2")
            row.pack(fill="x", pady=3)
            text_fg = FG if (not has_active or is_selected) else SUBTLE
            name_label = tk.Label(row, text=full_name[:20], bg=row_bg, fg=text_fg, font=FONT_TINY, width=18, anchor="w", cursor="hand2")
            name_label.pack(side="left")
            bar_wrap = tk.Frame(row, bg="#1B2A38", width=95, height=6, cursor="hand2")
            bar_wrap.pack(side="left", padx=(4, 6))
            bar_wrap.pack_propagate(False)
            width = int(95 * abs(float(r["net"])) / max_abs)
            bar = tk.Frame(bar_wrap, bg=GREEN if r["net"] >= 0 else RED, width=max(width, 2), height=6, cursor="hand2")
            bar.pack(side="left")
            net_label = tk.Label(row, text=fmt_money(float(r["net"])), bg=row_bg, fg=(GREEN if r["net"] >= 0 else RED), font=FONT_TINY, width=10, anchor="e", cursor="hand2")
            net_label.pack(side="right")
            trades_label = tk.Label(row, text=f"{int(r['trades'])}", bg=row_bg, fg=MUTED, font=FONT_TINY, width=7, anchor="e", cursor="hand2")
            trades_label.pack(side="right")
            wr_label = tk.Label(row, text=fmt_pct(float(r["winrate"])), bg=row_bg, fg=MUTED, font=FONT_TINY, width=8, anchor="e", cursor="hand2")
            wr_label.pack(side="right")
            for widget in [row, name_label, bar_wrap, bar, net_label, trades_label, wr_label]:
                widget.bind("<Button-1>", lambda _e, s=full_name: self.select_strategy(s))

    def _update_heatmap(self) -> None:
        clear_children(self.heatmap_body)
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        tk.Label(self.heatmap_body, text="", bg=CARD, width=6).grid(row=0, column=0)
        for c, m in enumerate(months, start=1):
            tk.Label(self.heatmap_body, text=m, bg=CARD, fg=MUTED, font=FONT_TINY, width=7).grid(row=0, column=c, padx=1, pady=1)
        d = self._symbol_filtered_raw_df(self.raw_df)
        if d.empty or "close_time_utc" not in d.columns:
            tk.Label(self.heatmap_body, text="No data", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=1, column=0, columnspan=13, pady=20)
            return
        d = d.copy()
        d["year"] = pd.to_datetime(d["close_time_utc"], utc=True).dt.year
        d["month"] = pd.to_datetime(d["close_time_utc"], utc=True).dt.month
        g = d.groupby(["year", "month"])["net_sum"].sum().reset_index()
        g["ret"] = 100.0 * g["net_sum"] / START_EQUITY
        years = sorted(g["year"].unique(), reverse=True)[:3]
        val_map = {(int(r["year"]), int(r["month"])): float(r["ret"]) for _, r in g.iterrows()}
        for r, y in enumerate(years, start=1):
            tk.Label(self.heatmap_body, text=str(y), bg=CARD, fg=FG, font=FONT_TINY, width=6).grid(row=r, column=0, padx=1, pady=2)
            for m in range(1, 13):
                val = val_map.get((int(y), m))
                if val is None:
                    bg = CARD_DARK
                    text = "-"
                    fg = MUTED
                else:
                    bg = "#12382F" if val >= 0 else "#3A1D26"
                    text = f"{val:.1f}%"
                    fg = GREEN if val >= 0 else RED
                tk.Label(self.heatmap_body, text=text, bg=bg, fg=fg, font=FONT_TINY, width=7, height=2).grid(row=r, column=m, padx=1, pady=2, sticky="nsew")

    def _update_top_table(self) -> None:
        for item in self.top_table.get_children():
            self.top_table.delete(item)
        self.top_table_period_var.set(f"{calendar.month_name[self.calendar_month]} {self.calendar_year}")
        d = self._calendar_source_df()
        if d.empty or "symbol" not in d.columns:
            return
        g = d.groupby("symbol").agg(net=("net_sum", "sum"), trades=("net_sum", "size"), wins=("net_sum", lambda x: int((x > 0).sum()))).reset_index()
        g["winrate"] = 100.0 * g["wins"] / g["trades"].replace(0, pd.NA)
        g = g.sort_values("net", ascending=False)
        for _, r in g.iterrows():
            self.top_table.insert("", "end", values=(str(r["symbol"]), fmt_money(float(r["net"])), int(r["trades"]), fmt_pct(float(r["winrate"]))))




# ============================================================
# LIVE PORTFOLIO BUILDER
# ============================================================

class LivePortfolioBuilder(tk.Toplevel):
    """
    Full live portfolio builder embedded into Account Monitor.

    It uses the already loaded monitor trades for the selected account, but it
    does not use the active monitor date/symbol/strategy filters. Therefore the
    candidate list contains all strategies in that account, including positive,
    negative and flat strategies.

    Main features:
    - Candidate table with all strategies
    - Search, symbol filter, direction filter
    - Ranking by Highest PnL, Lowest DD, PF, Win Rate, Trades, Avg Trade, Sharpe, Score
    - Min/Max filters
    - Portfolio builder with weights
    - Portfolio facts similar to Scaling dashboard
    - Daily / Weekly / Monthly analytics
    - Save / Load as JSON + CSV in 2_Baseline/Trades/Live/Portfolios
    """

    def __init__(self, parent: tk.Widget, trades: pd.DataFrame, account_name: str, account_root: Path):
        super().__init__(parent)
        self.title(f"Live Portfolio Builder · {account_name}")
        self.geometry("1760x980")
        self.minsize(1350, 820)
        self.configure(bg=BG)

        self.account_name = str(account_name)
        self.account_root = Path(account_root)
        self.trades = self._prepare_trades(trades)
        self.strategy_stats = self._build_strategy_stats(self.trades)
        self.filtered_stats = self.strategy_stats.copy()
        self.portfolio = pd.DataFrame(columns=["strategy", "symbol", "direction", "weight"])
        self.portfolio_trades = pd.DataFrame()

        self.portfolio_root = DATA_DIR / "feature_engineered" / "trades" / "live" / "Portfolios"
        self.portfolio_root.mkdir(parents=True, exist_ok=True)

        self.search_var = tk.StringVar(value="")
        self.symbol_var = tk.StringVar(value="ALL")
        self.direction_var = tk.StringVar(value="ALL")
        self.min_trades_var = tk.StringVar(value="0")
        self.min_pf_var = tk.StringVar(value="")
        self.min_pnl_var = tk.StringVar(value="")
        self.max_dd_var = tk.StringVar(value="")
        self.metric_min_var = tk.StringVar(value="")
        self.metric_max_var = tk.StringVar(value="")
        self.sort_var = tk.StringVar(value="Score")
        self.order_var = tk.StringVar(value="High First")
        self.weight_var = tk.StringVar(value="1.00")
        self.portfolio_name_var = tk.StringVar(value=f"{self.account_name}_portfolio")
        self.saved_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        self.show_components_var = tk.BooleanVar(value=True)

        self._setup_style()
        self._build_ui()
        self._populate_filter_values()
        self.apply_filters()
        self._refresh_saved_combo()
        self._update_portfolio_view()

    # --------------------------------------------------------
    # SETUP

    def _setup_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "Live.Treeview",
            background=CARD_DARK,
            foreground=FG,
            fieldbackground=CARD_DARK,
            rowheight=24,
            font=FONT_SMALL,
            borderwidth=0,
        )
        style.configure(
            "Live.Treeview.Heading",
            background=CARD_SOFT,
            foreground=FG,
            font=FONT_TINY,
            borderwidth=0,
        )
        style.map("Live.Treeview", background=[("selected", CARD_SOFT)], foreground=[("selected", FG)])

    def _build_ui(self) -> None:
        root = tk.Frame(self, bg=BG)
        root.pack(fill="both", expand=True, padx=14, pady=12)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)

        self._build_header(root)
        self._build_filter_bar(root)
        self._build_main_area(root)
        self._build_footer(root)

    def _build_header(self, root: tk.Frame) -> None:
        header = tk.Frame(root, bg=BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.columnconfigure(0, weight=1)

        left = tk.Frame(header, bg=BG)
        left.grid(row=0, column=0, sticky="w")
        tk.Label(left, text="Live Portfolio Builder", bg=BG, fg=FG, font=FONT_TITLE).pack(anchor="w")
        tk.Label(
            left,
            text=f"Account: {self.account_name}  |  Source: {self.account_root}  |  Strategies: {len(self.strategy_stats)}",
            bg=BG,
            fg=MUTED,
            font=FONT_SMALL,
        ).pack(anchor="w", pady=(2, 0))

        actions = tk.Frame(header, bg=BG)
        actions.grid(row=0, column=1, sticky="e")
        tk.Button(actions, text="Add Top 10", command=lambda: self.add_top_n(10), bg=GREEN, fg="#06130F", relief="flat", font=FONT_NAV, padx=16, pady=9).pack(side="left", padx=(0, 8))
        tk.Button(actions, text="Add Top Filtered", command=self.add_top_filtered, bg=BLUE, fg=FG, relief="flat", font=FONT_NAV, padx=16, pady=9).pack(side="left", padx=(0, 8))
        tk.Button(actions, text="Clear", command=self.clear_portfolio, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=16, pady=9).pack(side="left", padx=(0, 8))
        tk.Button(actions, text="Close", command=self.destroy, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=16, pady=9).pack(side="left")

    def _build_filter_bar(self, root: tk.Frame) -> None:
        bar = tk.Frame(root, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        bar.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        bar.columnconfigure(1, weight=1)

        tk.Label(bar, text="Search", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=0, padx=(10, 5), pady=8)
        search = tk.Entry(bar, textvariable=self.search_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL)
        search.grid(row=0, column=1, sticky="ew", ipady=5, padx=(0, 8))
        search.bind("<KeyRelease>", lambda _e: self.apply_filters())

        tk.Label(bar, text="Symbol", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=2, padx=(0, 5))
        self.symbol_combo = ttk.Combobox(bar, textvariable=self.symbol_var, state="readonly", width=12)
        self.symbol_combo.grid(row=0, column=3, padx=(0, 8))
        self.symbol_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        tk.Label(bar, text="Direction", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=4, padx=(0, 5))
        self.direction_combo = ttk.Combobox(bar, textvariable=self.direction_var, state="readonly", width=9)
        self.direction_combo.grid(row=0, column=5, padx=(0, 8))
        self.direction_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        tk.Label(bar, text="Min Trades", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=6, padx=(0, 5))
        tk.Entry(bar, textvariable=self.min_trades_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, width=8).grid(row=0, column=7, padx=(0, 8), ipady=5)

        tk.Label(bar, text="Min PF", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=8, padx=(0, 5))
        tk.Entry(bar, textvariable=self.min_pf_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, width=8).grid(row=0, column=9, padx=(0, 8), ipady=5)

        tk.Label(bar, text="Min PnL", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=10, padx=(0, 5))
        tk.Entry(bar, textvariable=self.min_pnl_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, width=9).grid(row=0, column=11, padx=(0, 8), ipady=5)

        tk.Label(bar, text="Max DD $", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=12, padx=(0, 5))
        tk.Entry(bar, textvariable=self.max_dd_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, width=9).grid(row=0, column=13, padx=(0, 8), ipady=5)

        tk.Label(bar, text="Weight", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=14, padx=(0, 5))
        tk.Entry(bar, textvariable=self.weight_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL, width=7).grid(row=0, column=15, padx=(0, 8), ipady=5)

        tk.Label(bar, text="Rank", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=16, padx=(0, 5))
        self.sort_combo = ttk.Combobox(bar, textvariable=self.sort_var, state="readonly", width=16)
        self.sort_combo["values"] = [
            "Score", "Highest PnL", "Lowest DD", "Profit Factor", "Win Rate", "Most Trades",
            "Avg Trade", "Best Trade", "Worst Trade", "Sharpe", "Worst Day", "Best Month",
        ]
        self.sort_combo.grid(row=0, column=17, padx=(0, 6))
        self.sort_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        self.order_combo = ttk.Combobox(bar, textvariable=self.order_var, state="readonly", width=10)
        self.order_combo["values"] = ["High First", "Low First"]
        self.order_combo.grid(row=0, column=18, padx=(0, 8))
        self.order_combo.bind("<<ComboboxSelected>>", lambda _e: self.apply_filters())

        tk.Button(bar, text="Apply", command=self.apply_filters, bg=BLUE, fg=FG, relief="flat", font=FONT_NAV, padx=12, pady=6).grid(row=0, column=19, padx=(0, 4))
        tk.Button(bar, text="Reset", command=self.reset_filters, bg=CARD_SOFT, fg=FG, relief="flat", font=FONT_NAV, padx=12, pady=6).grid(row=0, column=20, padx=(0, 10))

        adv = tk.Frame(bar, bg=CARD)
        adv.grid(row=1, column=0, columnspan=21, sticky="ew", padx=10, pady=(0, 8))
        tk.Label(adv, text="Metric Min", bg=CARD, fg=MUTED, font=FONT_TINY).pack(side="left", padx=(0, 4))
        tk.Entry(adv, textvariable=self.metric_min_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_TINY, width=12).pack(side="left", padx=(0, 10), ipady=3)
        tk.Label(adv, text="Metric Max", bg=CARD, fg=MUTED, font=FONT_TINY).pack(side="left", padx=(0, 4))
        tk.Entry(adv, textvariable=self.metric_max_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_TINY, width=12).pack(side="left", padx=(0, 10), ipady=3)
        tk.Label(adv, text="Ranking und Filter wirken nur auf Candidate-Tabelle. Keine Strategie wird standardmäßig versteckt.", bg=CARD, fg=SUBTLE, font=FONT_TINY).pack(side="left", padx=10)

    def _build_main_area(self, root: tk.Frame) -> None:
        body = tk.Frame(root, bg=BG)
        body.grid(row=3, column=0, sticky="nsew")
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)
        self._build_candidate_section(left)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(7, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        self._build_portfolio_section(right)
        self._build_analytics_section(right)

    def _build_candidate_section(self, parent: tk.Frame) -> None:
        panel = tk.Frame(parent, bg=BG)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        tk.Label(panel, text="Candidate Strategies", bg=BG, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w", pady=(0, 6))

        table_frame = tk.Frame(panel, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        table_frame.grid(row=1, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = ("strategy", "symbol", "dir", "trades", "net", "pf", "dd", "wr", "avg", "score")
        self.strategy_table = ttk.Treeview(table_frame, columns=cols, show="headings", style="Live.Treeview", height=28)
        specs = [
            ("strategy", "Strategy", 270, "w"),
            ("symbol", "Symbol", 90, "center"),
            ("dir", "Dir", 55, "center"),
            ("trades", "Trades", 65, "e"),
            ("net", "Net PnL", 100, "e"),
            ("pf", "PF", 70, "e"),
            ("dd", "Max DD", 100, "e"),
            ("wr", "Win %", 75, "e"),
            ("avg", "Avg", 85, "e"),
            ("score", "Score", 80, "e"),
        ]
        for col, text, width, anchor in specs:
            self.strategy_table.heading(col, text=text)
            self.strategy_table.column(col, width=width, anchor=anchor, stretch=True)
        self.strategy_table.grid(row=0, column=0, sticky="nsew")
        self.strategy_table.bind("<Double-1>", lambda _e: self.add_selected_strategy())
        self.strategy_table.tag_configure("positive", foreground=GREEN)
        self.strategy_table.tag_configure("negative", foreground=RED)
        self.strategy_table.tag_configure("flat", foreground=FG)

        scroll_y = tk.Scrollbar(table_frame, orient="vertical", command=self.strategy_table.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
        self.strategy_table.configure(yscrollcommand=scroll_y.set)

        bottom = tk.Frame(panel, bg=BG)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        tk.Button(bottom, text="Add Selected", command=self.add_selected_strategy, bg=BLUE, fg=FG, relief="flat", font=FONT_NAV, padx=14, pady=8).pack(side="left", padx=(0, 8))
        tk.Button(bottom, text="Add Top Filtered", command=self.add_top_filtered, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=14, pady=8).pack(side="left", padx=(0, 8))
        tk.Button(bottom, text="Open Strategy Facts", command=self.open_selected_strategy_facts, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=14, pady=8).pack(side="left", padx=(0, 8))

    def _build_portfolio_section(self, parent: tk.Frame) -> None:
        top = tk.Frame(parent, bg=BG)
        top.grid(row=0, column=0, sticky="nsew")
        top.rowconfigure(2, weight=1)
        top.columnconfigure(0, weight=1)

        controls = tk.Frame(top, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(1, weight=1)

        tk.Label(controls, text="Portfolio Name", bg=CARD, fg=MUTED, font=FONT_SMALL).grid(row=0, column=0, sticky="w", padx=(10, 6), pady=8)
        tk.Entry(controls, textvariable=self.portfolio_name_var, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=FONT_SMALL).grid(row=0, column=1, sticky="ew", ipady=5, padx=(0, 8))
        tk.Button(controls, text="Save", command=self.save_portfolio, bg=GREEN, fg="#06130F", relief="flat", font=FONT_NAV, padx=12, pady=6).grid(row=0, column=2, padx=(0, 8))
        tk.Button(controls, text="Load", command=self.load_portfolio, bg=CARD_SOFT, fg=FG, relief="flat", font=FONT_NAV, padx=12, pady=6).grid(row=0, column=3, padx=(0, 8))
        self.saved_combo = ttk.Combobox(controls, textvariable=self.saved_var, state="readonly", width=28)
        self.saved_combo.grid(row=0, column=4, padx=(0, 10))

        tk.Label(top, text="Selected Live Portfolio", bg=BG, fg=FG, font=FONT_H2).grid(row=1, column=0, sticky="w", pady=(0, 6))
        table_frame = tk.Frame(top, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        table_frame.grid(row=2, column=0, sticky="nsew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        cols = ("strategy", "symbol", "dir", "weight")
        self.portfolio_table = ttk.Treeview(table_frame, columns=cols, show="headings", style="Live.Treeview", height=12)
        for col, text, width, anchor in [
            ("strategy", "Strategy", 330, "w"),
            ("symbol", "Symbol", 90, "center"),
            ("dir", "Dir", 55, "center"),
            ("weight", "Weight", 80, "e"),
        ]:
            self.portfolio_table.heading(col, text=text)
            self.portfolio_table.column(col, width=width, anchor=anchor, stretch=True)
        self.portfolio_table.grid(row=0, column=0, sticky="nsew")
        scroll = tk.Scrollbar(table_frame, orient="vertical", command=self.portfolio_table.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.portfolio_table.configure(yscrollcommand=scroll.set)

        buttons = tk.Frame(top, bg=BG)
        buttons.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        tk.Button(buttons, text="Remove Selected", command=self.remove_selected, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=12, pady=8).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Equal Weight", command=self.equal_weight_portfolio, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=12, pady=8).pack(side="left", padx=(0, 8))
        tk.Button(buttons, text="Refresh Portfolio", command=self._update_portfolio_view, bg=CARD, fg=FG, relief="flat", font=FONT_NAV, padx=12, pady=8).pack(side="left", padx=(0, 8))
        tk.Checkbutton(buttons, text="Components", variable=self.show_components_var, command=self._update_portfolio_view, bg=BG, fg=MUTED, selectcolor=CARD_DARK, activebackground=BG, activeforeground=FG, font=FONT_SMALL).pack(side="left", padx=(10, 0))

    def _build_analytics_section(self, parent: tk.Frame) -> None:
        area = tk.Frame(parent, bg=BG)
        area.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        area.rowconfigure(2, weight=1)
        area.columnconfigure(0, weight=1)

        kpis = tk.Frame(area, bg=BG)
        kpis.grid(row=0, column=0, sticky="ew")
        for i in range(5):
            kpis.columnconfigure(i, weight=1)
        self.kpi_vars: Dict[str, Tuple[tk.StringVar, tk.Label]] = {}
        for i, (key, title, color) in enumerate([
            ("strategies", "Strategies", BLUE),
            ("net", "Net PnL", GREEN),
            ("dd", "Max DD", RED),
            ("pf", "Profit Factor", YELLOW),
            ("wr", "Win Rate", CYAN),
        ]):
            card = tk.Frame(kpis, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=i, sticky="ew", padx=4)
            tk.Label(card, text=title, bg=CARD, fg=MUTED, font=FONT_TINY).pack(anchor="w", padx=10, pady=(7, 0))
            var = tk.StringVar(value="-")
            lbl = tk.Label(card, textvariable=var, bg=CARD, fg=color, font=("Segoe UI", 13, "bold"))
            lbl.pack(anchor="w", padx=10, pady=(2, 8))
            self.kpi_vars[key] = (var, lbl)

        facts = tk.Frame(area, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        facts.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        for i in range(8):
            facts.columnconfigure(i, weight=1)
        tk.Label(facts, text="Portfolio Facts", bg=CARD, fg=FG, font=FONT_H2).grid(row=0, column=0, columnspan=8, sticky="w", padx=10, pady=(8, 4))
        self.fact_vars: Dict[str, Tuple[tk.StringVar, tk.Label]] = {}
        for i, name in enumerate(["Trades", "Wins", "Losses", "Avg Trade", "Best Trade", "Worst Trade", "Sharpe", "Worst Day"]):
            box = tk.Frame(facts, bg=CARD_DARK, highlightthickness=1, highlightbackground=BORDER)
            box.grid(row=1, column=i, sticky="ew", padx=4, pady=(0, 8))
            tk.Label(box, text=name, bg=CARD_DARK, fg=MUTED, font=FONT_TINY).pack(anchor="w", padx=7, pady=(5, 0))
            var = tk.StringVar(value="-")
            lbl = tk.Label(box, textvariable=var, bg=CARD_DARK, fg=FG, font=("Segoe UI", 9, "bold"))
            lbl.pack(anchor="w", padx=7, pady=(2, 5))
            self.fact_vars[name] = (var, lbl)

        lower = tk.Frame(area, bg=BG)
        lower.grid(row=2, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=3)
        lower.columnconfigure(1, weight=2)
        lower.rowconfigure(0, weight=1)

        chart_box = tk.Frame(lower, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        chart_box.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        chart_box.rowconfigure(1, weight=1)
        chart_box.columnconfigure(0, weight=1)
        tk.Label(chart_box, text="Portfolio Equity Curve", bg=CARD, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))
        self.fig = Figure(figsize=(6.2, 3.2), dpi=100)
        self.fig.patch.set_facecolor(CARD)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_box)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

        time_box = tk.Frame(lower, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        time_box.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        time_box.rowconfigure(1, weight=1)
        time_box.columnconfigure(0, weight=1)
        tk.Label(time_box, text="Daily / Weekly / Monthly Facts", bg=CARD, fg=FG, font=FONT_H2).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))
        self.time_text = tk.Text(time_box, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=("Consolas", 9), wrap="none")
        self.time_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)

    def _build_footer(self, root: tk.Frame) -> None:
        footer = tk.Frame(root, bg=BG)
        footer.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        tk.Label(footer, textvariable=self.status_var, bg=BG, fg=MUTED, font=FONT_SMALL).pack(side="left")

    # --------------------------------------------------------
    # DATA PREP

    def _prepare_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades is None or trades.empty:
            return pd.DataFrame()
        d = trades.copy()
        for col in ["open_time_utc", "close_time_utc"]:
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], errors="coerce", utc=True)
        for col in ["net_sum", "profit_sum", "commission_sum", "swap_sum"]:
            if col in d.columns:
                d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
        if "net_sum" not in d.columns:
            d["net_sum"] = 0.0
        for col in ["symbol", "direction", "canonical_bucket", "source_bucket", "strategy_id"]:
            if col in d.columns:
                d[col] = d[col].astype(str).fillna("").str.strip()
        if "direction" in d.columns:
            d["direction"] = d["direction"].str.upper().replace({"LONG": "BUY", "SHORT": "SELL"})
        else:
            d["direction"] = ""
        if "symbol" not in d.columns:
            d["symbol"] = "UNKNOWN"
        if "canonical_bucket" not in d.columns:
            if "source_bucket" in d.columns:
                d["canonical_bucket"] = d["source_bucket"]
            else:
                d["canonical_bucket"] = "UNKNOWN"
        if "close_time_utc" in d.columns:
            d = d.dropna(subset=["close_time_utc"]).sort_values("close_time_utc")
        return d.reset_index(drop=True)

    def _build_strategy_stats(self, trades: pd.DataFrame) -> pd.DataFrame:
        if trades.empty:
            return pd.DataFrame(columns=[
                "strategy", "symbol", "direction", "trades", "net_pnl", "profit_factor", "max_dd", "win_rate",
                "avg_trade", "best_trade", "worst_trade", "sharpe", "worst_day", "best_month", "score"
            ])
        rows = []
        grouped = trades.groupby("canonical_bucket", dropna=False)
        for strategy, g in grouped:
            g = g.sort_values("close_time_utc").copy()
            net = pd.to_numeric(g["net_sum"], errors="coerce").fillna(0.0)
            n = int(len(g))
            wins = int((net > 0).sum())
            losses = int((net < 0).sum())
            gross_profit = float(net[net > 0].sum())
            gross_loss = abs(float(net[net < 0].sum()))
            pf = gross_profit / gross_loss if gross_loss > 0 else math.inf if gross_profit > 0 else 0.0
            eq = net.cumsum()
            dd = eq - eq.cummax()
            max_dd = float(dd.min()) if len(dd) else 0.0

            symbol = str(g["symbol"].dropna().astype(str).iloc[0]) if "symbol" in g.columns and len(g) else "UNKNOWN"
            direction = str(g["direction"].dropna().astype(str).iloc[0]) if "direction" in g.columns and len(g) else ""

            daily = g.set_index("close_time_utc")["net_sum"].resample("1D").sum().fillna(0.0)
            sharpe = 0.0
            if len(daily) > 1 and float(daily.std()) != 0.0:
                sharpe = float((daily.mean() / daily.std()) * math.sqrt(252))
            worst_day = float(daily.min()) if len(daily) else 0.0
            monthly = g.set_index("close_time_utc")["net_sum"].resample("1M").sum()
            best_month = float(monthly.max()) if len(monthly) else 0.0
            score = self._score(float(net.sum()), max_dd, pf, wins / n if n else 0.0, n)
            rows.append({
                "strategy": str(strategy),
                "symbol": symbol,
                "direction": direction,
                "trades": n,
                "net_pnl": float(net.sum()),
                "profit_factor": pf,
                "max_dd": max_dd,
                "win_rate": wins / n if n else 0.0,
                "avg_trade": float(net.mean()) if n else 0.0,
                "best_trade": float(net.max()) if n else 0.0,
                "worst_trade": float(net.min()) if n else 0.0,
                "sharpe": sharpe,
                "worst_day": worst_day,
                "best_month": best_month,
                "score": score,
            })
        return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    def _score(self, pnl: float, max_dd: float, pf: float, win_rate_v: float, trades: int) -> float:
        risk_adj = pnl / abs(max_dd) if max_dd < 0 else pnl
        pf_part = 0.0 if math.isinf(pf) else pf
        return float(risk_adj + pf_part + win_rate_v + math.log(max(trades, 1)))

    def _populate_filter_values(self) -> None:
        symbols = ["ALL"]
        directions = ["ALL"]
        if not self.strategy_stats.empty:
            symbols += sorted([x for x in self.strategy_stats["symbol"].dropna().astype(str).unique().tolist() if x])
            directions += sorted([x for x in self.strategy_stats["direction"].dropna().astype(str).unique().tolist() if x])
        self.symbol_combo["values"] = symbols
        self.direction_combo["values"] = directions
        if self.symbol_var.get() not in symbols:
            self.symbol_var.set("ALL")
        if self.direction_var.get() not in directions:
            self.direction_var.set("ALL")

    # --------------------------------------------------------
    # FILTERS / TABLE

    def apply_filters(self) -> None:
        d = self.strategy_stats.copy()
        if d.empty:
            self.filtered_stats = d
            self._refresh_strategy_table()
            return
        q = self.search_var.get().strip().lower()
        if q:
            d = d[
                d["strategy"].astype(str).str.lower().str.contains(q, na=False)
                | d["symbol"].astype(str).str.lower().str.contains(q, na=False)
                | d["direction"].astype(str).str.lower().str.contains(q, na=False)
            ]
        symbol = self.symbol_var.get().strip()
        if symbol != "ALL":
            d = d[d["symbol"].astype(str) == symbol]
        direction = self.direction_var.get().strip()
        if direction != "ALL":
            d = d[d["direction"].astype(str) == direction]
        try:
            min_trades = int(float(self.min_trades_var.get() or 0))
        except Exception:
            min_trades = 0
        d = d[d["trades"] >= min_trades]
        try:
            min_pf = float(self.min_pf_var.get()) if self.min_pf_var.get().strip() else None
        except Exception:
            min_pf = None
        if min_pf is not None:
            d = d[d["profit_factor"] >= min_pf]
        try:
            min_pnl = float(self.min_pnl_var.get()) if self.min_pnl_var.get().strip() else None
        except Exception:
            min_pnl = None
        if min_pnl is not None:
            d = d[d["net_pnl"] >= min_pnl]
        try:
            max_dd = abs(float(self.max_dd_var.get())) if self.max_dd_var.get().strip() else None
        except Exception:
            max_dd = None
        if max_dd is not None:
            d = d[d["max_dd"].abs() <= max_dd]

        sort_map = {
            "Score": "score",
            "Highest PnL": "net_pnl",
            "Lowest DD": "max_dd_abs",
            "Profit Factor": "profit_factor",
            "Win Rate": "win_rate",
            "Most Trades": "trades",
            "Avg Trade": "avg_trade",
            "Best Trade": "best_trade",
            "Worst Trade": "worst_trade",
            "Sharpe": "sharpe",
            "Worst Day": "worst_day",
            "Best Month": "best_month",
        }
        d["max_dd_abs"] = d["max_dd"].abs()
        sort_col = sort_map.get(self.sort_var.get(), "score")
        try:
            metric_min = float(self.metric_min_var.get()) if self.metric_min_var.get().strip() else None
        except Exception:
            metric_min = None
        try:
            metric_max = float(self.metric_max_var.get()) if self.metric_max_var.get().strip() else None
        except Exception:
            metric_max = None
        if metric_min is not None and sort_col in d.columns:
            d = d[d[sort_col] >= metric_min]
        if metric_max is not None and sort_col in d.columns:
            d = d[d[sort_col] <= metric_max]

        high_first = self.order_var.get() != "Low First"
        if self.sort_var.get() == "Lowest DD" and self.order_var.get() == "High First":
            high_first = False
        d = d.sort_values(sort_col, ascending=not high_first).reset_index(drop=True)
        self.filtered_stats = d
        self._refresh_strategy_table()

    def reset_filters(self) -> None:
        self.search_var.set("")
        self.symbol_var.set("ALL")
        self.direction_var.set("ALL")
        self.min_trades_var.set("0")
        self.min_pf_var.set("")
        self.min_pnl_var.set("")
        self.max_dd_var.set("")
        self.metric_min_var.set("")
        self.metric_max_var.set("")
        self.sort_var.set("Score")
        self.order_var.set("High First")
        self.apply_filters()

    def _strategy_table_values(self, row: pd.Series) -> tuple[str, str, str, str, str, str, str, str, str, str]:
        strategy = str(row.get("strategy", ""))
        symbol = str(row.get("symbol", ""))
        direction = str(row.get("direction", ""))
        trades = str(int(float(row.get("trades", 0) or 0)))
        net_pnl = fmt_money(float(row.get("net_pnl", 0.0) or 0.0))
        profit_factor_text = fmt_num(float(row.get("profit_factor", 0.0) or 0.0), 2)
        max_dd_text = fmt_money(float(row.get("max_dd", 0.0) or 0.0))
        win_rate_text = fmt_pct(float(row.get("win_rate", 0.0) or 0.0), 1)
        avg_trade_text = fmt_money(float(row.get("avg_trade", 0.0) or 0.0))
        score_text = fmt_num(float(row.get("score", 0.0) or 0.0), 2)
        return (
            strategy,
            symbol,
            direction,
            trades,
            net_pnl,
            profit_factor_text,
            max_dd_text,
            win_rate_text,
            avg_trade_text,
            score_text,
        )

    def _refresh_strategy_table(self) -> None:
        self.strategy_table.delete(*self.strategy_table.get_children())
        if self.filtered_stats.empty:
            self.status_var.set("Filtered: 0 | Total: 0")
            return

        for idx, row in self.filtered_stats.iterrows():
            net_value = float(row.get("net_pnl", 0.0) or 0.0)
            tag = "positive" if net_value > 0 else "negative" if net_value < 0 else "flat"
            values = self._strategy_table_values(row)
            self.strategy_table.insert("", "end", iid=str(idx), values=values, tags=(tag,))

        total = len(self.strategy_stats)
        positive_count = int((self.filtered_stats["net_pnl"] > 0).sum())
        negative_count = int((self.filtered_stats["net_pnl"] < 0).sum())
        flat_count = int((self.filtered_stats["net_pnl"] == 0).sum())
        filtered_count = len(self.filtered_stats)

        self.status_var.set(
            f"Filtered: {filtered_count} | Total: {total} | "
            f"Positive: {positive_count} | Negative: {negative_count} | Flat: {flat_count}"
        )

    # --------------------------------------------------------
    # PORTFOLIO ACTIONS

    def _selected_strategy_rows(self) -> pd.DataFrame:
        sel = self.strategy_table.selection()
        if not sel:
            return pd.DataFrame()
        rows = []
        for iid in sel:
            try:
                idx = int(iid)
                if idx in self.filtered_stats.index:
                    rows.append(self.filtered_stats.loc[idx])
            except Exception:
                pass
        return pd.DataFrame(rows)

    def add_selected_strategy(self) -> None:
        rows = self._selected_strategy_rows()
        if rows.empty:
            self.status_var.set("No candidate selected")
            return
        try:
            weight = float(self.weight_var.get() or 1.0)
        except Exception:
            weight = 1.0
        for _, r in rows.iterrows():
            self._add_portfolio_row(r, weight)
        self._update_portfolio_view()

    def add_top_n(self, n: int) -> None:
        try:
            weight = float(self.weight_var.get() or 1.0)
        except Exception:
            weight = 1.0
        for _, r in self.filtered_stats.head(n).iterrows():
            self._add_portfolio_row(r, weight)
        self._update_portfolio_view()

    def add_top_filtered(self) -> None:
        self.add_top_n(len(self.filtered_stats))

    def _add_portfolio_row(self, r: pd.Series, weight: float) -> None:
        strategy = str(r.get("strategy", ""))
        if not strategy:
            return
        if not self.portfolio.empty and (self.portfolio["strategy"].astype(str) == strategy).any():
            return
        row = pd.DataFrame([{
            "strategy": strategy,
            "symbol": str(r.get("symbol", "")),
            "direction": str(r.get("direction", "")),
            "weight": float(weight),
        }])
        self.portfolio = pd.concat([self.portfolio, row], ignore_index=True)

    def remove_selected(self) -> None:
        sel = self.portfolio_table.selection()
        if not sel or self.portfolio.empty:
            return
        names = []
        for iid in sel:
            vals = self.portfolio_table.item(iid, "values")
            if vals:
                names.append(str(vals[0]))
        if names:
            self.portfolio = self.portfolio[~self.portfolio["strategy"].astype(str).isin(names)].reset_index(drop=True)
            self._update_portfolio_view()

    def equal_weight_portfolio(self) -> None:
        if self.portfolio.empty:
            return
        self.portfolio["weight"] = 1.0 / len(self.portfolio)
        self._update_portfolio_view()

    def clear_portfolio(self) -> None:
        self.portfolio = self.portfolio.iloc[0:0].copy()
        self._update_portfolio_view()

    def _portfolio_trades(self) -> pd.DataFrame:
        if self.portfolio.empty or self.trades.empty:
            return pd.DataFrame()
        frames = []
        for _, r in self.portfolio.iterrows():
            strategy = str(r.get("strategy", ""))
            weight = float(r.get("weight", 1.0))
            g = self.trades[self.trades["canonical_bucket"].astype(str) == strategy].copy()
            if g.empty:
                continue
            g["net_sum"] = pd.to_numeric(g["net_sum"], errors="coerce").fillna(0.0) * weight
            g["portfolio_strategy"] = strategy
            g["portfolio_weight"] = weight
            frames.append(g)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("close_time_utc").reset_index(drop=True)

    def _update_portfolio_view(self) -> None:
        self.portfolio_table.delete(*self.portfolio_table.get_children())
        if not self.portfolio.empty:
            for _, r in self.portfolio.iterrows():
                self.portfolio_table.insert("", "end", values=(
                    str(r.get("strategy", "")),
                    str(r.get("symbol", "")),
                    str(r.get("direction", "")),
                    f"{float(r.get('weight', 1.0)):.4f}",
                ))
        self.portfolio_trades = self._portfolio_trades()
        self._update_kpis_and_facts(self.portfolio_trades)
        self._plot_portfolio_equity(self.portfolio_trades)
        self._update_time_text(self.portfolio_trades)

    # --------------------------------------------------------
    # METRICS / PLOTS

    def _portfolio_metrics(self, trades: pd.DataFrame) -> dict:
        if trades.empty:
            return {
                "strategies": 0, "net": 0.0, "dd": 0.0, "pf": 0.0, "wr": 0.0, "trades": 0,
                "wins": 0, "losses": 0, "avg": 0.0, "best": 0.0, "worst": 0.0, "sharpe": 0.0, "worst_day": 0.0,
            }
        d = trades.copy().sort_values("close_time_utc")
        net = pd.to_numeric(d["net_sum"], errors="coerce").fillna(0.0)
        eq = net.cumsum()
        dd = eq - eq.cummax()
        wins = int((net > 0).sum())
        losses = int((net < 0).sum())
        gp = float(net[net > 0].sum())
        gl = abs(float(net[net < 0].sum()))
        pf = gp / gl if gl > 0 else math.inf if gp > 0 else 0.0
        daily = d.set_index("close_time_utc")["net_sum"].resample("1D").sum().fillna(0.0)
        sharpe = 0.0
        if len(daily) > 1 and float(daily.std()) != 0.0:
            sharpe = float((daily.mean() / daily.std()) * math.sqrt(252))
        return {
            "strategies": len(self.portfolio),
            "net": float(net.sum()),
            "dd": float(dd.min()) if len(dd) else 0.0,
            "pf": pf,
            "wr": wins / len(net) if len(net) else 0.0,
            "trades": int(len(d)),
            "wins": wins,
            "losses": losses,
            "avg": float(net.mean()) if len(net) else 0.0,
            "best": float(net.max()) if len(net) else 0.0,
            "worst": float(net.min()) if len(net) else 0.0,
            "sharpe": sharpe,
            "worst_day": float(daily.min()) if len(daily) else 0.0,
        }

    def _set_kpi(self, key: str, value: str, color: Optional[str] = None) -> None:
        if key not in self.kpi_vars:
            return
        value_var, value_label = self.kpi_vars[key]
        value_var.set(value)
        if color is not None:
            value_label.configure(fg=color)

    def _set_fact(self, key: str, value: str, color: Optional[str] = None) -> None:
        if key not in self.fact_vars:
            return
        value_var, value_label = self.fact_vars[key]
        value_var.set(value)
        if color is not None:
            value_label.configure(fg=color)

    def _update_kpis_and_facts(self, trades: pd.DataFrame) -> None:
        metrics = self._portfolio_metrics(trades)

        strategy_count = int(metrics.get("strategies", 0))
        net_value = float(metrics.get("net", 0.0))
        drawdown_value = float(metrics.get("dd", 0.0))
        profit_factor_value = float(metrics.get("pf", 0.0))
        win_rate_value = float(metrics.get("wr", 0.0))
        trade_count = int(metrics.get("trades", 0))
        wins = int(metrics.get("wins", 0))
        losses = int(metrics.get("losses", 0))
        avg_trade = float(metrics.get("avg", 0.0))
        best_trade = float(metrics.get("best", 0.0))
        worst_trade = float(metrics.get("worst", 0.0))
        sharpe = float(metrics.get("sharpe", 0.0))
        worst_day = float(metrics.get("worst_day", 0.0))

        self._set_kpi("strategies", str(strategy_count))
        self._set_kpi("net", fmt_money(net_value), pnl_color(net_value))
        self._set_kpi("dd", fmt_money(drawdown_value), dd_color(drawdown_value))
        self._set_kpi("pf", fmt_num(profit_factor_value, 2))
        self._set_kpi("wr", fmt_pct(win_rate_value, 1))

        self._set_fact("Trades", str(trade_count))
        self._set_fact("Wins", str(wins), GREEN)
        self._set_fact("Losses", str(losses), RED)
        self._set_fact("Avg Trade", fmt_money(avg_trade), pnl_color(avg_trade))
        self._set_fact("Best Trade", fmt_money(best_trade), GREEN)
        self._set_fact("Worst Trade", fmt_money(worst_trade), RED)
        self._set_fact("Sharpe", fmt_num(sharpe, 2))
        self._set_fact("Worst Day", fmt_money(worst_day), RED if worst_day < 0 else GREEN)

    def _plot_portfolio_equity(self, trades: pd.DataFrame) -> None:
        self.ax.clear()
        self.fig.patch.set_facecolor(CARD)
        self.ax.set_facecolor(CARD)
        if trades.empty:
            self.ax.text(0.5, 0.5, "No portfolio selected", color=MUTED, ha="center", va="center", transform=self.ax.transAxes)
            style_ax(self.ax)
            self.canvas.draw_idle()
            return
        d = trades.copy().sort_values("close_time_utc")
        if self.show_components_var.get() and "portfolio_strategy" in d.columns:
            for name, g in d.groupby("portfolio_strategy"):
                daily = g.set_index("close_time_utc")["net_sum"].resample("1D").sum().fillna(0.0)
                curve = START_EQUITY + daily.cumsum()
                self.ax.plot(curve.index, curve.values, linewidth=0.8, alpha=0.45, label=str(name)[:20])
        daily_total = d.set_index("close_time_utc")["net_sum"].resample("1D").sum().fillna(0.0)
        curve = START_EQUITY + daily_total.cumsum()
        self.ax.plot(curve.index, curve.values, color=GREEN, linewidth=2.0, label="Portfolio")
        self.ax.fill_between(curve.index, curve.values, START_EQUITY, color=GREEN, alpha=0.16)
        self.ax.axhline(START_EQUITY, color=GRID, linewidth=0.9)
        self.ax.yaxis.set_major_formatter(lambda x, _pos: f"${x / 1000:.0f}K")
        if self.show_components_var.get():
            self.ax.legend(loc="upper left", frameon=False, labelcolor=MUTED, fontsize=6)
        style_ax(self.ax)
        self.fig.tight_layout(pad=1.0)
        self.canvas.draw_idle()

    def _period_metrics(self, trades: pd.DataFrame, freq: str) -> dict:
        if trades.empty:
            return {"periods": 0, "avg": 0.0, "best": 0.0, "worst": 0.0, "q95": 0.0, "q75": 0.0, "median": 0.0, "q25": 0.0, "q05": 0.0, "pf": 0.0, "wr": 0.0}
        d = trades.copy().sort_values("close_time_utc")
        frame = d.set_index("close_time_utc")["net_sum"].resample(freq).sum()
        frame = frame[frame != 0]
        if frame.empty:
            return {"periods": 0, "avg": 0.0, "best": 0.0, "worst": 0.0, "q95": 0.0, "q75": 0.0, "median": 0.0, "q25": 0.0, "q05": 0.0, "pf": 0.0, "wr": 0.0}
        gp = float(frame[frame > 0].sum())
        gl = abs(float(frame[frame < 0].sum()))
        pf = gp / gl if gl > 0 else math.inf if gp > 0 else 0.0
        return {
            "periods": int(len(frame)),
            "avg": float(frame.mean()),
            "best": float(frame.max()),
            "worst": float(frame.min()),
            "q95": float(frame.quantile(0.95)),
            "q75": float(frame.quantile(0.75)),
            "median": float(frame.quantile(0.50)),
            "q25": float(frame.quantile(0.25)),
            "q05": float(frame.quantile(0.05)),
            "pf": pf,
            "wr": float((frame > 0).sum() / len(frame)),
        }

    def _update_time_text(self, trades: pd.DataFrame) -> None:
        self.time_text.delete("1.0", "end")
        if trades.empty:
            self.time_text.insert("end", "No portfolio selected.")
            return
        lines = []
        for title, freq in [("DAILY", "1D"), ("WEEKLY", "1W"), ("MONTHLY", "1M")]:
            m = self._period_metrics(trades, freq)
            lines.extend([
                title,
                "-" * 48,
                f"Periods       : {m['periods']}",
                f"Avg PnL       : {fmt_money(m['avg'])}",
                f"Best Period   : {fmt_money(m['best'])}",
                f"Worst Period  : {fmt_money(m['worst'])}",
                f"Q95 / Q75     : {fmt_money(m['q95'])} / {fmt_money(m['q75'])}",
                f"Median        : {fmt_money(m['median'])}",
                f"Q25 / Q05     : {fmt_money(m['q25'])} / {fmt_money(m['q05'])}",
                f"Win Rate      : {fmt_pct(m['wr'], 1)}",
                "Profit Factor : " + fmt_num(float(m.get("pf", 0.0)), 2),
                "",
            ])
        self.time_text.insert("end", "\n".join(lines))

    # --------------------------------------------------------
    # STRATEGY FACTS POPUP

    def open_selected_strategy_facts(self) -> None:
        rows = self._selected_strategy_rows()
        if rows.empty:
            self.status_var.set("No candidate selected")
            return
        r = rows.iloc[0]
        strategy = str(r["strategy"])
        g = self.trades[self.trades["canonical_bucket"].astype(str) == strategy].copy()
        win = tk.Toplevel(self)
        win.title(f"Strategy Facts · {strategy}")
        win.geometry("900x600")
        win.configure(bg=BG)
        tk.Label(win, text=strategy, bg=BG, fg=FG, font=FONT_TITLE).pack(anchor="w", padx=16, pady=(14, 6))
        txt = tk.Text(win, bg=CARD_DARK, fg=FG, insertbackground=FG, relief="flat", font=("Consolas", 10))
        txt.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        info = self._strategy_report_text(r, g)
        txt.insert("end", info)

    def _strategy_report_text(self, row: pd.Series, trades: pd.DataFrame) -> str:
        lines = []
        lines.append("BASIC")
        lines.append("-" * 50)
        for key in ["symbol", "direction", "trades", "net_pnl", "profit_factor", "max_dd", "win_rate", "avg_trade", "best_trade", "worst_trade", "sharpe", "worst_day", "best_month", "score"]:
            val = row.get(key, "")
            if key in {"net_pnl", "max_dd", "avg_trade", "best_trade", "worst_trade", "worst_day", "best_month"}:
                val = fmt_money(val)
            elif key in {"win_rate"}:
                val = fmt_pct(val, 1)
            elif key in {"profit_factor", "sharpe", "score"}:
                val = fmt_num(val, 2)
            lines.append(f"{key:18}: {val}")
        lines.append("")
        for title, freq in [("DAILY", "1D"), ("WEEKLY", "1W"), ("MONTHLY", "1M")]:
            m = self._period_metrics(trades, freq)
            lines.append(title)
            lines.append("-" * 50)
            lines.append(f"Periods       : {m['periods']}")
            lines.append(f"Avg PnL       : {fmt_money(m['avg'])}")
            lines.append(f"Best Period   : {fmt_money(m['best'])}")
            lines.append(f"Worst Period  : {fmt_money(m['worst'])}")
            lines.append(f"Median        : {fmt_money(m['median'])}")
            lines.append(f"Q05/Q95       : {fmt_money(m['q05'])} / {fmt_money(m['q95'])}")
            lines.append("")
        return "\n".join(lines)

    # --------------------------------------------------------
    # SAVE / LOAD

    def _portfolio_path(self, name: str | None = None) -> Path:
        raw_name = name if name is not None else self.portfolio_name_var.get()
        safe_name = sanitize_portfolio_name(raw_name)
        return self.portfolio_root / f"{safe_name}.json"

    def save_portfolio(self) -> None:
        path = self._portfolio_path()
        payload = {
            "portfolio_name": path.stem,
            "account_name": self.account_name,
            "account_root": str(self.account_root),
            "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
            "items": self.portfolio.to_dict(orient="records"),
            "metrics": self._portfolio_metrics(self.portfolio_trades),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.portfolio.to_csv(path.with_suffix(".csv"), index=False)
        self._refresh_saved_combo()
        self.status_var.set(f"Saved: {path}")

    def _refresh_saved_combo(self) -> None:
        names = sorted([p.stem for p in self.portfolio_root.glob("*.json")]) if self.portfolio_root.exists() else []
        self.saved_combo["values"] = names
        if names and not self.saved_var.get():
            self.saved_var.set(names[0])

    def load_portfolio(self) -> None:
        name = self.saved_var.get().strip()
        if not name:
            return
        path = self._portfolio_path(name)
        if not path.exists():
            self.status_var.set(f"Not found: {path}")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("items", [])
        self.portfolio = pd.DataFrame(items)
        for col in ["strategy", "symbol", "direction", "weight"]:
            if col not in self.portfolio.columns:
                self.portfolio[col] = 1.0 if col == "weight" else ""
        self.portfolio["weight"] = pd.to_numeric(self.portfolio["weight"], errors="coerce").fillna(1.0)
        self.portfolio_name_var.set(path.stem)
        self._update_portfolio_view()
        self.status_var.set(f"Loaded: {path}")


# ============================================================
# MAIN BOARD ENTRYPOINT
# ============================================================

def build_panel(parent, repository=None, **kwargs):
    """
    Main.py-compatible dashboard building block API.
    """
    repo = repository if isinstance(repository, AccountTradeRepository) else None
    return AccountMonitorPanel(parent, repo=repo, **kwargs)


# ============================================================
# STANDALONE RUN
# ============================================================

def main() -> None:
    app = tk.Tk()
    app.title("Live Trades Performance Dashboard")
    app.geometry("1720x980")
    app.minsize(1350, 820)
    app.configure(bg=BG)
    panel = AccountMonitorPanel(app)
    panel.pack(fill="both", expand=True)
    app.mainloop()


if __name__ == "__main__":
    main()

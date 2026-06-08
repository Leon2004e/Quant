# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: market_multi_ohlc_chart_dashboard
# script_name: Market Multi OHLC Chart Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Market
# asset_type: Dashboard Building Block
# purpose: Bloomberg-style multi OHLC chart viewer for QUANT market data
# inputs:
# - Data_Center/Data/1_Pipeline/Market/ohcl
# outputs:
# - Dashboard UI
# dependencies:
# - tkinter
# - pathlib
# - pandas
# schedule: manual
# version: v3.0.0
# last_reviewed: 2026-06-06
# ============================================================

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from datetime import datetime
import pandas as pd


BG = "#000000"
PANEL = "#050505"
HEADER = "#080808"
GRID = "#151515"
BORDER = "#5A3300"
ORANGE = "#FF9900"
ORANGE_DARK = "#6A3A00"
GREEN = "#00FF66"
RED = "#FF3333"
WHITE = "#E8E8E8"
GRAY = "#888888"
YELLOW = "#FFD400"

FONT_TITLE = ("Consolas", 14, "bold")
FONT_HEAD = ("Consolas", 10, "bold")
FONT_MAIN = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)


def find_quant_root(start: Path) -> Path:
    start = Path(start).resolve()
    for p in [start] + list(start.parents):
        if (p / "Dashboard").exists() and (p / "Data_Center").exists():
            return p
    raise FileNotFoundError("QUANT root not found. Expected Dashboard/ and Data_Center/.")


class MarketMultiOHLCChartDashboard(tk.Frame):
    def __init__(self, parent, repository=None, **kwargs):
        super().__init__(parent, bg=BG, **kwargs)

        self.repository = repository
        self.quant_root = find_quant_root(Path(__file__))
        self.data_root = self.quant_root / "Data_Center" / "Data" / "1_Pipeline" / "Market" / "ohcl"

        self.symbols = []
        self.timeframes = []
        self.selected_symbol = "AUDJPY"
        self.selected_timeframe = "D1"

        self.charts = []
        self.data_cache = {}

        self.symbol_hitboxes = []
        self.tf_hitboxes = []
        self.button_hitboxes = []
        self.chart_close_hitboxes = []

        self.canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_click)

        self.refresh_data()
        self.tick_clock()

    # ========================================================
    # DATA
    # ========================================================

    def refresh_data(self):
        try:
            if not self.data_root.exists():
                raise FileNotFoundError(str(self.data_root))

            order = ["M5", "M15", "H1", "H4", "H8", "H12", "D1", "W1", "MN1", "Q", "Y"]
            found = sorted([p.name for p in self.data_root.iterdir() if p.is_dir()])
            self.timeframes = [x for x in order if x in found] + [x for x in found if x not in order]

            symbols = set()
            for tf in self.timeframes:
                for f in (self.data_root / tf).glob("*.parquet"):
                    symbols.add(f.stem)

            self.symbols = sorted(symbols)

            if self.selected_symbol not in self.symbols and self.symbols:
                self.selected_symbol = self.symbols[0]

            if self.selected_timeframe not in self.timeframes and self.timeframes:
                self.selected_timeframe = self.timeframes[0]

            if not self.charts:
                self.charts.append((self.selected_symbol, self.selected_timeframe))

            self.data_cache.clear()
            self.redraw()

        except Exception as e:
            messagebox.showerror("Refresh Error", str(e))

    def load_chart_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)

        if key in self.data_cache:
            return self.data_cache[key]

        path = self.data_root / timeframe / f"{symbol}.parquet"

        if not path.exists():
            df = pd.DataFrame()
            self.data_cache[key] = df
            return df

        try:
            df = pd.read_parquet(path)

            for col in ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
                if col not in df.columns:
                    df[col] = pd.NA

            df["time"] = pd.to_datetime(df["time"], errors="coerce")
            df = df.dropna(subset=["time"]).sort_values("time")

            for col in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "high", "low", "close"])
            self.data_cache[key] = df
            return df

        except Exception:
            df = pd.DataFrame()
            self.data_cache[key] = df
            return df

    # ========================================================
    # DRAW BASIC
    # ========================================================

    def redraw(self):
        c = self.canvas
        c.delete("all")

        self.symbol_hitboxes.clear()
        self.tf_hitboxes.clear()
        self.button_hitboxes.clear()
        self.chart_close_hitboxes.clear()

        w = max(c.winfo_width(), 1300)
        h = max(c.winfo_height(), 850)

        self.draw_header(0, 0, w, 42)
        self.draw_control_bar(6, 48, w - 12, 66)

        main_y = 122
        footer_h = 28
        main_h = h - main_y - footer_h - 8

        left_w = 310
        gap = 6

        left_x = 6
        charts_x = left_x + left_w + gap
        charts_w = w - charts_x - 6

        self.draw_left(left_x, main_y, left_w, main_h)
        self.draw_multi_chart_area(charts_x, main_y, charts_w, main_h)
        self.draw_footer(6, h - footer_h - 3, w - 12, footer_h)

    def rect(self, x, y, w, h, fill=PANEL, outline=BORDER, width=1):
        self.canvas.create_rectangle(x, y, x + w, y + h, fill=fill, outline=outline, width=width)

    def text(self, x, y, text, fill=WHITE, font=FONT_MAIN, anchor="nw"):
        self.canvas.create_text(x, y, text=text, fill=fill, font=font, anchor=anchor)

    def line(self, x1, y1, x2, y2, fill=GRID, width=1, dash=None):
        self.canvas.create_line(x1, y1, x2, y2, fill=fill, width=width, dash=dash)

    def button(self, x, y, w, h, label, action, active=False):
        fill = ORANGE if active else BG
        fg = BG if active else ORANGE
        self.rect(x, y, w, h, fill=fill, outline=ORANGE)
        self.text(x + w / 2, y + h / 2, label, fg, FONT_HEAD, "center")
        self.button_hitboxes.append((action, x, y, w, h))

    # ========================================================
    # HEADER
    # ========================================================

    def draw_header(self, x, y, w, h):
        self.rect(x, y, w, h, fill=BG, outline=ORANGE)

        self.text(14, y + 12, "QUANT", WHITE, FONT_TITLE)
        self.text(73, y + 12, "TERMINAL", ORANGE, FONT_TITLE)
        self.text(230, y + 13, datetime.now().strftime("%H:%M:%S"), WHITE, FONT_HEAD)

        nav = ["OVRV", "PORT", "STRAT", "TRDS", "RSCH", "MKT", "PIPE", "REG", "SET"]
        nx = 360

        for item in nav:
            active = item == "MKT"
            if active:
                self.rect(nx - 5, y + 5, 54, 30, fill=ORANGE, outline=ORANGE)
            self.text(nx + 8, y + 14, item, BG if active else WHITE, FONT_HEAD)
            nx += 66

        self.text(w - 300, y + 13, "REFRESH", ORANGE, FONT_HEAD)
        self.text(w - 210, y + 13, "MSG", WHITE, FONT_HEAD)
        self.rect(w - 174, y + 8, 24, 22, fill=YELLOW, outline=YELLOW)
        self.text(w - 168, y + 12, "3", BG, FONT_HEAD)
        self.text(w - 125, y + 13, "<HELP>", GREEN, FONT_HEAD)

        self.button_hitboxes.append(("refresh", w - 310, y, 90, h))

    def draw_control_bar(self, x, y, w, h):
        self.rect(x, y, w, h, fill=PANEL, outline=BORDER)

        self.dropdown(x + 12, y + 10, 210, "SYMBOL", self.selected_symbol)
        self.dropdown(x + 245, y + 10, 140, "TIMEFRAME", self.selected_timeframe)
        self.dropdown(x + 410, y + 10, 270, "DATA SET", "1_Pipeline/Market/ohcl")

        self.text(x + 710, y + 26, "DATA STATUS: LIVE", GREEN, FONT_HEAD)

        bx = x + w - 450
        self.button(bx, y + 18, 80, 30, "+ CHART", "add_chart")
        self.button(bx + 90, y + 18, 70, 30, "1", "layout_1", len(self.charts) == 1)
        self.button(bx + 168, y + 18, 70, 30, "2", "layout_2", len(self.charts) == 2)
        self.button(bx + 246, y + 18, 70, 30, "4", "layout_4", len(self.charts) >= 3)
        self.button(bx + 324, y + 18, 110, 30, "CLEAR", "clear_charts")

    def dropdown(self, x, y, w, label, value):
        self.text(x, y, label, ORANGE, FONT_SMALL)
        self.rect(x, y + 20, w, 28, fill=BG, outline=BORDER)
        self.text(x + 8, y + 27, value, WHITE, FONT_MAIN)
        self.text(x + w - 18, y + 27, "▼", GRAY, FONT_SMALL)

    # ========================================================
    # LEFT PANEL
    # ========================================================

    def draw_left(self, x, y, w, h):
        symbols_h = int(h * 0.58)
        tf_h = h - symbols_h - 6

        self.draw_symbol_panel(x, y, w, symbols_h)
        self.draw_timeframe_panel(x, y + symbols_h + 6, w, tf_h)

    def panel_header(self, x, y, w, title):
        self.rect(x, y, w, 28, fill=HEADER, outline=BORDER)
        self.text(x + 10, y + 7, title, ORANGE, FONT_HEAD)

    def draw_symbol_panel(self, x, y, w, h):
        self.rect(x, y, w, h, fill=PANEL, outline=BORDER)
        self.panel_header(x, y, w, f"1) SYMBOLS ({len(self.symbols)})")

        table_y = y + 48
        row_h = 24

        self.text(x + 18, table_y - 22, "SYMBOL", ORANGE, FONT_SMALL)
        self.text(x + 130, table_y - 22, "LAST", ORANGE, FONT_SMALL)
        self.text(x + 210, table_y - 22, "CHG %", ORANGE, FONT_SMALL)

        max_rows = max(8, int((h - 60) / row_h))

        for i, sym in enumerate(self.symbols[:max_rows]):
            yy = table_y + i * row_h
            selected = sym == self.selected_symbol

            if selected:
                self.rect(x + 8, yy - 2, w - 16, row_h, fill=ORANGE_DARK, outline=ORANGE_DARK)

            last, chg, pct = self.snapshot(sym)
            col = GREEN if self.safe_float(pct) >= 0 else RED

            self.text(x + 18, yy + 3, sym, WHITE, FONT_MAIN)
            self.text(x + 130, yy + 3, last, WHITE, FONT_MAIN)
            self.text(x + 210, yy + 3, pct, col, FONT_MAIN)

            self.symbol_hitboxes.append((sym, x + 8, yy - 2, w - 16, row_h))
            self.line(x + 10, yy + row_h - 1, x + w - 10, yy + row_h - 1, "#111111")

    def draw_timeframe_panel(self, x, y, w, h):
        self.rect(x, y, w, h, fill=PANEL, outline=BORDER)
        self.panel_header(x, y, w, f"2) TIMEFRAMES ({len(self.timeframes)})")

        desc = {
            "M5": "5 Minutes", "M15": "15 Minutes", "H1": "1 Hour",
            "H4": "4 Hours", "H8": "8 Hours", "H12": "12 Hours",
            "D1": "1 Day", "W1": "1 Week", "MN1": "1 Month",
            "Q": "1 Quarter", "Y": "1 Year"
        }

        table_y = y + 48
        row_h = 24

        self.text(x + 35, table_y - 22, "TF", ORANGE, FONT_SMALL)
        self.text(x + 105, table_y - 22, "DESCRIPTION", ORANGE, FONT_SMALL)
        self.text(x + 250, table_y - 22, "STATUS", ORANGE, FONT_SMALL)

        for i, tf in enumerate(self.timeframes[:12]):
            yy = table_y + i * row_h
            selected = tf == self.selected_timeframe

            if selected:
                self.rect(x + 8, yy - 2, w - 16, row_h, fill=ORANGE_DARK, outline=ORANGE_DARK)

            self.text(x + 35, yy + 3, tf, WHITE, FONT_MAIN)
            self.text(x + 105, yy + 3, desc.get(tf, tf), WHITE, FONT_MAIN)
            self.text(x + 255, yy + 3, "OK", GREEN, FONT_MAIN)

            self.tf_hitboxes.append((tf, x + 8, yy - 2, w - 16, row_h))
            self.line(x + 10, yy + row_h - 1, x + w - 10, yy + row_h - 1, "#111111")

    # ========================================================
    # MULTI CHART AREA
    # ========================================================

    def draw_multi_chart_area(self, x, y, w, h):
        n = len(self.charts)

        if n <= 1:
            slots = [(x, y, w, h)]
        elif n == 2:
            gap = 6
            slots = [
                (x, y, (w - gap) / 2, h),
                (x + (w + gap) / 2, y, (w - gap) / 2, h),
            ]
        else:
            gap = 6
            cw = (w - gap) / 2
            ch = (h - gap) / 2
            slots = [
                (x, y, cw, ch),
                (x + cw + gap, y, cw, ch),
                (x, y + ch + gap, cw, ch),
                (x + cw + gap, y + ch + gap, cw, ch),
            ]

        for i, slot in enumerate(slots[:len(self.charts)]):
            sx, sy, sw, sh = slot
            symbol, tf = self.charts[i]
            df = self.load_chart_data(symbol, tf)
            self.draw_chart_panel(sx, sy, sw, sh, symbol, tf, df, i)

    def draw_chart_panel(self, x, y, w, h, symbol, timeframe, df, idx):
        self.rect(x, y, w, h, fill=PANEL, outline=BORDER)

        header_h = 30
        self.rect(x, y, w, header_h, fill=HEADER, outline=BORDER)

        self.text(x + 10, y + 8, f"{idx + 1}) {symbol} · {timeframe}", ORANGE, FONT_HEAD)
        self.text(x + w - 74, y + 8, "CANDLE", ORANGE, FONT_SMALL)
        self.text(x + w - 20, y + 8, "X", RED, FONT_HEAD, "center")

        self.chart_close_hitboxes.append((idx, x + w - 35, y, 35, 30))

        self.draw_chart(x + 8, y + 38, w - 16, h - 46, symbol, timeframe, df)

    def draw_chart(self, x, y, w, h, symbol, timeframe, df):
        self.rect(x, y, w, h, fill=BG, outline="#111111")

        if df.empty:
            self.text(x + 20, y + 20, "NO DATA", RED, FONT_HEAD)
            return

        data = df.tail(300).copy()
        last = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else last

        chg = float(last["close"]) - float(prev["close"])
        pct = chg / float(prev["close"]) * 100 if float(prev["close"]) else 0
        chg_col = GREEN if chg >= 0 else RED

        self.text(
            x + 10,
            y + 8,
            f"O {self._fmt(last['open'])}  H {self._fmt(last['high'])}  L {self._fmt(last['low'])}  C {self._fmt(last['close'])}  {chg:+.3f} ({pct:+.2f}%)",
            chg_col,
            FONT_HEAD,
        )

        plot_x = x + 45
        plot_y = y + 40
        plot_w = w - 95
        plot_h = h - 70

        if plot_w < 100 or plot_h < 100:
            return

        price_h = int(plot_h * 0.76)
        vol_gap = 14
        vol_h = plot_h - price_h - vol_gap
        vol_y = plot_y + price_h + vol_gap

        max_p = float(data["high"].max())
        min_p = float(data["low"].min())

        if max_p == min_p:
            return

        pad = (max_p - min_p) * 0.05
        max_p += pad
        min_p -= pad

        def xp(i):
            return plot_x + i * plot_w / max(len(data) - 1, 1)

        def yp(price):
            return plot_y + (max_p - price) / (max_p - min_p) * price_h

        for i in range(7):
            yy = plot_y + i * price_h / 6
            price = max_p - i * (max_p - min_p) / 6
            self.line(plot_x, yy, plot_x + plot_w, yy, GRID)
            self.text(plot_x + plot_w + 8, yy - 5, f"{price:.3f}", WHITE, FONT_TINY)

        for i in range(10):
            xx = plot_x + i * plot_w / 9
            self.line(xx, plot_y, xx, vol_y + vol_h, "#101010")

        candle_w = max(2, min(6, plot_w / len(data) * 0.62))
        max_vol = float(data["tick_volume"].max()) if "tick_volume" in data else 0

        for i, (_, r) in enumerate(data.iterrows()):
            xx = xp(i)
            o = float(r["open"])
            hi = float(r["high"])
            lo = float(r["low"])
            cl = float(r["close"])

            color = GREEN if cl >= o else RED

            self.line(xx, yp(lo), xx, yp(hi), color)

            y1 = yp(max(o, cl))
            y2 = yp(min(o, cl))

            if abs(y2 - y1) < 2:
                y2 += 2

            self.canvas.create_rectangle(
                xx - candle_w / 2,
                y1,
                xx + candle_w / 2,
                y2,
                fill=color,
                outline=color,
            )

            if max_vol > 0:
                vh = float(r["tick_volume"]) / max_vol * vol_h
                self.canvas.create_rectangle(
                    xx - candle_w / 2,
                    vol_y + vol_h - vh,
                    xx + candle_w / 2,
                    vol_y + vol_h,
                    fill=color,
                    outline=color,
                )

        last_close = float(last["close"])
        yy = yp(last_close)

        self.line(plot_x, yy, plot_x + plot_w, yy, ORANGE, dash=(2, 2))
        self.canvas.create_rectangle(
            plot_x + plot_w + 6,
            yy - 10,
            x + w - 8,
            yy + 10,
            fill=GREEN if chg >= 0 else RED,
            outline=GREEN if chg >= 0 else RED,
        )
        self.text(plot_x + plot_w + 10, yy - 6, self._fmt(last_close), BG, FONT_SMALL)
        self.text(plot_x, vol_y - 16, f"VOLUME {self._fmt(last['tick_volume'], 0)}", GREEN, FONT_HEAD)

    # ========================================================
    # FOOTER / EVENTS
    # ========================================================

    def draw_footer(self, x, y, w, h):
        self.rect(x, y, w, h, fill=BG, outline=ORANGE)
        self.text(x + 12, y + 8, "DATA SOURCE:", GRAY, FONT_SMALL)
        self.text(x + 115, y + 8, "1_Pipeline/Market/ohcl", WHITE, FONT_SMALL)
        self.text(x + 340, y + 8, "ACTIVE CHARTS:", GRAY, FONT_SMALL)
        self.text(x + 455, y + 8, str(len(self.charts)), GREEN, FONT_SMALL)
        self.text(x + w - 180, y + 8, "STATUS: LIVE", GREEN, FONT_SMALL)

    def on_click(self, event):
        for sym, x, y, w, h in self.symbol_hitboxes:
            if x <= event.x <= x + w and y <= event.y <= y + h:
                self.selected_symbol = sym
                self.redraw()
                return

        for tf, x, y, w, h in self.tf_hitboxes:
            if x <= event.x <= x + w and y <= event.y <= y + h:
                self.selected_timeframe = tf
                self.redraw()
                return

        for idx, x, y, w, h in self.chart_close_hitboxes:
            if x <= event.x <= x + w and y <= event.y <= y + h:
                if 0 <= idx < len(self.charts):
                    self.charts.pop(idx)
                    if not self.charts:
                        self.charts.append((self.selected_symbol, self.selected_timeframe))
                    self.redraw()
                return

        for action, x, y, w, h in self.button_hitboxes:
            if x <= event.x <= x + w and y <= event.y <= y + h:
                self.handle_action(action)
                return

    def handle_action(self, action):
        if action == "refresh":
            self.refresh_data()
            return

        if action == "add_chart":
            chart = (self.selected_symbol, self.selected_timeframe)
            if chart not in self.charts:
                self.charts.append(chart)
            if len(self.charts) > 4:
                self.charts = self.charts[-4:]
            self.redraw()
            return

        if action == "layout_1":
            self.charts = [(self.selected_symbol, self.selected_timeframe)]
            self.redraw()
            return

        if action == "layout_2":
            base = [(self.selected_symbol, self.selected_timeframe)]
            for s in self.symbols:
                c = (s, self.selected_timeframe)
                if c not in base:
                    base.append(c)
                if len(base) == 2:
                    break
            self.charts = base
            self.redraw()
            return

        if action == "layout_4":
            base = [(self.selected_symbol, self.selected_timeframe)]
            for s in self.symbols:
                c = (s, self.selected_timeframe)
                if c not in base:
                    base.append(c)
                if len(base) == 4:
                    break
            self.charts = base
            self.redraw()
            return

        if action == "clear_charts":
            self.charts = [(self.selected_symbol, self.selected_timeframe)]
            self.redraw()
            return

    def tick_clock(self):
        self.redraw()
        self.after(1000, self.tick_clock)

    # ========================================================
    # HELPERS
    # ========================================================

    def snapshot(self, symbol):
        path = self.data_root / self.selected_timeframe / f"{symbol}.parquet"
        try:
            df = pd.read_parquet(path, columns=["close"])
            if len(df) < 2:
                return "-", "-", "-"
            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-2])
            chg = last - prev
            pct = chg / prev * 100 if prev else 0
            return f"{last:.5g}", f"{chg:+.4g}", f"{pct:+.2f}%"
        except Exception:
            return "-", "-", "-"

    @staticmethod
    def safe_float(x):
        try:
            return float(str(x).replace("%", ""))
        except Exception:
            return 0.0

    @staticmethod
    def _fmt(v, digits=3):
        try:
            if pd.isna(v):
                return "-"
            return f"{float(v):,.{digits}f}"
        except Exception:
            return str(v)


def build_panel(parent, repository=None, **kwargs):
    return MarketMultiOHLCChartDashboard(parent, repository=repository, **kwargs)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("QUANT TERMINAL - MARKET MULTI OHLC CHARTS")
    root.configure(bg=BG)
    root.geometry("1700x1000")
    root.minsize(1350, 820)
    build_panel(root).pack(fill="both", expand=True)
    root.mainloop()
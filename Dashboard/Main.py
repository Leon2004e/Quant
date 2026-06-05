# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: main_dashboard
# script_name: Main Dashboard
# owner: Leon
# status: active
# layer: Dashboard
# domain: Dashboard
# asset_type: Dashboard
# purpose: Freeform main dashboard with Bloomberg-terminal-inspired UI, Building Block cards, and draggable/resizable dashboard panels on a canvas workspace.
# inputs:
#   - Dashboard/Building_Blocks
# outputs:
#   - Dashboard UI
#   - Dashboard/layouts/main_dashboard_layout.json
# dependencies:
#   - tkinter
#   - pathlib
#   - importlib
#   - json
# schedule: manual
# version: v2.1.1_dynamic_import_fix
# last_reviewed: 2026-06-01
# ============================================================

from __future__ import annotations

import json
import sys
import traceback
import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path


APP_TITLE = "QUANT TERMINAL"
LAYOUT_FILE_NAME = "main_dashboard_layout.json"

SIDEBAR_WIDTH = 360
SIDEBAR_MIN_WIDTH = 260

DEFAULT_PANEL_W = 760
DEFAULT_PANEL_H = 520
MIN_PANEL_W = 360
MIN_PANEL_H = 260

WORKSPACE_W = 3200
WORKSPACE_H = 2200


COLORS = {
    # Bloomberg-terminal-inspired theme. No logos/trademarks, only terminal-style UI.
    "bg": "#000000",
    "workspace": "#050505",
    "workspace_grid": "#161616",

    "panel": "#0A0A0A",
    "panel_2": "#111111",
    "panel_3": "#151515",

    "slot": "#0A0A0A",
    "slot_border": "#2A2A2A",
    "slot_empty": "#050505",

    "text": "#FFFFFF",
    "muted": "#B7B7B7",
    "faint": "#6F6F6F",

    "accent": "#FF9900",
    "accent_2": "#FFD400",
    "accent_soft": "#332000",

    "success": "#00FF66",
    "success_soft": "#062D16",

    "warning": "#FFD400",
    "warning_soft": "#332A00",

    "danger": "#FF4444",
    "danger_soft": "#300A0A",

    "info": "#00AEEF",
    "info_soft": "#061B28",

    "button": "#111111",
    "button_hover": "#332000",
    "button_border": "#2A2A2A",

    "shadow": "#000000",
}


FONT_TITLE = ("Consolas", 13, "bold")
FONT_HEAD = ("Consolas", 10, "bold")
FONT_MAIN = ("Consolas", 9)
FONT_SMALL = ("Consolas", 8)
FONT_TINY = ("Consolas", 7)
FONT_MONO = ("Consolas", 9)

TERMINAL_NAV_ITEMS = [
    ("OVRV", "Overview"),
    ("BLKS", "Blocks"),
    ("LAYT", "Layout"),
    ("REG", "Registry"),
    ("SYS", "System"),
]


def find_quant_root(start: Path) -> Path:
    current = start.resolve()
    if current.is_file():
        current = current.parent

    for p in [current] + list(current.parents):
        if (p / "Dashboard").is_dir() and (p / "Data_Center").is_dir():
            return p

    raise RuntimeError("QUANT root not found. Expected Dashboard/ and Data_Center/.")


def display_name(folder_name: str) -> str:
    return folder_name.replace("_", " ")


class BuildingBlockRegistry:
    def __init__(self, building_blocks_dir: Path):
        self.building_blocks_dir = building_blocks_dir
        self.blocks: dict[str, dict] = {}

    def scan(self) -> dict[str, dict]:
        self.blocks = {}

        if not self.building_blocks_dir.exists():
            return self.blocks

        for folder in self.building_blocks_dir.iterdir():
            if not folder.is_dir():
                continue
            if folder.name == "__pycache__":
                continue

            code_file = folder / "code.py"
            if code_file.exists():
                self.blocks[folder.name] = {
                    "name": folder.name,
                    "display": display_name(folder.name),
                    "folder": folder,
                    "code_file": code_file,
                }

        self.blocks = dict(sorted(self.blocks.items(), key=lambda x: x[0].lower()))
        return self.blocks

    def load_module(self, block_name: str):
        if block_name not in self.blocks:
            raise ValueError(f"Unknown Building Block: {block_name}")

        code_file = self.blocks[block_name]["code_file"]
        module_name = f"quant_building_block_{block_name}"

        spec = importlib.util.spec_from_file_location(module_name, code_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec: {code_file}")

        module = importlib.util.module_from_spec(spec)

        # IMPORTANT:
        # Register module before exec_module().
        # dataclasses, typing and some libraries inspect sys.modules during import.
        # Without this, dynamically loaded Building Blocks can fail in Main.py
        # even when they run standalone.
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            # Prevent broken partial modules from staying registered after failed import.
            sys.modules.pop(module_name, None)
            raise

        return module


class DashboardPanel(tk.Frame):
    def __init__(
        self,
        parent_canvas: tk.Canvas,
        app,
        block_name: str,
        x: int,
        y: int,
        width: int = DEFAULT_PANEL_W,
        height: int = DEFAULT_PANEL_H,
    ):
        super().__init__(
            parent_canvas,
            bg=COLORS["panel"],
            highlightbackground=COLORS["slot_border"],
            highlightthickness=1,
            bd=0,
        )
        self.canvas = parent_canvas
        self.app = app
        self.block_name = block_name
        self.panel_x = int(x)
        self.panel_y = int(y)
        self.panel_w = int(width)
        self.panel_h = int(height)
        self.shadow_id: int | None = None
        self.window_id: int | None = None
        self.drag_start: tuple[int, int] | None = None
        self.resize_start: tuple[int, int, int, int] | None = None
        self.content_frame = None

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_shell()
        self.shadow_id = self.canvas.create_rectangle(
            self.panel_x + 8,
            self.panel_y + 8,
            self.panel_x + self.panel_w + 8,
            self.panel_y + self.panel_h + 8,
            fill=COLORS["shadow"],
            outline="",
            tags=("panel_shadow",),
        )
        self.window_id = self.canvas.create_window(
            self.panel_x,
            self.panel_y,
            window=self,
            anchor="nw",
            width=self.panel_w,
            height=self.panel_h,
            tags=("dashboard_panel",),
        )
        self.canvas.tag_lower(self.shadow_id, self.window_id)
        self.load_block()

    def _build_shell(self):
        self.header = tk.Frame(self, bg=COLORS["panel_2"], height=26)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_columnconfigure(0, weight=1)

        self.title_label = tk.Label(
            self.header,
            text=display_name(self.block_name).upper(),
            bg=COLORS["panel_2"],
            fg=COLORS["text"],
            font=FONT_SMALL,
            padx=8,
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        self.reload_btn = self._header_button("↻", self.load_block, COLORS["accent"], 4)
        self.reload_btn.grid(row=0, column=1, padx=(0, 4), pady=5)

        self.close_btn = self._header_button("×", self.close, COLORS["danger"], 4)
        self.close_btn.grid(row=0, column=2, padx=(0, 5), pady=5)

        self.body = tk.Frame(self, bg=COLORS["panel"], highlightbackground=COLORS["slot_border"], highlightthickness=1)
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.grid_rowconfigure(0, weight=1)
        self.body.grid_columnconfigure(0, weight=1)

        self.resize_grip = tk.Label(
            self,
            text="◢",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_HEAD,
            cursor="size_nw_se",
        )
        self.resize_grip.place(relx=1.0, rely=1.0, anchor="se", x=-4, y=-4)

        for widget in (self.header, self.title_label):
            widget.bind("<ButtonPress-1>", self.start_move)
            widget.bind("<B1-Motion>", self.do_move)
            widget.bind("<ButtonRelease-1>", self.end_move)

        self.resize_grip.bind("<ButtonPress-1>", self.start_resize)
        self.resize_grip.bind("<B1-Motion>", self.do_resize)
        self.resize_grip.bind("<ButtonRelease-1>", self.end_resize)

    def _header_button(self, text: str, command, fg: str, width: int):
        return tk.Button(
            self.header,
            text=text,
            command=command,
            bg=COLORS["button"],
            fg=fg,
            activebackground=COLORS["button_hover"],
            activeforeground=fg,
            relief="flat",
            bd=0,
            width=width,
            cursor="hand2",
            font=FONT_SMALL,
        )

    def start_move(self, event):
        self.lift_panel()
        self.drag_start = (event.x_root, event.y_root)

    def do_move(self, event):
        if self.drag_start is None:
            return

        dx = event.x_root - self.drag_start[0]
        dy = event.y_root - self.drag_start[1]

        self.panel_x = max(0, self.panel_x + dx)
        self.panel_y = max(0, self.panel_y + dy)

        if self.window_id is not None:
            self.canvas.coords(self.window_id, self.panel_x, self.panel_y)
        if self.shadow_id is not None:
            self.canvas.coords(
                self.shadow_id,
                self.panel_x + 8,
                self.panel_y + 8,
                self.panel_x + self.panel_w + 8,
                self.panel_y + self.panel_h + 8,
            )

        self.drag_start = (event.x_root, event.y_root)
        self.app.update_workspace_scrollregion()

    def end_move(self, _event=None):
        self.drag_start = None
        self.app.update_status(f"Moved: {display_name(self.block_name)}")

    def start_resize(self, event):
        self.lift_panel()
        self.resize_start = (event.x_root, event.y_root, self.panel_w, self.panel_h)

    def do_resize(self, event):
        if self.resize_start is None:
            return

        x0, y0, w0, h0 = self.resize_start
        new_w = max(MIN_PANEL_W, w0 + (event.x_root - x0))
        new_h = max(MIN_PANEL_H, h0 + (event.y_root - y0))

        self.panel_w = int(new_w)
        self.panel_h = int(new_h)

        if self.window_id is not None:
            self.canvas.itemconfigure(self.window_id, width=self.panel_w, height=self.panel_h)
        if self.shadow_id is not None:
            self.canvas.coords(
                self.shadow_id,
                self.panel_x + 8,
                self.panel_y + 8,
                self.panel_x + self.panel_w + 8,
                self.panel_y + self.panel_h + 8,
            )

        self.app.update_workspace_scrollregion()

    def end_resize(self, _event=None):
        self.resize_start = None
        self.app.update_status(f"Resized: {display_name(self.block_name)}")

    def lift_panel(self):
        if self.shadow_id is not None:
            self.canvas.tag_raise(self.shadow_id)
        if self.window_id is not None:
            self.canvas.tag_raise(self.window_id)

    def load_block(self):
        for child in self.body.winfo_children():
            child.destroy()

        try:
            module = self.app.registry.load_module(self.block_name)

            panel = None
            if hasattr(module, "build_panel"):
                panel = module.build_panel(self.body)
            elif hasattr(module, "create_panel"):
                panel = module.create_panel(self.body)
            elif hasattr(module, "DashboardPanel"):
                panel = module.DashboardPanel(self.body)
            else:
                raise AttributeError(
                    "Building Block needs build_panel(parent), "
                    "create_panel(parent), or class DashboardPanel(tk.Frame)."
                )

            if panel is None:
                raise RuntimeError("Building Block returned None instead of a tk widget.")

            try:
                panel.grid(row=0, column=0, sticky="nsew")
            except Exception:
                panel.pack(fill="both", expand=True)

            self.content_frame = panel
            self.app.update_status(f"Loaded: {display_name(self.block_name)}")

        except Exception:
            error_text = traceback.format_exc()
            self.show_error(error_text)
            self.app.update_status(f"Load error: {display_name(self.block_name)}")

    def show_error(self, error_text: str):
        body = tk.Frame(self.body, bg=COLORS["danger_soft"])
        body.grid(row=0, column=0, sticky="nsew")
        body.grid_rowconfigure(1, weight=1)
        body.grid_columnconfigure(0, weight=1)

        tk.Label(
            body,
            text="BUILDING BLOCK LOAD ERROR",
            bg=COLORS["danger_soft"],
            fg=COLORS["danger"],
            font=FONT_HEAD,
        ).grid(row=0, column=0, sticky="nw", padx=12, pady=(12, 4))

        box = tk.Text(
            body,
            bg=COLORS["danger_soft"],
            fg="#7F1D1D",
            insertbackground=COLORS["text"],
            relief="flat",
            wrap="word",
            font=FONT_MONO,
        )
        box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        box.insert("1.0", error_text)
        box.config(state="disabled")

    def close(self):
        if self.shadow_id is not None:
            self.canvas.delete(self.shadow_id)
        if self.window_id is not None:
            self.canvas.delete(self.window_id)
        try:
            self.destroy()
        except Exception:
            pass
        self.app.panels = [p for p in self.app.panels if p is not self]
        self.app.update_status(f"Closed: {display_name(self.block_name)}")

    def to_layout_dict(self) -> dict:
        return {
            "block_name": self.block_name,
            "x": int(self.panel_x),
            "y": int(self.panel_y),
            "width": int(self.panel_w),
            "height": int(self.panel_h),
        }


class MainDashboard(tk.Tk):
    def __init__(self):
        super().__init__()

        self.quant_root = find_quant_root(Path(__file__))
        self.dashboard_dir = self.quant_root / "Dashboard"
        self.building_blocks_dir = self.dashboard_dir / "Building_Blocks"
        self.layouts_dir = self.dashboard_dir / "layouts"
        self.layout_file = self.layouts_dir / LAYOUT_FILE_NAME

        self.registry = BuildingBlockRegistry(self.building_blocks_dir)
        self.registry.scan()

        self.selected_block_name = tk.StringVar(value="")
        self.block_search_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready.")

        self.panels: list[DashboardPanel] = []
        self.dragging_block_name: str | None = None
        self.drag_label: tk.Label | None = None
        self.last_drop_xy = (80, 80)
        self.next_panel_offset = 0

        self.title(APP_TITLE)
        self.geometry("1850x1050")
        self.minsize(1350, 780)
        self.configure(bg=COLORS["bg"])

        self._setup_style()
        self._build_ui()

        self.bind("<Motion>", self._on_drag_motion, add="+")
        self.bind("<ButtonRelease-1>", self._on_global_release, add="+")

        self._refresh_sidebar()
        self.update_status(f"Blocks loaded: {len(self.registry.blocks)}")

    def _setup_style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            ".",
            background=COLORS["bg"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["panel"],
            font=FONT_MAIN,
            bordercolor=COLORS["slot_border"],
            lightcolor=COLORS["slot_border"],
            darkcolor=COLORS["slot_border"],
        )
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONT_MAIN)
        style.configure(
            "Treeview",
            background="#000000",
            foreground=COLORS["text"],
            fieldbackground="#000000",
            rowheight=20,
            borderwidth=1,
            font=FONT_SMALL,
        )
        style.configure(
            "Treeview.Heading",
            background="#111111",
            foreground=COLORS["accent"],
            font=("Consolas", 8, "bold"),
            borderwidth=1,
        )
        style.map(
            "Treeview",
            background=[("selected", COLORS["accent_soft"])],
            foreground=[("selected", COLORS["accent_2"])],
        )

    def update_status(self, text: str):
        self.status_var.set(text)

    def _button(self, parent, text: str, command, fg=None):
        outer = tk.Frame(parent, bg=COLORS["button_border"])
        btn = tk.Button(
            outer,
            text=text,
            bg=COLORS["button"],
            fg=fg or COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=fg or COLORS["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=7,
            command=command,
            cursor="hand2",
            font=FONT_MAIN,
        )
        btn.pack(fill="both", expand=True, padx=1, pady=1)
        return outer

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        self.topbar = tk.Frame(
            self,
            bg=COLORS["panel"],
            height=42,
            highlightbackground=COLORS["slot_border"],
            highlightthickness=1,
        )
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(1, weight=1)

        brand = tk.Frame(self.topbar, bg=COLORS["panel"])
        brand.grid(row=0, column=0, sticky="w", padx=10, pady=6)

        tk.Label(
            brand,
            text="QUANT TERMINAL",
            bg=COLORS["panel"],
            fg=COLORS["accent"],
            font=FONT_TITLE,
        ).pack(side="left")

        tk.Label(
            brand,
            text="WORKSPACE",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=FONT_HEAD,
        ).pack(side="left", padx=(8, 0))

        tk.Label(
            self.topbar,
            text="BUILDING BLOCK WORKSPACE | DRAG PANELS | SAVE/LOAD LAYOUT | TERMINAL STYLE",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_MAIN,
            anchor="w",
        ).grid(row=0, column=1, sticky="ew", padx=8)

        actions = tk.Frame(self.topbar, bg=COLORS["panel"])
        actions.grid(row=0, column=2, sticky="e", padx=8, pady=6)

        self._button(actions, "Refresh Blocks", self.refresh_blocks, fg=COLORS["accent"]).pack(side="left", padx=4)
        self._button(actions, "Clear Workspace", self.clear_workspace, fg=COLORS["danger"]).pack(side="left", padx=4)
        self._button(actions, "Load Layout", self.load_layout).pack(side="left", padx=4)
        self._button(actions, "Save Layout", self.save_layout, fg=COLORS["success"]).pack(side="left", padx=4)

        self.main_pane = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=COLORS["bg"],
            sashwidth=8,
            bd=0,
            showhandle=False,
        )
        self.main_pane.grid(row=1, column=0, sticky="nsew")

        self.sidebar = tk.Frame(
            self.main_pane,
            bg=COLORS["panel"],
            width=SIDEBAR_WIDTH,
            highlightbackground=COLORS["slot_border"],
            highlightthickness=1,
        )
        self.sidebar.grid_propagate(False)

        self.workspace_wrap = tk.Frame(self.main_pane, bg=COLORS["workspace"])
        self.workspace_wrap.grid_rowconfigure(0, weight=1)
        self.workspace_wrap.grid_columnconfigure(0, weight=1)

        self.main_pane.add(self.sidebar, minsize=SIDEBAR_MIN_WIDTH)
        self.main_pane.add(self.workspace_wrap, minsize=720)

        self._build_sidebar()
        self._build_workspace()

        self.statusbar = tk.Frame(
            self,
            bg=COLORS["panel"],
            height=28,
            highlightbackground=COLORS["slot_border"],
            highlightthickness=1,
        )
        self.statusbar.grid(row=2, column=0, sticky="ew")
        self.statusbar.grid_columnconfigure(0, weight=1)

        tk.Label(
            self.statusbar,
            textvariable=self.status_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
            anchor="w",
            padx=12,
        ).grid(row=0, column=0, sticky="ew")

    def _build_sidebar(self):
        head = tk.Frame(self.sidebar, bg=COLORS["panel"])
        head.pack(fill="x", padx=12, pady=(14, 8))

        tk.Label(
            head,
            text="BUILDING BLOCKS",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=FONT_HEAD,
        ).pack(anchor="w")

        self.block_count_label = tk.Label(
            head,
            text="0 blocks",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
        )
        self.block_count_label.pack(anchor="w", pady=(2, 0))

        search_wrap = tk.Frame(self.sidebar, bg=COLORS["button_border"])
        search_wrap.pack(fill="x", padx=12, pady=(0, 10))

        search_entry = tk.Entry(
            search_wrap,
            textvariable=self.block_search_var,
            bg=COLORS["button"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="solid",
            bd=1,
            font=FONT_MAIN,
        )
        search_entry.pack(fill="x", padx=1, pady=1, ipady=7)
        search_entry.bind("<KeyRelease>", lambda _e: self._refresh_sidebar())

        self.block_canvas = tk.Canvas(self.sidebar, bg=COLORS["panel"], highlightthickness=0, bd=0)
        self.block_scroll = tk.Scrollbar(self.sidebar, orient="vertical", command=self.block_canvas.yview)
        self.block_list_frame = tk.Frame(self.block_canvas, bg=COLORS["panel"])

        self.block_list_frame.bind(
            "<Configure>",
            lambda _e: self.block_canvas.configure(scrollregion=self.block_canvas.bbox("all")),
        )

        self.block_canvas_window = self.block_canvas.create_window((0, 0), window=self.block_list_frame, anchor="nw")
        self.block_canvas.configure(yscrollcommand=self.block_scroll.set)

        self.block_canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=(0, 10))
        self.block_scroll.pack(side="right", fill="y", padx=(0, 6), pady=(0, 10))

        self.block_canvas.bind("<Configure>", self._resize_block_canvas)
        self.block_canvas.bind("<MouseWheel>", self._sidebar_mousewheel)

    def _build_workspace(self):
        self.workspace_canvas = tk.Canvas(
            self.workspace_wrap,
            bg=COLORS["workspace"],
            highlightthickness=0,
            bd=0,
            scrollregion=(0, 0, WORKSPACE_W, WORKSPACE_H),
        )
        self.workspace_y = tk.Scrollbar(self.workspace_wrap, orient="vertical", command=self.workspace_canvas.yview)
        self.workspace_x = tk.Scrollbar(self.workspace_wrap, orient="horizontal", command=self.workspace_canvas.xview)

        self.workspace_canvas.configure(
            yscrollcommand=self.workspace_y.set,
            xscrollcommand=self.workspace_x.set,
        )

        self.workspace_canvas.grid(row=0, column=0, sticky="nsew")
        self.workspace_y.grid(row=0, column=1, sticky="ns")
        self.workspace_x.grid(row=1, column=0, sticky="ew")

        self.workspace_canvas.bind("<ButtonRelease-1>", self._workspace_release)
        self.workspace_canvas.bind("<MouseWheel>", self._workspace_mousewheel)
        self.workspace_canvas.bind("<Shift-MouseWheel>", self._workspace_shift_mousewheel)

        self._draw_workspace_hint()

    def _draw_workspace_grid(self):
        self.workspace_canvas.delete("workspace_grid")
        step = 40
        for x in range(0, WORKSPACE_W + 1, step):
            self.workspace_canvas.create_line(
                x,
                0,
                x,
                WORKSPACE_H,
                fill=COLORS["workspace_grid"],
                width=1,
                tags=("workspace_grid",),
            )
        for y in range(0, WORKSPACE_H + 1, step):
            self.workspace_canvas.create_line(
                0,
                y,
                WORKSPACE_W,
                y,
                fill=COLORS["workspace_grid"],
                width=1,
                tags=("workspace_grid",),
            )
        self.workspace_canvas.tag_lower("workspace_grid")

    def _draw_workspace_hint(self):
        self.workspace_canvas.delete("workspace_hint")
        self._draw_workspace_grid()
        self._draw_workspace_grid()
        if self.panels:
            return

        self.workspace_canvas.create_rectangle(
            62,
            52,
            760,
            150,
            fill=COLORS["panel"],
            outline=COLORS["slot_border"],
            width=1,
            tags=("workspace_hint",),
        )
        self.workspace_canvas.create_text(
            80,
            70,
            text="QUANT TERMINAL WORKSPACE",
            anchor="nw",
            fill=COLORS["text"],
            font=FONT_TITLE,
            tags=("workspace_hint",),
        )
        self.workspace_canvas.create_text(
            80,
            108,
            text="DRAG BUILDING BLOCKS FROM LEFT SIDEBAR. MOVE BY HEADER. RESIZE WITH BOTTOM-RIGHT GRIP.",
            anchor="nw",
            fill=COLORS["muted"],
            font=("Segoe UI", 10),
            tags=("workspace_hint",),
        )

    def _resize_block_canvas(self, event):
        self.block_canvas.itemconfigure(self.block_canvas_window, width=event.width)

    def _sidebar_mousewheel(self, event):
        try:
            self.block_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _workspace_mousewheel(self, event):
        try:
            self.workspace_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _workspace_shift_mousewheel(self, event):
        try:
            self.workspace_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _card_accent(self, block_name: str):
        lower = block_name.lower()
        if "registry" in lower:
            return COLORS["accent"], COLORS["accent_soft"]
        if "catalog" in lower:
            return COLORS["success"], COLORS["success_soft"]
        if "pipeline" in lower:
            return COLORS["warning"], COLORS["warning_soft"]
        return COLORS["accent"], COLORS["accent_soft"]

    def _filtered_blocks(self) -> dict[str, dict]:
        q = self.block_search_var.get().strip().lower()
        return {
            name: info
            for name, info in self.registry.blocks.items()
            if not q or q in name.lower() or q in info["display"].lower()
        }

    def _refresh_sidebar(self):
        for child in self.block_list_frame.winfo_children():
            child.destroy()

        blocks = self._filtered_blocks()
        self.block_count_label.config(text=f"{len(blocks)} / {len(self.registry.blocks)} blocks")

        if not blocks:
            tk.Label(
                self.block_list_frame,
                text="NO BUILDING BLOCKS FOUND.",
                bg=COLORS["panel"],
                fg=COLORS["muted"],
                font=FONT_MAIN,
                wraplength=280,
            ).pack(anchor="w", pady=10, padx=6)
            return

        if not self.selected_block_name.get() or self.selected_block_name.get() not in self.registry.blocks:
            self.selected_block_name.set(next(iter(self.registry.blocks.keys()), ""))

        for block_name, info in blocks.items():
            self._create_sidebar_card(block_name, info)

    def _create_sidebar_card(self, block_name: str, info: dict):
        selected = self.selected_block_name.get() == block_name
        accent, soft = self._card_accent(block_name)

        outer = tk.Frame(
            self.block_list_frame,
            bg=accent if selected else COLORS["button_border"],
        )
        outer.pack(fill="x", padx=4, pady=6)

        card = tk.Frame(
            outer,
            bg=soft if selected else COLORS["button"],
            cursor="hand2",
        )
        card.pack(fill="x", padx=1, pady=1)

        top = tk.Frame(card, bg=card["bg"])
        top.pack(fill="x", padx=12, pady=(11, 4))

        tk.Label(
            top,
            text="●",
            bg=card["bg"],
            fg=accent,
            font=FONT_HEAD,
        ).pack(side="left")

        tk.Label(
            top,
            text=info["display"],
            bg=card["bg"],
            fg=COLORS["text"],
            font=FONT_HEAD,
            anchor="w",
        ).pack(side="left", padx=(8, 0), fill="x", expand=True)

        path_text = str(info["folder"].relative_to(self.quant_root)).replace("\\", "/")
        tk.Label(
            card,
            text=path_text,
            bg=card["bg"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
            anchor="w",
            wraplength=285,
            justify="left",
        ).pack(fill="x", padx=12, pady=(0, 8))

        footer = tk.Frame(card, bg=card["bg"])
        footer.pack(fill="x", padx=12, pady=(0, 11))

        tk.Label(
            footer,
            text="DRAG TO WORKSPACE",
            bg=card["bg"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
        ).pack(side="left")

        tk.Button(
            footer,
            text="ADD",
            bg=COLORS["panel"],
            fg=accent,
            activebackground=COLORS["button_hover"],
            activeforeground=accent,
            relief="solid",
            bd=1,
            padx=8,
            pady=3,
            cursor="hand2",
            command=lambda b=block_name: self.add_panel_auto(b),
            font=("Segoe UI", 8, "bold"),
        ).pack(side="right")

        for w in (outer, card, top, footer):
            w.bind("<ButtonPress-1>", lambda e, b=block_name: self.start_drag(b, e))
            w.bind("<Double-1>", lambda _e, b=block_name: self.add_panel_auto(b))

    def start_drag(self, block_name: str, event):
        self.dragging_block_name = block_name
        self.selected_block_name.set(block_name)
        self._refresh_sidebar()

        if self.drag_label is not None:
            self.drag_label.destroy()

        self.drag_label = tk.Label(
            self,
            text=f"  {display_name(block_name)}  ",
            bg=COLORS["accent"],
            fg="#000000",
            font=FONT_HEAD,
            padx=12,
            pady=7,
        )
        self.drag_label.place(
            x=event.x_root - self.winfo_rootx() + 12,
            y=event.y_root - self.winfo_rooty() + 12,
        )
        self.update_status(f"Dragging: {display_name(block_name)}")

    def _on_drag_motion(self, event):
        if self.drag_label is None:
            return

        self.drag_label.place(
            x=event.x_root - self.winfo_rootx() + 12,
            y=event.y_root - self.winfo_rooty() + 12,
        )

    def _on_global_release(self, event):
        if not self.dragging_block_name:
            return

        if self._pointer_inside_workspace(event.x_root, event.y_root):
            self.drop_panel_at_pointer(event.x_root, event.y_root)
        else:
            self.cancel_drag()

    def _workspace_release(self, event):
        if not self.dragging_block_name:
            return
        self.drop_panel_at_canvas_xy(event.x, event.y)

    def _pointer_inside_workspace(self, x_root: int, y_root: int) -> bool:
        try:
            x1 = self.workspace_canvas.winfo_rootx()
            y1 = self.workspace_canvas.winfo_rooty()
            x2 = x1 + self.workspace_canvas.winfo_width()
            y2 = y1 + self.workspace_canvas.winfo_height()
            return x1 <= x_root <= x2 and y1 <= y_root <= y2
        except Exception:
            return False

    def drop_panel_at_pointer(self, x_root: int, y_root: int):
        canvas_x = self.workspace_canvas.canvasx(x_root - self.workspace_canvas.winfo_rootx())
        canvas_y = self.workspace_canvas.canvasy(y_root - self.workspace_canvas.winfo_rooty())
        self.drop_panel_at_canvas_xy(canvas_x, canvas_y)

    def drop_panel_at_canvas_xy(self, x: int | float, y: int | float):
        if not self.dragging_block_name:
            return

        block_name = self.dragging_block_name
        self.cancel_drag(clear_status=False)
        self.add_panel(block_name, int(x), int(y))

    def cancel_drag(self, clear_status: bool = True):
        self.dragging_block_name = None
        if self.drag_label is not None:
            self.drag_label.destroy()
            self.drag_label = None
        if clear_status:
            self.update_status("Drag cancelled.")

    def add_panel_auto(self, block_name: str):
        offset = self.next_panel_offset * 34
        self.next_panel_offset = (self.next_panel_offset + 1) % 12
        x = 80 + offset
        y = 90 + offset
        self.add_panel(block_name, x, y)

    def add_panel(self, block_name: str, x: int, y: int, width: int = DEFAULT_PANEL_W, height: int = DEFAULT_PANEL_H):
        self.workspace_canvas.delete("workspace_hint")
        x = max(0, int(x))
        y = max(0, int(y))

        panel = DashboardPanel(
            self.workspace_canvas,
            app=self,
            block_name=block_name,
            x=x,
            y=y,
            width=width,
            height=height,
        )
        self.panels.append(panel)
        self.update_workspace_scrollregion()
        self.update_status(f"Added panel: {display_name(block_name)}")

    def update_workspace_scrollregion(self):
        max_x = WORKSPACE_W
        max_y = WORKSPACE_H
        for panel in self.panels:
            max_x = max(max_x, panel.panel_x + panel.panel_w + 200)
            max_y = max(max_y, panel.panel_y + panel.panel_h + 200)
        self.workspace_canvas.configure(scrollregion=(0, 0, max_x, max_y))

    def clear_workspace(self):
        if not self.panels:
            return
        if not messagebox.askyesno("Clear Workspace", "Alle Panels entfernen?"):
            return
        for panel in list(self.panels):
            panel.close()
        self.panels.clear()
        self._draw_workspace_hint()
        self.update_status("Workspace cleared.")

    def refresh_blocks(self):
        current = self.selected_block_name.get()
        self.registry.scan()
        self._refresh_sidebar()

        if current in self.registry.blocks:
            self.selected_block_name.set(current)

        self.update_status(f"Blocks refreshed: {len(self.registry.blocks)}")

    def save_layout(self):
        self.layouts_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "v2.0.0",
            "mode": "freeform",
            "panels": [panel.to_layout_dict() for panel in self.panels],
        }

        with self.layout_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self.update_status(f"Layout saved: {self.layout_file}")
        messagebox.showinfo("Layout", f"Layout saved:\n{self.layout_file}")

    def load_layout(self):
        if not self.layout_file.exists():
            messagebox.showwarning("Layout", f"No layout file found:\n{self.layout_file}")
            return

        try:
            with self.layout_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for panel in list(self.panels):
                panel.close()
            self.panels.clear()
            self.workspace_canvas.delete("all")

            panels_data = data.get("panels", [])

            # Backward compatibility for old page/slot layouts.
            if not panels_data and "pages" in data:
                x, y = 80, 90
                for page in data.get("pages", []):
                    for block_name in page.get("slots", []):
                        if block_name:
                            panels_data.append(
                                {
                                    "block_name": block_name,
                                    "x": x,
                                    "y": y,
                                    "width": DEFAULT_PANEL_W,
                                    "height": DEFAULT_PANEL_H,
                                }
                            )
                            x += 60
                            y += 60

            for item in panels_data:
                block_name = item.get("block_name")
                if not block_name:
                    continue
                if block_name not in self.registry.blocks:
                    continue

                self.add_panel(
                    block_name=block_name,
                    x=int(item.get("x", 80)),
                    y=int(item.get("y", 90)),
                    width=int(item.get("width", DEFAULT_PANEL_W)),
                    height=int(item.get("height", DEFAULT_PANEL_H)),
                )

            if not self.panels:
                self._draw_workspace_hint()

            self.update_status(f"Layout loaded: {self.layout_file}")

        except Exception as e:
            messagebox.showerror("Layout Load Error", str(e))
            self.update_status("Layout load error.")


if __name__ == "__main__":
    app = MainDashboard()
    app.mainloop()

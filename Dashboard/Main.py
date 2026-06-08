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
# version: v2.6.3_final_clean_bloomberg_nav
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

SIDEBAR_WIDTH = 380
SIDEBAR_MIN_WIDTH = 320

DEFAULT_PANEL_W = 760
DEFAULT_PANEL_H = 520
MIN_PANEL_W = 360
MIN_PANEL_H = 260

WORKSPACE_W = 3200
WORKSPACE_H = 2200


COLORS = {
    # Bloomberg-terminal-inspired theme. No logos/trademarks, only terminal-style UI.
    "bg": "#050607",
    "workspace": "#07090B",
    "workspace_grid": "#151A1F",

    "panel": "#0B0D10",
    "panel_2": "#101318",
    "panel_3": "#151A20",

    "slot": "#0A0A0A",
    "slot_border": "#252B33",
    "slot_empty": "#050505",

    "text": "#F4F7FA",
    "muted": "#9AA4AF",
    "faint": "#687381",

    "accent": "#F5B041",
    "accent_2": "#FFD166",
    "accent_soft": "#211806",

    "success": "#00FF66",
    "success_soft": "#062D16",

    "warning": "#FFD400",
    "warning_soft": "#332A00",

    "danger": "#FF4444",
    "danger_soft": "#300A0A",

    "info": "#00AEEF",
    "info_soft": "#061B28",

    "button": "#0F1217",
    "button_hover": "#1C2530",
    "button_border": "#29313A",

    "shadow": "#000000",
}


FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_HEAD = ("Segoe UI", 10, "bold")
FONT_MAIN = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_TINY = ("Segoe UI", 7)
FONT_MONO = ("Cascadia Mono", 9)

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
    # Supports both top-level blocks like "Data_Catalog" and nested blocks
    # like "Market/Monitoring" or "Trades/Live".
    return " / ".join(part.replace("_", " ") for part in folder_name.replace("\\", "/").split("/"))


def module_name_from_block(block_name: str) -> str:
    # Dynamic import module names must not contain slashes or spaces.
    safe = block_name.replace("\\", "/").replace("/", "__").replace(" ", "_").replace("-", "_")
    return f"quant_building_block_{safe}"


class BuildingBlockRegistry:
    def __init__(self, building_blocks_dir: Path):
        self.building_blocks_dir = building_blocks_dir
        self.blocks: dict[str, dict] = {}

    def scan(self) -> dict[str, dict]:
        """Recursively scans Dashboard/Building_Blocks for every code.py.

        Supports:
        - Dashboard/Building_Blocks/Code_Registry/code.py
        - Dashboard/Building_Blocks/Data_Catalog/code.py
        - Dashboard/Building_Blocks/Market/code.py
        - Dashboard/Building_Blocks/Market/Monitoring/code.py
        - Dashboard/Building_Blocks/Pipeline_Management/code.py
        - Dashboard/Building_Blocks/Trades/Live/code.py

        The block key is the relative folder path, e.g.:
        - "Market"
        - "Market/Monitoring"
        - "Trades/Live"
        """
        self.blocks = {}

        if not self.building_blocks_dir.exists():
            return self.blocks

        for code_file in self.building_blocks_dir.rglob("code.py"):
            if "__pycache__" in code_file.parts:
                continue

            folder = code_file.parent
            try:
                rel_folder = folder.relative_to(self.building_blocks_dir)
            except ValueError:
                continue

            if not rel_folder.parts:
                continue

            block_name = rel_folder.as_posix()
            self.blocks[block_name] = {
                "name": block_name,
                "display": display_name(block_name),
                "folder": folder,
                "code_file": code_file,
                "depth": len(rel_folder.parts),
            }

        self.blocks = dict(sorted(self.blocks.items(), key=lambda x: x[0].lower()))
        return self.blocks

    def load_module(self, block_name: str):
        if block_name not in self.blocks:
            raise ValueError(f"Unknown Building Block: {block_name}")

        code_file = self.blocks[block_name]["code_file"]
        module_name = module_name_from_block(block_name)

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
        self.display_mode_var = tk.StringVar(value="FREE")
        self.fixed_panel: DashboardPanel | None = None
        self.fixed_2_panels: list[DashboardPanel] = []
        self.fixed_2_next_slot = 0

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
            height=64,
            highlightbackground=COLORS["slot_border"],
            highlightthickness=1,
        )
        self.topbar.grid(row=0, column=0, sticky="ew")
        self.topbar.grid_columnconfigure(1, weight=1)

        brand = tk.Frame(self.topbar, bg=COLORS["panel"])
        brand.grid(row=0, column=0, sticky="w", padx=18, pady=8)

        logo = tk.Label(
            brand,
            text="B",
            bg="#F4F7FA",
            fg="#000000",
            font=("Segoe UI", 15, "bold"),
            width=2,
            height=1,
        )
        logo.pack(side="left", padx=(0, 8))

        tk.Label(
            brand,
            text="Bloomberg",
            bg=COLORS["panel"],
            fg="#F4F7FA",
            font=("Segoe UI Semibold", 15),
        ).pack(side="left")

        mode_wrap = tk.Frame(self.topbar, bg=COLORS["panel"])
        mode_wrap.grid(row=0, column=1, sticky="w", padx=(36, 8), pady=8)

        self.mode_buttons = {}
        self._mode_button(mode_wrap, "0", "FREE", self.set_free_dashboard_mode, "FREE").pack(side="left", padx=3)
        self._mode_button(mode_wrap, "1", "FIXED DISPLAY", self.set_fixed_display_mode, "FIXED").pack(side="left", padx=3)
        self._mode_button(mode_wrap, "2", "FIXED 2 DISPLAY", self.set_fixed_2_display_mode, "FIXED_2").pack(side="left", padx=3)

        actions = tk.Frame(self.topbar, bg=COLORS["panel"])
        actions.grid(row=0, column=2, sticky="e", padx=16, pady=8)

        self._top_action(actions, "⟳", "REFRESH", self.refresh_blocks).pack(side="left", padx=5)
        self._top_action(actions, "×", "CLEAR", self.clear_workspace).pack(side="left", padx=5)
        self._top_action(actions, "⇧", "LOAD", self.load_layout).pack(side="left", padx=5)
        self._top_action(actions, "▣", "SAVE", self.save_layout).pack(side="left", padx=5)

        self.main_pane = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=COLORS["bg"],
            sashwidth=4,
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
            height=26,
            highlightbackground=COLORS["slot_border"],
            highlightthickness=1,
        )
        self.statusbar.grid(row=2, column=0, sticky="ew")
        self.statusbar.grid_columnconfigure(0, weight=1)
        self.statusbar.grid_columnconfigure(1, weight=0)

        tk.Label(
            self.statusbar,
            textvariable=self.status_var,
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
            anchor="w",
            padx=12,
        ).grid(row=0, column=0, sticky="ew")

        tk.Label(
            self.statusbar,
            textvariable=self.display_mode_var,
            bg=COLORS["panel"],
            fg=COLORS["accent"],
            font=FONT_SMALL,
            anchor="e",
            padx=12,
        ).grid(row=0, column=1, sticky="e")

        self._update_mode_buttons()

    def _mode_button(self, parent, number: str, label: str, command, mode_value: str):
        outer = tk.Frame(parent, bg=COLORS["slot_border"], width=132, height=48)
        outer.pack_propagate(False)

        btn = tk.Button(
            outer,
            text=f"{number}\n{label}",
            bg=COLORS["button"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            relief="flat",
            bd=0,
            command=command,
            cursor="hand2",
            font=("Segoe UI", 8),
            justify="center",
        )
        btn.pack(fill="both", expand=True, padx=1, pady=1)
        self.mode_buttons[mode_value] = (outer, btn)
        return outer

    def _top_action(self, parent, icon: str, label: str, command):
        outer = tk.Frame(parent, bg=COLORS["slot_border"], width=120, height=48)
        outer.pack_propagate(False)

        btn = tk.Button(
            outer,
            text=f"{icon}  {label}",
            bg=COLORS["button"],
            fg=COLORS["text"],
            activebackground=COLORS["button_hover"],
            activeforeground=COLORS["text"],
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            command=command,
            cursor="hand2",
            font=("Segoe UI", 8),
        )
        btn.pack(fill="both", expand=True, padx=1, pady=1)
        return outer

    def _update_mode_buttons(self):
        if not hasattr(self, "mode_buttons"):
            return
        active_mode = self.display_mode_var.get()
        for mode, (_outer, btn) in self.mode_buttons.items():
            if mode == active_mode:
                btn.configure(bg="#151A20", fg=COLORS["accent_2"])
            else:
                btn.configure(bg=COLORS["button"], fg=COLORS["text"])


    def set_fixed_display_mode(self):
        """
        Fixed Display Mode:
        - One selected Building Block fills the complete workspace area.
        - No dragging.
        - No resizing.
        - Clicking another module replaces the current full-screen module.
        """
        self.display_mode_var.set("FIXED")
        self.fixed_2_panels = []
        self.fixed_2_next_slot = 0
        self.clear_workspace(force=True, ask=False)
        self.workspace_canvas.delete("all")
        self.workspace_canvas.configure(scrollregion=(0, 0, WORKSPACE_W, WORKSPACE_H))
        self._draw_fixed_hint()
        self._update_mode_buttons()
        self.update_status("Mode: FIXED DISPLAY. Click a module to open it full workspace.")

    def set_fixed_2_display_mode(self):
        """
        Fixed 2 Display Mode:
        - Two modules side-by-side.
        - Each module gets half of the workspace.
        - Clicking modules fills left then right, then replaces left/right alternately.
        - No dragging and no resizing.
        """
        self.display_mode_var.set("FIXED_2")
        self.clear_workspace(force=True, ask=False)
        self.fixed_panel = None
        self.fixed_2_panels = []
        self.fixed_2_next_slot = 0
        self.workspace_canvas.delete("all")
        self.workspace_canvas.configure(scrollregion=(0, 0, WORKSPACE_W, WORKSPACE_H))
        self._draw_fixed_2_hint()
        self._update_mode_buttons()
        self.update_status("Mode: FIXED 2 DISPLAY. Click two modules to open left/right.")

    def set_free_dashboard_mode(self):
        """
        Free Dashboard Mode:
        - Original draggable/resizable panel workspace.
        - Multiple panels possible.
        - Save/load layout supported.
        """
        self.display_mode_var.set("FREE")
        self.fixed_panel = None
        self.fixed_2_panels = []
        self.fixed_2_next_slot = 0
        self.fixed_2_panels = []
        self.fixed_2_next_slot = 0
        self.clear_workspace(force=True, ask=False)
        self.workspace_canvas.delete("all")
        self._draw_workspace_hint()
        self._update_mode_buttons()
        self.update_status("Mode: FREE DASHBOARD. Click modules as panels. Right-click drag optional.")

    def _draw_fixed_hint(self):
        self.workspace_canvas.delete("workspace_hint")
        self._draw_workspace_grid()
        if self.panels:
            return

        w = max(1000, self.workspace_canvas.winfo_width())
        h = max(700, self.workspace_canvas.winfo_height())
        cx = w // 2
        cy = h // 2

        self.workspace_canvas.create_text(cx, cy - 60, text="▦", anchor="center", fill=COLORS["muted"], font=("Segoe UI", 34, "bold"), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy - 8, text="WORKSPACE", anchor="center", fill="#E9EEF5", font=("Segoe UI", 22, "bold"), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy + 34, text="Select a module from the left to open it here.", anchor="center", fill="#A7B0BA", font=("Segoe UI", 11), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy + 62, text="Mode 1: one module uses the full display.", anchor="center", fill=COLORS["muted"], font=("Segoe UI", 10), tags=("workspace_hint",))


    def _draw_fixed_2_hint(self):
        self.workspace_canvas.delete("workspace_hint")
        self._draw_workspace_grid()
        if self.panels:
            return

        w = max(1000, self.workspace_canvas.winfo_width())
        h = max(700, self.workspace_canvas.winfo_height())
        cx = w // 2
        cy = h // 2

        self.workspace_canvas.create_text(cx, cy - 60, text="▦", anchor="center", fill=COLORS["muted"], font=("Segoe UI", 34, "bold"), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy - 8, text="WORKSPACE", anchor="center", fill="#E9EEF5", font=("Segoe UI", 22, "bold"), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy + 34, text="Select two modules from the left.", anchor="center", fill="#A7B0BA", font=("Segoe UI", 11), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy + 62, text="Mode 2: each module receives half of the display.", anchor="center", fill=COLORS["muted"], font=("Segoe UI", 10), tags=("workspace_hint",))


    def _build_sidebar(self):
        head = tk.Frame(self.sidebar, bg=COLORS["panel"])
        head.pack(fill="x", padx=18, pady=(16, 10))

        search_wrap = tk.Frame(head, bg=COLORS["button_border"])
        search_wrap.pack(fill="x", pady=(0, 12))

        search_entry = tk.Entry(
            search_wrap,
            textvariable=self.block_search_var,
            bg=COLORS["button"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="solid",
            bd=1,
            font=("Segoe UI", 10),
        )
        search_entry.pack(side="left", fill="x", expand=True, padx=1, pady=1, ipady=8)
        search_entry.bind("<KeyRelease>", lambda _e: self._refresh_sidebar())

        search_icon = tk.Label(
            search_wrap,
            text="⌕",
            bg=COLORS["button"],
            fg=COLORS["muted"],
            font=("Segoe UI", 15),
            width=3,
        )
        search_icon.pack(side="right", padx=1, pady=1, ipady=4)

        self.block_count_label = tk.Label(
            head,
            text="0 modules",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=FONT_SMALL,
        )
        self.block_count_label.pack(anchor="w")

        self.block_canvas = tk.Canvas(self.sidebar, bg=COLORS["panel"], highlightthickness=0, bd=0)
        self.block_scroll = tk.Scrollbar(self.sidebar, orient="vertical", command=self.block_canvas.yview)
        self.block_list_frame = tk.Frame(self.block_canvas, bg=COLORS["panel"])

        self.block_list_frame.bind(
            "<Configure>",
            lambda _e: self.block_canvas.configure(scrollregion=self.block_canvas.bbox("all")),
        )

        self.block_canvas_window = self.block_canvas.create_window((0, 0), window=self.block_list_frame, anchor="nw")
        self.block_canvas.configure(yscrollcommand=self.block_scroll.set)

        self.block_canvas.pack(side="left", fill="both", expand=True, padx=(12, 0), pady=(0, 10))
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
        self.workspace_canvas.bind("<Configure>", self._on_workspace_resize)

        self._draw_workspace_hint()

    def _on_workspace_resize(self, event):
        """
        Keep fixed display panels filling the workspace when the window is resized.
        """
        if self.display_mode_var.get() == "FIXED":
            if self.fixed_panel is None or self.fixed_panel.window_id is None:
                return

            w = max(MIN_PANEL_W, int(event.width) - 28)
            h = max(MIN_PANEL_H, int(event.height) - 28)

            self.fixed_panel.panel_x = 14
            self.fixed_panel.panel_y = 14
            self.fixed_panel.panel_w = w
            self.fixed_panel.panel_h = h

            self.workspace_canvas.coords(self.fixed_panel.window_id, 14, 14)
            self.workspace_canvas.itemconfigure(self.fixed_panel.window_id, width=w, height=h)

            if self.fixed_panel.shadow_id is not None:
                self.workspace_canvas.coords(self.fixed_panel.shadow_id, 22, 22, w + 22, h + 22)

            self.workspace_canvas.configure(scrollregion=(0, 0, max(w + 40, WORKSPACE_W), max(h + 40, WORKSPACE_H)))
            return

        if self.display_mode_var.get() == "FIXED_2":
            active = [p for p in self.fixed_2_panels if p is not None]
            if not active:
                return

            total_w = max(MIN_PANEL_W * 2 + 40, int(event.width) - 28)
            total_h = max(MIN_PANEL_H, int(event.height) - 28)
            gap = 12
            slot_w = max(MIN_PANEL_W, int((total_w - gap) / 2))
            slot_h = max(MIN_PANEL_H, total_h)

            for slot, panel in enumerate(self.fixed_2_panels[:2]):
                if panel is None or panel.window_id is None:
                    continue

                x = 14 if slot == 0 else 14 + slot_w + gap
                y = 14

                panel.panel_x = x
                panel.panel_y = y
                panel.panel_w = slot_w
                panel.panel_h = slot_h

                self.workspace_canvas.coords(panel.window_id, x, y)
                self.workspace_canvas.itemconfigure(panel.window_id, width=slot_w, height=slot_h)

                if panel.shadow_id is not None:
                    self.workspace_canvas.coords(panel.shadow_id, x + 8, y + 8, x + slot_w + 8, y + slot_h + 8)

            self.workspace_canvas.configure(
                scrollregion=(0, 0, max(total_w + 40, WORKSPACE_W), max(total_h + 40, WORKSPACE_H))
            )

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
        if self.panels:
            return

        w = max(1000, self.workspace_canvas.winfo_width())
        h = max(700, self.workspace_canvas.winfo_height())
        cx = w // 2
        cy = h // 2

        self.workspace_canvas.create_text(cx, cy - 60, text="▦", anchor="center", fill=COLORS["muted"], font=("Segoe UI", 34, "bold"), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy - 8, text="WORKSPACE", anchor="center", fill="#E9EEF5", font=("Segoe UI", 22, "bold"), tags=("workspace_hint",))
        self.workspace_canvas.create_text(cx, cy + 34, text="Free mode: open multiple modules, move by header, resize with grip.", anchor="center", fill="#A7B0BA", font=("Segoe UI", 11), tags=("workspace_hint",))


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
        if "market" in lower:
            return COLORS["info"], COLORS["info_soft"]
        if "trade" in lower:
            return COLORS["success"], COLORS["success_soft"]
        return COLORS["accent"], COLORS["accent_soft"]

    def _filtered_blocks(self) -> dict[str, dict]:
        q = self.block_search_var.get().strip().lower()
        if q in {"search modules...", "search modules"}:
            q = ""
        return {
            name: info
            for name, info in self.registry.blocks.items()
            if not q or q in name.lower() or q in info["display"].lower()
        }

    def _refresh_sidebar(self):
        for child in self.block_list_frame.winfo_children():
            child.destroy()

        blocks = self._filtered_blocks()
        self.block_count_label.config(text=f"{len(blocks)} / {len(self.registry.blocks)} modules")

        if not blocks:
            tk.Label(
                self.block_list_frame,
                text="NO MODULES FOUND.",
                bg=COLORS["panel"],
                fg=COLORS["muted"],
                font=FONT_MAIN,
                wraplength=280,
            ).pack(anchor="w", pady=10, padx=6)
            return

        if not self.selected_block_name.get() or self.selected_block_name.get() not in self.registry.blocks:
            self.selected_block_name.set(next(iter(self.registry.blocks.keys()), ""))

        groups = {
            "MARKET RESEARCH": [],
            "TRADING & EXECUTION": [],
            "DATA & SYSTEM": [],
            "GENERAL": [],
        }

        for block_name, info in blocks.items():
            lower = block_name.lower()
            if "market" in lower or "correlation" in lower or "season" in lower or "session" in lower:
                groups["MARKET RESEARCH"].append((block_name, info))
            elif "trade" in lower or "live" in lower:
                groups["TRADING & EXECUTION"].append((block_name, info))
            elif "pipeline" in lower or "catalog" in lower or "registry" in lower or "data" in lower:
                groups["DATA & SYSTEM"].append((block_name, info))
            else:
                groups["GENERAL"].append((block_name, info))

        for group_name, items in groups.items():
            if not items:
                continue

            group = tk.Frame(self.block_list_frame, bg=COLORS["panel"])
            group.pack(fill="x", padx=4, pady=(8, 4))

            head = tk.Frame(group, bg=COLORS["panel"])
            head.pack(fill="x", padx=4, pady=(0, 3))

            tk.Label(
                head,
                text=group_name,
                bg=COLORS["panel"],
                fg=COLORS["text"],
                font=FONT_HEAD,
                anchor="w",
            ).pack(side="left")

            tk.Label(
                head,
                text=str(len(items)),
                bg=COLORS["panel"],
                fg=COLORS["success"],
                font=FONT_TINY,
                anchor="e",
            ).pack(side="right")

            line = tk.Frame(group, bg=COLORS["slot_border"], height=1)
            line.pack(fill="x", padx=4, pady=(0, 3))

            for block_name, info in items:
                self._create_sidebar_card(block_name, info)

    def _create_sidebar_card(self, block_name: str, info: dict):
        """
        Clean Bloomberg-style module row.
        """
        selected = self.selected_block_name.get() == block_name
        accent, _soft = self._card_accent(block_name)

        row_bg = "#101A23" if selected else COLORS["panel"]
        border_bg = accent if selected else COLORS["panel"]

        row = tk.Frame(
            self.block_list_frame,
            bg=border_bg,
            height=42,
            cursor="hand2",
        )
        row.pack(fill="x", padx=8, pady=2)
        row.pack_propagate(False)

        inner = tk.Frame(row, bg=row_bg, cursor="hand2")
        inner.pack(fill="both", expand=True, padx=(2 if selected else 0), pady=(1 if selected else 0))

        dot = tk.Label(
            inner,
            text="●",
            bg=row_bg,
            fg=accent,
            font=("Segoe UI", 8),
            cursor="hand2",
        )
        dot.pack(side="left", padx=(12, 10))

        title = tk.Label(
            inner,
            text=info["display"],
            bg=row_bg,
            fg=COLORS["text"],
            font=("Segoe UI", 10, "bold"),
            anchor="w",
            cursor="hand2",
        )
        title.pack(side="left", fill="x", expand=True)

        arrow = tk.Label(
            inner,
            text="›",
            bg=row_bg,
            fg=COLORS["text"],
            font=("Segoe UI", 16, "bold"),
            cursor="hand2",
        )
        arrow.pack(side="right", padx=(6, 12))

        def open_block(_event=None, b=block_name):
            self.selected_block_name.set(b)
            self.add_panel_auto(b)
            self._refresh_sidebar()

        for w in (row, inner, dot, title, arrow):
            w.bind("<Button-1>", open_block)
            w.bind("<Double-1>", open_block)
            w.bind("<ButtonPress-3>", lambda e, b=block_name: self.start_drag(b, e))


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

        if self.display_mode_var.get() == "FIXED":
            block_name = self.dragging_block_name
            self.cancel_drag(clear_status=False)
            self.open_fixed_panel(block_name)
            return

        if self.display_mode_var.get() == "FIXED_2":
            block_name = self.dragging_block_name
            self.cancel_drag(clear_status=False)
            self.open_fixed_2_panel(block_name)
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
        if self.display_mode_var.get() == "FIXED":
            self.open_fixed_panel(block_name)
            return

        if self.display_mode_var.get() == "FIXED_2":
            self.open_fixed_2_panel(block_name)
            return

        offset = self.next_panel_offset * 34
        self.next_panel_offset = (self.next_panel_offset + 1) % 12
        x = 80 + offset
        y = 90 + offset
        self.add_panel(block_name, x, y)

    def open_fixed_panel(self, block_name: str):
        """
        Opens exactly one full-workspace panel.
        Existing panel is removed first.
        """
        for panel in list(self.panels):
            panel.close()
        self.panels.clear()
        self.workspace_canvas.delete("all")
        self._draw_workspace_grid()

        # Use current visible canvas size. Fallback to reasonable size during first render.
        self.update_idletasks()
        w = max(MIN_PANEL_W, self.workspace_canvas.winfo_width() - 28)
        h = max(MIN_PANEL_H, self.workspace_canvas.winfo_height() - 28)

        panel = DashboardPanel(
            self.workspace_canvas,
            app=self,
            block_name=block_name,
            x=14,
            y=14,
            width=w,
            height=h,
        )
        panel.resize_grip.place_forget()

        # Disable move/resize bindings in fixed mode.
        for widget in (panel.header, panel.title_label):
            widget.unbind("<ButtonPress-1>")
            widget.unbind("<B1-Motion>")
            widget.unbind("<ButtonRelease-1>")

        self.panels.append(panel)
        self.fixed_panel = panel
        self.workspace_canvas.configure(scrollregion=(0, 0, max(w + 40, WORKSPACE_W), max(h + 40, WORKSPACE_H)))
        self.update_status(f"Fixed display: {display_name(block_name)}")

    def open_fixed_2_panel(self, block_name: str):
        """
        Opens/replaces a module in one of two fixed half-screen slots.
        Slot 0 = left, Slot 1 = right.
        """
        self.workspace_canvas.delete("workspace_hint")
        self._draw_workspace_grid()
        self.update_idletasks()

        total_w = max(MIN_PANEL_W * 2 + 40, self.workspace_canvas.winfo_width() - 28)
        total_h = max(MIN_PANEL_H, self.workspace_canvas.winfo_height() - 28)
        gap = 12
        slot_w = max(MIN_PANEL_W, int((total_w - gap) / 2))
        slot_h = max(MIN_PANEL_H, total_h)

        slot = self.fixed_2_next_slot % 2
        self.fixed_2_next_slot = (self.fixed_2_next_slot + 1) % 2

        # If slot already exists, remove it.
        if len(self.fixed_2_panels) > slot and self.fixed_2_panels[slot] is not None:
            old_panel = self.fixed_2_panels[slot]
            try:
                old_panel.close()
            except Exception:
                pass

        while len(self.fixed_2_panels) < 2:
            self.fixed_2_panels.append(None)

        x = 14 if slot == 0 else 14 + slot_w + gap
        y = 14

        panel = DashboardPanel(
            self.workspace_canvas,
            app=self,
            block_name=block_name,
            x=x,
            y=y,
            width=slot_w,
            height=slot_h,
        )
        panel.resize_grip.place_forget()

        for widget in (panel.header, panel.title_label):
            widget.unbind("<ButtonPress-1>")
            widget.unbind("<B1-Motion>")
            widget.unbind("<ButtonRelease-1>")

        self.fixed_2_panels[slot] = panel

        # Rebuild panel list cleanly from available fixed-2 panels.
        self.panels = [p for p in self.fixed_2_panels if p is not None]

        self.workspace_canvas.configure(
            scrollregion=(0, 0, max(total_w + 40, WORKSPACE_W), max(total_h + 40, WORKSPACE_H))
        )
        self.update_status(f"Fixed 2 display slot {slot + 1}: {display_name(block_name)}")

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

    def clear_workspace(self, force: bool = False, ask: bool = True):
        if not self.panels and not force:
            return
        if ask and not force:
            if not messagebox.askyesno("Clear Workspace", "Alle Panels entfernen?"):
                return

        for panel in list(self.panels):
            panel.close()
        self.panels.clear()
        self.fixed_panel = None
        self.fixed_2_panels = []
        self.fixed_2_next_slot = 0
        self.workspace_canvas.delete("all")

        if self.display_mode_var.get() == "FIXED":
            self._draw_fixed_hint()
        elif self.display_mode_var.get() == "FIXED_2":
            self._draw_fixed_2_hint()
        else:
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
        self.display_mode_var.set("FREE")
        self.fixed_panel = None
        self.fixed_2_panels = []
        self.fixed_2_next_slot = 0

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

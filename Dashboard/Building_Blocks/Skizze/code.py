# ============================================================
# CODE_REGISTRY
# ============================================================
# script_id: skizze_building_block
# script_name: Skizze Building Block
# owner: Leon
# status: active
# layer: Dashboard
# domain: UI_Design
# asset_type: Dashboard Building Block
# purpose: Sketch board building block for drawing dashboard/UI mockups before coding. Supports boxes, text, straight lines, freehand drawing, eraser, select/move/delete, save/load JSON, export PNG, and copy UI prompt.
# inputs:
#   - manual drawing
# outputs:
#   - sketch JSON
#   - sketch PNG
# dependencies:
#   - tkinter
#   - pillow optional for PNG export
# schedule: manual
# version: v1.1.0_free_draw
# last_reviewed: 2026-06-08
# required_api:
#   - build_panel(parent, **kwargs)
# ============================================================

from __future__ import annotations

import json
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# ============================================================
# THEME
# ============================================================

APP_BG = "#0B0F14"
SURFACE = "#10161D"
SURFACE_2 = "#151C25"
SURFACE_3 = "#1B2430"
BORDER = "#273241"

TEXT = "#E6EDF3"
TEXT_2 = "#AAB4C0"
MUTED = "#6F7B88"

ACCENT = "#4F8EF7"
BLUE = "#4F8EF7"
GREEN = "#3CCF91"
YELLOW = "#F5B041"
RED = "#E74C3C"

CANVAS_BG = "#070B10"
GRID = "#151B23"

FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_SECTION = ("Segoe UI", 9, "bold")
FONT_BODY = ("Segoe UI", 9)
FONT_SMALL = ("Segoe UI", 8)
FONT_MONO = ("Consolas", 9)

DEFAULT_BOX_FILL = "#111821"
DEFAULT_BOX_OUTLINE = "#344154"
DEFAULT_TEXT_COLOR = "#E6EDF3"


# ============================================================
# APP
# ============================================================

class QuantSketchBoard(tk.Frame):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, bg=APP_BG, **kwargs)
        self.parent = parent

        self.tool = tk.StringVar(value="select")
        self.snap = tk.BooleanVar(value=True)
        self.show_grid = tk.BooleanVar(value=True)

        self.items: list[dict] = []
        self.selected_id: int | None = None
        self.drag_start = None
        self.draw_start = None
        self.preview_item = None
        self.current_stroke: list[tuple[int, int]] = []
        self.next_id = 1
        self.current_file: Path | None = None

        self._build_ui()
        self._bind_shortcuts()
        self.redraw()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._build_topbar()
        self._build_sidebar()
        self._build_canvas()
        self._build_statusbar()

    def _build_topbar(self):
        top = tk.Frame(self, bg=APP_BG, height=58, highlightbackground=BORDER, highlightthickness=1)
        top.grid(row=0, column=0, columnspan=2, sticky="ew")
        top.grid_columnconfigure(1, weight=1)

        tk.Label(top, text="QUANT SKETCH BOARD", bg=APP_BG, fg=TEXT, font=FONT_TITLE).grid(
            row=0, column=0, padx=18, pady=14, sticky="w"
        )

        self.file_label = tk.Label(top, text="Untitled Sketch", bg=APP_BG, fg=TEXT_2, font=FONT_BODY)
        self.file_label.grid(row=0, column=1, sticky="w")

        actions = tk.Frame(top, bg=APP_BG)
        actions.grid(row=0, column=2, sticky="e", padx=14)

        self._btn(actions, "New", self.new_sketch).pack(side="left", padx=4)
        self._btn(actions, "Save JSON", self.save_json).pack(side="left", padx=4)
        self._btn(actions, "Load JSON", self.load_json).pack(side="left", padx=4)
        self._btn(actions, "Export PNG", self.export_png).pack(side="left", padx=4)
        self._btn(actions, "Copy Prompt", self.copy_prompt).pack(side="left", padx=4)

    def _build_sidebar(self):
        side = tk.Frame(self, bg=SURFACE, width=250, highlightbackground=BORDER, highlightthickness=1)
        side.grid(row=1, column=0, sticky="nsw")
        side.grid_propagate(False)

        tk.Label(side, text="TOOLS", bg=SURFACE, fg=TEXT, font=FONT_SECTION).pack(anchor="w", padx=16, pady=(18, 8))

        tools = [
            ("select", "Select / Move"),
            ("box", "Draw Box"),
            ("text", "Add Text"),
            ("line", "Draw Line"),
            ("pen", "Free Draw"),
            ("eraser", "Eraser"),
            ("delete", "Delete"),
        ]

        for value, label in tools:
            self._tool_radio(side, value, label).pack(fill="x", padx=12, pady=3)

        self._sep(side)

        tk.Label(side, text="QUICK BOXES", bg=SURFACE, fg=TEXT, font=FONT_SECTION).pack(anchor="w", padx=16, pady=(14, 8))
        self._btn(side, "Header Box", lambda: self.add_template_box("Header", 60, 50, 920, 64)).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Sidebar Box", lambda: self.add_template_box("Sidebar", 60, 130, 260, 560)).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Workspace Box", lambda: self.add_template_box("Workspace", 340, 130, 640, 560)).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Actions Box", lambda: self.add_template_box("Actions", 60, 710, 920, 78)).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Output Box", lambda: self.add_template_box("Output", 60, 600, 920, 90)).pack(fill="x", padx=12, pady=3)

        self._sep(side)

        tk.Label(side, text="OPTIONS", bg=SURFACE, fg=TEXT, font=FONT_SECTION).pack(anchor="w", padx=16, pady=(14, 8))

        tk.Checkbutton(
            side, text="Show grid", variable=self.show_grid, command=self.redraw,
            bg=SURFACE, fg=TEXT_2, selectcolor=SURFACE_3,
            activebackground=SURFACE, activeforeground=TEXT, font=FONT_BODY
        ).pack(anchor="w", padx=14, pady=2)

        tk.Checkbutton(
            side, text="Snap to grid", variable=self.snap,
            bg=SURFACE, fg=TEXT_2, selectcolor=SURFACE_3,
            activebackground=SURFACE, activeforeground=TEXT, font=FONT_BODY
        ).pack(anchor="w", padx=14, pady=2)

        self._sep(side)

        tk.Label(side, text="EDIT", bg=SURFACE, fg=TEXT, font=FONT_SECTION).pack(anchor="w", padx=16, pady=(14, 8))
        self._btn(side, "Duplicate Selected", self.duplicate_selected).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Delete Selected", self.delete_selected).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Bring Forward", lambda: self.reorder_selected(1)).pack(fill="x", padx=12, pady=3)
        self._btn(side, "Send Backward", lambda: self.reorder_selected(-1)).pack(fill="x", padx=12, pady=3)

        tk.Label(
            side,
            text="Workflow:\n1. Draw rough UI\n2. Export PNG\n3. Send screenshot to ChatGPT\n4. Generate code",
            bg=SURFACE, fg=MUTED, font=FONT_SMALL, justify="left"
        ).pack(anchor="w", padx=16, pady=(28, 0))

    def _build_canvas(self):
        wrap = tk.Frame(self, bg=APP_BG)
        wrap.grid(row=1, column=1, sticky="nsew")
        wrap.grid_rowconfigure(0, weight=1)
        wrap.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(wrap, bg=CANVAS_BG, highlightthickness=0, bd=0, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=14, pady=14)

        self.canvas.bind("<Button-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=SURFACE, height=34, highlightbackground=BORDER, highlightthickness=1)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)

        self.status = tk.Label(bar, text="Ready", bg=SURFACE, fg=TEXT_2, font=FONT_SMALL)
        self.status.grid(row=0, column=0, sticky="w", padx=16, pady=8)

        tk.Label(bar, text="Shortcuts: B=Box | T=Text | L=Line | P=Free Draw | E=Eraser | V=Select | Del=Delete | Ctrl+S=Save", bg=SURFACE, fg=MUTED, font=FONT_SMALL).grid(
            row=0, column=1, sticky="e", padx=16
        )

    def _btn(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=SURFACE_2,
            fg=TEXT,
            activebackground=SURFACE_3,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2",
            font=FONT_BODY,
        )

    def _tool_radio(self, parent, value, text):
        return tk.Radiobutton(
            parent,
            text=text,
            variable=self.tool,
            value=value,
            bg=SURFACE,
            fg=TEXT_2,
            activebackground=SURFACE,
            activeforeground=TEXT,
            selectcolor=SURFACE_3,
            indicatoron=False,
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            font=FONT_BODY,
            command=lambda: self.set_status(f"Tool: {text}"),
        )

    def _sep(self, parent):
        tk.Frame(parent, height=1, bg=BORDER).pack(fill="x", padx=12, pady=14)

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    def _bind_shortcuts(self):
        self.bind("<Control-s>", lambda _e: self.save_json())
        self.bind("<Control-o>", lambda _e: self.load_json())
        self.bind("<Delete>", lambda _e: self.delete_selected())
        self.bind("<BackSpace>", lambda _e: self.delete_selected())
        self.bind("b", lambda _e: self.tool.set("box"))
        self.bind("B", lambda _e: self.tool.set("box"))
        self.bind("t", lambda _e: self.tool.set("text"))
        self.bind("T", lambda _e: self.tool.set("text"))
        self.bind("l", lambda _e: self.tool.set("line"))
        self.bind("L", lambda _e: self.tool.set("line"))
        self.bind("p", lambda _e: self.tool.set("pen"))
        self.bind("P", lambda _e: self.tool.set("pen"))
        self.bind("e", lambda _e: self.tool.set("eraser"))
        self.bind("E", lambda _e: self.tool.set("eraser"))
        self.bind("v", lambda _e: self.tool.set("select"))
        self.bind("V", lambda _e: self.tool.set("select"))

    def on_left_down(self, event):
        x, y = self.snap_xy(event.x, event.y)
        tool = self.tool.get()

        if tool == "select":
            item = self.find_item_at(x, y)
            self.selected_id = item["id"] if item else None
            self.drag_start = (x, y)
            self.redraw()
            return

        if tool == "delete":
            item = self.find_item_at(x, y)
            if item:
                self.items = [i for i in self.items if i["id"] != item["id"]]
                self.selected_id = None
                self.redraw()
            return

        if tool == "eraser":
            item = self.find_item_at(x, y)
            if item:
                self.items = [i for i in self.items if i["id"] != item["id"]]
                self.selected_id = None
                self.redraw()
            return

        if tool == "pen":
            self.current_stroke = [(x, y)]
            self.preview_item = None
            return

        if tool == "text":
            text = simpledialog.askstring("Text", "Label text:", parent=self)
            if text:
                self.items.append({
                    "id": self._new_id(),
                    "type": "text",
                    "x": x,
                    "y": y,
                    "text": text,
                    "color": DEFAULT_TEXT_COLOR,
                    "font_size": 14,
                })
                self.selected_id = self.items[-1]["id"]
                self.redraw()
            return

        if tool in {"box", "line"}:
            self.draw_start = (x, y)
            self.preview_item = None

    def on_left_drag(self, event):
        x, y = self.snap_xy(event.x, event.y)
        tool = self.tool.get()

        if tool == "select" and self.selected_id is not None and self.drag_start:
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.move_item(self.selected_id, dx, dy)
            self.drag_start = (x, y)
            self.redraw()
            return

        if tool == "eraser":
            item = self.find_item_at(x, y)
            if item:
                self.items = [i for i in self.items if i["id"] != item["id"]]
                self.selected_id = None
                self.redraw()
            return

        if tool == "pen" and self.current_stroke:
            last_x, last_y = self.current_stroke[-1]
            # avoid too many points
            if abs(x - last_x) >= 2 or abs(y - last_y) >= 2:
                self.current_stroke.append((x, y))
                self.canvas.create_line(last_x, last_y, x, y, fill=ACCENT, width=2, capstyle="round", smooth=True)
            return

        if tool in {"box", "line"} and self.draw_start:
            if self.preview_item:
                self.canvas.delete(self.preview_item)
            x1, y1 = self.draw_start
            if tool == "box":
                self.preview_item = self.canvas.create_rectangle(x1, y1, x, y, outline=ACCENT, dash=(4, 3), width=2)
            else:
                self.preview_item = self.canvas.create_line(x1, y1, x, y, fill=ACCENT, dash=(4, 3), width=2)

    def on_left_up(self, event):
        x, y = self.snap_xy(event.x, event.y)
        tool = self.tool.get()

        if self.preview_item:
            self.canvas.delete(self.preview_item)
            self.preview_item = None

        if tool == "pen" and self.current_stroke:
            if len(self.current_stroke) >= 2:
                self.items.append({
                    "id": self._new_id(),
                    "type": "freehand",
                    "points": self.current_stroke[:],
                    "color": ACCENT,
                    "width": 2,
                })
                self.selected_id = self.items[-1]["id"]
            self.current_stroke = []
            self.redraw()
            return

        if tool == "box" and self.draw_start:
            x1, y1 = self.draw_start
            if abs(x - x1) > 16 and abs(y - y1) > 16:
                label = simpledialog.askstring("Box label", "Label:", parent=self)
                self.items.append({
                    "id": self._new_id(),
                    "type": "box",
                    "x1": min(x1, x),
                    "y1": min(y1, y),
                    "x2": max(x1, x),
                    "y2": max(y1, y),
                    "label": label or "Box",
                    "fill": DEFAULT_BOX_FILL,
                    "outline": DEFAULT_BOX_OUTLINE,
                })
                self.selected_id = self.items[-1]["id"]

        if tool == "line" and self.draw_start:
            x1, y1 = self.draw_start
            if abs(x - x1) > 8 or abs(y - y1) > 8:
                self.items.append({
                    "id": self._new_id(),
                    "type": "line",
                    "x1": x1,
                    "y1": y1,
                    "x2": x,
                    "y2": y,
                    "color": BORDER,
                })
                self.selected_id = self.items[-1]["id"]

        self.draw_start = None
        self.redraw()

    def on_double_click(self, event):
        x, y = self.snap_xy(event.x, event.y)
        item = self.find_item_at(x, y)
        if not item:
            return

        if item["type"] == "box":
            label = simpledialog.askstring("Edit label", "Label:", initialvalue=item.get("label", ""), parent=self)
            if label is not None:
                item["label"] = label
        elif item["type"] == "text":
            text = simpledialog.askstring("Edit text", "Text:", initialvalue=item.get("text", ""), parent=self)
            if text is not None:
                item["text"] = text
        self.redraw()

    def on_right_click(self, event):
        x, y = self.snap_xy(event.x, event.y)
        item = self.find_item_at(x, y)
        if item:
            self.selected_id = item["id"]
            self.redraw()

        menu = tk.Menu(self, tearoff=0, bg=SURFACE_2, fg=TEXT, activebackground=SURFACE_3, activeforeground=TEXT)
        menu.add_command(label="Edit Text / Label", command=self.edit_selected)
        menu.add_command(label="Duplicate", command=self.duplicate_selected)
        menu.add_command(label="Delete", command=self.delete_selected)
        menu.add_separator()
        menu.add_command(label="Bring Forward", command=lambda: self.reorder_selected(1))
        menu.add_command(label="Send Backward", command=lambda: self.reorder_selected(-1))
        menu.tk_popup(event.x_root, event.y_root)

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------

    def redraw(self):
        self.canvas.delete("all")
        if self.show_grid.get():
            self.draw_grid()

        for item in self.items:
            self.draw_item(item)

        self.set_status(f"Items: {len(self.items)} | Selected: {self.selected_id or '-'} | Tool: {self.tool.get()}")

    def draw_grid(self):
        w = max(self.canvas.winfo_width(), 1200)
        h = max(self.canvas.winfo_height(), 760)

        for x in range(0, w, 24):
            self.canvas.create_line(x, 0, x, h, fill=GRID)
        for y in range(0, h, 24):
            self.canvas.create_line(0, y, w, y, fill=GRID)

        for x in range(0, w, 120):
            self.canvas.create_line(x, 0, x, h, fill="#1B2330")
        for y in range(0, h, 120):
            self.canvas.create_line(0, y, w, y, fill="#1B2330")

    def draw_item(self, item):
        selected = item["id"] == self.selected_id

        if item["type"] == "box":
            outline = ACCENT if selected else item.get("outline", DEFAULT_BOX_OUTLINE)
            width = 2 if selected else 1
            self.canvas.create_rectangle(
                item["x1"], item["y1"], item["x2"], item["y2"],
                fill=item.get("fill", DEFAULT_BOX_FILL),
                outline=outline,
                width=width,
            )
            self.canvas.create_text(
                item["x1"] + 12, item["y1"] + 12,
                text=item.get("label", ""),
                anchor="nw",
                fill=TEXT,
                font=FONT_SECTION,
            )

        elif item["type"] == "text":
            self.canvas.create_text(
                item["x"], item["y"],
                text=item.get("text", ""),
                anchor="nw",
                fill=item.get("color", DEFAULT_TEXT_COLOR),
                font=("Segoe UI", int(item.get("font_size", 14)), "bold"),
            )
            if selected:
                bbox = self.canvas.bbox("all")
                self.canvas.create_rectangle(item["x"] - 4, item["y"] - 4, item["x"] + 180, item["y"] + 28, outline=ACCENT, dash=(3, 2))

        elif item["type"] == "line":
            self.canvas.create_line(
                item["x1"], item["y1"], item["x2"], item["y2"],
                fill=ACCENT if selected else item.get("color", BORDER),
                width=2 if selected else 1,
            )

        elif item["type"] == "freehand":
            pts = item.get("points", [])
            if len(pts) >= 2:
                flat = []
                for px, py in pts:
                    flat.extend([px, py])
                self.canvas.create_line(
                    *flat,
                    fill=ACCENT if selected else item.get("color", ACCENT),
                    width=int(item.get("width", 2)) + (1 if selected else 0),
                    capstyle="round",
                    joinstyle="round",
                    smooth=True,
                )
                if selected:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    self.canvas.create_rectangle(min(xs)-4, min(ys)-4, max(xs)+4, max(ys)+4, outline=ACCENT, dash=(3, 2))

    # --------------------------------------------------------
    # Item logic
    # --------------------------------------------------------

    def _new_id(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    def snap_xy(self, x, y):
        if not self.snap.get():
            return x, y
        grid = 12
        return round(x / grid) * grid, round(y / grid) * grid

    def find_item_at(self, x, y):
        for item in reversed(self.items):
            if item["type"] == "box":
                if item["x1"] <= x <= item["x2"] and item["y1"] <= y <= item["y2"]:
                    return item
            elif item["type"] == "text":
                if item["x"] <= x <= item["x"] + max(80, len(item.get("text", "")) * 9) and item["y"] <= y <= item["y"] + 26:
                    return item
            elif item["type"] == "line":
                if min(item["x1"], item["x2"]) - 6 <= x <= max(item["x1"], item["x2"]) + 6 and min(item["y1"], item["y2"]) - 6 <= y <= max(item["y1"], item["y2"]) + 6:
                    return item
            elif item["type"] == "freehand":
                pts = item.get("points", [])
                if pts:
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    if min(xs) - 8 <= x <= max(xs) + 8 and min(ys) - 8 <= y <= max(ys) + 8:
                        return item
        return None

    def move_item(self, item_id, dx, dy):
        item = self.get_item(item_id)
        if not item:
            return

        if item["type"] == "box":
            item["x1"] += dx
            item["x2"] += dx
            item["y1"] += dy
            item["y2"] += dy
        elif item["type"] == "text":
            item["x"] += dx
            item["y"] += dy
        elif item["type"] == "line":
            item["x1"] += dx
            item["x2"] += dx
            item["y1"] += dy
            item["y2"] += dy
        elif item["type"] == "freehand":
            item["points"] = [(px + dx, py + dy) for px, py in item.get("points", [])]

    def get_item(self, item_id):
        for item in self.items:
            if item["id"] == item_id:
                return item
        return None

    def add_template_box(self, label, x, y, w, h):
        self.items.append({
            "id": self._new_id(),
            "type": "box",
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
            "label": label,
            "fill": DEFAULT_BOX_FILL,
            "outline": DEFAULT_BOX_OUTLINE,
        })
        self.selected_id = self.items[-1]["id"]
        self.redraw()

    def edit_selected(self):
        item = self.get_item(self.selected_id)
        if not item:
            return
        if item["type"] == "box":
            label = simpledialog.askstring("Edit label", "Label:", initialvalue=item.get("label", ""), parent=self)
            if label is not None:
                item["label"] = label
        elif item["type"] == "text":
            text = simpledialog.askstring("Edit text", "Text:", initialvalue=item.get("text", ""), parent=self)
            if text is not None:
                item["text"] = text
        self.redraw()

    def duplicate_selected(self):
        item = self.get_item(self.selected_id)
        if not item:
            return
        new_item = dict(item)
        new_item["id"] = self._new_id()
        if new_item["type"] == "box":
            new_item["x1"] += 24
            new_item["x2"] += 24
            new_item["y1"] += 24
            new_item["y2"] += 24
        elif new_item["type"] == "text":
            new_item["x"] += 24
            new_item["y"] += 24
        elif new_item["type"] == "line":
            new_item["x1"] += 24
            new_item["x2"] += 24
            new_item["y1"] += 24
            new_item["y2"] += 24
        elif new_item["type"] == "freehand":
            new_item["points"] = [(px + 24, py + 24) for px, py in new_item.get("points", [])]
        self.items.append(new_item)
        self.selected_id = new_item["id"]
        self.redraw()

    def delete_selected(self):
        if self.selected_id is None:
            return
        self.items = [item for item in self.items if item["id"] != self.selected_id]
        self.selected_id = None
        self.redraw()

    def reorder_selected(self, direction: int):
        if self.selected_id is None:
            return
        idx = next((i for i, item in enumerate(self.items) if item["id"] == self.selected_id), None)
        if idx is None:
            return
        new_idx = max(0, min(len(self.items) - 1, idx + direction))
        self.items[idx], self.items[new_idx] = self.items[new_idx], self.items[idx]
        self.redraw()

    # --------------------------------------------------------
    # File actions
    # --------------------------------------------------------

    def new_sketch(self):
        if self.items and not messagebox.askyesno("New sketch", "Clear current sketch?"):
            return
        self.items.clear()
        self.selected_id = None
        self.next_id = 1
        self.current_file = None
        self.file_label.config(text="Untitled Sketch")
        self.redraw()

    def save_json(self):
        path = self.current_file
        if path is None:
            name = f"quant_sketch_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path_str = filedialog.asksaveasfilename(
                defaultextension=".json",
                initialfile=name,
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if not path_str:
                return
            path = Path(path_str)

        data = {
            "app": "QUANT Sketch Board",
            "version": "1.0.0",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": self.items,
            "next_id": self.next_id,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.current_file = path
        self.file_label.config(text=path.name)
        self.set_status(f"Saved: {path}")

    def load_json(self):
        path_str = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.items = data.get("items", [])
            self.next_id = int(data.get("next_id", len(self.items) + 1))
            self.selected_id = None
            self.current_file = path
            self.file_label.config(text=path.name)
            self.redraw()
        except Exception as exc:
            messagebox.showerror("Load error", str(exc))

    def export_png(self):
        if not PIL_AVAILABLE:
            messagebox.showerror("Missing dependency", "Install Pillow first:\n\npip install pillow")
            return

        path_str = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=f"quant_sketch_{time.strftime('%Y%m%d_%H%M%S')}.png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if not path_str:
            return

        width = max(self.canvas.winfo_width(), 1200)
        height = max(self.canvas.winfo_height(), 760)
        img = Image.new("RGB", (width, height), CANVAS_BG)
        draw = ImageDraw.Draw(img)

        if self.show_grid.get():
            for x in range(0, width, 24):
                draw.line((x, 0, x, height), fill=GRID)
            for y in range(0, height, 24):
                draw.line((0, y, width, y), fill=GRID)
            for x in range(0, width, 120):
                draw.line((x, 0, x, height), fill="#1B2330")
            for y in range(0, height, 120):
                draw.line((0, y, width, y), fill="#1B2330")

        try:
            font_title = ImageFont.truetype("arial.ttf", 16)
            font_text = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()

        for item in self.items:
            if item["type"] == "box":
                draw.rectangle(
                    (item["x1"], item["y1"], item["x2"], item["y2"]),
                    fill=item.get("fill", DEFAULT_BOX_FILL),
                    outline=item.get("outline", DEFAULT_BOX_OUTLINE),
                    width=1,
                )
                draw.text((item["x1"] + 12, item["y1"] + 12), item.get("label", ""), fill=TEXT, font=font_title)
            elif item["type"] == "text":
                draw.text((item["x"], item["y"]), item.get("text", ""), fill=item.get("color", DEFAULT_TEXT_COLOR), font=font_title)
            elif item["type"] == "line":
                draw.line((item["x1"], item["y1"], item["x2"], item["y2"]), fill=item.get("color", BORDER), width=2)
            elif item["type"] == "freehand":
                pts = item.get("points", [])
                if len(pts) >= 2:
                    draw.line(pts, fill=item.get("color", ACCENT), width=int(item.get("width", 2)))

        img.save(path_str)
        self.set_status(f"Exported PNG: {path_str}")

    def copy_prompt(self):
        description = self.build_prompt_text()
        self.clipboard_clear()
        self.clipboard_append(description)
        self.update()
        self.set_status("Copied UI prompt to clipboard")

    def build_prompt_text(self) -> str:
        lines = [
            "Build this dashboard/UI based on my sketch.",
            "",
            "Rules:",
            "- Keep the exact layout structure from the sketch.",
            "- Do not add new panels unless explicitly needed.",
            "- First create the backend/functionality skeleton.",
            "- Then apply visual polish.",
            "- Style: clean dark professional desktop software, like VS Code / JetBrains / Obsidian.",
            "- No Bloomberg, no SaaS, no TradingView, no mobile design.",
            "",
            "Sketch elements:",
        ]

        for item in self.items:
            if item["type"] == "box":
                lines.append(f"- BOX '{item.get('label', '')}' at ({item['x1']},{item['y1']}) size {item['x2']-item['x1']}x{item['y2']-item['y1']}")
            elif item["type"] == "text":
                lines.append(f"- TEXT '{item.get('text', '')}' at ({item['x']},{item['y']})")
            elif item["type"] == "line":
                lines.append(f"- LINE from ({item['x1']},{item['y1']}) to ({item['x2']},{item['y2']})")
            elif item["type"] == "freehand":
                pts = item.get("points", [])
                lines.append(f"- FREEHAND sketch stroke with {len(pts)} points")
        return "\n".join(lines)

    # --------------------------------------------------------
    # Status / close
    # --------------------------------------------------------

    def set_status(self, text):
        self.status.config(text=text)

    def close_app(self):
        try:
            self.destroy()
        except Exception:
            pass


# ============================================================
# Building Block API
# ============================================================

def build_panel(parent, **kwargs):
    return QuantSketchBoard(parent, **kwargs)


def create_panel(parent, **kwargs):
    return build_panel(parent, **kwargs)


DashboardPanel = QuantSketchBoard


def main():
    root = tk.Tk()
    root.title("QUANT WORKSPACE - Skizze")
    root.configure(bg=APP_BG)
    root.geometry("1320x860")
    root.minsize(1100, 720)
    root.protocol("WM_DELETE_WINDOW", root.destroy)

    app = build_panel(root)
    app.pack(fill="both", expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()

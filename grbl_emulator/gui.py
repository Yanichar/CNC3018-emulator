"""tkinter GUI for the CNC3018 GRBL emulator.

The window shows the machine state, a digital read-out, the virtual work table
with a moving cross carriage (which spins together with the simulated spindle)
and a serial console.  It can also be used without a port: commands typed into
the console are executed by an internal engine, so the animation can be tried
out immediately.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from .engine import Engine, Snapshot
from .pcb import (
    Board,
    Hole,
    PcbError,
    discover_sources,
    guess_outline_layer,
    parse_excellon_file,
    parse_gerber_contours,
)
from .settings import SETTINGS
from .transport import SerialTransport
from .workfield import (
    FieldGeometry,
    PX_PER_MM,
    center_offset,
    default_output_path,
    render_workfield,
)

# --------------------------------------------------------------------------- theme
BG = "#0b0f14"
PANEL = "#121922"
PANEL2 = "#0d131b"
BORDER = "#263241"
TEXT = "#d7e0ea"
MUTED = "#7d8b9c"
ACCENT = "#4cc9f0"
OK = "#3ddc97"
WARN = "#ffb703"
ERR = "#ef476f"
VIOLET = "#c77dff"

# Light aluminium of the machine table (the bed is always shown as aluminium).
ALU_LIGHT = "#d7dbdf"
ALU_PANEL = "#c2c8ce"
ALU_DARK = "#a9b0b7"
ALU_EDGE = "#8d959d"
ALU_LINE = "#b7bdc4"
ALU_TEXT = "#4f5861"
ALU_MUTED = "#6f7881"
BOARD_FILL = "#c87533"
BOARD_EDGE = "#8a4c1e"
BOARD_SHADOW = "#adb5bc"
BOARD_HOLE = "#000000"

STATE_COLORS = {
    "Idle": OK,
    "Run": ACCENT,
    "Jog": "#7bdff2",
    "Hold": WARN,
    "Alarm": ERR,
    "Door": "#f4a261",
    "Check": "#9b8cff",
    "Home": VIOLET,
    "Sleep": "#6c757d",
}

STATE_HINTS = {
    "Idle": "ready",
    "Run": "executing motion",
    "Jog": "jogging",
    "Hold": "feed hold / pause",
    "Alarm": "commands locked",
    "Door": "safety door open",
    "Check": "G-code check mode",
    "Home": "homing cycle",
    "Sleep": "sleeping (reset to wake)",
}

FONT_UI = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_TINY = ("Segoe UI", 8)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_STATE = ("Segoe UI", 24, "bold")
FONT_MONO = ("Consolas", 9)
FONT_DRO = ("Consolas", 21, "bold")
FONT_DRO_SMALL = ("Consolas", 9)


def available_ports() -> list[str]:
    try:
        from serial.tools import list_ports

        return [port.device for port in list_ports.comports()]
    except Exception:
        return []


class Host:
    """Owns the engine and (optionally) the serial port."""

    def __init__(self, engine: Engine, enqueue_log) -> None:
        self.engine = engine
        self.transport: SerialTransport | None = None
        self._enqueue = enqueue_log

    @property
    def connected(self) -> bool:
        return self.transport is not None and self.transport.connected

    def connect(self, port: str, baudrate: int) -> None:
        transport = SerialTransport(self.engine, on_message=self._enqueue)
        transport.open(port, baudrate)
        self.transport = transport

    def disconnect(self) -> None:
        transport = self.transport
        self.transport = None
        if transport is not None:
            transport.close()

    def snapshot(self) -> Snapshot:
        if self.connected and self.transport is not None:
            return self.transport.snapshot
        return self.engine.snapshot(connected=False)

    def local_tick(self, dt: float) -> None:
        for message in self.engine.tick(dt):
            self._enqueue(self._classify(message), message)

    def send_line(self, line: str) -> None:
        if self.connected and self.transport is not None:
            self.transport.send_line(line)
            return
        self._enqueue("tx", line)
        for message in self.engine.feed_bytes(line.encode("utf-8") + b"\n"):
            self._enqueue(self._classify(message), message)

    def send_realtime(self, byte: int) -> None:
        if self.connected and self.transport is not None:
            self.transport.send_realtime(byte)
            return
        self._enqueue("rt", f"0x{byte:02X}")
        for message in self.engine.feed_bytes(bytes([byte])):
            self._enqueue(self._classify(message), message)

    @staticmethod
    def _classify(message: str) -> str:
        if message.startswith("error") or message.startswith("ALARM"):
            return "err"
        if message.startswith("["):
            return "msg"
        if message.startswith("<"):
            return "status"
        return "rx"


def _polygon_area(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _point_in_polygon(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    inside = False
    count = len(points)
    for index in range(count):
        x1, y1 = points[index]
        x2, y2 = points[(index + 1) % count]
        if (y1 > y) != (y2 > y):
            cross_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if cross_x > x:
                inside = not inside
    return inside


def _shade(color: str, factor: float) -> str:
    """Lighten (factor > 1) or darken (factor < 1) a ``#rrggbb`` colour."""

    def clamp(value: float) -> int:
        return max(0, min(255, int(round(value))))

    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"#{clamp(red * factor):02x}{clamp(green * factor):02x}{clamp(blue * factor):02x}"


def _span_intersections(
    loops: list[list[tuple[float, float]]], y: float
) -> list[tuple[float, float]]:
    """Even-odd scanline spans of screen-space polygons at *y*."""

    crossings: list[float] = []
    for loop in loops:
        if len(loop) < 2:
            continue
        previous = loop[-1]
        for point in loop:
            x1, y1 = previous
            x2, y2 = point
            if (y1 <= y < y2) or (y2 <= y < y1):
                crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
            previous = point
    crossings.sort()
    spans: list[tuple[float, float]] = []
    for index in range(0, len(crossings) - 1, 2):
        left, right = crossings[index], crossings[index + 1]
        if right - left > 0.5:
            spans.append((left, right))
    return spans


# --------------------------------------------------------------------------- machine view
class MachineView(tk.Canvas):
    """The virtual work table: a fixed bed and a moving (and spinning) cross."""

    def __init__(self, master, **kwargs) -> None:
        super().__init__(
            master,
            bg=ALU_DARK,
            highlightthickness=0,
            bd=0,
            **kwargs,
        )
        self.travel = (300.0, 180.0, 45.0)
        self.mpos = (0.0, 0.0, 0.0)
        self.angle = 0.0
        self.state = "Idle"
        self.rpm = 0.0
        self.board: Board | None = None
        self._board_offset = (0.0, 0.0)
        self._board_items: list[int] = []
        self.trail: deque[tuple[float, float]] = deque(maxlen=900)
        self._geometry = (40.0, 100.0, 1.0)
        self._cross_ids: list[int] = []
        self._trail_id = self.create_line(
            0, 0, 1, 1, fill="#1f6f8b", width=1, state="hidden", smooth=True
        )
        self._build_cross_items()
        self.bind("<Configure>", lambda event: self._rebuild())
        self._rebuild()

    # ---------------------------------------------------------------- geometry
    def _build_cross_items(self) -> None:
        self._cross_arms = [
            self.create_line(0, 0, 1, 1, fill="#2b3540", width=2, capstyle="round")
            for _ in range(2)
        ]
        self._cross_index = self.create_line(0, 0, 1, 1, fill="#c2410c", width=2, capstyle="round")
        self._cross_ring = self.create_oval(0, 0, 1, 1, outline="#7b848d", width=1)
        self._cross_hub = self.create_oval(0, 0, 1, 1, outline="#5f6871", width=1, fill="#eef1f3")
        self._cross_dot = self.create_oval(0, 0, 1, 1, outline="", fill="#c2410c")
        self._labels: list[int] = []

    def _compute_geometry(self) -> tuple[float, float, float]:
        width = max(self.winfo_width(), 100)
        height = max(self.winfo_height(), 100)
        travel_x = max(self.travel[0], 1.0)
        travel_y = max(self.travel[1], 1.0)
        left, right = 58.0, 84.0
        top, bottom = 46.0, 52.0
        usable_w = max(width - left - right, 40.0)
        usable_h = max(height - top - bottom, 40.0)
        scale = min(usable_w / travel_x, usable_h / travel_y)
        x0 = left + (usable_w - travel_x * scale) / 2.0
        y0 = top + (usable_h + travel_y * scale) / 2.0
        return x0, y0, scale

    def _to_screen(self, x: float, y: float) -> tuple[float, float]:
        x0, y0, scale = self._geometry
        return x0 + x * scale, y0 - y * scale

    # ---------------------------------------------------------------- drawing
    def _rebuild(self) -> None:
        self._geometry = self._compute_geometry()
        x0, y0, scale = self._geometry
        tx, ty, tz = self.travel
        width = max(self.winfo_width(), 100)
        height = max(self.winfo_height(), 100)

        for item in self._labels:
            self.delete(item)
        self._labels.clear()

        # bed: a light aluminium plate, empty until a board is loaded
        x1 = x0 + tx * scale
        y1 = y0 - ty * scale
        self._bed_id = getattr(self, "_bed_id", None)
        if self._bed_id is None:
            self._bed_id = self.create_rectangle(
                x0, y1, x1, y0, outline=ALU_EDGE, fill=ALU_LIGHT, width=2
            )
        else:
            self.coords(self._bed_id, x0, y1, x1, y0)
            self.itemconfigure(self._bed_id, fill=ALU_LIGHT, outline=ALU_EDGE)
            self.tag_raise(self._bed_id)

        # brushed aluminium: fine horizontal streaks
        streak_colors = ("#dadde0", "#d2d7db", "#d8dcdf", "#cfd4d9", "#d5dade")
        streak = 0
        pixel_y = int(round(y1)) + 2
        while pixel_y < y0 - 2:
            self._labels.append(
                self.create_line(
                    x0 + 2,
                    pixel_y,
                    x1 - 2,
                    pixel_y,
                    fill=streak_colors[streak % len(streak_colors)],
                    width=1,
                )
            )
            pixel_y += 3
            streak += 1

        # T-slot grooves of the aluminium bed, running along X
        for groove_y in (30.0, 90.0, 150.0):
            if groove_y > ty:
                continue
            _, groove_top = self._to_screen(0.0, groove_y + 4.0)
            _, groove_bottom = self._to_screen(0.0, groove_y - 4.0)
            self._labels.append(
                self.create_rectangle(
                    x0 + 2, groove_top, x1 - 2, groove_bottom, fill="#b8bec5", outline=""
                )
            )
            self._labels.append(
                self.create_line(x0 + 2, groove_top, x1 - 2, groove_top, fill="#9aa2aa", width=1)
            )
            self._labels.append(
                self.create_line(
                    x0 + 2, groove_bottom, x1 - 2, groove_bottom, fill="#d9dde1", width=1
                )
            )

        # grid
        step = 20.0 if tx >= 120 else 10.0
        gx = 0.0
        while gx <= tx + 1e-6:
            sx, _ = self._to_screen(gx, 0)
            self._labels.append(
                self.create_line(sx, y1, sx, y0, fill=ALU_LINE, width=1)
            )
            gx += step
        gy = 0.0
        while gy <= ty + 1e-6:
            _, sy = self._to_screen(0, gy)
            self._labels.append(
                self.create_line(x0, sy, x1, sy, fill=ALU_LINE, width=1)
            )
            gy += step

        # ruler labels
        label_step = 50.0 if tx >= 250 else 25.0
        gx = 0.0
        while gx <= tx + 1e-6:
            sx, _ = self._to_screen(gx, 0)
            self._labels.append(
                self.create_text(sx, y0 + 12, text=f"{gx:g}", fill=ALU_MUTED, font=FONT_TINY)
            )
            gx += label_step
        gy = 0.0
        while gy <= ty + 1e-6:
            _, sy = self._to_screen(0, gy)
            self._labels.append(
                self.create_text(x0 - 16, sy, text=f"{gy:g}", fill=ALU_MUTED, font=FONT_TINY, anchor="e")
            )
            gy += label_step

        # axes at the machine zero corner
        self._labels.append(
            self.create_line(x0, y0, x0 + 42, y0, fill="#c0392b", width=2, arrow="last")
        )
        self._labels.append(
            self.create_line(x0, y0, x0, y0 - 42, fill="#2e7d32", width=2, arrow="last")
        )
        self._labels.append(self.create_text(x0 + 50, y0 + 2, text="X", fill="#c0392b", font=FONT_BOLD))
        self._labels.append(self.create_text(x0 - 10, y0 - 50, text="Y", fill="#2e7d32", font=FONT_BOLD))
        self._labels.append(
            self.create_text(
                (x0 + x1) / 2,
                22,
                text=f"CNC 3018 work table · {tx:g} × {ty:g} × {tz:g} mm",
                fill=ALU_TEXT,
                font=FONT_SMALL,
            )
        )

        # PCB (drawn above the table, below the tool and the Z gauge)
        for item in self._board_items:
            self.delete(item)
        self._board_items.clear()
        self._draw_board()

        # Z gauge
        gauge_x = width - 44
        z_top = y0 - tz * scale
        z_bottom = y0
        self._labels.append(
            self.create_line(gauge_x, z_top, gauge_x, z_bottom, fill=ALU_EDGE, width=6)
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            ty_px = z_bottom - (z_bottom - z_top) * fraction
            self._labels.append(
                self.create_line(gauge_x - 8, ty_px, gauge_x + 8, ty_px, fill="#8f99a2", width=1)
            )
        self._labels.append(
            self.create_text(gauge_x + 2, z_top - 14, text="Z", fill=ALU_TEXT, font=FONT_SMALL)
        )
        self._z_top = z_top
        self._z_bottom = z_bottom
        self._z_gauge_x = gauge_x
        self._z_marker = getattr(self, "_z_marker", None)
        if self._z_marker is None:
            self._z_marker = self.create_line(gauge_x - 11, z_bottom, gauge_x + 11, z_bottom, fill="#0e7490", width=3)
        self._z_text = getattr(self, "_z_text", None)
        if self._z_text is None:
            self._z_text = self.create_text(gauge_x, z_bottom + 18, text="Z 0.000", fill="#0e7490", font=FONT_TINY)
        self.coords(self._z_text, gauge_x, z_bottom + 18)
        self._update_trail()
        self._raise_tool()

    def _raise_tool(self) -> None:
        for item in (
            self._cross_arms[0],
            self._cross_arms[1],
            self._cross_index,
            self._cross_ring,
            self._cross_hub,
            self._cross_dot,
            self._z_marker,
            self._z_text,
        ):
            self.tag_raise(item)

    # ---------------------------------------------------------------- board
    def set_board(self, board: Board | None) -> None:
        """Show *board* on the table (``None`` clears the table)."""

        self.board = board
        self._board_offset = center_offset(board) if board is not None else (0.0, 0.0)
        self._rebuild()

    def clear_board(self) -> None:
        self.set_board(None)

    def _draw_board(self) -> None:
        if self.board is None:
            return
        offset_x, offset_y = self._board_offset
        loops = [
            [self._to_screen(x + offset_x, y + offset_y) for x, y in loop]
            for loop in self.board.loops
        ]
        loops = [loop for loop in loops if len(loop) >= 3]
        if not loops:
            return

        fill_order = sorted(range(len(loops)), key=lambda i: -abs(_polygon_area(loops[i])))
        outer: list[list[tuple[float, float]]] = []
        cutouts: list[list[tuple[float, float]]] = []
        for index in fill_order:
            loop = loops[index]
            if any(_point_in_polygon(loop[0][0], loop[0][1], other) for other in outer):
                cutouts.append(loop)
            else:
                outer.append(loop)

        # Soft shadow on the aluminium: light comes from the top-left.
        for loop in outer:
            for dx, dy, color in ((6.0, 7.0, "#b9c0c6"), (3.0, 4.0, "#a9b1b9")):
                self._board_items.append(
                    self.create_polygon(
                        [
                            coordinate
                            for point in loop
                            for coordinate in (point[0] + dx, point[1] + dy)
                        ],
                        fill=color,
                        outline="",
                        width=0,
                    )
                )

        # Copper foil base and cut-outs.
        for loop in outer:
            self._board_items.append(
                self.create_polygon(
                    [coordinate for point in loop for coordinate in point],
                    fill=BOARD_FILL,
                    outline="",
                    width=0,
                )
            )
        for loop in cutouts:
            self._board_items.append(
                self.create_polygon(
                    [coordinate for point in loop for coordinate in point],
                    fill=ALU_PANEL,
                    outline="",
                    width=0,
                )
            )

        # Brushed copper: a gentle sheen with fine horizontal grain.
        top = min(point[1] for loop in loops for point in loop)
        bottom = max(point[1] for loop in loops for point in loop)
        height = max(bottom - top, 1.0)
        for pixel_y in range(int(math.floor(top)), int(math.ceil(bottom)) + 1):
            line_y = pixel_y + 0.5
            factor = 1.10 - 0.20 * ((line_y - top) / height)
            factor += 0.015 * math.sin(pixel_y * 1.7) + 0.008 * math.sin(pixel_y * 5.3)
            for left, right in _span_intersections(loops, line_y):
                span = right - left
                if span <= 0.0:
                    continue
                for part in range(3):
                    part_left = left + span * part / 3.0
                    part_right = left + span * (part + 1) / 3.0
                    part_factor = factor + (0.014 if part == 0 else -0.014 if part == 2 else 0.0)
                    self._board_items.append(
                        self.create_line(
                            part_left,
                            line_y,
                            part_right,
                            line_y,
                            fill=_shade(BOARD_FILL, part_factor),
                            width=1,
                        )
                    )

        # Holes stay pure black, as requested.
        for hole in self.board.holes:
            center_x, center_y = self._to_screen(hole.x + offset_x, hole.y + offset_y)
            radius = max(hole.diameter / 2.0 * self._geometry[2], 0.6)
            self._board_items.append(
                self.create_oval(
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                    fill=BOARD_HOLE,
                    outline="",
                )
            )

        # Milled edge of the board.
        for loop in loops:
            coordinates = [coordinate for point in loop for coordinate in point]
            self._board_items.append(
                self.create_line(
                    *coordinates,
                    coordinates[0],
                    coordinates[1],
                    fill=BOARD_EDGE,
                    width=2,
                )
            )

    def _update_trail(self) -> None:
        if len(self.trail) < 2:
            self.itemconfigure(self._trail_id, state="hidden")
            return
        coords: list[float] = []
        for x, y in self.trail:
            sx, sy = self._to_screen(x, y)
            coords.extend((sx, sy))
        self.coords(self._trail_id, *coords)
        self.itemconfigure(self._trail_id, state="normal")
        self.tag_raise(self._trail_id)
        self._raise_tool()

    def _update_cross(self) -> None:
        cx, cy = self._to_screen(self.mpos[0], self.mpos[1])
        arm = 22.0
        rad = math.radians(self.angle)
        dx = math.cos(rad) * arm
        dy = math.sin(rad) * arm
        px = -dy
        py = dx
        self.coords(self._cross_arms[0], cx - dx, cy + dy, cx + dx, cy - dy)
        self.coords(self._cross_arms[1], cx - px, cy + py, cx + px, cy - py)
        # index arm (shows the rotation direction even at a glance)
        self.coords(self._cross_index, cx, cy, cx + dx, cy - dy)
        self.coords(self._cross_ring, cx - 26, cy - 26, cx + 26, cy + 26)
        self.coords(self._cross_hub, cx - 6, cy - 6, cx + 6, cy + 6)
        dot_x = cx + dx * 1.35
        dot_y = cy - dy * 1.35
        self.coords(self._cross_dot, dot_x - 3, dot_y - 3, dot_x + 3, dot_y + 3)

        z = self.mpos[2]
        tz = max(self.travel[2], 1e-6)
        fraction = min(max(z / tz, 0.0), 1.0)
        zy = self._z_bottom - (self._z_bottom - self._z_top) * fraction
        self.coords(self._z_marker, self._z_gauge_x - 11, zy, self._z_gauge_x + 11, zy)
        self.itemconfigure(self._z_text, text=f"Z {z:7.3f}")

    def update_snapshot(self, snapshot: Snapshot, dt: float) -> None:
        travel = tuple(float(v) for v in snapshot.travel)
        if travel != self.travel:
            self.travel = travel
            self._rebuild()
        self.mpos = snapshot.mpos
        self.state = snapshot.state
        self.rpm = snapshot.rpm
        if snapshot.rpm > 0.0:
            self.angle = (self.angle + snapshot.rpm / 10_000.0 * 360.0 * dt) % 360.0
        self._update_cross()

        if snapshot.state in ("Run", "Jog", "Home"):
            if not self.trail:
                self.trail.append((snapshot.mpos[0], snapshot.mpos[1]))
            else:
                last = self.trail[-1]
                if math.hypot(snapshot.mpos[0] - last[0], snapshot.mpos[1] - last[1]) > 0.15:
                    self.trail.append((snapshot.mpos[0], snapshot.mpos[1]))
                    self._update_trail()

    def clear_trail(self) -> None:
        self.trail.clear()
        self.itemconfigure(self._trail_id, state="hidden")


# --------------------------------------------------------------------------- settings dialog
class SettingsDialog(tk.Toplevel):
    def __init__(self, app: "App") -> None:
        super().__init__(app)
        self.app = app
        self.title("GRBL settings — EEPROM")
        self.geometry("780x560")
        self.configure(bg=BG)
        self.transient(app)
        self.resizable(True, True)

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x")
        tk.Label(
            header,
            text="$$ settings",
            bg=PANEL,
            fg=TEXT,
            font=FONT_TITLE,
        ).pack(side="left", padx=14, pady=10)
        tk.Label(
            header,
            text="Changes are written with $<id>=<value> and stored in the emulated EEPROM",
            bg=PANEL,
            fg=MUTED,
            font=FONT_SMALL,
        ).pack(side="left", padx=6)

        tree_frame = tk.Frame(self, bg=BG)
        tree_frame.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        columns = ("id", "value", "description")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")
        self.tree.heading("id", text="ID")
        self.tree.heading("value", text="Value")
        self.tree.heading("description", text="Description")
        self.tree.column("id", width=56, anchor="center", stretch=False)
        self.tree.column("value", width=110, anchor="e", stretch=False)
        self.tree.column("description", width=520, anchor="w")
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        edit = tk.Frame(self, bg=BG)
        edit.pack(fill="x", padx=12, pady=10)
        tk.Label(edit, text="$", bg=BG, fg=MUTED, font=FONT_MONO).pack(side="left")
        self.id_var = tk.StringVar()
        tk.Entry(
            edit,
            textvariable=self.id_var,
            width=5,
            bg=PANEL2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=FONT_MONO,
        ).pack(side="left", padx=(2, 4), ipady=4)
        tk.Label(edit, text="=", bg=BG, fg=MUTED, font=FONT_MONO).pack(side="left")
        self.value_var = tk.StringVar()
        value_entry = tk.Entry(
            edit,
            textvariable=self.value_var,
            width=14,
            bg=PANEL2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=FONT_MONO,
        )
        value_entry.pack(side="left", padx=(4, 8), ipady=4)
        value_entry.bind("<Return>", lambda event: self._apply())
        ttk.Button(edit, text="Apply", command=self._apply).pack(side="left", padx=3)
        ttk.Button(edit, text="Refresh", command=self._populate).pack(side="left", padx=3)
        ttk.Button(edit, text="Defaults ($RST=$)", command=self._restore).pack(side="left", padx=3)
        ttk.Button(edit, text="Close", command=self.destroy).pack(side="right")

        self.hint = tk.Label(
            self,
            text="Select a setting, edit the value and press Apply.  Changes take effect immediately.",
            bg=BG,
            fg=MUTED,
            font=FONT_SMALL,
        )
        self.hint.pack(fill="x", padx=14, pady=(0, 10))
        self._populate()

    def _populate(self) -> None:
        snapshot = self.app.host.snapshot()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for number, definition in SETTINGS.items():
            value = snapshot.settings.get(number, definition.default)
            self.tree.insert(
                "",
                "end",
                iid=str(number),
                values=(number, definition.format(value), definition.label),
            )

    def _on_select(self, _event=None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        item = self.tree.item(selection[0])
        sid, value, _ = item["values"]
        self.id_var.set(str(sid))
        self.value_var.set(str(value))

    def _apply(self) -> None:
        sid = self.id_var.get().strip()
        value = self.value_var.get().strip()
        if not sid.isdigit() or not value:
            messagebox.showwarning("Settings", "Enter a numeric setting id and a value.", parent=self)
            return
        self.app.send_line(f"${sid}={value}")
        self.app.after(350, self._populate)

    def _restore(self) -> None:
        if messagebox.askyesno(
            "Restore defaults",
            "Restore all GRBL settings to the CNC3018 defaults?\nThe emulator will reset.",
            parent=self,
        ):
            self.app.send_line("$RST=$")
            self.app.after(600, self._populate)


# --------------------------------------------------------------------------- main window
class App(tk.Tk):
    def __init__(
        self,
        port: str = "COM7",
        baudrate: int = 115200,
        eeprom_path: str | None = None,
        autoconnect: bool = False,
        workfield_path: str | None = None,
    ) -> None:
        super().__init__()
        self.title("CNC 3018 — GRBL 1.1 emulator")
        self.geometry("1320x1020")
        self.minsize(1100, 720)
        self.configure(bg=BG)

        self._log_queue: "queue.Queue[tuple[float, str, str]]" = queue.Queue()
        self.engine = Engine(eeprom_path=eeprom_path)
        self.host = Host(self.engine, self._enqueue_log)
        self._last_tick = time.monotonic()
        self._history: list[str] = []
        self._history_index = 0
        self._settings_dialog: SettingsDialog | None = None
        self._connected_at: float | None = None
        self._last_console_kind = ""

        # PCB / work-field state
        self._pcb_dir: Path | None = None
        self._pcb_layers: dict[str, Path] = {}
        self._pcb_holes: list[Hole] = []
        self._pcb_board: Board | None = None
        self._pcb_offset = (0.0, 0.0)
        self._workfield_path = Path(workfield_path) if workfield_path else default_output_path()
        self._workfield_token = 0
        self._workfield_results: "queue.Queue[tuple[int, Path, Exception | None]]" = queue.Queue()

        self._configure_styles()
        self._build_ui(port, baudrate)
        self._refresh_ports(initial=port)
        self._log_system("Emulator ready.  Open a port, or type G-code below to try the simulation.")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(28, self._tick)
        if autoconnect:
            self.after(250, self._connect)

    # ------------------------------------------------------------------ styles
    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, font=FONT_UI)
        style.configure("TButton", background="#1b2733", foreground=TEXT, padding=(10, 5), borderwidth=1)
        style.configure("Pad.TButton", background="#1b2733", foreground=TEXT, padding=(6, 2), borderwidth=1)
        style.map(
            "TButton",
            background=[("active", "#273949"), ("pressed", "#31465a"), ("disabled", "#141b23")],
            foreground=[("disabled", MUTED)],
        )
        style.configure("Accent.TButton", background="#0e7490", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#0f8bab"), ("pressed", "#0c627a")])
        style.configure("Danger.TButton", background="#7f1d3a", foreground="#ffffff")
        style.configure("DangerPad.TButton", background="#7f1d3a", foreground="#ffffff", padding=(6, 2), borderwidth=1)
        style.configure(
            "Action.TButton",
            background="#1b2733",
            foreground=TEXT,
            padding=(3, 2),
            borderwidth=1,
            font=FONT_TINY,
        )
        style.map("Action.TButton", background=[("active", "#273949"), ("pressed", "#31465a")])
        style.configure(
            "ResetAction.TButton",
            background="#7f1d3a",
            foreground="#ffffff",
            padding=(3, 2),
            borderwidth=1,
            font=FONT_TINY,
        )
        style.map("ResetAction.TButton", background=[("active", "#a02549")])
        style.map("Danger.TButton", background=[("active", "#a02549")])
        style.configure("TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=TEXT, arrowcolor=TEXT)
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PANEL2)],
            foreground=[("readonly", TEXT)],
            selectbackground=[("readonly", PANEL2)],
            selectforeground=[("readonly", TEXT)],
        )
        style.configure("TEntry", fieldbackground=PANEL2, foreground=TEXT, insertcolor=TEXT)
        style.configure("TCheckbutton", background=BG, foreground=MUTED)
        style.map("TCheckbutton", background=[("active", BG)], foreground=[("active", TEXT)])
        style.configure(
            "Treeview",
            background=PANEL2,
            fieldbackground=PANEL2,
            foreground=TEXT,
            rowheight=22,
            borderwidth=0,
        )
        style.map("Treeview", background=[("selected", "#1d3c52")], foreground=[("selected", TEXT)])
        style.configure("Treeview.Heading", background=PANEL, foreground=MUTED, relief="flat")
        style.configure("TSpinbox", fieldbackground=PANEL2, foreground=TEXT, arrowcolor=TEXT)
        style.configure(
            "Vertical.TScrollbar",
            background="#1b2733",
            troughcolor="#080c11",
            bordercolor=BG,
            arrowcolor=MUTED,
            lightcolor="#263241",
            darkcolor="#263241",
        )
        style.map("Vertical.TScrollbar", background=[("active", "#31465a")])
        self.option_add("*TCombobox*Listbox.background", PANEL2)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", "#1d3c52")

    # ------------------------------------------------------------------ layout
    def _card(self, parent, title: str) -> tk.Frame:
        card = tk.Frame(parent, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        tk.Label(card, text=title, bg=PANEL, fg=MUTED, font=FONT_TINY).pack(
            anchor="w", padx=12, pady=(8, 0)
        )
        return card

    def _build_ui(self, port: str, baudrate: int) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0, minsize=362)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0, minsize=176)

        self._build_header(port, baudrate)

        left = tk.Frame(self, bg=BG)
        left.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=(4, 6))
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)
        self._build_pcb_toolbar(left)
        self.machine_view = MachineView(left, width=820, height=560)
        self.machine_view.grid(row=1, column=0, sticky="nsew")
        footer = tk.Frame(left, bg=BG)
        footer.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        tk.Label(
            footer,
            text="Крест вращается вместе со шпинделем: 10 000 об/мин → 60 об/мин. "
            "Разгон/торможение шпинделя — 2 с.",
            bg=BG,
            fg=MUTED,
            font=FONT_SMALL,
        ).pack(side="left")
        ttk.Button(footer, text="Clear trail", command=self.machine_view.clear_trail).pack(side="right")

        right_container = tk.Frame(self, bg=BG)
        right_container.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=(4, 6))
        right_container.grid_rowconfigure(0, weight=1)
        right_container.grid_columnconfigure(0, weight=1)
        self.right_canvas = tk.Canvas(right_container, bg=BG, highlightthickness=0, width=350)
        right_scroll = ttk.Scrollbar(right_container, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scroll.grid(row=0, column=1, sticky="ns")
        self.right_inner = tk.Frame(self.right_canvas, bg=BG)
        self._right_window = self.right_canvas.create_window((0, 0), window=self.right_inner, anchor="nw")
        self.right_inner.bind(
            "<Configure>",
            lambda event: self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all")),
        )
        self.right_canvas.bind(
            "<Configure>",
            lambda event: self.right_canvas.itemconfigure(self._right_window, width=event.width),
        )
        self.right_canvas.configure(yscrollcommand=right_scroll.set)
        self.right_canvas.bind("<MouseWheel>", self._scroll_right)
        self.right_inner.bind("<MouseWheel>", self._scroll_right)
        self._build_status_panel(self.right_inner)

        self._build_console()
        self.bind("<F5>", lambda event: self.send_realtime(0x18))
        self.bind("<Control-l>", lambda event: self._clear_console())

    def _scroll_right(self, event) -> None:
        self.right_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    # ------------------------------------------------------------------ PCB / work field
    def _build_pcb_toolbar(self, parent: tk.Frame) -> None:
        bar = tk.Frame(parent, bg=BG)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(
            bar,
            text="Загрузить гербер + сверловка",
            style="Accent.TButton",
            command=self._choose_pcb_directory,
        ).pack(side="left")
        tk.Label(bar, text="Контур:", bg=BG, fg=MUTED, font=FONT_SMALL).pack(side="left", padx=(12, 4))
        self.layer_var = tk.StringVar()
        self.layer_combo = ttk.Combobox(
            bar, textvariable=self.layer_var, state="disabled", width=34, values=[]
        )
        self.layer_combo.pack(side="left", ipady=2)
        self.layer_combo.bind("<<ComboboxSelected>>", self._on_layer_selected)
        self.pcb_status = tk.Label(bar, text="Стол пуст", bg=BG, fg=MUTED, font=FONT_SMALL)
        self.pcb_status.pack(side="left", padx=(12, 0))

    def _choose_pcb_directory(self) -> None:
        initial = str(self._pcb_dir) if self._pcb_dir is not None else str(Path.cwd())
        directory = filedialog.askdirectory(
            parent=self,
            title="Выберите папку с Gerber-файлами и сверловкой",
            initialdir=initial,
        )
        if directory:
            self.load_pcb_directory(directory)

    def load_pcb_directory(self, directory: str | Path) -> None:
        """Load all drill files and the Gerber layer list from *directory*."""

        folder = Path(directory)
        try:
            drills, gerbers = discover_sources(folder)
        except OSError as exc:
            messagebox.showerror("Gerber", f"Не удалось прочитать папку:\n{exc}", parent=self)
            return
        if not gerbers:
            messagebox.showwarning(
                "Gerber",
                "В папке нет Gerber-файлов (.gbr, .ger, .gtl, .gbl ...).",
                parent=self,
            )
            return

        holes: list[Hole] = []
        for drill_path in drills:
            try:
                holes.extend(parse_excellon_file(drill_path))
            except PcbError as exc:
                self._log_system(f"Drill {drill_path.name}: {exc}")

        self._pcb_dir = folder
        self._pcb_layers = {path.name: path for path in gerbers}
        self._pcb_holes = holes
        names = list(self._pcb_layers)
        self.layer_combo.configure(values=names, state="readonly")
        preferred = guess_outline_layer(names, fallback=False)
        if preferred is None:
            self.layer_var.set("")
            self.pcb_status.configure(text="Выберите слой контура")
        else:
            self.layer_var.set(preferred)
        self._log_system(
            f"PCB folder {folder}: {len(names)} Gerber layers, "
            f"{len(drills)} drill files, {len(holes)} holes"
        )
        if preferred is not None:
            self._apply_pcb_layer()

    def _on_layer_selected(self, _event=None) -> None:
        self._apply_pcb_layer()

    def _apply_pcb_layer(self) -> None:
        name = self.layer_var.get()
        path = self._pcb_layers.get(name)
        if path is None:
            return
        try:
            loops = parse_gerber_contours(path)
            board = Board(loops=loops, holes=list(self._pcb_holes), source=name)
        except PcbError as exc:
            messagebox.showerror("Gerber", f"{name}:\n{exc}", parent=self)
            self.pcb_status.configure(text="Контур не загружен")
            return

        self._pcb_board = board
        self._pcb_offset = center_offset(board)
        self.machine_view.set_board(board)
        info = (
            f"{name} · {board.width:.2f} × {board.height:.2f} мм · "
            f"{len(board.holes)} отв."
        )
        self.pcb_status.configure(text=info)
        self._log_system(f"PCB loaded: {info}")
        self._render_workfield_async(board)

    def _render_workfield_async(self, board: Board) -> None:
        """Draw the 20 px/mm camera image in a worker thread."""

        self._workfield_token += 1
        token = self._workfield_token
        offset = self._pcb_offset
        path = self._workfield_path

        def worker() -> None:
            error: Exception | None = None
            try:
                render_workfield(path, board, offset)
            except Exception as exc:  # pragma: no cover - defensive
                error = exc
            self._workfield_results.put((token, path, error))

        threading.Thread(target=worker, name="workfield-render", daemon=True).start()

    def _drain_workfield_results(self) -> None:
        while True:
            try:
                token, path, error = self._workfield_results.get_nowait()
            except queue.Empty:
                return
            if token != self._workfield_token:
                continue
            if error is not None:
                self._log_system(f"Work-field PNG failed: {error}")
                continue
            geometry = FieldGeometry()
            self._log_system(
                f"Work-field PNG: {path} "
                f"({geometry.image_width}×{geometry.image_height} px, {PX_PER_MM:g} px/mm)"
            )

    def _build_header(self, port: str, baudrate: int) -> None:
        header = tk.Frame(self, bg=PANEL, height=62)
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        title_box = tk.Frame(header, bg=PANEL)
        title_box.grid(row=0, column=0, padx=(16, 8), pady=8, sticky="w")
        tk.Label(
            title_box,
            text="CNC 3018",
            bg=PANEL,
            fg=TEXT,
            font=FONT_TITLE,
        ).pack(side="left")
        tk.Label(
            title_box,
            text="GRBL 1.1 emulator",
            bg=PANEL,
            fg=ACCENT,
            font=("Segoe UI", 11),
        ).pack(side="left", padx=(8, 0), pady=(3, 0))

        self.connection_label = tk.Label(header, text="Disconnected", bg=PANEL, fg=MUTED, font=FONT_SMALL)
        self.connection_label.grid(row=0, column=1, sticky="e", padx=(0, 10))

        controls = tk.Frame(header, bg=PANEL)
        controls.grid(row=0, column=2, padx=(0, 14), pady=8, sticky="e")
        self.led = tk.Canvas(controls, width=12, height=12, bg=PANEL, highlightthickness=0)
        self.led.pack(side="left", padx=(0, 10))
        self._led_item = self.led.create_oval(2, 2, 10, 10, fill="#39424d", outline="")
        tk.Label(controls, text="Port", bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(side="left", padx=(0, 4))
        self.port_var = tk.StringVar(value=port)
        self.port_combo = ttk.Combobox(controls, textvariable=self.port_var, width=10, values=[port])
        self.port_combo.pack(side="left", ipady=2)
        self.port_combo.bind("<Return>", lambda event: self._connect())
        ttk.Button(controls, text="⟳", width=3, command=self._refresh_ports).pack(side="left", padx=3)
        tk.Label(controls, text="Baud", bg=PANEL, fg=MUTED, font=FONT_SMALL).pack(side="left", padx=(8, 4))
        self.baud_var = tk.StringVar(value=str(baudrate))
        ttk.Combobox(
            controls,
            textvariable=self.baud_var,
            width=8,
            values=["115200", "250000", "57600", "9600"],
        ).pack(side="left", ipady=2)
        self.connect_button = ttk.Button(
            controls, text="Connect", style="Accent.TButton", command=self._connect_or_disconnect
        )
        self.connect_button.pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Settings", command=self._open_settings).pack(side="left", padx=(6, 0))
        ttk.Button(controls, text="Help", command=self._show_help).pack(side="left", padx=(6, 0))

    def _build_status_panel(self, parent: tk.Frame) -> None:
        parent.grid_columnconfigure(0, weight=1)
        row = 0

        state_card = self._card(parent, "MACHINE STATE")
        state_card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        state_body = tk.Frame(state_card, bg=PANEL)
        state_body.pack(fill="x", padx=12, pady=(2, 10))
        self.state_label = tk.Label(
            state_body, text="Idle", bg=PANEL, fg=OK, font=FONT_STATE, anchor="w"
        )
        self.state_label.pack(side="left")
        self.state_hint = tk.Label(
            state_body, text="ready", bg=PANEL, fg=MUTED, font=FONT_SMALL, anchor="w"
        )
        self.state_hint.pack(side="left", padx=(10, 0), pady=(10, 0))
        self.flags_label = tk.Label(
            state_card, text="", bg=PANEL, fg=MUTED, font=FONT_TINY, anchor="w"
        )
        self.flags_label.pack(fill="x", padx=12)
        self.program_label = tk.Label(
            state_card, text="queue 0   ·   Grbl 1.1f", bg=PANEL, fg="#5c6b7c", font=FONT_TINY, anchor="w"
        )
        self.program_label.pack(fill="x", padx=12, pady=(1, 8))
        row += 1

        dro_card = self._card(parent, "POSITION  ·  machine / work")
        dro_card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        dro_body = tk.Frame(dro_card, bg=PANEL)
        dro_body.pack(fill="x", padx=12, pady=(4, 10))
        self.dro_vars: dict[str, tk.StringVar] = {}
        self.wpos_vars: dict[str, tk.StringVar] = {}
        for index, axis in enumerate(("X", "Y", "Z")):
            dro_body.grid_columnconfigure(1, weight=1)
            tk.Label(dro_body, text=axis, bg=PANEL, fg=ACCENT, font=("Consolas", 14, "bold")).grid(
                row=index, column=0, sticky="w", padx=(0, 10)
            )
            var = tk.StringVar(value="0.000")
            self.dro_vars[axis] = var
            tk.Label(dro_body, textvariable=var, bg=PANEL, fg=TEXT, font=FONT_DRO, anchor="e").grid(
                row=index, column=1, sticky="e"
            )
            wvar = tk.StringVar(value="0.000")
            self.wpos_vars[axis] = wvar
            tk.Label(dro_body, textvariable=wvar, bg=PANEL, fg=MUTED, font=FONT_DRO_SMALL, anchor="e").grid(
                row=index, column=2, sticky="e", padx=(14, 0)
            )
        tk.Label(dro_body, text="mm", bg=PANEL, fg=MUTED, font=FONT_TINY).grid(
            row=0, column=3, sticky="ne", padx=(4, 0)
        )
        row += 1

        feed_card = self._card(parent, "FEED  ·  SPINDLE")
        feed_card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        feed_body = tk.Frame(feed_card, bg=PANEL)
        feed_body.pack(fill="x", padx=12, pady=(4, 10))
        self.feed_var = tk.StringVar(value="0 mm/min")
        self.spindle_var = tk.StringVar(value="0 / 0 RPM")
        self.override_var = tk.StringVar(value="Ov 100 / 100 / 100 %")
        for index, var in enumerate((self.feed_var, self.spindle_var)):
            tk.Label(feed_body, textvariable=var, bg=PANEL, fg=TEXT, font=("Consolas", 12), anchor="w").grid(
                row=index, column=0, sticky="w"
            )
        self.rpm_canvas = tk.Canvas(feed_body, height=6, bg=PANEL, highlightthickness=0)
        self.rpm_canvas.grid(row=2, column=0, sticky="ew", pady=(6, 2))
        feed_body.grid_columnconfigure(0, weight=1)
        self.rpm_bar = self.rpm_canvas.create_rectangle(0, 0, 0, 6, fill=OK, outline="")
        self.rpm_canvas.bind("<Configure>", lambda event: self._draw_rpm_bar())
        tk.Label(feed_body, textvariable=self.override_var, bg=PANEL, fg=MUTED, font=FONT_TINY, anchor="w").grid(
            row=3, column=0, sticky="w"
        )
        row += 1

        jog_card = self._card(parent, "JOG  ·  /  MANUAL CONTROL")
        jog_card.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        jog_body = tk.Frame(jog_card, bg=PANEL)
        jog_body.pack(fill="x", padx=12, pady=(4, 10))

        options = tk.Frame(jog_body, bg=PANEL)
        options.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        tk.Label(options, text="Step", bg=PANEL, fg=MUTED, font=FONT_TINY).pack(side="left")
        self.step_var = tk.StringVar(value="10")
        ttk.Combobox(
            options, textvariable=self.step_var, width=5, values=["0.1", "1", "10", "50"]
        ).pack(side="left", padx=(4, 10))
        tk.Label(options, text="Feed", bg=PANEL, fg=MUTED, font=FONT_TINY).pack(side="left")
        self.jog_feed_var = tk.StringVar(value="500")
        ttk.Spinbox(
            options, textvariable=self.jog_feed_var, width=6, from_=10, to=5000, increment=50
        ).pack(side="left", padx=(4, 0))

        pad = tk.Frame(jog_body, bg=PANEL)
        pad.grid(row=1, column=0, columnspan=3)
        self._jog_button(pad, "Y+", "Y", +1, row=0, column=1)
        self._jog_button(pad, "X−", "X", -1, row=1, column=0)
        self._jog_button(pad, "Y−", "Y", -1, row=2, column=1)
        self._jog_button(pad, "X+", "X", +1, row=1, column=2)
        self._jog_button(pad, "Z+", "Z", +1, row=0, column=2)
        self._jog_button(pad, "Z−", "Z", -1, row=2, column=2)
        ttk.Button(pad, text="XY 0", width=6, style="Pad.TButton", command=lambda: self.send_line("G90 G0 X0 Y0")).grid(
            row=1, column=1, padx=2, pady=2
        )

        actions = tk.Frame(jog_card, bg=PANEL)
        actions.pack(fill="x", padx=12, pady=(0, 10))
        for label, command, danger in (
            ("Reset", lambda: self.send_realtime(0x18), True),
            ("$X", lambda: self.send_line("$X"), False),
            ("$H", lambda: self.send_line("$H"), False),
            ("Hold", lambda: self.send_realtime(0x21), False),
            ("Run", lambda: self.send_realtime(0x7E), False),
        ):
            self._action_button(actions, label, command, danger).pack(
                side="left", expand=True, fill="x", padx=2
            )

    def _action_button(self, parent, text: str, command, danger: bool = False) -> tk.Button:
        background = "#7f1d3a" if danger else "#1b2733"
        active = "#a02549" if danger else "#273949"
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg="#ffffff" if danger else TEXT,
            activebackground=active,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=FONT_TINY,
            width=6,
            padx=2,
            pady=3,
            cursor="hand2",
        )

    def _jog_button(self, parent, text: str, axis: str, direction: int, row: int, column: int) -> None:
        ttk.Button(
            parent,
            text=text,
            width=6,
            style="Pad.TButton",
            command=lambda: self.jog_axis(axis, direction),
        ).grid(row=row, column=column, padx=2, pady=2)

    def _build_console(self) -> None:
        frame = tk.Frame(self, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=12, pady=(0, 12))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        header = tk.Frame(frame, bg=PANEL)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))
        tk.Label(header, text="SERIAL CONSOLE", bg=PANEL, fg=MUTED, font=FONT_TINY).pack(side="left")
        self.autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(header, text="Autoscroll", variable=self.autoscroll_var).pack(side="right", padx=(8, 0))
        ttk.Button(header, text="Clear", command=self._clear_console).pack(side="right", padx=4)
        self.console_count = tk.Label(header, text="", bg=PANEL, fg=MUTED, font=FONT_TINY)
        self.console_count.pack(side="right", padx=8)

        self.console = ScrolledText(
            frame,
            height=7,
            bg="#06090d",
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            font=FONT_MONO,
            wrap="none",
            state="disabled",
        )
        self.console.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        self.console.tag_configure("tx", foreground=ACCENT)
        self.console.tag_configure("rx", foreground="#c9d1d9")
        self.console.tag_configure("rt", foreground="#8b949e")
        self.console.tag_configure("status", foreground="#5f7d94")
        self.console.tag_configure("msg", foreground=OK)
        self.console.tag_configure("err", foreground=ERR)
        self.console.tag_configure("sys", foreground="#8b949e")

        quick = tk.Frame(frame, bg=PANEL)
        quick.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 4))
        for label, command in (
            ("?", lambda: self.send_realtime(0x3F)),
            ("$$", lambda: self.send_line("$$")),
            ("$G", lambda: self.send_line("$G")),
            ("$#", lambda: self.send_line("$#")),
            ("$I", lambda: self.send_line("$I")),
            ("M3 S10000", lambda: self.send_line("M3 S10000")),
            ("M5", lambda: self.send_line("M5")),
            ("G0 X0 Y0", lambda: self.send_line("G90 G0 X0 Y0")),
        ):
            ttk.Button(quick, text=label, command=command).pack(side="left", padx=(0, 4))

        entry_row = tk.Frame(frame, bg=PANEL)
        entry_row.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 8))
        entry_row.grid_columnconfigure(1, weight=1)
        tk.Label(entry_row, text="grbl>", bg=PANEL, fg=OK, font=FONT_MONO).grid(row=0, column=0, padx=(0, 6))
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(
            entry_row,
            textvariable=self.entry_var,
            bg=PANEL2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Consolas", 10),
        )
        self.entry.grid(row=0, column=1, sticky="ew", ipady=5)
        self.entry.bind("<Return>", lambda event: self._send_entry())
        self.entry.bind("<Up>", lambda event: self._history_move(-1))
        self.entry.bind("<Down>", lambda event: self._history_move(1))
        ttk.Button(entry_row, text="Send", style="Accent.TButton", command=self._send_entry).grid(
            row=0, column=2, padx=(6, 0)
        )
        self.entry.focus_set()

    # ------------------------------------------------------------------ logging
    def _enqueue_log(self, kind: str, text: str) -> None:
        self._log_queue.put((time.time(), kind, text))

    def _log_system(self, text: str) -> None:
        self._enqueue_log("sys", text)

    def _drain_log(self) -> None:
        appended = False
        for _ in range(200):
            try:
                timestamp, kind, text = self._log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_console(timestamp, kind, text)
            appended = True
        if appended:
            self._update_console_count()

    def _append_console(self, timestamp: str | float, kind: str, text: str) -> None:
        stamp = time.strftime("%H:%M:%S", time.localtime(timestamp)) + f".{int((timestamp % 1) * 1000):03d}"
        if kind == "tx":
            prefix = ">>"
        elif kind == "rt":
            prefix = "**"
        elif kind == "err":
            prefix = "!!"
        elif kind == "status":
            prefix = "<<"
        elif kind == "sys":
            prefix = "--"
        else:
            prefix = "<<"
        line = f"{stamp}  {prefix}  {text}\n"
        self.console.configure(state="normal")
        self.console.insert("end", line, kind)
        # Keep the widget small and responsive during long streams.
        total = int(self.console.index("end-1c").split(".")[0])
        if total > 2600:
            self.console.delete("1.0", f"{total - 2000}.0")
        self.console.configure(state="disabled")
        if self.autoscroll_var.get():
            self.console.see("end")

    def _update_console_count(self) -> None:
        lines = int(self.console.index("end-1c").split(".")[0])
        self.console_count.configure(text=f"{max(lines - 1, 0)} lines")

    def _clear_console(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.configure(state="disabled")
        self._update_console_count()

    # ------------------------------------------------------------------ commands
    def send_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        self.host.send_line(line)
        self._history.append(line)
        self._history_index = len(self._history)
        if self.host.connected:
            self.entry_var.set("")

    def send_realtime(self, byte: int) -> None:
        self.host.send_realtime(byte)

    def _send_entry(self) -> None:
        text = self.entry_var.get()
        if not text.strip():
            return
        self.send_line(text)
        self.entry_var.set("")

    def _history_move(self, delta: int) -> None:
        if not self._history:
            return
        self._history_index = max(0, min(len(self._history), self._history_index + delta))
        if self._history_index < len(self._history):
            self.entry_var.set(self._history[self._history_index])
        else:
            self.entry_var.set("")

    def jog_axis(self, axis: str, direction: int) -> None:
        try:
            step = float(self.step_var.get())
        except ValueError:
            step = 10.0
        try:
            feed = float(self.jog_feed_var.get())
        except ValueError:
            feed = 500.0
        feed = max(feed, 1.0)
        self.send_line(f"$J=G91 {axis}{direction * step:g} F{feed:g}")

    # ------------------------------------------------------------------ connection
    def _refresh_ports(self, initial: str | None = None) -> None:
        ports = available_ports()
        current = self.port_var.get().strip() if hasattr(self, "port_var") else ""
        if initial:
            if initial not in ports:
                ports = [initial, *ports]
            self.port_combo.configure(values=ports)
            return
        if current and current not in ports:
            ports = [current, *ports]
        if not ports:
            ports = ["COM7", "COM8"]
        self.port_combo.configure(values=ports)

    def _connect_or_disconnect(self) -> None:
        if self.host.connected:
            self._disconnect()
        else:
            self._connect()

    def _connect(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("Serial port", "Choose a COM port first.")
            return
        try:
            baudrate = int(self.baud_var.get())
        except ValueError:
            baudrate = 115200
        try:
            self.host.connect(port, baudrate)
        except Exception as exc:
            self.host.disconnect()
            messagebox.showerror(
                "Serial port",
                f"Cannot open {port} at {baudrate} baud:\n{exc}\n\n"
                "Close Candle or any other program that may be using this port.",
            )
            self._log_system(f"Failed to open {port}: {exc}")
            return
        self._connected_at = time.monotonic()
        self._log_system(f"Listening on {port} at {baudrate} 8-N-1")
        self._log_system("Connect Candle to the paired port (for com0com: the other end of the pair).")
        self.connect_button.configure(text="Disconnect", style="TButton")
        self.connection_label.configure(text=f"{port} @ {baudrate}", fg=OK)
        self.led.itemconfigure(self._led_item, fill=OK)

    def _disconnect(self) -> None:
        port = self.port_var.get().strip()
        self.host.disconnect()
        self._connected_at = None
        self._log_system(f"Closed {port}")
        self.connect_button.configure(text="Connect", style="Accent.TButton")
        self.connection_label.configure(text="Disconnected", fg=MUTED)
        self.led.itemconfigure(self._led_item, fill="#39424d")

    def _check_transport(self) -> None:
        transport = self.host.transport
        if transport is None:
            return
        if transport.connected:
            return
        error = transport.error
        self.host.disconnect()
        self.connect_button.configure(text="Connect", style="Accent.TButton")
        self.connection_label.configure(text="Disconnected", fg=MUTED)
        self.led.itemconfigure(self._led_item, fill="#39424d")
        self._log_system(f"Serial connection closed{f': {error}' if error else ''}")

    # ------------------------------------------------------------------ per frame
    def _tick(self) -> None:
        now = time.monotonic()
        dt = min(max(now - self._last_tick, 0.0), 0.1)
        self._last_tick = now

        self._check_transport()
        if not self.host.connected:
            self.host.local_tick(dt)
        self._drain_log()
        self._drain_workfield_results()
        snapshot = self.host.snapshot()
        self._apply_snapshot(snapshot, dt)
        self.after(28, self._tick)

    def _apply_snapshot(self, snapshot: Snapshot, dt: float) -> None:
        color = STATE_COLORS.get(snapshot.state, TEXT)
        state_text = snapshot.state + (f":{snapshot.substate}" if snapshot.substate is not None else "")
        self.state_label.configure(text=state_text, fg=color)
        hint = STATE_HINTS.get(snapshot.state, "")
        if snapshot.alarm is not None:
            hint = (
                f"alarm {snapshot.alarm} · reset required"
                if snapshot.critical
                else f"alarm {snapshot.alarm} · $X to unlock"
            )
        self.state_hint.configure(text=hint)

        for axis, index in (("X", 0), ("Y", 1), ("Z", 2)):
            self.dro_vars[axis].set(f"{snapshot.mpos[index]:10.3f}")
            self.wpos_vars[axis].set(f"{snapshot.wpos[index]:9.3f}")

        self.feed_var.set(f"F  {snapshot.feed:8.1f} mm/min")
        self.spindle_var.set(
            f"S  {snapshot.rpm:7.0f} / {snapshot.rpm_target:.0f} RPM"
            + ("  CW" if snapshot.spindle_dir > 0 else "  CCW" if snapshot.spindle_dir < 0 else "")
        )
        self.override_var.set(
            f"Ov {snapshot.overrides[0]} / {snapshot.overrides[1]} / {snapshot.overrides[2]} %"
        )
        self._draw_rpm_bar(snapshot)

        flags = []
        flags.append("HOMED" if snapshot.homed else "NOT HOMED")
        if snapshot.check_mode:
            flags.append("CHECK MODE")
        if snapshot.door:
            flags.append("DOOR")
        if snapshot.alarm is not None:
            flags.append(f"ALARM {snapshot.alarm}")
        if snapshot.connected:
            flags.append("SERIAL")
        self.flags_label.configure(text="   ·   ".join(flags))
        self.program_label.configure(
            text=f"queue {snapshot.queued}   ·   Grbl {snapshot.version}"
        )

        self.machine_view.update_snapshot(snapshot, dt)
        self._draw_rpm_bar(snapshot)

    def _draw_rpm_bar(self, snapshot: Snapshot | None = None) -> None:
        canvas = getattr(self, "rpm_canvas", None)
        if canvas is None:
            return
        width = max(canvas.winfo_width(), 1)
        if snapshot is None:
            snapshot = self.host.snapshot()
        max_rpm = max(snapshot.settings.get(30, 10_000) or 10_000, 1)
        fraction = min(max(snapshot.rpm / max_rpm, 0.0), 1.0)
        canvas.coords(self.rpm_bar, 0, 0, width * fraction, 6)
        canvas.itemconfigure(self.rpm_bar, fill=OK if fraction else "#263241")

    # ------------------------------------------------------------------ dialogs
    def _open_settings(self) -> None:
        if self._settings_dialog is not None and self._settings_dialog.winfo_exists():
            self._settings_dialog.lift()
            self._settings_dialog.focus_force()
            return
        self._settings_dialog = SettingsDialog(self)

    def _show_help(self) -> None:
        window = tk.Toplevel(self)
        window.title("About / help")
        window.geometry("640x520")
        window.configure(bg=BG)
        window.transient(self)
        text = tk.Text(
            window,
            bg=PANEL2,
            fg=TEXT,
            relief="flat",
            font=("Segoe UI", 10),
            wrap="word",
            padx=16,
            pady=14,
        )
        text.pack(fill="both", expand=True, padx=12, pady=12)
        text.insert(
            "end",
            "CNC 3018 — GRBL 1.1 emulator\n"
            "================================\n\n"
            "The program opens a serial port at 115200 8-N-1 and behaves like a "
            "CNC3018 controller running Grbl 1.1: it understands G-code, $ system "
            "commands, real-time characters and reports machine status.\n\n"
            "Typical use with com0com\n"
            "------------------------\n"
            "The emulator opens one end of a virtual null-modem pair (COM7 by default). "
            "Open Candle on the other end (COM8) and connect. Candle sends Ctrl-X; the "
            "emulator answers with the Grbl banner, settings and status reports.\n\n"
            "Without a port you can type G-code in the console; the same simulation "
            "engine executes it locally so the carriage and spindle animate.\n\n"
            "Spindle model\n"
            "-------------\n"
            "The spindle ramps up and down in 2 seconds (0 ↔ $30 RPM). The cross on "
            "the table rotates at 1/166.67 of the spindle speed: 10 000 RPM → 60 RPM.\n\n"
            "Useful commands\n"
            "---------------\n"
            "$$\tshow settings\n"
            "$G\tshow modal state\n"
            "$#\tshow coordinate offsets\n"
            "$H\thome (needs $22=1)\n"
            "$X\tunlock alarm\n"
            "$J=G91 X10 F500\tjog 10 mm\n"
            "G1 X10 Y5 F600\tmove\n"
            "M3 S10000 / M5\tspindle on/off\n"
            "! / ~\tfeed hold / resume\n"
            "Ctrl-X\tsoft reset\n\n"
            "Feedback messages are shown in the console and in the state panel.",
        )
        text.configure(state="disabled")

    # ------------------------------------------------------------------ close
    def _on_close(self) -> None:
        try:
            self.host.disconnect()
        except Exception:
            pass
        self.destroy()


def run(
    port: str = "COM7",
    baudrate: int = 115200,
    eeprom_path: str | None = None,
    autoconnect: bool = False,
    workfield_path: str | None = None,
) -> None:
    app = App(
        port=port,
        baudrate=baudrate,
        eeprom_path=eeprom_path,
        autoconnect=autoconnect,
        workfield_path=workfield_path,
    )
    app.mainloop()

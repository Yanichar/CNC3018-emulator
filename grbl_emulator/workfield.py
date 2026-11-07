"""Photorealistic image of the machine work field.

The image is the base for the future "virtual camera": it is rendered at a
fixed scale of 20 px/mm, so the default 300 x 180 mm CNC3018 table becomes a
6000 x 3600 px PNG.  Machine coordinates (millimetres, origin at the front-left
corner, Y pointing away from the operator) map to image pixels through
:class:`FieldGeometry`:

    px = x_mm * px_per_mm
    py = (height_mm - y_mm) * px_per_mm

A camera can crop any region of the image without re-rendering, and holes and
the board are drawn at their true size.
"""

from __future__ import annotations

import math
import os
import random
import struct
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .pcb import Board, Hole, Point

#: Scale of the work-field image: 20 pixels per millimetre.
PX_PER_MM = 20.0

#: Default CNC3018 travel ($130 x $131), millimetres.
WORK_AREA_MM = (300.0, 180.0)

#: Colours required by the task.
BOARD_COLOR = (0xC8, 0x75, 0x33)
HOLE_COLOR = (0x00, 0x00, 0x00)

#: Darker bevel around the board and the soft shadow it casts on the table.
BOARD_EDGE_COLOR = tuple(int(channel * 0.72) for channel in BOARD_COLOR)
SHADOW_COLOR = (44, 50, 56)

#: Light aluminium of the machine table.
ALUMINIUM_COLOR = (203, 206, 209)


@dataclass(frozen=True)
class FieldGeometry:
    """Mapping between the machine table (mm) and the work-field image (px)."""

    width_mm: float = WORK_AREA_MM[0]
    height_mm: float = WORK_AREA_MM[1]
    px_per_mm: float = PX_PER_MM

    @property
    def image_width(self) -> int:
        return max(1, int(round(self.width_mm * self.px_per_mm)))

    @property
    def image_height(self) -> int:
        return max(1, int(round(self.height_mm * self.px_per_mm)))

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_width, self.image_height

    def to_pixel(self, x_mm: float, y_mm: float) -> Point:
        """Continuous image coordinates of a machine point (Y axis points down)."""

        return (x_mm * self.px_per_mm, (self.height_mm - y_mm) * self.px_per_mm)

    def to_machine(self, px: float, py: float) -> Point:
        """Machine coordinates of a continuous image point."""

        return (px / self.px_per_mm, self.height_mm - py / self.px_per_mm)

    def crop_box(
        self, x_mm: float, y_mm: float, width_mm: float, height_mm: float
    ) -> tuple[int, int, int, int]:
        """Integer ``(left, top, right, bottom)`` crop for a machine-space window."""

        left, top = self.to_pixel(x_mm, y_mm + height_mm)
        right, bottom = self.to_pixel(x_mm + width_mm, y_mm)
        return (
            max(0, int(math.floor(left))),
            max(0, int(math.floor(top))),
            min(self.image_width, int(math.ceil(right))),
            min(self.image_height, int(math.ceil(bottom))),
        )


DEFAULT_GEOMETRY = FieldGeometry()


def default_output_path() -> Path:
    """Where the work-field PNG is written by default (project-root ``workfield``)."""

    return Path(__file__).resolve().parent.parent / "workfield" / "workfield.png"


def center_offset(
    board: Board,
    geometry: FieldGeometry = DEFAULT_GEOMETRY,
) -> tuple[float, float]:
    """Offset that puts the board's bounding box in the middle of the table."""

    min_x, min_y, max_x, max_y = board.bounds
    return (
        geometry.width_mm / 2.0 - (min_x + max_x) / 2.0,
        geometry.height_mm / 2.0 - (min_y + max_y) / 2.0,
    )


# --------------------------------------------------------------------------- aluminium
def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _grain_pattern(rng: random.Random, width: int) -> bytes:
    """Fine horizontal grain: random values stretched to 2-pixel runs."""

    values = rng.randbytes((width + 1) // 2)
    buffer = bytearray(width)
    buffer[0::2] = values[: len(buffer[0::2])]
    buffer[1::2] = values[: len(buffer[1::2])]
    return bytes(buffer)


def _channel_table(base: int, level: int) -> bytes:
    return bytes(_clamp_channel(base + level + (value - 128) // 32) for value in range(256))


class _AluminiumSurface:
    """Light, horizontally brushed aluminium built from repeated bands.

    Every band of 4-16 rows shares one grain pattern and one brightness level,
    which looks like brushed metal and keeps the PNG small.
    """

    def __init__(self, width: int, height: int, seed: int = 3018) -> None:
        self.width = width
        self.height = height
        rng = random.Random(seed)
        self._grain = [_grain_pattern(rng, width) for _ in range(48)]
        self._band_of_row = [0] * height
        self._band_rows: list[bytes] = []
        self._levels: list[int] = []

        row = 0
        level = 0
        while row < height:
            height_px = min(rng.randint(4, 16), height - row)
            level = max(-5, min(5, level + rng.choice((-1, -1, 0, 0, 0, 1, 1))))
            if rng.random() < 0.04:
                level = max(-5, min(5, level + rng.choice((-4, 4))))
            grain = self._grain[rng.randrange(len(self._grain))]
            index = len(self._band_rows)
            self._band_rows.append(self._render_row(grain, level))
            self._levels.append(level)
            for pixel_row in range(row, row + height_px):
                self._band_of_row[pixel_row] = index
            row += height_px

    @staticmethod
    def _render_row(grain: bytes, level: int) -> bytes:
        base_r, base_g, base_b = ALUMINIUM_COLOR
        row = bytearray(len(grain) * 3)
        row[0::3] = grain.translate(_channel_table(base_r - 2, level))
        row[1::3] = grain.translate(_channel_table(base_g, level))
        row[2::3] = grain.translate(_channel_table(base_b + 2, level))
        return bytes(row)

    def row(self, index: int) -> bytes:
        return self._band_rows[self._band_of_row[index]]


# --------------------------------------------------------------------------- geometry helpers
def _polygon_edges(loops: Sequence[Sequence[Point]]) -> list[tuple[Point, Point]]:
    edges: list[tuple[Point, Point]] = []
    for loop in loops:
        if len(loop) < 2:
            continue
        previous = loop[-1]
        for point in loop:
            if previous[1] != point[1]:
                edges.append((previous, point))
            previous = point
    return edges


def _scanline_spans(edges: Sequence[tuple[Point, Point]], y: float) -> list[tuple[float, float]]:
    crossings: list[float] = []
    for (x1, y1), (x2, y2) in edges:
        if (y1 <= y < y2) or (y2 <= y < y1):
            crossings.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
    crossings.sort()
    spans: list[tuple[float, float]] = []
    for index in range(0, len(crossings) - 1, 2):
        left, right = crossings[index], crossings[index + 1]
        if right - left > 1e-9:
            spans.append((left, right))
    return spans


def _clip_span(left: float, right: float, width: int) -> tuple[int, int]:
    start = max(0, int(math.floor(left)))
    end = min(width, int(math.ceil(right)))
    return start, end


def _spans_for_loops(
    loops: Sequence[Sequence[Point]], geometry: FieldGeometry
) -> dict[int, list[tuple[float, float]]]:
    edges = _polygon_edges(loops)
    ys = [point[1] for loop in loops for point in loop]
    if not ys:
        return {}
    first_row = max(0, int(math.floor(min(ys))))
    last_row = min(geometry.image_height - 1, int(math.ceil(max(ys))))
    spans: dict[int, list[tuple[float, float]]] = {}
    for row in range(first_row, last_row + 1):
        line = _scanline_spans(edges, row + 0.5)
        if line:
            spans[row] = line
    return spans


def _shift_loops(loops: Sequence[Sequence[Point]], dx: float, dy: float) -> list[list[Point]]:
    return [[(x + dx, y + dy) for x, y in loop] for loop in loops]


def _subtract_spans(
    base: Sequence[tuple[float, float]], cut: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Intervals of *base* that are not covered by *cut* (all in pixels)."""

    result: list[tuple[float, float]] = []
    for start, end in base:
        pieces = [(start, end)]
        for cut_start, cut_end in cut:
            remaining: list[tuple[float, float]] = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    remaining.append((piece_start, piece_end))
                    continue
                if cut_start > piece_start:
                    remaining.append((piece_start, min(cut_start, piece_end)))
                if cut_end < piece_end:
                    remaining.append((max(cut_end, piece_start), piece_end))
            pieces = remaining
        result.extend(pieces)
    return result


def _blend_span(
    pixels: bytearray,
    left: float,
    right: float,
    color: tuple[int, int, int],
    alpha: float,
    width: int,
) -> None:
    start, end = _clip_span(left, right, width)
    if end <= start:
        return
    inverse = 1.0 - alpha
    r = color[0] * alpha
    g = color[1] * alpha
    b = color[2] * alpha
    for index in range(start * 3, end * 3, 3):
        pixels[index] = int(pixels[index] * inverse + r)
        pixels[index + 1] = int(pixels[index + 1] * inverse + g)
        pixels[index + 2] = int(pixels[index + 2] * inverse + b)


def _line_pixels(x0: float, y0: float, x1: float, y1: float) -> Iterator[tuple[int, int]]:
    """Bresenham line in integer image coordinates."""

    x0i, y0i = int(round(x0)), int(round(y0))
    x1i, y1i = int(round(x1)), int(round(y1))
    dx = abs(x1i - x0i)
    dy = -abs(y1i - y0i)
    step_x = 1 if x0i < x1i else -1
    step_y = 1 if y0i < y1i else -1
    error = dx + dy
    while True:
        yield x0i, y0i
        if x0i == x1i and y0i == y1i:
            break
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0i += step_x
        if doubled <= dx:
            error += dx
            y0i += step_y


def _board_texture(
    loops: Sequence[Sequence[Point]],
    board_spans: dict[int, list[tuple[float, float]]],
    geometry: FieldGeometry,
) -> tuple[int, dict[int, bytes], dict[int, set[int]]]:
    """Copper-foil texture for the board: highlights, mottling, grain, edge."""

    xs = [point[0] for loop in loops for point in loop]
    ys = [point[1] for loop in loops for point in loop]
    if not xs or not ys:
        return 0, {}, {}
    x0 = max(0, int(math.floor(min(xs))))
    x1 = min(geometry.image_width, int(math.ceil(max(xs))))
    y0 = max(0, int(math.floor(min(ys))))
    y1 = min(geometry.image_height, int(math.ceil(max(ys))))
    texture_width = max(1, x1 - x0)
    texture_height = max(1, y1 - y0)

    # The board edge: a darker bevel all around the contour.
    edge: dict[int, set[int]] = {}
    for loop in loops:
        points = list(loop)
        points.append(points[0])
        for (ax, ay), (bx, by) in zip(points, points[1:]):
            for x, y in _line_pixels(ax, ay, bx, by):
                if 0 <= x < geometry.image_width and 0 <= y < geometry.image_height:
                    edge.setdefault(y, set()).add(x)

    rng = random.Random(0xB0A2D5EED)
    noise = rng.randbytes(texture_width * texture_height)
    # Rolled copper foil: a faint brushed streak level per row of the board.
    streaks = [(rng.random() * 2.0 - 1.0) * 0.012 for _ in range(texture_height)]
    light_x = [0.060 * (0.5 - i / max(texture_width - 1, 1)) for i in range(texture_width)]
    sin_x1 = [math.sin(i * 0.055) for i in range(texture_width)]
    cos_x1 = [math.cos(i * 0.055) for i in range(texture_width)]
    sin_x2 = [math.sin(i * 0.021) for i in range(texture_width)]
    cos_x2 = [math.cos(i * 0.021) for i in range(texture_width)]
    # Broad specular highlight in the upper-left quadrant: gives the foil a sheen.
    sheen_x = [
        math.exp(-(((i / max(texture_width - 1, 1)) - 0.30) ** 2) / 0.08)
        for i in range(texture_width)
    ]
    sheen_y_values = [
        math.exp(-(((j / max(texture_height - 1, 1)) - 0.25) ** 2) / 0.15)
        for j in range(texture_height)
    ]
    base_r, base_g, base_b = BOARD_COLOR

    rows: dict[int, bytes] = {}
    for j, image_row in enumerate(range(y0, y1)):
        if image_row not in board_spans:
            continue
        light_y = 0.080 * (0.5 - j / max(texture_height - 1, 1))
        sheen_y = sheen_y_values[j]
        sin_y1, cos_y1 = math.sin(j * 0.043), math.cos(j * 0.043)
        sin_y2, cos_y2 = math.sin(j * 0.017), math.cos(j * 0.017)
        buffer = bytearray(texture_width * 3)
        noise_offset = j * texture_width
        for i in range(texture_width):
            mottle = 0.025 * (sin_x1[i] * cos_y1 + cos_x1[i] * sin_y1) + 0.015 * (
                sin_x2[i] * cos_y2 - cos_x2[i] * sin_y2
            )
            grain = (noise[noise_offset + i] - 128) * 0.00018
            factor = (
                1.0
                + light_x[i]
                + light_y
                + mottle
                + grain
                + streaks[j]
                + 0.08 * sheen_x[i] * sheen_y
            )
            index = i * 3
            buffer[index] = _clamp_channel(base_r * factor)
            buffer[index + 1] = _clamp_channel(base_g * factor)
            buffer[index + 2] = _clamp_channel(base_b * factor)
        rows[image_row] = bytes(buffer)
    return x0, rows, edge


def _hole_spans(
    holes: Iterable[Hole],
    geometry: FieldGeometry,
    offset: tuple[float, float],
) -> dict[int, list[tuple[float, float]]]:
    offset_x, offset_y = offset
    spans: dict[int, list[tuple[float, float]]] = {}
    for hole in holes:
        center_x, center_y = geometry.to_pixel(hole.x + offset_x, hole.y + offset_y)
        radius = hole.diameter * geometry.px_per_mm / 2.0
        if radius <= 0.0:
            continue
        first_row = max(0, int(math.floor(center_y - radius)))
        last_row = min(geometry.image_height - 1, int(math.ceil(center_y + radius)))
        for row in range(first_row, last_row + 1):
            delta = row + 0.5 - center_y
            if abs(delta) >= radius:
                continue
            half = math.sqrt(max(radius * radius - delta * delta, 0.0))
            spans.setdefault(row, []).append((center_x - half, center_x + half))
    return spans


# --------------------------------------------------------------------------- PNG writer
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _write_png(
    path: Path,
    width: int,
    height: int,
    rows: Iterable[bytes],
    compress_level: int = 6,
) -> None:
    compressor = zlib.compressobj(compress_level)
    idat_chunks: list[bytes] = []
    for row in rows:
        chunk = compressor.compress(b"\x00" + row)  # filter type 0 (None)
        if chunk:
            idat_chunks.append(chunk)
    idat_chunks.append(compressor.flush())
    idat = b"".join(idat_chunks)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    with open(path, "wb") as handle:
        handle.write(b"\x89PNG\r\n\x1a\n")
        handle.write(_png_chunk(b"IHDR", ihdr))
        handle.write(_png_chunk(b"IDAT", idat))
        handle.write(_png_chunk(b"IEND", b""))


# --------------------------------------------------------------------------- rendering
def render_workfield(
    path: str | Path,
    board: Board | None = None,
    offset: tuple[float, float] | None = None,
    geometry: FieldGeometry = DEFAULT_GEOMETRY,
) -> Path:
    """Render the table (and optionally a board) to a PNG file.

    Returns the written path.  With ``board=None`` the image is an empty
    aluminium table, which is also what the machine shows before a board is
    loaded.
    """

    output = Path(path)
    width, height = geometry.image_size
    aluminium = _AluminiumSurface(width, height)

    board_spans: dict[int, list[tuple[float, float]]] = {}
    hole_spans: dict[int, list[tuple[float, float]]] = {}
    shadow_layers: list[tuple[dict[int, list[tuple[float, float]]], float]] = []
    texture_x = 0
    texture_rows: dict[int, bytes] = {}
    edge_pixels: dict[int, set[int]] = {}
    if board is not None:
        if offset is None:
            offset = center_offset(board, geometry)
        loops = [
            [geometry.to_pixel(x + offset[0], y + offset[1]) for x, y in loop]
            for loop in board.loops
        ]
        board_spans = _spans_for_loops(loops, geometry)
        hole_spans = _hole_spans(board.holes, geometry, offset)
        texture_x, texture_rows, edge_pixels = _board_texture(
            loops, board_spans, geometry
        )
        # A soft, two-step shadow cast on the aluminium (light from the top-left).
        for dx, dy, alpha in ((7.0, 8.0, 0.07), (4.0, 5.0, 0.09)):
            shadow_layers.append(
                (_spans_for_loops(_shift_loops(loops, dx, dy), geometry), alpha)
            )

    board_bytes = bytes(BOARD_COLOR)
    edge_bytes = bytes(BOARD_EDGE_COLOR)
    hole_bytes = bytes(HOLE_COLOR)

    def rows() -> Iterator[bytes]:
        for row in range(height):
            fill = board_spans.get(row)
            drills = hole_spans.get(row)
            shadows: list[tuple[list[tuple[float, float]], float]] = []
            for spans, alpha in shadow_layers:
                layer = spans.get(row)
                if layer:
                    visible = _subtract_spans(layer, fill or ())
                    if visible:
                        shadows.append((visible, alpha))
            if not fill and not drills and not shadows:
                yield aluminium.row(row)
                continue

            pixels = bytearray(aluminium.row(row))
            for visible, alpha in shadows:
                for left, right in visible:
                    _blend_span(pixels, left, right, SHADOW_COLOR, alpha, width)

            if fill:
                texture = texture_rows.get(row)
                for left, right in fill:
                    start, end = _clip_span(left, right, width)
                    if end <= start:
                        continue
                    if texture is None:
                        pixels[start * 3 : end * 3] = board_bytes * (end - start)
                        continue
                    texture_start = max(0, start - texture_x)
                    texture_end = min(len(texture) // 3, end - texture_x)
                    if texture_end > texture_start:
                        pixels[start * 3 : start * 3 + (texture_end - texture_start) * 3] = (
                            texture[texture_start * 3 : texture_end * 3]
                        )
                    else:
                        pixels[start * 3 : end * 3] = board_bytes * (end - start)

            for x in edge_pixels.get(row, ()):
                if 0 <= x < width:
                    pixels[x * 3 : x * 3 + 3] = edge_bytes

            for left, right in drills or ():
                start, end = _clip_span(left, right, width)
                if end > start:
                    pixels[start * 3 : end * 3] = hole_bytes * (end - start)
            yield bytes(pixels)

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".workfield-", suffix=".png", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_png(temporary, width, height, rows())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass
    return output

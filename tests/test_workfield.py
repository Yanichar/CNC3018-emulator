"""Work-field PNG: geometry, placement and pixel content."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from grbl_emulator.pcb import Board, Hole
from grbl_emulator.workfield import (
    BOARD_COLOR,
    HOLE_COLOR,
    FieldGeometry,
    center_offset,
    render_workfield,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_png(path: str | Path) -> tuple[int, int, list[bytes]]:
    data = Path(path).read_bytes()
    assert data[:8] == PNG_SIGNATURE
    position = 8
    idat = bytearray()
    width = height = 0
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        tag = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length
        if tag == b"IHDR":
            width, height = struct.unpack(">II", chunk[:8])
        elif tag == b"IDAT":
            idat += chunk
    raw = zlib.decompress(bytes(idat))
    stride = width * 3 + 1
    rows: list[bytes] = []
    for row in range(height):
        line = raw[row * stride : (row + 1) * stride]
        assert line[0] == 0  # the renderer always writes filter type 0
        rows.append(line[1:])
    return width, height, rows


def pixel(rows: list[bytes], x: int, y: int) -> tuple[int, int, int]:
    offset = x * 3
    return tuple(rows[y][offset : offset + 3])  # type: ignore[return-value]


def test_default_geometry_is_20px_per_mm() -> None:
    geometry = FieldGeometry()
    assert geometry.image_size == (6000, 3600)
    assert geometry.width_mm == 300.0
    assert geometry.height_mm == 180.0
    assert geometry.px_per_mm == 20.0


def test_camera_transform_round_trip() -> None:
    geometry = FieldGeometry()
    for x, y in ((0.0, 0.0), (300.0, 180.0), (123.4, 56.7)):
        px, py = geometry.to_pixel(x, y)
        back_x, back_y = geometry.to_machine(px, py)
        assert back_x == pytest.approx(x)
        assert back_y == pytest.approx(y)
    assert geometry.to_pixel(0.0, 0.0) == (0.0, 3600.0)
    assert geometry.to_pixel(300.0, 180.0) == (6000.0, 0.0)


def test_crop_box() -> None:
    geometry = FieldGeometry()
    left, top, right, bottom = geometry.crop_box(10.0, 20.0, 50.0, 30.0)
    assert (left, top, right, bottom) == (200, 2600, 1200, 3200)


def test_center_offset() -> None:
    board = Board(
        loops=[[(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)]],
        holes=[],
    )
    assert center_offset(board) == pytest.approx((130.0, 75.0))


def test_render_png_contains_board_and_holes(tmp_path) -> None:
    geometry = FieldGeometry(width_mm=60.0, height_mm=40.0, px_per_mm=10.0)
    board = Board(
        loops=[[(10.0, 10.0), (30.0, 10.0), (30.0, 25.0), (10.0, 25.0)]],
        holes=[Hole(15.0, 15.0, 3.0)],
    )
    offset = center_offset(board, geometry)
    output = render_workfield(tmp_path / "field.png", board, offset, geometry)

    width, height, rows = read_png(output)
    assert (width, height) == (600, 400)
    assert output.stat().st_size > 0

    # Board centre (machine 30, 20) after centring: textured copper, still
    # close to the requested #C87533.
    centre = pixel(rows, 300, 200)
    assert all(abs(channel - base) <= 24 for channel, base in zip(centre, BOARD_COLOR))
    samples = {
        pixel(rows, x, y) for x, y in ((260, 160), (340, 180), (300, 240), (360, 150))
    }
    assert len(samples) > 1  # the foil is textured, not a flat fill
    # The board has a darker milled edge.
    assert sum(pixel(rows, 400, 200)) < sum(centre) - 60
    # Hole centre (machine 25, 17.5).
    assert pixel(rows, 250, 225) == HOLE_COLOR
    # A soft shadow is cast below/right of the board.
    assert sum(pixel(rows, 403, 210)) < sum(pixel(rows, 10, 10))
    # A corner far from the board is light aluminium.
    corner = pixel(rows, 10, 10)
    assert all(180 <= channel <= 225 for channel in corner)
    assert max(corner) - min(corner) <= 16


def test_render_empty_table(tmp_path) -> None:
    geometry = FieldGeometry(width_mm=10.0, height_mm=10.0, px_per_mm=5.0)
    output = render_workfield(tmp_path / "empty.png", None, geometry=geometry)
    width, height, rows = read_png(output)
    assert (width, height) == (50, 50)
    for y in range(height):
        for x in range(width):
            color = pixel(rows, x, y)
            assert all(180 <= channel <= 225 for channel in color)

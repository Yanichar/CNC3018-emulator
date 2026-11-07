"""Gerber / Excellon parsing and board geometry."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from grbl_emulator.pcb import (
    Board,
    Hole,
    PcbError,
    discover_sources,
    guess_outline_layer,
    load_board,
    parse_excellon,
    parse_gerber,
    parse_gerber_contours,
)

GERBER_RECT = """\
%TF.GenerationSoftware,KiCad,Pcbnew,10.0.1*%
%FSLAX46Y46*%
%MOMM*%
%LPD*%
G01*
%ADD10C,0.127000*%
D10*
X10000000Y2000000D02*
X20000000Y2000000D01*
X20000000Y7000000D01*
X10000000Y7000000D01*
X10000000Y2000000D01*
M02*
"""

GERBER_ARC = """\
%FSLAX46Y46*%
%MOMM*%
%ADD10C,0.100000*%
D10*
G75*
G03*
X5000000Y0D02*
X0Y0I-2500000J0D01*
X-5000000Y0D01*
X5000000Y0D01*
M02*
"""

EXCELLON = """\
M48
; FORMAT={-:-/ absolute / metric / decimal}
FMAT,2
METRIC
T1C0.800
T2C3.200
%
G90
G05
T1
X1.000Y2.000
X3.500Y4.250
T2
X10.000Y-5.000
M30
"""

EXCELLON_INTEGER = """\
M48
METRIC,LZ
T1C1.000
%
G90
T1
X010000Y020000
X015000Y000000
M30
"""


def test_parse_rectangle_contour() -> None:
    loops = parse_gerber(GERBER_RECT)
    assert len(loops) == 1
    loop = loops[0]
    assert len(loop) == 4
    xs = sorted(point[0] for point in loop)
    ys = sorted(point[1] for point in loop)
    assert xs == pytest.approx([10.0, 10.0, 20.0, 20.0])
    assert ys == pytest.approx([2.0, 2.0, 7.0, 7.0])


def test_parse_arc_contour() -> None:
    loops = parse_gerber(GERBER_ARC)
    assert len(loops) == 1
    loop = loops[0]
    xs = [point[0] for point in loop]
    ys = [point[1] for point in loop]
    assert min(xs) == pytest.approx(-5.0, abs=1e-4)
    assert max(xs) == pytest.approx(5.0, abs=1e-4)
    assert min(ys) == pytest.approx(0.0, abs=1e-4)
    assert len(loop) > 10  # the semicircle is flattened


def test_gerber_without_contour_raises() -> None:
    with pytest.raises(PcbError):
        parse_gerber("%FSLAX46Y46*%\n%MOMM*%\n%ADD10C,0.1*%\nD10*\nM02*\n")


def test_parse_excellon_holes() -> None:
    holes = parse_excellon(EXCELLON)
    assert len(holes) == 3
    assert holes[0].x == pytest.approx(1.0)
    assert holes[0].y == pytest.approx(2.0)
    assert holes[0].diameter == pytest.approx(0.8)
    assert holes[2].diameter == pytest.approx(3.2)
    assert holes[2].y == pytest.approx(-5.0)
    assert all(hole.plated for hole in holes)


def test_parse_excellon_integer_coordinates() -> None:
    holes = parse_excellon(EXCELLON_INTEGER)
    assert [hole.x for hole in holes] == pytest.approx([10.0, 15.0])
    assert [hole.y for hole in holes] == pytest.approx([20.0, 0.0])


def test_board_bounds_and_size() -> None:
    board = Board(
        loops=[[(0.0, 0.0), (40.0, 0.0), (40.0, 30.0), (0.0, 30.0)]],
        holes=[Hole(10.0, 10.0, 3.0)],
    )
    assert board.bounds == (0.0, 0.0, 40.0, 30.0)
    assert board.width == pytest.approx(40.0)
    assert board.height == pytest.approx(30.0)


def test_guess_outline_layer() -> None:
    names = ["ecc83-pp-top_cu.gbr", "ecc83-pp-Edge_Cuts.gbr", "ecc83-pp-F_Mask.gbr"]
    assert guess_outline_layer(names) == "ecc83-pp-Edge_Cuts.gbr"
    assert guess_outline_layer(["a.gbr", "b.gbr"]) == "a.gbr"
    assert guess_outline_layer(["a.gbr", "b.gbr"], fallback=False) is None
    assert guess_outline_layer([]) is None


def test_load_board_merges_all_drills(tmp_path) -> None:
    gerber = tmp_path / "board-Edge_Cuts.gbr"
    gerber.write_text(GERBER_RECT, encoding="utf-8")
    first = tmp_path / "board-PTH.drl"
    first.write_text(EXCELLON, encoding="utf-8")
    second = tmp_path / "board-NPTH.drl"
    second.write_text(EXCELLON_INTEGER, encoding="utf-8")

    board = load_board(gerber, [first, second])
    assert len(board.holes) == 5
    assert board.source == "board-Edge_Cuts.gbr"
    assert board.holes[0].plated is True
    # NPTH is detected from the file name.
    assert board.holes[3].plated is False


def test_discover_sources(tmp_path) -> None:
    (tmp_path / "a-Edge_Cuts.gbr").write_text(GERBER_RECT, encoding="utf-8")
    (tmp_path / "a-PTH.drl").write_text(EXCELLON, encoding="utf-8")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    drills, gerbers = discover_sources(tmp_path)
    assert [path.name for path in drills] == ["a-PTH.drl"]
    assert [path.name for path in gerbers] == ["a-Edge_Cuts.gbr"]


def test_reference_project_if_available() -> None:
    # The task's reference project lives next to this repository.
    reference = Path(__file__).resolve().parents[2] / "ecc83" / "gerber"
    if not reference.is_dir():
        pytest.skip("reference project is not available")
    board = load_board(
        reference / "ecc83-pp-Edge_Cuts.gbr",
        [reference / "ecc83-pp-NPTH.drl", reference / "ecc83-pp-PTH.drl"],
    )
    assert board.width == pytest.approx(52.07, abs=1e-6)
    assert board.height == pytest.approx(46.355, abs=1e-6)
    assert len(board.holes) == 33
    assert sorted({round(hole.diameter, 3) for hole in board.holes}) == [
        0.8,
        1.0,
        1.02,
        1.5,
        3.2,
    ]
    assert all(
        math.isfinite(value)
        for loop in board.loops
        for point in loop
        for value in point
    )

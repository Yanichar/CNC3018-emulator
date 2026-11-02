"""Parser level tests: syntax, modal state, arcs, offsets and error codes."""

from __future__ import annotations

import math

import pytest

from grbl_emulator.gcode import GCodeError, GCodeParser
from grbl_emulator.settings import SettingsStore


def make_parser() -> GCodeParser:
    return GCodeParser(SettingsStore(None))


def parse(parser: GCodeParser, line: str, mpos=(0.0, 0.0, 0.0)):
    return parser.parse(line, mpos)


def error_code(parser: GCodeParser, line: str, mpos=(0.0, 0.0, 0.0)) -> int:
    with pytest.raises(GCodeError) as info:
        parser.parse(line, mpos)
    return info.value.code


def test_default_modal_state():
    parser = make_parser()
    assert parser.state.motion == 0
    assert parser.state.wcs == "G54"
    assert parser.state.plane == "G17"
    assert parser.state.distance == "G90"
    assert parser.state.units == 21
    assert parser.state.spindle == "M5"
    assert parser.state.coolant == "M9"
    assert parser.gc_state_string() == "[GC:G0 G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]"


def test_absolute_and_incremental_moves():
    parser = make_parser()
    block = parse(parser, "G1 X10 Y5 F600")
    assert len(block.moves) == 1
    assert block.moves[0].target == (10.0, 5.0, 0.0)
    assert block.moves[0].feed == pytest.approx(600.0)
    assert block.moves[0].rapid is False

    block = parse(parser, "G91")
    assert block.moves == []
    block = parse(parser, "G1 X2 F600", mpos=(10.0, 5.0, 0.0))
    assert block.moves[0].target == (12.0, 5.0, 0.0)


def test_rapid_and_modal_motion():
    parser = make_parser()
    block = parse(parser, "G0 X1 Y2")
    assert block.moves[0].rapid
    block = parse(parser, "X3", mpos=(1.0, 2.0, 0.0))  # modal G0
    assert block.moves[0].rapid
    assert block.moves[0].target == (3.0, 2.0, 0.0)


def test_units_inches():
    parser = make_parser()
    parse(parser, "G20 G90")
    block = parse(parser, "G1 X1 F10")
    assert block.moves[0].target[0] == pytest.approx(25.4)
    assert block.moves[0].feed == pytest.approx(254.0)
    assert parser.state.units == 20


def test_work_offsets_applied():
    parser = make_parser()
    parse(parser, "G10 L2 P1 X10 Y20 Z30")
    parse(parser, "G54")
    block = parse(parser, "G1 X5 Y5 F500")
    assert block.moves[0].target == pytest.approx((15.0, 25.0, 0.0))
    block = parse(parser, "G1 Z0 F500", mpos=(0.0, 0.0, 0.0))
    assert block.moves[0].target == pytest.approx((0.0, 0.0, 30.0))


def test_g92_sets_work_position():
    parser = make_parser()
    parse(parser, "G92 X0 Y0 Z0", mpos=(10.0, 5.0, 2.0))
    assert parser.wco() == pytest.approx([10.0, 5.0, 2.0])
    block = parse(parser, "G1 X0 F500", mpos=(10.0, 5.0, 2.0))
    assert block.moves[0].target == pytest.approx((10.0, 5.0, 2.0))
    parse(parser, "G92.1")
    assert parser.wco() == pytest.approx([0.0, 0.0, 0.0])


def test_g10_l20_uses_current_position():
    parser = make_parser()
    parse(parser, "G10 L20 P1 X0 Y0", mpos=(10.0, 4.0, 0.0))
    assert parser.store.wcs_offset("G54") == pytest.approx([10.0, 4.0, 0.0])


def test_g28_and_g30():
    parser = make_parser()
    parse(parser, "G28.1", mpos=(12.0, 7.0, 0.0))
    assert parser.store.g28 == pytest.approx([12.0, 7.0, 0.0])
    block = parse(parser, "G28", mpos=(0.0, 0.0, 0.0))
    assert block.moves[-1].target == pytest.approx((12.0, 7.0, 0.0))
    assert block.moves[-1].rapid

    parse(parser, "G30.1", mpos=(1.0, 2.0, 3.0))
    block = parse(parser, "G30", mpos=(20.0, 20.0, 20.0))
    assert block.moves[-1].target == pytest.approx((1.0, 2.0, 3.0))


def test_g53_machine_coordinates():
    parser = make_parser()
    parse(parser, "G10 L2 P1 X100")
    block = parse(parser, "G53 G0 X10")
    assert block.moves[0].target == (10.0, 0.0, 0.0)
    assert error_code(parser, "G53 G2 X10 Y10 I1 J0 F100") == 30
    assert error_code(parser, "G53 X10") == 30


def test_arc_ijk_geometry():
    parser = make_parser()
    block = parse(parser, "G2 X10 Y10 I10 J0 F600")
    arc = block.moves[0].arc
    assert arc is not None
    assert arc.center == pytest.approx((10.0, 0.0, 0.0))
    assert arc.radius == pytest.approx(10.0)
    assert arc.sweep < 0  # G2 is clockwise
    assert block.moves[0].target == pytest.approx((10.0, 10.0, 0.0))


def test_arc_full_circle():
    parser = make_parser()
    block = parse(parser, "G3 X0 Y0 I5 J0 F500")
    arc = block.moves[0].arc
    assert arc is not None
    assert abs(arc.sweep) == pytest.approx(2.0 * math.pi)
    assert arc.length == pytest.approx(2.0 * math.pi * 5.0)


def test_arc_radius_form():
    parser = make_parser()
    block = parse(parser, "G2 X10 Y0 R5 F500")
    arc = block.moves[0].arc
    assert arc is not None
    assert arc.radius == pytest.approx(5.0)
    assert block.moves[0].target == pytest.approx((10.0, 0.0, 0.0))


def test_arc_in_other_planes():
    parser = make_parser()
    block = parse(parser, "G18 G2 X10 Z10 I10 K0 F500")
    assert block.moves[0].arc is not None
    block = parse(parser, "G19 G3 Y10 Z10 J10 K0 F500")
    assert block.moves[0].arc is not None


def test_arc_errors():
    parser = make_parser()
    assert error_code(parser, "G2 X10 Y10 F500") == 35
    assert error_code(parser, "G2 X10 Y10 I5 J0 F500") == 33
    assert error_code(parser, "G2 X30 Y0 R5 F500") == 34
    assert error_code(parser, "G2 X10 Y10 I5 J0 R5 F500") == 34
    assert error_code(parser, "G2 X10 Y10 K1 F500") == 36


def test_dwell():
    parser = make_parser()
    block = parse(parser, "G4 P1.5")
    assert block.dwell == pytest.approx(1.5)
    assert error_code(parser, "G4") == 28
    assert error_code(parser, "G4 P-1") == 4


def test_spindle_and_coolant_words():
    parser = make_parser()
    block = parse(parser, "M3 S10000")
    assert block.spindle_cmd == "M3"
    assert block.speed_changed
    assert parser.state.spindle == "M3"
    assert parser.state.speed == pytest.approx(10000)
    block = parse(parser, "M8")
    assert block.coolant_cmd == "M8"
    assert parser.state.coolant == "M8"
    parse(parser, "M5 M9")
    assert parser.state.spindle == "M5"
    assert parser.state.coolant == "M9"


def test_program_pause_and_end():
    parser = make_parser()
    assert parse(parser, "M0").pause
    assert parse(parser, "M2").program_end
    parser.reset_modes()
    assert parser.gc_state_string() == "[GC:G0 G54 G17 G21 G90 G94 M5 M9 T0 F0 S0]"


@pytest.mark.parametrize(
    "line, code",
    [
        ("G1 X", 2),
        ("10", 1),
        ("G1", 26),
        ("G1 X10", 22),
        ("G1 X1 X2", 25),
        ("G1 Q1", 20),
        ("G59.1", 29),
        ("G90.1", 20),
        ("M3.5", 20),
        ("T2.5", 23),
        ("T300", 38),
        ("N0 G1 X1 F100", 27),
        ("G1 X10 F500 P1", 36),
        ("G43.1 X1", 37),
        ("G43.1", 26),
        ("G10 L2 P7 X0", 28),
        ("G10 L2 X0", 28),
        ("G10 L3 P1 X0", 28),
        ("G92.1 X1", 36),
        ("G80 X1", 31),
        ("G1 X10 G4 P1", 36),
        ("G1 X10 F500 M2 M30", 21),
        ("G20 G21", 21),
        ("G2 X10 Y10 I5 J0 R5 F100", 34),
    ],
)
def test_error_codes(line: str, code: int):
    parser = make_parser()
    assert error_code(parser, line) == code


def test_parser_state_is_restored_on_error():
    parser = make_parser()
    parse(parser, "G21 G90")
    assert error_code(parser, "G1 X10 F0") == 22
    # Feed rate must not stay at zero after the failed block.
    assert parser.state.feed == 0.0
    parse(parser, "G1 X10 F600")
    assert parser.state.feed == pytest.approx(600.0)
    assert parser.state.units == 21


def test_validate_does_not_change_state():
    parser = make_parser()
    assert parser.validate("G20 G1 X1 F10", (0, 0, 0)) == 0
    assert parser.state.units == 21
    assert parser.state.feed == 0.0
    assert parser.validate("G1 X", (0, 0, 0)) == 2

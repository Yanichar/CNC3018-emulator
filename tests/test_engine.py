"""Engine level tests: protocol responses, motion, spindle, alarms."""

from __future__ import annotations

import math

import pytest

from grbl_emulator.engine import Engine


def make_engine() -> Engine:
    engine = Engine(eeprom_path=None, spindle_ramp=2.0)
    engine.reset()
    return engine


def advance(engine: Engine, seconds: float, dt: float = 0.005) -> None:
    steps = max(1, int(seconds / dt))
    for _ in range(steps):
        engine.tick(dt)


def status_fields(engine: Engine) -> dict[str, str]:
    report = engine.status_report()
    assert report is not None
    assert report.startswith("<") and report.endswith(">")
    fields: dict[str, str] = {}
    for part in report[1:-1].split("|"):
        if ":" in part:
            key, value = part.split(":", 1)
            fields[key] = value
        else:
            fields["state"] = part
    return fields


def test_reset_banner_and_startup_lines():
    engine = make_engine()
    messages = engine.reset()
    assert messages[0].strip() == "Grbl 1.1f ['$' for help]"
    assert engine.execute("$N0=G54G20") == ["ok"]
    messages = engine.reset()
    assert ">G54G20:ok" in messages
    engine.execute("$N0=")


def test_settings_dump_and_parameters():
    engine = make_engine()
    messages = engine.execute("$$")
    assert messages[-1] == "ok"
    assert "$0=10" in messages
    assert "$30=10000" in messages
    assert "$130=300.000" in messages

    messages = engine.execute("$#")
    assert messages[-1] == "ok"
    assert "[G54:0.000,0.000,0.000]" in messages
    assert "[TLO:0.000]" in messages
    assert "[PRB:0.000,0.000,0.000:0]" in messages


def test_build_info_and_gc_state():
    engine = make_engine()
    messages = engine.execute("$I")
    assert messages[0].startswith("[VER:1.1f.")
    assert messages[1] == "[OPT:V,15,128]"
    assert engine.execute("$G")[0].startswith("[GC:G0 G54 G17 G21 G90")


def test_status_report_basics():
    engine = make_engine()
    fields = status_fields(engine)
    assert fields["state"] == "Idle"
    assert fields["MPos"] == "0.000,0.000,0.000"
    assert fields["FS"] == "0.0,0"
    assert fields["Ov"] == "100,100,100"


def test_status_mask_and_inches():
    engine = make_engine()
    assert engine.execute("$10=2") == ["ok"]
    assert engine.execute("$13=1") == ["ok"]
    engine.execute("G1 X25.4 F500")
    advance(engine, 5.0)
    fields = status_fields(engine)
    assert "WPos" in fields
    assert "Bf" in fields
    assert fields["WPos"].split(",")[0] == "1.000"
    engine.execute("$13=0")
    engine.execute("$10=1")


def test_linear_motion_and_timing():
    engine = make_engine()
    assert engine.execute("G21 G90") == ["ok"]
    assert engine.execute("G1 X10 F600") == ["ok"]
    assert engine.snapshot().state == "Run"
    advance(engine, 2.0)
    snapshot = engine.snapshot()
    assert snapshot.state == "Idle"
    assert snapshot.mpos == pytest.approx((10.0, 0.0, 0.0))
    assert snapshot.feed == pytest.approx(0.0)


def test_rapid_uses_axis_max_rates():
    engine = make_engine()
    engine.execute("G0 X100")
    assert engine.snapshot().state == "Run"
    advance(engine, 12.0)
    assert engine.snapshot().mpos[0] == pytest.approx(100.0)
    assert engine.snapshot().state == "Idle"


def test_spindle_ramp_is_two_seconds():
    engine = make_engine()
    engine.execute("M3 S10000")
    advance(engine, 1.0)
    assert engine.snapshot().rpm == pytest.approx(5000.0, abs=60.0)
    advance(engine, 1.1)
    assert engine.snapshot().rpm == pytest.approx(10000.0, abs=10.0)

    engine.execute("M5")
    advance(engine, 1.0)
    assert engine.snapshot().rpm == pytest.approx(5000.0, abs=60.0)
    advance(engine, 1.1)
    assert engine.snapshot().rpm == pytest.approx(0.0, abs=1.0)


def test_spindle_marker_ratio():
    # 10 000 RPM -> 60 RPM marker == one turn per second.
    rpm = 10_000.0
    degrees_per_second = rpm / 10_000.0 * 360.0
    assert degrees_per_second == pytest.approx(360.0)


def test_feed_hold_and_resume():
    engine = make_engine()
    engine.execute("G1 X100 F800")
    advance(engine, 1.0)
    engine.realtime(0x21)  # '!'
    advance(engine, 3.0)
    assert engine.snapshot().state == "Hold"
    held_position = engine.snapshot().mpos[0]
    assert 0.0 < held_position < 100.0
    advance(engine, 1.0)
    assert engine.snapshot().mpos[0] == pytest.approx(held_position, abs=1e-6)

    engine.realtime(0x7E)  # '~'
    advance(engine, 30.0)
    assert engine.snapshot().state == "Idle"
    assert engine.snapshot().mpos[0] == pytest.approx(100.0)


def test_soft_reset_while_moving_raises_alarm():
    engine = make_engine()
    engine.execute("G1 X100 F800")
    advance(engine, 0.5)
    messages = engine.reset()
    assert "ALARM:3" in messages
    assert engine.snapshot().state == "Alarm"
    assert engine.execute("G1 X1") == ["error:9"]
    assert engine.execute("$X")[-1] == "ok"
    assert engine.snapshot().state == "Idle"


def test_soft_limits_require_homing():
    engine = make_engine()
    assert engine.execute("$20=1") == ["error:10"]
    assert engine.execute("$22=1") == ["ok"]
    messages = engine.reset()
    assert engine.snapshot().state == "Alarm"
    assert any("to unlock" in message for message in messages)
    assert engine.execute("$20=1") == ["error:9"]  # alarm locks $ commands
    assert engine.execute("$X")[-1] == "ok"
    assert engine.execute("$20=1") == ["ok"]  # homing is enabled
    assert engine.execute("$H") == ["ok"]
    advance(engine, 5.0)
    assert engine.snapshot().homed

    messages = engine.execute("G1 X400 F500")
    assert "ALARM:2" in messages
    assert engine.snapshot().state == "Alarm"
    # Critical alarm: everything is ignored until a reset.
    assert engine.execute("G1 X1") == []
    assert engine.status_report() is None
    engine.reset()
    assert engine.snapshot().state == "Alarm"  # homing is enabled
    assert engine.execute("$X")[-1] == "ok"


def test_homing_from_positive_position():
    engine = make_engine()
    engine.execute("$22=1")
    engine.reset()
    engine.execute("$X")
    engine.execute("G0 X50 Y30 Z10")
    advance(engine, 12.0)
    assert engine.snapshot().mpos[0] == pytest.approx(50.0)
    assert engine.execute("$H") == ["ok"]
    advance(engine, 20.0)
    snapshot = engine.snapshot()
    assert snapshot.state == "Idle"
    assert snapshot.homed
    assert snapshot.mpos == pytest.approx((0.0, 0.0, 0.0))


def test_homing_disabled():
    engine = make_engine()
    assert engine.execute("$H") == ["error:5"]


def test_jog_and_cancel():
    engine = make_engine()
    assert engine.execute("$J=G91 X10 F500") == ["ok"]
    assert engine.snapshot().state == "Jog"
    advance(engine, 3.0)
    assert engine.snapshot().state == "Idle"
    assert engine.snapshot().mpos[0] == pytest.approx(10.0)

    assert engine.execute("$J=G91 X50 F500") == ["ok"]
    engine.realtime(0x85)  # jog cancel
    advance(engine, 3.0)
    snapshot = engine.snapshot()
    assert snapshot.state == "Idle"
    assert snapshot.mpos[0] < 60.0


def test_jog_errors():
    engine = make_engine()
    assert engine.execute("$J=X10") == ["error:22"]  # no feed
    assert engine.execute("$J=G91 X10 F500 M3") == ["error:16"]
    assert engine.execute("$J=G91 F500") == ["error:16"]
    assert engine.execute("$J=") == ["error:16"]


def test_check_mode_does_not_move():
    engine = make_engine()
    messages = engine.execute("$C")
    assert "[MSG:Enabled]" in messages
    assert engine.execute("G1 X10 F500") == ["ok"]
    advance(engine, 2.0)
    assert engine.snapshot().mpos == pytest.approx((0.0, 0.0, 0.0))
    messages = engine.execute("$C")
    assert "[MSG:Disabled]" in messages
    assert engine.snapshot().state == "Idle"


def test_door_state():
    engine = make_engine()
    engine.execute("M3 S10000")
    engine.execute("G1 X100 F800")
    advance(engine, 0.5)
    messages = engine.realtime(0x84)
    assert "[MSG:Check Door]" in messages
    advance(engine, 3.0)
    assert engine.snapshot().state == "Door"
    assert engine.snapshot().rpm_target == 0.0
    engine.realtime(0x7E)
    assert engine.snapshot().state == "Idle"


def test_feed_and_spindle_overrides():
    engine = make_engine()
    engine.realtime(0x91)  # feed +10 %
    engine.realtime(0x96)  # rapid 50 %
    engine.realtime(0x9A)  # spindle +10 %
    snapshot = engine.snapshot()
    assert snapshot.overrides == (110, 50, 110)
    engine.realtime(0x90)
    engine.realtime(0x95)
    engine.realtime(0x99)
    assert engine.snapshot().overrides == (100, 100, 100)


def test_units_and_offsets_together():
    engine = make_engine()
    engine.execute("G20 G90")
    engine.execute("G1 X1 F10")
    advance(engine, 8.0)
    assert engine.snapshot().mpos[0] == pytest.approx(25.4)
    engine.execute("G92 X0")
    assert engine.snapshot().wpos[0] == pytest.approx(0.0)
    engine.execute("G21 G1 X0 F500")
    advance(engine, 3.0)
    assert engine.snapshot().mpos[0] == pytest.approx(25.4)


def test_program_end_resets_modes():
    engine = make_engine()
    engine.execute("M3 S10000 G1 X10 F500")
    engine.execute("M8")
    advance(engine, 2.0)
    engine.execute("M2")
    advance(engine, 0.2)
    snapshot = engine.snapshot()
    assert snapshot.coolant == "M9"
    gc = engine.execute("$G")[0]
    assert "M5" in gc and "M9" in gc


def test_overlong_line_reports_error_11():
    engine = make_engine()
    messages = engine.feed_bytes(b"A" * 100 + b"\n")
    assert messages == ["error:11"]


def test_crlf_does_not_double_ok():
    engine = make_engine()
    assert engine.feed_bytes(b"$G\r\n")[-1] == "ok"
    assert len(engine.feed_bytes(b"\r\n")) == 1  # one empty line, one ok


def test_realtime_status_byte():
    engine = make_engine()
    messages = engine.feed_bytes(b"?")
    assert len(messages) == 1
    assert messages[0].startswith("<Idle|")


def test_settings_persistence(tmp_path):
    path = tmp_path / "eeprom.json"
    engine = Engine(eeprom_path=str(path))
    engine.reset()
    assert engine.execute("$30=12000") == ["ok"]
    engine.execute("G10 L2 P1 X12.5 Y3")

    other = Engine(eeprom_path=str(path))
    assert other.settings.get(30) == pytest.approx(12000.0)
    assert other.settings.wcs_offset("G54") == pytest.approx([12.5, 3.0, 0.0])


def test_restore_defaults():
    engine = make_engine()
    engine.execute("$30=5000")
    engine.execute("$RST=$")
    assert engine.settings.get(30) == pytest.approx(10000.0)


def test_sleep_requires_reset():
    engine = make_engine()
    messages = engine.execute("$SLP")
    assert "[MSG:Sleeping]" in messages
    assert engine.execute("G1 X1 F100") == ["error:9"]
    engine.reset()
    assert engine.snapshot().state == "Idle"


def test_alarm_startup_with_homing_enabled():
    engine = Engine(eeprom_path=None)
    engine.execute("$22=1")
    messages = engine.reset()
    assert engine.snapshot().state == "Alarm"
    assert "[MSG:'$H'|'$X' to unlock]" in messages
    assert engine.execute("$$") == ["error:9"]
    assert engine.execute("$H") == ["ok"]

def test_arc_motion_reaches_endpoint():
    engine = make_engine()
    engine.execute("G2 X10 Y10 I10 J0 F600")
    advance(engine, 6.0)
    snapshot = engine.snapshot()
    assert snapshot.state == "Idle"
    assert snapshot.mpos == pytest.approx((10.0, 10.0, 0.0))


def test_probe_records_parameter():
    engine = make_engine()
    engine.execute("G38.2 Z-5 F100")
    advance(engine, 5.0)
    snapshot = engine.snapshot()
    assert snapshot.probe[:3] == pytest.approx((-0.0, -0.0, -5.0), abs=1e-6)
    assert snapshot.probe[3] is False
    assert "[PRB:0.000,0.000,-5.000:0]" in engine.execute("$#")


def test_jog_soft_limit_error():
    engine = make_engine()
    engine.execute("$22=1")
    engine.reset()
    engine.execute("$X")
    engine.execute("$20=1")
    assert engine.execute("$J=G91 X400 F500") == ["error:15"]
    assert engine.execute("$J=G91 X10 F500") == ["ok"]
    advance(engine, 2.0)


def test_jog_in_alarm_is_locked():
    engine = make_engine()
    engine.execute("G1 X100 F500")
    engine.reset()
    assert engine.execute("$J=G91 X1 F100") == ["error:9"]


def test_step_rate_limit_error():
    engine = make_engine()
    assert engine.execute("$100=3000") == ["error:12"]


def test_program_pause_and_resume():
    engine = make_engine()
    engine.execute("G1 X10 F500")
    engine.execute("M0")
    advance(engine, 3.0)
    assert engine.snapshot().state == "Hold"
    engine.realtime(0x7E)
    advance(engine, 0.5)
    snapshot = engine.snapshot()
    assert snapshot.state == "Idle"
    assert snapshot.mpos[0] == pytest.approx(10.0)

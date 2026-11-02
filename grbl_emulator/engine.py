"""The Grbl "machine": parser + motion simulation + protocol responses.

``Engine`` has no serial I/O.  :class:`~grbl_emulator.transport.SerialTransport`
feeds it bytes and writes back whatever messages it returns, while the GUI reads
immutable :class:`Snapshot` objects.  Keeping the engine free of threads makes
it fully testable: ``tick(dt)`` advances the physical simulation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

from .gcode import Block, GCodeError, GCodeParser, _scan_words, strip_comments
from .motion import Segment, SpindleSim, build_segment
from .settings import SettingsStore, WCS_NAMES

MAX_LINE = 79  # Grbl's serial RX line buffer, in characters

REALTIME_BYTES = {
    0x18,  # ctrl-x soft reset
    0x21,  # '!' feed hold
    0x3F,  # '?' status report
    0x7E,  # '~' resume
    0x84,  # safety door
    0x85,  # jog cancel
    0x90,
    0x91,
    0x92,
    0x93,
    0x94,  # feed override
    0x95,
    0x96,
    0x97,  # rapid override
    0x99,
    0x9A,
    0x9B,
    0x9C,
    0x9D,  # spindle override
    0x9E,  # spindle stop toggle
    0xA0,
    0xA1,  # coolant toggles
}

PLANNER_BLOCKS = 15
RX_BUFFER = 128


@dataclass
class MoveTask:
    segment: Segment
    kind: str = "G1"
    distance: float = 0.0
    entry_v: float = 0.0
    exit_free: bool = False


@dataclass
class DwellTask:
    remaining: float


@dataclass
class PauseTask:
    pass


@dataclass
class EndTask:
    pass


@dataclass
class HomeTask:
    phases: list[Segment] = field(default_factory=list)
    index: int = 0
    distance: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    connected: bool
    state: str
    substate: Optional[int]
    mpos: tuple[float, float, float]
    wpos: tuple[float, float, float]
    feed: float
    rpm: float
    rpm_target: float
    spindle_dir: int
    coolant: str
    overrides: tuple[int, int, int]
    homed: bool
    alarm: Optional[int]
    queued: int
    check_mode: bool
    travel: tuple[float, float, float]
    settings: dict[int, float]
    version: str
    door: bool
    probe: tuple[float, float, float, bool]
    critical: bool

    @property
    def spindle_on(self) -> bool:
        return self.spindle_dir != 0 and self.rpm_target > 0


class Engine:
    VERSION = "1.1f"

    def __init__(
        self,
        eeprom_path: str | None = None,
        spindle_ramp: float = 2.0,
        version: str = VERSION,
    ) -> None:
        self.version = version
        self.store = SettingsStore(eeprom_path)
        self.parser = GCodeParser(self.store)
        self.mpos = [0.0, 0.0, 0.0]
        self.planned = [0.0, 0.0, 0.0]
        self.queue: deque = deque()
        self.current = None
        self.v = 0.0
        self.state = "Idle"
        self.alarm_code: Optional[int] = None
        self.critical_alarm = False
        self.homed = False
        self.check_mode = False
        self.hold_pending = False
        self.hold_active = False
        self.hold_reason: Optional[str] = None
        self.door = False
        self.spindle = SpindleSim(spindle_ramp)
        self.spindle_inhibit = False
        self.spindle_stop_override = False
        self.overrides = [100, 100, 100]
        self.rx_buffer = bytearray()
        self.rx_overlong = False
        self._last_cr = False
        self.report_count = 0
        self._last_wco: Optional[tuple[float, float, float]] = None
        self._last_tangent: Optional[tuple[float, float, float]] = None
        self.probe = [0.0, 0.0, 0.0, False]
        self.jog_active = False

    # ================================================================== helpers
    @property
    def settings(self) -> SettingsStore:
        return self.store

    def _travel(self) -> tuple[float, float, float]:
        return (
            self.store.get(130),
            self.store.get(131),
            self.store.get(132),
        )

    def wco(self) -> list[float]:
        return self.parser.wco()

    def wpos(self) -> list[float]:
        offsets = self.wco()
        return [self.mpos[i] - offsets[i] for i in range(3)]

    def _target_rpm(self) -> float:
        if self.spindle_inhibit or self.spindle_stop_override:
            return 0.0
        if self.state in ("Alarm", "Door", "Sleep", "Home"):
            return 0.0
        if self.parser.state.spindle not in ("M3", "M4"):
            return 0.0
        speed = self.parser.state.speed * self.overrides[2] / 100.0
        return max(0.0, min(speed, self.store.get(30)))

    def _spindle_direction(self) -> int:
        if self.parser.state.spindle == "M3":
            return 1
        if self.parser.state.spindle == "M4":
            return -1
        return 0

    def _position_string(self, values: Iterable[float]) -> str:
        scale = 25.4 if self.store.is_set(13) else 1.0
        return ",".join(f"{value / scale:.3f}" for value in values)

    # ================================================================== reset
    def reset(self) -> list[str]:
        """Ctrl-X / power-on.  Returns the banner and state messages."""
        was_moving = isinstance(self.current, (MoveTask, HomeTask)) or bool(self.queue)
        self.queue.clear()
        self.current = None
        self.v = 0.0
        self.planned = list(self.mpos)
        self.rx_buffer.clear()
        self.rx_overlong = False
        self._last_cr = False
        self.parser.reset()
        self.spindle.stop_immediately()
        self.spindle_inhibit = False
        self.spindle_stop_override = False
        self.overrides = [100, 100, 100]
        self.check_mode = False
        self.hold_pending = False
        self.hold_active = False
        self.hold_reason = None
        self.door = False
        self.alarm_code = None
        self.critical_alarm = False
        self.homed = False
        self.jog_active = False
        self.probe = [0.0, 0.0, 0.0, False]
        self._last_tangent = None
        self._last_wco = None
        self.report_count = 0

        messages = [f"\r\nGrbl {self.version} ['$' for help]"]
        if self.store.is_set(22):
            self.state = "Alarm"
            messages.append("[MSG:'$H'|'$X' to unlock]")
        elif was_moving:
            self.state = "Alarm"
            self.alarm_code = 3
            messages.append("ALARM:3")
            messages.append("[MSG:'$H'|'$X' to unlock]")
        else:
            self.state = "Idle"
            messages.extend(self._run_startup_lines())
        return messages

    def _run_startup_lines(self) -> list[str]:
        messages: list[str] = []
        for index, line in enumerate(self.store.startup_lines):
            if not line.strip():
                continue
            code = self.parser.validate(line, tuple(self.planned))
            if code == 0:
                try:
                    block = self.parser.parse(line, tuple(self.planned))
                except GCodeError:
                    messages.append(f">{line}:error:20")
                    continue
                for plan in block.moves:
                    segment = build_segment(
                        tuple(self.planned),
                        plan.target,
                        plan.motion,
                        plan.feed,
                        plan.rapid,
                        self.store,
                        plan.arc,
                    )
                    if segment is not None:
                        self.queue.append(MoveTask(segment, kind="startup"))
                        self.planned = list(plan.target)
                messages.append(f">{line}:ok")
            else:
                messages.append(f">{line}:error:{code}")
        if self.queue and self.current is None:
            self._start_next_task()
        return messages

    # ================================================================== bytes in
    def feed_bytes(self, data: bytes) -> list[str]:
        messages: list[str] = []
        for byte in data:
            if byte in REALTIME_BYTES:
                self._last_cr = False
                messages.extend(self.realtime(byte))
                continue
            if byte == 0x0A and self._last_cr:
                self._last_cr = False
                continue
            if byte in (0x0A, 0x0D):
                self._last_cr = byte == 0x0D
                line = bytes(self.rx_buffer).decode("ascii", "replace").strip()
                self.rx_buffer.clear()
                if self.rx_overlong:
                    self.rx_overlong = False
                    messages.append("error:11")
                    continue
                messages.extend(self.execute(line))
                continue
            self._last_cr = False
            if byte == 0x09:
                continue
            if len(self.rx_buffer) >= MAX_LINE:
                self.rx_overlong = True
                continue
            self.rx_buffer.append(byte)
        return messages

    # ================================================================== line in
    def execute(self, line: str) -> list[str]:
        if self.critical_alarm:
            # Grbl requires a reset before it accepts anything else.
            return []
        if not line:
            return ["ok"]
        if line.startswith("$"):
            return self._system(line)
        if self.state in ("Alarm", "Sleep", "Home", "Door"):
            return ["error:9"]
        try:
            block = self.parser.parse(line, tuple(self.planned))
        except GCodeError as exc:
            return [f"error:{exc.code}"]
        return self._execute_block(block)

    def _execute_block(self, block: Block) -> list[str]:
        # Soft limit check happens before anything is queued.
        if self.store.is_set(20):
            travel = self._travel()
            for plan in block.moves:
                for axis in range(3):
                    if not (-1e-9 <= plan.target[axis] <= travel[axis] + 1e-9):
                        if self.check_mode:
                            return ["error:15"]
                        return self._trigger_alarm(2, critical=True)

        if self.check_mode:
            return ["ok"]

        tasks: list = []
        start = list(self.planned)
        for plan in block.moves:
            segment = build_segment(
                tuple(start),
                plan.target,
                plan.motion,
                plan.feed,
                plan.rapid,
                self.store,
                plan.arc,
            )
            if segment is not None:
                kind = "G0" if plan.rapid else f"G{int(plan.motion)}"
                tasks.append(MoveTask(segment, kind=kind))
            start = list(plan.target)
        self.planned = start

        if block.dwell is not None:
            tasks.append(DwellTask(block.dwell))
        if block.pause:
            tasks.append(PauseTask())
        if block.program_end:
            tasks.append(EndTask())

        if block.probe:
            self.probe = [*block.probe_target, False]

        if block.spindle_cmd:
            self.spindle_inhibit = False
            self.spindle_stop_override = False
        if block.coolant_cmd:
            pass

        self.queue.extend(tasks)
        if tasks and not self.hold_active:
            if self.current is None:
                self._start_next_task()
        return ["ok"]

    # ================================================================== jog
    def _jog(self, argument: str) -> list[str]:
        if self.state == "Alarm":
            return ["error:9"]
        if self.state not in ("Idle", "Jog"):
            return ["error:8"]
        if not argument:
            return ["error:16"]
        text = strip_comments(argument).upper()
        try:
            words = _scan_words(text)
        except GCodeError as exc:
            return [f"error:{exc.code}"]

        units = self.parser.state.units
        distance_mode = self.parser.state.distance
        machine_coords = False
        axis_words: dict[str, float] = {}
        feed: Optional[float] = None
        for letter, value in words:
            if letter == "G":
                rounded = round(value, 6)
                if rounded in (20, 21):
                    units = int(rounded)
                elif rounded == 90:
                    distance_mode = "G90"
                elif rounded == 91:
                    distance_mode = "G91"
                elif rounded == 53:
                    machine_coords = True
                else:
                    return ["error:16"]
            elif letter in ("X", "Y", "Z"):
                if letter in axis_words:
                    return ["error:25"]
                axis_words[letter] = value
            elif letter == "F":
                if feed is not None:
                    return ["error:25"]
                feed = value
            else:
                return ["error:16"]

        if not axis_words:
            return ["error:16"]
        if feed is None or feed <= 0.0:
            return ["error:22"]

        scale = 25.4 if units == 20 else 1.0
        offsets = self.wco()
        target = list(self.planned)
        for axis, value in axis_words.items():
            index = "XYZ".index(axis)
            if machine_coords:
                target[index] = value * scale
            elif distance_mode == "G90":
                target[index] = value * scale + offsets[index]
            else:
                target[index] = self.planned[index] + value * scale

        if self.store.is_set(20):
            travel = self._travel()
            for axis in range(3):
                if not (0.0 <= target[axis] <= travel[axis]):
                    return ["error:15"]

        segment = build_segment(
            tuple(self.planned),
            tuple(target),
            1,
            feed * scale,
            False,
            self.store,
        )
        if segment is None:
            return ["ok"]
        self.queue.append(MoveTask(segment, kind="Jog"))
        self.planned = target
        self.jog_active = True
        self.state = "Jog"
        if self.current is None:
            self._start_next_task()
        return ["ok"]

    # ================================================================== system
    def _system(self, line: str) -> list[str]:
        body = line[1:]
        if body == "":
            return [
                "[HLP:$$ $# $G $I $N $x=val $Nx=line $J=line $SLP $C $X $H ~ ! ? ctrl-x]",
                "ok",
            ]
        if body == "X":
            return self._unlock()
        if body == "H":
            return self._home()
        if body == "I":
            return self._build_info()
        if body.startswith("I="):
            self.store.build_info = body[2:][:79]
            self.store.save()
            return ["ok"]

        if self.state == "Alarm":
            return ["error:9"]
        if self.state == "Sleep":
            return ["error:9"]

        if body == "$":
            if self.state != "Idle":
                return ["error:8"]
            return self.store.dump() + ["ok"]
        if body == "#":
            if self.state != "Idle":
                return ["error:8"]
            return self._dump_parameters()
        if body == "G":
            return [self.parser.gc_state_string(), "ok"]
        if body == "N":
            if self.state != "Idle":
                return ["error:8"]
            return [
                f"$N0={self.store.startup_lines[0]}",
                f"$N1={self.store.startup_lines[1]}",
                "ok",
            ]
        if body.startswith("N") and "=" in body:
            return self._store_startup_line(body)
        if body.startswith("J="):
            return self._jog(body[2:])
        if body == "C":
            return self._toggle_check_mode()
        if body == "SLP":
            if self.state not in ("Idle", "Alarm"):
                return ["error:8"]
            self.state = "Sleep"
            self.spindle_inhibit = True
            self.spindle.stop_immediately()
            return ["[MSG:Sleeping]", "ok"]
        if body.startswith("RST="):
            return self._restore(body[4:])
        if "=" in body:
            return self._write_setting(body)
        return ["error:3"]

    def _build_info(self) -> list[str]:
        info = self.store.build_info
        version = f"[VER:{self.version}.20261102:{info}]" if info else f"[VER:{self.version}.20261102]"
        return [version, f"[OPT:V,{PLANNER_BLOCKS},{RX_BUFFER}]", "ok"]

    def _dump_parameters(self) -> list[str]:
        messages: list[str] = []
        for name in WCS_NAMES:
            messages.append(f"[{name}:{self._position_string(self.store.wcs_offset(name))}]")
        messages.append(f"[G28:{self._position_string(self.store.g28)}]")
        messages.append(f"[G30:{self._position_string(self.store.g30)}]")
        g92 = [self.parser.g92[0], self.parser.g92[1], self.parser.g92[2]]
        messages.append(f"[G92:{self._position_string(g92)}]")
        messages.append(f"[TLO:{self._position_string([self.parser.tlo])}]")
        if self.store.is_set(13):
            probe = [value / 25.4 for value in self.probe[:3]]
        else:
            probe = list(self.probe[:3])
        messages.append(
            f"[PRB:{probe[0]:.3f},{probe[1]:.3f},{probe[2]:.3f}:{1 if self.probe[3] else 0}]"
        )
        messages.append("ok")
        return messages

    def _store_startup_line(self, body: str) -> list[str]:
        if self.state != "Idle":
            return ["error:8"]
        index_text, _, content = body[1:].partition("=")
        if index_text not in ("0", "1"):
            return ["error:3"]
        index = int(index_text)
        content = content.strip().upper()
        if content:
            code = self.parser.validate(content, tuple(self.planned))
            if code:
                return [f"error:{code}"]
        self.store.set_startup_line(index, content)
        return ["ok"]

    def _write_setting(self, body: str) -> list[str]:
        if self.state != "Idle":
            return ["error:8"]
        number_text, _, value_text = body.partition("=")
        if not number_text.isdigit():
            return ["error:3"]
        code = self.store.set(int(number_text), value_text.strip())
        return ["ok"] if code == 0 else [f"error:{code}"]

    def _toggle_check_mode(self) -> list[str]:
        if self.state not in ("Idle", "Check"):
            return ["error:8"]
        if not self.check_mode:
            self.check_mode = True
            self.state = "Check"
            return ["[MSG:Enabled]", "ok"]
        self.check_mode = False
        messages = ["[MSG:Disabled]", "ok"]
        messages.extend(self.reset())
        return messages

    def _restore(self, what: str) -> list[str]:
        if self.state != "Idle":
            return ["error:8"]
        if not self.store.restore(what):
            return ["error:3"]
        messages = ["[MSG:Restoring defaults]", "ok"]
        messages.extend(self.reset())
        return messages

    # ================================================================== alarms
    def _trigger_alarm(self, code: int, critical: bool) -> list[str]:
        self.queue.clear()
        self.current = None
        self.v = 0.0
        self.spindle_inhibit = True
        self.hold_active = False
        self.hold_pending = False
        self.state = "Alarm"
        self.alarm_code = code
        self.planned = list(self.mpos)
        messages = [f"ALARM:{code}"]
        if critical:
            self.critical_alarm = True
            messages.append("[MSG:Reset to continue]")
        return messages

    def _unlock(self) -> list[str]:
        if self.state != "Alarm":
            return ["ok"]
        self.state = "Idle"
        self.alarm_code = None
        self.critical_alarm = False
        self.spindle_inhibit = True
        return ["[MSG:Caution: Unlocked]", "ok"]

    def _home(self) -> list[str]:
        if self.state not in ("Idle", "Alarm"):
            return ["error:8"]
        if not self.store.is_set(22):
            return ["error:5"]
        self.queue.clear()
        self.current = None
        self.v = 0.0
        self.alarm_code = None
        self.critical_alarm = False
        self.spindle_inhibit = True
        self.hold_active = False
        self.hold_pending = False
        self.planned = list(self.mpos)
        phases: list[Segment] = []
        seek = self.store.get(25)
        cursor = list(self.mpos)
        # Z first (lift to machine zero), then X and Y together.
        if abs(cursor[2]) > 1e-9:
            target = [cursor[0], cursor[1], 0.0]
            segment = build_segment(tuple(cursor), tuple(target), 0, seek, False, self.store)
            if segment:
                phases.append(segment)
            cursor = target
        if abs(cursor[0]) > 1e-9 or abs(cursor[1]) > 1e-9:
            target = [0.0, 0.0, cursor[2]]
            segment = build_segment(tuple(cursor), tuple(target), 0, seek, False, self.store)
            if segment:
                phases.append(segment)
            cursor = target
        self.current = HomeTask(phases=phases)
        self.state = "Home"
        self.jog_active = False
        return ["ok"]

    # ================================================================== realtime
    def realtime(self, byte: int) -> list[str]:
        if byte == 0x18:
            return self.reset()
        if self.critical_alarm:
            return []
        if byte == 0x3F:
            report = self.status_report()
            return [report] if report else []
        if byte == 0x21:
            return self._feed_hold()
        if byte == 0x7E:
            return self._resume()
        if byte == 0x84:
            return self._safety_door()
        if byte == 0x85:
            return self._jog_cancel()
        if 0x90 <= byte <= 0x94:
            return self._feed_override(byte)
        if 0x95 <= byte <= 0x97:
            return self._rapid_override(byte)
        if 0x99 <= byte <= 0x9D:
            return self._spindle_override(byte)
        if byte == 0x9E:
            if self.state == "Hold":
                self.spindle_stop_override = not self.spindle_stop_override
            return []
        if byte == 0xA0:
            self.parser.state.coolant = "M9" if self.parser.state.coolant == "M8" else "M8"
            return []
        if byte == 0xA1:
            self.parser.state.coolant = "M9" if self.parser.state.coolant == "M7" else "M7"
            return []
        return []

    def _feed_hold(self) -> list[str]:
        if self.state in ("Run", "Jog"):
            self.hold_pending = True
            self.hold_reason = "feed"
        return []

    def _resume(self) -> list[str]:
        if self.state == "Hold":
            self.hold_active = False
            self.hold_pending = False
            self.hold_reason = None
            self.spindle_stop_override = False
            self.parser.state.program = None
            if self.current is None and not self.queue:
                self.state = "Idle"
            else:
                self.state = "Jog" if self.jog_active else "Run"
        elif self.state == "Door":
            self.door = False
            self.spindle_inhibit = False
            self.state = "Idle"
        return []

    def _safety_door(self) -> list[str]:
        if self.state not in ("Idle", "Run", "Jog", "Hold"):
            return []
        self.door = True
        self.spindle_inhibit = True
        self.hold_pending = self.state in ("Run", "Jog")
        if self.state in ("Idle", "Hold"):
            self.state = "Door"
            self.queue.clear()
            self.current = None
            self.planned = list(self.mpos)
        return ["[MSG:Check Door]"]

    def _jog_cancel(self) -> list[str]:
        if self.state != "Jog":
            return []
        self.queue.clear()
        self.hold_pending = True
        self.hold_reason = "jog_cancel"
        return []

    def _feed_override(self, byte: int) -> list[str]:
        if byte == 0x90:
            self.overrides[0] = 100
        elif byte == 0x91:
            self.overrides[0] = min(200, self.overrides[0] + 10)
        elif byte == 0x92:
            self.overrides[0] = max(10, self.overrides[0] - 10)
        elif byte == 0x93:
            self.overrides[0] = min(200, self.overrides[0] + 1)
        elif byte == 0x94:
            self.overrides[0] = max(10, self.overrides[0] - 1)
        return []

    def _rapid_override(self, byte: int) -> list[str]:
        self.overrides[1] = {0x95: 100, 0x96: 50, 0x97: 25}[byte]
        return []

    def _spindle_override(self, byte: int) -> list[str]:
        if byte == 0x99:
            self.overrides[2] = 100
        elif byte == 0x9A:
            self.overrides[2] = min(200, self.overrides[2] + 10)
        elif byte == 0x9B:
            self.overrides[2] = max(10, self.overrides[2] - 10)
        elif byte == 0x9C:
            self.overrides[2] = min(200, self.overrides[2] + 1)
        elif byte == 0x9D:
            self.overrides[2] = max(10, self.overrides[2] - 1)
        return []

    # ================================================================== status
    def _state_string(self) -> tuple[str, Optional[int]]:
        if self.state == "Hold":
            return "Hold", 1 if self.hold_pending else 0
        if self.state == "Door":
            return "Door", 1 if self.door else 0
        return self.state, None

    def status_report(self) -> Optional[str]:
        if self.state == "Home" or self.critical_alarm:
            return None
        state, substate = self._state_string()
        first = f"{state}:{substate}" if substate is not None else state
        parts = [first]

        use_machine = bool(self.store.get_int(10) & 1)
        position = self.mpos if use_machine else self.wpos()
        label = "MPos" if use_machine else "WPos"
        parts.append(f"{label}:{self._position_string(position)}")

        if self.store.get_int(10) & 2:
            blocks = max(0, PLANNER_BLOCKS - len(self.queue) - (1 if self.current else 0))
            parts.append(f"Bf:{blocks},{RX_BUFFER - len(self.rx_buffer)}")

        feed = self.v * 60.0 if self.state in ("Run", "Jog") else 0.0
        if self.store.is_set(13):
            feed /= 25.4
        parts.append(f"FS:{feed:.1f},{int(round(self.spindle.rpm))}")

        wco = tuple(self.wco())
        self.report_count += 1
        if self._last_wco is None or self.report_count % 10 == 1 or wco != self._last_wco:
            parts.append(f"WCO:{self._position_string(wco)}")
            self._last_wco = wco

        parts.append(f"Ov:{self.overrides[0]},{self.overrides[1]},{self.overrides[2]}")

        accessory = ""
        direction = self._spindle_direction()
        if (
            direction > 0
            and not self.spindle_inhibit
            and self.parser.state.spindle == "M3"
            and self._target_rpm() > 0
        ):
            accessory += "S"
        elif (
            direction < 0
            and not self.spindle_inhibit
            and self.parser.state.spindle == "M4"
            and self._target_rpm() > 0
        ):
            accessory += "C"
        if self.parser.state.coolant == "M8":
            accessory += "F"
        elif self.parser.state.coolant == "M7":
            accessory += "M"
        if accessory:
            parts.append(f"A:{accessory}")

        return "<" + "|".join(parts) + ">"

    # ================================================================== tick
    def tick(self, dt: float) -> list[str]:
        dt = min(max(dt, 0.0), 0.25)
        messages: list[str] = []
        self.spindle.update(dt, self._target_rpm(), self.store.get(30))
        if self.state == "Home":
            self._tick_home(dt)
        elif self.state in ("Run", "Jog", "Hold") or self.current is not None:
            self._tick_motion(dt, messages)
        elif self.state == "Door":
            pass
        if self.state in ("Run", "Jog") and self.current is None and not self.queue:
            self.state = "Idle"
        return messages

    def _tick_home(self, dt: float) -> None:
        task = self.current
        if not isinstance(task, HomeTask):
            self.state = "Idle"
            return
        while task.index < len(task.phases):
            segment = task.phases[task.index]
            remaining = segment.length - task.distance
            if remaining <= 1e-9:
                task.index += 1
                task.distance = 0.0
                self.v = 0.0
                continue
            target_v = min(segment.base_vmax, math.sqrt(2.0 * segment.accel * remaining))
            step = segment.accel * dt
            if target_v > self.v:
                self.v = min(target_v, self.v + step)
            else:
                self.v = max(target_v, self.v - step)
            step_distance = self.v * dt
            if step_distance >= remaining:
                self.mpos = list(segment.end)
                task.index += 1
                task.distance = 0.0
                self.v = 0.0
            else:
                task.distance += step_distance
                self.mpos = list(segment.point(task.distance))
            break
        else:
            self.mpos = [0.0, 0.0, 0.0]
            self.planned = [0.0, 0.0, 0.0]
            self.homed = True
            self.current = None
            self.state = "Idle"
            self.v = 0.0

    def _tick_motion(self, dt: float, messages: list[str]) -> None:
        if self.hold_pending:
            self._advance_stop(dt)
            return
        if self.hold_active:
            return
        if self.current is None:
            if not self._start_next_task():
                self.state = "Idle"
                return
        task = self.current
        if isinstance(task, DwellTask):
            task.remaining -= dt
            if task.remaining <= 0.0:
                self.current = None
                if self.queue:
                    self._start_next_task()
                else:
                    self._end_of_queue()
            return
        if isinstance(task, PauseTask):
            self.current = None
            self.hold_active = True
            self.hold_reason = "program"
            self.parser.state.program = "M0"
            self.state = "Hold"
            return
        if isinstance(task, EndTask):
            self.current = None
            self.parser.reset_modes()
            self.spindle_inhibit = False
            messages.append("[MSG:Pgm End]")
            if self.queue:
                self._start_next_task()
            else:
                self._end_of_queue()
            return
        if isinstance(task, MoveTask):
            self._advance_move(task, dt)

    def _advance_stop(self, dt: float) -> None:
        """Decelerate to a standstill for a feed hold / jog cancel."""
        task = self.current
        if isinstance(task, MoveTask):
            decel = task.segment.accel
            self.v = max(0.0, self.v - decel * dt)
            if self.v > 0.0:
                task.distance = min(task.segment.length, task.distance + self.v * dt)
                self.mpos = list(task.segment.point(task.distance))
            if self.v <= 0.01:
                self.v = 0.0
                self.hold_pending = False
                self.hold_active = True
                if self.jog_active or (isinstance(task, MoveTask) and task.kind == "Jog"):
                    self.queue.clear()
                    self.current = None
                    self.jog_active = False
                    self.planned = list(self.mpos)
                    self.state = "Idle"
                    self.hold_active = False
                elif self.hold_reason == "jog_cancel":
                    self.queue.clear()
                    self.current = None
                    self.jog_active = False
                    self.planned = list(self.mpos)
                    self.state = "Idle"
                elif self.door:
                    self.queue.clear()
                    self.current = None
                    self.planned = list(self.mpos)
                    self.state = "Door"
                else:
                    self.state = "Hold"
        else:
            self.v = 0.0
            self.hold_pending = False
            self.hold_active = True
            self.state = "Hold"

    def _start_next_task(self) -> bool:
        while self.queue:
            task = self.queue.popleft()
            self.current = task
            if isinstance(task, MoveTask):
                entry = 0.0
                if self.v > 0.01 and self._last_tangent is not None:
                    tangent = _segment_tangent(task.segment)
                    if _dot(tangent, self._last_tangent) > 0.999:
                        entry = self.v
                task.entry_v = entry
                self.v = entry
                task.exit_free = self._next_collinear(task)
                self.jog_active = task.kind == "Jog"
                self.state = "Jog" if self.jog_active else "Run"
                return True
            if isinstance(task, DwellTask):
                self.state = "Run"
                return True
            # PauseTask / EndTask are handled by _tick_motion.
            return True
        return False

    def _next_collinear(self, task: MoveTask) -> bool:
        if task.kind == "Jog" or not self.queue:
            return False
        nxt = self.queue[0]
        if not isinstance(nxt, MoveTask):
            return False
        if nxt.kind == "Jog":
            return False
        a = task.segment
        b = nxt.segment
        if a.arc is not None or b.arc is not None:
            return False
        if a.rapid != b.rapid:
            return False
        if abs(a.base_vmax - b.base_vmax) > 1e-6:
            return False
        return _dot(_segment_tangent(a), _segment_tangent(b)) > 0.999

    def _advance_move(self, task: MoveTask, dt: float) -> None:
        segment = task.segment
        remaining = segment.length - task.distance
        override = self.overrides[1] if segment.rapid else self.overrides[0]
        vmax = segment.base_vmax * override / 100.0
        if segment.hard_limit > 0.0:
            vmax = min(vmax, segment.hard_limit)
        if task.exit_free:
            target_v = vmax
        else:
            target_v = min(vmax, math.sqrt(max(0.0, 2.0 * segment.accel * remaining)))
        step = segment.accel * dt
        if target_v > self.v:
            self.v = min(target_v, self.v + step)
        else:
            self.v = max(target_v, self.v - step)
        distance = self.v * dt
        if distance >= remaining:
            self.mpos = list(segment.end)
            task.distance = segment.length
            self.current = None
            self._last_tangent = _segment_tangent(segment, end=True)
            if not task.exit_free:
                self.v = 0.0
            if not self.queue:
                self.jog_active = False
                self.planned = list(self.mpos)
                self.state = "Idle"
            elif self.queue:
                self._start_next_task()
        else:
            task.distance += distance
            self.mpos = list(segment.point(task.distance))

    def _end_of_queue(self) -> None:
        self.planned = list(self.mpos)
        self.v = 0.0
        if not self.hold_active:
            self.state = "Idle"

    # ================================================================== snapshot
    def snapshot(self, connected: bool = True) -> Snapshot:
        state, substate = self._state_string()
        return Snapshot(
            connected=connected,
            state=state,
            substate=substate,
            mpos=(self.mpos[0], self.mpos[1], self.mpos[2]),
            wpos=tuple(self.wpos()),
            feed=self.v * 60.0,
            rpm=float(self.spindle.rpm),
            rpm_target=float(self._target_rpm()),
            spindle_dir=self._spindle_direction(),
            coolant=self.parser.state.coolant,
            overrides=(self.overrides[0], self.overrides[1], self.overrides[2]),
            homed=self.homed,
            alarm=self.alarm_code,
            queued=len(self.queue) + (1 if isinstance(self.current, MoveTask) else 0),
            check_mode=self.check_mode,
            travel=self._travel(),
            settings=dict(self.store.settings),
            version=self.version,
            door=self.door,
            probe=(self.probe[0], self.probe[1], self.probe[2], bool(self.probe[3])),
            critical=self.critical_alarm,
        )


def _segment_tangent(segment: Segment, end: bool = False) -> tuple[float, float, float]:
    if segment.arc is not None:
        return segment.arc.tangent(1.0 if end else 0.0)
    dx = segment.end[0] - segment.start[0]
    dy = segment.end[1] - segment.start[1]
    dz = segment.end[2] - segment.start[2]
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (dx / length, dy / length, dz / length)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

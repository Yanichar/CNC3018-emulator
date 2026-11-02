"""A Grbl 1.1 compatible G-code parser (modal state, arcs, error codes).

``parse()`` validates one block, mutates the modal parser state and returns a
:class:`Block` describing the motions/actions the engine has to execute.
Every syntax or semantic problem is reported with the Grbl ``error:N`` code.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .motion import PLANE_MAP, ArcError, ArcSpec, make_arc
from .settings import SettingsStore

AXES = ("X", "Y", "Z")
MAX_TOOL_NUMBER = 255

# G codes and the modal group they belong to.
G_MODAL_GROUPS: dict[int, int] = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    38.2: 0,
    38.3: 0,
    38.4: 0,
    38.5: 0,
    80: 0,
    54: 1,
    55: 1,
    56: 1,
    57: 1,
    58: 1,
    59: 1,
    17: 2,
    18: 2,
    19: 2,
    90: 3,
    91: 3,
    93: 4,
    94: 4,
    20: 5,
    21: 5,
    40: 6,
    43.1: 7,
    49: 7,
}

SUPPORTED_G = set(G_MODAL_GROUPS) | {
    4,
    10,
    28,
    28.1,
    30,
    30.1,
    53,
    59.1,
    59.2,
    59.3,
    90.1,
    92,
    92.1,
}

G_NON_MODAL = {4, 10, 28, 28.1, 30, 30.1, 92, 92.1, 43.1}

M_MODAL_GROUPS: dict[int, int] = {
    0: 0,
    1: 0,
    2: 0,
    30: 0,
    3: 1,
    4: 1,
    5: 1,
    7: 2,
    8: 2,
    9: 2,
}

SUPPORTED_M = set(M_MODAL_GROUPS) | {56}

MOTION_FEED = {1, 2, 3}
MOTION_RAPID = {0}
MOTION_PROBE = {38.2, 38.3, 38.4, 38.5}
MOTION_ARC = {2, 3}

_COMMENT_RE = re.compile(r"\([^)]*\)")


class GCodeError(Exception):
    """A Grbl ``error:N`` condition."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"error:{code}")


@dataclass
class ParserState:
    motion: int = 0
    wcs: str = "G54"
    plane: str = "G17"
    distance: str = "G90"
    arc_distance: str = "G91.1"
    feed_mode: str = "G94"
    units: int = 21
    tool_offset: str = "G49"
    program: Optional[str] = None
    spindle: str = "M5"
    coolant: str = "M9"
    feed: float = 0.0  # mm/min (always millimeters internally)
    speed: float = 0.0  # RPM
    tool: int = 0
    line_number: int = 0


@dataclass
class MovePlan:
    target: tuple[float, float, float]
    rapid: bool
    motion: int
    feed: float  # mm/min
    arc: Optional[ArcSpec] = None


@dataclass
class Block:
    moves: list[MovePlan] = field(default_factory=list)
    dwell: Optional[float] = None
    pause: bool = False
    program_end: bool = False
    spindle_cmd: Optional[str] = None
    coolant_cmd: Optional[str] = None
    speed_changed: bool = False
    probe: bool = False
    probe_target: Optional[tuple[float, float, float]] = None

    @property
    def is_empty(self) -> bool:
        return not (
            self.moves
            or self.dwell is not None
            or self.pause
            or self.program_end
        )


def strip_comments(line: str) -> str:
    line = _COMMENT_RE.sub("", line)
    semicolon = line.find(";")
    if semicolon >= 0:
        line = line[:semicolon]
    return line


def _scan_words(line: str) -> list[tuple[str, float]]:
    """Tokenise ``G1X10.5`` into ``[("G", 1.0), ("X", 10.5)]``."""
    words: list[tuple[str, float]] = []
    index = 0
    length = len(line)
    while index < length:
        char = line[index]
        if char in " \t\r":
            index += 1
            continue
        if not char.isalpha():
            raise GCodeError(1)
        letter = char.upper()
        index += 1
        while index < length and line[index] in " \t":
            index += 1
        start = index
        if index < length and line[index] in "+-":
            index += 1
        digits = 0
        while index < length and line[index].isdigit():
            index += 1
            digits += 1
        if index < length and line[index] == ".":
            index += 1
            while index < length and line[index].isdigit():
                index += 1
                digits += 1
        if digits == 0:
            raise GCodeError(2)
        if index < length and (line[index].isdigit() or line[index] == "."):
            raise GCodeError(2)
        try:
            value = float(line[start:index])
        except ValueError:
            raise GCodeError(2) from None
        if not math.isfinite(value):
            raise GCodeError(2)
        words.append((letter, value))
    return words


def _is_integer(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def _g_label(value: float) -> str:
    if _is_integer(value):
        return f"G{int(round(value))}"
    return f"G{value:g}"


class GCodeParser:
    def __init__(self, store: SettingsStore) -> None:
        self.store = store
        self.state = ParserState()
        self.g92 = [0.0, 0.0, 0.0]
        self.tlo = 0.0

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Soft reset: modal defaults, volatile offsets cleared."""
        self.state = ParserState()
        self.g92 = [0.0, 0.0, 0.0]
        self.tlo = 0.0

    def reset_modes(self) -> None:
        """M2/M30 program end: everything back to defaults, offsets retained."""
        self.state = ParserState()
        self.g92 = [0.0, 0.0, 0.0]
        self.tlo = 0.0

    def _snapshot(self) -> tuple:
        return (
            copy.deepcopy(self.state),
            list(self.g92),
            self.tlo,
            {name: list(value) for name, value in self.store.wcs.items()},
            list(self.store.g28),
            list(self.store.g30),
        )

    def _restore(self, snapshot: tuple) -> None:
        state, g92, tlo, wcs, g28, g30 = snapshot
        self.state = state
        self.g92 = g92
        self.tlo = tlo
        self.store.wcs = {name: list(value) for name, value in wcs.items()}
        self.store.g28 = g28
        self.store.g30 = g30

    def wco(self) -> list[float]:
        base = self.store.wcs_offset(self.state.wcs)
        return [
            base[0] + self.g92[0],
            base[1] + self.g92[1],
            base[2] + self.g92[2] + self.tlo,
        ]

    @property
    def units_scale(self) -> float:
        return 25.4 if self.state.units == 20 else 1.0

    def gc_state_string(self) -> str:
        state = self.state
        feed = state.feed / self.units_scale
        parts = [
            _g_label(state.motion),
            state.wcs,
            state.plane,
            "G20" if state.units == 20 else "G21",
            state.distance,
            state.feed_mode,
            state.spindle,
            state.coolant,
            f"T{state.tool}",
            f"F{feed:g}",
            f"S{state.speed:g}",
        ]
        if state.program in ("M0", "M1", "M2", "M30"):
            parts.append(state.program)
        return "[GC:" + " ".join(parts) + "]"

    # ------------------------------------------------------------------ parse
    def validate(self, line: str, mpos: tuple[float, float, float]) -> int:
        """Parse a line only to report its Grbl error code (0 == valid)."""
        snapshot = self._snapshot()
        try:
            self.parse(line, mpos)
            return 0
        except GCodeError as exc:
            return exc.code
        finally:
            self._restore(snapshot)

    def parse(
        self, line: str, mpos: tuple[float, float, float]
    ) -> Block:
        text = strip_comments(line).upper()
        words = _scan_words(text)
        snapshot = self._snapshot()
        try:
            block = self._parse_words(words, mpos)
        except GCodeError:
            self._restore(snapshot)
            raise
        except Exception:
            self._restore(snapshot)
            raise
        return block

    # ------------------------------------------------------------------ internals
    def _parse_words(
        self, words: list[tuple[str, float]], mpos: tuple[float, float, float]
    ) -> Block:
        g_values: list[float] = []
        m_values: list[float] = []
        params: dict[str, float] = {}
        for letter, value in words:
            if letter == "G":
                g_values.append(value)
            elif letter == "M":
                m_values.append(value)
            else:
                if letter in params:
                    raise GCodeError(25)
                if letter not in "XYZIJKRFSPTLN":
                    raise GCodeError(20)
                params[letter] = value

        if "N" in params:
            line_number = params["N"]
            if not _is_integer(line_number) or not (1 <= line_number <= 9_999_999):
                raise GCodeError(27)
            self.state.line_number = int(round(line_number))

        block = Block()
        used: set[str] = {"N"}

        # Modal group collision detection for G and M words.
        g_modal: dict[str, float] = {}
        g_actions: set[int] = set()
        for value in g_values:
            rounded = round(value, 6)
            if rounded not in SUPPORTED_G:
                raise GCodeError(20)
            if rounded in (59.1, 59.2, 59.3):
                raise GCodeError(29)
            if rounded == 90.1:
                raise GCodeError(20)
            group = G_MODAL_GROUPS.get(rounded)
            if group is not None:
                key = str(group)
                if key in g_modal:
                    raise GCodeError(21)
                g_modal[key] = rounded
            else:
                if rounded in g_actions:
                    raise GCodeError(21)
                g_actions.add(rounded)

        m_modal: dict[str, int] = {}
        m56 = False
        for value in m_values:
            if not _is_integer(value):
                raise GCodeError(20)
            command = int(round(value))
            if command not in SUPPORTED_M:
                raise GCodeError(20)
            group = M_MODAL_GROUPS.get(command)
            if group is not None:
                key = str(group)
                if key in m_modal:
                    raise GCodeError(21)
                m_modal[key] = command
            else:
                m56 = True

        # ------------------------------------------------------------------ units
        if "5" in g_modal:
            units = int(g_modal["5"])
            self.state.units = 20 if units == 20 else 21
        scale = self.units_scale
        distance_mode = self.state.distance
        if "3" in g_modal:
            distance_mode = "G90" if g_modal["3"] == 90 else "G91"
            self.state.distance = distance_mode
        if "2" in g_modal:
            plane = int(g_modal["2"])
            self.state.plane = f"G{plane}"
        if "4" in g_modal:
            self.state.feed_mode = "G93" if g_modal["4"] == 93 else "G94"
        if "1" in g_modal:
            self.state.wcs = f"G{int(g_modal['1'])}"
        if "6" in g_modal:
            self.state.tool_offset = "G49"
        if "7" in g_modal:
            if g_modal["7"] == 43.1:
                self.state.tool_offset = "G43.1"
            else:
                self.state.tool_offset = "G49"
        if self.state.tool_offset == "G49":
            self.tlo = 0.0

        motion_word: Optional[float] = None
        if "0" in g_modal:
            motion_word = g_modal["0"]
            self.state.motion = int(motion_word) if _is_integer(motion_word) else motion_word

        # ------------------------------------------------------------------ M words
        if "1" in m_modal:
            spindle = m_modal["1"]
            self.state.spindle = f"M{spindle}"
            block.spindle_cmd = f"M{spindle}"
        if "2" in m_modal:
            coolant = m_modal["2"]
            self.state.coolant = f"M{coolant}"
            block.coolant_cmd = f"M{coolant}"
        if "0" in m_modal:
            program = m_modal["0"]
            if program in (0, 1):
                block.pause = True
                self.state.program = f"M{program}"
            else:
                block.program_end = True
                self.state.program = f"M{program}"
        if m56:
            if "P" not in params:
                raise GCodeError(28)
            if not _is_integer(params["P"]):
                raise GCodeError(23)
            used.add("P")

        if "S" in params:
            if params["S"] < 0:
                raise GCodeError(4)
            self.state.speed = params["S"]
            block.speed_changed = True
            used.add("S")
        if "T" in params:
            tool = params["T"]
            if not _is_integer(tool):
                raise GCodeError(23)
            if tool < 0:
                raise GCodeError(4)
            if tool > MAX_TOOL_NUMBER:
                raise GCodeError(38)
            self.state.tool = int(round(tool))
            used.add("T")

        if "F" in params:
            feed_value = params["F"]
            if feed_value < 0:
                raise GCodeError(4)
            # G93 feed is inverse-time and interpreted after the path length.
            if self.state.feed_mode != "G93":
                self.state.feed = feed_value * scale
            used.add("F")

        # ------------------------------------------------------------------ G actions
        axis_words = {
            axis: params[axis] for axis in AXES if axis in params
        }
        actions: list[float] = []
        for action in g_actions:
            if action in G_NON_MODAL:
                actions.append(action)
        if 43.1 in g_modal.values():
            actions.append(43.1)

        if len(actions) > 1:
            raise GCodeError(21)

        action = actions[0] if actions else None

        if action == 4:
            if "P" not in params:
                raise GCodeError(28)
            dwell = params["P"]
            if dwell < 0:
                raise GCodeError(4)
            if axis_words:
                raise GCodeError(36)
            block.dwell = dwell
            used.add("P")

        elif action == 10:
            self._handle_g10(params, mpos, scale)
            used.update(("P", "L"))
            consumed = set(axis_words)
            used.update(consumed)

        elif action == 28.1:
            self.store.set_g28(list(mpos))
        elif action == 30.1:
            self.store.set_g30(list(mpos))

        elif action in (28, 30):
            target = list(self.store.g28 if action == 28 else self.store.g30)
            if axis_words:
                intermediate = list(mpos)
                offsets = self.wco()
                for axis, value in axis_words.items():
                    index = AXES.index(axis)
                    intermediate[index] = value * scale + offsets[index]
                block.moves.append(
                    MovePlan(tuple(intermediate), True, 0, 0.0)
                )
                used.update(axis_words)
            block.moves.append(MovePlan(tuple(target), True, 0, 0.0))

        elif action == 92:
            if not axis_words:
                raise GCodeError(26)
            base = self.store.wcs_offset(self.state.wcs)
            for axis, value in axis_words.items():
                index = AXES.index(axis)
                target = value * scale
                tlo = self.tlo if index == 2 else 0.0
                self.g92[index] = mpos[index] - target - base[index] - tlo
                used.add(axis)

        elif action == 92.1:
            self.g92 = [0.0, 0.0, 0.0]
            if axis_words:
                raise GCodeError(36)

        elif action == 43.1:
            if not axis_words:
                raise GCodeError(26)
            for axis in axis_words:
                if axis != "Z":
                    raise GCodeError(37)
            self.tlo = axis_words["Z"] * scale
            self.state.tool_offset = "G43.1"
            used.add("Z")

        # G53 is a modifier for a normal G0/G1 motion.
        g53 = 53 in g_values

        # ------------------------------------------------------------------ motion
        explicit_motion = motion_word is not None
        has_axis = bool(axis_words)
        motion = self.state.motion

        if g53:
            if motion not in (0, 1) or not explicit_motion:
                raise GCodeError(30)
            if not has_axis:
                raise GCodeError(26)

        if action is None:
            if motion == 80:
                if has_axis:
                    raise GCodeError(31)
            elif explicit_motion or has_axis:
                if not has_axis:
                    raise GCodeError(26)
                target = list(mpos)
                offsets = self.wco()
                for axis, value in axis_words.items():
                    index = AXES.index(axis)
                    if g53:
                        target[index] = value * scale
                    elif distance_mode == "G90":
                        target[index] = value * scale + offsets[index]
                    else:
                        target[index] = mpos[index] + value * scale
                used.update(axis_words)
                self._append_motion(
                    block, mpos, tuple(target), motion, params, scale, used
                )

        # ------------------------------------------------------------------ leftovers
        leftover = set(params) - used
        if leftover:
            raise GCodeError(36)

        return block

    # ------------------------------------------------------------------ helpers
    def _handle_g10(
        self, params: dict[str, float], mpos: tuple[float, float, float], scale: float
    ) -> None:
        if "L" not in params or "P" not in params:
            raise GCodeError(28)
        mode = params["L"]
        p_value = params["P"]
        if not _is_integer(mode) or int(round(mode)) not in (2, 20):
            raise GCodeError(28)
        if not _is_integer(p_value):
            raise GCodeError(23)
        system = int(round(p_value))
        if not 1 <= system <= 6:
            raise GCodeError(28)
        name = f"G{53 + system}"
        current = self.store.wcs_offset(name)
        for axis in AXES:
            if axis not in params:
                continue
            index = AXES.index(axis)
            value = params[axis] * scale
            if int(round(mode)) == 2:
                current[index] = value
            else:
                tlo = self.tlo if index == 2 else 0.0
                current[index] = mpos[index] - value - self.g92[index] - tlo
        self.store.set_wcs(name, current)

    def _append_motion(
        self,
        block: Block,
        mpos: tuple[float, float, float],
        target: tuple[float, float, float],
        motion: int,
        params: dict[str, float],
        scale: float,
        used: set[str],
    ) -> None:
        rapid = motion == 0
        arc: Optional[ArcSpec] = None

        if motion in MOTION_ARC:
            plane = self.state.plane
            u, v, w = PLANE_MAP[plane]
            plane_letters = {
                axis for axis, index in (("X", 0), ("Y", 1), ("Z", 2)) if index in (u, v)
            }
            offsets = [0.0, 0.0, 0.0]
            has_offsets = False
            for axis, letters in (("I", "X"), ("J", "Y"), ("K", "Z")):
                if axis in params:
                    if letters not in plane_letters:
                        raise GCodeError(36)
                    offsets[AXES.index(letters)] = params[axis] * scale
                    has_offsets = True
                    used.add(axis)
            radius: Optional[float] = None
            if "R" in params:
                radius = params["R"] * scale
                used.add("R")
            if radius is not None and has_offsets:
                raise GCodeError(34)
            if radius is None and not has_offsets:
                raise GCodeError(35)
            try:
                arc = make_arc(
                    mpos,
                    target,
                    plane,
                    cw=motion == 2,
                    offsets=tuple(offsets) if has_offsets else None,
                    radius=radius,
                )
            except ArcError as exc:
                raise GCodeError(exc.code) from None
            path_length = arc.length
        else:
            dx = target[0] - mpos[0]
            dy = target[1] - mpos[1]
            dz = target[2] - mpos[2]
            path_length = math.sqrt(dx * dx + dy * dy + dz * dz)

        feed = self.state.feed
        if motion in MOTION_ARC or motion in MOTION_FEED or motion in MOTION_PROBE:
            if self.state.feed_mode == "G93":
                if "F" not in params:
                    raise GCodeError(22)
                inverse_time = params["F"]
                if inverse_time <= 0.0:
                    raise GCodeError(22)
                feed = inverse_time * path_length
                self.state.feed = feed
            if feed <= 0.0:
                raise GCodeError(22)

        if motion in MOTION_PROBE and all(
            abs(target[axis] - mpos[axis]) < 1e-9 for axis in range(3)
        ):
            raise GCodeError(33)

        block.moves.append(
            MovePlan(target=target, rapid=rapid, motion=motion, feed=feed, arc=arc)
        )
        if motion in MOTION_PROBE:
            block.probe = True
            block.probe_target = target

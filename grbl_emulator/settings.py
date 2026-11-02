"""GRBL ``$`` settings, machine parameter data and EEPROM-like persistence.

The emulator behaves like a CNC3018 running Grbl 1.1: the ``$`` settings are
stored in a JSON file (the "EEPROM"), work coordinate offsets and startup
lines are persistent, while ``G92``/``G43.1`` offsets are volatile.

Only the settings that Grbl 1.1 actually exposes are implemented.  The
default values follow a common CNC3018 / Woodpecker profile: 300 x 180 x 45 mm
travel, 800 steps/mm, 10 000 RPM spindle.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

AXIS_NAMES = ("X", "Y", "Z")
WCS_NAMES = ("G54", "G55", "G56", "G57", "G58", "G59")


@dataclass(frozen=True)
class SettingDef:
    """Description of one ``$x`` setting."""

    number: int
    label: str
    default: float
    kind: str = "float"  # "int" | "float" | "bool"
    minimum: float = 0.0
    maximum: float = 1_000_000.0

    def format(self, value: float) -> str:
        if self.kind in ("int", "bool"):
            return str(int(round(value)))
        return f"{value:.3f}"


# Ordered exactly like Grbl prints them with ``$$``.
SETTINGS: "OrderedDict[int, SettingDef]" = OrderedDict(
    [
        (0, SettingDef(0, "Step pulse time, microseconds", 10, "int", 3, 1000)),
        (1, SettingDef(1, "Step idle delay, milliseconds", 25, "int", 0, 255)),
        (2, SettingDef(2, "Step pulse invert, mask", 0, "int", 0, 255)),
        (3, SettingDef(3, "Step direction invert, mask", 0, "int", 0, 255)),
        (4, SettingDef(4, "Invert step enable pin, boolean", 0, "bool", 0, 1)),
        (5, SettingDef(5, "Invert limit pins, boolean", 0, "bool", 0, 1)),
        (6, SettingDef(6, "Invert probe pin, boolean", 0, "bool", 0, 1)),
        (10, SettingDef(10, "Status report options, mask", 1, "int", 0, 255)),
        (11, SettingDef(11, "Junction deviation, millimeters", 0.010, "float", 0, 5)),
        (12, SettingDef(12, "Arc tolerance, millimeters", 0.002, "float", 0, 5)),
        (13, SettingDef(13, "Report in inches, boolean", 0, "bool", 0, 1)),
        (20, SettingDef(20, "Soft limits enable, boolean", 0, "bool", 0, 1)),
        (21, SettingDef(21, "Hard limits enable, boolean", 0, "bool", 0, 1)),
        (22, SettingDef(22, "Homing cycle enable, boolean", 0, "bool", 0, 1)),
        (23, SettingDef(23, "Homing direction invert, mask", 0, "int", 0, 255)),
        (24, SettingDef(24, "Homing locate feed rate, mm/min", 25.0, "float", 0, 10_000)),
        (25, SettingDef(25, "Homing search seek rate, mm/min", 500.0, "float", 0, 10_000)),
        (26, SettingDef(26, "Homing switch debounce delay, milliseconds", 250, "int", 0, 1000)),
        (27, SettingDef(27, "Homing switch pull-off distance, millimeters", 1.0, "float", 0, 100)),
        (30, SettingDef(30, "Maximum spindle speed, RPM", 10_000, "int", 1, 100_000)),
        (31, SettingDef(31, "Minimum spindle speed, RPM", 0, "int", 0, 100_000)),
        (32, SettingDef(32, "Laser-mode enable, boolean", 0, "bool", 0, 1)),
        (100, SettingDef(100, "X-axis travel resolution, step/mm", 800.0, "float", 1, 100_000)),
        (101, SettingDef(101, "Y-axis travel resolution, step/mm", 800.0, "float", 1, 100_000)),
        (102, SettingDef(102, "Z-axis travel resolution, step/mm", 800.0, "float", 1, 100_000)),
        (110, SettingDef(110, "X-axis maximum rate, mm/min", 800.0, "float", 1, 100_000)),
        (111, SettingDef(111, "Y-axis maximum rate, mm/min", 800.0, "float", 1, 100_000)),
        (112, SettingDef(112, "Z-axis maximum rate, mm/min", 800.0, "float", 1, 100_000)),
        (120, SettingDef(120, "X-axis acceleration, mm/sec^2", 50.0, "float", 1, 100_000)),
        (121, SettingDef(121, "Y-axis acceleration, mm/sec^2", 50.0, "float", 1, 100_000)),
        (122, SettingDef(122, "Z-axis acceleration, mm/sec^2", 50.0, "float", 1, 100_000)),
        (130, SettingDef(130, "X-axis maximum travel, millimeters", 300.0, "float", 1, 10_000)),
        (131, SettingDef(131, "Y-axis maximum travel, millimeters", 180.0, "float", 1, 10_000)),
        (132, SettingDef(132, "Z-axis maximum travel, millimeters", 45.0, "float", 1, 10_000)),
    ]
)

#: Grbl looks at these pairs to reject settings that would out-run the AVR.
_STEP_RATE_LIMIT = 30_000  # steps / second


def default_settings() -> dict[int, float]:
    return {number: definition.default for number, definition in SETTINGS.items()}


def _default_eeprom() -> dict[str, Any]:
    return {
        "settings": default_settings(),
        "wcs": {name: [0.0, 0.0, 0.0] for name in WCS_NAMES},
        "g28": [0.0, 0.0, 0.0],
        "g30": [0.0, 0.0, 0.0],
        "startup_lines": ["", ""],
        "build_info": "CNC3018 Emulator",
    }


def _coerce_vec(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return list(default)
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return list(default)


class SettingsStore:
    """Persistent Grbl state: ``$`` settings, offsets, startup lines."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.settings: dict[int, float] = default_settings()
        self.wcs: dict[str, list[float]] = {name: [0.0, 0.0, 0.0] for name in WCS_NAMES}
        self.g28: list[float] = [0.0, 0.0, 0.0]
        self.g30: list[float] = [0.0, 0.0, 0.0]
        self.startup_lines: list[str] = ["", ""]
        self.build_info: str = "CNC3018 Emulator"
        if path:
            self.load(path)

    # ------------------------------------------------------------------ load/save
    def load(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        stored = raw.get("settings")
        if isinstance(stored, dict):
            for key, value in stored.items():
                try:
                    number = int(key)
                except (TypeError, ValueError):
                    continue
                if number in SETTINGS:
                    try:
                        self.settings[number] = float(value)
                    except (TypeError, ValueError):
                        continue
        wcs = raw.get("wcs")
        if isinstance(wcs, dict):
            for name in WCS_NAMES:
                self.wcs[name] = _coerce_vec(wcs.get(name), [0.0, 0.0, 0.0])
        self.g28 = _coerce_vec(raw.get("g28"), self.g28)
        self.g30 = _coerce_vec(raw.get("g30"), self.g30)
        lines = raw.get("startup_lines")
        if isinstance(lines, list) and len(lines) == 2 and all(isinstance(x, str) for x in lines):
            self.startup_lines = list(lines)
        info = raw.get("build_info")
        if isinstance(info, str):
            self.build_info = info

    def save(self) -> None:
        if not self.path:
            return
        payload = {
            "settings": {str(number): value for number, value in self.settings.items()},
            "wcs": self.wcs,
            "g28": self.g28,
            "g30": self.g30,
            "startup_lines": self.startup_lines,
            "build_info": self.build_info,
        }
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(prefix=".grbl-eeprom-", dir=directory)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(tmp_path, self.path)
        except OSError:
            # A read-only EEPROM must not crash the emulator.
            pass

    # ------------------------------------------------------------------ settings
    def get(self, number: int) -> float:
        return self.settings.get(number, SETTINGS[number].default)

    def get_int(self, number: int) -> int:
        return int(round(self.get(number)))

    def is_set(self, number: int) -> bool:
        return bool(self.get_int(number))

    def set(self, number: int, raw: str) -> int:
        """Apply ``$x=val``.  Returns a Grbl error code (0 == ok)."""
        definition = SETTINGS.get(number)
        if definition is None:
            return 3  # STATUS_SETTING_UNKNOWN
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 2
        if not math.isfinite(value):
            return 2
        if definition.kind in ("int", "bool") and abs(value - round(value)) > 1e-9:
            return 2
        if value < 0:
            return 4
        if number == 0 and value < 3:
            return 6
        if definition.kind == "bool" and value not in (0.0, 1.0):
            return 2
        if number == 20 and value and not self.is_set(22):
            return 10  # soft limits require homing
        if number == 22 and not value and self.is_set(20):
            return 10
        if number == 30 and value < 1:
            return 2
        if value < definition.minimum or value > definition.maximum:
            return 2

        previous = self.settings.get(number)
        self.settings[number] = value

        # Grbl refuses a steps/mm or max-rate change that exceeds the step rate.
        if number in (100, 101, 102) or number in (110, 111, 112):
            axis = (number - 100) if number >= 100 and number < 103 else (number - 110)
            resolution = self.get(100 + axis)
            rate = self.get(110 + axis)
            if resolution * rate / 60.0 > _STEP_RATE_LIMIT:
                if previous is not None:
                    self.settings[number] = previous
                return 12
        self.save()
        return 0

    def dump(self) -> list[str]:
        return [
            f"${number}={definition.format(self.get(number))}"
            for number, definition in SETTINGS.items()
        ]

    # ------------------------------------------------------------------ parameters
    def wcs_offset(self, name: str) -> list[float]:
        return list(self.wcs.get(name, [0.0, 0.0, 0.0]))

    def set_wcs(self, name: str, values: list[float]) -> None:
        if name in self.wcs:
            self.wcs[name] = [float(values[0]), float(values[1]), float(values[2])]
            self.save()

    def set_g28(self, values: list[float]) -> None:
        self.g28 = [float(values[0]), float(values[1]), float(values[2])]
        self.save()

    def set_g30(self, values: list[float]) -> None:
        self.g30 = [float(values[0]), float(values[1]), float(values[2])]
        self.save()

    def set_startup_line(self, index: int, line: str) -> None:
        if 0 <= index < len(self.startup_lines):
            self.startup_lines[index] = line
            self.save()

    # ------------------------------------------------------------------ restore
    def restore(self, what: str) -> bool:
        """``$RST=$`` / ``$RST=#`` / ``$RST=*``."""
        if what == "$":
            self.settings = default_settings()
            self.save()
            return True
        if what == "#":
            self.wcs = {name: [0.0, 0.0, 0.0] for name in WCS_NAMES}
            self.g28 = [0.0, 0.0, 0.0]
            self.g30 = [0.0, 0.0, 0.0]
            self.save()
            return True
        if what == "*":
            fresh = _default_eeprom()
            self.settings = fresh["settings"]
            self.wcs = fresh["wcs"]
            self.g28 = fresh["g28"]
            self.g30 = fresh["g30"]
            self.startup_lines = fresh["startup_lines"]
            self.build_info = fresh["build_info"]
            self.save()
            return True
        return False

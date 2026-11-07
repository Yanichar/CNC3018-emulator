"""PCB sources: Gerber contours and Excellon drill hits.

The work-table view only needs two things from a PCB project: the board
outline (normally the ``Edge_Cuts`` Gerber layer) and the drilled holes (the
``*.drl`` files).  This module parses that subset of both formats and returns
plain millimetre geometry; nothing here knows about the GUI or about the
machine.

Coordinates stay in the file's own coordinate system (KiCad plots in page
coordinates, so a board often sits around X=120, Y=-130 mm).  Placing the
board on the table is the job of :func:`grbl_emulator.workfield.center_offset`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]

GERBER_SUFFIXES = frozenset(
    {
        ".gbr",
        ".ger",
        ".gtl",
        ".gbl",
        ".gts",
        ".gbs",
        ".gto",
        ".gbo",
        ".gko",
        ".gm1",
        ".gml",
        ".pho",
        ".cmp",
        ".sol",
        ".art",
    }
)
DRILL_SUFFIXES = frozenset({".drl", ".drd", ".xln", ".exc"})

_OUTLINE_NAME_HINTS = ("edge", "outline", "profile", "contour", "board", "mill")


class PcbError(Exception):
    """Raised when a Gerber/Excellon file cannot be used as a board."""


@dataclass
class Hole:
    """One drilled hole in file/millimetre coordinates."""

    x: float
    y: float
    diameter: float
    plated: bool = True


@dataclass
class Board:
    """A board outline plus its holes, all in millimetres.

    ``loops`` are closed polygons in the Gerber coordinate system; a loop that
    lies inside another one is treated as a cut-out (even-odd fill).
    """

    loops: list[list[Point]]
    holes: list[Hole] = field(default_factory=list)
    source: str = ""

    def __post_init__(self) -> None:
        self.loops = [loop for loop in self.loops if len(loop) >= 3]
        if not self.loops:
            raise PcbError("The selected Gerber layer has no closed contour")

    @property
    def bounds(self) -> Bounds:
        xs = [point[0] for loop in self.loops for point in loop]
        ys = [point[1] for loop in self.loops for point in loop]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def width(self) -> float:
        min_x, _, max_x, _ = self.bounds
        return max_x - min_x

    @property
    def height(self) -> float:
        _, min_y, _, max_y = self.bounds
        return max_y - min_y


# --------------------------------------------------------------------------- discovery
def discover_sources(directory: str | Path) -> tuple[list[Path], list[Path]]:
    """Return ``(drill_files, gerber_files)`` found directly in *directory*."""

    path = Path(directory)
    drills: list[Path] = []
    gerbers: list[Path] = []
    for entry in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if not entry.is_file():
            continue
        suffix = entry.suffix.lower()
        if suffix in DRILL_SUFFIXES:
            drills.append(entry)
        elif suffix in GERBER_SUFFIXES:
            gerbers.append(entry)
    return drills, gerbers


def guess_outline_layer(names: list[str], fallback: bool = True) -> str | None:
    """Pick the most likely board-outline layer from a list of file names.

    With ``fallback=False`` no name is returned when nothing looks like an
    outline layer, which lets the GUI wait for an explicit choice.
    """

    lowered = {name: name.lower() for name in names}
    for hint in _OUTLINE_NAME_HINTS:
        for name in names:
            if hint in lowered[name]:
                return name
    if fallback:
        return names[0] if names else None
    return None


# --------------------------------------------------------------------------- Gerber
_EXTENDED_RE = re.compile(r"%([^%]*)%|([^%]+)", re.S)
_COORD_RE = re.compile(r"([XYIJ])([+-]?(?:\d+(?:\.\d*)?|\.\d+))")
_GCODE_RE = re.compile(r"G(\d+(?:\.\d+)?)")
_DCODE_RE = re.compile(r"D(\d+)")
_FS_RE = re.compile(r"FS([LT])([AI])X(\d)(\d)Y(\d)(\d)")
_AD_RE = re.compile(r"ADD(\d+)([A-Za-z]),?(.*)")
_MAX_ARC_STEP = math.radians(4.0)


def _decode(raw: str, int_digits: int, dec_digits: int, trailing: bool) -> float:
    """Decode one Gerber/Excellon coordinate token into a float."""

    if "." in raw:
        return float(raw)
    sign = -1.0 if raw.startswith("-") else 1.0
    digits = raw.lstrip("+-")
    if not digits.isdigit():
        raise PcbError(f"Bad coordinate {raw!r}")
    if trailing:
        digits = digits.ljust(int_digits + dec_digits, "0")
    return sign * int(digits) / (10 ** dec_digits)


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _GerberParser:
    def __init__(self) -> None:
        self.scale = 1.0
        self.int_digits = 4
        self.dec_digits = 6
        self.trailing = False
        self.apertures: dict[int, float] = {}
        self.aperture: int | None = None
        self.x = 0.0
        self.y = 0.0
        self.operation = 1
        self.interpolation = 1
        self.multi_quadrant = True
        self.region = False
        self.region_points: list[Point] = []
        self.regions: list[list[Point]] = []
        self.segments: list[tuple[Point, Point, float]] = []

    # -- commands -----------------------------------------------------------
    def feed_extended(self, command: str) -> None:
        command = command.strip()
        if command.startswith("FS"):
            match = _FS_RE.match(command)
            if match:
                self.trailing = match.group(1) == "T"
                self.int_digits = int(match.group(3))
                self.dec_digits = int(match.group(4))
            return
        if command.startswith("MO"):
            self.scale = 25.4 if command.startswith("MOIN") else 1.0
            return
        if command.startswith("AD"):
            match = _AD_RE.match(command)
            if not match:
                return
            number = int(match.group(1))
            shape = match.group(2).upper()
            parts = [part for part in match.group(3).split("X") if part]
            try:
                values = [float(part) for part in parts]
            except ValueError:
                return
            if not values:
                width = 0.1
            elif shape in ("R", "O") and len(values) >= 2:
                width = min(values[0], values[1])
            else:
                width = values[0]
            self.apertures[number] = width * self.scale

    def feed(self, command: str) -> None:
        text = command.strip()
        if not text or text.startswith("G04"):
            return

        coords = {axis: value for axis, value in _COORD_RE.findall(text)}
        for raw_g in _GCODE_RE.findall(text):
            code = float(raw_g)
            if code == 1:
                self.interpolation = 1
            elif code == 2:
                self.interpolation = 2
            elif code == 3:
                self.interpolation = 3
            elif code == 36:
                self.region = True
                self.region_points = []
            elif code == 37:
                self._finish_region()
            elif code == 74:
                self.multi_quadrant = False
            elif code == 75:
                self.multi_quadrant = True

        match = _DCODE_RE.search(text)
        if match:
            code = int(match.group(1))
            if code >= 10:
                self.aperture = code
                return
            if code in (1, 2, 3):
                self.operation = code

        if not coords:
            return

        target = (
            self._coordinate("X", coords, self.x),
            self._coordinate("Y", coords, self.y),
        )
        if self.operation == 2:
            self._move(target)
        elif self.operation == 1:
            offsets = (
                self._coordinate("I", coords, 0.0),
                self._coordinate("J", coords, 0.0),
            )
            self._interpolate(target, offsets)
        # D03 flashes are not part of board contours.

    # -- geometry -----------------------------------------------------------
    def _coordinate(self, axis: str, coords: dict[str, str], default: float) -> float:
        raw = coords.get(axis)
        if raw is None:
            return default
        return _decode(raw, self.int_digits, self.dec_digits, self.trailing) * self.scale

    def _move(self, target: Point) -> None:
        self.x, self.y = target
        if self.region and not self.region_points:
            self.region_points = [target]

    def _interpolate(self, target: Point, offsets: Point) -> None:
        start = (self.x, self.y)
        if self.interpolation == 1:
            points = [target]
        else:
            points = self._arc_points(
                start, target, offsets, clockwise=self.interpolation == 2
            )

        if self.region:
            if not self.region_points:
                self.region_points = [start]
            self.region_points.extend(points)
        else:
            width = self.apertures.get(self.aperture, 0.1) if self.aperture else 0.1
            previous = start
            for point in points:
                if _distance(previous, point) > 1e-9:
                    self.segments.append((previous, point, width))
                previous = point
        self.x, self.y = target

    def _arc_points(
        self, start: Point, end: Point, offsets: Point, clockwise: bool
    ) -> list[Point]:
        center = (start[0] + offsets[0], start[1] + offsets[1])
        radius = math.hypot(start[0] - center[0], start[1] - center[1])
        if radius < 1e-9:
            return [end]
        start_angle = math.atan2(start[1] - center[1], start[0] - center[0])
        end_angle = math.atan2(end[1] - center[1], end[0] - center[0])
        if _distance(start, end) < 1e-9:
            sweep = -2.0 * math.pi if clockwise else 2.0 * math.pi
        else:
            sweep = end_angle - start_angle
            if clockwise:
                while sweep >= 0.0:
                    sweep -= 2.0 * math.pi
            else:
                while sweep <= 0.0:
                    sweep += 2.0 * math.pi
        steps = max(2, int(abs(sweep) / _MAX_ARC_STEP) + 1)
        points: list[Point] = []
        for step in range(1, steps + 1):
            angle = start_angle + sweep * step / steps
            points.append(
                (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
            )
        points[-1] = end
        return points

    def _finish_region(self) -> None:
        if self.region and len(self.region_points) >= 3:
            self.regions.append(self.region_points)
        self.region = False
        self.region_points = []


def _clean_loop(points: list[Point], tolerance: float = 1e-6) -> list[Point] | None:
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or _distance(cleaned[-1], point) > tolerance:
            cleaned.append(point)
    if len(cleaned) >= 2 and _distance(cleaned[0], cleaned[-1]) <= tolerance:
        cleaned.pop()
    return cleaned if len(cleaned) >= 3 else None


def _chain_segments(
    segments: list[tuple[Point, Point]], tolerance: float = 1e-3
) -> list[list[Point]]:
    """Join drawn line segments into closed loops."""

    def key(point: Point) -> tuple[int, int]:
        return (round(point[0] / tolerance), round(point[1] / tolerance))

    unused = set(range(len(segments)))
    incident: dict[tuple[int, int], list[int]] = {}
    for index, (start, end) in enumerate(segments):
        incident.setdefault(key(start), []).append(index)
        incident.setdefault(key(end), []).append(index)

    loops: list[list[Point]] = []
    while unused:
        first = min(unused)
        unused.remove(first)
        start, end = segments[first]
        chain = [start, end]
        while True:
            endpoint = chain[-1]
            endpoint_key = key(endpoint)
            following = None
            for index in incident.get(endpoint_key, ()):
                if index in unused:
                    following = index
                    break
            if following is None:
                break
            seg_start, seg_end = segments[following]
            unused.remove(following)
            if key(seg_start) == endpoint_key:
                chain.append(seg_end)
            else:
                chain.append(seg_start)
            if key(chain[-1]) == key(chain[0]):
                break
        loop = _clean_loop(chain)
        if loop:
            loops.append(loop)
    return loops


def _iter_commands(text: str):
    for match in _EXTENDED_RE.finditer(text):
        block, plain = match.group(1), match.group(2)
        extended = block is not None
        source = block if extended else plain
        for piece in source.split("*"):
            piece = piece.strip()
            if piece:
                yield extended, piece


def parse_gerber(text: str) -> list[list[Point]]:
    """Parse Gerber text and return closed contours in millimetres."""

    parser = _GerberParser()
    for extended, command in _iter_commands(text):
        if extended:
            parser.feed_extended(command)
        else:
            parser.feed(command)
    parser._finish_region()

    loops: list[list[Point]] = []
    for region in parser.regions:
        loop = _clean_loop(region)
        if loop:
            loops.append(loop)
    if parser.segments:
        loops.extend(_chain_segments([(start, end) for start, end, _ in parser.segments]))
    if not loops:
        raise PcbError("The Gerber file has no closed contour (is this the Edge_Cuts layer?)")
    return loops


def parse_gerber_contours(path: str | Path) -> list[list[Point]]:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PcbError(f"Cannot read {file_path.name}: {exc}") from exc
    return parse_gerber(text)


# --------------------------------------------------------------------------- Excellon
_TOOL_DEF_RE = re.compile(r"T(\d+)C([0-9.]+)", re.IGNORECASE)
_TOOL_SELECT_RE = re.compile(r"T(\d+)", re.IGNORECASE)
_FORMAT_RE = re.compile(r"FORMAT=\{(\d+):(\d+)", re.IGNORECASE)
_DRILL_COORD_RE = re.compile(r"([XY])([+-]?(?:\d+(?:\.\d*)?|\.\d+))")


def parse_excellon(text: str, plated: bool = True) -> list[Hole]:
    """Parse an Excellon drill file into a list of :class:`Hole`."""

    scale = 1.0
    int_digits = 3
    dec_digits = 3
    trailing = False
    tools: dict[int, float] = {}
    current_tool: int | None = None
    holes: list[Hole] = []

    for raw_line in text.splitlines():
        to_match = _FORMAT_RE.search(raw_line)
        if to_match:
            int_digits = int(to_match.group(1))
            dec_digits = int(to_match.group(2))

        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("M48"):
            continue
        if line == "%":
            continue
        if upper.startswith("M30") or upper.startswith("M00"):
            break
        if upper.startswith("METRIC") or upper.startswith("M71"):
            scale = 1.0
            trailing = ",TZ" in upper
            continue
        if upper.startswith("INCH") or upper.startswith("M72"):
            scale = 25.4
            trailing = ",TZ" in upper
            continue
        if upper.startswith("FMAT") or upper.startswith("G90") or upper.startswith("G05"):
            continue

        if upper.startswith("T"):
            definition = _TOOL_DEF_RE.match(line)
            if definition:
                tools[int(definition.group(1))] = float(definition.group(2)) * scale
                continue
            selection = _TOOL_SELECT_RE.match(line)
            if selection:
                current_tool = int(selection.group(1))
                continue

        coordinates = dict(_DRILL_COORD_RE.findall(line))
        if not coordinates:
            continue
        if current_tool is None or current_tool not in tools:
            continue
        try:
            x = (
                _decode(coordinates["X"], int_digits, dec_digits, trailing) * scale
                if "X" in coordinates
                else 0.0
            )
            y = (
                _decode(coordinates["Y"], int_digits, dec_digits, trailing) * scale
                if "Y" in coordinates
                else 0.0
            )
        except (PcbError, ValueError):
            continue
        holes.append(Hole(x=x, y=y, diameter=tools[current_tool], plated=plated))
    return holes


def parse_excellon_file(path: str | Path) -> list[Hole]:
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise PcbError(f"Cannot read {file_path.name}: {exc}") from exc
    plated = "npth" not in file_path.name.lower()
    return parse_excellon(text, plated=plated)


# --------------------------------------------------------------------------- board
def load_board(gerber_path: str | Path, drill_paths: list[str | Path]) -> Board:
    """Load an outline layer and merge all drill files into one board."""

    loops = parse_gerber_contours(gerber_path)
    holes: list[Hole] = []
    for drill_path in drill_paths:
        holes.extend(parse_excellon_file(drill_path))
    return Board(
        loops=loops,
        holes=holes,
        source=Path(gerber_path).name,
    )


def load_directory(directory: str | Path, layer: str | None = None) -> Board:
    """Load a directory, choosing the outline layer (or *layer* by name)."""

    drills, gerbers = discover_sources(directory)
    if not gerbers:
        raise PcbError(f"No Gerber files found in {directory}")
    names = [path.name for path in gerbers]
    chosen = layer or guess_outline_layer(names)
    if chosen not in names:
        raise PcbError(f"Layer {chosen!r} is not in {directory}")
    gerber_path = next(path for path in gerbers if path.name == chosen)
    return load_board(gerber_path, drills)

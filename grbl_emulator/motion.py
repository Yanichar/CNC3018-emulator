"""Motion geometry and spindle simulation for the GRBL emulator.

The emulator is not a step generator: it integrates a velocity profile for
every commanded segment (trapezoidal, limited by the ``$110``/``$120`` axis
settings) and evaluates the segment path analytically.  Arcs are handled as
true circular/helical paths, so the GUI gets smooth motion and exact timing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .settings import SettingsStore

#: Plane definitions: (u, v, w) axis indices.  u x v == +w, so a positive
#: angle sweep is counter-clockwise (G3) in every plane.
PLANE_MAP = {
    "G17": (0, 1, 2),  # XY plane, Z normal
    "G18": (2, 0, 1),  # ZX plane, Y normal (u=Z, v=X)
    "G19": (1, 2, 0),  # YZ plane, X normal (u=Y, v=Z)
}

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}

#: Marker rotation: a 10 000 RPM spindle turns the cross at 60 RPM
#: (one revolution per second).
MARKER_REFERENCE_RPM = 10_000.0
MARKER_RPM_AT_REFERENCE = 60.0


class ArcError(Exception):
    """Arc geometry is impossible; carries a Grbl error code."""

    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"error:{code}")


@dataclass(frozen=True)
class ArcSpec:
    """A circular (optionally helical) arc expressed in machine coordinates."""

    plane: str
    center: tuple[float, float, float]
    radius: float
    start_angle: float
    sweep: float  # radians, signed; positive == counter-clockwise
    start_w: float
    delta_w: float
    length: float
    u: int
    v: int
    w: int

    def point(self, fraction: float) -> tuple[float, float, float]:
        angle = self.start_angle + self.sweep * fraction
        coords = [0.0, 0.0, 0.0]
        coords[self.u] = self.center[self.u] + self.radius * math.cos(angle)
        coords[self.v] = self.center[self.v] + self.radius * math.sin(angle)
        coords[self.w] = self.start_w + self.delta_w * fraction
        return (coords[0], coords[1], coords[2])

    def tangent(self, fraction: float) -> tuple[float, float, float]:
        """Unit 3D tangent along increasing arc length."""
        angle = self.start_angle + self.sweep * fraction
        sign = 1.0 if self.sweep >= 0 else -1.0
        du = -self.radius * math.sin(angle) * sign
        dv = self.radius * math.cos(angle) * sign
        dw = self.delta_w / max(abs(self.sweep), 1e-9)
        length = math.hypot(math.hypot(du, dv), dw)
        coords = [0.0, 0.0, 0.0]
        coords[self.u] = du / length
        coords[self.v] = dv / length
        coords[self.w] = dw / length
        return (coords[0], coords[1], coords[2])


def _normalise_sweep(start_angle: float, end_angle: float, ccw: bool) -> float:
    if ccw:
        sweep = (end_angle - start_angle) % (2.0 * math.pi)
        return sweep if sweep > 1e-12 else 2.0 * math.pi
    sweep = -((start_angle - end_angle) % (2.0 * math.pi))
    return sweep if sweep < -1e-12 else -2.0 * math.pi


def make_arc(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    plane: str,
    cw: bool,
    offsets: Optional[tuple[float, float, float]] = None,
    radius: Optional[float] = None,
    tolerance: float = 0.01,
) -> ArcSpec:
    """Build an :class:`ArcSpec` from ``IJK`` or ``R`` words.

    Raises :class:`ArcError` with Grbl error code 33/34 for impossible arcs.
    """
    if plane not in PLANE_MAP:
        raise ArcError(33)
    u, v, w = PLANE_MAP[plane]
    start_u, start_v = start[u], start[v]
    end_u, end_v = end[u], end[v]

    if offsets is not None:
        off_u, off_v = offsets[u], offsets[v]
        center_u = start_u + off_u
        center_v = start_v + off_v
        start_radius = math.hypot(start_u - center_u, start_v - center_v)
        end_radius = math.hypot(end_u - center_u, end_v - center_v)
        if start_radius < 1e-9:
            raise ArcError(33)
        if abs(start_radius - end_radius) > tolerance:
            raise ArcError(33)
        if abs(end_u - start_u) < 1e-9 and abs(end_v - start_v) < 1e-9:
            # Full circle: direction still matters.
            sweep = 2.0 * math.pi if not cw else -2.0 * math.pi
            start_angle = math.atan2(start_v - center_v, start_u - center_u)
        else:
            start_angle = math.atan2(start_v - center_v, start_u - center_u)
            end_angle = math.atan2(end_v - center_v, end_u - center_u)
            sweep = _normalise_sweep(start_angle, end_angle, ccw=not cw)
        radius_value = start_radius
    else:
        if radius is None or abs(radius) < 1e-9:
            raise ArcError(34)
        dx = end_u - start_u
        dy = end_v - start_v
        chord = math.hypot(dx, dy)
        if chord < 1e-9 or chord > 2.0 * abs(radius) + 1e-9:
            raise ArcError(34)
        mid_u = (start_u + end_u) / 2.0
        mid_v = (start_v + end_v) / 2.0
        height = math.sqrt(max(radius * radius - (chord / 2.0) ** 2, 0.0))
        normal_u = -dy / chord
        normal_v = dx / chord
        candidates = (
            (mid_u + normal_u * height, mid_v + normal_v * height),
            (mid_u - normal_u * height, mid_v - normal_v * height),
        )
        minor = radius > 0
        chosen = None
        for center_u, center_v in candidates:
            start_angle = math.atan2(start_v - center_v, start_u - center_u)
            end_angle = math.atan2(end_v - center_v, end_u - center_u)
            sweep = _normalise_sweep(start_angle, end_angle, ccw=not cw)
            if minor == (abs(sweep) <= math.pi + 1e-9):
                chosen = (center_u, center_v, start_angle, sweep)
                break
        if chosen is None:
            center_u, center_v = candidates[0]
            start_angle = math.atan2(start_v - center_v, start_u - center_u)
            end_angle = math.atan2(end_v - center_v, end_u - center_u)
            sweep = _normalise_sweep(start_angle, end_angle, ccw=not cw)
            chosen = (center_u, center_v, start_angle, sweep)
        center_u, center_v, start_angle, sweep = chosen
        radius_value = abs(radius)

    center = [0.0, 0.0, 0.0]
    center[u] = center_u
    center[v] = center_v
    center[w] = start[w]
    arc_length = math.hypot(abs(sweep) * radius_value, end[w] - start[w])
    if arc_length < 1e-9:
        raise ArcError(33)
    return ArcSpec(
        plane=plane,
        center=(center[0], center[1], center[2]),
        radius=radius_value,
        start_angle=start_angle,
        sweep=sweep,
        start_w=start[w],
        delta_w=end[w] - start[w],
        length=arc_length,
        u=u,
        v=v,
        w=w,
    )


@dataclass
class Segment:
    """One queued motion, ready to be integrated by the engine."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    length: float
    base_vmax: float  # mm/s at 100 % override
    accel: float  # mm/s^2
    rapid: bool = False
    motion: int = 0
    arc: Optional[ArcSpec] = None
    hard_limit: float = 0.0

    def point(self, distance: float) -> tuple[float, float, float]:
        if self.length <= 0.0:
            return self.end
        fraction = min(max(distance / self.length, 0.0), 1.0)
        if self.arc is not None:
            return self.arc.point(fraction)
        return tuple(
            self.start[index] + (self.end[index] - self.start[index]) * fraction
            for index in range(3)
        )


def _axis_limits(
    settings: SettingsStore, tangent: tuple[float, float, float]
) -> tuple[float, float]:
    """Return (max path speed mm/s, path acceleration mm/s^2) for a direction."""
    max_speed = float("inf")
    max_accel = float("inf")
    for axis in range(3):
        component = abs(tangent[axis])
        if component < 1e-9:
            continue
        max_speed = min(max_speed, settings.get(110 + axis) / 60.0 / component)
        max_accel = min(max_accel, settings.get(120 + axis) / component)
    if not math.isfinite(max_speed):
        max_speed = 1.0
    if not math.isfinite(max_accel):
        max_accel = 1.0
    return max_speed, max_accel


def build_segment(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    motion: int,
    feed_mm_min: float,
    rapid: bool,
    settings: SettingsStore,
    arc: Optional[ArcSpec] = None,
) -> Optional[Segment]:
    """Create a :class:`Segment`, or ``None`` for a zero-length move."""
    if arc is not None:
        length = arc.length
        directions = [arc.tangent(index / 32.0) for index in range(33)]
    else:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dz = end[2] - start[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-9:
            return None
        directions = [(dx / length, dy / length, dz / length)]

    speed_limit = float("inf")
    accel_limit = float("inf")
    for direction in directions:
        speed, accel = _axis_limits(settings, direction)
        speed_limit = min(speed_limit, speed)
        accel_limit = min(accel_limit, accel)

    if rapid:
        base_vmax = speed_limit
    else:
        base_vmax = min(max(feed_mm_min, 0.0) / 60.0, speed_limit)
    if base_vmax <= 0.0:
        base_vmax = 0.0
    return Segment(
        start=start,
        end=end,
        length=length,
        base_vmax=base_vmax,
        accel=max(accel_limit, 1e-3),
        rapid=rapid,
        motion=motion,
        arc=arc,
        hard_limit=speed_limit,
    )


class SpindleSim:
    """A spindle with a linear acceleration/deceleration ramp.

    By default 0 -> ``$30`` takes 2 seconds; stopping from full speed takes
    the same time.  Rotation of the GUI cross is derived from the actual
    (ramped) speed, so the cross spins up and down with the spindle.
    """

    def __init__(self, ramp_seconds: float = 2.0) -> None:
        self.ramp_seconds = max(float(ramp_seconds), 0.0)
        self.rpm = 0.0

    def update(self, dt: float, target: float, max_rpm: float) -> None:
        target = max(0.0, min(float(target), max(float(max_rpm), 1.0)))
        if self.ramp_seconds <= 0.0:
            self.rpm = target
            return
        rate = max(float(max_rpm), 1.0) / self.ramp_seconds
        step = rate * max(dt, 0.0)
        if abs(target - self.rpm) <= step:
            self.rpm = target
        elif target > self.rpm:
            self.rpm += step
        else:
            self.rpm -= step
        self.rpm = max(0.0, min(self.rpm, max(float(max_rpm), 1.0)))

    def stop_immediately(self) -> None:
        self.rpm = 0.0

    @property
    def rotation_degrees_per_second(self) -> float:
        """Marker rotation for the GUI (10 000 RPM -> 60 RPM marker)."""
        return self.rpm / MARKER_REFERENCE_RPM * MARKER_RPM_AT_REFERENCE * 360.0 / 60.0

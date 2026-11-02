"""Manual end-to-end check of the emulator over a real serial pair.

Run this on the *other* end of the pair while the emulator (GUI or --no-gui)
owns COM7::

    python -m grbl_emulator --port COM7          # terminal 1
    python tools/grbl_probe.py --port COM8       # terminal 2

The script performs a Candle-like handshake and a few motions, then prints a
PASS/FAIL summary.  It is intentionally dependency-free (pyserial only).
"""

from __future__ import annotations

import argparse
import sys
import time

import serial


def read_lines(ser: serial.Serial, timeout: float = 0.6) -> list[str]:
    deadline = time.time() + timeout
    data = b""
    while time.time() < deadline:
        chunk = ser.read(4096)
        if chunk:
            data += chunk
            deadline = time.time() + 0.12
    text = data.decode("ascii", "replace")
    return [line for line in text.replace("\r", "").split("\n") if line]


def send(ser: serial.Serial, line: str, timeout: float = 0.8) -> list[str]:
    ser.write(line.encode("ascii") + b"\n")
    return read_lines(ser, timeout)


def query_status(ser: serial.Serial, timeout: float = 0.8) -> dict[str, str]:
    ser.write(b"?")
    lines = read_lines(ser, timeout)
    for line in lines:
        if line.startswith("<") and line.endswith(">"):
            fields: dict[str, str] = {}
            parts = line[1:-1].split("|")
            if not parts:
                return {}
            fields["state"] = parts[0]
            for part in parts[1:]:
                if ":" in part:
                    key, value = part.split(":", 1)
                    fields[key] = value
            return fields
    return {}


def wait_state(ser: serial.Serial, wanted: str, timeout: float = 30.0) -> dict[str, str]:
    deadline = time.time() + timeout
    last: dict[str, str] = {}
    while time.time() < deadline:
        last = query_status(ser)
        if last.get("state", "").split(":")[0] == wanted:
            return last
        time.sleep(0.1)
    return last


def position(status: dict[str, str]) -> tuple[float, float, float]:
    raw = status.get("MPos") or status.get("WPos") or "0,0,0"
    values = [float(part) for part in raw.split(",")]
    return values[0], values[1], values[2]


def rpm(status: dict[str, str]) -> float:
    raw = status.get("FS", "0,0")
    try:
        return float(raw.split(",")[1])
    except (IndexError, ValueError):
        return 0.0


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
        if not ok:
            self.failures += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe a GRBL emulator over serial")
    parser.add_argument("--port", default="COM8")
    parser.add_argument("--baud", type=int, default=115200)
    args = parser.parse_args(argv)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
    except Exception as exc:
        print(f"Cannot open {args.port}: {exc}", file=sys.stderr)
        return 2

    report = Report()
    print(f"Probing {args.port} @ {args.baud} 8-N-1")
    try:
        ser.write(b"\x18")
        lines = read_lines(ser, 1.5)
        report.check(
            "soft reset -> Grbl banner",
            any(line.startswith("Grbl") for line in lines),
            "; ".join(lines[:2]),
        )

        lines = send(ser, "$I")
        report.check("$I build info", any(line.startswith("[VER:") for line in lines), lines[0] if lines else "")

        lines = send(ser, "$G")
        report.check("$G modal state", any(line.startswith("[GC:") for line in lines), lines[0] if lines else "")

        lines = send(ser, "$#")
        report.check("$# parameters", any("[G54:" in line for line in lines), "")

        status = query_status(ser)
        report.check("? status report", status.get("state") == "Idle", str(status.get("state")))

        send(ser, "G21 G90")
        send(ser, "G0 X10 Y5")
        status = wait_state(ser, "Idle")
        x, y, _ = position(status)
        report.check("G0 X10 Y5", abs(x - 10.0) < 0.01 and abs(y - 5.0) < 0.01, f"MPos={x:.3f},{y:.3f}")

        send(ser, "M3 S10000")
        time.sleep(1.0)
        half = rpm(query_status(ser))
        report.check("spindle ramp up ~50 % after 1 s", 3500 <= half <= 6500, f"{half:.0f} RPM")
        time.sleep(1.2)
        full = rpm(query_status(ser))
        report.check("spindle reaches 10 000 RPM", 9500 <= full <= 10000, f"{full:.0f} RPM")

        send(ser, "M5")
        time.sleep(2.2)
        stopped = rpm(query_status(ser))
        report.check("spindle stops", stopped < 100, f"{stopped:.0f} RPM")

        send(ser, "G1 X40 F600")
        time.sleep(0.6)
        ser.write(b"!")
        held = wait_state(ser, "Hold", 5.0)
        report.check("feed hold (!)", held.get("state", "").startswith("Hold"), str(held.get("state")))
        ser.write(b"~")
        resumed = wait_state(ser, "Idle", 30.0)
        x, _, _ = position(resumed)
        report.check("cycle start (~) completes the move", abs(x - 40.0) < 0.02, f"MPos X={x:.3f}")

        send(ser, "G1 X80 F600")
        time.sleep(0.4)
        ser.write(b"\x18")
        lines = read_lines(ser, 1.0)
        report.check(
            "reset during motion -> ALARM:3",
            any(line.startswith("ALARM:3") for line in lines) or any("Alarm" in line for line in lines),
            "; ".join(lines[:3]),
        )

        send(ser, "$X")
        status = query_status(ser)
        report.check("$X unlock returns to Idle", status.get("state") == "Idle", str(status.get("state")))
    finally:
        ser.close()

    print()
    if report.failures:
        print(f"RESULT: {report.failures} check(s) FAILED")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

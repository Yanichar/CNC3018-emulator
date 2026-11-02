"""Command line entry point for the CNC3018 GRBL emulator."""

from __future__ import annotations

import argparse
import os
import sys
import time

from .engine import Engine
from .transport import SerialTransport


def default_eeprom_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "CNC3018-emulator", "grbl-eeprom.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cnc3018-emulator",
        description=(
            "CNC3018 / Grbl 1.1 emulator: opens a serial port at 115200 8-N-1, "
            "answers like a real controller and draws the machine in a GUI."
        ),
    )
    parser.add_argument("--port", default="COM7", help="serial port to open (default: COM7)")
    parser.add_argument("--baud", type=int, default=115200, help="baud rate (default: 115200)")
    parser.add_argument("--no-gui", action="store_true", help="run the serial server without a GUI")
    parser.add_argument(
        "--eeprom",
        default=None,
        help=f"settings/EEPROM file (default: {default_eeprom_path()})",
    )
    parser.add_argument("--no-eeprom", action="store_true", help="keep settings in memory only")
    parser.add_argument(
        "--spindle-ramp",
        type=float,
        default=2.0,
        help="spindle acceleration/deceleration time in seconds (default: 2.0)",
    )
    parser.add_argument(
        "--autoconnect",
        action="store_true",
        help="GUI: open the port immediately at startup",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="CNC3018 GRBL emulator 1.0",
    )
    return parser


def run_headless(args: argparse.Namespace) -> int:
    engine = Engine(eeprom_path=None if args.no_eeprom else (args.eeprom or default_eeprom_path()),
                    spindle_ramp=args.spindle_ramp)

    def log(kind: str, text: str) -> None:
        prefix = {"tx": ">>", "rx": "<<", "rt": "**", "err": "!!"}.get(kind, "<<")
        print(f"{prefix} {text}", flush=True)

    transport = SerialTransport(engine, on_message=log)
    try:
        transport.open(args.port, args.baud)
    except Exception as exc:
        print(f"Cannot open {args.port} at {args.baud}: {exc}", file=sys.stderr)
        return 1
    print(
        f"CNC3018 GRBL emulator listening on {args.port} @ {args.baud} 8-N-1.  Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        transport.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eeprom = None if args.no_eeprom else (args.eeprom or default_eeprom_path())
    if args.no_gui:
        return run_headless(args)
    try:
        from .gui import run as run_gui
    except ImportError as exc:  # pragma: no cover - tkinter is part of CPython
        print(f"GUI is not available ({exc}); use --no-gui.", file=sys.stderr)
        return 1
    run_gui(
        port=args.port,
        baudrate=args.baud,
        eeprom_path=eeprom,
        autoconnect=args.autoconnect,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

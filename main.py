"""Run the CNC3018 GRBL emulator: ``python main.py [--port COM7] [--no-gui]``."""

from grbl_emulator.main import main

if __name__ == "__main__":
    raise SystemExit(main())

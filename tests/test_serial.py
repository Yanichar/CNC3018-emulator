"""Real serial loopback test over a com0com-style port pair.

Skipped automatically when the pair is not available (e.g. on CI).
Set GRBL_EMULATOR_PORT / GRBL_EMULATOR_PEER to use other ports.
"""

from __future__ import annotations

import os
import time

import pytest

serial = pytest.importorskip("serial")

from grbl_emulator.engine import Engine
from grbl_emulator.transport import SerialTransport

PORT = os.environ.get("GRBL_EMULATOR_PORT", "COM7")
PEER = os.environ.get("GRBL_EMULATOR_PEER", "COM8")


def pair_available() -> bool:
    try:
        first = serial.Serial(PORT, 115200, timeout=0.1)
        second = serial.Serial(PEER, 115200, timeout=0.1)
        first.close()
        second.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not pair_available(), reason=f"serial pair {PORT}/{PEER} not available")


def read_until(client, timeout: float = 1.0) -> str:
    deadline = time.time() + timeout
    data = b""
    while time.time() < deadline:
        chunk = client.read(4096)
        if chunk:
            data += chunk
            deadline = time.time() + 0.15
    return data.decode("ascii", "replace")


def test_serial_loopback():
    engine = Engine(eeprom_path=None)
    transport = SerialTransport(engine)
    transport.open(PORT, 115200)
    client = serial.Serial(PEER, 115200, timeout=0.05)
    try:
        client.write(b"\x18")
        banner = read_until(client)
        assert "Grbl 1.1f" in banner

        client.write(b"$I\n")
        answer = read_until(client)
        assert "[VER:1.1f." in answer
        assert "ok" in answer

        client.write(b"?")
        answer = read_until(client)
        assert answer.strip().startswith("<Idle|")

        client.write(b"G21 G90\n")
        assert "ok" in read_until(client)
        client.write(b"G1 X10 F600\n")
        assert "ok" in read_until(client)
        time.sleep(1.6)
        client.write(b"?")
        answer = read_until(client)
        assert "MPos:10.000,0.000,0.000" in answer

        client.write(b"$G\n")
        answer = read_until(client)
        assert "[GC:" in answer

        client.write(b"$30=12000\n")
        answer = read_until(client)
        assert "ok" in answer
    finally:
        client.close()
        transport.close()

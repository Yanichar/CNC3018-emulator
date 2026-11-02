"""Serial transport: the emulator "device" side of the COM port.

Owns the pyserial handle, reads bytes, feeds them to :class:`Engine`, writes
the engine's responses back and publishes periodic :class:`Snapshot` objects
for the GUI.  A serial port is opened at 115200 8-N-1 by default, exactly like
a CNC3018 controller.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Optional

from .engine import Engine, Snapshot

MessageCallback = Callable[[str, str], None]


class SerialTransport:
    def __init__(self, engine: Engine, on_message: Optional[MessageCallback] = None) -> None:
        self.engine = engine
        self._on_message = on_message
        self._serial = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._outbox: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self._snapshot = engine.snapshot(connected=False)
        self._last_snapshot = 0.0
        self._last_tick = 0.0
        self.error: Optional[str] = None
        self.port: Optional[str] = None
        self.baudrate = 115200
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ state
    @property
    def connected(self) -> bool:
        return bool(self._running and self._serial is not None and self._serial.is_open)

    @property
    def snapshot(self) -> Snapshot:
        return self._snapshot

    # ------------------------------------------------------------------ open/close
    def open(self, port: str, baudrate: int = 115200) -> None:
        if self.connected:
            raise RuntimeError("port already open")
        import serial  # imported lazily so tests can run without pyserial

        self.error = None
        self.port = port
        self.baudrate = baudrate
        self._serial = serial.serial_for_url(
            port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.005,
            write_timeout=0.2,
        )
        try:
            self._serial.reset_input_buffer()
        except Exception:
            pass
        self._running = True
        self._last_tick = time.monotonic()
        self._last_snapshot = 0.0
        self._thread = threading.Thread(target=self._loop, name="grbl-serial", daemon=True)
        self._thread.start()
        # A com0com pair blocks writes until the other side is opened.  A real
        # controller would also be waiting, so a failed banner is harmless:
        # any GUI toggles DTR / sends Ctrl-X and triggers a fresh banner.
        self._write(self.engine.reset(), notify=False)

    def close(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)
        self._thread = None
        serial_handle = self._serial
        self._serial = None
        if serial_handle is not None:
            try:
                serial_handle.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ API for GUI
    def send_line(self, line: str) -> None:
        if not self.connected:
            raise RuntimeError("serial port is not open")
        self._outbox.put(("line", line))

    def send_realtime(self, byte: int) -> None:
        if not self.connected:
            raise RuntimeError("serial port is not open")
        self._outbox.put(("byte", int(byte) & 0xFF))

    # ------------------------------------------------------------------ internals
    def _write(self, messages: list[str], notify: bool = True) -> None:
        if not messages or self._serial is None:
            return
        payload = "".join(
            message if message.endswith("\n") else message + "\r\n" for message in messages
        )
        try:
            with self._lock:
                self._serial.write(payload.encode("ascii", "replace"))
        except Exception as exc:
            # Typically the peer end of a virtual pair is not open yet.
            self.error = str(exc)
            callback = self._on_message
            if notify and callback is not None:
                callback("error", f"write failed: {exc}")
            return
        callback = self._on_message
        if callback is not None:
            for message in messages:
                callback("rx", message.strip())

    def _loop(self) -> None:
        while self._running and self._serial is not None:
            now = time.monotonic()
            dt = now - self._last_tick
            self._last_tick = now
            try:
                data = self._serial.read(1024)
            except Exception as exc:  # port disappeared, USB unplugged, ...
                self.error = str(exc)
                self._running = False
                callback = self._on_message
                if callback is not None:
                    callback("error", str(exc))
                break

            messages: list[str] = []
            if data:
                messages.extend(self.engine.feed_bytes(data))

            try:
                while True:
                    kind, payload = self._outbox.get_nowait()
                    if kind == "line":
                        text = str(payload)
                        callback = self._on_message
                        if callback is not None:
                            callback("tx", text)
                        messages.extend(self.engine.feed_bytes(text.encode("utf-8") + b"\n"))
                    else:
                        callback = self._on_message
                        if callback is not None:
                            callback("rt", f"0x{int(payload):02X}")
                        messages.extend(self.engine.feed_bytes(bytes([int(payload)])))
            except queue.Empty:
                pass

            messages.extend(self.engine.tick(dt))
            if messages:
                try:
                    self._write(messages)
                except Exception as exc:
                    self.error = str(exc)
                    self._running = False
                    callback = self._on_message
                    if callback is not None:
                        callback("error", str(exc))
                    break

            if now - self._last_snapshot >= 1.0 / 30.0:
                self._snapshot = self.engine.snapshot(connected=True)
                self._last_snapshot = now

        self._running = False
        self._snapshot = self.engine.snapshot(connected=False)

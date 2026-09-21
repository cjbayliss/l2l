from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TextIO

from zh2en.monads import IO


@dataclass(frozen=True)
class StatusView:
    label: str = "Working"
    started: float = 0.0
    tokens: int = 0
    prefix: str = ""
    drawn: str = ""


def status_line_text(view: StatusView, now_value: float) -> str:
    line = "%s: time elapsed: %.2fs, tokens received: %d" % (
        view.label,
        now_value - view.started,
        view.tokens,
    )
    return view.prefix + line if view.prefix else line


def status_erase_text(view: StatusView) -> str:
    return "\r" + " " * len(view.drawn) + "\r" if view.drawn else ""


class StatusLine:
    def __init__(
        self,
        stream: TextIO,
        live: bool,
        monotonic: Callable[[], float] = time.monotonic,
    ):
        self._stream = stream
        self._live = live
        self._monotonic = monotonic
        self._view = StatusView()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def write_partial(self, text: str) -> IO[None]:
        def thunk() -> None:
            with self._lock:
                self._view = replace(self._view, prefix=self._view.prefix + text)
                self._stream.write(text)
                self._stream.flush()

        return IO(thunk)

    def start(self, label: str) -> IO[None]:
        def thunk() -> None:
            if not self._live:
                return None

            with self._lock:
                self._view = replace(
                    self._view, label=label, started=self._monotonic(), tokens=0
                )
                self._draw(self._monotonic())

            self._stop.clear()
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
            return None

        return IO(thunk)

    def progress(self, label: str, count: int = 1) -> IO[None]:
        def thunk() -> None:
            if not self._live:
                return None

            with self._lock:
                self._view = replace(
                    self._view, label=label, tokens=self._view.tokens + count
                )

            return None

        return IO(thunk)

    def stop(self) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self._lock:
                if self._erase() and self._view.prefix:
                    self._stream.write(self._view.prefix)
                    self._view = replace(self._view, drawn=self._view.prefix)

                self._stream.flush()

            return None

        return IO(thunk)

    def interrupt(self) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self._lock:
                had_drawn = self._erase()
                if self._view.prefix:
                    if not (self._live and had_drawn):
                        self._stream.write("\n")

                    self._view = replace(self._view, prefix="")

                self._stream.flush()

            return None

        return IO(thunk)

    def finish(self, text: str) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self._lock:
                self._stream.write(
                    (self._view.prefix if self._erase() else "") + text + "\n"
                )
                self._view = replace(self._view, prefix="", drawn="")
                self._stream.flush()

            return None

        return IO(thunk)

    def _halt(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _tick(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                self._draw(self._monotonic())

    def _draw(self, now_value: float) -> None:
        full = status_line_text(self._view, now_value)
        self._stream.write(
            "\r" + full + " " * max(len(self._view.drawn) - len(full), 0)
        )
        self._stream.flush()
        self._view = replace(self._view, drawn=full)

    def _erase(self) -> bool:
        if not self._view.drawn:
            return False

        self._stream.write(status_erase_text(self._view))
        self._view = replace(self._view, drawn="")
        return True


@dataclass(frozen=True)
class Console:
    stream: TextIO
    status: StatusLine

    def log(self, message: str) -> IO[None]:
        def thunk() -> None:
            self.status.interrupt().run()
            print(message, file=self.stream, flush=True)

        return IO(thunk)

    def write_partial(self, text: str) -> IO[None]:
        return self.status.write_partial(text)

    def start(self, label: str) -> IO[None]:
        return self.status.start(label)

    def progress(self, label: str, count: int = 1) -> IO[None]:
        return self.status.progress(label, count)

    def stop(self) -> IO[None]:
        return self.status.stop()

    def interrupt(self) -> IO[None]:
        return self.status.interrupt()

    def finish(self, text: str) -> IO[None]:
        return self.status.finish(text)

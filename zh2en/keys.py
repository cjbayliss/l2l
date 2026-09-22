"""Interactive Tab toggle: listens on the controlling TTY for Tab keys.

stdin carries the translation input, so the listener opens `/dev/tty`
instead. The terminal is put into cbreak mode (Tab arrives without
waiting for Enter while Ctrl-C still raises KeyboardInterrupt) and the
original attributes are restored on exit.
"""

from __future__ import annotations

import contextlib
import os
import select
import sys
import termios
import threading
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from zh2en.console import Console
from zh2en.monads import NOTHING, Just, Maybe, maybe_either, maybe_or

POLL_SECONDS = 0.1


def is_toggle_key(data: bytes) -> bool:
    return data == b"\t"


def open_tty() -> Maybe[int]:
    def from_tty_device() -> Maybe[int]:
        try:
            return Just(os.open("/dev/tty", os.O_RDONLY))
        except OSError:
            return NOTHING

    def from_stdin() -> Maybe[int]:
        try:
            fd = sys.stdin.fileno()
        except AttributeError, OSError, ValueError:
            return NOTHING

        return Just(fd) if os.isatty(fd) else NOTHING

    return maybe_or(from_tty_device(), from_stdin())


@dataclass
class TabListener:
    fd: int
    saved: list[Any]
    toggle: Callable[[], None]
    halt: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    restored: bool = False

    def start(self) -> None:
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.halt.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

        self.restore()

    def restore(self) -> None:
        if self.restored:
            return

        self.restored = True
        with contextlib.suppress(OSError, termios.error):
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def _listen(self) -> None:
        while not self.halt.is_set():
            try:
                ready, _, _ = select.select([self.fd], [], [], POLL_SECONDS)
                if not ready:
                    continue

                if is_toggle_key(os.read(self.fd, 1)):
                    self.toggle()
            except OSError, termios.error:
                return


def start_tab_listener(
    console: Console, toggle: Callable[[], None]
) -> Callable[[], None]:
    """Poll the TTY for Tab presses and invoke `toggle` on each press.

    `toggle` performs the mode flip and re-render; the program edge
    supplies it, so this module never executes IO itself. Returns a
    no-op when stderr is not a terminal or no TTY is available.
    The returned callable stops the listener and restores the terminal.
    """
    if not console.status.live:
        return lambda: None

    def attach(fd: int) -> Callable[[], None]:
        try:
            saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except OSError, termios.error:
            return lambda: None

        listener = TabListener(fd=fd, saved=saved, toggle=toggle)
        listener.start()
        return listener.stop

    return maybe_either(open_tty(), attach, lambda: lambda: None)

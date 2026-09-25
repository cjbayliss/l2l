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

from l2l.console import Console
from l2l.monads import (
    IO,
    NOTHING,
    Just,
    Maybe,
    Nothing,
    Ref,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    maybe_either,
    maybe_or,
    modify_ref_with,
    read_ref,
    repeat_until,
    write_ref,
)

POLL_SECONDS = 0.1


type TermiosState = list[Any]


def is_toggle_key(data: bytes) -> bool:
    return data == b"\t"


def open_tty() -> IO[Maybe[int]]:
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

    return IO(lambda: maybe_or(from_tty_device(), from_stdin()))


def tty_attributes(fd: int) -> IO[Maybe[TermiosState]]:
    def thunk() -> Maybe[TermiosState]:
        try:
            return Just(termios.tcgetattr(fd))
        except OSError, termios.error:
            return NOTHING

    return IO(thunk)


def enter_cbreak(fd: int) -> IO[bool]:
    def thunk() -> bool:
        try:
            tty.setcbreak(fd)
        except OSError, termios.error:
            return False

        return True

    return IO(thunk)


def no_op() -> None:
    return None


@dataclass(frozen=True)
class TabListener:
    fd: int
    saved: TermiosState
    toggle: Callable[[], None]
    halt: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    thread: Ref[threading.Thread | None] = field(
        default_factory=lambda: Ref(None), repr=False, compare=False
    )
    restored: Ref[bool] = field(
        default_factory=lambda: Ref(False), repr=False, compare=False
    )

    def listen_action(self) -> IO[None]:
        def thunk() -> None:
            try:
                ready, _, _ = select.select([self.fd], [], [], POLL_SECONDS)
                if ready and is_toggle_key(os.read(self.fd, 1)):
                    self.toggle()
            except OSError, termios.error:
                self.halt.set()

        return IO(thunk)

    def start(self) -> IO[None]:
        def launched(_: threading.Thread | None) -> IO[None]:
            thread = threading.Thread(
                target=repeat_until(self.listen_action(), self.halt, POLL_SECONDS),
                daemon=True,
            )

            def boot(_: threading.Thread | None) -> IO[None]:
                def thunk() -> None:
                    self.halt.clear()
                    thread.start()

                return IO(thunk)

            return io_bind(write_ref(self.thread, thread), boot)

        return io_bind(read_ref(self.thread), launched)

    def restore(self) -> IO[None]:
        def claim(restored: bool) -> tuple[bool, bool]:
            return True, not restored

        def apply(should_restore: bool) -> IO[None]:
            def thunk() -> None:
                if should_restore:
                    with contextlib.suppress(OSError, termios.error):
                        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

            return IO(thunk)

        return io_bind(modify_ref_with(self.restored, claim), apply)

    def stop(self) -> IO[None]:
        def halt_worker(worker: threading.Thread | None) -> IO[None]:
            def thunk() -> None:
                self.halt.set()
                if worker is not None and worker.is_alive():
                    worker.join(timeout=1.0)

            return IO(thunk)

        return io_and_then(io_bind(read_ref(self.thread), halt_worker), self.restore())


def start_tab_listener(console: Console, toggle: Callable[[], None]) -> IO[IO[None]]:
    if not console.status.live:
        return io_pure(IO(no_op))

    def attach(fd: int) -> IO[IO[None]]:
        def with_attributes(attributes: Maybe[TermiosState]) -> IO[IO[None]]:
            if isinstance(attributes, Nothing):
                return io_pure(IO(no_op))

            def with_cbreak(entered: bool) -> IO[IO[None]]:
                if not entered:
                    return io_pure(IO(no_op))

                listener = TabListener(fd=fd, saved=attributes.value, toggle=toggle)
                return io_map(listener.start(), lambda _: listener.stop())

            return io_bind(enter_cbreak(fd), with_cbreak)

        return io_bind(tty_attributes(fd), with_attributes)

    def with_tty(fd: Maybe[int]) -> IO[IO[None]]:
        return maybe_either(fd, attach, lambda: io_pure(IO(no_op)))

    return io_bind(open_tty(), with_tty)

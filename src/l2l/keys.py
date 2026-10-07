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
    Reference,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    maybe_either,
    maybe_or,
    modify_reference_with,
    read_reference,
    repeat_until,
    write_reference,
)

POLL_SECONDS = 0.1


type TermiosState = tuple[Any, ...]


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
            file_descriptor = sys.stdin.fileno()
        except AttributeError, OSError, ValueError:
            return NOTHING

        return Just(file_descriptor) if os.isatty(file_descriptor) else NOTHING

    return IO(lambda: maybe_or(from_tty_device(), from_stdin()))


def tty_attributes(file_descriptor: int) -> IO[Maybe[TermiosState]]:
    def thunk() -> Maybe[TermiosState]:
        try:
            return Just(tuple(termios.tcgetattr(file_descriptor)))
        except OSError, termios.error:
            return NOTHING

    return IO(thunk)


def enter_cbreak(file_descriptor: int) -> IO[bool]:
    def thunk() -> bool:
        try:
            tty.setcbreak(file_descriptor)
        except OSError, termios.error:
            return False

        return True

    return IO(thunk)


def no_op() -> None:
    return None


@dataclass(frozen=True)
class TabListener:
    file_descriptor: int
    saved: TermiosState
    toggle: Callable[[], None]
    halt: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    thread: Reference[threading.Thread | None] = field(
        default_factory=lambda: Reference(None), repr=False, compare=False
    )
    restored: Reference[bool] = field(
        default_factory=lambda: Reference(False), repr=False, compare=False
    )

    def listen_action(self) -> IO[None]:
        def thunk() -> None:
            try:
                ready, _, _ = select.select(
                    [self.file_descriptor], [], [], POLL_SECONDS
                )
                if ready and is_toggle_key(os.read(self.file_descriptor, 1)):
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

            return io_bind(write_reference(self.thread, thread), boot)

        return io_bind(read_reference(self.thread), launched)

    def restore(self) -> IO[None]:
        def claim(restored: bool) -> tuple[bool, bool]:
            return True, not restored

        def apply(should_restore: bool) -> IO[None]:
            def thunk() -> None:
                if should_restore:
                    with contextlib.suppress(OSError, termios.error):
                        termios.tcsetattr(
                            self.file_descriptor, termios.TCSADRAIN, list(self.saved)
                        )

            return IO(thunk)

        return io_bind(modify_reference_with(self.restored, claim), apply)

    def stop(self) -> IO[None]:
        def halt_worker(worker: threading.Thread | None) -> IO[None]:
            def thunk() -> None:
                self.halt.set()
                if worker is not None and worker.is_alive():
                    worker.join(timeout=1.0)

            return IO(thunk)

        return io_and_then(
            io_bind(read_reference(self.thread), halt_worker), self.restore()
        )


def start_tab_listener(console: Console, toggle: Callable[[], None]) -> IO[IO[None]]:
    if not console.status.live:
        return io_pure(IO(no_op))

    def attach(file_descriptor: int) -> IO[IO[None]]:
        def with_attributes(attributes: Maybe[TermiosState]) -> IO[IO[None]]:
            if isinstance(attributes, Nothing):
                return io_pure(IO(no_op))

            def with_cbreak(entered: bool) -> IO[IO[None]]:
                if not entered:
                    return io_pure(IO(no_op))

                listener = TabListener(
                    file_descriptor=file_descriptor,
                    saved=attributes.value,
                    toggle=toggle,
                )
                return io_map(listener.start(), lambda _: listener.stop())

            return io_bind(enter_cbreak(file_descriptor), with_cbreak)

        return io_bind(tty_attributes(file_descriptor), with_attributes)

    def with_tty(file_descriptor: Maybe[int]) -> IO[IO[None]]:
        return maybe_either(file_descriptor, attach, lambda: io_pure(IO(no_op)))

    return io_bind(open_tty(), with_tty)

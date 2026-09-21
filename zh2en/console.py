from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TextIO

from zh2en.monads import (
    IO,
    Ref,
    io_when,
    modify_ref,
    modify_ref_with,
    read_ref,
    write_ref,
)


@dataclass(frozen=True)
class StatusView:
    label: str = "Working"
    started: float = 0.0
    tokens: int = 0
    prefix: str = ""
    drawn: str = ""
    raw: bool = False
    raw_open: bool = False


def status_line_text(view: StatusView, now_value: float) -> str:
    line = "%s: time elapsed: %.2fs, tokens received: %d" % (
        view.label,
        now_value - view.started,
        view.tokens,
    )
    return view.prefix + line if view.prefix else line


def status_erase_text(view: StatusView) -> str:
    return "\r" + " " * len(view.drawn) + "\r" if view.drawn else ""


def draw_render(view: StatusView, now_value: float) -> tuple[StatusView, str]:
    full = status_line_text(view, now_value)
    text = "\r" + full + " " * max(len(view.drawn) - len(full), 0)
    return replace(view, drawn=full), text


def start_render(
    view: StatusView, label: str, now_value: float
) -> tuple[StatusView, str]:
    return draw_render(
        replace(
            view,
            label=label,
            started=now_value,
            tokens=0,
            raw=False,
            raw_open=False,
        ),
        now_value,
    )


def progress_render(view: StatusView, label: str, count: int) -> StatusView:
    return replace(view, label=label, tokens=view.tokens + count)


def raw_begin_render(view: StatusView) -> tuple[StatusView, str]:
    if view.raw:
        return view, ""

    return (
        replace(view, raw=True, drawn="", prefix="", raw_open=False),
        status_erase_text(view),
    )


def raw_write_render(view: StatusView, text: str) -> StatusView:
    if not text:
        return view

    return replace(view, raw=True, raw_open=not text.endswith("\n"))


def raw_end_render(view: StatusView) -> tuple[StatusView, str]:
    if not view.raw:
        return view, ""

    terminated = "\n" if view.raw_open else ""
    return replace(view, raw=False, drawn="", raw_open=False), terminated


def stop_render(view: StatusView) -> tuple[StatusView, str]:
    if view.raw:
        return raw_end_render(view)

    erased = status_erase_text(view)
    if not erased:
        return view, ""

    if view.prefix:
        return replace(view, drawn=view.prefix), erased + view.prefix

    return replace(view, drawn=""), erased


def interrupt_render(view: StatusView, live: bool) -> tuple[StatusView, str]:
    if view.raw:
        return raw_end_render(view)

    erased = status_erase_text(view)
    had_drawn = bool(view.drawn)
    view = replace(view, drawn="")
    if not view.prefix:
        return view, erased

    if live and had_drawn:
        return replace(view, prefix=""), erased

    return replace(view, prefix=""), erased + "\n"


def finish_render(view: StatusView, text: str) -> tuple[StatusView, str]:
    separator = "\n" if view.raw and view.raw_open else ""
    erased = status_erase_text(view)
    prefix = view.prefix if erased else ""
    return (
        replace(view, raw=False, raw_open=False, prefix="", drawn=""),
        separator + erased + prefix + text + "\n",
    )


@dataclass(frozen=True, eq=False)
class StatusLine:
    stream: TextIO
    live: bool
    monotonic: Callable[[], float] = time.monotonic
    view: Ref[StatusView] = field(default_factory=lambda: Ref(StatusView()))
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    halt: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )
    worker: Ref[threading.Thread | None] = field(
        default_factory=lambda: Ref(None), repr=False, compare=False
    )

    def write_partial(self, text: str) -> IO[None]:
        def thunk() -> None:
            with self.lock:
                modify_ref(
                    self.view,
                    lambda view: replace(view, prefix=view.prefix + text),
                ).run()
                self.stream.write(text)
                self.stream.flush()

        return IO(thunk)

    def begin_raw(self) -> IO[None]:
        def thunk() -> None:
            with self.lock:
                text = modify_ref_with(self.view, raw_begin_render).run()
                if text:
                    self.stream.write(text)
                    self.stream.flush()

        return IO(thunk)

    def write_raw(self, text: str) -> IO[None]:
        def thunk() -> None:
            with self.lock:
                modify_ref(self.view, lambda view: raw_write_render(view, text)).run()
                self.stream.write(text)
                self.stream.flush()

        return IO(thunk)

    def end_raw(self) -> IO[None]:
        def thunk() -> None:
            with self.lock:
                text = modify_ref_with(self.view, raw_end_render).run()
                if text:
                    self.stream.write(text)
                    self.stream.flush()

        return IO(thunk)

    def start(self, label: str) -> IO[None]:
        def thunk() -> None:
            if not self.live:
                return None

            with self.lock:
                text = modify_ref_with(
                    self.view,
                    lambda view: start_render(view, label, self.monotonic()),
                ).run()
                self.stream.write(text)
                self.stream.flush()

            self.halt.clear()
            thread = threading.Thread(target=self._tick, daemon=True)
            write_ref(self.worker, thread).run()
            thread.start()
            return None

        return IO(thunk)

    def progress(self, label: str, count: int = 1) -> IO[None]:
        def thunk() -> None:
            if not self.live:
                return None

            modify_ref(
                self.view, lambda view: progress_render(view, label, count)
            ).run()
            return None

        return IO(thunk)

    def stop(self) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self.lock:
                text = modify_ref_with(self.view, stop_render).run()
                self.stream.write(text)
                self.stream.flush()

            return None

        return IO(thunk)

    def interrupt(self) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self.lock:
                text = modify_ref_with(
                    self.view, lambda view: interrupt_render(view, self.live)
                ).run()
                self.stream.write(text)
                self.stream.flush()

            return None

        return IO(thunk)

    def finish(self, text: str) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self.lock:
                line = modify_ref_with(
                    self.view, lambda view: finish_render(view, text)
                ).run()
                self.stream.write(line)
                self.stream.flush()

            return None

        return IO(thunk)

    def _halt(self) -> None:
        worker = read_ref(self.worker).run()
        if worker is not None and worker.is_alive():
            self.halt.set()
            worker.join(timeout=1.0)

    def _tick(self) -> None:
        while not self.halt.wait(0.1):
            with self.lock:
                if self.view.value.raw:
                    continue

                text = modify_ref_with(
                    self.view, lambda view: draw_render(view, self.monotonic())
                ).run()
                self.stream.write(text)
                self.stream.flush()


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

    def begin_raw(self) -> IO[None]:
        return self.status.begin_raw()

    def write_raw(self, text: str) -> IO[None]:
        return self.status.write_raw(text)

    def end_raw(self) -> IO[None]:
        return self.status.end_raw()

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


def log_when(verbose: bool, console: Console, message: str) -> IO[None]:
    return io_when(verbose, console.log(message))

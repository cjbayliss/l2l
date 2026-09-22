from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TextIO

from zh2en.monads import (
    IO,
    Ref,
    io_and_then,
    io_atomic,
    io_bind,
    io_map,
    io_pair,
    io_pure,
    io_when_unit,
    modify_ref,
    modify_ref_with,
    read_ref,
    repeat_until,
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


def stream_write(stream: TextIO, text: str) -> IO[None]:
    def thunk() -> None:
        stream.write(text)
        stream.flush()

    return IO(thunk)


TICK_SECONDS = 0.1


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

    def write(self, text: str) -> IO[None]:
        return stream_write(self.stream, text)

    def write_rendered(self, text: str) -> IO[None]:
        return io_when_unit(bool(text), self.write(text))

    def write_partial(self, text: str) -> IO[None]:
        def updated(view: StatusView) -> StatusView:
            return replace(view, prefix=view.prefix + text)

        return io_atomic(
            self.lock,
            io_and_then(modify_ref(self.view, updated), self.write(text)),
        )

    def begin_raw_action(self) -> IO[None]:
        return io_bind(
            modify_ref_with(self.view, raw_begin_render),
            self.write_rendered,
        )

    def begin_raw(self) -> IO[None]:
        return io_atomic(self.lock, self.begin_raw_action())

    def write_raw_action(self, text: str) -> IO[None]:
        return io_and_then(
            modify_ref(self.view, lambda view: raw_write_render(view, text)),
            self.write(text),
        )

    def write_raw(self, text: str) -> IO[None]:
        return io_atomic(self.lock, self.write_raw_action(text))

    def end_raw_action(self) -> IO[None]:
        return io_bind(
            modify_ref_with(self.view, raw_end_render),
            self.write_rendered,
        )

    def end_raw(self) -> IO[None]:
        return io_atomic(self.lock, self.end_raw_action())

    def start(self, label: str) -> IO[None]:
        def render(view: StatusView) -> tuple[StatusView, str]:
            return start_render(view, label, self.monotonic())

        def drawn(_: None) -> IO[None]:
            return io_atomic(
                self.lock,
                io_bind(modify_ref_with(self.view, render), self.write_rendered),
            )

        def launch(_: None) -> IO[None]:
            thread = threading.Thread(
                target=repeat_until(self.tick_action(), self.halt, TICK_SECONDS),
                daemon=True,
            )

            def boot(_: threading.Thread | None) -> IO[None]:
                def thunk() -> None:
                    self.halt.clear()
                    thread.start()

                return IO(thunk)

            return io_bind(write_ref(self.worker, thread), boot)

        return io_when_unit(self.live, io_and_then(drawn(None), launch(None)))

    def progress(self, label: str, count: int = 1) -> IO[None]:
        def updated(view: StatusView) -> StatusView:
            return progress_render(view, label, count)

        return io_when_unit(
            self.live,
            io_atomic(
                self.lock,
                io_map(modify_ref(self.view, updated), lambda _: None),
            ),
        )

    def stop_worker(self) -> IO[None]:
        def join_if_running(worker: threading.Thread | None) -> IO[None]:
            def thunk() -> None:
                if worker is not None and worker.is_alive():
                    self.halt.set()
                    worker.join(timeout=1.0)

            return IO(thunk)

        return io_bind(read_ref(self.worker), join_if_running)

    def stop_action(self) -> IO[None]:
        return io_bind(
            modify_ref_with(self.view, stop_render),
            self.write_rendered,
        )

    def stop(self) -> IO[None]:
        return io_and_then(
            self.stop_worker(),
            io_atomic(self.lock, self.stop_action()),
        )

    def interrupt_action(self) -> IO[None]:
        return io_bind(
            modify_ref_with(self.view, lambda view: interrupt_render(view, self.live)),
            self.write_rendered,
        )

    def interrupt(self) -> IO[None]:
        return io_and_then(
            self.stop_worker(),
            io_atomic(self.lock, self.interrupt_action()),
        )

    def finish_action(self, text: str) -> IO[None]:
        return io_bind(
            modify_ref_with(self.view, lambda view: finish_render(view, text)),
            self.write_rendered,
        )

    def finish(self, text: str) -> IO[None]:
        return io_and_then(
            self.stop_worker(),
            io_atomic(self.lock, self.finish_action(text)),
        )

    def tick_action(self) -> IO[None]:
        def redraw(view: StatusView) -> tuple[StatusView, IO[None]]:
            if view.raw:
                return view, io_pure(None)

            updated, text = draw_render(view, self.monotonic())
            return updated, self.write(text)

        def emit(effect: IO[None]) -> IO[None]:
            return effect

        return io_atomic(
            self.lock,
            io_bind(modify_ref_with(self.view, redraw), emit),
        )


@dataclass(frozen=True)
class LogEvent:
    text: str
    verbose_only: bool
    raw: bool
    rows: int


def terminal_size() -> tuple[int, int]:
    try:
        size = os.get_terminal_size(sys.stderr.fileno())
    except OSError, ValueError, AttributeError:
        size = shutil.get_terminal_size()

    columns = size.columns if size.columns > 0 else 80
    lines = size.lines if size.lines > 0 else 24
    return columns, lines


def row_count(text: str, width: int) -> int:
    columns = max(width, 1)
    segments = text.split("\n")
    wrapped = sum(max(1, -(-len(segment) // columns)) for segment in segments[:-1])
    tail = segments[-1]
    return wrapped + (-(-len(tail) // columns) if tail else 0)


def event_visible(event: LogEvent, verbose: bool) -> bool:
    return verbose if event.verbose_only else True


def event_line(event: LogEvent) -> str:
    if event.raw:
        return event.text if event.text.endswith("\n") else event.text + "\n"

    return event.text + "\n"


def replay_text(events: tuple[LogEvent, ...], pending: str, verbose: bool) -> str:
    parts = [event_line(event) for event in events if event_visible(event, verbose)]
    if verbose and pending:
        parts.append(pending)

    return "".join(parts)


def replay_rows(
    events: tuple[LogEvent, ...], pending: str, verbose: bool, width: int
) -> int:
    rows = sum(event.rows for event in events if event_visible(event, verbose))
    if verbose and pending:
        rows += row_count(pending, width)

    return rows


def erase_rows_render(rows: int, height: int) -> str:
    """Erase `rows` content rows, the last of which holds the cursor."""
    if rows < 0:
        return ""

    count = min(rows, max(height - 1, 0))
    up = "\x1b[%dA" % count if count > 0 else ""
    return up + "\x1b[J"


def toggle_verbose(console: Console) -> IO[bool]:
    return io_bind(
        modify_ref(console.verbose, lambda verbose: not verbose),
        lambda flipped: io_map(console.replay(), lambda _: flipped),
    )


@dataclass(frozen=True)
class Console:
    stream: TextIO
    status: StatusLine
    verbose: Ref[bool] = field(default_factory=lambda: Ref(False))
    events: Ref[tuple[LogEvent, ...]] = field(default_factory=lambda: Ref(()))
    pending_raw: Ref[str] = field(default_factory=lambda: Ref(""))
    displayed_rows: Ref[int] = field(default_factory=lambda: Ref(0))
    term_size: Callable[[], tuple[int, int]] = terminal_size

    def record(self, text: str, verbose_only: bool, raw: bool) -> IO[None]:
        """Append a session event; compose inside a status-lock `io_atomic`."""
        width = self.term_size()[0]
        event = LogEvent(
            text=text,
            verbose_only=verbose_only,
            raw=raw,
            rows=row_count(text, width),
        )

        def tracked(_: tuple[LogEvent, ...]) -> IO[None]:
            def counted(verbose: bool) -> IO[None]:
                if raw or verbose_only and not verbose:
                    return io_pure(None)

                return io_map(
                    modify_ref(self.displayed_rows, lambda rows: rows + event.rows),
                    lambda _: None,
                )

            return io_bind(read_ref(self.verbose), counted)

        return io_bind(
            modify_ref(self.events, lambda events: events + (event,)),
            tracked,
        )

    def commit_pending_raw(self) -> IO[None]:
        """Seal an open reasoning block; compose inside a locked scope."""

        def sealed(pending: str) -> IO[None]:
            if not pending:
                return io_pure(None)

            return io_and_then(
                io_map(write_ref(self.pending_raw, ""), lambda _: None),
                self.record(pending, verbose_only=True, raw=True),
            )

        return io_bind(read_ref(self.pending_raw), sealed)

    def log(self, message: str) -> IO[None]:
        return self._log_line(message, verbose_only=False)

    def log_verbose(self, message: str) -> IO[None]:
        return self._log_line(message, verbose_only=True)

    def _log_line(self, message: str, verbose_only: bool) -> IO[None]:
        def act(verbose: bool) -> IO[None]:
            shown = not verbose_only or verbose

            def erase(view: StatusView) -> tuple[StatusView, str]:
                return interrupt_render(view, self.status.live)

            def logged(erase_text: str) -> IO[None]:
                def committed(_: None) -> IO[None]:
                    def printed(_: None) -> IO[None]:
                        def recorded(__: None) -> IO[None]:
                            return self.record(message, verbose_only, raw=False)

                        return io_and_then(
                            io_when_unit(shown, self.status.write(message + "\n")),
                            recorded(None),
                        )

                    return io_and_then(self.commit_pending_raw(), printed(None))

                return io_and_then(
                    io_when_unit(
                        shown and bool(erase_text), self.status.write(erase_text)
                    ),
                    committed(None),
                )

            def section(_: None) -> IO[None]:
                return io_atomic(
                    self.status.lock,
                    io_bind(modify_ref_with(self.status.view, erase), logged),
                )

            return io_and_then(
                io_when_unit(shown, self.status.stop_worker()),
                section(None),
            )

        return io_bind(read_ref(self.verbose), act)

    def stream_reasoning(self, text: str) -> IO[None]:
        """Capture a reasoning delta; display it live only when verbose."""

        def act(verbose: bool) -> IO[None]:
            def extended(previous: str) -> IO[None]:
                pending = previous + text

                def displayed(_: None) -> IO[None]:
                    if not verbose:
                        return io_pure(None)

                    width = self.term_size()[0]

                    def counted(_: None) -> IO[None]:
                        return io_map(
                            modify_ref(
                                self.displayed_rows,
                                lambda rows: (
                                    rows
                                    + row_count(pending, width)
                                    - row_count(previous, width)
                                ),
                            ),
                            lambda _: None,
                        )

                    return io_and_then(
                        io_and_then(
                            self.status.begin_raw_action(),
                            self.status.write_raw_action(text),
                        ),
                        counted(None),
                    )

                return io_and_then(
                    io_map(write_ref(self.pending_raw, pending), lambda _: None),
                    displayed(None),
                )

            return io_bind(read_ref(self.pending_raw), extended)

        return io_atomic(self.status.lock, io_bind(read_ref(self.verbose), act))

    def end_raw(self) -> IO[None]:
        return io_atomic(
            self.status.lock,
            io_and_then(self.status.end_raw_action(), self.commit_pending_raw()),
        )

    def replay(self) -> IO[None]:
        """Erase this session's output and re-render it for the current mode."""

        def snapshot() -> IO[
            tuple[
                tuple[bool, tuple[tuple[LogEvent, ...], str]],
                tuple[int, StatusView],
            ]
        ]:
            return io_pair(
                io_pair(
                    read_ref(self.verbose),
                    io_pair(read_ref(self.events), read_ref(self.pending_raw)),
                ),
                io_pair(read_ref(self.displayed_rows), read_ref(self.status.view)),
            )

        def render(
            inputs: tuple[
                tuple[bool, tuple[tuple[LogEvent, ...], str]],
                tuple[int, StatusView],
            ],
        ) -> IO[None]:
            (verbose, (events, pending)), (shown_rows, view) = inputs
            width, height = self.term_size()
            inside = view.raw and view.raw_open
            erase = status_erase_text(view)
            if inside:
                erase += "\r"
                erase += erase_rows_render(shown_rows - 1, height)
            elif shown_rows > 0:
                erase += erase_rows_render(shown_rows, height)

            text = replay_text(events, pending, verbose)

            def written(_: None) -> IO[None]:
                shown_pending = verbose and bool(pending)
                rows = replay_rows(events, pending, verbose, width)
                return io_and_then(
                    write_ref(self.displayed_rows, rows),
                    io_map(
                        modify_ref(
                            self.status.view,
                            lambda current: replace(
                                current,
                                drawn="",
                                raw=shown_pending,
                                raw_open=shown_pending and not pending.endswith("\n"),
                                prefix="" if shown_pending else current.prefix,
                            ),
                        ),
                        lambda _: None,
                    ),
                )

            return io_and_then(
                io_when_unit(bool(erase or text), self.status.write(erase + text)),
                written(None),
            )

        def act(_: None) -> IO[None]:
            return io_atomic(self.status.lock, io_bind(snapshot(), render))

        return io_when_unit(self.status.live, act(None))

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
        def render(view: StatusView) -> tuple[StatusView, IO[None]]:
            prefix = view.prefix
            updated, line = finish_render(view, text)
            return updated, io_and_then(
                self.status.write(line),
                self.record(prefix + text, verbose_only=False, raw=False),
            )

        def emit(effect: IO[None]) -> IO[None]:
            return effect

        return io_and_then(
            self.status.stop_worker(),
            io_atomic(
                self.status.lock,
                io_bind(modify_ref_with(self.status.view, render), emit),
            ),
        )

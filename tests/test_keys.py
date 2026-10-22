import os
import sys
import termios
import threading
import tty
from pathlib import Path

import pytest
from fakes import make_console

from l2l import keys
from l2l.console import Console, toggle_verbose
from l2l.keys import (
    TabListener,
    enter_cbreak,
    is_toggle_key,
    open_tty,
    start_tab_listener,
)
from l2l.monads import IO, NOTHING, Just, Reference, read_reference, write_reference

pytestmark = pytest.mark.skipif(os.name == "nt", reason="requires a POSIX TTY")


def toggler(console: Console) -> None:
    toggle_verbose(console).run()


def listener_for(
    file_descriptor: int,
    console: Console,
    saved: tuple[object, ...] = (),
) -> TabListener:
    return TabListener(
        file_descriptor=file_descriptor,
        saved=saved,
        toggle=lambda: toggler(console),
    )


def test_is_toggle_key_matches_only_tab() -> None:
    assert is_toggle_key(b"\t")
    assert not is_toggle_key(b"x")
    assert not is_toggle_key(b"")
    assert not is_toggle_key(b"\n")


def test_open_tty_opens_the_tty_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "open", lambda path, flags: 42)
    assert open_tty().run() == Just(42)


def test_open_tty_falls_back_to_a_stdin_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdin:
        def fileno(self) -> int:
            return 7

    def refuse(path: str, flags: int) -> int:
        raise OSError("no /dev/tty")

    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(os, "isatty", lambda file_descriptor: True)
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    assert open_tty().run() == Just(7)


def test_open_tty_without_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdin:
        def fileno(self) -> int:
            return 7

    def refuse(path: str, flags: int) -> int:
        raise OSError("no /dev/tty")

    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(os, "isatty", lambda file_descriptor: False)
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    assert open_tty().run() is NOTHING


def test_enter_cbreak_fails_on_a_non_tty_fd(tmp_path: Path) -> None:
    file_descriptor = os.open(str(tmp_path / "plain"), os.O_RDWR | os.O_CREAT)
    try:
        assert enter_cbreak(file_descriptor).run() is False
    finally:
        os.close(file_descriptor)


def test_listen_action_toggles_on_tab_and_ignores_other_keys() -> None:
    read_fd, write_fd = os.pipe()
    console, _ = make_console(live=True)
    listener = listener_for(read_fd, console)
    try:
        os.write(write_fd, b"\t")
        listener.listen_action().run()
        assert console.verbose.value is True

        os.write(write_fd, b"x")
        listener.listen_action().run()
        assert console.verbose.value is True
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_listen_action_halts_when_the_descriptor_disappears() -> None:
    read_fd, write_fd = os.pipe()
    console, _ = make_console(live=True)
    listener = listener_for(read_fd, console)
    os.close(read_fd)
    try:
        listener.listen_action().run()
        assert listener.halt.is_set()
    finally:
        os.close(write_fd)


def test_start_and_stop_manage_the_listener_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []
    joined: list[float | None] = []
    restored: list[tuple[int, int, list[object]]] = []

    class RecordingThread:
        def __init__(self, target: object, daemon: bool) -> None:
            assert daemon
            self.alive = True

        def start(self) -> None:
            started.append(True)

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            self.alive = False
            joined.append(timeout)

    monkeypatch.setattr(threading, "Thread", RecordingThread)
    monkeypatch.setattr(
        termios,
        "tcsetattr",
        lambda file_descriptor, when, attributes: restored.append(
            (file_descriptor, when, attributes)
        ),
    )
    console, _ = make_console(live=True)
    listener = listener_for(0, console, saved=(1, 2, 3))
    listener.start().run()
    listener.stop().run()
    listener.stop().run()

    assert started == [True]
    assert joined == [1.0]
    assert restored == [(0, termios.TCSADRAIN, [1, 2, 3])]


def test_start_tab_listener_without_live_terminal() -> None:
    console, _ = make_console(live=False)
    stop = start_tab_listener(console, lambda: toggler(console)).run()
    stop.run()
    assert console.verbose.value is False


def test_start_tab_listener_without_cbreak_stays_detached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(file_descriptor: int) -> None:
        raise termios.error("cannot enter cbreak mode")

    monkeypatch.setattr(keys, "open_tty", lambda: IO(lambda: Just(0)))
    monkeypatch.setattr(termios, "tcgetattr", lambda file_descriptor: (1, 2, 3))
    monkeypatch.setattr(tty, "setcbreak", refuse)
    console, _ = make_console(live=True)
    stop = start_tab_listener(console, lambda: toggler(console)).run()
    stop.run()
    assert console.verbose.value is False


def test_start_tab_listener_attaches_to_a_working_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[bool] = []

    class RecordingThread:
        def __init__(self, target: object, daemon: bool) -> None:
            assert daemon

        def start(self) -> None:
            started.append(True)

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            return None

    monkeypatch.setattr(threading, "Thread", RecordingThread)
    monkeypatch.setattr(keys, "open_tty", lambda: IO(lambda: Just(0)))
    monkeypatch.setattr(termios, "tcgetattr", lambda file_descriptor: (1, 2, 3))
    monkeypatch.setattr(termios, "tcsetattr", lambda *arguments: None)
    monkeypatch.setattr(tty, "setcbreak", lambda file_descriptor: None)
    console, _ = make_console(live=True)
    stop = start_tab_listener(console, lambda: toggler(console)).run()
    stop.run()
    assert started == [True]


def test_stop_restores_terminal_attributes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    restored: list[int] = []
    monkeypatch.setattr(
        termios,
        "tcsetattr",
        lambda file_descriptor, when, attributes: restored.append(file_descriptor),
    )
    listener = TabListener(
        file_descriptor=0,
        saved=(),
        toggle=lambda: None,
        restored=Reference(False),
    )
    listener.stop().run()
    listener.stop().run()
    assert restored == [0]
    assert read_reference(listener.restored).run() is True


def test_toggle_writes_verbose_state() -> None:
    console, _ = make_console(live=True)
    write_reference(console.verbose, False).run()
    toggler(console)
    assert console.verbose.value is True

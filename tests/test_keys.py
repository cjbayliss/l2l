import os
import pty
import sys
import termios
import time

import pytest
from fakes import make_console

from zh2en import keys
from zh2en.keys import TabListener, is_toggle_key, open_tty, start_tab_listener
from zh2en.monads import NOTHING, Just

pytestmark = pytest.mark.skipif(os.name == "nt", reason="requires a POSIX TTY")


def test_is_toggle_key_matches_only_tab() -> None:
    assert is_toggle_key(b"\t")
    assert not is_toggle_key(b"x")
    assert not is_toggle_key(b"")
    assert not is_toggle_key(b"\n")


def test_open_tty_opens_the_tty_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "open", lambda path, flags: 42)
    assert open_tty() == Just(42)


def test_open_tty_falls_back_to_a_stdin_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    master, slave = pty.openpty()

    def refuse(path: str, flags: int) -> int:
        raise OSError("no /dev/tty")

    class FakeStdin:
        def fileno(self) -> int:
            return slave

    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(sys, "stdin", FakeStdin())
    assert open_tty() == Just(slave)
    os.close(master)
    os.close(slave)


def test_open_tty_without_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(path: str, flags: int) -> int:
        raise OSError("no /dev/tty")

    monkeypatch.setattr(os, "open", refuse)
    monkeypatch.setattr(sys, "stdin", 3)
    assert open_tty() is NOTHING


def test_start_tab_listener_disabled_without_live_terminal() -> None:
    console, _ = make_console(live=False)
    stop = start_tab_listener(console)
    stop()


def test_start_tab_listener_toggles_and_restores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, slave = pty.openpty()
    console, stream = make_console(live=True)
    monkeypatch.setattr(keys, "open_tty", lambda: Just(slave))
    saved = termios.tcgetattr(slave)
    stop = start_tab_listener(console)
    try:
        assert console.verbose.value is False
        os.write(master, b"\t")
        deadline = time.monotonic() + 5.0
        while console.verbose.value is False and time.monotonic() < deadline:
            time.sleep(0.01)
        assert console.verbose.value is True

        os.write(master, b"x")
        time.sleep(0.2)
        assert console.verbose.value is True
        assert stream.getvalue() == ""
    finally:
        stop()
        restored = termios.tcgetattr(slave)
        assert restored[3] & (termios.ECHO | termios.ICANON) == saved[3] & (
            termios.ECHO | termios.ICANON
        )
        assert restored[6][termios.VMIN] == saved[6][termios.VMIN]
        assert restored[6][termios.VTIME] == saved[6][termios.VTIME]
        os.close(master)
        os.close(slave)

    stop()


def test_start_tab_listener_twice_toggles_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, slave = pty.openpty()
    console, _ = make_console(live=True)
    monkeypatch.setattr(keys, "open_tty", lambda: Just(slave))
    stop = start_tab_listener(console)
    try:
        os.write(master, b"\t")
        deadline = time.monotonic() + 5.0
        while console.verbose.value is False and time.monotonic() < deadline:
            time.sleep(0.01)

        os.write(master, b"\t")
        while console.verbose.value is True and time.monotonic() < deadline:
            time.sleep(0.01)

        assert console.verbose.value is False
    finally:
        stop()
        os.close(master)
        os.close(slave)


def test_start_tab_listener_survives_a_non_tty_fd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(fd: object) -> list[object]:
        raise OSError("not a tty")

    monkeypatch.setattr(keys, "open_tty", lambda: Just(42))
    monkeypatch.setattr(termios, "tcgetattr", refuse)
    console, _ = make_console(live=True)
    stop = start_tab_listener(console)
    stop()


def test_tab_listener_stop_without_start() -> None:
    console, _ = make_console(live=True)
    listener = TabListener(console=console, fd=-1, saved=[], restored=True)
    listener.stop()


def test_tab_listener_exits_when_the_tty_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master, slave = pty.openpty()
    console, _ = make_console(live=True)
    monkeypatch.setattr(keys, "open_tty", lambda: Just(slave))
    stop = start_tab_listener(console)
    os.close(slave)
    time.sleep(0.3)
    stop()
    os.close(master)

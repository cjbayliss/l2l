from __future__ import annotations

import contextlib
import json
import os
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from l2l.errors import TranslationError, fail_config
from l2l.monads import (
    IO,
    NOTHING,
    Just,
    Maybe,
    Nothing,
    Ok,
    Result,
    fold_io,
    io_bind,
    io_map,
    io_result,
    result_or_else,
)
from l2l.text import cache_path

Clock = Callable[[], float]
Sleep = Callable[[float], None]


def now(clock: Clock) -> IO[float]:
    return IO(clock)


def time_sleep(seconds: float) -> None:
    time.sleep(seconds)


def read_stdin(stream: TextIO) -> IO[str]:
    return IO(stream.read)


def write_stdout(stream: TextIO, content: str) -> IO[None]:
    def thunk() -> None:
        stream.write(content)
        if not content.endswith("\n"):
            stream.write("\n")

    return IO(thunk)


def path_exists(path: str) -> IO[bool]:
    return IO(lambda: os.path.exists(path))


def cwd() -> IO[str]:
    return IO(os.getcwd)


def io_isatty(stream: TextIO) -> IO[bool]:
    def thunk() -> bool:
        try:
            return bool(stream.isatty())
        except AttributeError, OSError, ValueError:
            return False

    return IO(thunk)


def user_config_path(environment: Mapping[str, str]) -> IO[str]:
    return IO(
        lambda: os.path.join(
            environment.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
            "l2l",
            "config.toml",
        )
    )


def resolve_cache_dir(
    environment: Mapping[str, str], override: str | None = None
) -> IO[str]:
    def thunk() -> str:
        if override:
            directory = override
        else:
            directory = os.path.join(
                environment.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                "l2l",
            )

        os.makedirs(directory, exist_ok=True)
        return directory

    return IO(thunk)


def load_toml(
    path: str, description: str
) -> IO[Result[dict[str, Any], TranslationError]]:
    def thunk() -> Result[dict[str, Any], TranslationError]:
        try:
            with open(path, "rb") as handle:
                return Ok(tomllib.load(handle))
        except FileNotFoundError:
            return fail_config("%s not found: %s" % (description, path))
        except OSError as error:
            return fail_config("cannot read %s: %s" % (description, error))
        except tomllib.TOMLDecodeError as error:
            return fail_config("cannot parse %s %s: %s" % (description, path, error))

    return IO(thunk)


def read_text_file(path: str, description: str) -> IO[Result[str, TranslationError]]:
    def thunk() -> Result[str, TranslationError]:
        try:
            with open(path, encoding="utf-8") as handle:
                return Ok(handle.read())
        except FileNotFoundError:
            return fail_config("%s not found: %s" % (description, path))
        except OSError as error:
            return fail_config("cannot read %s: %s" % (description, error))

    return IO(thunk)


def cache_read(cache_directory: str, key: str) -> IO[Maybe[str]]:
    def thunk() -> Maybe[str]:
        try:
            with open(cache_path(cache_directory, key), encoding="utf-8") as handle:
                return Just(handle.read())
        except OSError:
            return NOTHING

    return IO(thunk)


def cache_write(cache_directory: str, key: str, value: str) -> IO[None]:
    def thunk() -> None:
        path = cache_path(cache_directory, key)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(value)

        os.replace(temporary_path, path)

    return IO(thunk)


def log_stamp(now_value: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_value))


def _no_append(_content: str) -> None:
    return None


def _no_close() -> None:
    return None


@dataclass(frozen=True)
class RunLog:
    path: Maybe[str]
    clock: Clock
    append: Callable[[str], None] = _no_append
    close: Callable[[], None] = _no_close


def run_log_path(cache_directory: str, now_value: float, pid_value: int) -> str:
    return os.path.join(
        cache_directory,
        "logs",
        "%s-%d.log"
        % (time.strftime("%Y%m%d-%H%M%S", time.gmtime(now_value)), pid_value),
    )


def open_run_log(cache_directory: str, clock: Clock) -> IO[RunLog]:
    def thunk() -> RunLog:
        path = run_log_path(cache_directory, clock(), os.getpid())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle = open(path, "a", encoding="utf-8")

        def append(content: str) -> None:
            with contextlib.suppress(OSError, ValueError):
                handle.write(content)

        def close() -> None:
            with contextlib.suppress(OSError):
                handle.close()

        return RunLog(path=Just(path), clock=clock, append=append, close=close)

    return IO(thunk)


def close_run_log(log: RunLog) -> IO[None]:
    def thunk() -> None:
        log.close()

    return IO(thunk)


def run_log_write(log: RunLog, content: str) -> IO[None]:
    def thunk() -> None:
        if isinstance(log.path, Nothing):
            return

        log.append(content)

    return IO(thunk)


def log_entry(log: RunLog, label: str, body: str) -> IO[None]:
    def stamped(now_value: float) -> IO[None]:
        return run_log_write(
            log, "== %s %s\n%s\n\n" % (log_stamp(now_value), label, body)
        )

    return io_bind(now(log.clock), stamped)


def log_request(
    log: RunLog, payload: Mapping[str, Any], label: str = "REQUEST"
) -> IO[None]:
    return log_entry(log, label, json.dumps(payload, indent=2, ensure_ascii=False))


def log_error(log: RunLog, detail: str) -> IO[None]:
    return log_entry(log, "ERROR", detail)


def cache_entry_paths(cache_directory: str) -> IO[tuple[str, ...]]:
    def thunk() -> tuple[str, ...]:
        try:
            entries = tuple(os.scandir(cache_directory))
        except OSError:
            return ()

        return tuple(
            entry.path
            for entry in entries
            if entry.is_file()
            and (entry.name.endswith(".txt") or entry.name.endswith(".txt.tmp"))
        )

    return IO(thunk)


def log_paths(cache_directory: str) -> IO[tuple[str, ...]]:
    def thunk() -> tuple[str, ...]:
        try:
            entries = tuple(os.scandir(os.path.join(cache_directory, "logs")))
        except OSError:
            return ()

        return tuple(
            entry.path
            for entry in entries
            if entry.is_file() and entry.name.endswith(".log")
        )

    return IO(thunk)


LOG_SECONDS_PER_DAY = 86400.0


def prune_old_logs(cache_directory: str, keep_days: int, clock: Clock) -> IO[int]:
    horizon = keep_days * LOG_SECONDS_PER_DAY

    def with_now(now_value: float) -> IO[int]:
        def step(count: int, path: str) -> IO[Result[int, TranslationError]]:
            def decide(age: float) -> IO[Result[int, TranslationError]]:
                if age <= horizon:
                    return io_result(Ok(count))

                return io_map(
                    remove_file(path),
                    lambda removed: Ok(count + (1 if removed else 0)),
                )

            return io_bind(file_age(path, now_value), decide)

        def report(pruned: Result[int, TranslationError]) -> int:
            return result_or_else(pruned, lambda: 0)

        return io_map(
            io_bind(
                log_paths(cache_directory), lambda paths: fold_io(paths, step, Ok(0))
            ),
            report,
        )

    return io_bind(now(clock), with_now)


def file_age(path: str, now_value: float) -> IO[float]:
    def thunk() -> float:
        try:
            return max(now_value - os.path.getmtime(path), 0.0)
        except OSError:
            return 0.0

    return IO(thunk)


def remove_file(path: str) -> IO[bool]:
    def thunk() -> bool:
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    return IO(thunk)

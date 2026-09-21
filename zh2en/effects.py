from __future__ import annotations

import json
import os
import time
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

from zh2en.errors import TranslationError, fail_config
from zh2en.monads import IO, Ok, Result
from zh2en.text import cache_path

Clock = Callable[[], float]


def now(clock: Clock) -> IO[float]:
    return IO(clock)


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
        except (AttributeError, OSError, ValueError):
            return False

    return IO(thunk)


def user_config_path(environment: Mapping[str, str]) -> IO[str]:
    return IO(
        lambda: os.path.join(
            environment.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
            "zh2en",
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
                "zh2en",
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


def cache_read(cache_directory: str, key: str) -> IO[str | None]:
    def thunk() -> str | None:
        try:
            with open(cache_path(cache_directory, key), encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return None

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


@dataclass(frozen=True)
class RunLog:
    path: str
    clock: Clock = time.time


def run_log_path(cache_directory: str, now_value: float) -> str:
    return os.path.join(
        cache_directory,
        "logs",
        "%s-%d.log"
        % (time.strftime("%Y%m%d-%H%M%S", time.gmtime(now_value)), os.getpid()),
    )


def open_run_log(cache_directory: str, clock: Clock) -> IO[RunLog]:
    def thunk() -> RunLog:
        path = run_log_path(cache_directory, clock())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return RunLog(path=path, clock=clock)

    return IO(thunk)


def run_log_write(log: RunLog, content: str) -> IO[None]:
    def thunk() -> None:
        if not log.path:
            return

        try:
            with open(log.path, "a", encoding="utf-8") as handle:
                handle.write(content)
        except OSError:
            pass

    return IO(thunk)


def log_entry(log: RunLog, label: str, body: str) -> IO[None]:
    def thunk() -> None:
        return run_log_write(
            log, "== %s %s\n%s\n\n" % (log_stamp(log.clock()), label, body)
        ).run()

    return IO(thunk)


def log_request(
    log: RunLog, payload: Mapping[str, Any], label: str = "REQUEST"
) -> IO[None]:
    return log_entry(log, label, json.dumps(payload, indent=2, ensure_ascii=False))


def log_error(log: RunLog, detail: str) -> IO[None]:
    return log_entry(log, "ERROR", detail)

from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from typing import Any, Literal

from zh2en.console import Console, StatusLine
from zh2en.effects import RunLog
from zh2en.errors import TranslationError
from zh2en.monads import NOTHING, Ok, Result, write_ref
from zh2en.settings import Config, Context, build_settings

SLEEPS: list[float] = []


def recording_sleep(seconds: float) -> None:
    SLEEPS.append(seconds)


def reset_sleeps() -> None:
    SLEEPS.clear()


class FakeStreamResponse:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._lines = [
            line
            for chunk in chunks
            for line in (
                b"data: " + json.dumps(chunk).encode("utf-8") + b"\n",
                b"\n",
            )
        ]

    def __iter__(self) -> Any:
        return iter(self._lines)

    def __enter__(self) -> FakeStreamResponse:
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        return False


class FakePlainResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def read(self) -> bytes:
        return json.dumps(self._body).encode("utf-8")

    def __enter__(self) -> FakePlainResponse:
        return self

    def __exit__(self, *args: object) -> Literal[False]:
        return False


class FakeHttp:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = responses
        self.requests: list[Any] = []

    def open(self, request: Any, timeout: float) -> Result[Any, TranslationError]:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return Ok(self.responses[index])


def make_console(
    live: bool = False,
    term_size: Callable[[], tuple[int, int]] | None = None,
) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    size = term_size if term_size is not None else lambda: (80, 24)
    console = Console(stream, StatusLine(stream, live=live), term_size=size)
    return console, stream


def make_context(
    console: Console,
    open_http: Callable[[Any, float], Result[Any, TranslationError]],
    cache_directory: str = "",
    use_cache: bool = False,
    max_tokens: int = 100000,
    log: RunLog | None = None,
    ensure_paragraphs: bool = False,
    stream: bool | None = None,
    verbose: bool = False,
) -> Context:
    config = Config(
        base_url="http://endpoint.test/v1",
        api_key="key",
        model="model-x",
        timeout=10.0,
        max_tokens=max_tokens,
        params={},
    )
    write_ref(console.verbose, verbose).run()
    return Context(
        config=config,
        settings=build_settings(),
        use_cache=use_cache,
        cache_directory=cache_directory,
        ensure_paragraphs=ensure_paragraphs,
        console=console,
        open_http=open_http,
        log=log if log is not None else RunLog(NOTHING, time.time),
        clock=time.time,
        sleep=recording_sleep,
        stream=stream,
    )


def stream_chunks(*texts: str) -> list[dict[str, Any]]:
    return [{"choices": [{"delta": {"content": text}}]} for text in texts]


def with_usage(
    chunks: list[dict[str, Any]], usage: dict[str, Any]
) -> list[dict[str, Any]]:
    return chunks + [{"choices": [{"delta": {}}], "usage": usage}]

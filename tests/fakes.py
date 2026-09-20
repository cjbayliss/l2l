from __future__ import annotations

import io
import json
from collections.abc import Callable
from typing import Any, Literal

import zh2en as z


class FakeStreamResponse:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._lines = [
            b"data: " + json.dumps(chunk).encode("utf-8") + b"\n" for chunk in chunks
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

    def open(self, request: Any, timeout: float) -> z.Result[Any, str]:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        return z.Ok(self.responses[index])


def make_console() -> tuple[z.Console, io.StringIO]:
    stream = io.StringIO()
    console = z.Console(stream, z.StatusLine(stream))
    return console, stream


def make_context(
    console: z.Console,
    open_http: Callable[[Any, float], z.Result[Any, str]],
    cache_directory: str = "",
    use_cache: bool = False,
    max_tokens: int = 100000,
    log: z.RunLog | None = None,
) -> z.Context:
    config = z.Config(
        base_url="http://endpoint.test/v1",
        api_key="key",
        model="model-x",
        timeout=10.0,
        max_tokens=max_tokens,
        params={},
    )
    return z.Context(
        config=config,
        settings=z.build_settings(),
        use_cache=use_cache,
        cache_directory=cache_directory,
        verbose=False,
        console=console,
        open_http=open_http,
        log=log if log is not None else z.RunLog(""),
    )


def stream_chunks(*texts: str) -> list[dict[str, Any]]:
    return [{"choices": [{"delta": {"content": text}}]} for text in texts]


def with_usage(
    chunks: list[dict[str, Any]], usage: dict[str, Any]
) -> list[dict[str, Any]]:
    return chunks + [{"choices": [{"delta": {}}], "usage": usage}]

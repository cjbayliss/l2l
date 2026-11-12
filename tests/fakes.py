from __future__ import annotations

import io
import json
import threading
import time
from typing import TYPE_CHECKING, cast, final

import pycurl

from l2l.console import Console, StatusLine
from l2l.context import Context, HttpRequest, HttpResponse, OpenHTTP
from l2l.effects import RunLog, Sleep
from l2l.errors import TranslationError, fail_http
from l2l.http import CurlResponse
from l2l.monads import (
    IO,
    NOTHING,
    Ok,
    Result,
    fold_io,
    io_result,
    write_reference,
)
from l2l.settings import (
    Config,
    PassDefinition,
    PassMode,
    build_settings,
    paragraph_tolerance,
)
lazy from l2l.glossary import GlossaryEntry
lazy from l2l.json_values import JsonObject

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def translate_pass(
    name: str = "translate",
    ascii_output: bool = False,
    ensure_paragraphs: bool | int = False,
    retranslate_untranslated: bool = False,
) -> PassDefinition:
    return PassDefinition(
        name,
        "T.",
        "translate",
        {},
        None,
        ascii_output,
        ensure_paragraphs,
        retranslate_untranslated,
        unit_paragraph_tolerance=paragraph_tolerance(ensure_paragraphs),
        pass_paragraph_tolerance=paragraph_tolerance(ensure_paragraphs),
    )


def single_paragraph_pass(
    name: str = "translate",
    retranslate_untranslated: bool = False,
    mode: PassMode = "translate",
) -> PassDefinition:
    return PassDefinition(
        name,
        "T.",
        mode,
        {},
        None,
        False,
        False,
        retranslate_untranslated,
        unit_paragraph_tolerance=0,
        pass_paragraph_tolerance=None,
        paragraphs_per_call=1,
    )


def update_pass(
    name: str = "glossary", target_language: str | None = None
) -> PassDefinition:
    return PassDefinition(
        name,
        "Extract.",
        "update_glossary",
        {},
        None,
        False,
        target_language=target_language,
    )


def no_sleep(seconds: float) -> None:
    return None


def make_sleep_recorder() -> tuple[Sleep, Callable[[], tuple[float, ...]]]:
    recorded: list[float] = []

    def sleep(seconds: float) -> None:
        recorded.append(seconds)

    def sleeps() -> tuple[float, ...]:
        return tuple(recorded)

    return sleep, sleeps


DONE_LINES: tuple[bytes, ...] = (b"data: [DONE]\n", b"\n")


@final
class FakeStreamResponse:
    def __init__(self, chunks: list[JsonObject], terminated: bool = True) -> None:
        frames = [
            line
            for chunk in chunks
            for line in (
                b"data: " + json.dumps(chunk).encode("utf-8") + b"\n",
                b"\n",
            )
        ]
        self._lines = frames + (list(DONE_LINES) if terminated else [])

    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(fail_http("protocol", "streaming response has no body"))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        return fold_io(self._lines, step, initial)


@final
class FakePlainResponse:
    def __init__(self, body: JsonObject) -> None:
        self._body = body

    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(Ok(json.dumps(self._body).encode("utf-8")))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        return io_result(initial)


@final
class FakeCurl:
    def __init__(
        self,
        status: int = 200,
        headers: tuple[bytes, ...] = (),
        chunks: tuple[bytes, ...] = (),
        error: Exception | None = None,
        stages: tuple[tuple[int, tuple[bytes, ...], tuple[bytes, ...]], ...] = (),
    ) -> None:
        self.status = status
        self.headers = headers
        self.chunks = chunks
        self.error = error
        self.stages = stages
        self.options: dict[int, object] = {}
        self.performed = 0
        self.closed = False

    def setopt(self, option: int, value: object) -> None:
        self.options[option] = value

    def perform(self) -> None:
        self.performed += 1
        on_header = cast("Callable[[bytes], None]", self.options[pycurl.HEADERFUNCTION])
        on_chunk = cast("Callable[[bytes], None]", self.options[pycurl.WRITEFUNCTION])
        responses = self.stages or ((self.status, self.headers, self.chunks),)
        for stage_status, stage_headers, stage_chunks in responses:
            for line in stage_headers:
                on_header(line)

            for chunk in stage_chunks:
                on_chunk(chunk)

            self.status = stage_status

        if self.error is not None:
            raise self.error

    def getinfo(self, info: int) -> int:
        return self.status

    def close(self) -> None:
        self.closed = True


@final
class FakeCurlHttp:
    def __init__(self, responses: list[FakeCurl]) -> None:
        self.responses = responses
        self.requests: list[HttpRequest] = []

    def open(
        self, request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.responses) - 1)
        curl = self.responses[index]
        return Ok(CurlResponse(request, timeout, {}, lambda: curl))


def sse_bytes(chunks: list[JsonObject]) -> tuple[bytes, ...]:
    return (
        tuple(
            line
            for chunk in chunks
            for line in (
                b"data: " + json.dumps(chunk).encode("utf-8") + b"\n",
                b"\n",
            )
        )
        + DONE_LINES
    )


@final
class FakeHttp:
    def __init__(self, responses: Sequence[HttpResponse]) -> None:
        self.responses = responses
        self.requests: list[HttpRequest] = []
        self.lock = threading.Lock()

    def open(
        self, request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        # Chunk calls run concurrently, so registration and index selection
        # must be one atomic step to hand each response to exactly one call.
        with self.lock:
            self.requests.append(request)
            index = min(len(self.requests) - 1, len(self.responses) - 1)
        return Ok(self.responses[index])


def make_console(
    live: bool = False,
    term_size: Callable[[], tuple[int, int]] | None = None,
) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()

    def default_size() -> tuple[int, int]:
        return 80, 24

    size = term_size if term_size is not None else default_size
    console = Console(
        stream,
        StatusLine(stream, live=live),
        term_size=lambda: IO(size),
    )
    return console, stream


def make_context(
    console: Console,
    open_http: OpenHTTP,
    cache_directory: str = "",
    use_cache: bool = False,
    maximum_tokens: int = 100000,
    log: RunLog | None = None,
    verbose: bool = False,
    clock: Callable[[], float] = time.time,
    sleep: Sleep = no_sleep,
    best_effort: bool = False,
    glossary_entries: tuple[GlossaryEntry, ...] = (),
    glossary_update_path: str | None = None,
    target_language: str = "English",
    source_language: str = "auto",
) -> Context:
    config = Config(
        base_url="http://endpoint.test/v1",
        api_key="key",
        model="model-x",
        timeout=10.0,
        maximum_tokens=maximum_tokens,
        parameters={},
    )
    write_reference(console.verbose, verbose).run()
    return Context(
        config=config,
        settings=build_settings(),
        use_cache=use_cache,
        cache_directory=cache_directory,
        console=console,
        open_http=open_http,
        log=log if log is not None else RunLog(NOTHING, clock),
        clock=clock,
        sleep=sleep,
        best_effort=best_effort,
        glossary_entries=glossary_entries,
        glossary_update_path=glossary_update_path,
        target_language=target_language,
        source_language=source_language,
    )


def stream_chunks(*texts: str) -> list[JsonObject]:
    return [{"choices": [{"delta": {"content": text}}]} for text in texts]


def with_usage(chunks: list[JsonObject], usage: JsonObject) -> list[JsonObject]:
    return [*chunks, {"choices": [{"delta": {}}], "usage": usage}]

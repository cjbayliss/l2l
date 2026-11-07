import json
import socket
from dataclasses import replace
from typing import TYPE_CHECKING, cast
lazy from collections.abc import Callable

import pycurl
from fakes import (
    FakeCurl,
    FakeCurlHttp,
    FakeHttp,
    FakePlainResponse,
    FakeStreamResponse,
    make_console,
    make_context,
    make_sleep_recorder,
    sse_bytes,
    stream_chunks,
    with_usage,
)
from http_server import JsonReply, json_reply, local_endpoint, raw_reply

from l2l.cli import curl_open_http
from l2l.console import Console, toggle_verbose
from l2l.errors import ConfigError, HttpError, TranslationError, describe, fail_http
from l2l.http import (
    CurlResponse,
    StreamState,
    chat,
    curl_error_code,
    curl_error_detail,
    curl_open,
    drive_stream,
    failed_provider_name,
    http_post_json,
    http_request,
    http_stream,
    is_error_status_line,
    is_openrouter,
    log_all,
    noproxy_for,
    proxy_for,
    retry_after_seconds,
    with_blocked_providers,
)
from l2l.monads import (
    IO,
    NOTHING,
    Err,
    Just,
    Ok,
    Result,
    cons_to_tuple,
    fold_io,
    io_and_then,
    io_map,
    io_pure,
    io_result,
    read_reference,
    write_reference,
)
from l2l.settings import UNSET
from l2l.text import Usage
lazy from l2l.json_values import JsonObject

if TYPE_CHECKING:
    from l2l.context import Context, HttpRequest, HttpResponse, OpenHTTP

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def test_curl_open_returns_a_configured_response() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeCurlHttp([]).open)
    request = http_request(run_context.config, {"model": "m"})
    opened = curl_open(request, 10.0, {})
    assert isinstance(opened, Ok)
    assert isinstance(opened.value, CurlResponse)
    assert opened.value.request == request
    assert opened.value.timeout == 10.0


def test_curl_open_reports_an_unreachable_endpoint() -> None:
    console, _ = make_console()
    curl = FakeCurl(error=pycurl.error(7, "Failed to connect to endpoint"))
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"
    assert "Failed to connect" in describe(failure)
    assert curl.closed


def test_curl_open_parses_status_and_retry_after() -> None:
    cases = (
        (
            429,
            (b"HTTP/1.1 429 Too Many Requests\r\n", b"Retry-After: 7\r\n"),
            b"slow down",
            429,
            "slow down",
            7.0,
        ),
        (500, (b"Retry-After: bogus\r\n",), b"boom", 500, "boom", None),
        (400, (), b"nope", 400, "nope", None),
    )
    for status, headers, chunks, expected_status, detail, retry_after in cases:
        console, _ = make_console()
        curl = FakeCurl(status=status, headers=headers, chunks=(chunks,))
        http = FakeCurlHttp([curl])
        run_context = make_context(console, http.open)
        result = http_post_json(run_context, {"model": "m"}).run()
        assert isinstance(result, Err)
        failure = result.error
        assert isinstance(failure, HttpError)
        assert failure.kind == "status"
        assert failure.status == expected_status
        assert failure.detail == detail
        assert failure.retry_after == retry_after
        assert curl.closed


def test_curl_open_survives_a_single_argument_error() -> None:
    console, _ = make_console()
    curl = FakeCurl(error=pycurl.error(28))
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"


def test_curl_error_helpers_tolerate_an_empty_error() -> None:
    empty = pycurl.error()
    assert curl_error_code(empty) == 0
    assert curl_error_detail(empty) == str(empty)


def test_curl_open_delivers_a_successful_body() -> None:
    console, _ = make_console()
    curl = FakeCurl(status=200, chunks=(b'{"ok": true}',))
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Ok)
    assert result.value == {"ok": True}


def test_http_request_builds_url_body_and_headers() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeCurlHttp([]).open)
    request = http_request(
        run_context.config, {"model": "m"}, accept="text/event-stream"
    )
    assert request.url == "http://endpoint.test/v1/chat/completions"
    assert json.loads(request.body) == {"model": "m"}
    assert request.headers["Authorization"] == "Bearer key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.headers["Accept"] == "text/event-stream"


def test_curl_response_sets_expected_options() -> None:
    console, _ = make_console()
    curl = FakeCurl()
    run_context = make_context(console, FakeCurlHttp([]).open)
    request = http_request(run_context.config, {"model": "m"})
    response = CurlResponse(request, 10.0, {}, lambda: curl)
    assert isinstance(response.body().run(), Ok)
    assert curl.options[pycurl.URL] == "http://endpoint.test/v1/chat/completions"
    assert curl.options[pycurl.POST] == 1
    assert curl.options[pycurl.POSTFIELDS] == request.body
    assert curl.options[pycurl.CONNECTTIMEOUT_MS] == 10000
    assert curl.options[pycurl.LOW_SPEED_LIMIT] == 1
    assert curl.options[pycurl.LOW_SPEED_TIME] == 10
    assert curl.options[pycurl.NOSIGNAL] == 1
    assert curl.options[pycurl.FOLLOWLOCATION] == 1
    assert curl.options[pycurl.MAXREDIRS] == 10
    headers = curl.options[pycurl.HTTPHEADER]
    assert isinstance(headers, list)
    configured_headers = cast("list[str]", headers)
    assert "Authorization: Bearer key" in configured_headers
    assert "Content-Type: application/json" in configured_headers
    assert any(header.startswith("User-Agent: l2l/") for header in configured_headers)
    assert curl.closed


def test_curl_response_configures_proxy_from_environment() -> None:
    console, _ = make_console()
    curl = FakeCurl(status=200, chunks=(b"{}",))
    run_context = make_context(console, FakeCurlHttp([]).open)
    request = http_request(run_context.config, {"model": "m"})
    environment = {"http_proxy": "http://proxy:1", "no_proxy": "localhost"}
    response = CurlResponse(request, 10.0, environment, lambda: curl)
    assert isinstance(response.body().run(), Ok)
    assert curl.options[pycurl.PROXY] == "http://proxy:1"
    assert curl.options[pycurl.NOPROXY] == "localhost"


def test_proxy_for_prefers_scheme_specific_variables() -> None:
    environment = {
        "http_proxy": "http://a:1",
        "https_proxy": "http://b:2",
        "ALL_PROXY": "http://c:3",
    }
    assert proxy_for("https://endpoint.test/v1", environment) == Just("http://b:2")
    assert proxy_for("http://endpoint.test/v1", environment) == Just("http://a:1")
    assert proxy_for("https://endpoint.test/v1", {}) is NOTHING
    assert noproxy_for({"NO_PROXY": "x"}) == Just("x")
    assert noproxy_for({}) is NOTHING


def test_retry_after_seconds_handles_garbage() -> None:
    assert retry_after_seconds(None) is NOTHING
    assert retry_after_seconds("") is NOTHING
    assert retry_after_seconds("bogus") is NOTHING
    assert retry_after_seconds("-3") == Just(0.0)
    assert retry_after_seconds("7") == Just(7.0)


def test_chat_retries_transient_failures_then_succeeds() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    calls: list[HttpRequest] = []

    def flaky_open(
        request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        calls.append(request)
        if len(calls) < 3:
            return fail_http("unreachable", "down")

        return Ok(FakeStreamResponse(chunks))

    run_context = make_context(console, flaky_open, sleep=sleep)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(calls) == 3
    assert sleeps() == (1.0, 2.0)


def test_chat_does_not_retry_protocol_failures() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    http = FakeHttp([FakePlainResponse({"nope": True})])
    run_context = make_context(console, http.open, sleep=sleep)
    result = chat(run_context, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Err)
    assert "unexpected response shape" in describe(result.error)
    assert len(http.requests) == 1
    assert sleeps() == ()


def test_chat_honours_retry_after_header() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    calls: list[HttpRequest] = []

    def rate_limited_then_ok(
        request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        calls.append(request)
        if len(calls) == 1:
            return fail_http("status", "slow down", 429, 9.0)

        return Ok(FakeStreamResponse(chunks))

    run_context = make_context(console, rate_limited_then_ok, sleep=sleep)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert len(calls) == 2
    assert sleeps() == (9.0,)


def test_chat_gives_up_after_retry_budget() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    calls: list[HttpRequest] = []

    def always_down(
        request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        calls.append(request)
        return fail_http("unreachable", "down")

    run_context = make_context(console, always_down, sleep=sleep)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "could not reach endpoint: down" in describe(result.error)
    assert len(calls) == 3
    assert sleeps() == (1.0, 2.0)


def test_chat_stream_setting_precedence() -> None:
    plain_body: JsonObject = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    cases: tuple[tuple[bool | None, dict[str, bool], bool, str], ...] = (
        (False, {}, False, "Plain"),
        (True, {"stream": False}, True, "Hi"),
        (None, {"stream": False}, False, "Plain"),
    )
    for context_stream, parameters, expected_stream, expected_text in cases:
        console, _ = make_console()
        response: FakePlainResponse | FakeStreamResponse = (
            FakeStreamResponse(with_usage(stream_chunks("Hi"), USAGE))
            if expected_stream
            else FakePlainResponse(plain_body)
        )
        http = FakeHttp([response])
        run_context = make_context(
            console,
            http.open,
            stream=UNSET if context_stream is None else context_stream,
        )
        result = chat(run_context, "s", "u", "m", parameters, Usage()).run()
        assert isinstance(result, Ok)
        assert result.value.text == expected_text
        assert json.loads(http.requests[0].body)["stream"] is expected_stream


class DyingPlainResponse:
    def body(self) -> IO[Result[bytes, TranslationError]]:
        def die() -> Result[bytes, TranslationError]:
            raise OSError("connection reset mid-body")

        return IO(die)

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        return io_result(initial)


def test_chat_reports_a_plain_response_that_dies_mid_body() -> None:
    console, _ = make_console()
    http = FakeHttp([DyingPlainResponse()])
    run_context = make_context(console, http.open, stream=False)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"
    assert "response read failed" in describe(failure)


class DyingStreamResponse:
    def __init__(self) -> None:
        self._lines: list[bytes] = [
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n',
            b"\n",
        ]

    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(fail_http("protocol", "streaming response has no body"))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        def die() -> Result[S, TranslationError]:
            raise OSError("connection reset mid-stream")

        return io_and_then(fold_io(self._lines, step, initial), IO(die))


def test_is_error_status_line_tolerates_malformed_codes() -> None:
    assert is_error_status_line(b"HTTP/1.1 oops Bad Request\r\n") is False
    assert is_error_status_line(b"HTTP/1.1 503 Service Unavailable\r\n") is True
    assert is_error_status_line(b"not a status line\r\n") is False


def test_chat_reports_an_interrupted_stream() -> None:
    console, _ = make_console()
    http = FakeHttp([DyingStreamResponse()])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "interrupted"
    assert "connection reset mid-stream" in describe(failure)


class TogglingStreamResponse:
    def __init__(
        self, chunks: list[JsonObject], console: Console, toggle_after_chunks: int
    ) -> None:
        self._lines = list(sse_bytes(chunks))
        self._console = console
        self._toggle_after = toggle_after_chunks * 2

    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(fail_http("protocol", "streaming response has no body"))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        def toggling(
            state: S, indexed: tuple[int, bytes]
        ) -> IO[Result[S, TranslationError]]:
            index, line = indexed
            return io_and_then(
                io_map(
                    (
                        toggle_verbose(self._console)
                        if index == self._toggle_after
                        else io_pure(False)
                    ),
                    lambda _: None,
                ),
                step(state, line),
            )

        return fold_io(tuple(enumerate(self._lines)), toggling, initial)


def test_chat_shows_streamed_reasoning_once_across_a_mid_stream_toggle() -> None:
    console, stderr = make_console()
    chunks: list[JsonObject] = [
        {"choices": [{"delta": {"reasoning_content": "Most"}}]},
        {"choices": [{"delta": {"reasoning_content": "ly fine."}}]},
        {"choices": [{"delta": {"reasoning_content": " Second line."}}]},
        {"choices": [{"delta": {"content": "Hi"}}]},
    ]
    http = FakeHttp([TogglingStreamResponse(chunks, console, toggle_after_chunks=2)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert stderr.getvalue() == " Second line.\n"
    sealed = [
        event.text
        for event in cons_to_tuple(console.events.value)
        if event.raw and event.verbose_only
    ]
    assert sealed == ["Mostly fine. Second line."]


class UnterminatedStreamResponse:
    def __init__(self, chunks: list[JsonObject]) -> None:
        self._lines = sse_bytes(chunks)[:-1]

    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(fail_http("protocol", "streaming response has no body"))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        return fold_io(self._lines, step, initial)


def test_chat_flushes_an_unterminated_final_frame() -> None:
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    http = FakeHttp([UnterminatedStreamResponse(chunks)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert result.value.usage == Usage(5, 6, 0.2)


class GarbagePlainResponse:
    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(Ok(b"{not json at all"))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        return io_result(initial)


def test_chat_reports_invalid_json_body_as_protocol_error() -> None:
    console, _ = make_console()
    http = FakeHttp([GarbagePlainResponse()])
    run_context = make_context(console, http.open, stream=False)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "invalid JSON response" in describe(result.error)


class ExplodingPlainResponse:
    def body(self) -> IO[Result[bytes, TranslationError]]:
        def explode() -> Result[bytes, TranslationError]:
            raise ValueError("bad body")

        return IO(explode)

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        return io_result(initial)


def test_chat_reports_a_plain_read_failure_as_protocol_error() -> None:
    console, _ = make_console()
    http = FakeHttp([ExplodingPlainResponse()])
    run_context = make_context(console, http.open, stream=False)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "protocol"
    assert "ValueError" in describe(failure)


def test_chat_reopens_a_stream_without_stream_options() -> None:
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    requests: list[HttpRequest] = []

    def picky_open(
        request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        requests.append(request)
        if "stream_options" in json.loads(request.body):
            return fail_http("status", "stream_options is unsupported", 400)

        return Ok(FakeStreamResponse(chunks))

    run_context = make_context(console, picky_open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(requests) == 2
    assert "stream_options" not in json.loads(requests[1].body)


def test_chat_reopens_a_curl_stream_without_stream_options() -> None:
    console, _ = make_console()
    curl = FakeCurl(
        status=400,
        headers=(b"Content-Type: application/json\r\n",),
        chunks=(b'{"error": {"message": "stream_options is unsupported"}}',),
    )
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "stream_options" in describe(result.error)
    assert len(http.requests) == 2
    assert curl.performed == 2


def test_http_stream_retries_only_when_the_request_sends_stream_options() -> None:
    console, _ = make_console()
    curl = FakeCurl(
        status=400,
        headers=(b"HTTP/1.1 400 Bad Request\r\n",),
        chunks=(b"stream_options is unsupported",),
    )
    guarded = FakeCurlHttp([curl])
    run_context = make_context(console, guarded.open)

    def plain_drive(
        response: HttpResponse,
    ) -> IO[Result[StreamState, TranslationError]]:
        return drive_stream(response, lambda label, count: io_pure(None))

    outcome = http_stream(run_context, {"model": "m"}, plain_drive).run()
    assert isinstance(outcome, Err)
    assert len(guarded.requests) == 1

    retrying = FakeCurlHttp([curl])
    run_context = make_context(console, retrying.open)
    outcome = http_stream(
        run_context, {"model": "m", "stream_options": {}}, plain_drive
    ).run()
    assert isinstance(outcome, Err)
    assert len(retrying.requests) == 2
    assert b"stream_options" not in retrying.requests[1].body


def test_chat_reports_endpoint_error_events_from_a_stream() -> None:
    console, _ = make_console()
    http = FakeHttp([FakeStreamResponse([{"error": {"message": "overloaded"}}])])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "overloaded" in describe(result.error)


class ExplodingStreamResponse:
    def body(self) -> IO[Result[bytes, TranslationError]]:
        return io_result(fail_http("protocol", "streaming response has no body"))

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        def explode() -> Result[S, TranslationError]:
            raise ValueError("bad frame")

        return IO(explode)


def test_chat_reports_malformed_stream_frames_as_protocol_errors() -> None:
    console, _ = make_console()
    http = FakeHttp([ExplodingStreamResponse()])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "protocol"
    assert "ValueError" in describe(failure)


def test_chat_flushes_a_partial_final_line_from_a_stream() -> None:
    console, _ = make_console()
    curl = FakeCurl(
        chunks=(
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
            b"data: [DONE]",
        )
    )
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"


def test_chat_rejects_a_stream_that_ends_without_a_completion_event() -> None:
    console, _ = make_console()
    http = FakeHttp([FakeStreamResponse(stream_chunks("Hi"), terminated=False)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "stream"
    assert "without a completion event" in describe(failure)


def test_chat_rejects_a_stream_truncated_by_length() -> None:
    console, _ = make_console()
    chunks = stream_chunks("Hi") + [
        {"choices": [{"delta": {}, "finish_reason": "length"}]}
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "stream"
    assert "finish_reason=length" in describe(failure)


def test_chat_rejects_a_plain_response_truncated_by_length() -> None:
    console, _ = make_console()
    body: JsonObject = {
        "choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}]
    }
    http = FakeHttp([FakePlainResponse(body)])
    run_context = make_context(console, http.open, stream=False)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "stream"
    assert "finish_reason=length" in describe(failure)


def test_chat_reports_malformed_sse_data_as_a_stream_error() -> None:
    console, _ = make_console()
    curl = FakeCurl(
        chunks=(
            b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
            b"data: {truncated json\n\n",
        )
    )
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "stream"
    assert "malformed SSE data" in describe(failure)


def test_chat_retries_a_stream_that_fails_to_connect() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    curl = FakeCurl(error=pycurl.error(7, "Failed to connect"))
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open, sleep=sleep)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"
    assert sleeps() == (1.0, 2.0)


def test_chat_reports_a_dropped_stream_without_retrying() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    curl = FakeCurl(
        chunks=sse_bytes(stream_chunks("Hi")),
        error=pycurl.error(56, "Recv failure: connection reset"),
    )
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open, sleep=sleep)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "interrupted"
    assert sleeps() == ()


def test_chat_does_not_feed_error_bodies_to_the_stream_pipeline() -> None:
    console, _ = make_console()
    curl = FakeCurl(
        status=400,
        headers=(b"HTTP/1.1 400 Bad Request\r\n",),
        chunks=(b'data: {"error": {"message": "poison"}}\n\n',),
    )
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "status"
    assert "poison" in failure.detail


def test_curl_response_uses_the_final_response_after_redirects() -> None:
    console, _ = make_console()
    curl = FakeCurl(
        stages=(
            (
                301,
                (b"HTTP/1.1 301 Moved Permanently\r\n", b"Retry-After: 9\r\n"),
                (b"redirecting away",),
            ),
            (
                429,
                (b"HTTP/1.1 429 Too Many Requests\r\n", b"Retry-After: 3\r\n"),
                (b"slow down",),
            ),
        )
    )
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "status"
    assert failure.status == 429
    assert failure.detail == "slow down"
    assert failure.retry_after == 3.0


def test_log_all_writes_each_message_when_verbose() -> None:
    console, stream = make_console()
    write_reference(console.verbose, True).run()
    outcome = log_all(console, ("one", "two")).run()
    assert outcome is None
    assert stream.getvalue() == "one\ntwo\n"


DIRECT_ENVIRONMENT = {"no_proxy": "127.0.0.1,localhost"}


def real_open_http(
    request: HttpRequest, timeout: float
) -> Result[HttpResponse, TranslationError]:
    return curl_open(request, timeout, DIRECT_ENVIRONMENT)


def local_context(console: Console, endpoint_url: str) -> Context:
    run_context = make_context(console, real_open_http)
    return replace(
        run_context, config=replace(run_context.config, base_url=endpoint_url)
    )


def closed_local_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def test_curl_open_http_wires_the_environment_into_curl() -> None:
    console, _ = make_console()
    with local_endpoint(lambda body, headers: json_reply({"echo": "ok"})) as endpoint:
        base = make_context(console, curl_open_http(DIRECT_ENVIRONMENT))
        run_context = replace(base, config=replace(base.config, base_url=endpoint.url))
        result = http_post_json(run_context, {"model": "m"}).run()

    assert isinstance(result, Ok)
    assert result.value == {"echo": "ok"}


def test_http_post_json_round_trips_a_local_server() -> None:
    console, _ = make_console()
    with local_endpoint(lambda body, headers: json_reply({"echo": "ok"})) as endpoint:
        result = http_post_json(
            local_context(console, endpoint.url), {"model": "m"}
        ).run()
    assert isinstance(result, Ok)
    assert result.value == {"echo": "ok"}
    assert len(endpoint.requests) == 1
    sent = endpoint.requests[0]
    assert sent.path == "/v1/chat/completions"
    assert sent.headers["Authorization"] == "Bearer key"
    assert sent.payload() == {"model": "m"}


def test_http_post_json_reports_a_status_failure_from_a_local_server() -> None:
    console, _ = make_console()

    def overloaded(body: bytes, headers: dict[str, str]) -> JsonReply:
        return (429, (("Retry-After", "7"),), b"slow down")

    with local_endpoint(overloaded) as endpoint:
        result = http_post_json(
            local_context(console, endpoint.url), {"model": "m"}
        ).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "status"
    assert failure.status == 429
    assert failure.detail == "slow down"
    assert failure.retry_after == 7.0


def test_http_post_json_reports_an_unreachable_local_port() -> None:
    console, _ = make_console()
    endpoint_url = "http://127.0.0.1:%d/v1" % closed_local_port()
    result = http_post_json(local_context(console, endpoint_url), {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"


def test_http_post_json_reports_invalid_json_from_a_local_server() -> None:
    console, _ = make_console()
    with local_endpoint(lambda body, headers: raw_reply(200, b"not json")) as endpoint:
        result = http_post_json(
            local_context(console, endpoint.url), {"model": "m"}
        ).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "protocol"
    assert "invalid JSON" in describe(failure)


def test_chat_streams_a_completion_from_a_local_server() -> None:
    console, _ = make_console()

    def stream_hello(body: bytes, headers: dict[str, str]) -> JsonReply:
        frames = (
            b'data: {"choices": [{"delta": {"content": "Hel"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "lo"}}]}\n\n',
            b"data: [DONE]\n\n",
        )
        return (200, (("Content-Type", "text/event-stream"),), b"".join(frames))

    with local_endpoint(stream_hello) as endpoint:
        run_context = replace(local_context(console, endpoint.url), stream=True)
        result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hello"
    sent = endpoint.requests[0]
    assert sent.headers["Accept"] == "text/event-stream"
    payload = sent.payload()
    assert payload["stream"] is True
    stream_options = payload["stream_options"]
    assert isinstance(stream_options, dict)
    assert "include_usage" in stream_options


OPENROUTER_429_DETAIL = (
    '{"error":{"message":"Provider returned error","code":429,'
    '"metadata":{"raw":"deepseek is temporarily rate-limited upstream. '
    'Please retry shortly: https://openrouter.ai/settings/integrations",'
    '"provider_name":"Modal","is_byok":false}}}'
)


def openrouter_context(console: Console, open_http: OpenHTTP) -> Context:
    run_context = make_context(console, open_http)
    return replace(
        run_context,
        config=replace(run_context.config, base_url="https://openrouter.ai/api/v1"),
    )


def test_is_openrouter_matches_the_base_url() -> None:
    console, _ = make_console()
    assert is_openrouter(openrouter_context(console, FakeHttp([]).open).config)
    assert not is_openrouter(make_context(console, FakeHttp([]).open).config)


def test_failed_provider_name_extracts_the_provider_from_the_detail() -> None:
    failure = fail_http("status", OPENROUTER_429_DETAIL, 429).error
    assert failed_provider_name(failure) == "Modal"
    assert failed_provider_name(fail_http("status", "slow down", 429).error) is None
    assert failed_provider_name(ConfigError("nope")) is None


def test_with_blocked_providers_adds_an_ignore_list_on_openrouter() -> None:
    console, _ = make_console()
    run_context = openrouter_context(console, FakeHttp([]).open)
    payload: JsonObject = {"model": "m", "provider": {"order": ["A"]}}
    scoped = with_blocked_providers(
        run_context.config, payload, frozenset({"Modal", "DeepInfra"})
    )
    assert scoped["provider"] == {
        "order": ["A"],
        "ignore": ["DeepInfra", "Modal"],
    }
    plain = {"model": "m"}
    assert with_blocked_providers(run_context.config, plain, frozenset()) is plain
    other = make_context(console, FakeHttp([]).open)
    assert with_blocked_providers(other.config, plain, frozenset({"Modal"})) is plain


def rate_limited_providers_open(
    requests: list[HttpRequest],
    chunks: list[JsonObject],
    providers: tuple[str, ...],
) -> OpenHTTP:
    def open_http(
        request: HttpRequest, timeout: float
    ) -> Result[HttpResponse, TranslationError]:
        requests.append(request)
        index = len(requests) - 1
        if index < len(providers):
            detail = OPENROUTER_429_DETAIL.replace("Modal", providers[index])
            return fail_http("status", detail, 429)

        return Ok(FakeStreamResponse(chunks))

    return open_http


def rate_limited_chat_context(
    providers: tuple[str, ...], provider_block_attempts: int | None = None
) -> tuple[Context, list[HttpRequest], Callable[[], tuple[float, ...]]]:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    requests: list[HttpRequest] = []
    run_context = openrouter_context(
        console, rate_limited_providers_open(requests, chunks, providers)
    )
    settings = run_context.settings
    if provider_block_attempts is not None:
        settings = replace(settings, provider_block_attempts=provider_block_attempts)
    return replace(run_context, sleep=sleep, settings=settings), requests, sleeps


def test_chat_blocks_a_failing_openrouter_provider_and_retries() -> None:
    run_context, requests, sleeps = rate_limited_chat_context(("Modal",))
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(requests) == 2
    first = json.loads(requests[0].body)
    second = json.loads(requests[1].body)
    assert "provider" not in first
    assert second["provider"] == {"ignore": ["Modal"]}
    assert read_reference(run_context.provider_blocklist).run() == frozenset({"Modal"})
    assert sleeps() == ()


def test_chat_blocks_each_new_rate_limited_provider_without_backoff() -> None:
    run_context, requests, sleeps = rate_limited_chat_context(
        ("Modal", "DeepInfra", "Crusoe")
    )
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(requests) == 4
    ignored = [
        json.loads(request.body).get("provider", {}).get("ignore", [])
        for request in requests
    ]
    assert ignored == [
        [],
        ["Modal"],
        ["DeepInfra", "Modal"],
        ["Crusoe", "DeepInfra", "Modal"],
    ]
    assert read_reference(run_context.provider_blocklist).run() == frozenset(
        {"Modal", "DeepInfra", "Crusoe"}
    )
    assert sleeps() == ()


def test_chat_falls_back_to_backoff_after_exhausting_provider_blocks() -> None:
    run_context, requests, sleeps = rate_limited_chat_context(
        ("Modal", "DeepInfra"), provider_block_attempts=1
    )
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(requests) == 3
    third = json.loads(requests[2].body)
    assert third["provider"] == {"ignore": ["DeepInfra", "Modal"]}
    assert sleeps() == (1.0,)


def test_chat_repeats_backoff_for_a_known_rate_limited_provider() -> None:
    run_context, requests, sleeps = rate_limited_chat_context(
        ("Modal", "Modal", "Modal")
    )
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(requests) == 4
    assert sleeps() == (1.0, 2.0)


def test_chat_blocklist_persists_across_chat_calls() -> None:
    run_context, requests, _ = rate_limited_chat_context(("Modal",))
    first = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(first, Ok)
    second = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(second, Ok)
    assert "provider" not in json.loads(requests[0].body)
    assert json.loads(requests[1].body)["provider"] == {"ignore": ["Modal"]}
    assert json.loads(requests[2].body)["provider"] == {"ignore": ["Modal"]}


def test_chat_does_not_block_providers_off_openrouter() -> None:
    sleep, sleeps = make_sleep_recorder()
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    requests: list[HttpRequest] = []
    run_context = make_context(
        console, rate_limited_providers_open(requests, chunks, ("Modal",)), sleep=sleep
    )
    result = chat(run_context, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert len(requests) == 2
    assert "provider" not in json.loads(requests[1].body)
    assert read_reference(run_context.provider_blocklist).run() == frozenset()
    assert sleeps() == (1.0,)
    assert read_reference(run_context.provider_blocklist).run() == frozenset()

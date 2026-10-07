import json
from typing import Any

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

from l2l.console import toggle_verbose
from l2l.errors import HttpError, TranslationError, describe, fail_http
from l2l.http import (
    CurlResponse,
    chat,
    curl_error_code,
    curl_error_detail,
    curl_open,
    drive_stream,
    http_post_json,
    http_request,
    http_stream,
    is_error_status_line,
    log_all,
    noproxy_for,
    proxy_for,
    retry_after_seconds,
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
    write_reference,
)
from l2l.text import Usage

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def test_curl_open_returns_a_configured_response() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeCurlHttp([]).open)
    request = http_request(run_context.config, {"model": "m"})
    opened = curl_open(request, 10.0, {})
    assert isinstance(opened, Ok)
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
    console, _ = make_console()
    curl = FakeCurl(
        status=429,
        headers=(b"HTTP/1.1 429 Too Many Requests\r\n", b"Retry-After: 7\r\n"),
        chunks=(b"slow down",),
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
    assert failure.retry_after == 7.0
    assert curl.closed


def test_curl_open_ignores_a_garbage_retry_after() -> None:
    console, _ = make_console()
    curl = FakeCurl(status=500, headers=(b"Retry-After: bogus\r\n",), chunks=(b"boom",))
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "status"
    assert failure.detail == "boom"
    assert failure.retry_after is None


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


def test_curl_open_reports_a_status_error_without_retry_after() -> None:
    console, _ = make_console()
    curl = FakeCurl(status=400, headers=(), chunks=(b"nope",))
    http = FakeCurlHttp([curl])
    run_context = make_context(console, http.open)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "status"
    assert failure.status == 400
    assert failure.detail == "nope"
    assert failure.retry_after is None


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
    assert "Authorization: Bearer key" in headers
    assert "Content-Type: application/json" in headers
    assert any(header.startswith("User-Agent: l2l/") for header in headers)
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
    calls: list[Any] = []

    def flaky_open(request: Any, timeout: float) -> Any:
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
    calls: list[Any] = []

    def rate_limited_then_ok(request: Any, timeout: float) -> Any:
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
    calls: list[Any] = []

    def always_down(request: Any, timeout: float) -> Any:
        calls.append(request)
        return fail_http("unreachable", "down")

    run_context = make_context(console, always_down, sleep=sleep)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "could not reach endpoint: down" in describe(result.error)
    assert len(calls) == 3
    assert sleeps() == (1.0, 2.0)


def test_chat_context_stream_false_forces_plain() -> None:
    console, _ = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    run_context = make_context(console, http.open, stream=False)
    result = chat(run_context, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Plain"
    assert json.loads(http.requests[0].body)["stream"] is False


def test_chat_context_stream_true_overrides_params() -> None:
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    run_context = make_context(console, http.open, stream=True)
    result = chat(run_context, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert json.loads(http.requests[0].body)["stream"] is True


def test_chat_without_override_keeps_params_stream_false() -> None:
    console, _ = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    run_context = make_context(console, http.open)
    result = chat(run_context, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert json.loads(http.requests[0].body)["stream"] is False


class DyingPlainResponse:
    def body(self) -> IO[Result[bytes, TranslationError]]:
        def die() -> Result[bytes, TranslationError]:
            raise OSError("connection reset mid-body")

        return IO(die)


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

    def consume(
        self,
        step: Any,
        initial: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        def die() -> Result[Any, TranslationError]:
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
        self, chunks: list[dict[str, Any]], console: Any, toggle_after_chunks: int
    ) -> None:
        self._lines = list(sse_bytes(chunks))
        self._console = console
        self._toggle_after = toggle_after_chunks * 2

    def consume(
        self,
        step: Any,
        initial: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        def toggling(state: Any, indexed: tuple[int, bytes]) -> Any:
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
    chunks = [
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
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._lines = sse_bytes(chunks)[:-1]

    def consume(
        self,
        step: Any,
        initial: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
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


def test_http_post_json_reports_an_unreachable_endpoint() -> None:
    console, _ = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, TranslationError]:
        return fail_http("unreachable", "down")

    run_context = make_context(console, open_fail)
    result = http_post_json(run_context, {"model": "m"}).run()
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"


def test_chat_reopens_a_stream_without_stream_options() -> None:
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    requests: list[Any] = []

    def picky_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
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

    def plain_drive(response: Any) -> Any:
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
    def consume(
        self,
        step: Any,
        initial: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        def explode() -> Result[Any, TranslationError]:
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
    body = {"choices": [{"message": {"content": "Partial"}, "finish_reason": "length"}]}
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
    assert outcome == Ok(())
    assert stream.getvalue() == "one\ntwo\n"

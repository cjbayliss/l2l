import io
import json
import urllib.error
from email.message import Message
from typing import Any

from fakes import (
    SLEEPS,
    FakeHttp,
    FakePlainResponse,
    FakeStreamResponse,
    make_console,
    make_context,
    reset_sleeps,
    stream_chunks,
    with_usage,
)

from zh2en.errors import HttpError, describe, fail_http
from zh2en.http import chat, retry_after_seconds, urllib_open
from zh2en.monads import Err, Ok
from zh2en.text import Usage

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def _retry_after_headers(value: str) -> Message:
    headers = Message()
    headers["Retry-After"] = value
    return headers


def test_urllib_open_reports_unreachable(monkeypatch: Any) -> None:
    def raise_url_error(request: Any, timeout: float) -> Any:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", raise_url_error)
    result = urllib_open("request", 1.0)
    assert isinstance(result, Err)
    assert isinstance(result.error, HttpError)
    assert result.error.kind == "unreachable"


def test_urllib_open_parses_status_and_retry_after(monkeypatch: Any) -> None:
    error = urllib.error.HTTPError(
        "http://endpoint",
        429,
        "Too Many Requests",
        _retry_after_headers("7"),
        io.BytesIO(b"slow down"),
    )

    def raise_http_error(request: Any, timeout: float) -> Any:
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)
    result = urllib_open("request", 1.0)
    assert isinstance(result, Err)
    failure = result.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "status"
    assert failure.status == 429
    assert failure.detail == "slow down"
    assert failure.retry_after == 7.0


def test_retry_after_seconds_handles_garbage() -> None:
    assert retry_after_seconds(_retry_after_headers("bogus")) is None
    assert retry_after_seconds(Message()) is None
    assert retry_after_seconds(None) is None
    assert retry_after_seconds(_retry_after_headers("-3")) == 0.0
    assert retry_after_seconds(_retry_after_headers("7")) == 7.0


def test_chat_retries_transient_failures_then_succeeds() -> None:
    reset_sleeps()
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    calls: list[Any] = []

    def flaky_open(request: Any, timeout: float) -> Any:
        calls.append(request)
        if len(calls) < 3:
            return fail_http("unreachable", "down")

        return Ok(FakeStreamResponse(chunks))

    ctx = make_context(console, flaky_open)
    result = chat(ctx, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert len(calls) == 3
    assert SLEEPS == [1.0, 2.0]


def test_chat_does_not_retry_protocol_failures() -> None:
    reset_sleeps()
    console, _ = make_console()
    http = FakeHttp([FakePlainResponse({"nope": True})])
    ctx = make_context(console, http.open)
    result = chat(ctx, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Err)
    assert "unexpected response shape" in describe(result.error)
    assert len(http.requests) == 1
    assert SLEEPS == []


def test_chat_honours_retry_after_header() -> None:
    reset_sleeps()
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    calls: list[Any] = []

    def rate_limited_then_ok(request: Any, timeout: float) -> Any:
        calls.append(request)
        if len(calls) == 1:
            return fail_http("status", "slow down", 429, 9.0)

        return Ok(FakeStreamResponse(chunks))

    ctx = make_context(console, rate_limited_then_ok)
    result = chat(ctx, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert len(calls) == 2
    assert SLEEPS == [9.0]


def test_chat_gives_up_after_retry_budget() -> None:
    reset_sleeps()
    console, _ = make_console()
    calls: list[Any] = []

    def always_down(request: Any, timeout: float) -> Any:
        calls.append(request)
        return fail_http("unreachable", "down")

    ctx = make_context(console, always_down)
    result = chat(ctx, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "could not reach endpoint: down" in describe(result.error)
    assert len(calls) == 3
    assert SLEEPS == [1.0, 2.0]


def test_chat_context_stream_false_forces_plain() -> None:
    console, _ = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    ctx = make_context(console, http.open, stream=False)
    result = chat(ctx, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Plain"
    assert json.loads(http.requests[0].data)["stream"] is False


def test_chat_context_stream_true_overrides_params() -> None:
    console, _ = make_console()
    chunks = with_usage(stream_chunks("Hi"), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    ctx = make_context(console, http.open, stream=True)
    result = chat(ctx, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert json.loads(http.requests[0].data)["stream"] is True


def test_chat_without_override_keeps_params_stream_false() -> None:
    console, _ = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    ctx = make_context(console, http.open)
    result = chat(ctx, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert json.loads(http.requests[0].data)["stream"] is False

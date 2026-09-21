from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any, TypeVar

from zh2en import __version__
from zh2en.console import Console
from zh2en.effects import log_entry, log_error, log_request, run_log_write
from zh2en.errors import HttpError, TranslationError, describe, fail_budget, fail_http
from zh2en.monads import (
    IO,
    NOTHING,
    Err,
    Just,
    Maybe,
    Nothing,
    Ok,
    Result,
    fold_io,
    fold_io_lazy,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_when,
    maybe_either,
    maybe_or,
    result_bind,
    result_map,
)
from zh2en.plans import plan_backoff, retry_delay, transient
from zh2en.settings import Config, Context
from zh2en.text import (
    SseState,
    ThinkState,
    Translated,
    Usage,
    add_usage,
    estimate_tokens,
    sse_step,
    strip_think_tag,
    think_step,
)

T = TypeVar("T")

ProgressCallback = Callable[[str, int], IO[None]]
RawLineLogger = Callable[[bytes], IO[None]]


def build_chat_payload(
    model: str, system: str, user: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **{
            "model": model,
            "messages": (
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
        },
        **{key: value for key, value in params.items() if value is not None},
    }


def extract_message(
    body: Any,
) -> Result[tuple[Mapping[str, Any], Any], TranslationError]:
    try:
        message = body["choices"][0]["message"]
        return Ok((message, message["content"]))
    except (KeyError, IndexError, TypeError):
        return fail_http("protocol", "unexpected response shape: %s" % str(body)[:500])


def collect_thinking_texts(part: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        entry["text"]
        for entry in (part.get("thinking") or [])
        if isinstance(entry, dict) and entry.get("text")
    )


def collect_content_parts(parts: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def step(
        accumulator: tuple[tuple[str, ...], tuple[str, ...]], part: Any
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        texts, thoughts = accumulator
        if not isinstance(part, dict):
            return texts + (str(part),), thoughts

        if part.get("type") == "thinking":
            return texts, thoughts + collect_thinking_texts(part)

        if part.get("text"):
            return texts + (part["text"],), thoughts

        return accumulator

    initial: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    return reduce(step, parts, initial)


def flatten_content_parts(content: Any) -> tuple[Any, tuple[str, ...]]:
    if not isinstance(content, list):
        return content, ()

    texts, thoughts = collect_content_parts(content)
    return "".join(texts), thoughts


def message_reasoning_texts(message: Mapping[str, Any]) -> tuple[str, ...]:
    def reasoning_at(key: str) -> Maybe[str]:
        value = message.get(key)
        return (
            Just(value.rstrip())
            if isinstance(value, str) and value.strip()
            else NOTHING
        )

    found = maybe_or(reasoning_at("reasoning_content"), reasoning_at("reasoning"))
    return maybe_either(found, lambda text: (text,), tuple)


def parse_chunk_delta(chunk: Mapping[str, Any]) -> dict[str, Any]:
    try:
        choices = chunk.get("choices")
        if not choices:
            return {}

        return choices[0].get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return {}


def delta_reasoning_text(delta: Mapping[str, Any]) -> Maybe[str]:
    def reasoning_at(key: str) -> Maybe[str]:
        value = delta.get(key)
        return Just(value) if isinstance(value, str) and value else NOTHING

    return maybe_or(reasoning_at("reasoning_content"), reasoning_at("reasoning"))


def delta_text(delta: Mapping[str, Any]) -> str:
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))

    return ""


@dataclass(frozen=True)
class ProgressRequest:
    label: str
    count: int


@dataclass(frozen=True)
class StreamState:
    reasoning: tuple[str, ...] = ()
    contents: tuple[str, ...] = ()
    reported: Mapping[str, Any] | None = None
    counted: int = 0
    content_started: bool = False
    think: ThinkState = ThinkState()
    sse: SseState = SseState()
    progress_request: ProgressRequest | None = None


def stream_step(
    state: StreamState, chunk: Mapping[str, Any]
) -> Result[StreamState, TranslationError]:
    usage_report = chunk.get("usage")
    reported = (
        MappingProxyType(usage_report)
        if isinstance(usage_report, dict)
        else state.reported
    )
    delta = parse_chunk_delta(chunk)
    reasoning_found = delta_reasoning_text(delta)
    text = delta_text(delta)
    if isinstance(reasoning_found, Nothing) and not text:
        return Ok(replace(state, reported=reported, progress_request=None))

    think_state, thinking, visible = (
        think_step(state.think, text) if text else (state.think, False, "")
    )
    reasoning_texts: tuple[str, ...] = maybe_either(
        reasoning_found, lambda value: (value,), lambda: ()
    )
    thinking_progress = 1 if reasoning_texts and not state.content_started else 0
    text_progress = 1 if text else 0
    progress_request: ProgressRequest | None = None
    if thinking_progress or text_progress:
        label = ("Thinking" if thinking else "Working") if text else "Thinking"
        progress_request = ProgressRequest(label, thinking_progress + text_progress)

    return Ok(
        StreamState(
            reasoning=state.reasoning + reasoning_texts,
            contents=state.contents + (visible,) if visible else state.contents,
            reported=reported,
            counted=state.counted + thinking_progress + text_progress,
            content_started=state.content_started or bool(text),
            think=think_state,
            progress_request=progress_request,
        )
    )


def step_stream(
    state: StreamState, raw_line: bytes
) -> Result[StreamState, TranslationError]:
    sse, chunk = sse_step(state.sse, raw_line)
    state = state if sse == state.sse else replace(state, sse=sse)
    if chunk is None:
        return Ok(replace(state, progress_request=None))

    if isinstance(chunk.get("error"), dict):
        return fail_http("stream", str(chunk["error"])[:500])

    return stream_step(state, chunk)


@dataclass(frozen=True)
class ChatReply:
    content: Any
    reasoning: tuple[str, ...] = ()
    reported: Mapping[str, Any] | None = None
    counted: int = 0


def plain_reply(body: Any) -> Result[ChatReply, TranslationError]:
    message_result = extract_message(body)
    if isinstance(message_result, Err):
        return message_result

    message, content = message_result.value
    texts, thoughts = flatten_content_parts(content)
    usage_report = body.get("usage")
    reported = (
        MappingProxyType(usage_report) if isinstance(usage_report, dict) else None
    )
    return Ok(
        ChatReply(
            content=texts,
            reasoning=message_reasoning_texts(message) + thoughts,
            reported=reported,
            counted=0,
        )
    )


def http_request(
    config: Config, payload: Mapping[str, Any], accept: str | None = None
) -> urllib.request.Request:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + config.api_key,
        "User-Agent": "zh2en/" + __version__,
    }
    if accept:
        headers = {**headers, "Accept": accept}

    return urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def urllib_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
    try:
        return Ok(urllib.request.urlopen(request, timeout=timeout))
    except urllib.error.HTTPError as error:
        return fail_http(
            "status",
            error.read().decode("utf-8", "replace")[:500],
            error.code,
            retry_after_seconds(error.headers),
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return fail_http("unreachable", str(error))


def retry_after_seconds(headers: Any) -> float | None:
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        return None

    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return None


def verbose_retry_log(
    ctx: Context, retry_number: int, retries: int, wait: float
) -> IO[None]:
    return io_when(
        ctx.verbose,
        ctx.console.log(
            "zh2en: transient failure; retry %d/%d in %.1fs"
            % (retry_number, retries, wait)
        ),
    )


def with_retries(
    ctx: Context, attempt: Callable[[], IO[Result[T, TranslationError]]]
) -> IO[Result[T, TranslationError]]:
    delays = plan_backoff(
        ctx.settings.retry_base_delay,
        ctx.settings.retry_cap,
        ctx.settings.retry_attempts,
    )

    def attempt_at(index: int) -> IO[Result[T, TranslationError]]:
        def decide(
            outcome: Result[T, TranslationError],
        ) -> IO[Result[T, TranslationError]]:
            if isinstance(outcome, Ok):
                return io_result(outcome)

            failure = outcome.error
            if index >= len(delays) or not transient(failure):
                return io_result(outcome)

            wait = retry_delay(failure, delays[index])
            return io_bind(
                verbose_retry_log(ctx, index + 1, len(delays), wait),
                lambda _: io_and_then(
                    IO(lambda: ctx.sleep(wait)),
                    attempt_at(index + 1),
                ),
            )

        return io_bind(attempt(), decide)

    return attempt_at(0)


def http_post_json(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[dict[str, Any], TranslationError]]:
    def thunk() -> Result[dict[str, Any], TranslationError]:
        log_request(ctx.log, payload).run()
        opened = ctx.open_http(http_request(ctx.config, payload), ctx.config.timeout)
        if isinstance(opened, Err):
            log_error(ctx.log, describe(opened.error)).run()
            return opened

        with opened.value as response:
            try:
                body = response.read().decode("utf-8")
                log_entry(ctx.log, "RESPONSE", body).run()
                return Ok(json.loads(body))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                failure = fail_http("protocol", "invalid JSON response: %s" % error)
                log_error(ctx.log, describe(failure.error)).run()
                return failure

    return IO(thunk)


def http_open_stream(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[Any, TranslationError]]:
    def open_stream(
        body: Mapping[str, Any], label: str
    ) -> Result[Any, TranslationError]:
        log_request(ctx.log, body, label).run()
        opened = ctx.open_http(
            http_request(ctx.config, body, accept="text/event-stream"),
            ctx.config.timeout,
        )
        if isinstance(opened, Err):
            log_error(ctx.log, describe(opened.error)).run()

        return opened

    def thunk() -> Result[Any, TranslationError]:
        opened = open_stream(payload, "REQUEST (stream)")
        if (
            isinstance(opened, Ok)
            or "stream_options" not in payload
            or not (
                isinstance(opened.error, HttpError)
                and "stream_options" in opened.error.detail
            )
        ):
            return opened

        return open_stream(
            {key: value for key, value in payload.items() if key != "stream_options"},
            "REQUEST (stream, retry)",
        )

    return IO(thunk)


def drive_stream(
    response: Any,
    on_progress: ProgressCallback,
    on_raw_line: RawLineLogger | None = None,
) -> IO[Result[StreamState, TranslationError]]:
    def advance(
        state: StreamState, raw_line: bytes
    ) -> IO[Result[StreamState, TranslationError]]:
        def notify(
            outcome: Result[StreamState, TranslationError],
        ) -> IO[Result[StreamState, TranslationError]]:
            if isinstance(outcome, Err):
                return io_result(outcome)

            request = outcome.value.progress_request
            if request is None:
                return io_result(outcome)

            return io_map(
                on_progress(request.label, request.count), lambda _: outcome
            )

        def after_log(_: None) -> IO[Result[StreamState, TranslationError]]:
            return io_bind(io_result(step_stream(state, raw_line)), notify)

        logged = io_pure(None) if on_raw_line is None else on_raw_line(raw_line)
        return io_bind(logged, after_log)

    return fold_io_lazy(response, advance, Ok(StreamState()))


def collect_stream(
    ctx: Context, payload: Mapping[str, Any], on_progress: ProgressCallback
) -> IO[Result[StreamState, TranslationError]]:
    def on_raw_line(raw_line: bytes) -> IO[None]:
        return run_log_write(ctx.log, raw_line.decode("utf-8", "replace"))

    def respond(
        opened: Result[Any, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        if isinstance(opened, Err):
            return io_result(opened)

        def thunk() -> Result[StreamState, TranslationError]:
            try:
                with opened.value as response:
                    outcome = drive_stream(response, on_progress, on_raw_line).run()
                    if isinstance(outcome, Err):
                        log_error(ctx.log, describe(outcome.error)).run()
                    else:
                        run_log_write(ctx.log, "\n").run()

                    return outcome
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                failure = fail_http("interrupted", str(error))
                log_error(ctx.log, describe(failure.error)).run()
                return failure

        return IO(thunk)

    return io_bind(http_open_stream(ctx, payload), respond)


def log_all(
    console: Console, messages: tuple[str, ...]
) -> IO[Result[tuple[()], TranslationError]]:
    def step(_: tuple[()], message: str) -> IO[Result[tuple[()], TranslationError]]:
        return io_map(console.log(message), lambda _: Ok(()))

    return fold_io(messages, step, Ok(()))


def apply_stream_override(ctx: Context, payload: dict[str, Any]) -> dict[str, Any]:
    if ctx.stream is None:
        return payload

    return {**payload, "stream": ctx.stream}


def chat(
    ctx: Context,
    system: str,
    user: str,
    model: str,
    params: Mapping[str, Any],
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > ctx.config.max_tokens:
        return io_result(fail_budget(estimated, ctx.config.max_tokens))

    payload = apply_stream_override(
        ctx,
        build_chat_payload(model, system, user, params),
    )
    call = with_retries(
        ctx,
        lambda: streamed_call(ctx, payload)
        if payload.get("stream", True)
        else plain_call(ctx, payload),
    )

    def stopped(
        reply_result: Result[ChatReply, TranslationError],
    ) -> IO[Result[Translated, TranslationError]]:
        return io_and_then(
            ctx.console.stop(),
            conclude_chat(ctx, reply_result, usage, estimated),
        )

    return io_and_then(ctx.console.start("Working"), io_bind(call, stopped))


def streamed_call(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[ChatReply, TranslationError]]:
    full_payload = {**payload, "stream": True}
    if "stream_options" not in full_payload:
        full_payload = {**full_payload, "stream_options": {"include_usage": True}}

    def on_progress(label: str, count: int) -> IO[None]:
        return ctx.console.progress(label, count)

    def to_reply(state: StreamState) -> ChatReply:
        return ChatReply(
            content="".join(state.contents)
            + (
                state.think.held
                if state.think.checking and not state.think.open
                else ""
            ),
            reasoning=state.reasoning,
            reported=state.reported,
            counted=state.counted,
        )

    return io_map(
        collect_stream(ctx, full_payload, on_progress),
        lambda outcome: result_map(outcome, to_reply),
    )


def plain_call(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[ChatReply, TranslationError]]:
    return io_bind(
        http_post_json(ctx, payload),
        lambda body_result: io_result(result_bind(body_result, plain_reply)),
    )


def conclude_chat(
    ctx: Context,
    reply_result: Result[ChatReply, TranslationError],
    usage: Usage,
    estimated: int,
) -> IO[Result[Translated, TranslationError]]:
    if isinstance(reply_result, Err):
        return io_result(reply_result)

    reply = reply_result.value
    content, think_text = strip_think_tag(reply.content)
    if not isinstance(content, str):
        return io_result(
            fail_http(
                "protocol", "unexpected content type: %s" % type(content).__name__
            )
        )

    think_texts: tuple[str, ...] = maybe_either(
        think_text, lambda text: (text,), lambda: ()
    )
    messages = (
        tuple(
            text.rstrip()
            for text in reply.reasoning + think_texts
            if text.strip()
        )
        if ctx.verbose
        else ()
    )

    def finish(
        _: Result[tuple[()], TranslationError],
    ) -> Result[Translated, TranslationError]:
        reported = reply.reported
        return Ok(
            Translated(
                content,
                add_usage(usage, reported)
                if reported
                else add_usage(
                    usage,
                    {
                        "prompt_tokens": estimated,
                        "completion_tokens": reply.counted,
                    },
                ),
            )
        )

    return io_map(log_all(ctx.console, messages), finish)

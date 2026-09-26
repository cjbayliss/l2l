from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any, TypedDict

from l2l import __version__
from l2l.console import Console
from l2l.effects import log_entry, log_error, log_request, run_log_write
from l2l.errors import (
    HttpError,
    TranslationError,
    describe,
    fail_budget,
    fail_http,
)
from l2l.monads import (
    IO,
    NOTHING,
    Cons,
    Err,
    Just,
    Maybe,
    Nothing,
    Ok,
    Result,
    cons,
    cons_all,
    cons_to_tuple,
    fold_io,
    fold_io_lazy,
    io_and_then,
    io_bind,
    io_catch_result,
    io_map,
    io_pure,
    io_result,
    io_using,
    maybe_either,
    maybe_map,
    maybe_or,
    maybe_to_optional,
    read_ref,
    result_bind,
    result_map,
)
from l2l.plans import plan_backoff, retry_delay, transient
from l2l.settings import Config, Context
from l2l.text import (
    SseState,
    ThinkState,
    Translated,
    Usage,
    add_usage,
    estimate_tokens,
    parse_float,
    sse_step,
    strip_think_tag,
    think_step,
)

ProgressCallback = Callable[[str, int], IO[None]]
RawLineLogger = Callable[[bytes], IO[None]]
ReasoningLogger = Callable[[str], IO[None]]


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


class ChatMessage(TypedDict, total=False):
    role: str
    content: object
    reasoning_content: object
    reasoning: object


class ChatChoice(TypedDict, total=False):
    message: ChatMessage
    delta: ChatMessage


class UsageReport(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    cost: float


class ChatCompletion(TypedDict, total=False):
    choices: list[ChatChoice]
    usage: UsageReport


class StreamChunk(ChatCompletion, total=False):
    error: dict[str, Any]


TRANSPORT_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    OSError,
    http.client.HTTPException,
)


def extract_message(
    body: Any,
) -> Result[tuple[Mapping[str, Any], Any], TranslationError]:
    try:
        message = body["choices"][0]["message"]
        return Ok((message, message["content"]))
    except KeyError, IndexError, TypeError:
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


REASONING_KEYS = ("reasoning_content", "reasoning")


def reasoning_at(
    container: Mapping[str, Any], clean: Callable[[str], Maybe[str]]
) -> Maybe[str]:
    def from_key(key: str) -> Maybe[str]:
        value = container.get(key)
        return clean(value) if isinstance(value, str) else NOTHING

    initial: Maybe[str] = NOTHING
    return reduce(maybe_or, (from_key(key) for key in REASONING_KEYS), initial)


def keep_raw(value: str) -> Maybe[str]:
    return Just(value) if value else NOTHING


def keep_trimmed(value: str) -> Maybe[str]:
    return Just(value.rstrip()) if value.strip() else NOTHING


def message_reasoning_texts(message: Mapping[str, Any]) -> tuple[str, ...]:
    found = reasoning_at(message, keep_trimmed)
    return maybe_either(found, lambda text: (text,), tuple)


def parse_chunk_delta(chunk: Mapping[str, Any]) -> dict[str, Any]:
    try:
        choices = chunk.get("choices")
        if not choices:
            return {}

        return choices[0].get("delta") or {}
    except AttributeError, IndexError, KeyError, TypeError:
        return {}


def delta_text(delta: Mapping[str, Any]) -> str:
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))

    return ""


@dataclass(frozen=True)
class ProgressUpdate:
    label: str
    count: int
    reasoning: str = ""


@dataclass(frozen=True)
class StreamState:
    reasoning: Cons[str] | None = None
    contents: Cons[str] | None = None
    reported_usage: Mapping[str, Any] | None = None
    chunk_count: int = 0
    content_started: bool = False
    think: ThinkState = ThinkState()
    sse: SseState = SseState()
    progress_update: ProgressUpdate | None = None


def stream_text(state: StreamState) -> str:
    return "".join(cons_to_tuple(state.contents)) + (
        state.think.held if state.think.checking and not state.think.open else ""
    )


def stream_step(
    state: StreamState, chunk: Mapping[str, Any]
) -> Result[StreamState, TranslationError]:
    usage_report = chunk.get("usage")
    reported_usage = (
        MappingProxyType(usage_report)
        if isinstance(usage_report, dict)
        else state.reported_usage
    )
    delta = parse_chunk_delta(chunk)
    reasoning_found = reasoning_at(delta, keep_raw)
    text = delta_text(delta)
    if isinstance(reasoning_found, Nothing) and not text:
        return Ok(replace(state, reported_usage=reported_usage, progress_update=None))

    think_state, thinking, visible = (
        think_step(state.think, text) if text else (state.think, False, "")
    )
    reasoning_texts: tuple[str, ...] = maybe_either(
        reasoning_found, lambda value: (value,), lambda: ()
    )
    thinking_progress = 1 if reasoning_texts and not state.content_started else 0
    text_progress = 1 if text else 0
    progress_update: ProgressUpdate | None = None
    if thinking_progress or text_progress:
        label = ("Thinking" if thinking else "Working") if text else "Thinking"
        progress_update = ProgressUpdate(
            label,
            thinking_progress + text_progress,
            "".join(reasoning_texts),
        )

    return Ok(
        StreamState(
            reasoning=cons_all(reasoning_texts, state.reasoning),
            contents=cons(visible, state.contents) if visible else state.contents,
            reported_usage=reported_usage,
            chunk_count=state.chunk_count + thinking_progress + text_progress,
            content_started=state.content_started or bool(text),
            think=think_state,
            progress_update=progress_update,
        )
    )


def ingest_raw_line(
    state: StreamState, raw_line: bytes
) -> Result[StreamState, TranslationError]:
    sse, chunk = sse_step(state.sse, raw_line)
    updated = state if sse == state.sse else replace(state, sse=sse)
    if chunk is None:
        return Ok(replace(updated, progress_update=None))

    if isinstance(chunk.get("error"), dict):
        return fail_http("stream", str(chunk["error"])[:500])

    return stream_step(updated, chunk)


@dataclass(frozen=True)
class ChatReply:
    content: str
    reasoning: tuple[str, ...] = ()
    reported_usage: Mapping[str, Any] | None = None
    chunk_count: int = 0
    reasoning_shown: bool = False


def plain_reply(body: Any) -> Result[ChatReply, TranslationError]:
    message_result = extract_message(body)
    if isinstance(message_result, Err):
        return message_result

    message, content = message_result.value
    texts, thoughts = flatten_content_parts(content)
    if not isinstance(texts, str):
        return fail_http(
            "protocol", "unexpected content type: %s" % type(texts).__name__
        )

    usage_report = body.get("usage")
    reported_usage = (
        MappingProxyType(usage_report) if isinstance(usage_report, dict) else None
    )
    return Ok(
        ChatReply(
            content=texts,
            reasoning=message_reasoning_texts(message) + thoughts,
            reported_usage=reported_usage,
            chunk_count=0,
        )
    )


def http_request(
    config: Config, payload: Mapping[str, Any], accept: str | None = None
) -> urllib.request.Request:
    headers = {
        **{
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.api_key,
            "User-Agent": "l2l/" + __version__,
        },
        **({"Accept": accept} if accept else {}),
    }

    return urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", "replace")[:500]
    except TRANSPORT_ERRORS:
        return str(error)


def urllib_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
    try:
        return Ok(urllib.request.urlopen(request, timeout=timeout))
    except urllib.error.HTTPError as error:
        return fail_http(
            "status",
            http_error_detail(error),
            error.code,
            maybe_to_optional(retry_after_seconds(error.headers)),
        )
    except TRANSPORT_ERRORS as error:
        return fail_http("unreachable", str(error))


def retry_after_seconds(headers: Any) -> Maybe[float]:
    raw = headers.get("Retry-After") if headers is not None else None
    if raw is None:
        return NOTHING

    return maybe_map(parse_float(raw), lambda seconds: max(seconds, 0.0))


def verbose_retry_log(
    ctx: Context, retry_number: int, retries: int, wait: float
) -> IO[None]:
    return ctx.console.log_verbose(
        "l2l: transient failure; retry %d/%d in %.1fs" % (retry_number, retries, wait)
    )


def with_retries[T](
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
    def opened() -> Result[Any, TranslationError]:
        return ctx.open_http(http_request(ctx.config, payload), ctx.config.timeout)

    def read(
        outcome: Result[Any, TranslationError],
    ) -> Result[tuple[str, Any], TranslationError]:
        if isinstance(outcome, Err):
            return outcome

        with outcome.value as response:
            try:
                body = response.read().decode("utf-8")
                return Ok((body, json.loads(body)))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                return fail_http("protocol", "invalid JSON response: %s" % error)
            except TRANSPORT_ERRORS as error:
                return fail_http("unreachable", "response read failed: %s" % error)

    def attempt() -> IO[Result[tuple[str, Any], TranslationError]]:
        def thunk() -> Result[tuple[str, Any], TranslationError]:
            return read(opened())

        return IO(thunk)

    def record(
        outcome: Result[tuple[str, Any], TranslationError],
    ) -> IO[Result[dict[str, Any], TranslationError]]:
        if isinstance(outcome, Err):
            return io_map(
                log_error(ctx.log, describe(outcome.error)),
                lambda _: Err(outcome.error),
            )

        body, loaded = outcome.value
        return io_map(log_entry(ctx.log, "RESPONSE", body), lambda _: Ok(loaded))

    return io_bind(
        io_and_then(log_request(ctx.log, payload), attempt()),
        record,
    )


def http_open_stream(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[Any, TranslationError]]:
    def open_stream(
        body: Mapping[str, Any], label: str
    ) -> IO[Result[Any, TranslationError]]:
        return io_and_then(
            log_request(ctx.log, body, label),
            IO(
                lambda: ctx.open_http(
                    http_request(ctx.config, body, accept="text/event-stream"),
                    ctx.config.timeout,
                )
            ),
        )

    def logged(
        opened: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        if isinstance(opened, Ok):
            return io_result(opened)

        return io_map(
            log_error(ctx.log, describe(opened.error)),
            lambda _: opened,
        )

    def decide(
        opened: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        if (
            isinstance(opened, Ok)
            or "stream_options" not in payload
            or not (
                isinstance(opened.error, HttpError)
                and "stream_options" in opened.error.detail
            )
        ):
            return io_result(opened)

        return io_bind(
            open_stream(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "stream_options"
                },
                "REQUEST (stream, retry)",
            ),
            logged,
        )

    return io_bind(io_bind(open_stream(payload, "REQUEST (stream)"), logged), decide)


def drive_stream(
    response: Any,
    on_progress: ProgressCallback,
    on_raw_line: RawLineLogger | None = None,
    on_reasoning: ReasoningLogger | None = None,
) -> IO[Result[StreamState, TranslationError]]:
    def advance(
        state: StreamState, raw_line: bytes
    ) -> IO[Result[StreamState, TranslationError]]:
        def notify(
            outcome: Result[StreamState, TranslationError],
        ) -> IO[Result[StreamState, TranslationError]]:
            if isinstance(outcome, Err):
                return io_result(outcome)

            request = outcome.value.progress_update
            if request is None:
                return io_result(outcome)

            def after_reasoning(
                _: None,
            ) -> IO[Result[StreamState, TranslationError]]:
                return io_map(
                    on_progress(request.label, request.count), lambda _: outcome
                )

            if on_reasoning is None or not request.reasoning:
                return after_reasoning(None)

            return io_bind(on_reasoning(request.reasoning), after_reasoning)

        def after_log(_: None) -> IO[Result[StreamState, TranslationError]]:
            return io_bind(io_result(ingest_raw_line(state, raw_line)), notify)

        logged = io_pure(None) if on_raw_line is None else on_raw_line(raw_line)
        return io_bind(logged, after_log)

    def flush(
        outcome: Result[StreamState, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        if isinstance(outcome, Err):
            return io_result(outcome)

        return io_result(ingest_raw_line(outcome.value, b""))

    return io_bind(fold_io_lazy(response, advance, Ok(StreamState())), flush)


def collect_stream(
    ctx: Context, payload: Mapping[str, Any], on_progress: ProgressCallback
) -> IO[Result[StreamState, TranslationError]]:
    def on_reasoning(text: str) -> IO[None]:
        return ctx.console.stream_reasoning(text)

    def note_progress(label: str, count: int) -> IO[None]:
        def continued(_: None) -> IO[None]:
            return ctx.console.progress(label, count)

        return (
            io_and_then(ctx.console.end_raw(), continued(None))
            if label == "Working"
            else ctx.console.progress(label, count)
        )

    def on_raw_line(raw_line: bytes) -> IO[None]:
        return run_log_write(ctx.log, raw_line.decode("utf-8", "replace"))

    def conclude(
        outcome: Result[StreamState, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        def recorded(_: None) -> IO[Result[StreamState, TranslationError]]:
            if isinstance(outcome, Err):
                return io_map(
                    log_error(ctx.log, describe(outcome.error)),
                    lambda _: outcome,
                )

            return io_map(run_log_write(ctx.log, "\n"), lambda _: outcome)

        return io_and_then(ctx.console.end_raw(), recorded(None))

    def handle(error: Exception) -> Result[StreamState, TranslationError]:
        if isinstance(error, TRANSPORT_ERRORS):
            return fail_http("interrupted", str(error))

        return fail_http("protocol", "%s: %s" % (type(error).__name__, error))

    def respond(
        opened: Result[Any, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        if isinstance(opened, Err):
            return io_result(opened)

        return io_catch_result(
            io_using(
                opened.value,
                lambda response: io_bind(
                    drive_stream(response, note_progress, on_raw_line, on_reasoning),
                    conclude,
                ),
            ),
            handle,
        )

    return io_bind(http_open_stream(ctx, payload), respond)


def log_all(
    console: Console, messages: tuple[str, ...]
) -> IO[Result[tuple[()], TranslationError]]:
    def step(_: tuple[()], message: str) -> IO[Result[tuple[()], TranslationError]]:
        return io_map(console.log_verbose(message), lambda _: Ok(()))

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
        lambda: (
            streamed_call(ctx, payload)
            if payload.get("stream", True)
            else plain_call(ctx, payload)
        ),
    )

    def stopped(
        reply_result: Result[ChatReply, TranslationError],
    ) -> IO[Result[Translated, TranslationError]]:
        return io_and_then(
            ctx.console.stop(),
            conclude_chat(ctx, reply_result, usage, estimated),
        )

    return io_and_then(ctx.console.start("Working"), io_bind(call, stopped))


def to_chat_reply(state: StreamState) -> ChatReply:
    joined = "".join(cons_to_tuple(state.reasoning))
    return ChatReply(
        content=stream_text(state),
        reasoning=(joined,) if joined.strip() else (),
        reported_usage=state.reported_usage,
        chunk_count=state.chunk_count,
        reasoning_shown=True,
    )


def streamed_call(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[ChatReply, TranslationError]]:
    full_payload = {
        **payload,
        "stream": True,
        **(
            {}
            if "stream_options" in payload
            else {"stream_options": {"include_usage": True}}
        ),
    }

    def on_progress(label: str, count: int) -> IO[None]:
        return ctx.console.progress(label, count)

    return io_map(
        collect_stream(ctx, full_payload, on_progress),
        lambda outcome: result_map(outcome, lambda state: to_chat_reply(state)),
    )


def plain_call(
    ctx: Context,
    payload: Mapping[str, Any],
) -> IO[Result[ChatReply, TranslationError]]:
    return io_bind(
        http_post_json(ctx, payload),
        lambda body_result: io_result(result_bind(body_result, plain_reply)),
    )


type RepairOutcome = tuple[str, Usage, bool]


def repaired_call(
    ctx: Context,
    model: str,
    params: Mapping[str, Any],
    instruction: str,
    initial_user: str,
    validate: Callable[[str], Maybe[str]],
    build_retry_user: Callable[[str, str], str],
    max_repairs: int,
    on_repair: Callable[[int, str], IO[None]],
    on_exhausted: Callable[[str, str], IO[None]] | None,
) -> Callable[[Usage], IO[Result[RepairOutcome, TranslationError]]]:
    def attempt(
        user: str, failed: int, usage: Usage
    ) -> IO[Result[RepairOutcome, TranslationError]]:
        def assessed(
            reply_result: Result[Translated, TranslationError],
        ) -> IO[Result[RepairOutcome, TranslationError]]:
            if isinstance(reply_result, Err):
                return io_result(reply_result)

            reply = reply_result.value
            problem = validate(reply.text)
            if isinstance(problem, Nothing):
                return io_result(Ok((reply.text, reply.usage, True)))

            if failed > max_repairs:
                announce = (
                    on_exhausted(reply.text, problem.value)
                    if on_exhausted is not None
                    else io_pure(None)
                )
                return io_map(announce, lambda _: Ok((reply.text, reply.usage, False)))

            def continued(_: None) -> IO[Result[RepairOutcome, TranslationError]]:
                return attempt(
                    build_retry_user(reply.text, problem.value), failed + 1, reply.usage
                )

            return io_bind(on_repair(failed, problem.value), continued)

        return io_bind(chat(ctx, instruction, user, model, params, usage), assessed)

    return lambda usage: attempt(initial_user, 1, usage)


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
    think_texts: tuple[str, ...] = maybe_either(
        think_text, lambda text: (text,), lambda: ()
    )

    def conclude(verbose: bool) -> IO[Result[Translated, TranslationError]]:
        messages = (
            tuple(
                text.rstrip() for text in reply.reasoning + think_texts if text.strip()
            )
            if verbose and not reply.reasoning_shown
            else ()
        )

        def finish(
            _: Result[tuple[()], TranslationError],
        ) -> Result[Translated, TranslationError]:
            reported_usage = reply.reported_usage
            return Ok(
                Translated(
                    content,
                    (
                        add_usage(usage, reported_usage)
                        if reported_usage
                        else add_usage(
                            usage,
                            {
                                "prompt_tokens": estimated,
                                "completion_tokens": reply.chunk_count,
                            },
                        )
                    ),
                )
            )

        return io_map(log_all(ctx.console, messages), finish)

    return io_bind(read_ref(ctx.console.verbose), conclude)

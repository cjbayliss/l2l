from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any

from zh2en.config import Config, Context
from zh2en.console import Console
from zh2en.effects import log_entry, log_error, log_request, run_log_write
from zh2en.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_bind,
    io_map,
    io_pure,
    io_result,
    result_bind,
    result_map,
)
from zh2en.text import (
    ChatOutcome,
    ThinkState,
    Usage,
    add_usage,
    estimate_tokens,
    strip_think_tag,
    think_step,
)

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


def extract_message(body: Any) -> Result[tuple[Mapping[str, Any], Any], str]:
    try:
        message = body["choices"][0]["message"]
        return Ok((message, message["content"]))
    except (KeyError, IndexError, TypeError):
        return Err("unexpected response shape: %s" % str(body)[:500])


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
    def reasoning_at(key: str) -> tuple[str, ...]:
        value = message.get(key)
        return (value.rstrip(),) if isinstance(value, str) and value.strip() else ()

    return reduce(
        lambda found, key: found or reasoning_at(key),
        ("reasoning_content", "reasoning"),
        (),
    )


def parse_stream_line(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return None

    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None

    try:
        loaded = json.loads(data)
    except json.JSONDecodeError:
        return None

    return loaded if isinstance(loaded, dict) else None


def parse_chunk_delta(chunk: Mapping[str, Any]) -> dict[str, Any]:
    try:
        choices = chunk.get("choices")
        if not choices:
            return {}

        return choices[0].get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return {}


def delta_reasoning_text(delta: Mapping[str, Any]) -> str | None:
    def reasoning_at(key: str) -> str | None:
        value = delta.get(key)
        return value if isinstance(value, str) and value else None

    def step(found: str | None, key: str) -> str | None:
        return found if found is not None else reasoning_at(key)

    return reduce(step, ("reasoning_content", "reasoning"), None)


def delta_text(delta: Mapping[str, Any]) -> str:
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))

    return ""


@dataclass(frozen=True)
class StreamState:
    reasoning: tuple[str, ...] = ()
    contents: tuple[str, ...] = ()
    reported: Mapping[str, Any] | None = None
    counted: int = 0
    content_started: bool = False
    think: ThinkState = ThinkState()
    progress_request: tuple[str, int] | None = None


def stream_step(
    state: StreamState, chunk: Mapping[str, Any]
) -> Result[StreamState, str]:
    usage_report = chunk.get("usage")
    reported = (
        MappingProxyType(usage_report)
        if isinstance(usage_report, dict)
        else state.reported
    )
    delta = parse_chunk_delta(chunk)
    reasoning_text = delta_reasoning_text(delta)
    text = delta_text(delta)
    if not reasoning_text and not text:
        return Ok(replace(state, reported=reported, progress_request=None))

    think_state, thinking, visible = (
        think_step(state.think, text) if text else (state.think, False, "")
    )
    thinking_progress = 1 if reasoning_text and not state.content_started else 0
    text_progress = 1 if text else 0
    progress_request = None
    if thinking_progress or text_progress:
        progress_request = (
            ("Thinking" if thinking else "Working") if text else "Thinking",
            thinking_progress + text_progress,
        )

    return Ok(
        StreamState(
            reasoning=(
                state.reasoning + (reasoning_text,)
                if reasoning_text
                else state.reasoning
            ),
            contents=state.contents + (visible,) if visible else state.contents,
            reported=reported,
            counted=state.counted + thinking_progress + text_progress,
            content_started=state.content_started or bool(text),
            think=think_state,
            progress_request=progress_request,
        )
    )


def step_stream(state: StreamState, raw_line: bytes) -> Result[StreamState, str]:
    chunk = parse_stream_line(raw_line)
    if chunk is None:
        return Ok(state)

    if isinstance(chunk.get("error"), dict):
        return Err("endpoint stream error: %s" % str(chunk["error"])[:500])

    return stream_step(state, chunk)


@dataclass(frozen=True)
class ChatReply:
    content: Any
    reasoning: tuple[str, ...] = ()
    reported: Mapping[str, Any] | None = None
    counted: int = 0


def plain_reply(body: Any) -> Result[ChatReply, str]:
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
    }
    if accept:
        headers = {**headers, "Accept": accept}

    return urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def urllib_open(request: Any, timeout: float) -> Result[Any, str]:
    try:
        return Ok(urllib.request.urlopen(request, timeout=timeout))
    except urllib.error.HTTPError as error:
        return Err(
            "HTTP %s from endpoint: %s"
            % (error.code, error.read().decode("utf-8", "replace")[:500])
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return Err("could not reach endpoint: %s" % error)


def http_post_json(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[dict[str, Any], str]]:
    def thunk() -> Result[dict[str, Any], str]:
        log_request(ctx.log, payload).run()
        opened = ctx.open_http(http_request(ctx.config, payload), ctx.config.timeout)
        if isinstance(opened, Err):
            log_error(ctx.log, opened.error).run()
            return opened

        with opened.value as response:
            try:
                body = response.read().decode("utf-8")
                log_entry(ctx.log, "RESPONSE", body).run()
                return Ok(json.loads(body))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                log_error(ctx.log, "invalid JSON response: %s" % error).run()
                return Err("invalid JSON response: %s" % error)

    return IO(thunk)


def http_open_stream(ctx: Context, payload: Mapping[str, Any]) -> IO[Result[Any, str]]:
    def open_stream(body: Mapping[str, Any], label: str) -> Result[Any, str]:
        log_request(ctx.log, body, label).run()
        opened = ctx.open_http(
            http_request(ctx.config, body, accept="text/event-stream"),
            ctx.config.timeout,
        )
        if isinstance(opened, Err):
            log_error(ctx.log, opened.error).run()

        return opened

    def thunk() -> Result[Any, str]:
        opened = open_stream(payload, "REQUEST (stream)")
        if (
            isinstance(opened, Ok)
            or "stream_options" not in payload
            or "stream_options" not in str(opened.error)
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
) -> IO[Result[StreamState, str]]:
    def advance(state: StreamState, raw_line: bytes) -> IO[Result[StreamState, str]]:
        def notify(outcome: Result[StreamState, str]) -> IO[Result[StreamState, str]]:
            if isinstance(outcome, Err):
                return io_result(outcome)

            request = outcome.value.progress_request
            if request is None:
                return io_result(outcome)

            label, count = request
            return io_map(on_progress(label, count), lambda _: outcome)

        def after_log(_: None) -> IO[Result[StreamState, str]]:
            return io_bind(io_result(step_stream(state, raw_line)), notify)

        logged = io_pure(None) if on_raw_line is None else on_raw_line(raw_line)
        return io_bind(logged, after_log)

    def thunk() -> Result[StreamState, str]:
        outcome: Result[StreamState, str] = Ok(StreamState())
        lines = iter(response)
        while not isinstance(outcome, Err):
            try:
                raw_line = next(lines)
            except StopIteration:
                return outcome

            outcome = advance(outcome.value, raw_line).run()

        return outcome

    return IO(thunk)


def collect_stream(
    ctx: Context, payload: Mapping[str, Any], on_progress: ProgressCallback
) -> IO[Result[StreamState, str]]:
    def on_raw_line(raw_line: bytes) -> IO[None]:
        return run_log_write(ctx.log, raw_line.decode("utf-8", "replace"))

    def respond(opened: Result[Any, str]) -> IO[Result[StreamState, str]]:
        if isinstance(opened, Err):
            return io_result(opened)

        def thunk() -> Result[StreamState, str]:
            try:
                with opened.value as response:
                    outcome = drive_stream(response, on_progress, on_raw_line).run()
                    if isinstance(outcome, Err):
                        log_error(ctx.log, outcome.error).run()
                    else:
                        run_log_write(ctx.log, "\n").run()

                    return outcome
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                log_error(ctx.log, "stream interrupted: %s" % error).run()
                return Err("stream interrupted: %s" % error)

        return IO(thunk)

    return io_bind(http_open_stream(ctx, payload), respond)


def log_all(console: Console, messages: tuple[str, ...]) -> IO[Result[tuple[()], str]]:
    def step(_: tuple[()], message: str) -> IO[Result[tuple[()], str]]:
        return io_map(console.log(message), lambda _: Ok(()))

    return fold_io(messages, step, Ok(()))


def chat(
    ctx: Context,
    system: str,
    user: str,
    model: str,
    params: Mapping[str, Any],
    usage: Usage,
) -> IO[Result[ChatOutcome, str]]:
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > ctx.config.max_tokens:
        return io_result(
            Err(
                "request is ~%d tokens, over the %d-token budget; raise "
                "api.max_tokens (--max-tokens / TRANSLATE_MAX_TOKENS) or "
                "shorten the input" % (estimated, ctx.config.max_tokens)
            )
        )

    payload = build_chat_payload(model, system, user, params)

    def stopped(
        reply_result: Result[ChatReply, str],
    ) -> IO[Result[ChatOutcome, str]]:
        return io_bind(
            ctx.console.stop(),
            lambda _: conclude_chat(ctx, reply_result, usage, estimated),
        )

    return io_bind(
        ctx.console.start("Working"),
        lambda _: io_bind(
            (
                streamed_call(ctx, payload)
                if payload.get("stream", True)
                else plain_call(ctx, payload)
            ),
            stopped,
        ),
    )


def streamed_call(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[ChatReply, str]]:
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


def plain_call(ctx: Context, payload: Mapping[str, Any]) -> IO[Result[ChatReply, str]]:
    return io_bind(
        http_post_json(ctx, payload),
        lambda body_result: io_result(result_bind(body_result, plain_reply)),
    )


def conclude_chat(
    ctx: Context, reply_result: Result[ChatReply, str], usage: Usage, estimated: int
) -> IO[Result[ChatOutcome, str]]:
    if isinstance(reply_result, Err):
        return io_result(reply_result)

    reply = reply_result.value
    content, think_text = strip_think_tag(reply.content)
    if not isinstance(content, str):
        return io_result(Err("unexpected content type: %s" % type(content).__name__))

    messages = (
        tuple(
            text.rstrip()
            for text in reply.reasoning + ((think_text,) if think_text else ())
            if text.strip()
        )
        if ctx.verbose
        else ()
    )

    def finish(_: Result[tuple[()], str]) -> Result[ChatOutcome, str]:
        reported = reply.reported
        return Ok(
            (
                content,
                (
                    add_usage(usage, reported)
                    if reported
                    else add_usage(
                        usage,
                        {
                            "prompt_tokens": estimated,
                            "completion_tokens": reply.counted,
                        },
                    )
                ),
            )
        )

    if messages:
        return io_map(log_all(ctx.console, messages), finish)

    return io_result(finish(Ok(())))

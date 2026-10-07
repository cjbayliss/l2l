from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any

import pycurl

from l2l import __version__
from l2l.console import Console
from l2l.effects import log_entry, log_error, log_request, run_log_write
from l2l.errors import (
    HttpError,
    HttpKind,
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
    Reference,
    Result,
    cons,
    cons_all,
    cons_to_tuple,
    fold_io,
    fold_io_push,
    io_and_then,
    io_bind,
    io_catch_result,
    io_map,
    io_pair,
    io_pure,
    io_result,
    io_result_bind,
    line_push,
    maybe_either,
    maybe_from_optional,
    maybe_map,
    maybe_or,
    maybe_to_optional,
    read_reference,
    reference_collector,
    reference_gate,
    reference_write_when,
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
    model: str, system: str, user: str, parameters: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **{
            "model": model,
            "messages": (
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
        },
        **{key: value for key, value in parameters.items() if value is not None},
    }


TRANSPORT_ERRORS = (OSError, TimeoutError)

CONNECT_ERROR_CODES = frozenset(
    {1, 3, 5, 6, 7, 35, 51, 58, 60, 61, 64, 66, 77, 83, 90, 96, 200}
)


@dataclass(frozen=True)
class HttpRequest:
    url: str
    body: bytes
    headers: Mapping[str, str]


def curl_error_code(error: Any) -> int:
    return int(error.args[0]) if error.args else 0


def curl_error_detail(error: Any) -> str:
    return str(error.args[1]) if len(error.args) > 1 else str(error)


def curl_failure(error: Any, kind: HttpKind) -> Err[TranslationError]:
    resolved = "unreachable" if curl_error_code(error) in CONNECT_ERROR_CODES else kind
    return fail_http(resolved, curl_error_detail(error))


def proxy_for(url: str, environment: Mapping[str, str]) -> Maybe[str]:
    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    candidates = (scheme + "_proxy", "all_proxy", "ALL_PROXY")

    def lookup(name: str) -> Maybe[str]:
        return maybe_from_optional(environment.get(name) or None)

    initial: Maybe[str] = NOTHING
    return reduce(maybe_or, (lookup(name) for name in candidates), initial)


def noproxy_for(environment: Mapping[str, str]) -> Maybe[str]:
    def lookup(name: str) -> Maybe[str]:
        return maybe_from_optional(environment.get(name) or None)

    initial: Maybe[str] = NOTHING
    return reduce(
        maybe_or, (lookup(name) for name in ("no_proxy", "NO_PROXY")), initial
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
) -> HttpRequest:
    headers = MappingProxyType(
        {
            **{
                "Content-Type": "application/json",
                "Authorization": "Bearer " + config.api_key,
                "User-Agent": "l2l/" + __version__,
            },
            **({"Accept": accept} if accept else {}),
        }
    )

    return HttpRequest(
        url=config.base_url + "/chat/completions",
        body=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )


def retry_after_seconds(raw: str | None) -> Maybe[float]:
    if raw is None:
        return NOTHING

    return maybe_map(parse_float(raw), lambda seconds: max(seconds, 0.0))


def retry_after_header(lines: tuple[bytes, ...]) -> Maybe[float]:
    def matching(line: bytes) -> bool:
        return line.lower().startswith(b"retry-after:")

    found = next((line for line in lines if matching(line)), None)
    if found is None:
        return NOTHING

    raw = found.split(b":", 1)[1].decode("ascii", "replace").strip()
    return retry_after_seconds(raw or None)


def is_status_line(line: bytes) -> bool:
    return line.startswith(b"HTTP/")


def is_error_status_line(line: bytes) -> bool:
    parts = line.split(None, 2)
    if not is_status_line(line) or len(parts) < 2:
        return False

    try:
        return int(parts[1]) >= 400
    except ValueError:
        return False


def append_bytes(collected: tuple[bytes, ...], item: bytes) -> tuple[bytes, ...]:
    return collected + (item,)


def configure_curl(
    request: HttpRequest,
    timeout: float,
    environment: Mapping[str, str],
    curl: Any,
    on_chunk: Callable[[bytes], None],
    on_header: Callable[[bytes], None],
) -> None:
    curl.setopt(pycurl.URL, request.url)
    curl.setopt(pycurl.POST, 1)
    curl.setopt(pycurl.POSTFIELDS, request.body)
    curl.setopt(
        pycurl.HTTPHEADER,
        [name + ": " + value for name, value in request.headers.items()],
    )
    curl.setopt(pycurl.CONNECTTIMEOUT_MS, int(timeout * 1000))
    curl.setopt(pycurl.LOW_SPEED_LIMIT, 1)
    curl.setopt(pycurl.LOW_SPEED_TIME, max(int(timeout), 1))
    curl.setopt(pycurl.NOSIGNAL, 1)
    curl.setopt(pycurl.FOLLOWLOCATION, 1)
    curl.setopt(pycurl.MAXREDIRS, 10)
    curl.setopt(pycurl.WRITEFUNCTION, on_chunk)
    curl.setopt(pycurl.HEADERFUNCTION, on_header)

    proxy = proxy_for(request.url, environment)
    if isinstance(proxy, Just):
        curl.setopt(pycurl.PROXY, proxy.value)

    noproxy = noproxy_for(environment)
    if isinstance(noproxy, Just):
        curl.setopt(pycurl.NOPROXY, noproxy.value)


@dataclass(frozen=True)
class ResponseTaps:
    chunks: Reference[tuple[bytes, ...]]
    header_lines: Reference[tuple[bytes, ...]]
    on_header: Callable[[bytes], None]


def response_taps() -> ResponseTaps:
    chunks: Reference[tuple[bytes, ...]] = Reference(())
    header_lines: Reference[tuple[bytes, ...]] = Reference(())
    reset_body = reference_write_when(is_status_line, chunks, ())
    reset_headers = reference_write_when(is_status_line, header_lines, ())
    collect_header = reference_collector(append_bytes, header_lines)

    def on_header(line: bytes) -> None:
        reset_body(line)
        reset_headers(line)
        collect_header(line)

    return ResponseTaps(chunks=chunks, header_lines=header_lines, on_header=on_header)


def curl_transfer(curl: Any, kind: HttpKind) -> IO[Result[int, TranslationError]]:
    def thunk() -> Result[int, TranslationError]:
        try:
            curl.perform()
            return Ok(int(curl.getinfo(pycurl.RESPONSE_CODE)))
        except pycurl.error as error:
            return curl_failure(error, kind)
        finally:
            curl.close()

    return IO(thunk)


def status_failure(
    code: int, collected: tuple[bytes, ...], headers: tuple[bytes, ...]
) -> Err[TranslationError]:
    return fail_http(
        "status",
        b"".join(collected).decode("utf-8", "replace")[:500],
        code,
        maybe_to_optional(retry_after_header(headers)),
    )


@dataclass(frozen=True)
class CurlResponse:
    request: HttpRequest
    timeout: float
    environment: Mapping[str, str]
    new_curl: Callable[[], Any] = pycurl.Curl

    def body(self) -> IO[Result[bytes, TranslationError]]:
        taps = response_taps()
        curl = self.new_curl()
        configure_curl(
            self.request,
            self.timeout,
            self.environment,
            curl,
            on_chunk=reference_collector(append_bytes, taps.chunks),
            on_header=taps.on_header,
        )

        def finish(code: int) -> IO[Result[bytes, TranslationError]]:
            def decide(
                parts: tuple[tuple[bytes, ...], tuple[bytes, ...]],
            ) -> Result[bytes, TranslationError]:
                collected, headers = parts
                if code >= 400:
                    return status_failure(code, collected, headers)

                return Ok(b"".join(collected))

            return io_map(
                io_pair(read_reference(taps.chunks), read_reference(taps.header_lines)),
                decide,
            )

        return io_result_bind(curl_transfer(curl, "unreachable"), finish)

    def consume(
        self,
        step: Callable[[Any, bytes], IO[Result[Any, TranslationError]]],
        initial: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        state: Reference[Result[Any, TranslationError]] = Reference(initial)
        remainder: Reference[bytes] = Reference(b"")
        error_active: Reference[bool] = Reference(False)
        taps = response_taps()
        gated = reference_gate(error_active, fold_io_push(step, state))
        feed = line_push(gated, remainder)
        collect = reference_collector(append_bytes, taps.chunks)
        arm = reference_write_when(is_error_status_line, error_active, True)
        curl = self.new_curl()

        def on_chunk(chunk: bytes) -> None:
            collect(chunk)
            feed(chunk)

        def on_header(line: bytes) -> None:
            taps.on_header(line)
            arm(line)

        configure_curl(
            self.request,
            self.timeout,
            self.environment,
            curl,
            on_chunk=on_chunk,
            on_header=on_header,
        )

        def emit_remainder(remaining: bytes) -> None:
            if remaining:
                gated(remaining)

        def finish(code: int) -> IO[Result[Any, TranslationError]]:
            def decide(
                parts: tuple[
                    Result[Any, TranslationError],
                    tuple[tuple[bytes, ...], tuple[bytes, ...]],
                ],
            ) -> Result[Any, TranslationError]:
                outcome, rest = parts
                collected, headers = rest
                if code >= 400:
                    return status_failure(code, collected, headers)

                return outcome

            def drained(_: None) -> IO[Result[Any, TranslationError]]:
                return io_map(
                    io_pair(
                        read_reference(state),
                        io_pair(
                            read_reference(taps.chunks),
                            read_reference(taps.header_lines),
                        ),
                    ),
                    decide,
                )

            return io_and_then(
                io_map(read_reference(remainder), emit_remainder), drained(None)
            )

        return io_result_bind(curl_transfer(curl, "interrupted"), finish)


def curl_open(
    request: Any,
    timeout: float,
    environment: Mapping[str, str],
    new_curl: Callable[[], Any] = pycurl.Curl,
) -> Result[CurlResponse, TranslationError]:
    return Ok(CurlResponse(request, timeout, environment, new_curl))


def verbose_retry_log(
    run_context: Context, retry_number: int, retries: int, wait: float
) -> IO[None]:
    return run_context.console.log_verbose(
        "l2l: transient failure; retry %d/%d in %.1fs" % (retry_number, retries, wait)
    )


def with_retries[T](
    run_context: Context, attempt: Callable[[], IO[Result[T, TranslationError]]]
) -> IO[Result[T, TranslationError]]:
    delays = plan_backoff(
        run_context.settings.retry_base_delay,
        run_context.settings.retry_cap,
        run_context.settings.retry_attempts,
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
                verbose_retry_log(run_context, index + 1, len(delays), wait),
                lambda _: io_and_then(
                    IO(lambda: run_context.sleep(wait)),
                    attempt_at(index + 1),
                ),
            )

        return io_bind(attempt(), decide)

    return attempt_at(0)


def parse_json_body(
    body_result: Result[bytes, TranslationError],
) -> Result[tuple[str, Any], TranslationError]:
    if isinstance(body_result, Err):
        return body_result

    try:
        text = body_result.value.decode("utf-8")
        return Ok((text, json.loads(text)))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return fail_http("protocol", "invalid JSON response: %s" % error)


def read_failure(error: Exception) -> Result[tuple[str, Any], TranslationError]:
    if isinstance(error, TRANSPORT_ERRORS):
        return fail_http("unreachable", "response read failed: %s" % error)

    return fail_http("protocol", "%s: %s" % (type(error).__name__, error))


def http_post_json(
    run_context: Context, payload: Mapping[str, Any]
) -> IO[Result[dict[str, Any], TranslationError]]:
    def opened() -> Result[Any, TranslationError]:
        return run_context.open_http(
            http_request(run_context.config, payload), run_context.config.timeout
        )

    def transferred(
        outcome: Result[Any, TranslationError],
    ) -> IO[Result[tuple[str, Any], TranslationError]]:
        if isinstance(outcome, Err):
            return io_result(outcome)

        return io_catch_result(
            io_map(outcome.value.body(), parse_json_body), read_failure
        )

    def attempt() -> IO[Result[tuple[str, Any], TranslationError]]:
        return io_bind(IO(opened), transferred)

    def record(
        outcome: Result[tuple[str, Any], TranslationError],
    ) -> IO[Result[dict[str, Any], TranslationError]]:
        if isinstance(outcome, Err):
            return io_map(
                log_error(run_context.log, describe(outcome.error)),
                lambda _: Err(outcome.error),
            )

        body, loaded = outcome.value
        return io_map(
            log_entry(run_context.log, "RESPONSE", body), lambda _: Ok(loaded)
        )

    return io_bind(
        io_and_then(log_request(run_context.log, payload), attempt()),
        record,
    )


def http_stream(
    run_context: Context,
    payload: Mapping[str, Any],
    drive: Callable[[Any], IO[Result[StreamState, TranslationError]]],
) -> IO[Result[StreamState, TranslationError]]:
    def open_stream(
        body: Mapping[str, Any], label: str
    ) -> IO[Result[Any, TranslationError]]:
        return io_and_then(
            log_request(run_context.log, body, label),
            IO(
                lambda: run_context.open_http(
                    http_request(run_context.config, body, accept="text/event-stream"),
                    run_context.config.timeout,
                )
            ),
        )

    def logged(
        opened: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]:
        if isinstance(opened, Ok):
            return io_result(opened)

        return io_map(
            log_error(run_context.log, describe(opened.error)),
            lambda _: opened,
        )

    def respond(
        opened: Result[Any, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        if isinstance(opened, Err):
            return io_result(opened)

        return drive(opened.value)

    def attempt(
        body: Mapping[str, Any], label: str
    ) -> IO[Result[StreamState, TranslationError]]:
        return io_bind(io_bind(open_stream(body, label), logged), respond)

    def decide(
        outcome: Result[StreamState, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        if (
            not isinstance(outcome, Err)
            or "stream_options" not in payload
            or not (isinstance(outcome.error, HttpError))
            or "stream_options" not in outcome.error.detail
        ):
            return io_result(outcome)

        return attempt(
            {key: value for key, value in payload.items() if key != "stream_options"},
            "REQUEST (stream, retry)",
        )

    return io_bind(attempt(payload, "REQUEST (stream)"), decide)


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

    return io_bind(response.consume(advance, Ok(StreamState())), flush)


def collect_stream(
    run_context: Context, payload: Mapping[str, Any]
) -> IO[Result[StreamState, TranslationError]]:
    def on_reasoning(text: str) -> IO[None]:
        return run_context.console.stream_reasoning(text)

    def note_progress(label: str, count: int) -> IO[None]:
        def continued(_: None) -> IO[None]:
            return run_context.console.progress(label, count)

        return (
            io_and_then(run_context.console.end_raw(), continued(None))
            if label == "Working"
            else run_context.console.progress(label, count)
        )

    def on_raw_line(raw_line: bytes) -> IO[None]:
        return run_log_write(run_context.log, raw_line.decode("utf-8", "replace"))

    def conclude(
        outcome: Result[StreamState, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        def recorded(_: None) -> IO[Result[StreamState, TranslationError]]:
            if isinstance(outcome, Err):
                return io_map(
                    log_error(run_context.log, describe(outcome.error)),
                    lambda _: outcome,
                )

            return io_map(run_log_write(run_context.log, "\n"), lambda _: outcome)

        return io_and_then(run_context.console.end_raw(), recorded(None))

    def handle(error: Exception) -> Result[StreamState, TranslationError]:
        if isinstance(error, TRANSPORT_ERRORS):
            return fail_http("interrupted", str(error))

        return fail_http("protocol", "%s: %s" % (type(error).__name__, error))

    def drive(response: Any) -> IO[Result[StreamState, TranslationError]]:
        return io_catch_result(
            io_bind(
                drive_stream(response, note_progress, on_raw_line, on_reasoning),
                conclude,
            ),
            handle,
        )

    return http_stream(run_context, payload, drive)


def log_all(
    console: Console, messages: tuple[str, ...]
) -> IO[Result[tuple[()], TranslationError]]:
    def step(_: tuple[()], message: str) -> IO[Result[tuple[()], TranslationError]]:
        return io_map(console.log_verbose(message), lambda _: Ok(()))

    return fold_io(messages, step, Ok(()))


def apply_stream_override(
    run_context: Context, payload: dict[str, Any]
) -> dict[str, Any]:
    if run_context.stream is None:
        return payload

    return {**payload, "stream": run_context.stream}


def chat(
    run_context: Context,
    system: str,
    user: str,
    model: str,
    parameters: Mapping[str, Any],
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > run_context.config.maximum_tokens:
        return io_result(fail_budget(estimated, run_context.config.maximum_tokens))

    payload = apply_stream_override(
        run_context,
        build_chat_payload(model, system, user, parameters),
    )
    call = with_retries(
        run_context,
        lambda: (
            streamed_call(run_context, payload)
            if payload.get("stream", True)
            else plain_call(run_context, payload)
        ),
    )

    def stopped(
        reply_result: Result[ChatReply, TranslationError],
    ) -> IO[Result[Translated, TranslationError]]:
        return io_and_then(
            run_context.console.stop(),
            conclude_chat(run_context, reply_result, usage, estimated),
        )

    return io_and_then(run_context.console.start("Working"), io_bind(call, stopped))


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
    run_context: Context, payload: Mapping[str, Any]
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

    return io_map(
        collect_stream(run_context, full_payload),
        lambda outcome: result_map(outcome, lambda state: to_chat_reply(state)),
    )


def plain_call(
    run_context: Context,
    payload: Mapping[str, Any],
) -> IO[Result[ChatReply, TranslationError]]:
    return io_bind(
        http_post_json(run_context, payload),
        lambda body_result: io_result(result_bind(body_result, plain_reply)),
    )


type RepairOutcome = tuple[str, Usage, bool]


def repaired_call(
    run_context: Context,
    model: str,
    parameters: Mapping[str, Any],
    instruction: str,
    initial_user: str,
    validate: Callable[[str], Maybe[str]],
    build_retry_user: Callable[[str, str], str],
    maximum_repairs: int,
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

            if failed > maximum_repairs:
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

        return io_bind(
            chat(run_context, instruction, user, model, parameters, usage), assessed
        )

    return lambda usage: attempt(initial_user, 1, usage)


def conclude_chat(
    run_context: Context,
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

        return io_map(log_all(run_context.console, messages), finish)

    return io_bind(read_reference(run_context.console.verbose), conclude)

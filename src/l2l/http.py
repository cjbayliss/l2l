from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import reduce
from types import MappingProxyType
from typing import Final, Protocol, assert_never, final

import pycurl

from l2l import __version__
from l2l.context import Context, HttpRequest, HttpResponse, HttpStreamResponse
from l2l.effects import log_entry, log_error, log_request, run_log_write
from l2l.errors import (
    HttpError,
    HttpKind,
    TranslationError,
    describe,
    fail_budget,
    fail_http,
)
from l2l.json_values import (
    JsonArray,
    JsonObject,
    JsonValue,
    is_json_array,
    is_json_object,
    is_object_list,
    is_object_table,
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
    io_result_map,
    line_push,
    maybe_either,
    maybe_from_optional,
    maybe_map,
    maybe_or,
    maybe_to_optional,
    modify_reference_with,
    read_reference,
    reference_collector,
    reference_gate,
    reference_write_when,
    result_bind,
)
from l2l.text import (
    SseDone,
    SseEvent,
    SseMalformed,
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
lazy from l2l.console import Console
lazy from l2l.settings import Config


def transient(error: TranslationError) -> bool:
    match error:
        case HttpError(kind="unreachable"):
            return True

        case HttpError(kind="status", status=int(code)):
            return code == 429 or code >= 500

        case _:
            return False


def plan_backoff(base_delay: float, cap: float, attempts: int) -> tuple[float, ...]:
    def delay(index: int) -> float:
        exponential: float = base_delay * 2.0**index
        return min(exponential, cap)

    return tuple(delay(index) for index in range(max(attempts, 0)))


def retry_delay(error: TranslationError, planned: float) -> float:
    match error:
        case HttpError(retry_after=float(seconds)):
            return max(planned, seconds)

        case _:
            return planned


OPENROUTER_HOST: Final = "openrouter.ai"

PROVIDER_NAME_PATTERN: Final = re.compile(
    r"['\"]provider_name['\"]\s*:\s*['\"]([^'\"]+)['\"]"
)


def is_openrouter(config: Config) -> bool:
    return OPENROUTER_HOST in config.base_url


def failed_provider_name(error: TranslationError) -> str | None:
    match error:
        case HttpError(detail=str(detail)):
            found = PROVIDER_NAME_PATTERN.search(detail)
            return found.group(1) if found is not None else None

        case _:
            return None


def with_blocked_providers(
    config: Config, payload: JsonObject, blocked_providers: frozenset[str]
) -> JsonObject:
    if not is_openrouter(config) or not blocked_providers:
        return payload

    routing = payload.get("provider")
    existing_routing = (
        routing if is_json_object(routing) else MappingProxyType[str, JsonValue]({})
    )
    ignored = existing_routing.get("ignore")
    existing_ignored: set[str] = (
        {name for name in ignored if isinstance(name, str)}
        if is_json_array(ignored)
        else set()
    )
    return {
        **payload,
        "provider": {
            **existing_routing,
            "ignore": sorted(existing_ignored | blocked_providers),
        },
    }


def block_failing_provider(run_context: Context, failure: TranslationError) -> IO[bool]:
    provider = failed_provider_name(failure)
    if provider is None or not is_openrouter(run_context.config):
        return io_pure(False)

    def add_provider_when_absent(
        blocked: frozenset[str],
    ) -> tuple[frozenset[str], bool]:
        return (blocked | frozenset({provider}), provider not in blocked)

    def announce_blocked_provider(blocked_a_new_provider: bool) -> IO[bool]:
        if not blocked_a_new_provider:
            return io_pure(False)

        return io_and_then(
            run_context.console.log_verbose(
                f"l2l: openrouter provider {provider} failed; added to the blocklist, "
                "retrying without it"
            ),
            io_pure(True),
        )

    return io_bind(
        modify_reference_with(
            run_context.provider_blocklist,
            add_provider_when_absent,
        ),
        announce_blocked_provider,
    )


ProgressCallback = Callable[[str, int], IO[None]]
RawLineLogger = Callable[[bytes], IO[None]]
ReasoningLogger = Callable[[str], IO[None]]


def build_chat_payload(
    model: str, system: str, user: str, parameters: Mapping[str, JsonValue]
) -> JsonObject:
    return {
        "model": model,
        "messages": (
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ),
        **{key: value for key, value in parameters.items() if value is not None},
    }


TRANSPORT_ERRORS: Final = (OSError, TimeoutError)

CONNECT_ERROR_CODES: Final = frozenset(
    {1, 3, 5, 6, 7, 35, 51, 58, 60, 61, 64, 66, 77, 83, 90, 96, 200}
)


def curl_error_code(error: pycurl.error) -> int:
    match error.args:
        case (int(code), _):
            return code

        case _:
            return 0


def curl_error_detail(error: pycurl.error) -> str:
    match error.args:
        case (_, str(detail)):
            return detail

        case _:
            return str(error)


def curl_failure(error: pycurl.error, kind: HttpKind) -> Err[TranslationError]:
    resolved = "unreachable" if curl_error_code(error) in CONNECT_ERROR_CODES else kind
    return fail_http(resolved, curl_error_detail(error))


def environment_value(
    environment: Mapping[str, str], names: tuple[str, ...]
) -> Maybe[str]:
    initial: Maybe[str] = NOTHING
    return reduce(
        maybe_or,
        (maybe_from_optional(environment.get(name) or None) for name in names),
        initial,
    )


def proxy_for(url: str, environment: Mapping[str, str]) -> Maybe[str]:
    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    return environment_value(environment, (scheme + "_proxy", "all_proxy", "ALL_PROXY"))


def noproxy_for(environment: Mapping[str, str]) -> Maybe[str]:
    return environment_value(environment, ("no_proxy", "NO_PROXY"))


def unexpected_response(body: object) -> Err[TranslationError]:
    return fail_http("protocol", f"unexpected response shape: {str(body)[:500]}")


def extract_message(
    body: object,
) -> Result[tuple[JsonObject, JsonValue], TranslationError]:
    if not is_json_object(body):
        return unexpected_response(body)

    choices = body.get("choices")
    if not is_json_array(choices) or not choices:
        return unexpected_response(body)

    first_choice = choices[0]
    if not is_json_object(first_choice) or "message" not in first_choice:
        return unexpected_response(body)

    message = first_choice["message"]
    if not is_json_object(message) or "content" not in message:
        return unexpected_response(body)

    return Ok((message, message["content"]))


def thinking_entry_text(entry: object) -> str | None:
    text = entry.get("text") if is_json_object(entry) else None
    return text if isinstance(text, str) and text else None


def collect_thinking_texts(part: Mapping[str, object]) -> tuple[str, ...]:
    thinking = part.get("thinking")
    if not is_json_array(thinking):
        return ()

    return tuple(
        entry_text
        for entry_text in (thinking_entry_text(entry) for entry in thinking)
        if entry_text is not None
    )


def collect_content_parts(
    parts: JsonArray,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def step(
        accumulator: tuple[tuple[str, ...], tuple[str, ...]], part: JsonValue
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        texts, thoughts = accumulator
        if not is_json_object(part):
            return (*texts, str(part)), thoughts

        if part.get("type") == "thinking":
            return texts, thoughts + collect_thinking_texts(part)

        text = part.get("text")
        if isinstance(text, str) and text:
            return (*texts, text), thoughts

        return accumulator

    initial: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    return reduce(step, parts, initial)


def flatten_content_parts(
    content: JsonValue,
) -> tuple[JsonValue, tuple[str, ...]]:
    if not is_json_array(content):
        return content, ()

    texts, thoughts = collect_content_parts(content)
    return "".join(texts), thoughts


REASONING_KEYS: Final = ("reasoning_content", "reasoning")


def reasoning_at(
    container: Mapping[str, object], clean: Callable[[str], Maybe[str]]
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


def single_text_tuple(text: str) -> tuple[str, ...]:
    return (text,)


def message_reasoning_texts(message: Mapping[str, object]) -> tuple[str, ...]:
    found = reasoning_at(message, keep_trimmed)
    return maybe_either(found, single_text_tuple, tuple)


def first_choice_field(payload: Mapping[str, object], field: str) -> JsonValue | None:
    choices = payload.get("choices")
    if not is_json_array(choices) or not choices:
        return None

    first_choice = choices[0]
    if not is_json_object(first_choice):
        return None

    return first_choice.get(field)


def parse_chunk_delta(chunk: Mapping[str, object]) -> JsonObject:
    delta = first_choice_field(chunk, "delta")
    return delta if is_json_object(delta) else MappingProxyType({})


def completion_finish_reason(payload: Mapping[str, object]) -> str | None:
    found = first_choice_field(payload, "finish_reason")
    return found if isinstance(found, str) else None


def delta_text(delta: Mapping[str, object]) -> str:
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if not is_object_list(value):
        return ""

    return "".join(choice_delta_text(item) for item in value if is_object_table(item))


def choice_delta_text(delta: Mapping[object, object]) -> str:
    text = delta.get("text", "")
    return text if isinstance(text, str) else ""


@final
@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    label: str
    count: int
    reasoning: str = ""


@final
@dataclass(frozen=True, slots=True)
class StreamState:
    reasoning: Cons[str] | None = None
    contents: Cons[str] | None = None
    reported_usage: Mapping[str, JsonValue] | None = None
    chunk_count: int = 0
    content_started: bool = False
    think: ThinkState = field(default_factory=ThinkState)
    sse: SseState = field(default_factory=SseState)
    finish_reason: str | None = None
    completed: bool = False
    progress_update: ProgressUpdate | None = None


def stream_text(state: StreamState) -> str:
    visible_held_text = (
        state.think.buffered_text
        if state.think.scanning_opening_tag and not state.think.inside_reasoning
        else ""
    )
    return "".join(cons_to_tuple(state.contents)) + visible_held_text


def stream_step(
    state: StreamState, chunk: JsonObject
) -> Result[StreamState, TranslationError]:
    finish_reason = completion_finish_reason(chunk)
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
        return Ok(
            replace(
                state,
                reported_usage=reported_usage,
                finish_reason=finish_reason or state.finish_reason,
                progress_update=None,
            )
        )

    think_state, thinking, visible = (
        think_step(state.think, text) if text else (state.think, False, "")
    )
    reasoning_texts: tuple[str, ...] = maybe_either(
        reasoning_found, single_text_tuple, tuple
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
        replace(
            state,
            reasoning=cons_all(reasoning_texts, state.reasoning),
            contents=cons(visible, state.contents) if visible else state.contents,
            reported_usage=reported_usage,
            chunk_count=state.chunk_count + thinking_progress + text_progress,
            content_started=state.content_started or bool(text),
            think=think_state,
            finish_reason=finish_reason or state.finish_reason,
            progress_update=progress_update,
        )
    )


def ingest_raw_line(
    state: StreamState, raw_line: bytes
) -> Result[StreamState, TranslationError]:
    sse, frame = sse_step(state.sse, raw_line)
    updated = state if sse == state.sse else replace(state, sse=sse)
    match frame:
        case None:
            return Ok(replace(updated, progress_update=None))

        case SseDone():
            return Ok(replace(updated, completed=True, progress_update=None))

        case SseMalformed():
            return fail_http("stream", f"malformed SSE data: {frame.data[:200]!r}")

        case SseEvent():
            if isinstance(frame.payload.get("error"), dict):
                return fail_http("stream", str(frame.payload["error"])[:500])

            return stream_step(updated, frame.payload)

        case _:
            assert_never(frame)


def require_completed_stream(
    state: StreamState,
) -> Result[StreamState, TranslationError]:
    if state.finish_reason == "length":
        return fail_http("stream", "response truncated: finish_reason=length")

    if not (state.completed or state.finish_reason is not None):
        return fail_http("stream", "stream ended without a completion event")

    return Ok(state)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ChatReply:
    content: str
    reasoning: tuple[str, ...] = ()
    reported_usage: Mapping[str, JsonValue] | None = None
    chunk_count: int = 0
    reasoning_shown: bool = False


def plain_reply(body: object) -> Result[ChatReply, TranslationError]:
    if not is_json_object(body):
        return unexpected_response(body)

    message_result = extract_message(body)
    if isinstance(message_result, Err):
        return message_result

    if completion_finish_reason(body) == "length":
        return fail_http("stream", "response truncated: finish_reason=length")

    message, content = message_result.value
    texts, thoughts = flatten_content_parts(content)
    if not isinstance(texts, str):
        return fail_http("protocol", f"unexpected content type: {type(texts).__name__}")

    usage_report = body.get("usage")
    reported_usage = (
        MappingProxyType(usage_report) if is_json_object(usage_report) else None
    )
    return Ok(
        ChatReply(
            content=texts,
            reasoning=message_reasoning_texts(message) + thoughts,
            reported_usage=reported_usage,
        )
    )


def http_request(
    config: Config, payload: JsonObject, accept: str | None = None
) -> HttpRequest:
    headers = MappingProxyType(
        {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.api_key,
            "User-Agent": "l2l/" + __version__,
            **({"Accept": accept} if accept else {}),
        }
    )

    return HttpRequest(
        url=config.base_url + "/chat/completions",
        body=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )


def nonnegative_seconds(seconds: float) -> float:
    return max(seconds, 0.0)


def retry_after_seconds(raw: str | None) -> Maybe[float]:
    if raw is None:
        return NOTHING

    return maybe_map(parse_float(raw), nonnegative_seconds)


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
    return (*collected, item)


class CurlHandle(Protocol):
    def close(self) -> None: ...

    def getinfo(self, info: int) -> int: ...

    def perform(self) -> None: ...

    def setopt(self, option: int, value: object) -> None: ...


def configure_curl(
    request: HttpRequest,
    timeout: float,
    environment: Mapping[str, str],
    curl: CurlHandle,
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


@final
@dataclass(frozen=True, slots=True)
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


def curl_transfer(
    curl: CurlHandle, kind: HttpKind
) -> IO[Result[int, TranslationError]]:
    def thunk() -> Result[int, TranslationError]:
        try:
            curl.perform()
            return Ok(curl.getinfo(pycurl.RESPONSE_CODE))
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


@final
@dataclass(frozen=True, slots=True)
class CurlResponse:
    request: HttpRequest
    timeout: float
    environment: Mapping[str, str]
    new_curl: Callable[[], CurlHandle] = pycurl.Curl

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

    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]:
        state: Reference[Result[S, TranslationError]] = Reference(initial)
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

        def finish(code: int) -> IO[Result[S, TranslationError]]:
            def decide(
                parts: tuple[
                    Result[S, TranslationError],
                    tuple[tuple[bytes, ...], tuple[bytes, ...]],
                ],
            ) -> Result[S, TranslationError]:
                outcome, rest = parts
                collected, headers = rest
                if code >= 400:
                    return status_failure(code, collected, headers)

                return outcome

            def drained(_: None) -> IO[Result[S, TranslationError]]:
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
    request: HttpRequest,
    timeout: float,
    environment: Mapping[str, str],
    new_curl: Callable[[], CurlHandle] = pycurl.Curl,
) -> Result[HttpResponse, TranslationError]:
    return Ok(CurlResponse(request, timeout, environment, new_curl))


def verbose_retry_log(
    run_context: Context, retry_number: int, retries: int, wait: float
) -> IO[None]:
    return run_context.console.log_verbose(
        f"l2l: transient failure; retry {retry_number}/{retries} in {wait:.1f}s"
    )


def with_retries[T](
    run_context: Context,
    retryable_action: Callable[[], IO[Result[T, TranslationError]]],
) -> IO[Result[T, TranslationError]]:
    delays = plan_backoff(
        run_context.settings.retry_base_delay,
        run_context.settings.retry_cap,
        run_context.settings.retry_attempts,
    )

    def attempt_at(
        index: int, blocked_provider_retries: int
    ) -> IO[Result[T, TranslationError]]:
        def decide(
            outcome: Result[T, TranslationError],
        ) -> IO[Result[T, TranslationError]]:
            if isinstance(outcome, Ok):
                return io_result(outcome)

            failure = outcome.error
            if not transient(failure):
                return io_result(outcome)

            def proceed_after_blocking(
                blocked_a_new_provider: bool,
            ) -> IO[Result[T, TranslationError]]:
                if (
                    blocked_a_new_provider
                    and blocked_provider_retries
                    < run_context.settings.provider_block_attempts
                ):
                    return attempt_at(index, blocked_provider_retries + 1)

                if index >= len(delays):
                    return io_result(outcome)

                wait = retry_delay(failure, delays[index])

                def sleep_then_retry(
                    _: None,
                ) -> IO[Result[T, TranslationError]]:
                    return io_and_then(
                        IO(lambda: run_context.sleep(wait)),
                        attempt_at(index + 1, blocked_provider_retries),
                    )

                return io_bind(
                    verbose_retry_log(run_context, index + 1, len(delays), wait),
                    sleep_then_retry,
                )

            return io_bind(
                block_failing_provider(run_context, failure),
                proceed_after_blocking,
            )

        return io_bind(retryable_action(), decide)

    return attempt_at(0, 0)


def parse_json_body(
    body_result: Result[bytes, TranslationError],
) -> Result[tuple[str, JsonValue], TranslationError]:
    if isinstance(body_result, Err):
        return body_result

    try:
        text = body_result.value.decode("utf-8")
        loaded: JsonValue = json.loads(text)
        return Ok((text, loaded))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return fail_http("protocol", f"invalid JSON response: {error}")


def read_failure(error: Exception) -> Result[tuple[str, JsonValue], TranslationError]:
    if isinstance(error, TRANSPORT_ERRORS):
        return fail_http("unreachable", f"response read failed: {error}")

    return fail_http("protocol", f"{type(error).__name__}: {error}")


def log_failure[T](
    run_context: Context, outcome: Result[T, TranslationError]
) -> IO[Result[T, TranslationError]]:
    if isinstance(outcome, Ok):
        return io_result(outcome)

    return io_and_then(
        log_error(run_context.log, describe(outcome.error)), io_result(outcome)
    )


def http_post_json(
    run_context: Context, payload: JsonObject
) -> IO[Result[JsonValue, TranslationError]]:
    def opened() -> Result[HttpResponse, TranslationError]:
        return run_context.open_http(
            http_request(run_context.config, payload), run_context.config.timeout
        )

    def transferred(
        outcome: Result[HttpResponse, TranslationError],
    ) -> IO[Result[tuple[str, JsonValue], TranslationError]]:
        if isinstance(outcome, Err):
            return io_result(outcome)

        return io_catch_result(
            io_map(outcome.value.body(), parse_json_body), read_failure
        )

    def fetch_response_body() -> IO[Result[tuple[str, JsonValue], TranslationError]]:
        return io_bind(IO(opened), transferred)

    def record(
        outcome: Result[tuple[str, JsonValue], TranslationError],
    ) -> IO[Result[JsonValue, TranslationError]]:
        if isinstance(outcome, Err):
            return log_failure(run_context, outcome)

        body, loaded = outcome.value
        return io_and_then(
            log_entry(run_context.log, "RESPONSE", body), io_result(Ok(loaded))
        )

    return io_bind(
        io_and_then(log_request(run_context.log, payload), fetch_response_body()),
        record,
    )


def http_stream(
    run_context: Context,
    payload: JsonObject,
    drive: Callable[[HttpResponse], IO[Result[StreamState, TranslationError]]],
) -> IO[Result[StreamState, TranslationError]]:
    def open_stream(
        body: JsonObject, label: str
    ) -> IO[Result[HttpResponse, TranslationError]]:
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
        opened: Result[HttpResponse, TranslationError],
    ) -> IO[Result[HttpResponse, TranslationError]]:
        return log_failure(run_context, opened)

    def respond(
        opened: Result[HttpResponse, TranslationError],
    ) -> IO[Result[StreamState, TranslationError]]:
        if isinstance(opened, Err):
            return io_result(opened)

        return drive(opened.value)

    def open_stream_and_respond(
        body: JsonObject, label: str
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

        return open_stream_and_respond(
            {key: value for key, value in payload.items() if key != "stream_options"},
            "REQUEST (stream, retry)",
        )

    return io_bind(open_stream_and_respond(payload, "REQUEST (stream)"), decide)


def drive_stream(
    response: HttpStreamResponse,
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
                return io_and_then(
                    on_progress(request.label, request.count), io_result(outcome)
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

        return io_result(
            result_bind(ingest_raw_line(outcome.value, b""), require_completed_stream)
        )

    return io_bind(response.consume(advance, Ok(StreamState())), flush)


def collect_stream(
    run_context: Context, payload: JsonObject
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
                return log_failure(run_context, outcome)

            return io_and_then(run_log_write(run_context.log, "\n"), io_result(outcome))

        return io_and_then(run_context.console.end_raw(), recorded(None))

    def handle(error: Exception) -> Result[StreamState, TranslationError]:
        if isinstance(error, TRANSPORT_ERRORS):
            return fail_http("interrupted", str(error))

        return fail_http("protocol", f"{type(error).__name__}: {error}")

    def drive(response: HttpResponse) -> IO[Result[StreamState, TranslationError]]:
        return io_catch_result(
            io_bind(
                drive_stream(response, note_progress, on_raw_line, on_reasoning),
                conclude,
            ),
            handle,
        )

    return http_stream(run_context, payload, drive)


def log_all(console: Console, messages: tuple[str, ...]) -> IO[None]:
    def step(_: tuple[()], message: str) -> IO[Result[tuple[()], TranslationError]]:
        return io_and_then(console.log_verbose(message), io_result(Ok(())))

    def ignore_outcome(_: Result[tuple[()], TranslationError]) -> None:
        return None

    return io_map(fold_io(messages, step, Ok(())), ignore_outcome)


def chat(
    run_context: Context,
    system: str,
    user: str,
    model: str,
    parameters: Mapping[str, JsonValue],
    usage: Usage,
    *,
    response_format: Mapping[str, JsonValue] | None = None,
) -> IO[Result[Translated, TranslationError]]:
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > run_context.config.maximum_tokens:
        return io_result(fail_budget(estimated, run_context.config.maximum_tokens))

    payload = build_chat_payload(model, system, user, parameters)
    if response_format is not None:
        payload = {**payload, "response_format": dict(response_format)}

    def request_reply() -> IO[Result[ChatReply, TranslationError]]:
        def reply_with_blocklist(
            blocked_providers: frozenset[str],
        ) -> IO[Result[ChatReply, TranslationError]]:
            scoped_payload = with_blocked_providers(
                run_context.config, payload, blocked_providers
            )
            return (
                streamed_call(run_context, scoped_payload)
                if scoped_payload.get("stream", True)
                else plain_call(run_context, scoped_payload)
            )

        return io_bind(
            read_reference(run_context.provider_blocklist), reply_with_blocklist
        )

    call = with_retries(run_context, request_reply)

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
    run_context: Context, payload: JsonObject
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

    return io_result_map(collect_stream(run_context, full_payload), to_chat_reply)


def plain_reply_from_body(
    body_result: Result[JsonValue, TranslationError],
) -> IO[Result[ChatReply, TranslationError]]:
    return io_result(result_bind(body_result, plain_reply))


def plain_call(
    run_context: Context,
    payload: JsonObject,
) -> IO[Result[ChatReply, TranslationError]]:
    return io_bind(http_post_json(run_context, payload), plain_reply_from_body)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class RepairOutcome:
    text: str
    usage: Usage
    validated: bool
    problem: str | None = None


@final
@dataclass(frozen=True, slots=True)
class RepairPolicy:
    validate: Callable[[str], Maybe[str]]
    build_retry_user: Callable[[str, str], str]
    maximum_repairs: int
    on_repair: Callable[[int, str], IO[None]]
    on_exhausted: Callable[[str, str], IO[None]] | None = None


def exhausted_repair(
    policy: RepairPolicy, reply: Translated, problem: str
) -> IO[Result[RepairOutcome, TranslationError]]:
    announce = (
        policy.on_exhausted(reply.text, problem)
        if policy.on_exhausted is not None
        else io_pure(None)
    )

    unvalidated_outcome: Result[RepairOutcome, TranslationError] = Ok(
        RepairOutcome(
            text=reply.text,
            usage=reply.usage,
            validated=False,
            problem=problem,
        )
    )
    return io_and_then(announce, io_result(unvalidated_outcome))


def repaired_call(
    run_context: Context,
    model: str,
    parameters: Mapping[str, JsonValue],
    instruction: str,
    initial_user: str,
    policy: RepairPolicy,
) -> Callable[[Usage], IO[Result[RepairOutcome, TranslationError]]]:
    def request_and_repair_reply(
        user: str, failed: int, usage: Usage
    ) -> IO[Result[RepairOutcome, TranslationError]]:
        def assessed(
            reply_result: Result[Translated, TranslationError],
        ) -> IO[Result[RepairOutcome, TranslationError]]:
            if isinstance(reply_result, Err):
                return io_result(reply_result)

            reply = reply_result.value
            problem = policy.validate(reply.text)
            if isinstance(problem, Nothing):
                return io_result(
                    Ok(
                        RepairOutcome(
                            text=reply.text,
                            usage=reply.usage,
                            validated=True,
                            problem=None,
                        )
                    )
                )

            if failed > policy.maximum_repairs:
                return exhausted_repair(policy, reply, problem.value)

            def continued(_: None) -> IO[Result[RepairOutcome, TranslationError]]:
                return request_and_repair_reply(
                    policy.build_retry_user(reply.text, problem.value),
                    failed + 1,
                    reply.usage,
                )

            return io_bind(policy.on_repair(failed, problem.value), continued)

        return io_bind(
            chat(run_context, instruction, user, model, parameters, usage), assessed
        )

    def request_initial_repair(
        usage: Usage,
    ) -> IO[Result[RepairOutcome, TranslationError]]:
        return request_and_repair_reply(initial_user, 1, usage)

    return request_initial_repair


def reply_usage(usage: Usage, reply: ChatReply, estimated: int) -> Usage:
    reported_usage = reply.reported_usage
    if reported_usage:
        return add_usage(usage, reported_usage)

    return add_usage(
        usage,
        {
            "prompt_tokens": estimated,
            "completion_tokens": reply.chunk_count,
        },
    )


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
    think_texts: tuple[str, ...] = maybe_either(think_text, single_text_tuple, tuple)

    def conclude(verbose: bool) -> IO[Result[Translated, TranslationError]]:
        messages = (
            tuple(
                text.rstrip() for text in reply.reasoning + think_texts if text.strip()
            )
            if verbose and not reply.reasoning_shown
            else ()
        )

        def finish(
            _: None,
        ) -> Result[Translated, TranslationError]:
            return Ok(Translated(content, reply_usage(usage, reply, estimated)))

        return io_map(log_all(run_context.console, messages), finish)

    return io_bind(read_reference(run_context.console.verbose), conclude)

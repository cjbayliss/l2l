from collections.abc import Iterator

from fakes import FakeStreamResponse, stream_chunks

from zh2en.errors import describe
from zh2en.http import (
    ProgressRequest,
    StreamState,
    drive_stream,
    flatten_content_parts,
    plain_reply,
    step_stream,
    stream_step,
)
from zh2en.monads import IO, Err, Ok, io_pure
from zh2en.text import (
    THINK_CLOSE,
    SseState,
    ThinkState,
    sse_step,
    strip_think_tag,
    think_step,
)


def feed(state: StreamState, *texts: str) -> StreamState:
    for text in texts:
        result = stream_step(state, {"choices": [{"delta": {"content": text}}]})
        assert isinstance(result, Ok)
        state = result.value
    return state


def test_think_step_holds_partial_prefix() -> None:
    state, thinking, visible = think_step(ThinkState(), "<th")
    assert visible == ""
    assert (state.checking, state.open) == (True, False)

    state, thinking, visible = think_step(state, "ink>hidden")
    assert visible == ""
    assert (state.checking, state.open) == (True, True)


def test_think_step_close_across_chunks() -> None:
    state, _, _ = think_step(ThinkState(), "<think>abc")
    state, thinking, visible = think_step(state, "def</thi")
    assert visible == ""
    assert state.open

    state, thinking, visible = think_step(state, "nk>out")
    assert (thinking, visible) == (True, "out")
    assert (state.checking, state.open, state.held) == (False, False, "")


def test_think_step_plain_text() -> None:
    state, thinking, visible = think_step(ThinkState(), "Hello")
    assert (thinking, visible) == (False, "Hello")


def test_think_step_leading_whitespace() -> None:
    state, thinking, visible = think_step(ThinkState(), "  ")
    assert visible == ""
    state, thinking, visible = think_step(state, "hi")
    assert visible == "  hi"


def test_think_step_bounds_whitespace_hold() -> None:
    state = ThinkState()
    for _ in range(10):
        state, _, visible = think_step(state, "        ")
        assert visible == ""

    assert len(state.held) <= len(THINK_CLOSE)
    state, _, visible = think_step(state, "hi")
    assert visible.endswith("hi")


def test_stream_step_filters_think_content() -> None:
    state = feed(StreamState(), "<think>secret ", "reasoning</think>", "Translation.")
    assert "".join(state.contents) == "Translation."


def test_stream_step_accumulates_reasoning_and_usage() -> None:
    state = StreamState()
    result = stream_step(
        state,
        {
            "choices": [{"delta": {"reasoning_content": "ponder"}}],
        },
    )
    assert isinstance(result, Ok)
    state = result.value
    assert state.reasoning == ("ponder",)
    assert state.progress_request == ProgressRequest("Thinking", 1)

    result = stream_step(
        state,
        {
            "choices": [{"delta": {"content": "Hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.5},
        },
    )
    assert isinstance(result, Ok)
    state = result.value
    assert state.contents == ("Hi",)
    assert state.reported == {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.5}
    assert state.counted == 2
    assert state.progress_request == ProgressRequest("Working", 1)


def test_stream_step_content_part_list() -> None:
    state = StreamState()
    result = stream_step(
        state,
        {"choices": [{"delta": {"content": [{"type": "text", "text": "ok"}]}}]},
    )
    assert isinstance(result, Ok)
    assert result.value.contents == ("ok",)


def test_step_stream_ignores_non_dict_chunks() -> None:
    state = feed(StreamState(), "keep")
    for raw in (
        b"data: [1,2,3]\n",
        b"\n",
        b'data: "x"\n',
        b"\n",
        b": keep-alive\n",
        b"data: [DONE]\n",
        b"\n",
    ):
        result = step_stream(state, raw)
        assert isinstance(result, Ok)
        state = result.value

    assert state.contents == ("keep",)
    assert state.counted == 1


def test_step_stream_reports_endpoint_error() -> None:
    result = step_stream(
        StreamState(),
        b'data: {"error": {"message": "overloaded"}}\n',
    )
    assert isinstance(result, Ok)
    dispatched = step_stream(result.value, b"\n")
    assert isinstance(dispatched, Err)
    assert "overloaded" in describe(dispatched.error)


def test_sse_step_assembles_frames_on_blank_lines() -> None:
    state = SseState()
    state, chunk = sse_step(state, b'data: {"choices": []}\n')
    assert chunk is None
    state, chunk = sse_step(state, b"\n")
    assert chunk == {"choices": []}


def test_sse_step_joins_multi_line_data() -> None:
    state = SseState()
    state, _ = sse_step(state, b'data: {"a": \n')
    state, _ = sse_step(state, b"data:1}\n")
    state, chunk = sse_step(state, b"\r\n")
    assert chunk == {"a": 1}


def test_sse_step_ignores_comments_and_other_fields() -> None:
    state = SseState()
    for raw in (b": keep-alive\n", b"event: ping\n", b"id: 42\n"):
        state, chunk = sse_step(state, raw)
        assert (state, chunk) == (SseState(), None)


def test_sse_step_ignores_blank_lines_without_pending_data() -> None:
    assert sse_step(SseState(), b"\n") == (SseState(), None)


def test_sse_step_holds_back_incomplete_json_frames() -> None:
    state, chunk = sse_step(SseState(), b"data: not json\n")
    assert chunk is None
    state, chunk = sse_step(state, b"\n")
    assert chunk is None
    assert state == SseState()


def test_step_stream_dispatches_completed_frames() -> None:
    result = step_stream(StreamState(), b'data: {"choices": []}\n')
    assert isinstance(result, Ok)
    assert result.value.sse == SseState(pending=('{"choices": []}',))

    result = step_stream(result.value, b"\n")
    assert isinstance(result, Ok)
    assert result.value.sse == SseState()


def test_strip_think_tag() -> None:
    content, think = strip_think_tag("<think>hmm</think>Body")
    assert content == "Body"
    assert think == "hmm"
    content, think = strip_think_tag("Body")
    assert (content, think) == ("Body", None)
    content, think = strip_think_tag(42)
    assert (content, think) == (42, None)


def test_plain_reply() -> None:
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello", "reasoning": " why"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.1},
    }
    result = plain_reply(body)
    assert isinstance(result, Ok)
    reply = result.value
    assert reply.content == "Hello"
    assert reply.reasoning == (" why",)
    assert reply.reported is not None and reply.reported["cost"] == 0.1


def test_plain_reply_bad_shape() -> None:
    result = plain_reply({"nope": True})
    assert isinstance(result, Err)
    assert "unexpected response shape" in describe(result.error)


def test_flatten_content_parts_with_thinking() -> None:
    content = [
        {"type": "thinking", "thinking": [{"text": " t1 "}]},
        {"type": "text", "text": "A"},
        "B",
    ]
    texts, thoughts = flatten_content_parts(content)
    assert texts == "AB"
    assert thoughts == (" t1 ",)


def test_drive_stream_progress_labels() -> None:
    events: list[tuple[str, int]] = []
    chunks = stream_chunks("<think>thought", "</think>answer")
    chunks.append({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1}})
    body = FakeStreamResponse(chunks)

    def on_progress(label: str, count: int) -> IO[None]:
        events.append((label, count))
        return io_pure(None)

    result = drive_stream(body, on_progress).run()
    assert isinstance(result, Ok)
    assert events == [("Thinking", 1), ("Thinking", 1)]


def test_drive_stream_stops_on_error_without_reading_next() -> None:
    pulled: list[bytes] = []

    def lines() -> Iterator[bytes]:
        chunk = b'data: {"error": {"message": "boom"}}\n'
        pulled.append(chunk)
        yield chunk
        blank = b"\n"
        pulled.append(blank)
        yield blank
        pulled.append(b"data: next\n")
        yield b"data: next\n"

    result = drive_stream(lines(), lambda label, count: io_pure(None)).run()
    assert isinstance(result, Err)
    assert "boom" in describe(result.error)
    assert len(pulled) == 2

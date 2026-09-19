import json
from collections.abc import Iterator

from fakes import FakeStreamResponse, stream_chunks

import zh2en as z


def feed(state: z.StreamState, *texts: str) -> z.StreamState:
    for text in texts:
        result = z.stream_step(state, {"choices": [{"delta": {"content": text}}]})
        assert isinstance(result, z.Ok)
        state = result.value
    return state


def test_think_step_holds_partial_prefix() -> None:
    state, thinking, visible = z.think_step(z.ThinkState(), "<th")
    assert visible == ""
    assert (state.checking, state.open) == (True, False)

    state, thinking, visible = z.think_step(state, "ink>hidden")
    assert visible == ""
    assert (state.checking, state.open) == (True, True)


def test_think_step_close_across_chunks() -> None:
    state, _, _ = z.think_step(z.ThinkState(), "<think>abc")
    state, thinking, visible = z.think_step(state, "def</thi")
    assert visible == ""
    assert state.open

    state, thinking, visible = z.think_step(state, "nk>out")
    assert (thinking, visible) == (True, "out")
    assert (state.checking, state.open, state.held) == (False, False, "")


def test_think_step_plain_text() -> None:
    state, thinking, visible = z.think_step(z.ThinkState(), "Hello")
    assert (thinking, visible) == (False, "Hello")


def test_think_step_leading_whitespace() -> None:
    state, thinking, visible = z.think_step(z.ThinkState(), "  ")
    assert visible == ""
    state, thinking, visible = z.think_step(state, "hi")
    assert visible == "  hi"


def test_think_step_bounds_whitespace_hold() -> None:
    state = z.ThinkState()
    for _ in range(10):
        state, _, visible = z.think_step(state, "        ")
        assert visible == ""

    assert len(state.held) <= len(z.THINK_CLOSE)
    state, _, visible = z.think_step(state, "hi")
    assert visible.endswith("hi")


def test_stream_step_filters_think_content() -> None:
    state = feed(z.StreamState(), "<think>secret ", "reasoning</think>", "Translation.")
    assert "".join(state.contents) == "Translation."


def test_stream_step_accumulates_reasoning_and_usage() -> None:
    state = z.StreamState()
    result = z.stream_step(
        state,
        {
            "choices": [{"delta": {"reasoning_content": "ponder"}}],
        },
    )
    assert isinstance(result, z.Ok)
    state = result.value
    assert state.reasoning == ("ponder",)
    assert state.progress_request == ("Thinking", 1)

    result = z.stream_step(
        state,
        {
            "choices": [{"delta": {"content": "Hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.5},
        },
    )
    assert isinstance(result, z.Ok)
    state = result.value
    assert state.contents == ("Hi",)
    assert state.reported == {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.5}
    assert state.counted == 2
    assert state.progress_request == ("Working", 1)


def test_stream_step_content_part_list() -> None:
    state = z.StreamState()
    result = z.stream_step(
        state,
        {"choices": [{"delta": {"content": [{"type": "text", "text": "ok"}]}}]},
    )
    assert isinstance(result, z.Ok)
    assert result.value.contents == ("ok",)


def test_step_stream_ignores_non_dict_chunks() -> None:
    state = feed(z.StreamState(), "keep")
    for raw in (
        b"data: [1,2,3]\n",
        b'data: "x"\n',
        b": keep-alive\n",
        b"data: [DONE]\n",
    ):
        result = z.step_stream(state, raw)
        assert isinstance(result, z.Ok)
        assert result.value is state


def test_step_stream_reports_endpoint_error() -> None:
    result = z.step_stream(
        z.StreamState(),
        b'data: {"error": {"message": "overloaded"}}\n',
    )
    assert isinstance(result, z.Err)
    assert "overloaded" in result.error


def test_parse_stream_line() -> None:
    payload = json.dumps({"choices": []}).encode("utf-8")
    assert z.parse_stream_line(b"data: " + payload + b"\n") == {"choices": []}
    assert z.parse_stream_line(b"data: [DONE]\n") is None
    assert z.parse_stream_line(b"data: not json\n") is None
    assert z.parse_stream_line(b"event: ping\n") is None
    assert z.parse_stream_line(b"data: []\n") is None


def test_strip_think_tag() -> None:
    content, think = z.strip_think_tag("<think>hmm</think>Body")
    assert content == "Body"
    assert think == "hmm"
    content, think = z.strip_think_tag("Body")
    assert (content, think) == ("Body", None)
    content, think = z.strip_think_tag(42)
    assert (content, think) == (42, None)


def test_plain_reply() -> None:
    body = {
        "choices": [
            {"message": {"role": "assistant", "content": "Hello", "reasoning": " why"}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "cost": 0.1},
    }
    result = z.plain_reply(body)
    assert isinstance(result, z.Ok)
    reply = result.value
    assert reply.content == "Hello"
    assert reply.reasoning == (" why",)
    assert reply.reported is not None and reply.reported["cost"] == 0.1


def test_plain_reply_bad_shape() -> None:
    result = z.plain_reply({"nope": True})
    assert isinstance(result, z.Err)
    assert "unexpected response shape" in result.error


def test_flatten_content_parts_with_thinking() -> None:
    content = [
        {"type": "thinking", "thinking": [{"text": " t1 "}]},
        {"type": "text", "text": "A"},
        "B",
    ]
    texts, thoughts = z.flatten_content_parts(content)
    assert texts == "AB"
    assert thoughts == (" t1 ",)


def test_drive_stream_progress_labels() -> None:
    events: list[tuple[str, int]] = []
    chunks = stream_chunks("<think>thought", "</think>answer")
    chunks.append({"choices": [{"delta": {}}], "usage": {"prompt_tokens": 1}})
    body = FakeStreamResponse(chunks)

    def on_progress(label: str, count: int) -> None:
        events.append((label, count))

    result = z.drive_stream(body, on_progress)
    assert isinstance(result, z.Ok)
    assert events == [("Thinking", 1), ("Thinking", 1)]


def test_drive_stream_stops_on_error_without_reading_next() -> None:
    pulled: list[bytes] = []

    def lines() -> Iterator[bytes]:
        chunk = b'data: {"error": {"message": "boom"}}\n'
        pulled.append(chunk)
        yield chunk
        pulled.append(b"data: next\n")
        yield b"data: next\n"

    result = z.drive_stream(lines(), lambda label, count: None)
    assert isinstance(result, z.Err)
    assert "boom" in result.error
    assert len(pulled) == 1

from itertools import chain

from hypothesis import assume, given
from hypothesis import strategies as st

from zh2en.text import (
    ThinkState,
    cache_key,
    estimate_tokens,
    make_chunks,
    split_to_budget,
    think_step,
)

PLAIN_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x9FFF),
    max_size=120,
)


def feed_think(body: str, chunk_size: int) -> str:
    state = ThinkState()
    parts: list[str] = []
    for start in range(0, len(body), chunk_size):
        state, _thinking, visible = think_step(state, body[start : start + chunk_size])
        parts.append(visible)

    tail = state.held if state.checking and not state.open else ""
    return "".join(parts) + tail


@given(PLAIN_TEXT)
def test_think_filter_passes_plain_body_char_by_char(body: str) -> None:
    assume("<think>" not in body and "</think>" not in body)
    assert feed_think(body, 1) == body


@given(st.text(alphabet=" \t\n", max_size=8), PLAIN_TEXT, PLAIN_TEXT)
def test_think_filter_removes_leading_block(
    leading: str, inside: str, suffix: str
) -> None:
    assume(
        "<think>" not in inside
        and "</think>" not in inside
        and "<think>" not in suffix
        and "</think>" not in suffix
    )
    body = leading + "<think>" + inside + "</think>" + suffix
    assert feed_think(body, 1) == suffix


@given(PLAIN_TEXT, PLAIN_TEXT)
def test_think_filter_keeps_mid_content_tags(head: str, tail: str) -> None:
    assume(head.strip() and "<think>" not in head and "</think>" not in head)
    body = head + "<think>" + tail
    assert feed_think(body, 1) == body


@given(
    st.lists(st.text(min_size=1, max_size=40), max_size=20),
    st.integers(min_value=1, max_value=60),
)
def test_make_chunks_preserves_paragraphs_and_respects_budget(
    paragraphs: list[str], budget: int
) -> None:
    chunks = make_chunks(tuple(paragraphs), budget)
    assert tuple(chain.from_iterable(chunks)) == tuple(paragraphs)
    for chunk in chunks:
        if len(chunk) > 1:
            assert sum(estimate_tokens(part) for part in chunk) <= budget


@given(
    st.lists(
        st.text(alphabet="一二三abc。！", min_size=1, max_size=12).map(
            lambda sentence: sentence + "。"
        ),
        max_size=12,
    ),
    st.integers(min_value=1, max_value=25),
)
def test_split_to_budget_preserves_text(sentences: list[str], budget: int) -> None:
    text = "".join(sentences)
    pieces = split_to_budget(text, budget, "。！")
    assert "".join(pieces) == text


@given(PLAIN_TEXT, PLAIN_TEXT)
def test_cache_key_is_deterministic_and_content_sensitive(
    chunk_text: str, model: str
) -> None:
    key = cache_key(chunk_text, model)
    assert key == cache_key(chunk_text, model)
    assert key != cache_key(chunk_text, model + "x")
    assert key != cache_key(chunk_text + "x", model)

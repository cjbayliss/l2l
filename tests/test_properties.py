import json
from itertools import chain, zip_longest
from typing import Any

from hypothesis import assume, given
from hypothesis import strategies as st

from zh2en.http import StreamState, step_stream
from zh2en.monads import Ok, cons_to_tuple
from zh2en.plans import plan_backoff
from zh2en.settings import PartialApiSettings
from zh2en.text import (
    ThinkState,
    cache_key,
    count_paragraphs,
    ensure_blank_line_separators,
    estimate_tokens,
    make_chunks,
    split_paragraphs,
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


# --- split_paragraphs -------------------------------------------------------
# Paragraphs exclude all whitespace characters: whitespace-only runs adjacent
# to blank lines are absorbed into separators by `split_paragraphs`, so they
# are outside the contract these properties cover.
PARAGRAPH = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x4DBF).filter(
        lambda char: not char.isspace()
    ),
    min_size=1,
    max_size=20,
)
SEPARATOR = st.sampled_from(["\n\n", "\n\n\n", "\n \n", "\n\t\n"])


@st.composite
def paragraph_documents(draw: st.DrawFn) -> tuple[str, tuple[str, ...]]:
    count = draw(st.integers(min_value=1, max_value=6))
    paragraphs = tuple(draw(PARAGRAPH) for _ in range(count))
    separators = [draw(SEPARATOR) for _ in range(count - 1)]
    pieces: list[str] = []
    for index, paragraph in enumerate(paragraphs):
        pieces.append(paragraph)
        if index < count - 1:
            pieces.append(separators[index])

    return "".join(pieces), paragraphs


@given(paragraph_documents())
def test_split_paragraphs_reconstructs_the_document(
    document: tuple[str, tuple[str, ...]],
) -> None:
    text, expected = document
    paragraphs, separators = split_paragraphs(text)
    rebuilt = "".join(
        part + sep for part, sep in zip_longest(paragraphs, separators, fillvalue="")
    )
    assert paragraphs == expected
    assert rebuilt == text


@given(paragraph_documents())
def test_ensure_blank_line_separators_preserves_paragraph_count(
    document: tuple[str, tuple[str, ...]],
) -> None:
    text, expected = document
    joined = ensure_blank_line_separators(text)
    assert count_paragraphs(joined) == len(expected)


# --- SSE stream fold -------------------------------------------------------
# Content excludes "<" (the <think> tag machinery legitimately consumes such
# prefixes) and all-whitespace deltas (a whitespace-only prefix is held back
# while checking for a tag, so raw contents differ until the held flush).
PLAIN_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x10FFFF).filter(
        lambda char: char != "<"
    ),
    max_size=12,
).filter(lambda text: not text.isspace())
CONTENT_CHUNKS = st.fixed_dictionaries(
    {
        "choices": st.lists(
            st.fixed_dictionaries(
                {"delta": st.fixed_dictionaries({"content": PLAIN_TEXT})}
            ),
            min_size=1,
            max_size=1,
        )
    }
)
USAGE_CHUNKS = st.fixed_dictionaries(
    {
        "usage": st.fixed_dictionaries(
            {
                "prompt_tokens": st.integers(min_value=0, max_value=999),
                "completion_tokens": st.integers(min_value=0, max_value=999),
            }
        )
    }
)
STREAM_CHUNKS = st.lists(
    st.one_of(CONTENT_CHUNKS, USAGE_CHUNKS, st.just({})),
    max_size=15,
)


def sse_bytes(chunks: list[dict[str, Any]]) -> list[bytes]:
    lines: list[bytes] = []
    for chunk in chunks:
        lines.append(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n")
        lines.append(b"\n")

    return lines


@given(STREAM_CHUNKS)
def test_stream_fold_concatenates_deltas_and_keeps_last_usage(
    chunks: list[dict[str, Any]],
) -> None:
    state: StreamState = StreamState()
    for line in sse_bytes(chunks):
        outcome = step_stream(state, line)
        assert isinstance(outcome, Ok)
        state = outcome.value

    expected = "".join(
        chunk["choices"][0]["delta"]["content"]
        for chunk in chunks
        if "choices" in chunk
    )
    assert "".join(cons_to_tuple(state.contents)) == expected

    usages = [chunk["usage"] for chunk in chunks if "usage" in chunk]
    if usages:
        assert state.reported is not None
        assert dict(state.reported) == usages[-1]
    else:
        assert state.reported is None


# --- PartialApiSettings.merge ----------------------------------------------
OPTIONAL_TEXT = st.one_of(st.none(), st.text(min_size=1, max_size=4))
OPTIONAL_NUMBER = st.one_of(
    st.none(), st.integers(min_value=1, max_value=99).map(float)
)


@st.composite
def partial_settings(draw: st.DrawFn) -> PartialApiSettings:
    return PartialApiSettings(
        base_url=draw(OPTIONAL_TEXT),
        api_key=draw(OPTIONAL_TEXT),
        model=draw(OPTIONAL_TEXT),
        timeout=draw(OPTIONAL_NUMBER),
        max_tokens=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=99))),
        params=draw(
            st.one_of(
                st.none(),
                st.dictionaries(
                    st.text(min_size=1, max_size=2, alphabet="xyz"),
                    st.integers(min_value=0, max_value=9),
                ),
            )
        ),
    )


@given(partial_settings(), partial_settings(), partial_settings())
def test_partial_merge_is_associative(
    first: PartialApiSettings,
    second: PartialApiSettings,
    third: PartialApiSettings,
) -> None:
    assert first.merge(second).merge(third) == first.merge(second.merge(third))


@given(partial_settings())
def test_partial_merge_has_identity(settings: PartialApiSettings) -> None:
    assert settings.merge(PartialApiSettings()) == settings


@given(partial_settings(), partial_settings())
def test_partial_merge_params_deep_merges(
    base: PartialApiSettings, extra: PartialApiSettings
) -> None:
    merged = base.merge(extra)
    if base.params is not None and extra.params is not None:
        assert merged.params == {**base.params, **extra.params}


# --- plan_backoff ----------------------------------------------------------
@given(
    st.floats(min_value=0.1, max_value=10.0),
    st.floats(min_value=0.1, max_value=50.0),
    st.integers(min_value=0, max_value=8),
)
def test_plan_backoff_is_monotone_and_capped(
    base: float, cap: float, attempts: int
) -> None:
    delays = plan_backoff(base, cap, attempts)
    assert len(delays) == attempts
    assert tuple(delays) == tuple(sorted(delays))
    assert all(delay <= cap for delay in delays)

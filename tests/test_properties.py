import json
from itertools import chain, zip_longest

from fakes import DONE_LINES
from hypothesis import assume, given
from hypothesis import strategies as st

from l2l.config.values import is_paragraph_tri_state
from l2l.glossary import is_entries_table
from l2l.http import StreamState, ingest_raw_line, plan_backoff
from l2l.json_values import (
    JsonObject,
    JsonValue,
    is_json_array,
    is_json_object,
    is_json_value,
    is_non_empty_string,
    is_non_empty_string_array,
    is_non_empty_table_array,
    is_object_list,
    is_object_table,
    is_string_array,
    is_string_keyed_table,
)
from l2l.monads import Ok, cons_to_tuple
from l2l.settings import UNSET, PartialApiSettings, Unset
from l2l.text import (
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

    tail = (
        state.buffered_text
        if state.scanning_opening_tag and not state.inside_reasoning
        else ""
    )
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
        part + separator
        for part, separator in zip_longest(paragraphs, separators, fillvalue="")
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


STREAM_TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=0x10FFFF).filter(
        lambda char: char != "<"
    ),
    max_size=12,
).filter(lambda text: not text.isspace())
EMPTY_CHUNK: JsonObject = {}
CONTENT_CHUNKS = st.fixed_dictionaries(
    {
        "choices": st.lists(
            st.fixed_dictionaries(
                {"delta": st.fixed_dictionaries({"content": STREAM_TEXT})}
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
    st.one_of(CONTENT_CHUNKS, USAGE_CHUNKS, st.just(EMPTY_CHUNK)),
    max_size=15,
)


def sse_bytes(chunks: list[JsonObject]) -> list[bytes]:
    lines: list[bytes] = []
    for chunk in chunks:
        lines.append(b"data: " + json.dumps(chunk).encode("utf-8") + b"\n")
        lines.append(b"\n")
    lines.extend(DONE_LINES)
    return lines


def chunk_content(chunk: object) -> str:
    if not is_json_object(chunk):
        return ""

    choices = chunk.get("choices")
    if not is_json_array(choices) or not choices:
        return ""

    first_choice = choices[0]
    if not is_json_object(first_choice):
        return ""

    delta = first_choice.get("delta")
    if not is_json_object(delta):
        return ""

    content = delta.get("content")
    return content if isinstance(content, str) else ""


@given(STREAM_CHUNKS)
def test_stream_fold_concatenates_deltas_and_keeps_last_usage(
    chunks: list[JsonObject],
) -> None:
    state: StreamState = StreamState()
    for line in sse_bytes(chunks):
        outcome = ingest_raw_line(state, line)
        assert isinstance(outcome, Ok)
        state = outcome.value

    expected = "".join(chunk_content(chunk) for chunk in chunks)
    assert "".join(cons_to_tuple(state.contents)) == expected

    usages = [chunk["usage"] for chunk in chunks if "usage" in chunk]
    if usages:
        assert state.reported_usage is not None
        assert dict(state.reported_usage) == usages[-1]
    else:
        assert state.reported_usage is None


OPTIONAL_TEXT = st.one_of(st.just(UNSET), st.text(min_size=1, max_size=4))
OPTIONAL_NUMBER = st.one_of(
    st.just(UNSET), st.integers(min_value=1, max_value=99).map(float)
)


@st.composite
def partial_settings(draw: st.DrawFn) -> PartialApiSettings:
    return PartialApiSettings(
        base_url=draw(OPTIONAL_TEXT),
        api_key=draw(OPTIONAL_TEXT),
        model=draw(OPTIONAL_TEXT),
        timeout=draw(OPTIONAL_NUMBER),
        maximum_tokens=draw(
            st.one_of(st.just(UNSET), st.integers(min_value=1, max_value=99))
        ),
        parameters=draw(
            st.one_of(
                st.just(UNSET),
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
    if not isinstance(base.parameters, Unset) and not isinstance(
        extra.parameters, Unset
    ):
        assert merged.parameters == {**base.parameters, **extra.parameters}


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


JSON_SCALARS: st.SearchStrategy[JsonValue] = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.text(),
)


def json_containers(
    children: st.SearchStrategy[JsonValue],
) -> st.SearchStrategy[JsonValue]:
    return st.lists(children) | st.dictionaries(st.text(), children)


JSON_VALUES: st.SearchStrategy[JsonValue] = st.recursive(
    JSON_SCALARS,
    json_containers,
    max_leaves=20,
)

NON_JSON_VALUES: st.SearchStrategy[object] = st.one_of(
    st.binary(),
    st.complex_numbers(),
    st.sets(st.integers()),
    st.tuples(st.integers()),
)

WHITESPACE_TEXT: st.SearchStrategy[str] = st.text(
    alphabet=" \t\n\r\x0b\x0c", min_size=1
)

NON_STRING_JSON_VALUES: st.SearchStrategy[object] = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False),
    st.lists(JSON_VALUES),
    st.dictionaries(st.text(), JSON_VALUES),
)


@given(JSON_VALUES)
def test_json_value_guard_accepts_every_decodable_value(value: JsonValue) -> None:
    decoded: JsonValue = json.loads(json.dumps(value))
    assert is_json_value(value)
    assert is_json_value(decoded)


@given(JSON_VALUES)
def test_table_and_array_guards_track_decoded_structure(value: JsonValue) -> None:
    decoded: JsonValue = json.loads(json.dumps(value))
    if isinstance(decoded, dict):
        assert is_object_table(decoded)
        assert is_string_keyed_table(decoded)
        assert is_json_object(decoded)

    if isinstance(decoded, list):
        assert is_object_list(decoded)
        assert is_json_array(decoded)


@given(NON_JSON_VALUES)
def test_json_guards_reject_values_outside_json(value: object) -> None:
    assert not is_json_value(value)
    assert not is_json_object(value)
    assert not is_json_array(value)


@given(
    st.one_of(
        JSON_SCALARS,
        st.lists(JSON_VALUES),
        st.dictionaries(st.integers(), JSON_VALUES, min_size=1),
        st.dictionaries(st.text(), NON_JSON_VALUES, min_size=1),
    )
)
def test_json_object_guard_rejects_impure_and_non_string_keyed_tables(
    value: object,
) -> None:
    assert not is_json_object(value)


@given(
    st.one_of(
        JSON_SCALARS,
        st.dictionaries(st.text(), JSON_VALUES),
        st.lists(NON_JSON_VALUES, min_size=1),
    )
)
def test_json_array_guard_rejects_non_arrays_and_impure_arrays(
    value: object,
) -> None:
    assert not is_json_array(value)


@given(
    st.one_of(
        JSON_SCALARS,
        st.lists(JSON_VALUES),
        st.dictionaries(st.integers(), JSON_VALUES, min_size=1),
    )
)
def test_string_keyed_table_guard_rejects_non_string_keys(value: object) -> None:
    assert not is_string_keyed_table(value)


@given(st.one_of(JSON_SCALARS, st.lists(JSON_VALUES)))
def test_object_table_guard_rejects_non_mappings(value: object) -> None:
    assert not is_object_table(value)


@given(st.one_of(JSON_SCALARS, st.dictionaries(st.text(), JSON_VALUES)))
def test_object_list_guard_rejects_non_lists(value: object) -> None:
    assert not is_object_list(value)


@given(st.text())
def test_non_empty_string_guard_matches_stripped_content(value: str) -> None:
    assert is_non_empty_string(value) == bool(value.strip())


@given(
    st.one_of(
        NON_STRING_JSON_VALUES,
        WHITESPACE_TEXT,
    )
)
def test_non_empty_string_guard_rejects_non_strings_and_blanks(
    value: object,
) -> None:
    assert not is_non_empty_string(value)


@given(st.lists(st.text()))
def test_string_array_guards_match_their_elements(values: list[str]) -> None:
    assert is_string_array(values)
    assert is_non_empty_string_array(values) == all(
        bool(value.strip()) for value in values
    )


@given(
    st.one_of(
        JSON_SCALARS,
        st.dictionaries(st.text(), JSON_VALUES),
        st.lists(st.one_of(st.integers(), st.none()), min_size=1),
    )
)
def test_string_array_guards_reject_non_string_elements(value: object) -> None:
    assert not is_string_array(value)
    assert not is_non_empty_string_array(value)


@given(st.lists(WHITESPACE_TEXT, min_size=1))
def test_non_empty_string_array_guard_rejects_blank_elements(
    values: list[str],
) -> None:
    assert is_string_array(values)
    assert not is_non_empty_string_array(values)


@given(st.lists(st.dictionaries(st.text(), JSON_VALUES)))
def test_table_array_guard_accepts_exactly_the_non_empty_tables(
    tables: list[dict[str, JsonValue]],
) -> None:
    assert is_non_empty_table_array(tables) == bool(tables)


@given(st.lists(JSON_SCALARS, min_size=1))
def test_table_array_guard_rejects_non_table_elements(tables: list[JsonValue]) -> None:
    assert not is_non_empty_table_array(tables)


def test_table_array_guard_rejects_non_list_containers() -> None:
    assert not is_non_empty_table_array({"a": 1})
    assert not is_non_empty_table_array("tables")
    assert not is_non_empty_table_array(None)


@given(st.one_of(st.booleans(), st.integers(min_value=1)))
def test_paragraph_tri_state_guard_accepts_flags_and_positive_counts(
    value: bool | int,
) -> None:
    assert is_paragraph_tri_state(value)


@given(
    st.one_of(
        st.integers(max_value=0),
        st.floats(allow_nan=False),
        st.text(),
        st.none(),
        st.lists(st.integers()),
    )
)
def test_paragraph_tri_state_guard_rejects_everything_else(value: object) -> None:
    assert not is_paragraph_tri_state(value)


@given(st.dictionaries(st.text(), JSON_VALUES), JSON_VALUES)
def test_entries_table_guard_accepts_tables_with_entries(
    table: dict[str, JsonValue], entries: JsonValue
) -> None:
    assert is_entries_table({**table, "entries": entries})


@given(
    st.one_of(
        JSON_SCALARS,
        st.lists(JSON_VALUES),
        st.dictionaries(
            st.text().filter(lambda key: key != "entries"),
            JSON_VALUES,
            min_size=1,
        ),
    )
)
def test_entries_table_guard_rejects_tables_without_entries(
    value: object,
) -> None:
    assert not is_entries_table(value)

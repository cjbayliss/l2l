import json
from typing import Any

import pytest

from l2l.errors import describe
from l2l.glossary import (
    GlossaryEntry,
    append_glossary_entries,
    entries_to_json_text,
    extraction_user_message,
    instruction_with_glossary,
    merge_glossary_entries,
    parse_glossary_document,
    parse_model_glossary_reply,
    render_glossary_block,
)
from l2l.monads import Err, Ok, Result


def entry(original: str, **attributes: str) -> GlossaryEntry:
    result: Result[tuple[GlossaryEntry, ...], Any] = parse_glossary_document(
        "g.json", [{"original": original, **attributes}]
    )
    assert isinstance(result, Ok)
    return result.value[0]


def test_parse_valid_entries_preserves_attribute_order() -> None:
    result = parse_glossary_document(
        "g.json",
        [
            {
                "original": "林黛玉",
                "translated": "Lin Daiyu",
                "pronouns": "she/her",
            },
            {"original": "拳法", "pronouns": "it", "style": "internal martial art"},
        ],
    )
    assert isinstance(result, Ok)
    first, second = result.value
    assert first.original == "林黛玉"
    assert tuple(first.attributes.items()) == (
        ("translated", "Lin Daiyu"),
        ("pronouns", "she/her"),
    )
    assert tuple(second.attributes.items()) == (
        ("pronouns", "it"),
        ("style", "internal martial art"),
    )
    assert parse_glossary_document("g.json", []) == Ok(())


def test_null_attributes_are_treated_as_unset() -> None:
    result = parse_glossary_document(
        "g.json",
        [{"original": "南京", "translated": None, "location": "city"}],
    )
    assert isinstance(result, Ok)
    (parsed,) = result.value
    assert tuple(parsed.attributes.items()) == (("location", "city"),)


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        ({"original": "x"}, "glossary must be a JSON array"),
        (["x"], "glossary entry 1 must be an object"),
        ([{"translated": "x"}], 'needs a non-empty "original"'),
        ([{"original": "x"}], "needs at least one attribute"),
        ([{"original": "x", "translated": None}], "needs at least one attribute"),
        ([{"original": "x", "translated": ""}], "invalid value(s) for: translated"),
        ([{"original": "x", "translated": 3}], "invalid value(s) for: translated"),
    ],
)
def test_parse_glossary_document_rejects_invalid_documents(
    document: object, fragment: str
) -> None:
    result = parse_glossary_document("g.json", document)
    assert isinstance(result, Err)
    assert fragment in describe(result.error)


def test_parse_glossary_document_drops_later_duplicate_entries() -> None:
    result = parse_glossary_document(
        "g.json",
        [
            {"original": "南京", "translated": "Nanjing"},
            {"original": "甲", "translated": "Alpha"},
            {"original": "南京", "translated": "Nanking"},
        ],
    )
    assert isinstance(result, Ok)
    assert [item.original for item in result.value] == ["南京", "甲"]
    assert tuple(result.value[0].attributes.items()) == (("translated", "Nanjing"),)


def test_merge_later_entry_replaces_earlier_wholesale() -> None:
    merged = merge_glossary_entries(
        (
            (
                entry("甲", translated="Alpha", pronouns="she/her"),
                entry("乙", translated="Bee"),
            ),
            (entry("甲", translated="Aleph"),),
        )
    )
    assert [item.original for item in merged] == ["甲", "乙"]
    assert tuple(merged[0].attributes.items()) == (("translated", "Aleph"),)
    assert tuple(merged[1].attributes.items()) == (("translated", "Bee"),)


def test_render_glossary_block_and_instruction_integration() -> None:
    block = render_glossary_block(
        (
            entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
            entry("南京", translated="Nanjing"),
            entry("100", translated="one hundred"),
            entry("拳法", pronouns="it", style="internal martial art"),
        )
    )
    assert block.startswith("Glossary (mandatory):")
    assert '- 林黛玉 = "Lin Daiyu" (pronouns: she/her)' in block
    assert '- 南京 = "Nanjing"' in block
    assert '- 100 = "one hundred"' in block
    assert "- 拳法 (pronouns: it) (style: internal martial art)" in block
    assert instruction_with_glossary("Translate.", ()) == "Translate."
    combined = instruction_with_glossary(
        "Translate.", (entry("南京", translated="Nanjing"),)
    )
    assert combined.startswith("Translate.\n\nGlossary (mandatory):")
    assert '- 南京 = "Nanjing"' in combined


@pytest.mark.parametrize(
    "reply",
    [
        '```json\n[{"original": "南京", "translated": "Nanjing"}]\n```',
        ' thinkingplanning</think>\n[{"original": "南京", "translated": "Nanjing"}]',
        (
            "The source text contains no new names.\n\n"
            '[{"original": "南京", "translated": "Nanjing"}]\n\n'
            "All decided entries are carried over."
        ),
    ],
)
def test_parse_model_glossary_reply_accepts_json_around_noise(reply: str) -> None:
    result = parse_model_glossary_reply(reply, "out.json")
    assert isinstance(result, Ok)
    assert result.value[0].original == "南京"


@pytest.mark.parametrize(
    ("reply", "fragment"),
    [
        ("no names here", "model reply is not valid JSON"),
        ("not json", "model reply is not valid JSON"),
        ('[{"original": ', "model reply is not valid JSON"),
        ("[123]", "glossary entry 1 must be an object"),
    ],
)
def test_parse_model_glossary_reply_rejects_invalid_replies(
    reply: str, fragment: str
) -> None:
    result = parse_model_glossary_reply(reply, "out.json")
    assert isinstance(result, Err)
    assert fragment in describe(result.error)


def test_entries_to_json_text_round_trips() -> None:
    entries = (
        entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
        entry("拳法", pronouns="it", style="internal martial art"),
    )
    text = entries_to_json_text(entries)
    assert text.endswith("\n")
    assert parse_glossary_document("g.json", json.loads(text)) == Ok(entries)


def test_extraction_user_message_includes_optional_carried_entries() -> None:
    message = extraction_user_message(
        "French", (entry("南京", translated="Nanjing"),), "hello"
    )
    assert message.startswith("Target language: French")
    assert "Already decided entries" in message
    assert '"original": "南京"' in message
    assert message.endswith("Source text:\nhello")
    assert (
        extraction_user_message("English", (), "hello")
        == "Target language: English\n\nSource text:\nhello"
    )


def test_append_glossary_entries_preserves_existing_entries() -> None:
    existing = (
        entry("南京", translated="Nanjing"),
        entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
    )
    extracted = (
        entry("拳法", pronouns="it", style="internal martial art"),
        entry("大观园", translated="Grand View Garden"),
        entry("南京", translated="Nanking", note="variant"),
    )
    merged = append_glossary_entries(existing, extracted)
    assert [item.original for item in merged] == [
        "南京",
        "林黛玉",
        "拳法",
        "大观园",
    ]
    assert tuple(merged[0].attributes.items()) == (("translated", "Nanjing"),)
    assert tuple(merged[2].attributes.items()) == (
        ("pronouns", "it"),
        ("style", "internal martial art"),
    )
    assert append_glossary_entries(existing, ()) == existing

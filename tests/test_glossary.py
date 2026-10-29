import json
from typing import Any

import pytest

from l2l.errors import GlossaryReplyError, describe
from l2l.glossary import (
    GLOSSARY_ENTRY_TYPES,
    GlossaryEntry,
    entries_to_json_text,
    extraction_user_message,
    glossary_response_format,
    instruction_with_glossary,
    integrate_glossary_reply,
    known_originals_component,
    merge_glossary_entries,
    parse_glossary_document,
    parse_model_glossary_reply,
    render_glossary_block,
)
from l2l.monads import Err, Ok


def entry(
    original: str,
    translated: str | None = None,
    pronouns: str | None = None,
    entry_type: str = "other",
) -> GlossaryEntry:
    document: dict[str, Any] = {"original": original, "type": entry_type}
    if translated is not None:
        document["translated"] = translated
    if pronouns is not None:
        document["pronouns"] = pronouns
    result = parse_glossary_document("g.json", [document])
    assert isinstance(result, Ok)
    return result.value[0]


def test_parse_glossary_document_normalizes_entries() -> None:
    result = parse_glossary_document(
        "g.json",
        [
            {
                "original": "林黛玉",
                "translated": "Lin Daiyu",
                "pronouns": "she/her",
                "type": "character",
                "note": "protagonist",
                "extra": 3,
            },
            {"original": "金陵", "translated": "Nanjing", "type": "place"},
            {"original": "拳法"},
            {
                "original": "南京",
                "translated": "   ",
                "pronouns": "",
                "type": "other",
            },
        ],
    )
    assert isinstance(result, Ok)
    assert result.value == (
        GlossaryEntry(
            original="林黛玉",
            translated="Lin Daiyu",
            pronouns="she/her",
            type="character",
        ),
        GlossaryEntry(
            original="金陵", translated="Nanjing", pronouns=None, type="place"
        ),
        GlossaryEntry(original="拳法", translated=None, pronouns=None, type="other"),
        GlossaryEntry(original="南京", translated=None, pronouns=None, type="other"),
    )
    assert parse_glossary_document("g.json", []) == Ok(())


@pytest.mark.parametrize(
    ("document", "fragment"),
    [
        ({"original": "x"}, "glossary must be a JSON array"),
        (["x"], "glossary entry 1 must be an object"),
        ([{"translated": "x"}], 'needs a non-empty "original"'),
        ([{"original": "x", "type": "vehicle"}], "has invalid type 'vehicle'"),
        ([{"original": "x", "type": ""}], "has invalid type ''"),
        ([{"original": "x", "translated": 3}], "invalid value(s) for: translated"),
        ([{"original": "x", "pronouns": 3}], "invalid value(s) for: pronouns"),
    ],
)
def test_parse_glossary_document_rejects_invalid_documents(
    document: object, fragment: str
) -> None:
    result = parse_glossary_document("g.json", document)
    assert isinstance(result, Err)
    assert fragment in describe(result.error)


def test_parse_glossary_document_errors_carry_the_file_path() -> None:
    result = parse_glossary_document("book.json", [{"original": 3}])
    assert isinstance(result, Err)
    assert describe(result.error).startswith("l2l: book.json:")


def test_parse_glossary_document_drops_later_duplicate_entries() -> None:
    result = parse_glossary_document(
        "g.json",
        [
            {"original": "南京", "translated": "Nanjing", "type": "place"},
            {"original": "甲", "translated": "Alpha", "type": "other"},
            {"original": "南京", "translated": "Nanking", "type": "place"},
        ],
    )
    assert isinstance(result, Ok)
    assert [item.original for item in result.value] == ["南京", "甲"]
    assert result.value[0].translated == "Nanjing"


def test_merge_later_entry_replaces_earlier_wholesale() -> None:
    merged = merge_glossary_entries(
        (
            (entry("甲", translated="Alpha", pronouns="she/her"), entry("乙", "Bee")),
            (entry("甲", translated="Aleph"),),
        )
    )
    assert [item.original for item in merged] == ["甲", "乙"]
    assert merged[0].translated == "Aleph"
    assert merged[1].translated == "Bee"


def test_render_glossary_block_and_instruction_integration() -> None:
    block = render_glossary_block(
        (
            entry("林黛玉", "Lin Daiyu", "she/her", "character"),
            entry("南京", "Nanjing", entry_type="place"),
            entry("100", "one hundred"),
        )
    )
    assert block.startswith("Glossary (mandatory):")
    assert '- 林黛玉 = "Lin Daiyu" (pronouns: she/her) (type: character)' in block
    assert '- 南京 = "Nanjing" (type: place)' in block
    assert '- 100 = "one hundred"' in block
    assert instruction_with_glossary("Translate.", ()) == "Translate."
    combined = instruction_with_glossary(
        "Translate.", (entry("南京", "Nanjing", entry_type="place"),)
    )
    assert combined.startswith("Translate.\n\nGlossary (mandatory):")
    assert '- 南京 = "Nanjing" (type: place)' in combined


def test_parse_model_glossary_reply_accepts_the_wrapper_object() -> None:
    reply = (
        '```json\n{"entries": [{"original": "南京", "translated": "Nanjing", '
        '"pronouns": null, "type": "place"}]}\n```'
    )
    result = parse_model_glossary_reply(reply, "out.json")
    assert isinstance(result, Ok)
    assert result.value == (entry("南京", "Nanjing", entry_type="place"),)


def test_parse_model_glossary_reply_accepts_a_bare_array() -> None:
    result = parse_model_glossary_reply(
        '[{"original": "南京", "translated": "Nanjing", "type": "place"}]', "out.json"
    )
    assert isinstance(result, Ok)
    assert result.value == (entry("南京", "Nanjing", entry_type="place"),)


def test_parse_model_glossary_reply_rejects_a_non_array_wrapper() -> None:
    result = parse_model_glossary_reply('{"entries": 3}', "out.json")
    assert isinstance(result, Err)
    assert "entries" in describe(result.error)


@pytest.mark.parametrize(
    ("reply", "fragment"),
    [
        ("no names here", "model reply is not valid JSON"),
        ('[{"original": ', "model reply is not valid JSON"),
        ("[123]", "glossary entry 1 must be an object"),
        ('[{"original": "x", "type": "vehicle"}]', "has invalid type 'vehicle'"),
    ],
)
def test_parse_model_glossary_reply_rejects_invalid_replies(
    reply: str, fragment: str
) -> None:
    result = parse_model_glossary_reply(reply, "out.json")
    assert isinstance(result, Err)
    assert fragment in describe(result.error)


def test_reply_errors_are_glossary_reply_errors() -> None:
    result = parse_model_glossary_reply("not json", "out.json")
    assert isinstance(result, Err)
    assert isinstance(result.error, GlossaryReplyError)
    assert result.error.pass_name == "out.json"


def test_entries_to_json_text_round_trips() -> None:
    entries = (
        entry("林黛玉", "Lin Daiyu", "she/her", "character"),
        entry("南京", "Nanjing", entry_type="place"),
        entry("拳法", pronouns="it"),
    )
    text = entries_to_json_text(entries)
    assert text.endswith("\n")
    assert parse_glossary_document("g.json", json.loads(text)) == Ok(entries)


def test_known_originals_component_lists_sorted_names() -> None:
    component = known_originals_component((entry("乙"), entry("南京"), entry("甲")))
    expected = "Already known names (do not output entries for these):\n乙\n南京\n甲\n"
    assert component == expected
    assert known_originals_component(()) == ""


def test_extraction_user_message_carries_known_originals_and_source() -> None:
    message = extraction_user_message(
        "French", (entry("南京", "Nanjing", entry_type="place"),), "hello"
    )
    assert message.startswith("Target language: French")
    assert "Already known names" in message
    assert "南京" in message
    assert message.endswith("Source text:\nhello")
    assert (
        extraction_user_message("English", (), "hello")
        == "Target language: English\n\nSource text:\nhello"
    )


def test_integrate_glossary_reply_appends_new_entries() -> None:
    existing = (
        entry("南京", "Nanjing", entry_type="place"),
        entry("林黛玉", "Lin Daiyu", "she/her", "character"),
    )
    extracted = (
        entry("拳法", pronouns="it"),
        entry("大观园", "Grand View Garden", entry_type="place"),
        entry("南京", "Nanking", entry_type="place"),
    )
    merged = integrate_glossary_reply(existing, extracted)
    assert [item.original for item in merged] == [
        "南京",
        "林黛玉",
        "拳法",
        "大观园",
    ]
    assert merged[0].translated == "Nanjing"
    assert integrate_glossary_reply(existing, ()) == existing


def test_integrate_glossary_reply_filters_by_type() -> None:
    existing = (entry("南京", "Nanjing", entry_type="place"),)
    extracted = (
        entry("林黛玉", "Lin Daiyu", "she/her", "character"),
        entry("大观园", "Grand View Garden", entry_type="place"),
        entry("拳法", pronouns="it"),
    )
    merged = integrate_glossary_reply(existing, extracted, ("character",))
    assert [item.original for item in merged] == ["南京", "林黛玉"]
    assert integrate_glossary_reply(existing, extracted, ()) == existing + tuple(
        extracted
    )


def glossary_entry_schema() -> dict[str, Any]:
    response_format = glossary_response_format()
    assert isinstance(response_format, dict)
    entry_schema = response_format["json_schema"]["schema"]["properties"]["entries"][
        "items"
    ]
    assert isinstance(entry_schema, dict)
    return entry_schema


def test_glossary_response_format_is_strict_mode_compliant() -> None:
    entry_schema = glossary_entry_schema()
    assert entry_schema["additionalProperties"] is False
    assert sorted(entry_schema["required"]) == [
        "original",
        "pronouns",
        "translated",
        "type",
    ]
    assert entry_schema["properties"]["type"]["enum"] == list(GLOSSARY_ENTRY_TYPES)
    assert entry_schema["properties"]["translated"]["type"] == ["string", "null"]
    assert entry_schema["properties"]["pronouns"]["type"] == ["string", "null"]

    response_format = glossary_response_format()
    assert response_format is not None
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "glossary_entries"
    assert response_format["json_schema"]["strict"] is True
    reply_schema = response_format["json_schema"]["schema"]
    assert reply_schema["type"] == "object"
    assert reply_schema["required"] == ["entries"]
    assert reply_schema["properties"]["entries"]["items"] == entry_schema
    assert glossary_response_format(True) is None

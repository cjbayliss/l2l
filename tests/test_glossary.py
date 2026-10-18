import json
from typing import Any

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


def test_null_attributes_are_treated_as_unset() -> None:
    result = parse_glossary_document(
        "g.json",
        [{"original": "南京", "translated": None, "location": "city"}],
    )
    assert isinstance(result, Ok)
    (entry,) = result.value
    assert tuple(entry.attributes.items()) == (("location", "city"),)


def test_empty_array_is_valid() -> None:
    result = parse_glossary_document("g.json", [])
    assert result == Ok(())


def test_document_must_be_an_array() -> None:
    result = parse_glossary_document("g.json", {"original": "x"})
    assert isinstance(result, Err)
    assert "g.json: glossary must be a JSON array of entries" in describe(result.error)


def test_entry_must_be_an_object() -> None:
    result = parse_glossary_document("g.json", ["x"])
    assert isinstance(result, Err)
    assert "g.json: glossary entry 1 must be an object" in describe(result.error)


def test_missing_original_is_rejected() -> None:
    result = parse_glossary_document("g.json", [{"translated": "x"}])
    assert isinstance(result, Err)
    assert 'needs a non-empty "original"' in describe(result.error)


def test_entry_without_usable_attribute_is_rejected() -> None:
    for document in (
        [{"original": "x"}],
        [{"original": "x", "translated": None}],
    ):
        result = parse_glossary_document("g.json", document)
        assert isinstance(result, Err)
        assert "needs at least one attribute" in describe(result.error)


def test_empty_string_attribute_is_rejected() -> None:
    result = parse_glossary_document("g.json", [{"original": "x", "translated": ""}])
    assert isinstance(result, Err)
    assert "invalid value(s) for: translated" in describe(result.error)


def test_non_string_non_null_attribute_is_rejected() -> None:
    result = parse_glossary_document("g.json", [{"original": "x", "translated": 3}])
    assert isinstance(result, Err)
    assert "invalid value(s) for: translated" in describe(result.error)


def test_duplicate_original_within_one_file_is_rejected() -> None:
    result = parse_glossary_document(
        "g.json",
        [
            {"original": "南京", "translated": "Nanjing"},
            {"original": "南京", "translated": "Nanking"},
        ],
    )
    assert isinstance(result, Err)
    assert 'duplicate glossary "original" value(s): 南京' in describe(result.error)


def entry(original: str, **attributes: str) -> GlossaryEntry:
    result: Result[tuple[GlossaryEntry, ...], Any] = parse_glossary_document(
        "g.json", [{"original": original, **attributes}]
    )
    assert isinstance(result, Ok)
    return result.value[0]


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


def test_render_glossary_block_covers_attribute_shapes() -> None:
    block = render_glossary_block(
        (
            entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
            entry("南京", translated="Nanjing"),
            entry("100", translated="one hundred"),
            entry("拳法", pronouns="it", style="internal martial art"),
        )
    )
    assert block.splitlines() == [
        "Glossary (mandatory): use the exact renderings and attributes below on "
        "every occurrence; never vary a rendering, never assign pronouns to "
        "entries that carry none, and keep each character's pronouns consistent "
        "even where the source omits subjects:",
        '- 林黛玉 = "Lin Daiyu" (pronouns: she/her)',
        '- 南京 = "Nanjing"',
        '- 100 = "one hundred"',
        "- 拳法 (pronouns: it) (style: internal martial art)",
    ]


def test_instruction_with_glossary_is_a_no_op_without_entries() -> None:
    assert instruction_with_glossary("Translate.", ()) == "Translate."


def test_instruction_with_glossary_appends_block() -> None:
    combined = instruction_with_glossary(
        "Translate.", (entry("南京", translated="Nanjing"),)
    )
    assert combined.startswith("Translate.\n\nGlossary (mandatory):")
    assert '- 南京 = "Nanjing"' in combined


def test_parse_model_glossary_reply_accepts_markdown_fences() -> None:
    reply = '```json\n[{"original": "南京", "translated": "Nanjing"}]\n```'
    result = parse_model_glossary_reply(reply, "out.json")
    assert isinstance(result, Ok)
    assert result.value[0].original == "南京"


def test_parse_model_glossary_reply_strips_think_tags() -> None:
    reply = '<think>planning</think>\n[{"original": "南京", "translated": "Nanjing"}]'
    result = parse_model_glossary_reply(reply, "out.json")
    assert isinstance(result, Ok)
    assert result.value[0].original == "南京"


def test_parse_model_glossary_reply_rejects_invalid_json() -> None:
    result = parse_model_glossary_reply("not json", "out.json")
    assert isinstance(result, Err)
    assert "out.json: model reply is not valid JSON" in describe(result.error)


def test_parse_model_glossary_reply_revalidates_entries() -> None:
    result = parse_model_glossary_reply("[123]", "out.json")
    assert isinstance(result, Err)
    assert "out.json: glossary entry 1 must be an object" in describe(result.error)


def test_entries_to_json_text_round_trips() -> None:
    entries = (
        entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
        entry("拳法", pronouns="it", style="internal martial art"),
    )
    text = entries_to_json_text(entries)
    assert text.endswith("\n")
    assert parse_glossary_document("g.json", json.loads(text)) == Ok(entries)


def test_extraction_user_message_combines_language_carried_and_text() -> None:
    message = extraction_user_message(
        "French", (entry("南京", translated="Nanjing"),), "hello"
    )
    assert message.startswith("Target language: French\n\nAlready decided entries")
    assert '"original": "南京"' in message
    assert message.endswith("Source text:\nhello")


def test_extraction_user_message_omits_carried_section_when_empty() -> None:
    message = extraction_user_message("English", (), "hello")
    assert message == "Target language: English\n\nSource text:\nhello"


def test_append_glossary_entries_appends_new_originals_in_order() -> None:
    existing = (
        entry("南京", translated="Nanjing"),
        entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
    )
    extracted = (
        entry("拳法", pronouns="it", style="internal martial art"),
        entry("大观园", translated="Grand View Garden"),
    )
    merged = append_glossary_entries(existing, extracted)
    assert [item.original for item in merged] == ["南京", "林黛玉", "拳法", "大观园"]
    assert tuple(merged[2].attributes.items()) == (
        ("pronouns", "it"),
        ("style", "internal martial art"),
    )


def test_append_glossary_entries_keeps_existing_entries_on_conflicts() -> None:
    existing = (
        entry("南京", translated="Nanjing"),
        entry("林黛玉", translated="Lin Daiyu", pronouns="she/her"),
    )
    extracted = (
        entry("南京", translated="Nanking", note="variant"),
        entry("林黛玉", translated="Daiyu"),
        entry("贾宝玉", translated="Jia Baoyu"),
    )
    merged = append_glossary_entries(existing, extracted)
    assert [item.original for item in merged] == ["南京", "林黛玉", "贾宝玉"]
    assert tuple(merged[0].attributes.items()) == (("translated", "Nanjing"),)
    assert tuple(merged[1].attributes.items()) == (
        ("translated", "Lin Daiyu"),
        ("pronouns", "she/her"),
    )


def test_append_glossary_entries_without_extracts_returns_existing() -> None:
    existing = (entry("南京", translated="Nanjing"),)
    assert append_glossary_entries(existing, ()) == existing

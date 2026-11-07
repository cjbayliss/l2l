from __future__ import annotations

import json
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Final, Literal, TypeIs, final, get_args

from l2l.errors import (
    ConfigError,
    GlossaryReplyError,
    TranslationError,
)
from l2l.json_values import (
    JsonValue,
    is_json_array,
    is_json_object,
    is_non_empty_string,
    is_object_list,
    is_string_keyed_table,
)
from l2l.monads import (
    Err,
    Ok,
    Result,
    result_bind,
    result_map,
    result_map_error,
    result_sequence,
)
from l2l.text import strip_think_tag

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

GlossaryEntryType = Literal["character", "place", "other"]

GLOSSARY_ENTRY_TYPES: Final[tuple[GlossaryEntryType, ...]] = get_args(GlossaryEntryType)


@final
@dataclass(frozen=True, slots=True)
class GlossaryEntry:
    original: str
    translated: str | None
    pronouns: str | None
    type: GlossaryEntryType
    notes: str | None = None


GLOSSARY_BLOCK_HEADER: Final = (
    "Glossary (mandatory): use the exact renderings and attributes below on "
    "every occurrence; never vary a rendering, never assign pronouns to "
    "entries that carry none, and keep each character's pronouns consistent "
    "even where the source omits subjects:"
)


def render_glossary_entry_line(entry: GlossaryEntry) -> str:
    return "- {}{}{}{}{}".format(
        entry.original,
        (f' = "{entry.translated}"') if entry.translated is not None else "",
        (f" (pronouns: {entry.pronouns})") if entry.pronouns is not None else "",
        (f" (type: {entry.type})") if entry.type != "other" else "",
        (f" (notes: {entry.notes})") if entry.notes is not None else "",
    )


def render_glossary_block(entries: tuple[GlossaryEntry, ...]) -> str:
    return "\n".join(
        (
            GLOSSARY_BLOCK_HEADER,
            *tuple(render_glossary_entry_line(entry) for entry in entries),
        )
    )


def instruction_with_glossary(
    instruction: str, entries: tuple[GlossaryEntry, ...]
) -> str:
    if not entries:
        return instruction

    return instruction + "\n\n" + render_glossary_block(entries)


def longest_glossary_match_at(
    ordered_entries: tuple[GlossaryEntry, ...], text: str, position: int
) -> GlossaryEntry | None:
    return next(
        (
            entry
            for entry in ordered_entries
            if text.startswith(entry.original, position)
        ),
        None,
    )


def longest_match_glossary_entries(
    entries: tuple[GlossaryEntry, ...], text: str
) -> frozenset[GlossaryEntry]:
    ordered_entries = tuple(
        sorted(
            (entry for entry in entries if entry.original),
            key=lambda entry: len(entry.original),
            reverse=True,
        )
    )

    def advance(
        state: tuple[int, frozenset[GlossaryEntry]], _: int
    ) -> tuple[int, frozenset[GlossaryEntry]]:
        position, matched = state
        if position >= len(text):
            return state

        entry = longest_glossary_match_at(ordered_entries, text, position)
        if entry is None:
            return (position + 1, matched)

        return (position + len(entry.original), matched | {entry})

    _, matched = reduce(advance, range(len(text)), (0, frozenset[GlossaryEntry]()))
    return matched


def select_glossary_entries_for_texts(
    entries: tuple[GlossaryEntry, ...], texts: tuple[str, ...]
) -> tuple[GlossaryEntry, ...]:
    matched = frozenset(
        entry
        for text in texts
        for entry in longest_match_glossary_entries(entries, text)
    )
    return tuple(entry for entry in entries if entry in matched)


def untranslated_glossary_entries(
    entries: tuple[GlossaryEntry, ...], output: str
) -> tuple[GlossaryEntry, ...]:
    return tuple(
        entry
        for entry in entries
        if entry.translated is not None
        and entry.translated != entry.original
        and entry.original in output
    )


def text_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def glossary_entry_type_from_value(value: object) -> GlossaryEntryType | None:
    return next(
        (entry_type for entry_type in GLOSSARY_ENTRY_TYPES if entry_type == value),
        None,
    )


def glossary_entry_from_json(index: int, item: object) -> Result[GlossaryEntry, str]:
    if not is_json_object(item):
        return Err(f"glossary entry {index} must be an object")

    original_value = item.get("original")
    if not is_non_empty_string(original_value):
        return Err(f'glossary entry {index} needs a non-empty "original" string')

    original = original_value

    invalid_keys = tuple(
        key
        for key in ("translated", "pronouns", "notes")
        if key in item and item[key] is not None and not isinstance(item[key], str)
    )
    if invalid_keys:
        return Err(
            f'glossary entry {index} ("{original}") has invalid value(s) for: '
            f"{', '.join(invalid_keys)}; translated, pronouns, and notes must be "
            "a string or null"
        )

    type_value = item.get("type", "other")
    entry_type = glossary_entry_type_from_value(type_value)
    if entry_type is None:
        return Err(
            f'glossary entry {index} ("{original}") has invalid type {type_value!r}; '
            f"allowed types are {', '.join(GLOSSARY_ENTRY_TYPES)}"
        )

    return Ok(
        GlossaryEntry(
            original=original,
            translated=text_or_none(item.get("translated")),
            pronouns=text_or_none(item.get("pronouns")),
            type=entry_type,
            notes=text_or_none(item.get("notes")),
        )
    )


def drop_later_duplicate_entries(
    entries: tuple[GlossaryEntry, ...],
) -> tuple[GlossaryEntry, ...]:
    return tuple(
        entry
        for index, entry in enumerate(entries)
        if entry.original not in {earlier.original for earlier in entries[:index]}
    )


def parse_entries_document(
    document: object,
    parse_entry: Callable[[int, object], Result[GlossaryEntry, str]],
) -> Result[tuple[GlossaryEntry, ...], str]:
    if not is_json_array(document):
        return Err("glossary must be a JSON array of entries")

    return result_map(
        result_sequence(
            parse_entry(index, item) for index, item in enumerate(document, 1)
        ),
        drop_later_duplicate_entries,
    )


def parse_glossary_document(
    path: str, document: object
) -> Result[tuple[GlossaryEntry, ...], TranslationError]:
    return result_map_error(
        parse_entries_document(document, glossary_entry_from_json),
        lambda problem: ConfigError(f"{path}: {problem}"),
    )


def reply_entries_document(document: object) -> Result[list[object], str]:
    if is_object_list(document):
        return Ok(document)

    if not is_entries_table(document):
        return Err('model reply must be {"entries": [...]} or a bare JSON array')

    entries_value = document["entries"]
    if not is_object_list(entries_value):
        return Err('model reply must be {"entries": [...]} or a bare JSON array')

    return Ok(entries_value)


def is_entries_table(value: object) -> TypeIs[Mapping[str, object]]:
    if not is_string_keyed_table(value):
        return False

    return "entries" in value


def parse_model_glossary_reply(
    text: str, path: str
) -> Result[tuple[GlossaryEntry, ...], TranslationError]:
    without_think, _reasoning = strip_think_tag(text)
    content = strip_markdown_fence(without_think)
    decoded = decode_json_document(content)
    if isinstance(decoded, Err):
        return Err(
            GlossaryReplyError(
                pass_name=path,
                problem=f"model reply is not valid JSON: {decoded.error}",
            )
        )

    return result_map_error(
        result_bind(
            reply_entries_document(decoded.value),
            lambda entries: parse_entries_document(entries, glossary_entry_from_json),
        ),
        lambda problem: GlossaryReplyError(pass_name=path, problem=problem),
    )


def integrate_glossary_reply(
    existing: tuple[GlossaryEntry, ...],
    reply_entries: tuple[GlossaryEntry, ...],
    include_types: tuple[str, ...] = (),
) -> tuple[GlossaryEntry, ...]:
    known_originals = {entry.original for entry in existing}
    return existing + tuple(
        entry
        for entry in reply_entries
        if entry.original not in known_originals
        and (not include_types or entry.type in include_types)
    )


def merge_glossary_entries(
    groups: tuple[tuple[GlossaryEntry, ...], ...],
) -> tuple[GlossaryEntry, ...]:
    by_original = {entry.original: entry for group in groups for entry in group}
    return tuple(by_original.values())


def strip_markdown_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()[1:]
    body = lines[:-1] if lines and lines[-1].strip().startswith("```") else lines
    return "\n".join(body).strip()


def decode_json_document(content: str) -> Result[JsonValue, json.JSONDecodeError]:
    try:
        decoded: JsonValue = json.loads(content)
        return Ok(decoded)
    except json.JSONDecodeError as strict_error:
        array_start_index = content.find("[")
        if array_start_index < 0:
            return Err(strict_error)
        try:
            document, _end_index = json.JSONDecoder().raw_decode(
                content, array_start_index
            )
            return Ok(document)
        except json.JSONDecodeError:
            return Err(strict_error)


def glossary_response_format(
    no_structured_output: bool = False,
) -> dict[str, JsonValue] | None:
    if no_structured_output:
        return None

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "glossary_entries",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "original": {"type": "string"},
                                "translated": {"type": ["string", "null"]},
                                "pronouns": {"type": ["string", "null"]},
                                "notes": {"type": ["string", "null"]},
                                "type": {
                                    "type": "string",
                                    "enum": list(GLOSSARY_ENTRY_TYPES),
                                },
                            },
                            "required": [
                                "original",
                                "translated",
                                "pronouns",
                                "notes",
                                "type",
                            ],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["entries"],
                "additionalProperties": False,
            },
        },
    }


def entries_to_json_text(entries: tuple[GlossaryEntry, ...]) -> str:
    return (
        json.dumps(
            [
                {
                    "original": entry.original,
                    "translated": entry.translated,
                    "pronouns": entry.pronouns,
                    "notes": entry.notes,
                    "type": entry.type,
                }
                for entry in entries
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def extraction_user_message(target_language: str, text: str) -> str:
    return f"Target language: {target_language}\n\nSource text:\n{text}"

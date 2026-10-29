from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, get_args

from l2l.errors import (
    ConfigError,
    GlossaryReplyError,
    TranslationError,
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

GlossaryEntryType = Literal["character", "place", "other"]

GLOSSARY_ENTRY_TYPES: tuple[GlossaryEntryType, ...] = get_args(GlossaryEntryType)


@dataclass(frozen=True)
class GlossaryEntry:
    original: str
    translated: str | None
    pronouns: str | None
    type: GlossaryEntryType


GLOSSARY_BLOCK_HEADER = (
    "Glossary (mandatory): use the exact renderings and attributes below on "
    "every occurrence; never vary a rendering, never assign pronouns to "
    "entries that carry none, and keep each character's pronouns consistent "
    "even where the source omits subjects:"
)


def render_glossary_entry_line(entry: GlossaryEntry) -> str:
    return "- %s%s%s%s" % (
        entry.original,
        (' = "%s"' % entry.translated) if entry.translated is not None else "",
        (" (pronouns: %s)" % entry.pronouns) if entry.pronouns is not None else "",
        (" (type: %s)" % entry.type) if entry.type != "other" else "",
    )


def render_glossary_block(entries: tuple[GlossaryEntry, ...]) -> str:
    return "\n".join(
        (GLOSSARY_BLOCK_HEADER,)
        + tuple(render_glossary_entry_line(entry) for entry in entries)
    )


def instruction_with_glossary(
    instruction: str, entries: tuple[GlossaryEntry, ...]
) -> str:
    if not entries:
        return instruction

    return instruction + "\n\n" + render_glossary_block(entries)


def text_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def glossary_entry_from_json(index: int, item: Any) -> Result[GlossaryEntry, str]:
    if not isinstance(item, dict):
        return Err("glossary entry %d must be an object" % index)

    original = item.get("original")
    if not isinstance(original, str) or not original.strip():
        return Err('glossary entry %d needs a non-empty "original" string' % index)

    invalid_keys = tuple(
        key
        for key in ("translated", "pronouns")
        if key in item and item[key] is not None and not isinstance(item[key], str)
    )
    if invalid_keys:
        return Err(
            'glossary entry %d ("%s") has invalid value(s) for: %s; '
            "translated and pronouns must be a string or null"
            % (index, original, ", ".join(invalid_keys))
        )

    entry_type = item.get("type", "other")
    if entry_type not in GLOSSARY_ENTRY_TYPES:
        return Err(
            'glossary entry %d ("%s") has invalid type %r; allowed types are %s'
            % (
                index,
                original,
                entry_type,
                ", ".join(GLOSSARY_ENTRY_TYPES),
            )
        )

    return Ok(
        GlossaryEntry(
            original=original,
            translated=text_or_none(item.get("translated")),
            pronouns=text_or_none(item.get("pronouns")),
            type=entry_type,
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
    document: Any,
    parse_entry: Callable[[int, Any], Result[GlossaryEntry, str]],
) -> Result[tuple[GlossaryEntry, ...], str]:
    if not isinstance(document, list):
        return Err("glossary must be a JSON array of entries")

    return result_map(
        result_sequence(
            parse_entry(index, item) for index, item in enumerate(document, 1)
        ),
        drop_later_duplicate_entries,
    )


def parse_glossary_document(
    path: str, document: Any
) -> Result[tuple[GlossaryEntry, ...], TranslationError]:
    return result_map_error(
        parse_entries_document(document, glossary_entry_from_json),
        lambda problem: ConfigError("%s: %s" % (path, problem)),
    )


def reply_entries_document(document: Any) -> Result[list[Any], str]:
    match document:
        case {"entries": list(entries)}:
            return Ok(entries)

        case list(entries):
            return Ok(entries)

        case _:
            return Err('model reply must be {"entries": [...]} or a bare JSON array')


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
                problem="model reply is not valid JSON: %s" % decoded.error,
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


def decode_json_document(content: str) -> Result[Any, json.JSONDecodeError]:
    try:
        return Ok(json.loads(content))
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
) -> dict[str, Any] | None:
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
                                "type": {
                                    "type": "string",
                                    "enum": list(GLOSSARY_ENTRY_TYPES),
                                },
                            },
                            "required": [
                                "original",
                                "translated",
                                "pronouns",
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
                    "type": entry.type,
                }
                for entry in entries
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def known_originals_component(entries: tuple[GlossaryEntry, ...]) -> str:
    if not entries:
        return ""

    return "Already known names (do not output entries for these):\n%s\n" % "\n".join(
        sorted(entry.original for entry in entries)
    )


def extraction_user_message(
    target_language: str,
    known_entries: tuple[GlossaryEntry, ...],
    text: str,
) -> str:
    return "Target language: %s\n\n%sSource text:\n%s" % (
        target_language,
        known_originals_component(known_entries),
        text,
    )

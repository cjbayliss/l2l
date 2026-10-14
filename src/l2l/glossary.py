from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from l2l.errors import TranslationError, fail_config
from l2l.monads import Ok, Result, result_bind, result_sequence
from l2l.text import strip_think_tag


@dataclass(frozen=True)
class GlossaryEntry:
    original: str
    attributes: Mapping[str, str]


GLOSSARY_BLOCK_HEADER = (
    "Glossary (mandatory): use the exact renderings and attributes below on "
    "every occurrence; never vary a rendering, never assign pronouns to "
    "entries that carry none, and keep each character's pronouns consistent "
    "even where the source omits subjects:"
)


def render_glossary_entry_line(entry: GlossaryEntry) -> str:
    translated = entry.attributes.get("translated")
    remainder = tuple(
        "(%s: %s)" % (key, value)
        for key, value in entry.attributes.items()
        if key != "translated"
    )
    return "- %s%s%s" % (
        entry.original,
        (' = "%s"' % translated) if translated is not None else "",
        "".join(" " + item for item in remainder),
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


def parse_glossary_entry(
    path: str, index: int, item: Any
) -> Result[GlossaryEntry, TranslationError]:
    if not isinstance(item, dict):
        return fail_config("%s: glossary entry %d must be an object" % (path, index))

    original = item.get("original")
    if not isinstance(original, str) or not original.strip():
        return fail_config(
            '%s: glossary entry %d needs a non-empty "original" string' % (path, index)
        )

    invalid = tuple(
        key
        for key, value in item.items()
        if key != "original"
        and value is not None
        and (not isinstance(value, str) or not value.strip())
    )
    if invalid:
        return fail_config(
            '%s: glossary entry %d ("%s") has invalid value(s) for: %s; '
            "every attribute must be a non-empty string or null"
            % (path, index, original, ", ".join(str(key) for key in invalid))
        )

    attributes: Mapping[str, str] = MappingProxyType(
        {
            key: value
            for key, value in item.items()
            if key != "original" and value is not None
        }
    )
    if not attributes:
        return fail_config(
            '%s: glossary entry %d ("%s") needs at least one attribute '
            "besides original" % (path, index, original)
        )

    return Ok(GlossaryEntry(original=original, attributes=attributes))


def parse_glossary_document(
    path: str, document: Any
) -> Result[tuple[GlossaryEntry, ...], TranslationError]:
    if not isinstance(document, list):
        return fail_config("%s: glossary must be a JSON array of entries" % path)

    def check_duplicates(
        entries: tuple[GlossaryEntry, ...],
    ) -> Result[tuple[GlossaryEntry, ...], TranslationError]:
        originals = tuple(entry.original for entry in entries)
        duplicates = tuple(
            dict.fromkeys(
                original for original in originals if originals.count(original) > 1
            )
        )
        if duplicates:
            return fail_config(
                '%s: duplicate glossary "original" value(s): %s'
                % (path, ", ".join(duplicates))
            )

        return Ok(entries)

    return result_bind(
        result_sequence(
            parse_glossary_entry(path, index, item)
            for index, item in enumerate(document, 1)
        ),
        check_duplicates,
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


def parse_model_glossary_reply(
    text: str, path: str
) -> Result[tuple[GlossaryEntry, ...], TranslationError]:
    without_think, _reasoning = strip_think_tag(text)
    content = strip_markdown_fence(without_think)
    try:
        document = json.loads(content)
    except json.JSONDecodeError as error:
        return fail_config("%s: model reply is not valid JSON: %s" % (path, error))

    return parse_glossary_document(path, document)


def entries_to_json_text(entries: tuple[GlossaryEntry, ...]) -> str:
    return (
        json.dumps(
            [
                {"original": entry.original, **dict(entry.attributes)}
                for entry in entries
            ],
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def extraction_user_message(
    target_language: str,
    carried_entries: tuple[GlossaryEntry, ...],
    text: str,
) -> str:
    carried = (
        "Already decided entries (include them unchanged in your reply):\n"
        + entries_to_json_text(carried_entries)
        + "\n"
        if carried_entries
        else ""
    )
    return "Target language: %s\n\n%sSource text:\n%s" % (
        target_language,
        carried,
        text,
    )

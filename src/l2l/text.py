from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate, chain
from typing import Any

import unicodedataplus

from l2l.monads import NOTHING, Just, Maybe, maybe_or_else


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class Translated:
    text: str
    usage: Usage
    validated: bool = True
    problem: str | None = None


CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),
    (0x3040, 0x30FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
    (0x20000, 0x2FA1F),
)

HANGUL_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),
    (0xAC00, 0xD7AF),
)

THAI_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0E7F),
    (0x0E80, 0x0EFF),
)

ONE_CHAR_PER_TOKEN_RANGES = CJK_RANGES + HANGUL_RANGES
TWO_CHARS_PER_TOKEN_RANGES = THAI_RANGES


def in_ranges(char: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in ranges)


def is_cjk_char(char: str) -> bool:
    return in_ranges(char, CJK_RANGES)


def chars_per_token(char: str) -> int:
    if in_ranges(char, ONE_CHAR_PER_TOKEN_RANGES):
        return 1

    return 2 if in_ranges(char, TWO_CHARS_PER_TOKEN_RANGES) else 4


def estimate_tokens(text: str) -> int:
    return (sum(4 // chars_per_token(character) for character in text) + 3) // 4


def paragraph_separator_pattern(text: str) -> str:
    blank_lines = re.findall(r"\n\s*\n", text)
    non_blank_newlines = text.count("\n") - sum(
        blank.count("\n") for blank in blank_lines
    )
    return r"(\n+)" if non_blank_newlines >= 3 * len(blank_lines) else r"(\n\s*\n)"


def split_paragraphs(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    pattern = re.compile(paragraph_separator_pattern(text))
    parts = pattern.split(text)
    return (
        tuple(part for part in parts if part and not pattern.fullmatch(part)),
        tuple(part for part in parts if part and pattern.fullmatch(part)),
    )


def ensure_blank_line_separators(text: str) -> str:
    return "\n\n".join(split_paragraphs(text)[0])


def count_paragraphs(text: str) -> int:
    return len(split_paragraphs(text)[0])


def group_units_to_budget(
    units: tuple[str, ...], budget: int
) -> tuple[tuple[str, ...], ...]:
    def fold(
        accumulator: tuple[tuple[tuple[str, ...], ...], tuple[str, ...], int],
        unit: str,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], int]:
        groups, current, size = accumulator
        unit_size = estimate_tokens(unit)
        if current and size + unit_size > budget:
            return groups + (current,), (unit,), unit_size

        return groups, current + (unit,), size + unit_size

    initial: tuple[tuple[tuple[str, ...], ...], tuple[str, ...], int] = ((), (), 0)
    groups, current, _ = reduce(fold, units, initial)
    return groups + (current,) if current else groups


def make_chunks(
    paragraphs: tuple[str, ...], budget: int
) -> tuple[tuple[str, ...], ...]:
    return group_units_to_budget(paragraphs, budget)


def split_sentences(text: str, boundary_characters: str) -> tuple[str, ...]:
    pieces = re.split("(?<=[%s])" % re.escape(boundary_characters), text)
    trimmed = pieces[:-1] if pieces and pieces[-1] == "" else pieces
    return tuple(trimmed)


def split_to_budget(
    text: str, budget: int, boundary_characters: str
) -> tuple[str, ...]:
    sentences = split_sentences(text, boundary_characters)
    return tuple("".join(group) for group in group_units_to_budget(sentences, budget))


def split_units_to_budget(
    paragraphs: tuple[str, ...], budget: int, boundary_characters: str
) -> tuple[str, ...]:
    return tuple(
        chain.from_iterable(
            (
                (paragraph,)
                if estimate_tokens(paragraph) <= budget
                else split_to_budget(paragraph, budget, boundary_characters)
            )
            for paragraph in paragraphs
        )
    )


def separator_at(separators: tuple[str, ...], index: int) -> str:
    return separators[index] if index < len(separators) else ""


def unit_separators(
    plan: tuple[tuple[str, ...], ...], separators: tuple[str, ...]
) -> tuple[str, ...]:
    def separator_after_unit(index: int, end: int) -> str:
        if index >= len(plan) - 1:
            return ""

        return separator_at(separators, end - 1)

    return tuple(
        separator_after_unit(index, end)
        for index, end in enumerate(accumulate(map(len, plan)))
    )


def regroup_by_plan(
    paragraphs: tuple[str, ...], plan: tuple[tuple[str, ...], ...]
) -> Maybe[tuple[tuple[str, ...], ...]]:
    if len(paragraphs) != sum(map(len, plan)):
        return NOTHING

    return Just(
        tuple(
            tuple(paragraphs[start : start + len(chunk)])
            for start, chunk in zip(
                accumulate(map(len, plan), initial=0), plan, strict=False
            )
        )
    )


FLOAT_PATTERN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


def parse_float(value: Any) -> Maybe[float]:
    if isinstance(value, bool):
        return NOTHING

    if isinstance(value, (int, float)):
        return Just(float(value))

    if isinstance(value, str) and FLOAT_PATTERN.fullmatch(value.strip()):
        return Just(float(value))

    return NOTHING


def parse_cost(value: Any) -> float:
    return maybe_or_else(parse_float(value), lambda: 0.0)


def parse_tokens(value: Any) -> int:
    return int(maybe_or_else(parse_float(value), lambda: 0.0))


def add_usage(usage: Usage, reported: Mapping[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=usage.prompt_tokens + parse_tokens(reported.get("prompt_tokens")),
        completion_tokens=usage.completion_tokens
        + parse_tokens(reported.get("completion_tokens")),
        cost=usage.cost + parse_cost(reported.get("cost")),
    )


def usage_delta(start_usage: Usage, end_usage: Usage) -> tuple[int, int, float]:
    return (
        end_usage.prompt_tokens - start_usage.prompt_tokens,
        end_usage.completion_tokens - start_usage.completion_tokens,
        end_usage.cost - start_usage.cost,
    )


def usage_add(first: Usage, second: Usage) -> Usage:
    return Usage(
        prompt_tokens=first.prompt_tokens + second.prompt_tokens,
        completion_tokens=first.completion_tokens + second.completion_tokens,
        cost=first.cost + second.cost,
    )


def non_ascii_sample(text: str, limit: int = 12) -> str:
    return "".join(
        tuple(dict.fromkeys(char for char in text if not char.isascii()))[:limit]
    )


def has_letters(text: str) -> bool:
    return any(char.isalpha() for char in text)


def has_alphanumeric_characters(text: str) -> bool:
    return any(char.isalnum() for char in text)


UNKNOWN_SCRIPT = ""


def character_scripts(character: str) -> tuple[str, ...]:
    if not character.isalpha():
        return ()

    return tuple(str(script) for script in unicodedataplus.script_extensions(character))


def letter_script(character: str) -> str:
    scripts = character_scripts(character)
    return scripts[0] if scripts else UNKNOWN_SCRIPT


def dominant_script(text: str) -> str:
    scripts = tuple(script for script in map(letter_script, text) if script)
    tallies = Counter(scripts)
    return max(
        ((count, script) for script, count in tallies.items()),
        default=(0, UNKNOWN_SCRIPT),
    )[1]


def source_script(paragraphs: tuple[str, ...]) -> str:
    return dominant_script("".join(paragraphs))


def echoes_source(source: str, output: str, character_map: Mapping[str, str]) -> bool:
    return (
        to_ascii_mechanical(source, character_map).split()
        == to_ascii_mechanical(output, character_map).split()
    )


def has_ascii_letter(text: str) -> bool:
    return any(character.isascii() and character.isalpha() for character in text)


def contains_target_script(text: str, target_scripts: tuple[str, ...]) -> bool:
    return any(
        script in target_scripts
        for character in text
        for script in character_scripts(character)
    )


def untranslated_paragraph(
    source: str,
    output: str,
    target_scripts: tuple[str, ...],
    character_map: Mapping[str, str],
) -> bool:
    if not has_letters(source):
        return False

    if echoes_source(source, output, character_map):
        return True

    if not has_letters(output):
        return not has_alphanumeric_characters(output)

    if not target_scripts:
        return not has_ascii_letter(output)

    return not contains_target_script(output, target_scripts)


def paragraph_is_in_target_script(output: str, target_scripts: tuple[str, ...]) -> bool:
    if not has_letters(output):
        return False

    if not target_scripts:
        return all(char.isascii() for char in output if char.isalpha())

    return dominant_script(output) in target_scripts


def excerpt(text: str, limit: int = 60) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def to_ascii_mechanical(text: str, character_map: Mapping[str, str]) -> str:
    return "".join(
        character
        for character in unicodedata.normalize(
            "NFKD",
            reduce(
                lambda text, pair: text.replace(pair[0], pair[1]),
                character_map.items(),
                text,
            ),
        )
        if not unicodedata.combining(character)
    )


@dataclass(frozen=True)
class AsciiDrop:
    index: int
    sample: str
    attempts: int


def drop_non_ascii(
    text: str, character_map: Mapping[str, str], attempts: int, index: int
) -> tuple[str, Maybe[AsciiDrop]]:
    if text.isascii():
        return text, NOTHING

    fallback = to_ascii_mechanical(text, character_map)
    if fallback.isascii():
        return fallback, NOTHING

    return (
        re.sub(
            r"  +", " ", "".join(character for character in text if character.isascii())
        ),
        Just(AsciiDrop(index, non_ascii_sample(text), attempts)),
    )


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_think_close(text: str) -> tuple[str, str] | None:
    index = text.find(THINK_CLOSE)
    if index < 0:
        return None

    return text[:index], text[index + len(THINK_CLOSE) :]


def strip_think_tag(content: str) -> tuple[str, Maybe[str]]:
    stripped = content.lstrip()
    if not stripped.startswith(THINK_OPEN):
        return content, NOTHING

    reasoning = stripped[len(THINK_OPEN) :]
    closed = split_think_close(reasoning)
    if closed is None:
        return "", Just(reasoning.strip())

    captured, visible = closed
    return visible, Just(captured.strip())


@dataclass(frozen=True)
class SseState:
    pending_data: tuple[str, ...] = ()


@dataclass(frozen=True)
class SseDone:
    pass


@dataclass(frozen=True)
class SseEvent:
    payload: dict[str, Any]


@dataclass(frozen=True)
class SseMalformed:
    data: str


type SseFrame = SseDone | SseEvent | SseMalformed | None


def decode_sse_data(data: str) -> SseFrame:
    if not data:
        return None

    if data == "[DONE]":
        return SseDone()

    try:
        loaded = json.loads(data)
    except json.JSONDecodeError:
        return SseMalformed(data)

    return SseEvent(loaded) if isinstance(loaded, dict) else None


def sse_step(state: SseState, raw_line: bytes) -> tuple[SseState, SseFrame]:
    line = raw_line.decode("utf-8", "replace").strip("\r\n")
    if line == "":
        if not state.pending_data:
            return state, None

        return SseState(), decode_sse_data("\n".join(state.pending_data))

    if line.startswith(":"):
        return state, None

    if line.startswith("data:"):
        payload = line[len("data:") :].removeprefix(" ")

        return replace(state, pending_data=state.pending_data + (payload,)), None

    return state, None


@dataclass(frozen=True)
class ThinkState:
    scanning_opening_tag: bool = True
    inside_reasoning: bool = False
    buffered_text: str = ""


def buffered_close_length() -> int:
    return len(THINK_CLOSE) - 1


def think_step(state: ThinkState, text: str) -> tuple[ThinkState, bool, str]:
    if not state.scanning_opening_tag and not state.inside_reasoning:
        return (
            ThinkState(False, False, ""),
            False,
            state.buffered_text + text,
        )

    searched = state.buffered_text + text
    if state.inside_reasoning:
        closed = split_think_close(searched)
        if closed is None:
            return (
                ThinkState(False, True, searched[-buffered_close_length() :]),
                True,
                "",
            )

        return ThinkState(False, False, ""), True, closed[1]

    stripped = searched.lstrip()
    if stripped.startswith(THINK_OPEN):
        rest = stripped[len(THINK_OPEN) :]
        closed = split_think_close(rest)
        if closed is None:
            return (
                ThinkState(True, True, rest[-buffered_close_length() :]),
                True,
                "",
            )

        return ThinkState(False, False, ""), True, closed[1]

    if THINK_OPEN.startswith(stripped):
        return ThinkState(True, False, searched[-len(THINK_CLOSE) :]), False, ""

    return ThinkState(False, False, ""), False, searched


def cache_path(cache_directory: str, key: str) -> str:
    return os.path.join(cache_directory, key + ".txt")


CACHE_SALT_VERSION = "4"


def cache_key(
    chunk_text: str,
    model: str,
    pass_salt: str = "",
    work_text: str = "",
    overrides: Mapping[str, Any] | None = None,
    context: str = "",
    analysis: str = "",
) -> str:
    salt_bytes = pass_salt.encode("utf-8") + b"\x00" if pass_salt else b""
    work_bytes = b"\x00work\x00" + work_text.encode("utf-8") if work_text else b""
    context_bytes = b"\x00context\x00" + context.encode("utf-8") if context else b""
    analysis_bytes = b"\x00analysis\x00" + analysis.encode("utf-8") if analysis else b""
    override_bytes = (
        b"\x00"
        + json.dumps(
            dict(overrides), sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")
        if overrides
        else b""
    )
    parts = (
        salt_bytes,
        chunk_text.encode("utf-8"),
        work_bytes,
        context_bytes,
        analysis_bytes,
        b"\x00" + model.encode("utf-8"),
        override_bytes,
    )
    return hashlib.sha256(b"".join(parts)).hexdigest()

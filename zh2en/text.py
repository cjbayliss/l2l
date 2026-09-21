from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from itertools import accumulate, chain
from typing import Any


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class Translated:
    text: str
    usage: Usage


def is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3000 <= code <= 0x303F
        or 0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0xFF00 <= code <= 0xFFEF
        or 0x20000 <= code <= 0x2FA1F
    )


def estimate_tokens(text: str) -> int:
    cjk_count = sum(1 for char in text if is_cjk_char(char))
    return cjk_count + (len(text) - cjk_count + 3) // 4


def split_paragraphs(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    blanks = re.compile(r"\n\s*\n").findall(text)
    separator_pattern = (
        r"(\n+)"
        if text.count("\n") - sum(blank.count("\n") for blank in blanks)
        >= 3 * len(blanks)
        else r"(\n\s*\n)"
    )
    parts = re.split(separator_pattern, text)
    is_separator = re.compile(separator_pattern).fullmatch
    return (
        tuple(part for part in parts if part and not is_separator(part)),
        tuple(part for part in parts if part and is_separator(part)),
    )


def ensure_blank_line_separators(text: str) -> str:
    return "\n\n".join(split_paragraphs(text)[0])


def count_paragraphs(text: str) -> int:
    return len(split_paragraphs(text)[0])


def make_chunks(
    paragraphs: tuple[str, ...], budget: int
) -> tuple[tuple[str, ...], ...]:
    def fold(
        accumulator: tuple[tuple[tuple[str, ...], ...], tuple[str, ...], int],
        paragraph: str,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], int]:
        chunks, current, size = accumulator
        paragraph_size = estimate_tokens(paragraph)
        if current and size + paragraph_size > budget:
            return chunks + (current,), (paragraph,), paragraph_size

        return chunks, current + (paragraph,), size + paragraph_size

    initial: tuple[tuple[tuple[str, ...], ...], tuple[str, ...], int] = ((), (), 0)
    chunks, current, _ = reduce(fold, paragraphs, initial)
    return chunks + (current,) if current else chunks


def split_sentences(text: str, boundary_characters: str) -> tuple[str, ...]:
    pieces = re.split("(?<=[%s])" % re.escape(boundary_characters), text)
    if pieces and pieces[-1] == "":
        pieces = pieces[:-1]

    return tuple(pieces)


def split_to_budget(
    text: str, budget: int, boundary_characters: str
) -> tuple[str, ...]:
    def fold(
        accumulator: tuple[tuple[str, ...], tuple[str, ...], int], sentence: str
    ) -> tuple[tuple[str, ...], tuple[str, ...], int]:
        pieces, buffer, size = accumulator
        sentence_size = estimate_tokens(sentence)
        if buffer and size + sentence_size > budget:
            return pieces + ("".join(buffer),), (sentence,), sentence_size

        return pieces, buffer + (sentence,), size + sentence_size

    initial: tuple[tuple[str, ...], tuple[str, ...], int] = ((), (), 0)
    pieces, buffer, _ = reduce(
        fold, split_sentences(text, boundary_characters), initial
    )
    return pieces + ("".join(buffer),) if buffer else pieces


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


def unit_separators(
    plan: tuple[tuple[str, ...], ...], separators: tuple[str, ...]
) -> tuple[str, ...]:
    def trailing(index: int, end: int) -> str:
        if index >= len(plan) - 1:
            return ""

        position = end - 1
        return separators[position] if position < len(separators) else ""

    return tuple(
        trailing(index, end) for index, end in enumerate(accumulate(map(len, plan)))
    )


def regroup_by_plan(
    paragraphs: tuple[str, ...], plan: tuple[tuple[str, ...], ...]
) -> tuple[tuple[str, ...], ...] | None:
    if len(paragraphs) != sum(map(len, plan)):
        return None

    return tuple(
        tuple(paragraphs[start : start + len(chunk)])
        for start, chunk in zip(
            accumulate(map(len, plan), initial=0), plan, strict=False
        )
    )


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return "%.1fs" % seconds

    return "%dm%ds" % (int(seconds // 60), int(seconds % 60))


def usage_line(
    label: str, elapsed: float, prompt_tokens: int, completion_tokens: int, cost: float
) -> str:
    return "%s: %s, prompt=%d, completion=%d, %.1f tok/s, cost=$%.6f" % (
        label,
        fmt_duration(elapsed),
        prompt_tokens,
        completion_tokens,
        completion_tokens / elapsed if elapsed else 0.0,
        cost,
    )


def add_usage(usage: Usage, reported: Mapping[str, Any]) -> Usage:
    try:
        cost = float(reported.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    return Usage(
        prompt_tokens=usage.prompt_tokens + (reported.get("prompt_tokens") or 0),
        completion_tokens=usage.completion_tokens
        + (reported.get("completion_tokens") or 0),
        cost=usage.cost + cost,
    )


def usage_delta(start_usage: Usage, end_usage: Usage) -> tuple[int, int, float]:
    return (
        end_usage.prompt_tokens - start_usage.prompt_tokens,
        end_usage.completion_tokens - start_usage.completion_tokens,
        end_usage.cost - start_usage.cost,
    )


def non_ascii_sample(text: str, limit: int = 12) -> str:
    return "".join(
        tuple(dict.fromkeys(char for char in text if not char.isascii()))[:limit]
    )


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
) -> tuple[str, AsciiDrop | None]:
    if text.isascii():
        return text, None

    fallback = to_ascii_mechanical(text, character_map)
    if fallback.isascii():
        return fallback, None

    return (
        re.sub(
            r"  +", " ", "".join(character for character in text if character.isascii())
        ),
        AsciiDrop(index, non_ascii_sample(text), attempts),
    )


def strip_think_tag(content: Any) -> tuple[Any, str | None]:
    if not isinstance(content, str):
        return content, None

    match = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    if not match:
        return content, None

    return content[match.end() :], match.group(1).strip()


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass(frozen=True)
class ThinkState:
    checking: bool = True
    open: bool = False
    held: str = ""


def think_step(state: ThinkState, text: str) -> tuple[ThinkState, bool, str]:
    if not state.checking and not state.open:
        return ThinkState(False, False, ""), False, state.held + text

    searched = state.held + text
    if state.open:
        index = searched.find(THINK_CLOSE)
        if index < 0:
            return (
                ThinkState(
                    checking=False, open=True, held=searched[-(len(THINK_CLOSE) - 1) :]
                ),
                True,
                "",
            )

        return (
            ThinkState(False, False, ""),
            True,
            searched[index + len(THINK_CLOSE) :],
        )

    stripped = searched.lstrip()
    if stripped.startswith(THINK_OPEN):
        rest = stripped[len(THINK_OPEN) :]
        index = rest.find(THINK_CLOSE)
        if index < 0:
            return (
                ThinkState(
                    checking=True,
                    open=True,
                    held=rest[-(len(THINK_CLOSE) - 1) :],
                ),
                True,
                "",
            )

        return ThinkState(False, False, ""), True, rest[index + len(THINK_CLOSE) :]

    if THINK_OPEN.startswith(stripped):
        return ThinkState(True, False, searched[-len(THINK_CLOSE) :]), False, ""

    return ThinkState(False, False, ""), False, searched


def cache_path(cache_directory: str, key: str) -> str:
    return os.path.join(cache_directory, key + ".txt")


CACHE_SALT_VERSION = "2"


def cache_key(
    chunk_text: str,
    model: str,
    pass_salt: str = "",
    work_text: str = "",
    overrides: Mapping[str, Any] | None = None,
    context: str = "",
) -> str:
    hasher = hashlib.sha256()
    if pass_salt:
        hasher.update(pass_salt.encode("utf-8") + b"\x00")

    hasher.update(chunk_text.encode("utf-8"))
    if work_text:
        hasher.update(b"\x00work\x00" + work_text.encode("utf-8"))

    if context:
        hasher.update(b"\x00context\x00" + context.encode("utf-8"))

    hasher.update(b"\x00" + model.encode("utf-8"))
    if overrides:
        hasher.update(
            b"\x00"
            + json.dumps(
                overrides, sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8")
        )

    return hasher.hexdigest()

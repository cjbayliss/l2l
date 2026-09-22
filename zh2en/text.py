from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate, chain
from typing import Any

from zh2en.monads import NOTHING, Just, Maybe, maybe_or_else_get


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
    pieces = re.split(f"(?<=[{re.escape(boundary_characters)}])", text)
    trimmed = pieces[:-1] if pieces and pieces[-1] == "" else pieces
    return tuple(trimmed)


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
    """Parse a JSON-ish number without exceptions: numbers pass through,
    numeric strings are matched algebraically, everything else is Nothing."""
    if isinstance(value, bool):
        return NOTHING

    if isinstance(value, (int, float)):
        return Just(float(value))

    if isinstance(value, str) and FLOAT_PATTERN.fullmatch(value.strip()):
        return Just(float(value))

    return NOTHING


def parse_cost(value: Any) -> float:
    return maybe_or_else_get(parse_float(value), lambda: 0.0)


def add_usage(usage: Usage, reported: Mapping[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=usage.prompt_tokens + (reported.get("prompt_tokens") or 0),
        completion_tokens=usage.completion_tokens
        + (reported.get("completion_tokens") or 0),
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


def strip_think_tag(content: str) -> tuple[str, Maybe[str]]:
    match = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    if not match:
        return content, NOTHING

    return content[match.end() :], Just(match.group(1).strip())


@dataclass(frozen=True)
class SseState:
    pending: tuple[str, ...] = ()


def decode_sse_data(data: str) -> dict[str, Any] | None:
    if not data or data == "[DONE]":
        return None

    try:
        loaded = json.loads(data)
    except json.JSONDecodeError:
        return None

    return loaded if isinstance(loaded, dict) else None


def sse_step(
    state: SseState, raw_line: bytes
) -> tuple[SseState, dict[str, Any] | None]:
    line = raw_line.decode("utf-8", "replace").strip("\r\n")
    if line == "":
        if not state.pending:
            return state, None

        return SseState(), decode_sse_data("\n".join(state.pending))

    if line.startswith(":"):
        return state, None

    if line.startswith("data:"):
        payload = line[len("data:") :].removeprefix(" ")

        return replace(state, pending=state.pending + (payload,)), None

    return state, None


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
    salted = pass_salt.encode("utf-8") + b"\x00" if pass_salt else b""
    worked = b"\x00work\x00" + work_text.encode("utf-8") if work_text else b""
    contexted = b"\x00context\x00" + context.encode("utf-8") if context else b""
    overriden = (
        b"\x00"
        + json.dumps(overrides, sort_keys=True, ensure_ascii=False, default=str).encode(
            "utf-8"
        )
        if overrides
        else b""
    )
    parts = (
        salted,
        chunk_text.encode("utf-8"),
        worked,
        contexted,
        b"\x00" + model.encode("utf-8"),
        overriden,
    )
    return hashlib.sha256(b"".join(parts)).hexdigest()

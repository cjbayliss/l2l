#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
import tomllib
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate, chain
from types import MappingProxyType
from typing import Any, Generic, TextIO, TypeVar

__version__ = "0.1.0"

T = TypeVar("T")
E = TypeVar("E")
R = TypeVar("R")
A = TypeVar("A")
S = TypeVar("S")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


Result = Ok[T] | Err[E]


def result_map(result: Result[T, E], fn: Callable[[T], R]) -> Result[R, E]:
    return Ok(fn(result.value)) if isinstance(result, Ok) else result


def result_bind(result: Result[T, E], fn: Callable[[T], Result[R, E]]) -> Result[R, E]:
    return fn(result.value) if isinstance(result, Ok) else result


def result_bind_io(
    result: Result[T, E], fn: Callable[[T], IO[Result[R, E]]]
) -> IO[Result[R, E]]:
    if isinstance(result, Err):
        return io_result(result)

    return fn(result.value)


def results_sequence(results: Iterable[Result[T, E]]) -> Result[tuple[T, ...], E]:
    def step(
        accumulator: Result[tuple[T, ...], E], result: Result[T, E]
    ) -> Result[tuple[T, ...], E]:
        return result_bind(
            accumulator,
            lambda values: result_map(result, lambda value: values + (value,)),
        )

    initial: Result[tuple[T, ...], E] = Ok(())
    return reduce(step, tuple(results), initial)


@dataclass(frozen=True)
class IO(Generic[T]):
    run: Callable[[], T]


def io_pure(value: T) -> IO[T]:
    return IO(lambda: value)


def io_result(value: Result[T, E]) -> IO[Result[T, E]]:
    return IO(lambda: value)


def io_map(io_value: IO[T], fn: Callable[[T], R]) -> IO[R]:
    return IO(lambda: fn(io_value.run()))


def io_bind(io_value: IO[T], fn: Callable[[T], IO[R]]) -> IO[R]:
    return IO(lambda: fn(io_value.run()).run())


def io_sequence(io_values: Iterable[IO[T]]) -> IO[tuple[T, ...]]:
    return IO(lambda: tuple(io_value.run() for io_value in io_values))


def fold_io(
    items: Iterable[S],
    step: Callable[[A, S], IO[Result[A, E]]],
    initial: Result[A, E],
) -> IO[Result[A, E]]:
    def thunk() -> Result[A, E]:
        outcome = initial
        for item in tuple(items):
            if isinstance(outcome, Err):
                return outcome

            outcome = step(outcome.value, item).run()

        return outcome

    return IO(thunk)


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    params: Mapping[str, Any]


@dataclass(frozen=True)
class PassDefinition:
    name: str
    instruction: str
    mode: str
    params: Mapping[str, Any]
    model: str | None
    ascii: bool | None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0


@dataclass(frozen=True)
class Settings:
    chunk_budget_tokens: int
    analysis_reserve_tokens: int
    ascii_fix_attempts: int
    sentence_boundary_characters: str
    ascii_fix_instruction: str
    ascii_character_map: Mapping[str, str]
    unit_fix_attempts: int
    unit_output_max_ratio: float


@dataclass(frozen=True)
class State:
    text: str
    analysis: str | None
    usage: Usage


@dataclass(frozen=True)
class Arguments:
    config: str | None
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout: float | None
    max_tokens: int | None
    no_cache: bool
    ensure_paragraphs: bool
    verbose: bool
    show_log_path: bool
    cache_dir: str | None


@dataclass(frozen=True)
class Setup:
    config: Config
    passes: tuple[PassDefinition, ...]
    ensure_paragraphs: bool


OpenHTTP = Callable[[Any, float], Result[Any, str]]
ProgressCallback = Callable[[str, int], None]


@dataclass(frozen=True)
class Context:
    config: Config
    settings: Settings
    use_cache: bool
    cache_directory: str
    verbose: bool
    ensure_paragraphs: bool
    console: Console
    open_http: OpenHTTP
    log: RunLog


def build_settings() -> Settings:
    return Settings(
        chunk_budget_tokens=3500,
        analysis_reserve_tokens=128,
        ascii_fix_attempts=3,
        unit_fix_attempts=2,
        unit_output_max_ratio=6.0,
        sentence_boundary_characters="。！？!?；;\n",
        ascii_fix_instruction="""
This paragraph failed to be fully translated or contains non-ASCII
characters. Please analyse it and only output a clean translation
without any non-ASCII.
""",
        ascii_character_map=MappingProxyType(
            {
                "\u00a0": " ",
                "\u2007": " ",
                "\u2009": " ",
                "\u202f": " ",
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": ",",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
                "\u2032": "'",
                "\u2033": '"',
                "\u2039": "'",
                "\u203a": "'",
                "\u00ab": '"',
                "\u00bb": '"',
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
                "\u2026": "...",
                "\u2025": "..",
                "\u2022": "*",
                "\u2023": "*",
                "\u2024": "*",
                "\u2219": "*",
                "\u00b7": " ",
                "\u2190": "<-",
                "\u2192": "->",
                "\u2194": "<->",
                "\u2264": "<=",
                "\u2265": ">=",
                "\u2260": "!=",
                "\u3002": ".",
                "\u3001": ",",
                "\u300c": '"',
                "\u300d": '"',
                "\u300e": '"',
                "\u300f": '"',
                "\u3008": "<",
                "\u3009": ">",
                "\u300a": '"',
                "\u300b": '"',
                "\u3010": "[",
                "\u3011": "]",
            }
        ),
    )


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


def unit_output_problem(
    source_text: str, output: str, settings: Settings
) -> str | None:
    if not output.strip():
        return "the reply was empty"

    expected = count_paragraphs(source_text)
    found = count_paragraphs(output)
    if found != expected:
        return "the reply has %d paragraph(s) but the source has %d" % (
            found,
            expected,
        )

    source_estimate = estimate_tokens(source_text)
    output_estimate = estimate_tokens(output)
    if output_estimate > settings.unit_output_max_ratio * max(source_estimate, 1):
        return (
            "the reply is ~%d tokens against a source of ~%d tokens "
            "(limit %.0fx)"
            % (
                output_estimate,
                source_estimate,
                settings.unit_output_max_ratio,
            )
        )

    return None


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


API_SETTING_KEYS = ("base_url", "api_key", "model", "timeout", "max_tokens", "params")
PASS_KEYS = (
    "name",
    "instruction",
    "instruction_file",
    "mode",
    "ascii",
    "model",
    "params",
)


def default_api_settings() -> dict[str, Any]:
    return {
        "base_url": "",
        "api_key": "",
        "model": "",
        "timeout": 120.0,
        "max_tokens": 100000,
        "params": {},
    }


def validate_document(path: str, document: dict[str, Any]) -> Result[None, str]:
    if not document:
        return Ok(None)

    unknown = sorted(set(document) - {"api", "options", "pass"})
    if unknown:
        return Err(
            "%s: unknown top-level key(s): %s (expected [api], [options], [[pass]])"
            % (path, ", ".join(unknown))
        )

    return Ok(None)


def string_api_settings(
    path: str, table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    def add(settings: dict[str, Any], key: str) -> Result[dict[str, Any], str]:
        if key not in table:
            return Ok(settings)

        value = table[key]
        if not isinstance(value, str) or not value.strip():
            return Err("%s: [api] %s must be a non-empty string" % (path, key))

        return Ok({**settings, key: value.strip()})

    def step(
        settings_result: Result[dict[str, Any], str], key: str
    ) -> Result[dict[str, Any], str]:
        return result_bind(settings_result, lambda settings: add(settings, key))

    initial: Result[dict[str, Any], str] = Ok({})
    return reduce(step, ("base_url", "api_key", "model"), initial)


def timeout_api_setting(
    path: str, settings: dict[str, Any], table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "timeout" not in table:
        return Ok(settings)

    value = table["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return Err("%s: [api] timeout must be a positive number" % path)

    return Ok({**settings, "timeout": float(value)})


def max_tokens_api_setting(
    path: str, settings: dict[str, Any], table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "max_tokens" not in table:
        return Ok(settings)

    value = table["max_tokens"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return Err("%s: [api] max_tokens must be a positive integer" % path)

    return Ok({**settings, "max_tokens": value})


def params_api_setting(
    path: str, settings: dict[str, Any], table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "params" not in table:
        return Ok(settings)

    value = table["params"]
    if not isinstance(value, dict):
        return Err("%s: [api] params must be a table" % path)

    return Ok({**settings, "params": value})


def document_api_settings(
    path: str, document: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "api" not in document:
        return Ok({})

    table = document["api"]
    if not isinstance(table, dict):
        return Err("%s: [api] must be a table" % path)

    unknown = sorted(set(table) - set(API_SETTING_KEYS))
    if unknown:
        return Err("%s: [api]: unknown key(s): %s" % (path, ", ".join(unknown)))

    return result_bind(
        string_api_settings(path, table),
        lambda settings: result_bind(
            timeout_api_setting(path, settings, table),
            lambda settings: result_bind(
                max_tokens_api_setting(path, settings, table),
                lambda settings: params_api_setting(path, settings, table),
            ),
        ),
    )


def merge_api_settings(
    base: Mapping[str, Any], extra: Mapping[str, Any]
) -> dict[str, Any]:
    def merge_one(settings: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
        if key == "params" and isinstance(settings.get("params"), dict):
            return {**settings, "params": {**settings["params"], **value}}

        return {**settings, key: value}

    return reduce(
        lambda settings, entry: merge_one(settings, entry[0], entry[1]),
        extra.items(),
        dict(base),
    )


def parse_number_setting(
    environment: Mapping[str, str],
    settings: dict[str, Any],
    variable: str,
    key: str,
    convert: Callable[[str], Any],
    invalid_label: str,
) -> Result[dict[str, Any], str]:
    raw = environment.get(variable, "").strip()
    if not raw:
        return Ok(settings)

    try:
        value = convert(raw)
    except ValueError:
        return Err("%s must be %s, got %r" % (variable, invalid_label, raw))

    if value <= 0:
        return Err("%s must be positive" % variable)

    return Ok({**settings, key: value})


def api_settings_from_environment(
    environment: Mapping[str, str],
) -> Result[dict[str, Any], str]:
    def string_step(
        settings_result: Result[dict[str, Any], str], binding: tuple[str, str]
    ) -> Result[dict[str, Any], str]:
        variable, key = binding
        value = environment.get(variable, "").strip()
        return result_bind(
            settings_result,
            lambda settings: Ok({**settings, key: value}) if value else Ok(settings),
        )

    initial: Result[dict[str, Any], str] = Ok({})
    return result_bind(
        reduce(
            string_step,
            (
                ("TRANSLATE_BASE_URL", "base_url"),
                ("TRANSLATE_API_KEY", "api_key"),
                ("TRANSLATE_MODEL", "model"),
            ),
            initial,
        ),
        lambda settings: result_bind(
            parse_number_setting(
                environment, settings, "TRANSLATE_TIMEOUT", "timeout", float, "a number"
            ),
            lambda settings: parse_number_setting(
                environment,
                settings,
                "TRANSLATE_MAX_TOKENS",
                "max_tokens",
                int,
                "an integer",
            ),
        ),
    )


def api_settings_from_arguments(arguments: Arguments) -> dict[str, Any]:
    def step(settings: dict[str, Any], binding: tuple[Any, str]) -> dict[str, Any]:
        value, key = binding
        return {**settings, key: value} if value is not None else settings

    return reduce(
        step,
        (
            (arguments.base_url, "base_url"),
            (arguments.api_key, "api_key"),
            (arguments.model, "model"),
            (arguments.timeout, "timeout"),
            (arguments.max_tokens, "max_tokens"),
        ),
        {},
    )


def missing_api_settings(config: Config) -> tuple[str, ...]:
    return tuple(
        description
        for value, description in (
            (config.base_url, "api.base_url (--base-url / TRANSLATE_BASE_URL)"),
            (config.api_key, "api.api_key (--api-key / TRANSLATE_API_KEY)"),
            (config.model, "api.model (--model / TRANSLATE_MODEL)"),
        )
        if not value
    )


def build_config(settings: Mapping[str, Any]) -> Result[Config, str]:
    config = Config(
        base_url=settings["base_url"].rstrip("/"),
        api_key=settings["api_key"],
        model=settings["model"],
        timeout=settings["timeout"],
        max_tokens=settings["max_tokens"],
        params=MappingProxyType(dict(settings["params"])),
    )
    missing = missing_api_settings(config)
    if missing:
        return Err("missing required API settings: " + ", ".join(missing))

    return Ok(config)


def merged_api_settings(
    arguments: Arguments,
    environment: Mapping[str, str],
    user_path: str,
    user_document_result: Result[dict[str, Any], str],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], str],
) -> Result[Config, str]:
    def merge_document(
        path: str | None,
        document_result: Result[dict[str, Any], str],
        settings: dict[str, Any],
    ) -> Result[dict[str, Any], str]:
        return result_bind(
            document_result,
            lambda document: result_map(
                document_api_settings(path or "", document),
                lambda extra: merge_api_settings(settings, extra),
            ),
        )

    pipeline: Result[dict[str, Any], str] = Ok(default_api_settings())
    pipeline = result_bind(
        pipeline,
        lambda settings: merge_document(user_path, user_document_result, settings),
    )
    pipeline = result_bind(
        pipeline,
        lambda settings: merge_document(
            selected_path, selected_document_result, settings
        ),
    )
    pipeline = result_bind(
        pipeline,
        lambda settings: result_map(
            api_settings_from_environment(environment),
            lambda extra: merge_api_settings(settings, extra),
        ),
    )
    pipeline = result_bind(
        pipeline,
        lambda settings: Ok(
            merge_api_settings(settings, api_settings_from_arguments(arguments))
        ),
    )
    return result_bind(pipeline, build_config)


def parse_options_table(path: str, table: Any) -> Result[dict[str, bool], str]:
    if not isinstance(table, dict):
        return Err("%s: [options] must be a table" % path)

    unknown = sorted(set(table) - {"ascii", "ensure_paragraphs"})
    if unknown:
        return Err("%s: [options]: unknown key(s): %s" % (path, ", ".join(unknown)))

    ascii_value = table.get("ascii", False)
    if not isinstance(ascii_value, bool):
        return Err("%s: [options] ascii must be true or false" % path)

    ensure_paragraphs = table.get("ensure_paragraphs", False)
    if not isinstance(ensure_paragraphs, bool):
        return Err("%s: [options] ensure_paragraphs must be true or false" % path)

    return Ok({"ascii": ascii_value, "ensure_paragraphs": ensure_paragraphs})


def pass_definition_from(
    path: str, name: str, table: dict[str, Any], instruction: str
) -> Result[PassDefinition, str]:
    mode = table.get("mode", "chunk")
    if mode not in ("analysis", "chunk", "paragraph"):
        return Err(
            "%s: [[pass]] %s: mode must be analysis, chunk, or paragraph" % (path, name)
        )

    ascii_value = table.get("ascii")
    if ascii_value is not None and not isinstance(ascii_value, bool):
        return Err("%s: [[pass]] %s: ascii must be true or false" % (path, name))

    model = table.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return Err("%s: [[pass]] %s: model must be a non-empty string" % (path, name))

    params = table.get("params", {})
    if not isinstance(params, dict):
        return Err("%s: [[pass]] %s: params must be a table" % (path, name))

    return Ok(
        PassDefinition(
            name=name,
            instruction=instruction,
            mode=mode,
            params=MappingProxyType(dict(params)),
            model=model.strip() if model else None,
            ascii=ascii_value,
        )
    )


def apply_default_ascii(
    pass_definitions: tuple[PassDefinition, ...], options: Mapping[str, bool]
) -> tuple[PassDefinition, ...]:
    return tuple(
        (
            replace(pass_definition, ascii=options.get("ascii", False))
            if pass_definition.ascii is None
            else pass_definition
        )
        for pass_definition in pass_definitions
    )


def resolve_call_settings(
    config: Config, pass_definition: PassDefinition
) -> tuple[str, dict[str, Any]]:
    return (pass_definition.model or config.model), {
        **dict(config.params),
        **dict(pass_definition.params),
    }


def build_chat_payload(
    model: str, system: str, user: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        **{
            "model": model,
            "messages": (
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ),
        },
        **{key: value for key, value in params.items() if value is not None},
    }


def context_parts(context: tuple[str, ...]) -> tuple[str, ...]:
    if not context:
        return ()

    return (
        "Context paragraphs (reference only — do not translate them, do not "
        "continue the story from them, and do not include them in the "
        "output):",
        *context,
        "Translate only the paragraph that follows, as exactly one paragraph:",
    )


def build_pass_user(
    source_chunk: str,
    work_chunk: str | None,
    analysis: str | None,
    context: tuple[str, ...] = (),
) -> str:
    parts = [
        analysis.strip() if analysis else "",
        *context_parts(context),
        source_chunk,
    ]
    if work_chunk is not None and work_chunk != source_chunk:
        parts.append(work_chunk)
    return "\n\n".join(part for part in parts if part)


def pass_salt(pass_definition: PassDefinition) -> str:
    return "\x00".join(
        (CACHE_SALT_VERSION, pass_definition.name, pass_definition.instruction)
    )


def plan_info_message(
    pass_definition: PassDefinition,
    work_paragraphs: tuple[str, ...],
    work_groups: tuple[tuple[str, ...], ...],
) -> str:
    if pass_definition.mode == "paragraph":
        return "zh2en: [%s] %d paragraph(s), one call per paragraph" % (
            pass_definition.name,
            len(work_groups),
        )

    return "zh2en: [%s] %d paragraph(s) in %d chunk(s)" % (
        pass_definition.name,
        len(work_paragraphs),
        len(work_groups),
    )


def resolve_work_groups(
    mode: str,
    work_paragraphs: tuple[str, ...],
    plan: tuple[tuple[str, ...], ...],
    pass_name: str,
    budget: int,
) -> tuple[tuple[tuple[str, ...], ...], str | None]:
    groups = regroup_by_plan(work_paragraphs, plan)
    if groups is not None:
        return groups, None

    warning = (
        "zh2en: [%s] paragraph count changed by a previous pass; "
        "grouping working text independently" % pass_name
    )
    if mode == "paragraph":
        return tuple((paragraph,) for paragraph in work_paragraphs), warning

    return make_chunks(work_paragraphs, budget), warning


def extract_message(body: Any) -> Result[tuple[Mapping[str, Any], Any], str]:
    try:
        message = body["choices"][0]["message"]
        return Ok((message, message["content"]))
    except (KeyError, IndexError, TypeError):
        return Err("unexpected response shape: %s" % str(body)[:500])


def collect_thinking_texts(part: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        entry["text"]
        for entry in (part.get("thinking") or [])
        if isinstance(entry, dict) and entry.get("text")
    )


def collect_content_parts(parts: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    def step(
        accumulator: tuple[tuple[str, ...], tuple[str, ...]], part: Any
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        texts, thoughts = accumulator
        if not isinstance(part, dict):
            return texts + (str(part),), thoughts

        if part.get("type") == "thinking":
            return texts, thoughts + collect_thinking_texts(part)

        if part.get("text"):
            return texts + (part["text"],), thoughts

        return accumulator

    initial: tuple[tuple[str, ...], tuple[str, ...]] = ((), ())
    return reduce(step, parts, initial)


def flatten_content_parts(content: Any) -> tuple[Any, tuple[str, ...]]:
    if not isinstance(content, list):
        return content, ()

    texts, thoughts = collect_content_parts(content)
    return "".join(texts), thoughts


def message_reasoning_texts(message: Mapping[str, Any]) -> tuple[str, ...]:
    def reasoning_at(key: str) -> tuple[str, ...]:
        value = message.get(key)
        return (value.rstrip(),) if isinstance(value, str) and value.strip() else ()

    return reduce(
        lambda found, key: found or reasoning_at(key),
        ("reasoning_content", "reasoning"),
        (),
    )


def parse_stream_line(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return None

    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None

    try:
        loaded = json.loads(data)
    except json.JSONDecodeError:
        return None

    return loaded if isinstance(loaded, dict) else None


def parse_chunk_delta(chunk: Mapping[str, Any]) -> dict[str, Any]:
    try:
        choices = chunk.get("choices")
        if not choices:
            return {}

        return choices[0].get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return {}


def delta_reasoning_text(delta: Mapping[str, Any]) -> str | None:
    def reasoning_at(key: str) -> str | None:
        value = delta.get(key)
        return value if isinstance(value, str) and value else None

    def step(found: str | None, key: str) -> str | None:
        return found if found is not None else reasoning_at(key)

    return reduce(step, ("reasoning_content", "reasoning"), None)


def delta_text(delta: Mapping[str, Any]) -> str:
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))

    return ""


@dataclass(frozen=True)
class StreamState:
    reasoning: tuple[str, ...] = ()
    contents: tuple[str, ...] = ()
    reported: dict[str, Any] | None = None
    counted: int = 0
    content_started: bool = False
    think: ThinkState = ThinkState()
    progress_request: tuple[str, int] | None = None


def stream_step(
    state: StreamState, chunk: Mapping[str, Any]
) -> Result[StreamState, str]:
    reported = (
        chunk["usage"] if isinstance(chunk.get("usage"), dict) else state.reported
    )
    delta = parse_chunk_delta(chunk)
    reasoning_text = delta_reasoning_text(delta)
    text = delta_text(delta)
    if not reasoning_text and not text:
        return Ok(replace(state, reported=reported, progress_request=None))

    think_state, thinking, visible = (
        think_step(state.think, text) if text else (state.think, False, "")
    )
    thinking_progress = 1 if reasoning_text and not state.content_started else 0
    text_progress = 1 if text else 0
    progress_request = None
    if thinking_progress or text_progress:
        progress_request = (
            ("Thinking" if thinking else "Working") if text else "Thinking",
            thinking_progress + text_progress,
        )

    return Ok(
        StreamState(
            reasoning=(
                state.reasoning + (reasoning_text,)
                if reasoning_text
                else state.reasoning
            ),
            contents=state.contents + (visible,) if visible else state.contents,
            reported=reported,
            counted=state.counted + thinking_progress + text_progress,
            content_started=state.content_started or bool(text),
            think=think_state,
            progress_request=progress_request,
        )
    )


@dataclass(frozen=True)
class ChatReply:
    content: Any
    reasoning: tuple[str, ...] = ()
    reported: dict[str, Any] | None = None
    counted: int = 0


def plain_reply(body: Any) -> Result[ChatReply, str]:
    message_result = extract_message(body)
    if isinstance(message_result, Err):
        return message_result

    message, content = message_result.value
    texts, thoughts = flatten_content_parts(content)
    return Ok(
        ChatReply(
            content=texts,
            reasoning=message_reasoning_texts(message) + thoughts,
            reported=body.get("usage") or {},
            counted=0,
        )
    )


def build_ascii_fix_user(source_paragraph: str, output_paragraph: str) -> str:
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Rewrite the translated paragraph as pure ASCII English."
        % (source_paragraph, output_paragraph)
    )


def build_ascii_retry_user(
    source_paragraph: str, output_paragraph: str, result: str
) -> str:
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Your previous reply still contained these non-ASCII "
        "characters: %s. Rewrite the translated paragraph again, "
        "inferring English for every one of them from the source and "
        "context. Reply with ASCII characters only."
        % (source_paragraph, output_paragraph, non_ascii_sample(result))
    )


def build_unit_retry_user(
    source_chunk: str,
    context: tuple[str, ...],
    bad_output: str,
    problem: str,
) -> str:
    return "\n\n".join(
        part
        for part in (
            "Your previous reply below does not satisfy the output rules: %s."
            % problem,
            "Previous reply:\n%s" % bad_output,
            *context_parts(context),
            source_chunk,
            "Translate the source text again, fixing the problem; output only "
            "the translation.",
        )
        if part
    )


def drop_non_ascii(
    text: str, character_map: Mapping[str, str], attempts: int, index: int
) -> tuple[str, str | None]:
    if text.isascii():
        return text, None

    fallback = to_ascii_mechanical(text, character_map)
    if fallback.isascii():
        return fallback, None

    return (
        re.sub(
            r"  +", " ", "".join(character for character in text if character.isascii())
        ),
        (
            "zh2en: ascii: warning: paragraph %d still contained "
            "non-ASCII characters (%s) after %d LLM attempts; dropping "
            "them" % (index + 1, non_ascii_sample(text), attempts)
        ),
    )


def now(clock: Callable[[], float]) -> IO[float]:
    return IO(clock)


def read_stdin(stream: TextIO) -> IO[str]:
    return IO(stream.read)


def write_stdout(stream: TextIO, text: str) -> IO[None]:
    def thunk() -> None:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")

    return IO(thunk)


def path_exists(path: str) -> IO[bool]:
    return IO(lambda: os.path.exists(path))


def cwd() -> IO[str]:
    return IO(os.getcwd)


def user_config_path(environment: Mapping[str, str]) -> IO[str]:
    return IO(
        lambda: os.path.join(
            environment.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
            "zh2en",
            "config.toml",
        )
    )


def resolve_cache_dir(
    environment: Mapping[str, str], override: str | None = None
) -> IO[str]:
    def thunk() -> str:
        if override:
            directory = override
        else:
            directory = os.path.join(
                environment.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"),
                "zh2en",
            )

        os.makedirs(directory, exist_ok=True)
        return directory

    return IO(thunk)


def load_toml(path: str, description: str) -> IO[Result[dict[str, Any], str]]:
    def thunk() -> Result[dict[str, Any], str]:
        try:
            with open(path, "rb") as handle:
                return Ok(tomllib.load(handle))
        except FileNotFoundError:
            return Err("%s not found: %s" % (description, path))
        except OSError as error:
            return Err("cannot read %s: %s" % (description, error))
        except tomllib.TOMLDecodeError as error:
            return Err("cannot parse %s %s: %s" % (description, path, error))

    return IO(thunk)


def cache_read(cache_directory: str, key: str) -> IO[str | None]:
    def thunk() -> str | None:
        try:
            with open(cache_path(cache_directory, key), encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return None

    return IO(thunk)


def cache_write(cache_directory: str, key: str, value: str) -> IO[None]:
    def thunk() -> None:
        path = cache_path(cache_directory, key)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(value)

        os.replace(temporary_path, path)

    return IO(thunk)


def log_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class RunLog:
    path: str


def run_log_path(cache_directory: str) -> str:
    return os.path.join(
        cache_directory,
        "logs",
        "%s-%d.log" % (time.strftime("%Y%m%d-%H%M%S", time.gmtime()), os.getpid()),
    )


def open_run_log(cache_directory: str) -> IO[RunLog]:
    def thunk() -> RunLog:
        path = run_log_path(cache_directory)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return RunLog(path=path)

    return IO(thunk)


def run_log_write(log: RunLog, text: str) -> IO[None]:
    def thunk() -> None:
        if not log.path:
            return

        try:
            with open(log.path, "a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            pass

    return IO(thunk)


def log_entry(log: RunLog, label: str, body: str) -> IO[None]:
    return run_log_write(log, "== %s %s\n%s\n\n" % (log_stamp(), label, body))


def log_request(
    log: RunLog, payload: Mapping[str, Any], label: str = "REQUEST"
) -> IO[None]:
    return log_entry(log, label, json.dumps(payload, indent=2, ensure_ascii=False))


def log_error(log: RunLog, detail: str) -> IO[None]:
    return log_entry(log, "ERROR", detail)


def http_request(
    config: Config, payload: Mapping[str, Any], accept: str | None = None
) -> urllib.request.Request:
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + config.api_key,
    }
    if accept:
        headers = {**headers, "Accept": accept}

    return urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def urllib_open(request: Any, timeout: float) -> Result[Any, str]:
    try:
        return Ok(urllib.request.urlopen(request, timeout=timeout))
    except urllib.error.HTTPError as error:
        return Err(
            "HTTP %s from endpoint: %s"
            % (error.code, error.read().decode("utf-8", "replace")[:500])
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return Err("could not reach endpoint: %s" % error)


def http_post_json(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[dict[str, Any], str]]:
    def thunk() -> Result[dict[str, Any], str]:
        log_request(ctx.log, payload).run()
        opened = ctx.open_http(http_request(ctx.config, payload), ctx.config.timeout)
        if isinstance(opened, Err):
            log_error(ctx.log, opened.error).run()
            return opened

        with opened.value as response:
            try:
                body = response.read().decode("utf-8")
                log_entry(ctx.log, "RESPONSE", body).run()
                return Ok(json.loads(body))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                log_error(ctx.log, "invalid JSON response: %s" % error).run()
                return Err("invalid JSON response: %s" % error)

    return IO(thunk)


def http_open_stream(ctx: Context, payload: Mapping[str, Any]) -> IO[Result[Any, str]]:
    def open_stream(body: Mapping[str, Any], label: str) -> Result[Any, str]:
        log_request(ctx.log, body, label).run()
        opened = ctx.open_http(
            http_request(ctx.config, body, accept="text/event-stream"),
            ctx.config.timeout,
        )
        if isinstance(opened, Err):
            log_error(ctx.log, opened.error).run()

        return opened

    def thunk() -> Result[Any, str]:
        opened = open_stream(payload, "REQUEST (stream)")
        if (
            isinstance(opened, Ok)
            or "stream_options" not in payload
            or "stream_options" not in str(opened.error)
        ):
            return opened

        return open_stream(
            {key: value for key, value in payload.items() if key != "stream_options"},
            "REQUEST (stream, retry)",
        )

    return IO(thunk)


def drive_stream(
    response: Any,
    on_progress: ProgressCallback,
    on_raw_line: Callable[[bytes], None] | None = None,
) -> Result[StreamState, str]:
    state = StreamState()
    for raw_line in response:
        if on_raw_line is not None:
            on_raw_line(raw_line)

        outcome = step_stream(state, raw_line)
        if isinstance(outcome, Err):
            return outcome

        state = outcome.value
        if state.progress_request is not None:
            label, count = state.progress_request
            on_progress(label, count)

    return Ok(state)


def collect_stream(
    ctx: Context, payload: Mapping[str, Any], on_progress: ProgressCallback
) -> IO[Result[StreamState, str]]:
    def on_raw_line(raw_line: bytes) -> None:
        run_log_write(ctx.log, raw_line.decode("utf-8", "replace")).run()

    def respond(opened: Result[Any, str]) -> IO[Result[StreamState, str]]:
        if isinstance(opened, Err):
            return io_result(opened)

        def thunk() -> Result[StreamState, str]:
            try:
                with opened.value as response:
                    outcome = drive_stream(response, on_progress, on_raw_line)
                    if isinstance(outcome, Err):
                        log_error(ctx.log, outcome.error).run()
                    else:
                        run_log_write(ctx.log, "\n").run()

                    return outcome
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                log_error(ctx.log, "stream interrupted: %s" % error).run()
                return Err("stream interrupted: %s" % error)

        return IO(thunk)

    return io_bind(http_open_stream(ctx, payload), respond)


def step_stream(state: StreamState, raw_line: bytes) -> Result[StreamState, str]:
    chunk = parse_stream_line(raw_line)
    if chunk is None:
        return Ok(state)

    if isinstance(chunk.get("error"), dict):
        return Err("endpoint stream error: %s" % str(chunk["error"])[:500])

    return stream_step(state, chunk)


def verbose_log(ctx: Context, message: str) -> IO[None]:
    return ctx.console.log(message) if ctx.verbose else io_pure(None)


def log_all(console: Console, messages: tuple[str, ...]) -> IO[Result[tuple[()], str]]:
    def step(_: tuple[()], message: str) -> IO[Result[tuple[()], str]]:
        return io_map(console.log(message), lambda _: Ok(()))

    return fold_io(messages, step, Ok(()))


def chat(
    ctx: Context,
    system: str,
    user: str,
    model: str,
    params: Mapping[str, Any],
    usage: Usage,
) -> IO[Result[tuple[str, Usage], str]]:
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > ctx.config.max_tokens:
        return io_result(
            Err(
                "request is ~%d tokens, over the %d-token budget; raise "
                "api.max_tokens (--max-tokens / TRANSLATE_MAX_TOKENS) or "
                "shorten the input" % (estimated, ctx.config.max_tokens)
            )
        )

    payload = build_chat_payload(model, system, user, params)

    def stopped(
        reply_result: Result[ChatReply, str],
    ) -> IO[Result[tuple[str, Usage], str]]:
        return io_bind(
            ctx.console.stop(),
            lambda _: conclude_chat(ctx, reply_result, usage, estimated),
        )

    return io_bind(
        ctx.console.start("Working"),
        lambda _: io_bind(
            (
                streamed_call(ctx, payload)
                if payload.get("stream", True)
                else plain_call(ctx, payload)
            ),
            stopped,
        ),
    )


def streamed_call(
    ctx: Context, payload: Mapping[str, Any]
) -> IO[Result[ChatReply, str]]:
    full_payload = {**payload, "stream": True}
    if "stream_options" not in full_payload:
        full_payload = {**full_payload, "stream_options": {"include_usage": True}}

    def on_progress(label: str, count: int) -> None:
        ctx.console.progress(label, count).run()

    def to_reply(state: StreamState) -> ChatReply:
        return ChatReply(
            content="".join(state.contents)
            + (
                state.think.held
                if state.think.checking and not state.think.open
                else ""
            ),
            reasoning=state.reasoning,
            reported=state.reported,
            counted=state.counted,
        )

    return io_map(
        collect_stream(ctx, full_payload, on_progress),
        lambda outcome: result_map(outcome, to_reply),
    )


def plain_call(ctx: Context, payload: Mapping[str, Any]) -> IO[Result[ChatReply, str]]:
    return io_bind(
        http_post_json(ctx, payload),
        lambda body_result: io_result(result_bind(body_result, plain_reply)),
    )


def conclude_chat(
    ctx: Context, reply_result: Result[ChatReply, str], usage: Usage, estimated: int
) -> IO[Result[tuple[str, Usage], str]]:
    if isinstance(reply_result, Err):
        return io_result(reply_result)

    reply = reply_result.value
    content, think_text = strip_think_tag(reply.content)
    if not isinstance(content, str):
        return io_result(Err("unexpected content type: %s" % type(content).__name__))

    messages = (
        tuple(
            text.rstrip()
            for text in reply.reasoning + ((think_text,) if think_text else ())
            if text.strip()
        )
        if ctx.verbose
        else ()
    )

    def finish(_: Result[tuple[()], str]) -> Result[tuple[str, Usage], str]:
        reported = reply.reported
        return Ok(
            (
                content,
                (
                    add_usage(usage, reported)
                    if reported
                    else add_usage(
                        usage,
                        {
                            "prompt_tokens": estimated,
                            "completion_tokens": reply.counted,
                        },
                    )
                ),
            )
        )

    if messages:
        return io_map(log_all(ctx.console, messages), finish)

    return io_result(finish(Ok(())))


def run_analysis(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    usage: Usage,
) -> IO[Result[tuple[str, Usage], str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return io_map(
        chat(ctx, pass_definition.instruction, text, model, params, usage),
        lambda result: result_map(result, lambda pair: (pair[0].strip(), pair[1])),
    )


def analyze_document(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[tuple[str, Usage], str]]:
    budget = max(
        ctx.config.max_tokens
        - estimate_tokens(pass_definition.instruction)
        - ctx.settings.analysis_reserve_tokens,
        1,
    )
    chunks = make_chunks(
        split_units_to_budget(
            split_paragraphs(full_text)[0],
            budget,
            ctx.settings.sentence_boundary_characters,
        ),
        budget,
    )
    if len(chunks) > 1:
        return io_result(
            Err(
                "document requires analysis in %d parts, over the "
                "%d-token budget; raise api.max_tokens (--max-tokens / "
                "TRANSLATE_MAX_TOKENS) or shorten the input"
                % (len(chunks), ctx.config.max_tokens)
            )
        )

    return run_analysis(ctx, pass_definition, full_text, usage)


def run_analysis_once(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[tuple[str, Usage], str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    key = cache_key(full_text, model, pass_salt(pass_definition), overrides=params)

    def compute(current_usage: Usage) -> IO[Result[tuple[str, Usage], str]]:
        def store(
            result: Result[tuple[str, Usage], str],
        ) -> IO[Result[tuple[str, Usage], str]]:
            if isinstance(result, Err):
                return io_result(result)

            analysis, new_usage = result.value
            return io_map(
                (
                    cache_write(ctx.cache_directory, key, analysis)
                    if ctx.use_cache
                    else io_pure(None)
                ),
                lambda _: Ok((analysis, new_usage)),
            )

        return io_bind(
            analyze_document(ctx, pass_definition, full_text, current_usage), store
        )

    if not ctx.use_cache:
        return compute(usage)

    def use_cached(cached: str | None) -> IO[Result[tuple[str, Usage], str]]:
        if cached is None:
            return compute(usage)

        return io_map(
            verbose_log(ctx, "zh2en: [%s] cache hit" % pass_definition.name),
            lambda _: Ok((cached, usage)),
        )

    return io_bind(cache_read(ctx.cache_directory, key), use_cached)


def translate_chunk(
    ctx: Context,
    pass_definition: PassDefinition,
    source_chunk: str,
    work_chunk: str | None,
    analysis: str | None,
    usage: Usage,
    context: tuple[str, ...] = (),
) -> IO[Result[tuple[str, Usage], str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return chat(
        ctx,
        pass_definition.instruction,
        build_pass_user(source_chunk, work_chunk, analysis, context),
        model,
        params,
        usage,
    )


def ascii_fix_llm(
    ctx: Context,
    pass_definition: PassDefinition,
    source_paragraph: str,
    output_paragraph: str,
    usage: Usage,
) -> IO[Result[tuple[str, Usage], str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    key = cache_key(
        source_paragraph,
        model,
        "\x00".join(
            (
                CACHE_SALT_VERSION,
                "ascii-fix",
                pass_definition.name,
                ctx.settings.ascii_fix_instruction,
            )
        ),
        output_paragraph,
        overrides=params,
    )

    def attempt(
        index: int, user: str, last_result: str, current_usage: Usage
    ) -> IO[Result[tuple[str, Usage], str]]:
        if index > ctx.settings.ascii_fix_attempts:
            return io_result(Ok((last_result, current_usage)))

        return io_bind(
            chat(
                ctx,
                ctx.settings.ascii_fix_instruction,
                user,
                model,
                params,
                current_usage,
            ),
            lambda result: result_bind_io(
                result, lambda pair: ascii_outcome(index, pair[0], pair[1])
            ),
        )

    def ascii_outcome(
        index: int, result: str, current_usage: Usage
    ) -> IO[Result[tuple[str, Usage], str]]:
        if result.isascii():
            return io_map(
                (
                    cache_write(ctx.cache_directory, key, result)
                    if ctx.use_cache
                    else io_pure(None)
                ),
                lambda _: Ok((result, current_usage)),
            )

        return io_bind(
            verbose_log(
                ctx,
                "zh2en: ascii: attempt %d/%d still non-ASCII; retrying"
                % (index, ctx.settings.ascii_fix_attempts),
            ),
            lambda _: attempt(
                index + 1,
                build_ascii_retry_user(source_paragraph, output_paragraph, result),
                result,
                current_usage,
            ),
        )

    def start(current_usage: Usage) -> IO[Result[tuple[str, Usage], str]]:
        return attempt(
            1,
            build_ascii_fix_user(source_paragraph, output_paragraph),
            "",
            current_usage,
        )

    if not ctx.use_cache:
        return start(usage)

    def use_cached(cached: str | None) -> IO[Result[tuple[str, Usage], str]]:
        if cached is None or not cached.isascii():
            return start(usage)

        return io_map(
            verbose_log(ctx, "zh2en: ascii: cache hit"),
            lambda _: Ok((cached, usage)),
        )

    return io_bind(cache_read(ctx.cache_directory, key), use_cached)


def repair_paragraph(
    ctx: Context,
    pass_definition: PassDefinition,
    paragraph: str,
    separator: str,
    index: int,
    total: int,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
) -> IO[Result[tuple[str, Usage], str]]:
    mechanical = to_ascii_mechanical(paragraph, ctx.settings.ascii_character_map)
    if mechanical.isascii():
        return io_map(
            verbose_log(
                ctx,
                "zh2en: ascii: paragraph %d/%d converted mechanically"
                % (index + 1, total),
            ),
            lambda _: Ok((mechanical + separator, usage)),
        )

    def repaired(
        result: Result[tuple[str, Usage], str],
    ) -> IO[Result[tuple[str, Usage], str]]:
        if isinstance(result, Err):
            return io_result(result)

        final, warning = drop_non_ascii(
            result.value[0],
            ctx.settings.ascii_character_map,
            ctx.settings.ascii_fix_attempts,
            index,
        )

        def emit(_: None) -> Result[tuple[str, Usage], str]:
            return Ok((final + separator, result.value[1]))

        return (
            io_map(ctx.console.log(warning), emit) if warning else io_result(emit(None))
        )

    return io_bind(
        verbose_log(
            ctx,
            "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
            "LLM to repair it" % (index + 1, total),
        ),
        lambda _: io_bind(
            ascii_fix_llm(
                ctx,
                pass_definition,
                (
                    source_paragraphs[index]
                    if index < len(source_paragraphs)
                    else "(unavailable)"
                ),
                paragraph,
                usage,
            ),
            repaired,
        ),
    )


def ensure_ascii_output(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
) -> IO[Result[tuple[str, Usage], str]]:
    paragraphs, separators = split_paragraphs(text)
    accumulator_type = tuple[tuple[str, ...], Usage]

    def step(
        accumulator: accumulator_type, indexed: tuple[int, str]
    ) -> IO[Result[accumulator_type, str]]:
        outputs, current_usage = accumulator
        index, paragraph = indexed
        separator = separators[index] if index < len(separators) else ""
        if paragraph.isascii():
            return io_result(Ok((outputs + (paragraph + separator,), current_usage)))

        return io_map(
            repair_paragraph(
                ctx,
                pass_definition,
                paragraph,
                separator,
                index,
                len(paragraphs),
                source_paragraphs,
                current_usage,
            ),
            lambda result: result_map(
                result, lambda pair: (outputs + (pair[0],), pair[1])
            ),
        )

    initial: Result[accumulator_type, str] = Ok(((), usage))
    return io_map(
        fold_io(enumerate(paragraphs), step, initial),
        lambda result: result_map(result, lambda pair: ("".join(pair[0]), pair[1])),
    )


def enforce_pass_ascii(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
    started_at: float,
    clock: Callable[[], float],
) -> IO[Result[tuple[str, Usage], str]]:
    def conclude(
        result: Result[tuple[str, Usage], str],
    ) -> IO[Result[tuple[str, Usage], str]]:
        if isinstance(result, Err):
            return io_map(
                ctx.console.interrupt(),
                lambda _: Err(
                    "zh2en: pass [%s] ascii enforcement failed: %s"
                    % (pass_definition.name, result.error)
                ),
            )

        fixed, new_usage = result.value
        prompt, completion, cost = usage_delta(usage, new_usage)

        def report(ended_at: float) -> IO[Result[tuple[str, Usage], str]]:
            return io_map(
                ctx.console.finish(
                    usage_line("Done", ended_at - started_at, prompt, completion, cost)
                    if prompt or completion or cost
                    else "Done."
                ),
                lambda _: Ok((fixed, new_usage)),
            )

        return io_bind(now(clock), report)

    return io_bind(
        io_bind(
            ctx.console.write_partial("Enforcing ASCII... "),
            lambda _: ensure_ascii_output(
                ctx, pass_definition, text, source_paragraphs, usage
            ),
        ),
        conclude,
    )


def log_stage(
    console: Console,
    label: str,
    started_at: float,
    ended_at: float,
    start_usage: Usage,
    end_usage: Usage,
) -> IO[None]:
    prompt, completion, cost = usage_delta(start_usage, end_usage)
    return console.log(
        usage_line(label, ended_at - started_at, prompt, completion, cost)
    )


def conclude_state(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs: tuple[str, ...],
    clock: Callable[[], float],
) -> IO[Result[State, str]]:
    if not pass_definition.ascii:
        return io_result(Ok(state))

    def enforce(ascii_started: float) -> IO[Result[State, str]]:
        return io_map(
            enforce_pass_ascii(
                ctx,
                pass_definition,
                state.text,
                source_paragraphs,
                state.usage,
                ascii_started,
                clock,
            ),
            lambda result: result_map(
                result, lambda pair: State(pair[0], state.analysis, pair[1])
            ),
        )

    return io_bind(now(clock), enforce)


def run_analysis_pass(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    clock: Callable[[], float],
) -> IO[Result[State, str]]:
    def after_analysis(
        analysis_result: Result[tuple[str, Usage], str],
    ) -> IO[Result[State, str]]:
        if isinstance(analysis_result, Err):
            return io_result(
                Err(
                    "zh2en: pass [%s] failed: %s"
                    % (pass_definition.name, analysis_result.error)
                )
            )

        analysis, analysis_usage = analysis_result.value

        def after_stage(ended_at: float) -> IO[Result[State, str]]:
            return io_bind(
                log_stage(
                    ctx.console,
                    "Done",
                    started_at,
                    ended_at,
                    state.usage,
                    analysis_usage,
                ),
                lambda _: conclude_state(
                    ctx,
                    pass_definition,
                    State(text=state.text, analysis=analysis, usage=analysis_usage),
                    source_paragraphs,
                    clock,
                ),
            )

        return io_bind(now(clock), after_stage)

    return io_bind(
        verbose_log(
            ctx,
            "zh2en: [%s] whole-document analysis (%d characters, ~%d tokens)"
            % (pass_definition.name, len(state.text), estimate_tokens(state.text)),
        ),
        lambda _: io_bind(
            run_analysis_once(ctx, pass_definition, state.text, state.usage),
            after_analysis,
        ),
    )


def run_units(
    ctx: Context,
    pass_definition: PassDefinition,
    work_groups: tuple[tuple[str, ...], ...],
    plan: tuple[tuple[str, ...], ...],
    trailing_separators: tuple[str, ...],
    analysis: str | None,
    usage: Usage,
) -> IO[Result[tuple[tuple[str, ...], Usage], str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    total = len(work_groups)
    accumulator_type = tuple[tuple[str, ...], Usage]
    outcome_type = tuple[str, bool, Usage]
    flat_plan = tuple(chain.from_iterable(plan))
    plan_starts = tuple(accumulate(map(len, plan), initial=0))

    def neighbour_context(index: int) -> tuple[str, ...]:
        if pass_definition.mode != "paragraph" or index >= len(plan):
            return ()

        start = plan_starts[index]
        end = plan_starts[index + 1]
        return (flat_plan[start - 1 : start] if start > 0 else ()) + (
            flat_plan[end : end + 1] if end < len(flat_plan) else ()
        )

    def step(
        accumulator: accumulator_type, indexed: tuple[int, tuple[str, ...]]
    ) -> IO[Result[accumulator_type, str]]:
        outputs, current_usage = accumulator
        index, work_group = indexed
        source_chunk_text = "\n\n".join(plan[index]) if index < len(plan) else ""
        work_chunk_text = "\n\n".join(work_group)
        context = neighbour_context(index)
        key = cache_key(
            source_chunk_text,
            model,
            pass_salt(pass_definition),
            work_chunk_text,
            overrides=params,
            context="\n\n".join(context),
        )
        trailing_separator = (
            trailing_separators[index] if index < len(trailing_separators) else ""
        )

        def note(message: str) -> IO[None]:
            return verbose_log(ctx, message)

        if not source_chunk_text:
            return io_map(
                note(
                    "zh2en: [%s] unit %d/%d has no matching source; "
                    "passing it through unchanged"
                    % (pass_definition.name, index + 1, total)
                ),
                lambda _: Ok(
                    (outputs + (work_chunk_text + trailing_separator,), current_usage)
                ),
            )

        def assess_initial(
            translated: str, unit_usage: Usage
        ) -> IO[Result[outcome_type, str]]:
            problem = unit_output_problem(source_chunk_text, translated, ctx.settings)
            if problem is None:
                return io_result(Ok((translated, True, unit_usage)))

            return repair(1, translated, unit_usage, problem)

        def repair(
            attempt_index: int,
            bad_output: str,
            unit_usage: Usage,
            problem: str,
        ) -> IO[Result[outcome_type, str]]:
            if attempt_index > ctx.settings.unit_fix_attempts:
                return io_map(
                    ctx.console.log(
                        "zh2en: [%s] unit %d/%d failed validation %d time(s); "
                        "last problem: %s. Keeping the last reply, uncached"
                        % (
                            pass_definition.name,
                            index + 1,
                            total,
                            ctx.settings.unit_fix_attempts,
                            problem,
                        )
                    ),
                    lambda _: Ok((bad_output, False, unit_usage)),
                )

            return io_bind(
                verbose_log(
                    ctx,
                    "zh2en: [%s] unit %d/%d failed validation (%s); repair "
                    "attempt %d/%d"
                    % (
                        pass_definition.name,
                        index + 1,
                        total,
                        problem,
                        attempt_index,
                        ctx.settings.unit_fix_attempts,
                    ),
                ),
                lambda _: io_bind(
                    chat(
                        ctx,
                        pass_definition.instruction,
                        build_unit_retry_user(
                            source_chunk_text, context, bad_output, problem
                        ),
                        model,
                        params,
                        unit_usage,
                    ),
                    settled(attempt_index),
                ),
            )

        def settled(
            attempt_index: int,
        ) -> Callable[[Result[tuple[str, Usage], str]], IO[Result[outcome_type, str]]]:
            def continue_after(
                reply_result: Result[tuple[str, Usage], str],
            ) -> IO[Result[outcome_type, str]]:
                if isinstance(reply_result, Err):
                    return io_result(reply_result)

                translated, new_usage = reply_result.value
                problem = unit_output_problem(
                    source_chunk_text, translated, ctx.settings
                )
                if problem is None:
                    return io_result(Ok((translated, True, new_usage)))

                return repair(attempt_index + 1, translated, new_usage, problem)

            return continue_after

        def assessed(
            reply_result: Result[tuple[str, Usage], str],
        ) -> IO[Result[outcome_type, str]]:
            if isinstance(reply_result, Err):
                return io_result(reply_result)

            translated, new_usage = reply_result.value
            return assess_initial(translated, new_usage)

        def translate(
            unit_usage: Usage,
        ) -> IO[Result[accumulator_type, str]]:
            def store(
                result: Result[outcome_type, str],
            ) -> IO[Result[accumulator_type, str]]:
                if isinstance(result, Err):
                    return io_result(
                        Err(
                            "zh2en: [%s] failed on unit %d: %s"
                            % (pass_definition.name, index + 1, result.error)
                        )
                    )

                translated, valid, new_usage = result.value
                return io_map(
                    io_bind(
                        (
                            cache_write(ctx.cache_directory, key, translated)
                            if valid and ctx.use_cache
                            else io_pure(None)
                        ),
                        lambda _: note(
                            "zh2en: [%s] unit %d/%d done"
                            % (pass_definition.name, index + 1, total)
                        ),
                    ),
                    lambda _: Ok(
                        (
                            outputs + (translated + trailing_separator,),
                            new_usage,
                        )
                    ),
                )

            return io_bind(
                io_bind(
                    translate_chunk(
                        ctx,
                        pass_definition,
                        source_chunk_text,
                        work_chunk_text,
                        analysis,
                        unit_usage,
                        context,
                    ),
                    assessed,
                ),
                store,
            )

        if not ctx.use_cache:
            return translate(current_usage)

        def with_cached(cached: str | None) -> IO[Result[accumulator_type, str]]:
            if cached is not None:
                return io_map(
                    note(
                        "zh2en: [%s] unit %d/%d cache hit"
                        % (pass_definition.name, index + 1, total)
                    ),
                    lambda _: Ok(
                        (outputs + (cached + trailing_separator,), current_usage)
                    ),
                )

            return translate(current_usage)

        return io_bind(cache_read(ctx.cache_directory, key), with_cached)

    initial: Result[accumulator_type, str] = Ok(((), usage))
    return fold_io(enumerate(work_groups), step, initial)


def run_text_pass_once(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
    clock: Callable[[], float],
) -> IO[Result[State, str]]:
    plan = paragraph_plan if pass_definition.mode == "paragraph" else chunk_plan
    work_paragraphs, _ = split_paragraphs(state.text)
    work_groups, warning = resolve_work_groups(
        pass_definition.mode,
        work_paragraphs,
        plan,
        pass_definition.name,
        ctx.settings.chunk_budget_tokens,
    )

    def proceed(_: None) -> IO[Result[State, str]]:
        def after_units(
            units_result: Result[tuple[tuple[str, ...], Usage], str],
        ) -> IO[Result[State, str]]:
            if isinstance(units_result, Err):
                return io_result(units_result)

            next_state = State(
                text=ensure_blank_line_separators("".join(units_result.value[0])),
                analysis=state.analysis,
                usage=units_result.value[1],
            )

            def after_stage(ended_at: float) -> IO[Result[State, str]]:
                return io_bind(
                    log_stage(
                        ctx.console,
                        "Done",
                        started_at,
                        ended_at,
                        state.usage,
                        next_state.usage,
                    ),
                    lambda _: conclude_state(
                        ctx, pass_definition, next_state, source_paragraphs, clock
                    ),
                )

            return io_bind(now(clock), after_stage)

        return io_bind(
            verbose_log(
                ctx, plan_info_message(pass_definition, work_paragraphs, work_groups)
            ),
            lambda _: io_bind(
                run_units(
                    ctx,
                    pass_definition,
                    work_groups,
                    plan,
                    unit_separators(plan, separators),
                    state.analysis,
                    state.usage,
                ),
                after_units,
            ),
        )

    return io_bind(ctx.console.log(warning) if warning else io_pure(None), proceed)


def run_text_pass(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
    clock: Callable[[], float],
) -> IO[Result[State, str]]:
    def attempt(
        pass_to_run: PassDefinition, current_state: State, attempt_started: float
    ) -> IO[Result[State, str]]:
        return run_text_pass_once(
            ctx,
            pass_to_run,
            current_state,
            attempt_started,
            source_paragraphs,
            separators,
            chunk_plan,
            paragraph_plan,
            clock,
        )

    if not ctx.ensure_paragraphs:
        return attempt(pass_definition, state, started_at)

    def mismatch_message(count: int, action: str) -> str:
        return "zh2en: [%s] output has %d paragraph(s), source has %d; %s" % (
            pass_definition.name,
            count,
            len(source_paragraphs),
            action,
        )

    def after_retry(result: Result[State, str]) -> IO[Result[State, str]]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if count == len(source_paragraphs):
            return io_result(result)

        return io_map(
            ctx.console.log(
                "zh2en: [%s] paragraph count still differs (%d vs %d); continuing"
                % (pass_definition.name, count, len(source_paragraphs))
            ),
            lambda _: result,
        )

    def check(result: Result[State, str]) -> IO[Result[State, str]]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if count == len(source_paragraphs):
            return io_result(result)

        if pass_definition.mode != "chunk":
            return io_map(
                ctx.console.log(mismatch_message(count, "continuing")),
                lambda _: result,
            )

        return io_bind(
            ctx.console.log(
                mismatch_message(
                    count, "re-running the pass with one call per paragraph"
                )
            ),
            lambda _: io_bind(
                attempt(
                    replace(pass_definition, mode="paragraph"),
                    replace(state, usage=result.value.usage),
                    clock(),
                ),
                after_retry,
            ),
        )

    return io_bind(attempt(pass_definition, state, started_at), check)


def run_pass(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
    clock: Callable[[], float],
) -> IO[Result[State, str]]:
    if pass_definition.mode == "analysis":
        return run_analysis_pass(
            ctx, pass_definition, state, started_at, source_paragraphs, clock
        )

    return run_text_pass(
        ctx,
        pass_definition,
        state,
        started_at,
        source_paragraphs,
        separators,
        chunk_plan,
        paragraph_plan,
        clock,
    )


def run_passes(
    ctx: Context,
    pass_definitions: tuple[PassDefinition, ...],
    state: State,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
    clock: Callable[[], float],
) -> IO[Result[State, str]]:
    def step(
        current_state: State, indexed: tuple[int, PassDefinition]
    ) -> IO[Result[State, str]]:
        number, pass_definition = indexed

        def launch(stage_started: float) -> IO[Result[State, str]]:
            return io_bind(
                ctx.console.log(
                    "Starting pass %d/%d [%s]..."
                    % (number, len(pass_definitions), pass_definition.name)
                ),
                lambda _: run_pass(
                    ctx,
                    pass_definition,
                    current_state,
                    stage_started,
                    source_paragraphs,
                    separators,
                    chunk_plan,
                    paragraph_plan,
                    clock,
                ),
            )

        return io_bind(now(clock), launch)

    return fold_io(enumerate(pass_definitions, 1), step, Ok(state))


def load_instruction_text(
    path: str, name: str, table: dict[str, Any], base_directory: str
) -> IO[Result[str, str]]:
    has_inline = "instruction" in table
    if ("instruction_file" in table) == has_inline:
        return io_result(
            Err(
                "%s: [[pass]] %s: exactly one of instruction_file or instruction "
                "is required" % (path, name)
            )
        )

    if has_inline:
        instruction = table["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            return io_result(
                Err("%s: [[pass]] %s: instruction is empty" % (path, name))
            )

        return io_result(Ok(instruction.strip()))

    instruction_file = table["instruction_file"]
    if not isinstance(instruction_file, str) or not instruction_file.strip():
        return io_result(
            Err("%s: [[pass]] %s: instruction_file must be a path" % (path, name))
        )

    def thunk() -> Result[str, str]:
        try:
            with open(
                os.path.join(base_directory, instruction_file), encoding="utf-8"
            ) as handle:
                instruction = handle.read().strip()
        except OSError as error:
            return Err(
                "%s: [[pass]] %s: cannot read instruction file: %s"
                % (path, name, error)
            )

        if not instruction:
            return Err("%s: [[pass]] %s: instruction file is empty" % (path, name))

        return Ok(instruction)

    return IO(thunk)


def parse_pass_table(
    path: str, table: Any, base_directory: str
) -> IO[Result[PassDefinition, str]]:
    if not isinstance(table, dict):
        return io_result(Err("%s: [[pass]] entries must be tables" % path))

    unknown = sorted(set(table) - set(PASS_KEYS))
    if unknown:
        return io_result(
            Err("%s: [[pass]]: unknown key(s): %s" % (path, ", ".join(unknown)))
        )

    name = table.get("name")
    if not isinstance(name, str) or not name.strip():
        return io_result(Err("%s: [[pass]]: name must be a non-empty string" % path))

    return io_bind(
        load_instruction_text(path, name.strip(), table, base_directory),
        lambda instruction_result: io_result(
            result_bind(
                instruction_result,
                lambda instruction: pass_definition_from(
                    path, name.strip(), table, instruction
                ),
            )
        ),
    )


def document_passes(
    path: str, document: dict[str, Any]
) -> IO[Result[tuple[PassDefinition, ...], str]]:
    entries = document.get("pass")
    if entries is None:
        return io_result(Ok(()))

    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        return io_result(Err("%s: [[pass]] must define one or more pass tables" % path))

    return io_map(
        io_sequence(
            parse_pass_table(path, entry, os.path.dirname(os.path.abspath(path)))
            for entry in entries
        ),
        results_sequence,
    )


def resolve_passes(
    user_path: str,
    user_document_result: Result[dict[str, Any], str],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], str],
) -> IO[Result[tuple[dict[str, bool], tuple[PassDefinition, ...]], str]]:
    pair_type = tuple[dict[str, bool], tuple[PassDefinition, ...]]

    def resolve(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> IO[Result[pair_type, str]]:
        user_document, selected_document = documents
        sources: tuple[tuple[str | None, dict[str, Any]], ...] = ()
        if selected_document:
            sources += ((selected_path, selected_document),)

        if user_document:
            sources += ((user_path, user_document),)

        candidates = tuple(
            (path, document) for path, document in sources if "pass" in document
        )
        if not candidates:
            return io_result(
                Err(
                    "no [[pass]] tables found; define at least one pass in %s"
                    % (selected_path or user_path or "a config file (see --help)")
                )
            )

        option_sources = tuple(
            (path, document) for path, document in sources if "options" in document
        )
        options: Result[dict[str, bool], str]
        if option_sources:
            option_path, option_document = option_sources[0]
            options = parse_options_table(option_path or "", option_document["options"])
        else:
            options = Ok({})

        pass_path, pass_document = candidates[0]

        def combine(
            result: Result[tuple[PassDefinition, ...], str],
        ) -> Result[pair_type, str]:
            return result_bind(
                options,
                lambda option_values: result_map(
                    result, lambda passes: (option_values, passes)
                ),
            )

        return io_map(document_passes(pass_path or "", pass_document), combine)

    return result_bind_io(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        resolve,
    )


def read_document(
    path: str | None, description: str, exists: bool
) -> IO[Result[dict[str, Any], str]]:
    if not exists:
        return io_result(Ok({}))

    return io_map(
        load_toml(path or "", description),
        lambda result: result_bind(
            result,
            lambda document: result_map(
                validate_document(path or "", document), lambda _: document
            ),
        ),
    )


def resolve_config_path(
    requested: str | None, environment: Mapping[str, str]
) -> IO[str | None]:
    if requested:
        return io_pure(requested)

    configured = environment.get("TRANSLATE_CONFIG", "").strip()
    if configured:
        return io_pure(configured)

    def pick(cwd_value: str, user_path: str) -> str | None:
        local = os.path.join(cwd_value, "zh2en.toml")
        if os.path.exists(local):
            return local

        return user_path if os.path.exists(user_path) else None

    return io_bind(
        cwd(),
        lambda cwd_value: io_bind(
            user_config_path(environment),
            lambda user_path: IO(lambda: pick(cwd_value, user_path)),
        ),
    )


def load_setup(
    arguments: Arguments, environment: Mapping[str, str]
) -> IO[Result[Setup, str]]:
    def from_paths(user_path: str, selected_path: str | None) -> IO[Result[Setup, str]]:
        def after_user(user_exists: bool) -> IO[Result[Setup, str]]:
            def after_user_document(
                user_document_result: Result[dict[str, Any], str],
            ) -> IO[Result[Setup, str]]:
                def after_selected_document(
                    selected_document_result: Result[dict[str, Any], str],
                ) -> IO[Result[Setup, str]]:
                    def after_passes(
                        passes_result: Result[
                            tuple[dict[str, bool], tuple[PassDefinition, ...]], str
                        ],
                    ) -> IO[Result[Setup, str]]:
                        return io_result(
                            result_bind(
                                merged_api_settings(
                                    arguments,
                                    environment,
                                    user_path,
                                    user_document_result,
                                    selected_path,
                                    selected_document_result,
                                ),
                                lambda config: result_bind(
                                    passes_result,
                                    lambda pair: Ok(
                                        Setup(
                                            config=config,
                                            passes=apply_default_ascii(
                                                pair[1], pair[0]
                                            ),
                                            ensure_paragraphs=pair[0].get(
                                                "ensure_paragraphs", False
                                            ),
                                        )
                                    ),
                                ),
                            )
                        )

                    return io_bind(
                        resolve_passes(
                            user_path,
                            user_document_result,
                            selected_path,
                            selected_document_result,
                        ),
                        after_passes,
                    )

                return io_bind(
                    read_document(selected_path, "config file", bool(selected_path)),
                    after_selected_document,
                )

            return io_bind(
                read_document(user_path, "user config", user_exists),
                after_user_document,
            )

        return io_bind(path_exists(user_path), after_user)

    return io_bind(
        user_config_path(environment),
        lambda user_path: io_bind(
            resolve_config_path(arguments.config, environment),
            lambda selected_path: from_paths(user_path, selected_path),
        ),
    )


@dataclass(frozen=True)
class StatusView:
    label: str = "Working"
    started: float = 0.0
    tokens: int = 0
    prefix: str = ""
    drawn: str = ""


def status_line_text(view: StatusView, now_value: float) -> str:
    line = "%s: time elapsed: %.2fs, tokens received: %d" % (
        view.label,
        now_value - view.started,
        view.tokens,
    )
    return view.prefix + line if view.prefix else line


def status_erase_text(view: StatusView) -> str:
    return "\r" + " " * len(view.drawn) + "\r" if view.drawn else ""


class StatusLine:
    def __init__(self, stream: TextIO):
        self._stream = stream
        try:
            self._live = bool(stream.isatty())
        except (AttributeError, OSError, ValueError):
            self._live = False

        self._view = StatusView()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def write_partial(self, text: str) -> IO[None]:
        def thunk() -> None:
            with self._lock:
                self._view = replace(self._view, prefix=self._view.prefix + text)
                self._stream.write(text)
                self._stream.flush()

        return IO(thunk)

    def start(self, label: str) -> IO[None]:
        def thunk() -> None:
            if not self._live:
                return None

            with self._lock:
                self._view = replace(
                    self._view, label=label, started=time.monotonic(), tokens=0
                )
                self._draw(time.monotonic())

            self._stop.clear()
            self._thread = threading.Thread(target=self._tick, daemon=True)
            self._thread.start()
            return None

        return IO(thunk)

    def progress(self, label: str, count: int = 1) -> IO[None]:
        def thunk() -> None:
            if not self._live:
                return None

            with self._lock:
                self._view = replace(
                    self._view, label=label, tokens=self._view.tokens + count
                )

            return None

        return IO(thunk)

    def stop(self) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self._lock:
                if self._erase() and self._view.prefix:
                    self._stream.write(self._view.prefix)
                    self._view = replace(self._view, drawn=self._view.prefix)

                self._stream.flush()

            return None

        return IO(thunk)

    def interrupt(self) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self._lock:
                had_drawn = self._erase()
                if self._view.prefix:
                    if not (self._live and had_drawn):
                        self._stream.write("\n")

                    self._view = replace(self._view, prefix="")

                self._stream.flush()

            return None

        return IO(thunk)

    def finish(self, text: str) -> IO[None]:
        def thunk() -> None:
            self._halt()
            with self._lock:
                self._stream.write(
                    (self._view.prefix if self._erase() else "") + text + "\n"
                )
                self._view = replace(self._view, prefix="", drawn="")
                self._stream.flush()

            return None

        return IO(thunk)

    def _halt(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _tick(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                self._draw(time.monotonic())

    def _draw(self, now_value: float) -> None:
        full = status_line_text(self._view, now_value)
        self._stream.write(
            "\r" + full + " " * max(len(self._view.drawn) - len(full), 0)
        )
        self._stream.flush()
        self._view = replace(self._view, drawn=full)

    def _erase(self) -> bool:
        if not self._view.drawn:
            return False

        self._stream.write(status_erase_text(self._view))
        self._view = replace(self._view, drawn="")
        return True


@dataclass(frozen=True)
class Console:
    stream: TextIO
    status: StatusLine

    def log(self, message: str) -> IO[None]:
        def thunk() -> None:
            self.status.interrupt().run()
            print(message, file=self.stream, flush=True)

        return IO(thunk)

    def write_partial(self, text: str) -> IO[None]:
        return self.status.write_partial(text)

    def start(self, label: str) -> IO[None]:
        return self.status.start(label)

    def progress(self, label: str, count: int = 1) -> IO[None]:
        return self.status.progress(label, count)

    def stop(self) -> IO[None]:
        return self.status.stop()

    def interrupt(self) -> IO[None]:
        return self.status.interrupt()

    def finish(self, text: str) -> IO[None]:
        return self.status.finish(text)


def parse_args(arguments: Sequence[str]) -> Arguments:
    parser = argparse.ArgumentParser(
        prog="zh2en",
        description="Translate Chinese text from stdin to English on stdout.",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        nargs="?",
        help="TOML config file defining [api] settings and [[pass]] passes "
        "(default: $TRANSLATE_CONFIG, then ./zh2en.toml, then "
        "~/.config/zh2en/config.toml)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="zh2en " + __version__,
    )
    parser.add_argument(
        "--base-url",
        help="API endpoint; overrides [api] base_url and TRANSLATE_BASE_URL",
    )
    parser.add_argument(
        "--api-key",
        help="API key; overrides [api] api_key and TRANSLATE_API_KEY",
    )
    parser.add_argument(
        "--model",
        help="default model; overrides [api] model and TRANSLATE_MODEL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="request timeout in seconds; overrides [api] timeout and "
        "TRANSLATE_TIMEOUT",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="request token budget; overrides [api] max_tokens and "
        "TRANSLATE_MAX_TOKENS",
    )
    parser.add_argument(
        "--cache-dir",
        help="translation cache directory (default: $XDG_CACHE_HOME/zh2en)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    parser.add_argument(
        "--ensure-paragraphs",
        action="store_true",
        help="after each pass, check the output paragraph count against the "
        "source; re-run a mismatching pass with one call per paragraph",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) and LLM reasoning "
        "traces to stderr",
    )
    parser.add_argument(
        "--show-log-path",
        "-l",
        action="store_true",
        help="print the run log's path to stderr at startup",
    )
    parsed = parser.parse_args(arguments)
    return Arguments(
        config=parsed.config,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        model=parsed.model,
        timeout=parsed.timeout,
        max_tokens=parsed.max_tokens,
        no_cache=parsed.no_cache,
        ensure_paragraphs=parsed.ensure_paragraphs,
        verbose=parsed.verbose,
        show_log_path=parsed.show_log_path,
        cache_dir=parsed.cache_dir,
    )


def run_pipeline(
    ctx: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
    started: float,
    stdout: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    source_paragraphs, separators = split_paragraphs(text)

    def conclude(result: Result[State, str]) -> IO[int]:
        if isinstance(result, Err):
            return io_map(ctx.console.log(result.error), lambda _: 1)

        return finish_output(ctx, result.value, started, stdout, clock)

    return io_bind(
        run_passes(
            ctx,
            pass_definitions,
            State(text=text, analysis=None, usage=Usage()),
            source_paragraphs,
            separators,
            make_chunks(source_paragraphs, ctx.settings.chunk_budget_tokens),
            tuple((paragraph,) for paragraph in source_paragraphs),
            clock,
        ),
        conclude,
    )


def finish_output(
    ctx: Context,
    state: State,
    started: float,
    stdout: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    def after_write(_: None) -> IO[int]:
        total_elapsed = clock() - started

        def after_done(_: None) -> IO[int]:
            return io_map(
                ctx.console.log(
                    usage_line(
                        "TOTAL",
                        total_elapsed,
                        state.usage.prompt_tokens,
                        state.usage.completion_tokens,
                        state.usage.cost,
                    )
                ),
                lambda _: 0,
            )

        return io_bind(
            (
                ctx.console.log("zh2en: done in %.1fs" % total_elapsed)
                if ctx.verbose
                else io_pure(None)
            ),
            after_done,
        )

    return io_bind(write_stdout(stdout, state.text), after_write)


def main(
    arguments: Sequence[str],
    environment: Mapping[str, str],
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    parsed = parse_args(arguments)
    return io_bind(
        read_stdin(stdin),
        lambda text: run_main_program(parsed, environment, text, stdout, stderr, clock),
    )


def run_main_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    text: str,
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    if not text.strip():
        return io_pure(0)

    console = Console(stderr, StatusLine(stderr))
    started = clock()

    def use_setup(setup_result: Result[Setup, str]) -> IO[int]:
        if isinstance(setup_result, Err):
            return io_map(console.log("zh2en: %s" % setup_result.error), lambda _: 2)

        setup = setup_result.value

        def with_cache_dir(cache_directory: str) -> IO[int]:
            def with_log(log: RunLog) -> IO[int]:
                return io_bind(
                    (
                        console.log("zh2en: log: %s" % log.path)
                        if parsed.show_log_path
                        else io_pure(None)
                    ),
                    lambda _: run_pipeline(
                        Context(
                            config=setup.config,
                            settings=build_settings(),
                            use_cache=not parsed.no_cache,
                            cache_directory=cache_directory,
                            verbose=parsed.verbose,
                            ensure_paragraphs=parsed.ensure_paragraphs
                            or setup.ensure_paragraphs,
                            console=console,
                            open_http=urllib_open,
                            log=log,
                        ),
                        setup.passes,
                        text,
                        started,
                        stdout,
                        clock,
                    ),
                )

            return io_bind(open_run_log(cache_directory), with_log)

        return io_bind(resolve_cache_dir(environment, parsed.cache_dir), with_cache_dir)

    return io_bind(load_setup(parsed, environment), use_setup)


def cli() -> None:
    try:
        code = main(
            sys.argv[1:],
            dict(os.environ),
            sys.stdin,
            sys.stdout,
            sys.stderr,
            time.time,
        ).run()
    except KeyboardInterrupt:
        print("zh2en: interrupted", file=sys.stderr)
        code = 130

    sys.exit(code)


if __name__ == "__main__":
    cli()

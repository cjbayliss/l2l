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
from dataclasses import dataclass, replace
from functools import reduce
from itertools import accumulate, chain
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    Optional,
    TextIO,
    Tuple,
    TypeVar,
    Union,
)

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


Result = Union[Ok[T], Err[E]]


def is_ok(result: Result) -> bool:
    return isinstance(result, Ok)


def result_map(result: Result, fn: Callable[[Any], Any]) -> Result:
    return Ok(fn(result.value)) if isinstance(result, Ok) else result


def result_bind(result: Result, fn: Callable[[Any], Result]) -> Result:
    return fn(result.value) if isinstance(result, Ok) else result


def result_bind_io(result: Result, fn: Callable[[Any], "IO"]) -> "IO":
    return io_pure(result) if isinstance(result, Err) else fn(result.value)


def results_sequence(results: Iterable[Result]) -> Result:
    def step(accumulator: Result, result: Result) -> Result:
        return result_bind(
            accumulator,
            lambda values: result_map(result, lambda value: values + (value,)),
        )

    return reduce(step, tuple(results), Ok(()))


@dataclass(frozen=True)
class IO(Generic[T]):
    run: Callable[[], T]


def io_pure(value: Any) -> IO:
    return IO(lambda: value)


def io_map(io_value: IO, fn: Callable[[Any], Any]) -> IO:
    return IO(lambda: fn(io_value.run()))


def io_bind(io_value: IO, fn: Callable[[Any], IO]) -> IO:
    return IO(lambda: fn(io_value.run()).run())


def io_sequence(io_values: Iterable[IO]) -> IO:
    return IO(lambda: tuple(io_value.run() for io_value in io_values))


def fold_io(
    items: Iterable[Any], step: Callable[[Any, Any], IO], initial: Result
) -> IO:
    def chain_step(accumulator_io: IO, item: Any) -> IO:
        return io_bind(
            accumulator_io,
            lambda accumulator_result: result_bind_io(
                accumulator_result, lambda accumulator: step(accumulator, item)
            ),
        )

    return reduce(chain_step, tuple(items), io_pure(initial))


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    params: Mapping


@dataclass(frozen=True)
class PassDefinition:
    name: str
    instruction: str
    mode: str
    strict_fidelity: bool
    params: Mapping
    model: Optional[str]
    ascii: Optional[bool]


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
    strict_fidelity_suffix: str
    analysis_merge_instruction: str
    analysis_user_prefix: str
    merge_user_prefix: str
    ascii_fix_instruction: str
    ascii_character_map: Mapping


@dataclass(frozen=True)
class State:
    text: str
    analysis: Optional[str]
    usage: Usage


@dataclass(frozen=True)
class Arguments:
    config: Optional[str]
    base_url: Optional[str]
    api_key: Optional[str]
    model: Optional[str]
    timeout: Optional[float]
    max_tokens: Optional[int]
    no_cache: bool
    verbose: bool


@dataclass(frozen=True)
class Setup:
    config: Config
    passes: Tuple[PassDefinition, ...]


@dataclass(frozen=True)
class Context:
    config: Config
    settings: Settings
    use_cache: bool
    cache_directory: str
    verbose: bool
    console: "Console"


def build_settings() -> Settings:
    return Settings(
        chunk_budget_tokens=3500,
        analysis_reserve_tokens=128,
        ascii_fix_attempts=3,
        sentence_boundary_characters="。！？!?；;\n",
        strict_fidelity_suffix=(
            "\n- STRICT FIDELITY MODE: keep the exact number of paragraphs "
            "and the exact separators of the source. Do not merge "
            "or split paragraphs."
        ),
        analysis_merge_instruction="""
You are merging partial preparation briefs from a Chinese-to-English translation
pipeline. The source document was too long to read in one pass, so it was
analysed in parts. Merge the parts into one compact brief with exactly these
headings, in this order: OUTLINE, NAMES, HARD TO TRANSLATE.

- OUTLINE: a single numbered list of the major beats in source order.
- NAMES: one line per distinct term, formatted exactly as 原文 -> rendering (type),
  where type is one of: person, place, organization, title, term. On conflicts,
  prefer the first occurrence.
- HARD TO TRANSLATE: deduplicated traps with the recommended English handling.

Output only the merged brief. No preamble, no commentary. Keep it compact
(about 600 words at most).
""",
        analysis_user_prefix="Source text (full document, original Chinese):\n",
        merge_user_prefix=(
            "Partial preparation briefs (parts of one document, in source order):\n\n"
        ),
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


def split_paragraphs(text: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    blank_pattern = re.compile(r"\n\s*\n")
    blanks = blank_pattern.findall(text)
    lone_newlines = text.count("\n") - sum(blank.count("\n") for blank in blanks)
    separator_pattern = r"(\n+)" if lone_newlines >= 3 * len(blanks) else r"(\n\s*\n)"
    parts = re.split(separator_pattern, text)
    is_separator = re.compile(separator_pattern).fullmatch
    paragraphs = tuple(part for part in parts if part and not is_separator(part))
    separators = tuple(part for part in parts if part and is_separator(part))
    return paragraphs, separators


def ensure_blank_line_separators(text: str) -> str:
    return "\n\n".join(split_paragraphs(text)[0])


def make_chunks(
    paragraphs: Tuple[str, ...], budget: int
) -> Tuple[Tuple[str, ...], ...]:
    def fold(accumulator, paragraph):
        chunks, current, size = accumulator
        paragraph_size = estimate_tokens(paragraph)
        if current and size + paragraph_size > budget:
            return chunks + (current,), (paragraph,), paragraph_size

        return chunks, current + (paragraph,), size + paragraph_size

    chunks, current, _ = reduce(fold, paragraphs, ((), (), 0))
    return chunks + (current,) if current else chunks


def split_sentences(text: str, boundary_characters: str) -> Tuple[str, ...]:
    pieces = re.split("(?<=[%s])" % re.escape(boundary_characters), text)
    if pieces and pieces[-1] == "":
        pieces = pieces[:-1]

    return tuple(pieces)


def split_to_budget(
    text: str, budget: int, boundary_characters: str
) -> Tuple[str, ...]:
    def fold(accumulator, sentence):
        pieces, buffer, size = accumulator
        sentence_size = estimate_tokens(sentence)
        if buffer and size + sentence_size > budget:
            return pieces + ("".join(buffer),), (sentence,), sentence_size

        return pieces, buffer + (sentence,), size + sentence_size

    pieces, buffer, _ = reduce(
        fold, split_sentences(text, boundary_characters), ((), (), 0)
    )
    return pieces + ("".join(buffer),) if buffer else pieces


def split_units_to_budget(
    paragraphs: Tuple[str, ...], budget: int, boundary_characters: str
) -> Tuple[str, ...]:
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
    plan: Tuple[Tuple[str, ...], ...], separators: Tuple[str, ...]
) -> Tuple[str, ...]:
    ends = accumulate(map(len, plan))

    def trailing(index: int, end: int) -> str:
        if index >= len(plan) - 1:
            return ""

        position = end - 1
        return separators[position] if position < len(separators) else ""

    return tuple(trailing(index, end) for index, end in enumerate(ends))


def regroup_by_plan(
    paragraphs: Tuple[str, ...], plan: Tuple[Tuple[str, ...], ...]
) -> Optional[Tuple[Tuple[str, ...], ...]]:
    total = sum(map(len, plan))
    if len(paragraphs) != total:
        return None

    starts = accumulate(map(len, plan), initial=0)
    return tuple(
        tuple(paragraphs[start : start + len(chunk)])
        for start, chunk in zip(starts, plan)
    )


def analysis_block(analysis: Optional[str]) -> str:
    stripped = analysis.strip() if analysis else ""
    return stripped if stripped else "(none)"


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return "%.1fs" % seconds

    return "%dm%ds" % (int(seconds // 60), int(seconds % 60))


def usage_line(label, elapsed, prompt_tokens, completion_tokens, cost) -> str:
    rate = completion_tokens / elapsed if elapsed else 0.0
    return "%s: %s, prompt=%d, completion=%d, %.1f tok/s, cost=$%.6f" % (
        label,
        fmt_duration(elapsed),
        prompt_tokens,
        completion_tokens,
        rate,
        cost,
    )


def add_usage(usage: Usage, reported: Mapping) -> Usage:
    prompt = reported.get("prompt_tokens") or 0
    completion = reported.get("completion_tokens") or 0
    try:
        cost = float(reported.get("cost") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    return Usage(
        prompt_tokens=usage.prompt_tokens + prompt,
        completion_tokens=usage.completion_tokens + completion,
        cost=usage.cost + cost,
    )


def usage_delta(start_usage: Usage, end_usage: Usage) -> Tuple[int, int, float]:
    return (
        end_usage.prompt_tokens - start_usage.prompt_tokens,
        end_usage.completion_tokens - start_usage.completion_tokens,
        end_usage.cost - start_usage.cost,
    )


def non_ascii_sample(text: str, limit: int = 12) -> str:
    distinct = dict.fromkeys(char for char in text if not char.isascii())
    return "".join(tuple(distinct)[:limit])


def to_ascii_mechanical(text: str, character_map: Mapping) -> str:
    replaced = reduce(
        lambda text, pair: text.replace(pair[0], pair[1]),
        character_map.items(),
        text,
    )
    normalized = unicodedata.normalize("NFKD", replaced)
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


def strip_think_tag(content: Any) -> Tuple[Any, Optional[str]]:
    if not isinstance(content, str):
        return content, None

    match = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    if not match:
        return content, None

    return content[match.end() :], match.group(1).strip()


@dataclass(frozen=True)
class ThinkState:
    checking: bool = True
    open: bool = False
    tail: str = ""


def think_step(state: ThinkState, text: str) -> Tuple[ThinkState, bool]:
    if not state.checking:
        return state, False

    searched = state.tail + text
    tail = searched[-8:]
    if state.open:
        closed = "</think>" in searched
        return ThinkState(checking=not closed, open=not closed, tail=tail), not closed

    stripped = searched.lstrip()
    if stripped.startswith("<think>"):
        closed = "</think>" in stripped[7:]
        return ThinkState(checking=not closed, open=not closed, tail=tail), not closed

    done = not stripped.startswith("<") or len(stripped) > 64
    return ThinkState(checking=not done, open=False, tail=tail), False


def cache_path(cache_directory: str, key: str) -> str:
    return os.path.join(cache_directory, key + ".txt")


def cache_key(chunk_text, model, pass_salt="", work_text="", overrides=None) -> str:
    hasher = hashlib.sha256()
    if pass_salt:
        hasher.update(pass_salt.encode("utf-8") + b"\x00")

    hasher.update(chunk_text.encode("utf-8"))
    if work_text:
        hasher.update(b"\x00work\x00" + work_text.encode("utf-8"))

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
    "strict_fidelity",
    "ascii",
    "model",
    "params",
)


def default_api_settings() -> dict:
    return {
        "base_url": "",
        "api_key": "",
        "model": "",
        "timeout": 120.0,
        "max_tokens": 100000,
        "params": {},
    }


def validate_document(path: str, document: dict) -> Result:
    if not document:
        return Ok(None)

    unknown = sorted(set(document) - {"api", "options", "pass"})
    if unknown:
        return Err(
            "%s: unknown top-level key(s): %s (expected [api], [options], [[pass]])"
            % (path, ", ".join(unknown))
        )

    return Ok(None)


def string_api_settings(path: str, table: dict) -> Result:
    def add(settings: dict, key: str) -> Result:
        if key not in table:
            return Ok(settings)

        value = table[key]
        if not isinstance(value, str) or not value.strip():
            return Err("%s: [api] %s must be a non-empty string" % (path, key))

        return Ok({**settings, key: value.strip()})

    def step(settings_result: Result, key: str) -> Result:
        return result_bind(settings_result, lambda settings: add(settings, key))

    return reduce(step, ("base_url", "api_key", "model"), Ok({}))


def timeout_api_setting(path: str, settings: dict, table: dict) -> Result:
    if "timeout" not in table:
        return Ok(settings)

    value = table["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return Err("%s: [api] timeout must be a positive number" % path)

    return Ok({**settings, "timeout": float(value)})


def max_tokens_api_setting(path: str, settings: dict, table: dict) -> Result:
    if "max_tokens" not in table:
        return Ok(settings)

    value = table["max_tokens"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return Err("%s: [api] max_tokens must be a positive integer" % path)

    return Ok({**settings, "max_tokens": value})


def params_api_setting(path: str, settings: dict, table: dict) -> Result:
    if "params" not in table:
        return Ok(settings)

    value = table["params"]
    if not isinstance(value, dict):
        return Err("%s: [api] params must be a table" % path)

    return Ok({**settings, "params": value})


def document_api_settings(path: str, document: dict) -> Result:
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


def merge_api_settings(base: dict, extra: Mapping) -> dict:
    def merge_one(settings: dict, key: str, value: Any) -> dict:
        if key == "params" and isinstance(settings.get("params"), dict):
            return {**settings, "params": {**settings["params"], **value}}

        return {**settings, key: value}

    return reduce(
        lambda settings, entry: merge_one(settings, entry[0], entry[1]),
        extra.items(),
        dict(base),
    )


def parse_number_setting(
    environment, settings, variable, key, convert, invalid_label
) -> Result:
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


def api_settings_from_environment(environment: Mapping) -> Result:
    def string_step(settings_result: Result, binding) -> Result:
        variable, key = binding
        return result_bind(
            settings_result,
            lambda settings: (
                lambda value: Ok({**settings, key: value} if value else settings)
            )(environment.get(variable, "").strip()),
        )

    strings = reduce(
        string_step,
        (
            ("TRANSLATE_BASE_URL", "base_url"),
            ("TRANSLATE_API_KEY", "api_key"),
            ("TRANSLATE_MODEL", "model"),
        ),
        Ok({}),
    )

    return result_bind(
        strings,
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


def api_settings_from_arguments(arguments: Arguments) -> dict:
    bindings = (
        (arguments.base_url, "base_url"),
        (arguments.api_key, "api_key"),
        (arguments.model, "model"),
        (arguments.timeout, "timeout"),
        (arguments.max_tokens, "max_tokens"),
    )

    def step(settings: dict, binding) -> dict:
        value, key = binding
        return {**settings, key: value} if value is not None else settings

    return reduce(step, bindings, {})


def missing_api_settings(config: Config) -> Tuple[str, ...]:
    return tuple(
        description
        for value, description in (
            (config.base_url, "api.base_url (--base-url / TRANSLATE_BASE_URL)"),
            (config.api_key, "api.api_key (--api-key / TRANSLATE_API_KEY)"),
            (config.model, "api.model (--model / TRANSLATE_MODEL)"),
        )
        if not value
    )


def build_config(settings: dict) -> Result:
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
    environment: Mapping,
    user_path: str,
    user_document_result: Result,
    selected_path: Optional[str],
    selected_document_result: Result,
) -> Result:
    def merge_document(path: str, document_result: Result, settings: dict) -> Result:
        return result_bind(
            document_result,
            lambda document: result_map(
                document_api_settings(path, document),
                lambda extra: merge_api_settings(settings, extra),
            ),
        )

    pipeline = Ok(default_api_settings())
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


def parse_options_table(path: str, table: Any) -> Result:
    if not isinstance(table, dict):
        return Err("%s: [options] must be a table" % path)

    unknown = sorted(set(table) - {"ascii"})
    if unknown:
        return Err("%s: [options]: unknown key(s): %s" % (path, ", ".join(unknown)))

    ascii_value = table.get("ascii", False)
    if not isinstance(ascii_value, bool):
        return Err("%s: [options] ascii must be true or false" % path)

    return Ok({"ascii": ascii_value})


def pass_definition_from(path: str, name: str, table: dict, instruction: str) -> Result:
    mode = table.get("mode", "chunk")
    if mode not in ("analysis", "chunk", "paragraph"):
        return Err(
            "%s: [[pass]] %s: mode must be analysis, chunk, or paragraph" % (path, name)
        )

    strict_fidelity = table.get("strict_fidelity", False)
    if not isinstance(strict_fidelity, bool):
        return Err(
            "%s: [[pass]] %s: strict_fidelity must be true or false" % (path, name)
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
            strict_fidelity=strict_fidelity,
            params=MappingProxyType(dict(params)),
            model=model.strip() if model else None,
            ascii=ascii_value,
        )
    )


def apply_default_ascii(
    pass_definitions, options: Mapping
) -> Tuple[PassDefinition, ...]:
    default_ascii = options.get("ascii", False)
    return tuple(
        (
            replace(pass_definition, ascii=default_ascii)
            if pass_definition.ascii is None
            else pass_definition
        )
        for pass_definition in pass_definitions
    )


def resolve_call_settings(
    config: Config, pass_definition: PassDefinition
) -> Tuple[str, dict]:
    params = {**dict(config.params), **dict(pass_definition.params)}
    return (pass_definition.model or config.model), params


def merge_budget(config: Config, settings: Settings) -> int:
    overhead = estimate_tokens(settings.analysis_merge_instruction) + estimate_tokens(
        settings.merge_user_prefix
    )
    return max(config.max_tokens - overhead - settings.analysis_reserve_tokens, 1)


def build_chat_payload(model: str, system: str, user: str, params: Mapping) -> dict:
    core = {
        "model": model,
        "messages": (
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ),
    }
    return {
        **core,
        **{key: value for key, value in params.items() if value is not None},
    }


def build_pass_user(
    source_chunk: str, work_chunk: Optional[str], analysis: Optional[str]
) -> str:
    base = (
        "Preparation brief from a full read of the text "
        "(outline, names, hard-to-translate items):\n%s\n\n"
        "Source text (original Chinese):\n%s" % (analysis_block(analysis), source_chunk)
    )
    draft = (
        ("\n\nCurrent draft from the previous pass:\n%s" % work_chunk,)
        if work_chunk is not None and work_chunk != source_chunk
        else ()
    )
    return "".join((base,) + draft)


def plan_info_message(
    pass_definition: PassDefinition, work_paragraphs, work_groups
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


def resolve_work_groups(mode, work_paragraphs, plan, pass_name, budget):
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


def extract_message(body: Any) -> Result:
    try:
        message = body["choices"][0]["message"]
        return Ok((message, message["content"]))
    except (KeyError, IndexError, TypeError):
        return Err("unexpected response shape: %s" % str(body)[:500])


def collect_thinking_texts(part: Mapping) -> Tuple[str, ...]:
    return tuple(
        entry["text"]
        for entry in (part.get("thinking") or [])
        if isinstance(entry, dict) and entry.get("text")
    )


def collect_content_parts(parts) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    def step(accumulator, part):
        texts, thoughts = accumulator
        if not isinstance(part, dict):
            return texts + (str(part),), thoughts

        if part.get("type") == "thinking":
            return texts, thoughts + collect_thinking_texts(part)

        if part.get("text"):
            return texts + (part["text"],), thoughts

        return accumulator

    return reduce(step, parts, ((), ()))


def flatten_content_parts(content: Any) -> Tuple[Any, Tuple[str, ...]]:
    if not isinstance(content, list):
        return content, ()

    texts, thoughts = collect_content_parts(content)
    return "".join(texts), thoughts


def message_reasoning_texts(message: Mapping) -> Tuple[str, ...]:
    def reasoning_at(key: str) -> Tuple[str, ...]:
        value = message.get(key)
        return (value.rstrip(),) if isinstance(value, str) and value.strip() else ()

    return reduce(
        lambda found, key: found or reasoning_at(key),
        ("reasoning_content", "reasoning"),
        (),
    )


def parse_stream_line(raw_line: bytes) -> Result:
    line = raw_line.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return Ok(None)

    data = line[5:].strip()
    if not data or data == "[DONE]":
        return Ok(None)

    try:
        return Ok(json.loads(data))
    except json.JSONDecodeError:
        return Ok(None)


def parse_chunk_delta(chunk: Mapping) -> dict:
    try:
        choices = chunk.get("choices")
        if not choices:
            return {}

        return choices[0].get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return {}


def delta_reasoning_text(delta: Mapping) -> Optional[str]:
    def reasoning_at(key: str) -> Optional[str]:
        value = delta.get(key)
        return value if isinstance(value, str) and value else None

    return reduce(
        lambda found, key: found if found is not None else reasoning_at(key),
        ("reasoning_content", "reasoning"),
        None,
    )


def delta_text(delta: Mapping) -> str:
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))

    return ""


@dataclass(frozen=True)
class StreamState:
    reasoning: Tuple[str, ...] = ()
    contents: Tuple[str, ...] = ()
    reported: Optional[dict] = None
    counted: int = 0
    content_started: bool = False
    think: ThinkState = ThinkState()
    progress_request: Optional[Tuple[str, int]] = None


def stream_step(state: StreamState, chunk: Mapping) -> Result:
    reported = (
        chunk["usage"] if isinstance(chunk.get("usage"), dict) else state.reported
    )
    delta = parse_chunk_delta(chunk)
    reasoning_text = delta_reasoning_text(delta)
    text = delta_text(delta)
    if not reasoning_text and not text:
        return Ok(replace(state, reported=reported, progress_request=None))

    think_state, thinking = (
        think_step(state.think, text) if text else (state.think, False)
    )
    thinking_progress = 1 if reasoning_text and not state.content_started else 0
    text_progress = 1 if text else 0
    progress_request = None
    if thinking_progress or text_progress:
        label = ("Thinking" if thinking else "Working") if text else "Thinking"
        progress_request = (label, thinking_progress + text_progress)

    return Ok(
        StreamState(
            reasoning=(
                state.reasoning + (reasoning_text,)
                if reasoning_text
                else state.reasoning
            ),
            contents=state.contents + (text,) if text else state.contents,
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
    reasoning: Tuple[str, ...] = ()
    reported: Optional[dict] = None
    counted: int = 0


def plain_reply(body: Any) -> Result:
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


def drop_non_ascii(
    text: str, character_map: Mapping, attempts: int, index: int
) -> Tuple[str, Optional[str]]:
    if text.isascii():
        return text, None

    fallback = to_ascii_mechanical(text, character_map)
    if fallback.isascii():
        return fallback, None

    stripped = re.sub(
        r"  +", " ", "".join(character for character in text if character.isascii())
    )
    warning = (
        "zh2en: ascii: warning: paragraph %d still contained "
        "non-ASCII characters (%s) after %d LLM attempts; dropping "
        "them" % (index + 1, non_ascii_sample(text), attempts)
    )
    return stripped, warning


def now(clock: Callable[[], float]) -> IO:
    return IO(clock)


def read_stdin(stream: TextIO) -> IO:
    return IO(stream.read)


def write_stdout(stream: TextIO, text: str) -> IO:
    def thunk():
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")

    return IO(thunk)


def path_exists(path: str) -> IO:
    return IO(lambda: os.path.exists(path))


def cwd() -> IO:
    return IO(os.getcwd)


def user_config_path(environment: Mapping) -> IO:
    return IO(
        lambda: os.path.join(
            environment.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
            "zh2en",
            "config.toml",
        )
    )


def resolve_cache_dir(environment: Mapping) -> IO:
    def thunk():
        base = environment.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        directory = os.path.join(base, "zh2en")
        os.makedirs(directory, exist_ok=True)
        return directory

    return IO(thunk)


def load_toml(path: str, description: str) -> IO:
    def thunk():
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


def cache_read(cache_directory: str, key: str) -> IO:
    def thunk():
        try:
            with open(cache_path(cache_directory, key), encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return None

    return IO(thunk)


def cache_write(cache_directory: str, key: str, value: str) -> IO:
    def thunk():
        path = cache_path(cache_directory, key)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            handle.write(value)

        os.replace(temporary_path, path)

    return IO(thunk)


def http_request(config: Config, payload: dict, accept: Optional[str] = None):
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


def open_http(request, timeout: float) -> Result:
    try:
        return Ok(urllib.request.urlopen(request, timeout=timeout))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        return Err("HTTP %s from endpoint: %s" % (error.code, detail))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return Err("could not reach endpoint: %s" % error)


def http_post_json(config: Config, payload: dict) -> IO:
    def thunk():
        opened = open_http(http_request(config, payload), config.timeout)
        if isinstance(opened, Err):
            return opened

        with opened.value as response:
            try:
                return Ok(json.loads(response.read().decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                return Err("invalid JSON response: %s" % error)

    return IO(thunk)


def http_open_stream(config: Config, payload: dict) -> IO:
    def thunk():
        opened = open_http(
            http_request(config, payload, accept="text/event-stream"), config.timeout
        )
        # Endpoint rejected stream_options; retry without it.
        if (
            is_ok(opened)
            or "stream_options" not in payload
            or "stream_options" not in str(opened.error)
        ):
            return opened

        retried = {
            key: value for key, value in payload.items() if key != "stream_options"
        }
        return open_http(
            http_request(config, retried, accept="text/event-stream"), config.timeout
        )

    return IO(thunk)


def record_progress(
    on_progress: Callable[[str, int], None],
) -> Callable[[StreamState], StreamState]:
    def record(state: StreamState) -> StreamState:
        if state.progress_request is not None:
            label, count = state.progress_request
            on_progress(label, count)

        return state

    return record


def step_stream(state_result: Result, raw_line: bytes, on_progress) -> Result:
    line_result = parse_stream_line(raw_line)
    if isinstance(line_result, Err):
        return line_result

    chunk = line_result.value
    if chunk is None:
        return state_result

    if isinstance(chunk.get("error"), dict):
        return Err("endpoint stream error: %s" % str(chunk["error"])[:500])

    return result_map(
        stream_step(state_result.value, chunk), record_progress(on_progress)
    )


def drive_stream(response, on_progress) -> Result:
    state = Ok(StreamState())
    for raw_line in response:
        if isinstance(state, Err):
            return state

        state = step_stream(state, raw_line, on_progress)

    return state


def collect_stream(config: Config, payload: dict, on_progress) -> IO:
    def thunk():
        opened = http_open_stream(config, payload).run()
        if isinstance(opened, Err):
            return opened

        try:
            with opened.value as response:
                return drive_stream(response, on_progress)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return Err("stream interrupted: %s" % error)

    return IO(thunk)


def verbose_log(ctx: Context, message: str) -> IO:
    return ctx.console.log(message) if ctx.verbose else io_pure(None)


def log_all(console: "Console", messages: Tuple[str, ...]) -> IO:
    def step(_, message):
        return io_map(console.log(message), lambda _: Ok(()))

    return fold_io(messages, step, Ok(()))


def chat(
    ctx: Context, system: str, user: str, model: str, params: Mapping, usage: Usage
) -> IO:
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > ctx.config.max_tokens:
        return io_pure(
            Err(
                "request is ~%d tokens, over the %d-token budget; raise "
                "api.max_tokens (--max-tokens / TRANSLATE_MAX_TOKENS) or "
                "shorten the input" % (estimated, ctx.config.max_tokens)
            )
        )

    payload = build_chat_payload(model, system, user, params)
    call = (
        streamed_call(ctx, payload)
        if payload.get("stream", True)
        else plain_call(ctx, payload)
    )

    def run_and_stop():
        reply_result = call.run()
        ctx.console.stop().run()
        return reply_result

    program = io_bind(ctx.console.start("Working"), lambda _: IO(run_and_stop))
    return io_bind(
        program, lambda reply_result: conclude_chat(ctx, reply_result, usage, estimated)
    )


def streamed_call(ctx: Context, payload: dict) -> IO:
    full_payload = {**payload, "stream": True}
    if "stream_options" not in full_payload:
        full_payload = {**full_payload, "stream_options": {"include_usage": True}}

    def on_progress(label: str, count: int) -> None:
        ctx.console.progress(label, count).run()

    def thunk():
        return result_map(
            collect_stream(ctx.config, full_payload, on_progress).run(),
            lambda state: ChatReply(
                content="".join(state.contents),
                reasoning=state.reasoning,
                reported=state.reported,
                counted=state.counted,
            ),
        )

    return IO(thunk)


def plain_call(ctx: Context, payload: dict) -> IO:
    return IO(
        lambda: result_bind(http_post_json(ctx.config, payload).run(), plain_reply)
    )


def conclude_chat(
    ctx: Context, reply_result: Result, usage: Usage, estimated: int
) -> IO:
    if isinstance(reply_result, Err):
        return io_pure(reply_result)

    reply = reply_result.value
    content, think_text = strip_think_tag(reply.content)
    reasoning = reply.reasoning + ((think_text,) if think_text else ())
    messages = (
        tuple(text.rstrip() for text in reasoning if text.strip())
        if ctx.verbose
        else ()
    )

    def finish(_):
        reported = reply.reported
        new_usage = (
            add_usage(usage, reported)
            if reported
            else add_usage(
                usage, {"prompt_tokens": estimated, "completion_tokens": reply.counted}
            )
        )
        if not isinstance(content, str):
            return Err("unexpected content type: %s" % type(content).__name__)

        return Ok((content, new_usage))

    if messages:
        return io_map(log_all(ctx.console, messages), finish)

    return io_pure(finish(None))


def run_analysis(
    ctx: Context, pass_definition: PassDefinition, text: str, usage: Usage
) -> IO:
    user = ctx.settings.analysis_user_prefix + text
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return io_map(
        chat(ctx, pass_definition.instruction, user, model, params, usage),
        lambda result: result_map(result, lambda pair: (pair[0].strip(), pair[1])),
    )


def run_merge(
    ctx: Context, pass_definition: PassDefinition, briefs, usage: Usage
) -> IO:
    user = ctx.settings.merge_user_prefix + "\n\n".join(briefs)
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return io_map(
        chat(ctx, ctx.settings.analysis_merge_instruction, user, model, params, usage),
        lambda result: result_map(result, lambda pair: (pair[0].strip(), pair[1])),
    )


def merge_group(
    ctx: Context, pass_definition: PassDefinition, group, usage: Usage
) -> IO:
    if len(group) == 1:
        return io_pure(Ok((group[0], usage)))

    return run_merge(ctx, pass_definition, group, usage)


def merge_groups(
    ctx: Context, pass_definition: PassDefinition, groups, usage: Usage
) -> IO:
    def step(accumulator, group):
        merged, current_usage = accumulator
        return io_map(
            merge_group(ctx, pass_definition, group, current_usage),
            lambda result: result_map(
                result, lambda pair: (merged + (pair[0],), pair[1])
            ),
        )

    return fold_io(groups, step, Ok(((), usage)))


def merge_analysis(
    ctx: Context, pass_definition: PassDefinition, briefs, usage: Usage
) -> IO:
    if len(briefs) <= 1:
        return io_pure(Ok((briefs[0], usage)))

    budget = merge_budget(ctx.config, ctx.settings)
    groups = make_chunks(briefs, budget)
    if len(groups) == len(briefs):
        return io_pure(
            Err(
                "partial analysis brief (~%d tokens) does not fit the "
                "%d-token budget; raise api.max_tokens (--max-tokens / "
                "TRANSLATE_MAX_TOKENS) or shorten the input"
                % (
                    max(estimate_tokens(brief) for brief in briefs),
                    ctx.config.max_tokens,
                )
            )
        )

    return io_bind(
        merge_groups(ctx, pass_definition, groups, usage),
        lambda result: result_bind_io(
            result, lambda pair: merge_analysis(ctx, pass_definition, pair[0], pair[1])
        ),
    )


def analyze_document(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO:
    overhead = estimate_tokens(pass_definition.instruction) + estimate_tokens(
        ctx.settings.analysis_user_prefix
    )
    budget = max(
        ctx.config.max_tokens - overhead - ctx.settings.analysis_reserve_tokens, 1
    )
    paragraphs, _ = split_paragraphs(full_text)
    units = split_units_to_budget(
        paragraphs, budget, ctx.settings.sentence_boundary_characters
    )
    chunks = make_chunks(units, budget)
    if len(chunks) <= 1:
        return run_analysis(ctx, pass_definition, full_text, usage)

    intro_log = verbose_log(
        ctx,
        "zh2en: [%s] ~%d tokens over the %d-token budget; analysing in %d part(s)"
        % (
            pass_definition.name,
            estimate_tokens(full_text),
            ctx.config.max_tokens,
            len(chunks),
        ),
    )

    def step(accumulator, chunk):
        briefs, current_usage = accumulator
        return io_map(
            run_analysis(ctx, pass_definition, "\n\n".join(chunk), current_usage),
            lambda result: result_map(
                result, lambda pair: (briefs + (pair[0],), pair[1])
            ),
        )

    return io_bind(
        intro_log,
        lambda _: io_bind(
            fold_io(chunks, step, Ok(((), usage))),
            lambda result: result_bind_io(
                result,
                lambda pair: merge_analysis(ctx, pass_definition, pair[0], pair[1]),
            ),
        ),
    )


def run_analysis_once(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO:
    salt = pass_definition.name + "\x00" + pass_definition.instruction
    model, params = resolve_call_settings(ctx.config, pass_definition)
    key = cache_key(full_text, model, salt, overrides=params)

    def compute(current_usage: Usage) -> IO:
        def store(result: Result) -> IO:
            if isinstance(result, Err):
                return io_pure(result)

            analysis, new_usage = result.value
            write = (
                cache_write(ctx.cache_directory, key, analysis)
                if ctx.use_cache
                else io_pure(None)
            )
            return io_map(write, lambda _: Ok((analysis, new_usage)))

        return io_bind(
            analyze_document(ctx, pass_definition, full_text, current_usage), store
        )

    if not ctx.use_cache:
        return compute(usage)

    def use_cached(cached: Optional[str]) -> IO:
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
    work_chunk: Optional[str],
    analysis: Optional[str],
    usage: Usage,
) -> IO:
    system = pass_definition.instruction + (
        ctx.settings.strict_fidelity_suffix if pass_definition.strict_fidelity else ""
    )
    user = build_pass_user(source_chunk, work_chunk, analysis)
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return chat(ctx, system, user, model, params, usage)


def ascii_fix_llm(
    ctx: Context,
    pass_name: str,
    source_paragraph: str,
    output_paragraph: str,
    usage: Usage,
) -> IO:
    salt = "ascii-fix\x00" + pass_name + "\x00" + ctx.settings.ascii_fix_instruction
    key = cache_key(source_paragraph, ctx.config.model, salt, output_paragraph)

    def attempt(index: int, user: str, last_result: str, current_usage: Usage) -> IO:
        if index > ctx.settings.ascii_fix_attempts:
            return io_pure(Ok((last_result, current_usage)))

        return io_bind(
            chat(
                ctx,
                ctx.settings.ascii_fix_instruction,
                user,
                ctx.config.model,
                dict(ctx.config.params),
                current_usage,
            ),
            lambda result: result_bind_io(
                result, lambda pair: ascii_outcome(index, pair[0], pair[1])
            ),
        )

    def ascii_outcome(index: int, result: str, current_usage: Usage) -> IO:
        if result.isascii():
            write = (
                cache_write(ctx.cache_directory, key, result)
                if ctx.use_cache
                else io_pure(None)
            )
            return io_map(write, lambda _: Ok((result, current_usage)))

        retry_user = build_ascii_retry_user(source_paragraph, output_paragraph, result)
        retry_log = verbose_log(
            ctx,
            "zh2en: ascii: attempt %d/%d still non-ASCII; retrying"
            % (index, ctx.settings.ascii_fix_attempts),
        )
        return io_bind(
            retry_log, lambda _: attempt(index + 1, retry_user, result, current_usage)
        )

    def start(current_usage: Usage) -> IO:
        return attempt(
            1,
            build_ascii_fix_user(source_paragraph, output_paragraph),
            "",
            current_usage,
        )

    if not ctx.use_cache:
        return start(usage)

    def use_cached(cached: Optional[str]) -> IO:
        if cached is None or not cached.isascii():
            return start(usage)

        return io_map(
            verbose_log(ctx, "zh2en: ascii: cache hit"),
            lambda _: Ok((cached, usage)),
        )

    return io_bind(cache_read(ctx.cache_directory, key), use_cached)


def repair_paragraph(
    ctx: Context,
    pass_name: str,
    paragraph: str,
    separator: str,
    index: int,
    total: int,
    source_paragraphs: Tuple[str, ...],
    usage: Usage,
) -> IO:
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

    source = (
        source_paragraphs[index] if index < len(source_paragraphs) else "(unavailable)"
    )

    def repaired(result: Result) -> IO:
        if isinstance(result, Err):
            return io_pure(result)

        final, warning = drop_non_ascii(
            result.value[0],
            ctx.settings.ascii_character_map,
            ctx.settings.ascii_fix_attempts,
            index,
        )

        def emit(_):
            return Ok((final + separator, result.value[1]))

        return (
            io_bind(ctx.console.log(warning), emit) if warning else io_pure(emit(None))
        )

    return io_bind(
        verbose_log(
            ctx,
            "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
            "LLM to repair it" % (index + 1, total),
        ),
        lambda _: io_bind(
            ascii_fix_llm(ctx, pass_name, source, paragraph, usage), repaired
        ),
    )


def ensure_ascii_output(
    ctx: Context,
    pass_name: str,
    text: str,
    source_paragraphs: Tuple[str, ...],
    usage: Usage,
) -> IO:
    paragraphs, separators = split_paragraphs(text)
    total = len(paragraphs)

    def step(accumulator, indexed):
        outputs, current_usage = accumulator
        index, paragraph = indexed
        separator = separators[index] if index < len(separators) else ""
        if paragraph.isascii():
            return io_pure(Ok((outputs + (paragraph + separator,), current_usage)))

        return io_map(
            repair_paragraph(
                ctx,
                pass_name,
                paragraph,
                separator,
                index,
                total,
                source_paragraphs,
                current_usage,
            ),
            lambda result: result_map(
                result, lambda pair: (outputs + (pair[0],), pair[1])
            ),
        )

    return io_map(
        fold_io(enumerate(paragraphs), step, Ok(((), usage))),
        lambda result: result_map(result, lambda pair: ("".join(pair[0]), pair[1])),
    )


def enforce_pass_ascii(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: Tuple[str, ...],
    usage: Usage,
    started_at: float,
    clock: Callable[[], float],
) -> IO:
    program = io_bind(
        ctx.console.write_partial("Enforcing ASCII... "),
        lambda _: ensure_ascii_output(
            ctx, pass_definition.name, text, source_paragraphs, usage
        ),
    )

    def conclude(result: Result) -> IO:
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
        message = (
            usage_line("Done", now(clock).run() - started_at, prompt, completion, cost)
            if prompt or completion or cost
            else "Done."
        )
        return io_map(ctx.console.finish(message), lambda _: Ok((fixed, new_usage)))

    return io_bind(program, conclude)


def log_stage(
    console: "Console",
    label,
    started_at,
    ended_at,
    start_usage: Usage,
    end_usage: Usage,
) -> IO:
    prompt, completion, cost = usage_delta(start_usage, end_usage)
    return console.log(
        usage_line(label, ended_at - started_at, prompt, completion, cost)
    )


def conclude_state(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs,
    clock,
) -> IO:
    if not pass_definition.ascii:
        return io_pure(Ok(state))

    def enforce(ascii_started: float) -> IO:
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
    source_paragraphs,
    clock: Callable[[], float],
) -> IO:
    intro_log = verbose_log(
        ctx,
        "zh2en: [%s] whole-document analysis (%d characters, ~%d tokens)"
        % (pass_definition.name, len(state.text), estimate_tokens(state.text)),
    )

    def after_analysis(analysis_result: Result) -> IO:
        if isinstance(analysis_result, Err):
            return io_pure(
                Err(
                    "zh2en: pass [%s] failed: %s"
                    % (pass_definition.name, analysis_result.error)
                )
            )

        analysis, analysis_usage = analysis_result.value
        next_state = State(text=state.text, analysis=analysis, usage=analysis_usage)

        def after_stage(ended_at: float) -> IO:
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
                    ctx, pass_definition, next_state, source_paragraphs, clock
                ),
            )

        return io_bind(now(clock), after_stage)

    return io_bind(
        intro_log,
        lambda _: io_bind(
            run_analysis_once(ctx, pass_definition, state.text, state.usage),
            after_analysis,
        ),
    )


def run_units(
    ctx: Context,
    pass_definition: PassDefinition,
    work_groups,
    plan,
    trailing_separators,
    pass_salt: str,
    analysis: Optional[str],
    usage: Usage,
) -> IO:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    total = len(work_groups)

    def step(accumulator, indexed):
        outputs, current_usage = accumulator
        index, work_group = indexed
        source_chunk_text = "\n\n".join(plan[index]) if index < len(plan) else ""
        work_chunk_text = "\n\n".join(work_group)
        key = cache_key(
            source_chunk_text, model, pass_salt, work_chunk_text, overrides=params
        )
        trailing_separator = (
            trailing_separators[index] if index < len(trailing_separators) else ""
        )

        def note(message: str) -> IO:
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

        def translate(unit_usage: Usage) -> IO:
            def store(result: Result) -> IO:
                if isinstance(result, Err):
                    return Err(
                        "zh2en: pass [%s] failed on unit %d: %s"
                        % (pass_definition.name, index + 1, result.error)
                    )

                translated = result.value[0]
                write = (
                    cache_write(ctx.cache_directory, key, translated)
                    if ctx.use_cache
                    else io_pure(None)
                )
                return io_map(
                    io_bind(
                        write,
                        lambda _: note(
                            "zh2en: [%s] unit %d/%d done"
                            % (pass_definition.name, index + 1, total)
                        ),
                    ),
                    lambda _: Ok(
                        (outputs + (translated + trailing_separator,), result.value[1])
                    ),
                )

            return io_bind(
                translate_chunk(
                    ctx,
                    pass_definition,
                    source_chunk_text,
                    work_chunk_text,
                    analysis,
                    unit_usage,
                ),
                store,
            )

        if not ctx.use_cache:
            return translate(current_usage)

        def with_cached(cached: Optional[str]) -> IO:
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

    return fold_io(enumerate(work_groups), step, Ok(((), usage)))


def run_text_pass(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    clock: Callable[[], float],
) -> IO:
    plan = paragraph_plan if pass_definition.mode == "paragraph" else chunk_plan
    work_paragraphs, _ = split_paragraphs(state.text)
    work_groups, warning = resolve_work_groups(
        pass_definition.mode,
        work_paragraphs,
        plan,
        pass_definition.name,
        ctx.settings.chunk_budget_tokens,
    )

    def proceed(_) -> IO:
        trailing_separators = unit_separators(plan, separators)
        pass_salt = pass_definition.name + "\x00" + pass_definition.instruction

        def after_units(units_result: Result) -> IO:
            if isinstance(units_result, Err):
                return io_pure(units_result)

            text = ensure_blank_line_separators("".join(units_result.value[0]))
            next_state = State(
                text=text, analysis=state.analysis, usage=units_result.value[1]
            )

            def after_stage(ended_at: float) -> IO:
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
                    trailing_separators,
                    pass_salt,
                    state.analysis,
                    state.usage,
                ),
                after_units,
            ),
        )

    return io_bind(ctx.console.log(warning) if warning else io_pure(None), proceed)


def run_pass(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    clock: Callable[[], float],
) -> IO:
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
    pass_definitions: Tuple[PassDefinition, ...],
    state: State,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    clock: Callable[[], float],
) -> IO:
    total = len(pass_definitions)

    def step(current_state: State, indexed) -> IO:
        number, pass_definition = indexed

        def launch(stage_started: float) -> IO:
            return io_bind(
                ctx.console.log(
                    "Starting pass %d/%d [%s]..."
                    % (number, total, pass_definition.name)
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


def load_instruction_text(path: str, name: str, table: dict, base_directory: str) -> IO:
    has_file = "instruction_file" in table
    has_inline = "instruction" in table
    if has_file == has_inline:
        return io_pure(
            Err(
                "%s: [[pass]] %s: exactly one of instruction_file or instruction "
                "is required" % (path, name)
            )
        )

    if has_inline:
        instruction = table["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            return io_pure(Err("%s: [[pass]] %s: instruction is empty" % (path, name)))

        return io_pure(Ok(instruction.strip()))

    instruction_file = table["instruction_file"]
    if not isinstance(instruction_file, str) or not instruction_file.strip():
        return io_pure(
            Err("%s: [[pass]] %s: instruction_file must be a path" % (path, name))
        )

    instruction_path = os.path.join(base_directory, instruction_file)

    def thunk():
        try:
            with open(instruction_path, encoding="utf-8") as handle:
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


def parse_pass_table(path: str, table: Any, base_directory: str) -> IO:
    if not isinstance(table, dict):
        return io_pure(Err("%s: [[pass]] entries must be tables" % path))

    unknown = sorted(set(table) - set(PASS_KEYS))
    if unknown:
        return io_pure(
            Err("%s: [[pass]]: unknown key(s): %s" % (path, ", ".join(unknown)))
        )

    name = table.get("name")
    if not isinstance(name, str) or not name.strip():
        return io_pure(Err("%s: [[pass]]: name must be a non-empty string" % path))

    return io_bind(
        load_instruction_text(path, name.strip(), table, base_directory),
        lambda instruction_result: io_pure(
            result_bind(
                instruction_result,
                lambda instruction: pass_definition_from(
                    path, name.strip(), table, instruction
                ),
            )
        ),
    )


def document_passes(path: str, document: dict) -> IO:
    entries = document.get("pass")
    if entries is None:
        return io_pure(Ok(()))

    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        return io_pure(Err("%s: [[pass]] must define one or more pass tables" % path))

    base_directory = os.path.dirname(os.path.abspath(path))
    return io_map(
        io_sequence(parse_pass_table(path, entry, base_directory) for entry in entries),
        results_sequence,
    )


def resolve_passes(
    user_path: str,
    user_document_result: Result,
    selected_path: Optional[str],
    selected_document_result: Result,
) -> IO:
    def resolve(documents: Tuple[dict, dict]) -> IO:
        user_document, selected_document = documents
        sources = ()
        if selected_document:
            sources += ((selected_path, selected_document),)

        if user_document:
            sources += ((user_path, user_document),)

        candidates = tuple(
            (path, document) for path, document in sources if "pass" in document
        )
        if not candidates:
            return io_pure(
                Err(
                    "no [[pass]] tables found; define at least one pass in %s"
                    % (selected_path or user_path or "a config file (see --help)")
                )
            )

        option_sources = tuple(
            (path, document) for path, document in sources if "options" in document
        )
        if option_sources:
            option_path, option_document = option_sources[0]
            options = parse_options_table(option_path, option_document["options"])
        else:
            options = Ok({})

        pass_path, pass_document = candidates[0]
        return io_map(
            document_passes(pass_path, pass_document),
            lambda result: result_bind(
                options,
                lambda option_values: result_map(
                    result, lambda passes: (option_values, passes)
                ),
            ),
        )

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


def read_document(path: str, description: str, exists: bool) -> IO:
    if not exists:
        return io_pure(Ok({}))

    return io_map(
        load_toml(path, description),
        lambda result: result_bind(
            result,
            lambda document: result_map(
                validate_document(path, document), lambda _: document
            ),
        ),
    )


def resolve_config_path(requested: Optional[str], environment: Mapping) -> IO:
    if requested:
        return io_pure(requested)

    configured = environment.get("TRANSLATE_CONFIG", "").strip()
    if configured:
        return io_pure(configured)

    def pick(cwd_value: str, user_path: str) -> Optional[str]:
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


def load_setup(arguments: Arguments, environment: Mapping) -> IO:
    def from_paths(user_path: str, selected_path: Optional[str]) -> IO:
        return io_bind(
            path_exists(user_path),
            lambda user_exists: io_bind(
                read_document(user_path, "user config", user_exists),
                lambda user_document_result: io_bind(
                    read_document(selected_path, "config file", bool(selected_path)),
                    lambda selected_document_result: io_bind(
                        resolve_passes(
                            user_path,
                            user_document_result,
                            selected_path,
                            selected_document_result,
                        ),
                        lambda passes_result: io_pure(
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
                                        )
                                    ),
                                ),
                            )
                        ),
                    ),
                ),
            ),
        )

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
        self._thread: Optional[threading.Thread] = None

    def write_partial(self, text: str) -> IO:
        def thunk():
            with self._lock:
                self._view = replace(self._view, prefix=self._view.prefix + text)
                self._stream.write(text)
                self._stream.flush()

        return IO(thunk)

    def start(self, label: str) -> IO:
        def thunk():
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

    def progress(self, label: str, count: int = 1) -> IO:
        def thunk():
            if not self._live:
                return None

            with self._lock:
                self._view = replace(
                    self._view, label=label, tokens=self._view.tokens + count
                )

            return None

        return IO(thunk)

    def stop(self) -> IO:
        def thunk():
            self._halt()
            with self._lock:
                rewrote = self._erase()
                if rewrote and self._view.prefix:
                    self._stream.write(self._view.prefix)
                    self._view = replace(self._view, drawn=self._view.prefix)

                self._stream.flush()

            return None

        return IO(thunk)

    def interrupt(self) -> IO:
        def thunk():
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

    def finish(self, text: str) -> IO:
        def thunk():
            self._halt()
            with self._lock:
                rewrote = self._erase()
                prefix = self._view.prefix if rewrote else ""
                self._stream.write(prefix + text + "\n")
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
        padding = max(len(self._view.drawn) - len(full), 0)
        self._stream.write("\r" + full + " " * padding)
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

    def log(self, message: str) -> IO:
        def thunk():
            self.status.interrupt().run()
            print(message, file=self.stream, flush=True)

        return IO(thunk)

    def write_partial(self, text: str) -> IO:
        return self.status.write_partial(text)

    def start(self, label: str) -> IO:
        return self.status.start(label)

    def progress(self, label: str, count: int = 1) -> IO:
        return self.status.progress(label, count)

    def stop(self) -> IO:
        return self.status.stop()

    def interrupt(self) -> IO:
        return self.status.interrupt()

    def finish(self, text: str) -> IO:
        return self.status.finish(text)


def parse_args(arguments) -> Arguments:
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
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) and LLM reasoning "
        "traces to stderr",
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
        verbose=parsed.verbose,
    )


def run_pipeline(
    ctx: Context, pass_definitions, text: str, started: float, stdout: TextIO, clock
) -> IO:
    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(source_paragraphs, ctx.settings.chunk_budget_tokens)
    paragraph_plan = tuple((paragraph,) for paragraph in source_paragraphs)
    initial_state = State(text=text, analysis=None, usage=Usage())

    def conclude(result: Result) -> IO:
        if isinstance(result, Err):
            return io_map(ctx.console.log(result.error), lambda _: 1)

        return finish_output(ctx, result.value, started, stdout, clock)

    return io_bind(
        run_passes(
            ctx,
            pass_definitions,
            initial_state,
            source_paragraphs,
            separators,
            chunk_plan,
            paragraph_plan,
            clock,
        ),
        conclude,
    )


def finish_output(
    ctx: Context, state: State, started: float, stdout: TextIO, clock
) -> IO:
    def after_write(_) -> IO:
        total_elapsed = clock() - started
        done_log = (
            ctx.console.log("zh2en: done in %.1fs" % total_elapsed)
            if ctx.verbose
            else io_pure(None)
        )

        def after_done(_) -> IO:
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

        return io_bind(done_log, after_done)

    return io_bind(write_stdout(stdout, state.text), after_write)


def main(arguments, environment, stdin, stdout, stderr, clock) -> IO:
    parsed = parse_args(arguments)
    return io_bind(
        read_stdin(stdin),
        lambda text: run_main_program(parsed, environment, text, stdout, stderr, clock),
    )


def run_main_program(
    parsed: Arguments, environment, text: str, stdout, stderr, clock
) -> IO:
    if not text.strip():
        return io_pure(0)

    console = Console(stderr, StatusLine(stderr))
    started = clock()

    def use_setup(setup_result: Result) -> IO:
        if isinstance(setup_result, Err):
            return io_map(console.log("zh2en: %s" % setup_result.error), lambda _: 2)

        setup = setup_result.value

        def with_cache_dir(cache_directory: str) -> IO:
            ctx = Context(
                config=setup.config,
                settings=build_settings(),
                use_cache=not parsed.no_cache,
                cache_directory=cache_directory,
                verbose=parsed.verbose,
                console=console,
            )
            return run_pipeline(ctx, setup.passes, text, started, stdout, clock)

        return io_bind(resolve_cache_dir(environment), with_cache_dir)

    return io_bind(load_setup(parsed, environment), use_setup)


if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1:],
            dict(os.environ),
            sys.stdin,
            sys.stdout,
            sys.stderr,
            time.time,
        ).run()
    )

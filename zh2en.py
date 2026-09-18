#!/usr/bin/env python3

import argparse
import configparser
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Optional


class ConfigError(Exception):
    pass


class LLMError(Exception):
    pass


class PassError(Exception):
    pass


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int


@dataclass(frozen=True)
class PassDef:
    name: str
    instruction: str
    mode: str
    strict_fidelity: bool
    api_overrides: dict
    ascii: bool


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
    ascii_character_map: dict


@dataclass(frozen=True)
class State:
    text: str
    analysis: Optional[str]
    usage: Usage


@dataclass(frozen=True)
class PassResult:
    state: Optional[State]
    error: Optional[str]


def build_settings():
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
        ascii_character_map={
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
        },
    )


def build_config(environment):
    base_url = environment.get("TRANSLATE_BASE_URL", "").rstrip("/")
    api_key = environment.get("TRANSLATE_API_KEY", "")
    model = environment.get("TRANSLATE_MODEL", "")
    timeout = parse_float_setting(environment.get("TRANSLATE_TIMEOUT", "120"), 120.0)
    max_tokens = parse_int_setting(environment.get("TRANSLATE_MAX_TOKENS", "100000"), 0)
    return Config(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
    )


def parse_float_setting(raw, default):
    try:
        return float(raw)
    except ValueError:
        return default


def parse_int_setting(raw, default):
    try:
        return int(raw)
    except ValueError:
        return default


def validate_config(config):
    missing = missing_config_keys(config)
    if missing:
        raise ConfigError(
            "missing required environment variables: " + ", ".join(missing)
        )

    if config.max_tokens <= 0:
        raise ConfigError("TRANSLATE_MAX_TOKENS must be a positive integer")

    return config


def missing_config_keys(config):
    missing = []
    if not config.base_url:
        missing.append("TRANSLATE_BASE_URL")

    if not config.api_key:
        missing.append("TRANSLATE_API_KEY")

    return missing


def infer_type(value):
    trimmed = value.strip()
    if trimmed.lower() == "none":
        return "none"

    if trimmed.lower() in ("true", "false"):
        return trimmed.lower() == "true"

    parsed_int = parse_int_setting(trimmed, None)
    if parsed_int is not None:
        return parsed_int

    parsed_float = parse_float_setting(trimmed, None)
    if parsed_float is not None:
        return parsed_float

    return trimmed


def load_passes(path):
    try:
        with open(path, encoding="utf-8") as passes_file:
            text = passes_file.read()
    except OSError as error:
        raise PassError("cannot read passes file: %s" % error) from None

    return parse_passes(text, path)


def parse_passes(text, path):
    parser = configparser.ConfigParser()
    try:
        parser.read_string(text)
    except configparser.Error as error:
        raise PassError("cannot parse passes file: %s" % error) from None

    sections = parser.sections()
    if not sections:
        raise PassError("passes file %s defines no passes" % path)

    base_directory = os.path.dirname(os.path.abspath(path))
    options = {}
    pass_definitions = []
    for section in sections:
        section_options = dict(parser.items(section))
        if section.lower() == "options":
            options.update(parse_options_section(section, section_options))
            continue

        pass_definitions.append(
            parse_pass_section(section, section_options, base_directory)
        )

    return pass_definitions, options


def parse_options_section(section, section_options):
    ascii_raw = section_options.pop("ascii", "false")
    parse_bool_value(section, "ascii", ascii_raw, "")
    if section_options:
        raise PassError(
            "[%s]: unknown option(s): %s"
            % (section, ", ".join(sorted(section_options)))
        )

    return {"ascii": ascii_raw.strip().lower() == "true"}


def parse_bool_value(section, key, raw, prefix):
    value = raw.strip().lower()
    if value not in ("true", "false"):
        raise PassError("%s[%s]: %s must be true or false" % (prefix, section, key))

    return value == "true"


def parse_pass_section(section, section_options, base_directory):
    instruction = load_instruction(section, section_options, base_directory)
    strict_fidelity = parse_bool_value(
        section,
        "strict_fidelity",
        section_options.pop("strict_fidelity", "false"),
        "pass ",
    )
    ascii_override = parse_ascii_override(section, section_options)
    mode = parse_mode(section, section_options)
    api_overrides = collect_api_overrides(section_options)
    return PassDef(
        name=section,
        instruction=instruction,
        mode=mode,
        strict_fidelity=strict_fidelity,
        api_overrides=api_overrides,
        ascii=ascii_override,
    )


def parse_ascii_override(section, section_options):
    raw = section_options.pop("ascii", None)
    if raw is None:
        return None

    return parse_bool_value(section, "ascii", raw, "pass ")


def parse_mode(section, section_options):
    mode = section_options.pop("mode", "chunk").strip().lower()
    if mode not in ("analysis", "chunk", "paragraph"):
        raise PassError(
            "pass [%s]: mode must be analysis, chunk, or paragraph" % section
        )

    return mode


def load_instruction(section, section_options, base_directory):
    instruction_file = section_options.pop("instruction-file", None)
    if not instruction_file:
        raise PassError("pass [%s]: missing required key `instruction-file`" % section)

    instruction_path = os.path.join(base_directory, instruction_file)
    try:
        with open(instruction_path, encoding="utf-8") as instruction_handle:
            instruction = instruction_handle.read().strip()
    except OSError as error:
        raise PassError(
            "pass [%s]: cannot read instruction file: %s" % (section, error)
        ) from None

    if not instruction:
        raise PassError("pass [%s]: instruction file is empty" % section)

    return instruction


def collect_api_overrides(section_options):
    overrides = {}
    for key, value in section_options.items():
        parsed = infer_type(value)
        if parsed is not None:
            overrides[key] = parsed

    return overrides


def apply_default_ascii(pass_definitions, options):
    default_ascii = options.get("ascii", False)
    return [
        (
            replace(pass_definition, ascii=default_ascii)
            if pass_definition.ascii is None
            else pass_definition
        )
        for pass_definition in pass_definitions
    ]


def log(stderr, message):
    print(message, file=stderr, flush=True)


def fmt_duration(seconds):
    if seconds < 60:
        return "%.1fs" % seconds

    return "%dm%ds" % (int(seconds // 60), int(seconds % 60))


def usage_line(label, elapsed, prompt_tokens, completion_tokens, cost):
    rate = completion_tokens / elapsed if elapsed else 0.0
    return "%s: %s, prompt=%d, completion=%d, %.1f tok/s, cost=$%.6f" % (
        label,
        fmt_duration(elapsed),
        prompt_tokens,
        completion_tokens,
        rate,
        cost,
    )


def add_usage(usage, reported):
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


def usage_delta(start_usage, end_usage):
    return (
        end_usage.prompt_tokens - start_usage.prompt_tokens,
        end_usage.completion_tokens - start_usage.completion_tokens,
        end_usage.cost - start_usage.cost,
    )


def log_stage(label, started_at, ended_at, start_usage, end_usage, stderr):
    prompt, completion, cost = usage_delta(start_usage, end_usage)
    log(
        stderr,
        usage_line(label, ended_at - started_at, prompt, completion, cost),
    )


def log_total(total_elapsed, usage, stderr):
    log(
        stderr,
        usage_line(
            "TOTAL",
            total_elapsed,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.cost,
        ),
    )


def is_cjk_char(char):
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


def estimate_tokens(text):
    cjk_count = sum(1 for char in text if is_cjk_char(char))
    return cjk_count + (len(text) - cjk_count + 3) // 4


def split_paragraphs(text):
    blank_pattern = re.compile(r"\n\s*\n")
    blanks = blank_pattern.findall(text)
    lone = text.count("\n") - sum(blank.count("\n") for blank in blanks)
    separator_pattern = r"(\n+)" if lone >= 3 * len(blanks) else r"(\n\s*\n)"
    parts = re.split(separator_pattern, text)
    is_separator = re.compile(separator_pattern).fullmatch
    paragraphs = [part for part in parts if part and not is_separator(part)]
    separators = [part for part in parts if part and is_separator(part)]
    return paragraphs, separators


def ensure_blank_line_separators(text):
    paragraphs, ignored_separators = split_paragraphs(text)
    return "\n\n".join(paragraphs)


def make_chunks(paragraphs, budget):
    chunks = []
    current = []
    size = 0
    for paragraph in paragraphs:
        paragraph_size = estimate_tokens(paragraph)
        if current and size + paragraph_size > budget:
            chunks.append(current)
            current = []
            size = 0

        current.append(paragraph)
        size += paragraph_size

    if current:
        chunks.append(current)

    return chunks


def iter_sentences(text, boundary_characters):
    buffer = []
    for char in text:
        buffer.append(char)
        if char in boundary_characters:
            yield "".join(buffer)
            buffer = []

    if buffer:
        yield "".join(buffer)


def split_to_budget(text, budget, boundary_characters):
    pieces = []
    buffer = []
    size = 0
    for sentence in iter_sentences(text, boundary_characters):
        sentence_size = estimate_tokens(sentence)
        if buffer and size + sentence_size > budget:
            pieces.append("".join(buffer))
            buffer = []
            size = 0

        buffer.append(sentence)
        size += sentence_size

    if buffer:
        pieces.append("".join(buffer))

    return pieces


def split_units_to_budget(paragraphs, budget, boundary_characters):
    units = []
    for paragraph in paragraphs:
        if estimate_tokens(paragraph) <= budget:
            units.append(paragraph)
            continue

        units.extend(split_to_budget(paragraph, budget, boundary_characters))

    return units


def unit_separators(plan, separators):
    result = []
    consumed = 0
    for index, unit in enumerate(plan):
        consumed += len(unit)
        separator = separators[consumed - 1] if consumed - 1 < len(separators) else ""
        result.append("" if index >= len(plan) - 1 else separator)

    return result


def regroup_by_plan(paragraphs, plan):
    total = sum(len(chunk) for chunk in plan)
    if len(paragraphs) != total:
        return None

    groups = []
    index = 0
    for chunk in plan:
        groups.append(paragraphs[index : index + len(chunk)])
        index += len(chunk)

    return groups


def analysis_block(analysis):
    if analysis and analysis.strip():
        return analysis.strip()

    return "(none)"


def build_chat_payload(config, system, user, overrides):
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if "openrouter" in config.base_url.lower():
        payload["provider"] = {
            "allow_fallbacks": True,
            "sort": {"by": "throughput", "partition": None},
        }

    if overrides:
        payload.update(
            {key: value for key, value in overrides.items() if value is not None}
        )

    return payload


def http_chat(config, payload):
    request = urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise LLMError("HTTP %s from endpoint: %s" % (error.code, detail)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise LLMError("could not reach endpoint: %s" % error) from None


def extract_message_content(body):
    try:
        message = body["choices"][0]["message"]
        return message, message["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape: %s" % str(body)[:500]) from None


def thinking_texts(part):
    texts = []
    for entry in part.get("thinking") or []:
        if isinstance(entry, dict) and entry.get("text"):
            texts.append(entry["text"])

    return texts


def collect_content_parts(parts):
    texts = []
    thoughts = []
    for part in parts:
        if not isinstance(part, dict):
            texts.append(str(part))
            continue

        if part.get("type") == "thinking":
            thoughts.extend(thinking_texts(part))
            continue

        if part.get("text"):
            texts.append(part["text"])

    return texts, thoughts


def flatten_content_parts(content, verbose, stderr):
    if not isinstance(content, list):
        return content

    texts, thoughts = collect_content_parts(content)
    if verbose and thoughts:
        log(stderr, "\n".join(thoughts).rstrip())

    return "".join(texts)


def print_message_reasoning(message, verbose, stderr):
    if not verbose:
        return

    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        log(stderr, reasoning.rstrip())


def strip_think_tag(content, verbose, stderr):
    if not isinstance(content, str):
        return content

    match = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    if not match:
        return content

    if verbose:
        log(stderr, match.group(1).strip())

    return content[match.end() :]


def chat(config, system, user, overrides, verbose, stderr, usage):
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > config.max_tokens:
        raise LLMError(
            "request is ~%d tokens, over the %d-token budget; raise "
            "TRANSLATE_MAX_TOKENS or shorten the input" % (estimated, config.max_tokens)
        )

    payload = build_chat_payload(config, system, user, overrides)
    body = http_chat(config, payload)
    usage = add_usage(usage, body.get("usage") or {})
    message, content = extract_message_content(body)
    print_message_reasoning(message, verbose, stderr)
    content = flatten_content_parts(content, verbose, stderr)
    content = strip_think_tag(content, verbose, stderr)
    if not isinstance(content, str):
        raise LLMError("unexpected content type: %s" % str(body)[:500])

    return content, usage


def run_analysis(config, settings, pass_definition, text, verbose, stderr, usage):
    user = settings.analysis_user_prefix + text
    content, usage = chat(
        config,
        pass_definition.instruction,
        user,
        pass_definition.api_overrides,
        verbose,
        stderr,
        usage,
    )
    return content.strip(), usage


def run_merge(config, settings, pass_definition, briefs, verbose, stderr, usage):
    user = settings.merge_user_prefix + "\n\n".join(briefs)
    content, usage = chat(
        config,
        settings.analysis_merge_instruction,
        user,
        pass_definition.api_overrides,
        verbose,
        stderr,
        usage,
    )
    return content.strip(), usage


def merge_budget(config, settings):
    overhead = estimate_tokens(settings.analysis_merge_instruction) + estimate_tokens(
        settings.merge_user_prefix
    )
    return max(config.max_tokens - overhead - settings.analysis_reserve_tokens, 1)


def merge_groups(config, settings, pass_definition, groups, verbose, stderr, usage):
    merged = []
    total_usage = usage
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue

        brief, total_usage = run_merge(
            config, settings, pass_definition, group, verbose, stderr, total_usage
        )
        merged.append(brief)

    return merged, total_usage


def merge_analysis(config, settings, pass_definition, briefs, verbose, stderr, usage):
    current = briefs
    total_usage = usage
    while len(current) > 1:
        budget = merge_budget(config, settings)
        groups = make_chunks(current, budget)
        if len(groups) == len(current):
            raise LLMError(
                "partial analysis brief (~%d tokens) does not fit the "
                "%d-token budget; raise TRANSLATE_MAX_TOKENS or shorten "
                "the input"
                % (max(estimate_tokens(brief) for brief in current), config.max_tokens)
            )

        current, total_usage = merge_groups(
            config, settings, pass_definition, groups, verbose, stderr, total_usage
        )

    return current[0], total_usage


def analyze_document(
    config, settings, pass_definition, full_text, verbose, stderr, usage
):
    overhead = estimate_tokens(pass_definition.instruction) + estimate_tokens(
        settings.analysis_user_prefix
    )
    budget = max(config.max_tokens - overhead - settings.analysis_reserve_tokens, 1)
    paragraphs, ignored_separators = split_paragraphs(full_text)
    units = split_units_to_budget(
        paragraphs, budget, settings.sentence_boundary_characters
    )
    chunks = make_chunks(units, budget)
    if len(chunks) <= 1:
        return run_analysis(
            config, settings, pass_definition, full_text, verbose, stderr, usage
        )

    if verbose:
        log(
            stderr,
            "zh2en: [%s] ~%d tokens over the %d-token budget; analysing in %d part(s)"
            % (
                pass_definition.name,
                estimate_tokens(full_text),
                config.max_tokens,
                len(chunks),
            ),
        )

    briefs = []
    total_usage = usage
    for chunk in chunks:
        brief, total_usage = run_analysis(
            config,
            settings,
            pass_definition,
            "\n\n".join(chunk),
            verbose,
            stderr,
            total_usage,
        )
        briefs.append(brief)

    return merge_analysis(
        config, settings, pass_definition, briefs, verbose, stderr, total_usage
    )


def run_analysis_once(
    config,
    settings,
    pass_definition,
    full_text,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    usage,
):
    salt = pass_definition.name + "\x00" + pass_definition.instruction
    key = cache_key(
        full_text, config.model, salt, overrides=pass_definition.api_overrides
    )
    if use_cache:
        cached = cache_get(cache_directory, key)
        if cached is not None:
            if verbose:
                log(stderr, "zh2en: [%s] cache hit" % pass_definition.name)

            return cached, usage

    result, usage = analyze_document(
        config, settings, pass_definition, full_text, verbose, stderr, usage
    )
    if use_cache:
        cache_put(cache_directory, key, result)

    return result, usage


def build_pass_user(source_chunk, work_chunk, analysis):
    user = (
        "Preparation brief from a full read of the text "
        "(outline, names, hard-to-translate items):\n%s\n\n"
        "Source text (original Chinese):\n%s" % (analysis_block(analysis), source_chunk)
    )
    if work_chunk is not None and work_chunk != source_chunk:
        user += "\n\nCurrent draft from the previous pass:\n%s" % work_chunk

    return user


def run_pass(
    config,
    settings,
    pass_definition,
    source_chunk,
    work_chunk,
    analysis,
    verbose,
    stderr,
    usage,
):
    system = pass_definition.instruction
    if pass_definition.strict_fidelity:
        system += settings.strict_fidelity_suffix

    user = build_pass_user(source_chunk, work_chunk, analysis)
    content, usage = chat(
        config, system, user, pass_definition.api_overrides, verbose, stderr, usage
    )
    return content, usage


def to_ascii_mechanical(text, character_map):
    replaced = text
    for source, replacement in character_map.items():
        replaced = replaced.replace(source, replacement)

    normalized = unicodedata.normalize("NFKD", replaced)
    return "".join(
        character for character in normalized if not unicodedata.combining(character)
    )


def non_ascii_sample(text, limit=12):
    seen = []
    for character in text:
        if not character.isascii() and character not in seen:
            seen.append(character)
            if len(seen) >= limit:
                break

    return "".join(seen)


def build_ascii_fix_user(source_paragraph, output_paragraph):
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Rewrite the translated paragraph as pure ASCII English."
        % (source_paragraph, output_paragraph)
    )


def build_ascii_retry_user(source_paragraph, output_paragraph, result):
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Your previous reply still contained these non-ASCII "
        "characters: %s. Rewrite the translated paragraph again, "
        "inferring English for every one of them from the source and "
        "context. Reply with ASCII characters only."
        % (source_paragraph, output_paragraph, non_ascii_sample(result))
    )


def ascii_fix_llm(
    config,
    settings,
    pass_name,
    source_paragraph,
    output_paragraph,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    usage,
):
    salt = "ascii-fix\x00" + pass_name + "\x00" + settings.ascii_fix_instruction
    key = cache_key(source_paragraph, config.model, salt, output_paragraph)
    if use_cache:
        cached = cache_get(cache_directory, key)
        if cached is not None and cached.isascii():
            if verbose:
                log(stderr, "zh2en: ascii: cache hit")

            return cached, usage

    user = build_ascii_fix_user(source_paragraph, output_paragraph)
    result = ""
    total_usage = usage
    for attempt in range(1, settings.ascii_fix_attempts + 1):
        result, total_usage = chat(
            config,
            settings.ascii_fix_instruction,
            user,
            None,
            verbose,
            stderr,
            total_usage,
        )
        if result.isascii():
            if use_cache:
                cache_put(cache_directory, key, result)

            return result, total_usage

        if verbose:
            log(
                stderr,
                "zh2en: ascii: attempt %d/%d still non-ASCII; retrying"
                % (attempt, settings.ascii_fix_attempts),
            )

        user = build_ascii_retry_user(source_paragraph, output_paragraph, result)

    return result, total_usage


def drop_non_ascii(text, character_map, attempts, index, stderr):
    if text.isascii():
        return text

    fallback = to_ascii_mechanical(text, character_map)
    if fallback.isascii():
        return fallback

    log(
        stderr,
        "zh2en: ascii: warning: paragraph %d still contained "
        "non-ASCII characters (%s) after %d LLM attempts; dropping "
        "them" % (index + 1, non_ascii_sample(text), attempts),
    )
    stripped = "".join(character for character in text if character.isascii())
    return re.sub(r"  +", " ", stripped)


def repair_paragraph(
    config,
    settings,
    pass_name,
    paragraph,
    separator,
    index,
    total,
    source_paragraphs,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    usage,
):
    mechanical = to_ascii_mechanical(paragraph, settings.ascii_character_map)
    if mechanical.isascii():
        if verbose:
            log(
                stderr,
                "zh2en: ascii: paragraph %d/%d converted mechanically"
                % (index + 1, total),
            )

        return mechanical + separator, usage

    if verbose:
        log(
            stderr,
            "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
            "LLM to repair it" % (index + 1, total),
        )

    source = (
        source_paragraphs[index] if index < len(source_paragraphs) else "(unavailable)"
    )
    repaired, usage = ascii_fix_llm(
        config,
        settings,
        pass_name,
        source,
        paragraph,
        use_cache,
        cache_directory,
        verbose,
        stderr,
        usage,
    )
    final = drop_non_ascii(
        repaired,
        settings.ascii_character_map,
        settings.ascii_fix_attempts,
        index,
        stderr,
    )
    return final + separator, usage


def ensure_ascii_output(
    config,
    settings,
    pass_name,
    text,
    source_paragraphs,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    usage,
):
    paragraphs, separators = split_paragraphs(text)
    outputs = []
    total_usage = usage
    for index, paragraph in enumerate(paragraphs):
        separator = separators[index] if index < len(separators) else ""
        if paragraph.isascii():
            outputs.append(paragraph + separator)
            continue

        piece, total_usage = repair_paragraph(
            config,
            settings,
            pass_name,
            paragraph,
            separator,
            index,
            len(paragraphs),
            source_paragraphs,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            total_usage,
        )
        outputs.append(piece)

    return "".join(outputs), total_usage


def resolve_cache_dir(environment):
    base = environment.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    directory = os.path.join(base, "zh2en")
    os.makedirs(directory, exist_ok=True)
    return directory


def cache_key(chunk_text, model, pass_salt="", work_text="", overrides=None):
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
            + json.dumps(overrides, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )

    return hasher.hexdigest()


def cache_get(cache_directory, key):
    path = os.path.join(cache_directory, key + ".txt")
    try:
        with open(path, encoding="utf-8") as cache_file:
            return cache_file.read()
    except OSError:
        return None


def cache_put(cache_directory, key, value):
    path = os.path.join(cache_directory, key + ".txt")
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as temporary_file:
        temporary_file.write(value)

    os.replace(temporary_path, path)


def run_units(
    config,
    settings,
    pass_definition,
    work_groups,
    plan,
    trailing_separators,
    pass_salt,
    analysis,
    usage,
    use_cache,
    cache_directory,
    verbose,
    stderr,
):
    outputs = []
    total_usage = usage
    for index, work_group in enumerate(work_groups):
        source_chunk_text = "\n\n".join(plan[index]) if index < len(plan) else ""
        work_chunk_text = "\n\n".join(work_group)
        key = cache_key(
            source_chunk_text,
            config.model,
            pass_salt,
            work_chunk_text,
            overrides=pass_definition.api_overrides,
        )
        trailing_separator = (
            trailing_separators[index] if index < len(trailing_separators) else ""
        )
        if not source_chunk_text:
            if verbose:
                log(
                    stderr,
                    "zh2en: [%s] unit %d/%d has no matching source; "
                    "passing it through unchanged"
                    % (pass_definition.name, index + 1, len(work_groups)),
                )

            outputs.append(work_chunk_text + trailing_separator)
            continue

        if use_cache:
            cached = cache_get(cache_directory, key)
            if cached is not None:
                outputs.append(cached + trailing_separator)
                if verbose:
                    log(
                        stderr,
                        "zh2en: [%s] unit %d/%d cache hit"
                        % (pass_definition.name, index + 1, len(work_groups)),
                    )

                continue

        try:
            result, total_usage = run_pass(
                config,
                settings,
                pass_definition,
                source_chunk_text,
                work_chunk_text,
                analysis,
                verbose,
                stderr,
                total_usage,
            )
        except LLMError as error:
            return (
                None,
                total_usage,
                "zh2en: pass [%s] failed on unit %d: %s"
                % (pass_definition.name, index + 1, error),
            )

        if use_cache:
            cache_put(cache_directory, key, result)

        outputs.append(result + trailing_separator)
        if verbose:
            log(
                stderr,
                "zh2en: [%s] unit %d/%d done"
                % (pass_definition.name, index + 1, len(work_groups)),
            )

    return outputs, total_usage, None


def resolve_work_groups(mode, work_paragraphs, plan, pass_name, budget, stderr):
    groups = regroup_by_plan(work_paragraphs, plan)
    if groups is not None:
        return groups

    log(
        stderr,
        "zh2en: [%s] paragraph count changed by a previous pass; "
        "grouping working text independently" % pass_name,
    )
    if mode == "paragraph":
        return [[paragraph] for paragraph in work_paragraphs]

    return make_chunks(work_paragraphs, budget)


def log_plan_info(pass_definition, work_paragraphs, work_groups, stderr):
    if pass_definition.mode == "paragraph":
        log(
            stderr,
            "zh2en: [%s] %d paragraph(s), one call per paragraph"
            % (pass_definition.name, len(work_groups)),
        )
        return

    log(
        stderr,
        "zh2en: [%s] %d paragraph(s) in %d chunk(s)"
        % (pass_definition.name, len(work_paragraphs), len(work_groups)),
    )


def enforce_pass_ascii(
    config,
    settings,
    pass_definition,
    text,
    source_paragraphs,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    usage,
    started_at,
    clock,
):
    log(stderr, "Starting ascii enforcement for [%s]..." % pass_definition.name)
    try:
        fixed, updated_usage = ensure_ascii_output(
            config,
            settings,
            pass_definition.name,
            text,
            source_paragraphs,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            usage,
        )
    except LLMError as error:
        return (
            text,
            usage,
            "zh2en: pass [%s] ascii enforcement failed: %s"
            % (pass_definition.name, error),
        )

    log_stage("Done", started_at, clock(), usage, updated_usage, stderr)
    return fixed, updated_usage, None


def run_analysis_pass(
    config,
    settings,
    pass_definition,
    state,
    started_at,
    source_paragraphs,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    clock,
):
    if verbose:
        log(
            stderr,
            "zh2en: [%s] whole-document analysis (%d characters, ~%d tokens)"
            % (pass_definition.name, len(state.text), estimate_tokens(state.text)),
        )

    try:
        analysis, usage = run_analysis_once(
            config,
            settings,
            pass_definition,
            state.text,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            state.usage,
        )
    except LLMError as error:
        return PassResult(
            None, "zh2en: pass [%s] failed: %s" % (pass_definition.name, error)
        )

    log_stage("Done", started_at, clock(), state.usage, usage, stderr)
    if pass_definition.ascii:
        analysis, usage, error = enforce_pass_ascii(
            config,
            settings,
            pass_definition,
            analysis,
            source_paragraphs,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            usage,
            clock(),
            clock,
        )
        if error is not None:
            return PassResult(None, error)

    return PassResult(State(state.text, analysis, usage), None)


def run_text_pass(
    config,
    settings,
    pass_definition,
    state,
    started_at,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    clock,
):
    plan = paragraph_plan if pass_definition.mode == "paragraph" else chunk_plan
    work_paragraphs, ignored_separators = split_paragraphs(state.text)
    work_groups = resolve_work_groups(
        pass_definition.mode,
        work_paragraphs,
        plan,
        pass_definition.name,
        settings.chunk_budget_tokens,
        stderr,
    )
    if verbose:
        log_plan_info(pass_definition, work_paragraphs, work_groups, stderr)

    trailing_separators = unit_separators(plan, separators)
    pass_salt = pass_definition.name + "\x00" + pass_definition.instruction
    outputs, usage, error = run_units(
        config,
        settings,
        pass_definition,
        work_groups,
        plan,
        trailing_separators,
        pass_salt,
        state.analysis,
        state.usage,
        use_cache,
        cache_directory,
        verbose,
        stderr,
    )
    if error is not None:
        return PassResult(None, error)

    log_stage("Done", started_at, clock(), state.usage, usage, stderr)
    text = ensure_blank_line_separators("".join(outputs))
    if pass_definition.ascii:
        text, usage, error = enforce_pass_ascii(
            config,
            settings,
            pass_definition,
            text,
            source_paragraphs,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            usage,
            clock(),
            clock,
        )
        if error is not None:
            return PassResult(None, error)

    return PassResult(State(text, state.analysis, usage), None)


def run_one_pass(
    config,
    settings,
    pass_definition,
    state,
    started_at,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    clock,
):
    if pass_definition.mode == "analysis":
        return run_analysis_pass(
            config,
            settings,
            pass_definition,
            state,
            started_at,
            source_paragraphs,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            clock,
        )

    return run_text_pass(
        config,
        settings,
        pass_definition,
        state,
        started_at,
        source_paragraphs,
        separators,
        chunk_plan,
        paragraph_plan,
        use_cache,
        cache_directory,
        verbose,
        stderr,
        clock,
    )


def run_passes(
    config,
    settings,
    pass_definitions,
    state,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    use_cache,
    cache_directory,
    verbose,
    stderr,
    clock,
):
    stage_started = clock()
    for number, pass_definition in enumerate(pass_definitions, 1):
        log(
            stderr,
            "Starting pass %d/%d [%s]..."
            % (number, len(pass_definitions), pass_definition.name),
        )
        result = run_one_pass(
            config,
            settings,
            pass_definition,
            state,
            stage_started,
            source_paragraphs,
            separators,
            chunk_plan,
            paragraph_plan,
            use_cache,
            cache_directory,
            verbose,
            stderr,
            clock,
        )
        if result.error is not None:
            return result

        state = result.state
        stage_started = clock()

    return PassResult(state, None)


def write_output(stdout, text):
    stdout.write(text)
    if not text.endswith("\n"):
        stdout.write("\n")


def parse_args(arguments):
    parser = argparse.ArgumentParser(
        prog="zh2en",
        description="Translate Chinese text from stdin to English on stdout.",
    )
    parser.add_argument(
        "passes",
        metavar="PASSES_INI",
        help="INI file defining the translation passes (in execution order)",
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
    return parser.parse_args(arguments)


def main(arguments, environment, stdin, stdout, stderr, clock):
    parsed_arguments = parse_args(arguments)
    text = stdin.read()
    if not text.strip():
        return 0

    started = clock()
    try:
        config = validate_config(build_config(environment))
    except ConfigError as error:
        log(stderr, "zh2en: %s" % error)
        return 2

    try:
        pass_definitions, options = load_passes(parsed_arguments.passes)
    except PassError as error:
        log(stderr, "zh2en: %s" % error)
        return 2

    pass_definitions = apply_default_ascii(pass_definitions, options)
    settings = build_settings()
    cache_path = resolve_cache_dir(environment)
    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(source_paragraphs, settings.chunk_budget_tokens)
    paragraph_plan = [[paragraph] for paragraph in source_paragraphs]
    state = State(text=text, analysis=None, usage=Usage())
    result = run_passes(
        config,
        settings,
        pass_definitions,
        state,
        source_paragraphs,
        separators,
        chunk_plan,
        paragraph_plan,
        not parsed_arguments.no_cache,
        cache_path,
        parsed_arguments.verbose,
        stderr,
        clock,
    )
    if result.error is not None:
        log(stderr, result.error)
        return 1

    write_output(stdout, result.state.text)
    total_elapsed = clock() - started
    if parsed_arguments.verbose:
        log(stderr, "zh2en: done in %.1fs" % total_elapsed)

    log_total(total_elapsed, result.state.usage, stderr)
    return 0


if __name__ == "__main__":
    sys.exit(
        main(
            sys.argv[1:],
            dict(os.environ),
            sys.stdin,
            sys.stdout,
            sys.stderr,
            time.time,
        )
    )

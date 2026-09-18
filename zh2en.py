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
    sentence_boundary_chars: str
    strict_fidelity_suffix: str
    analysis_merge_instruction: str
    analysis_user_prefix: str
    merge_user_prefix: str
    ascii_fix_instruction: str
    ascii_char_map: dict


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
        sentence_boundary_chars="。！？!?；;\n",
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
        ascii_char_map={
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


def build_config(env):
    base_url = env.get("TRANSLATE_BASE_URL", "").rstrip("/")
    api_key = env.get("TRANSLATE_API_KEY", "")
    model = env.get("TRANSLATE_MODEL", "")
    timeout = parse_float_setting(env.get("TRANSLATE_TIMEOUT", "120"), 120.0)
    max_tokens = parse_int_setting(env.get("TRANSLATE_MAX_TOKENS", "100000"), 0)
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
    v = value.strip()
    if v.lower() == "none":
        return "none"

    if v.lower() in ("true", "false"):
        return v.lower() == "true"

    parsed_int = parse_int_setting(v, None)
    if parsed_int is not None:
        return parsed_int

    parsed_float = parse_float_setting(v, None)
    if parsed_float is not None:
        return parsed_float

    return v


def load_passes(path):
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        raise PassError("cannot read passes file: %s" % e) from None

    return parse_passes(text, path)


def parse_passes(text, path):
    parser = configparser.ConfigParser()
    try:
        parser.read_string(text)
    except configparser.Error as e:
        raise PassError("cannot parse passes file: %s" % e) from None

    sections = parser.sections()
    if not sections:
        raise PassError("passes file %s defines no passes" % path)

    base_dir = os.path.dirname(os.path.abspath(path))
    options = {}
    passdefs = []
    for section in sections:
        opts = dict(parser.items(section))
        if section.lower() == "options":
            options.update(parse_options_section(section, opts))
            continue

        passdefs.append(parse_pass_section(section, opts, base_dir))

    return passdefs, options


def parse_options_section(section, opts):
    ascii_raw = opts.pop("ascii", "false")
    parse_bool_value(section, "ascii", ascii_raw, "")
    if opts:
        raise PassError(
            "[%s]: unknown option(s): %s" % (section, ", ".join(sorted(opts)))
        )

    return {"ascii": ascii_raw.strip().lower() == "true"}


def parse_bool_value(section, key, raw, prefix):
    value = raw.strip().lower()
    if value not in ("true", "false"):
        raise PassError("%s[%s]: %s must be true or false" % (prefix, section, key))

    return value == "true"


def parse_pass_section(section, opts, base_dir):
    instruction = load_instruction(section, opts, base_dir)
    strict_fidelity = parse_bool_value(
        section, "strict_fidelity", opts.pop("strict_fidelity", "false"), "pass "
    )
    ascii_override = parse_ascii_override(section, opts)
    mode = parse_mode(section, opts)
    api_overrides = collect_api_overrides(opts)
    return PassDef(
        name=section,
        instruction=instruction,
        mode=mode,
        strict_fidelity=strict_fidelity,
        api_overrides=api_overrides,
        ascii=ascii_override,
    )


def parse_ascii_override(section, opts):
    raw = opts.pop("ascii", None)
    if raw is None:
        return None

    return parse_bool_value(section, "ascii", raw, "pass ")


def parse_mode(section, opts):
    mode = opts.pop("mode", "chunk").strip().lower()
    if mode not in ("analysis", "chunk", "paragraph"):
        raise PassError(
            "pass [%s]: mode must be analysis, chunk, or paragraph" % section
        )

    return mode


def load_instruction(section, opts, base_dir):
    instruction_file = opts.pop("instruction-file", None)
    if not instruction_file:
        raise PassError("pass [%s]: missing required key `instruction-file`" % section)

    instruction_path = os.path.join(base_dir, instruction_file)
    try:
        with open(instruction_path, encoding="utf-8") as f:
            instruction = f.read().strip()
    except OSError as e:
        raise PassError(
            "pass [%s]: cannot read instruction file: %s" % (section, e)
        ) from None

    if not instruction:
        raise PassError("pass [%s]: instruction file is empty" % section)

    return instruction


def collect_api_overrides(opts):
    overrides = {}
    for key, value in opts.items():
        parsed = infer_type(value)
        if parsed is not None:
            overrides[key] = parsed

    return overrides


def apply_default_ascii(passdefs, options):
    default_ascii = options.get("ascii", False)
    return [
        replace(passdef, ascii=default_ascii) if passdef.ascii is None else passdef
        for passdef in passdefs
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
    blank_re = re.compile(r"\n\s*\n")
    blanks = blank_re.findall(text)
    lone = text.count("\n") - sum(s.count("\n") for s in blanks)
    sep_pattern = r"(\n+)" if lone >= 3 * len(blanks) else r"(\n\s*\n)"
    parts = re.split(sep_pattern, text)
    is_sep = re.compile(sep_pattern).fullmatch
    paragraphs = [p for p in parts if p and not is_sep(p)]
    separators = [s for s in parts if s and is_sep(s)]
    return paragraphs, separators


def ensure_blank_line_separators(text):
    paragraphs, _ = split_paragraphs(text)
    return "\n\n".join(paragraphs)


def make_chunks(paragraphs, budget):
    chunks = []
    current = []
    size = 0
    for paragraph in paragraphs:
        p_size = estimate_tokens(paragraph)
        if current and size + p_size > budget:
            chunks.append(current)
            current = []
            size = 0

        current.append(paragraph)
        size += p_size

    if current:
        chunks.append(current)

    return chunks


def iter_sentences(text, boundary_chars):
    buf = []
    for char in text:
        buf.append(char)
        if char in boundary_chars:
            yield "".join(buf)
            buf = []

    if buf:
        yield "".join(buf)


def split_to_budget(text, budget, boundary_chars):
    pieces = []
    buf = []
    size = 0
    for sentence in iter_sentences(text, boundary_chars):
        s_size = estimate_tokens(sentence)
        if buf and size + s_size > budget:
            pieces.append("".join(buf))
            buf = []
            size = 0

        buf.append(sentence)
        size += s_size

    if buf:
        pieces.append("".join(buf))

    return pieces


def split_units_to_budget(paragraphs, budget, boundary_chars):
    units = []
    for para in paragraphs:
        if estimate_tokens(para) <= budget:
            units.append(para)
            continue

        units.extend(split_to_budget(para, budget, boundary_chars))

    return units


def unit_separators(plan, separators):
    result = []
    consumed = 0
    for i, unit in enumerate(plan):
        consumed += len(unit)
        sep = separators[consumed - 1] if consumed - 1 < len(separators) else ""
        result.append("" if i >= len(plan) - 1 else sep)

    return result


def regroup_by_plan(paragraphs, plan):
    total = sum(len(chunk) for chunk in plan)
    if len(paragraphs) != total:
        return None

    groups = []
    i = 0
    for chunk in plan:
        groups.append(paragraphs[i : i + len(chunk)])
        i += len(chunk)

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
        payload.update({k: v for k, v in overrides.items() if v is not None})

    return payload


def http_chat(config, payload):
    req = urllib.request.Request(
        config.base_url + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.api_key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise LLMError("HTTP %s from endpoint: %s" % (e.code, detail)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError("could not reach endpoint: %s" % e) from None


def extract_message_content(body):
    try:
        message = body["choices"][0]["message"]
        return message, message["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape: %s" % str(body)[:500]) from None


def thinking_texts(part):
    texts = []
    for sub in part.get("thinking") or []:
        if isinstance(sub, dict) and sub.get("text"):
            texts.append(sub["text"])

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
    new_usage = add_usage(usage, body.get("usage") or {})
    message, content = extract_message_content(body)
    print_message_reasoning(message, verbose, stderr)
    content = flatten_content_parts(content, verbose, stderr)
    content = strip_think_tag(content, verbose, stderr)
    if not isinstance(content, str):
        raise LLMError("unexpected content type: %s" % str(body)[:500])

    return content, new_usage


def run_analysis(config, settings, passdef, text, verbose, stderr, usage):
    user = settings.analysis_user_prefix + text
    content, new_usage = chat(
        config, passdef.instruction, user, passdef.api_overrides, verbose, stderr, usage
    )
    return content.strip(), new_usage


def run_merge(config, settings, passdef, briefs, verbose, stderr, usage):
    user = settings.merge_user_prefix + "\n\n".join(briefs)
    content, new_usage = chat(
        config,
        settings.analysis_merge_instruction,
        user,
        passdef.api_overrides,
        verbose,
        stderr,
        usage,
    )
    return content.strip(), new_usage


def merge_budget(config, settings):
    overhead = estimate_tokens(settings.analysis_merge_instruction) + estimate_tokens(
        settings.merge_user_prefix
    )
    return max(config.max_tokens - overhead - settings.analysis_reserve_tokens, 1)


def merge_groups(config, settings, passdef, groups, verbose, stderr, usage):
    merged = []
    total_usage = usage
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue

        brief, total_usage = run_merge(
            config, settings, passdef, group, verbose, stderr, total_usage
        )
        merged.append(brief)

    return merged, total_usage


def merge_analysis(config, settings, passdef, briefs, verbose, stderr, usage):
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
                % (max(estimate_tokens(b) for b in current), config.max_tokens)
            )

        current, total_usage = merge_groups(
            config, settings, passdef, groups, verbose, stderr, total_usage
        )

    return current[0], total_usage


def analyze_document(config, settings, passdef, full_text, verbose, stderr, usage):
    overhead = estimate_tokens(passdef.instruction) + estimate_tokens(
        settings.analysis_user_prefix
    )
    budget = max(config.max_tokens - overhead - settings.analysis_reserve_tokens, 1)
    paragraphs, _ = split_paragraphs(full_text)
    units = split_units_to_budget(paragraphs, budget, settings.sentence_boundary_chars)
    chunks = make_chunks(units, budget)
    if len(chunks) <= 1:
        return run_analysis(
            config, settings, passdef, full_text, verbose, stderr, usage
        )

    if verbose:
        log(
            stderr,
            "zh2en: [%s] ~%d tokens over the %d-token budget; analysing in %d part(s)"
            % (
                passdef.name,
                estimate_tokens(full_text),
                config.max_tokens,
                len(chunks),
            ),
        )

    briefs = []
    total_usage = usage
    for chunk in chunks:
        brief, total_usage = run_analysis(
            config, settings, passdef, "\n\n".join(chunk), verbose, stderr, total_usage
        )
        briefs.append(brief)

    return merge_analysis(
        config, settings, passdef, briefs, verbose, stderr, total_usage
    )


def run_analysis_once(
    config, settings, passdef, full_text, use_cache, cache_dir, verbose, stderr, usage
):
    salt = passdef.name + "\x00" + passdef.instruction
    key = cache_key(full_text, config.model, salt, overrides=passdef.api_overrides)
    if use_cache:
        cached = cache_get(cache_dir, key)
        if cached is not None:
            if verbose:
                log(stderr, "zh2en: [%s] cache hit" % passdef.name)

            return cached, usage

    result, new_usage = analyze_document(
        config, settings, passdef, full_text, verbose, stderr, usage
    )
    if use_cache:
        cache_put(cache_dir, key, result)

    return result, new_usage


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
    passdef,
    source_chunk,
    work_chunk,
    analysis,
    verbose,
    stderr,
    usage,
):
    system = passdef.instruction
    if passdef.strict_fidelity:
        system += settings.strict_fidelity_suffix

    user = build_pass_user(source_chunk, work_chunk, analysis)
    content, new_usage = chat(
        config, system, user, passdef.api_overrides, verbose, stderr, usage
    )
    return clean_translation(content), new_usage


def clean_translation(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)

    if len(t) >= 2:
        pairs = {'"': '"', "'": "'", "\u201c": "\u201d", "\u2018": "\u2019"}
        if t[0] in pairs and t[-1] == pairs[t[0]]:
            t = t[1:-1]

    t = re.sub(r"^Translation:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def to_ascii_mechanical(text, char_map):
    replaced = text
    for src, repl in char_map.items():
        replaced = replaced.replace(src, repl)

    normalized = unicodedata.normalize("NFKD", replaced)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def non_ascii_sample(text, limit=12):
    seen = []
    for c in text:
        if not c.isascii() and c not in seen:
            seen.append(c)
            if len(seen) >= limit:
                break

    return "".join(seen)


def build_ascii_fix_user(source_para, out_para):
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Rewrite the translated paragraph as pure ASCII English."
        % (source_para, out_para)
    )


def build_ascii_retry_user(source_para, out_para, result):
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Your previous reply still contained these non-ASCII "
        "characters: %s. Rewrite the translated paragraph again, "
        "inferring English for every one of them from the source and "
        "context. Reply with ASCII characters only."
        % (source_para, out_para, non_ascii_sample(result))
    )


def ascii_fix_llm(
    config,
    settings,
    pass_name,
    source_para,
    out_para,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    usage,
):
    salt = "ascii-fix\x00" + pass_name + "\x00" + settings.ascii_fix_instruction
    key = cache_key(source_para, config.model, salt, out_para)
    if use_cache:
        cached = cache_get(cache_dir, key)
        if cached is not None and cached.isascii():
            if verbose:
                log(stderr, "zh2en: ascii: cache hit")

            return cached, usage

    user = build_ascii_fix_user(source_para, out_para)
    result = ""
    total_usage = usage
    for attempt in range(1, settings.ascii_fix_attempts + 1):
        reply, total_usage = chat(
            config,
            settings.ascii_fix_instruction,
            user,
            None,
            verbose,
            stderr,
            total_usage,
        )
        result = clean_translation(reply)
        if result.isascii():
            if use_cache:
                cache_put(cache_dir, key, result)

            return result, total_usage

        if verbose:
            log(
                stderr,
                "zh2en: ascii: attempt %d/%d still non-ASCII; retrying"
                % (attempt, settings.ascii_fix_attempts),
            )

        user = build_ascii_retry_user(source_para, out_para, result)

    return result, total_usage


def drop_non_ascii(text, char_map, attempts, index, stderr):
    if text.isascii():
        return text

    fallback = to_ascii_mechanical(text, char_map)
    if fallback.isascii():
        return fallback

    log(
        stderr,
        "zh2en: ascii: warning: paragraph %d still contained "
        "non-ASCII characters (%s) after %d LLM attempts; dropping "
        "them" % (index + 1, non_ascii_sample(text), attempts),
    )
    stripped = "".join(c for c in text if c.isascii())
    return re.sub(r"  +", " ", stripped)


def repair_paragraph(
    config,
    settings,
    pass_name,
    para,
    sep,
    index,
    total,
    source_paragraphs,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    usage,
):
    mechanical = to_ascii_mechanical(para, settings.ascii_char_map)
    if mechanical.isascii():
        if verbose:
            log(
                stderr,
                "zh2en: ascii: paragraph %d/%d converted mechanically"
                % (index + 1, total),
            )

        return mechanical + sep, usage

    if verbose:
        log(
            stderr,
            "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
            "LLM to repair it" % (index + 1, total),
        )

    source = (
        source_paragraphs[index] if index < len(source_paragraphs) else "(unavailable)"
    )
    repaired, new_usage = ascii_fix_llm(
        config,
        settings,
        pass_name,
        source,
        para,
        use_cache,
        cache_dir,
        verbose,
        stderr,
        usage,
    )
    final = drop_non_ascii(
        repaired, settings.ascii_char_map, settings.ascii_fix_attempts, index, stderr
    )
    return final + sep, new_usage


def ensure_ascii_output(
    config,
    settings,
    pass_name,
    text,
    source_paragraphs,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    usage,
):
    paragraphs, separators = split_paragraphs(text)
    outputs = []
    total_usage = usage
    for i, para in enumerate(paragraphs):
        sep = separators[i] if i < len(separators) else ""
        if para.isascii():
            outputs.append(para + sep)
            continue

        piece, total_usage = repair_paragraph(
            config,
            settings,
            pass_name,
            para,
            sep,
            i,
            len(paragraphs),
            source_paragraphs,
            use_cache,
            cache_dir,
            verbose,
            stderr,
            total_usage,
        )
        outputs.append(piece)

    return "".join(outputs), total_usage


def resolve_cache_dir(env):
    base = env.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    directory = os.path.join(base, "zh2en")
    os.makedirs(directory, exist_ok=True)
    return directory


def cache_key(chunk_text, model, pass_salt="", work_text="", overrides=None):
    h = hashlib.sha256()
    if pass_salt:
        h.update(pass_salt.encode("utf-8") + b"\x00")

    h.update(chunk_text.encode("utf-8"))
    if work_text:
        h.update(b"\x00work\x00" + work_text.encode("utf-8"))

    h.update(b"\x00" + model.encode("utf-8"))
    if overrides:
        h.update(
            b"\x00"
            + json.dumps(overrides, sort_keys=True, ensure_ascii=False).encode("utf-8")
        )

    return h.hexdigest()


def cache_get(cache_dir, key):
    path = os.path.join(cache_dir, key + ".txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def cache_put(cache_dir, key, value):
    path = os.path.join(cache_dir, key + ".txt")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(value)

    os.replace(tmp, path)


def run_units(
    config,
    settings,
    passdef,
    work_groups,
    plan,
    unit_seps,
    pass_salt,
    analysis,
    usage,
    use_cache,
    cache_dir,
    verbose,
    stderr,
):
    outputs = []
    total_usage = usage
    for i, work_group in enumerate(work_groups):
        source_chunk_text = "\n\n".join(plan[i]) if i < len(plan) else ""
        work_chunk_text = "\n\n".join(work_group)
        key = cache_key(
            source_chunk_text,
            config.model,
            pass_salt,
            work_chunk_text,
            overrides=passdef.api_overrides,
        )
        trailing_sep = unit_seps[i] if i < len(unit_seps) else ""
        if not source_chunk_text:
            if verbose:
                log(
                    stderr,
                    "zh2en: [%s] unit %d/%d has no matching source; "
                    "passing it through unchanged"
                    % (passdef.name, i + 1, len(work_groups)),
                )

            outputs.append(work_chunk_text + trailing_sep)
            continue

        if use_cache:
            cached = cache_get(cache_dir, key)
            if cached is not None:
                outputs.append(cached + trailing_sep)
                if verbose:
                    log(
                        stderr,
                        "zh2en: [%s] unit %d/%d cache hit"
                        % (passdef.name, i + 1, len(work_groups)),
                    )

                continue

        try:
            result, total_usage = run_pass(
                config,
                settings,
                passdef,
                source_chunk_text,
                work_chunk_text,
                analysis,
                verbose,
                stderr,
                total_usage,
            )
        except LLMError as e:
            return (
                None,
                total_usage,
                "zh2en: pass [%s] failed on unit %d: %s" % (passdef.name, i + 1, e),
            )

        if use_cache:
            cache_put(cache_dir, key, result)

        outputs.append(result + trailing_sep)
        if verbose:
            log(
                stderr,
                "zh2en: [%s] unit %d/%d done" % (passdef.name, i + 1, len(work_groups)),
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
        return [[p] for p in work_paragraphs]

    return make_chunks(work_paragraphs, budget)


def log_plan_info(passdef, work_paragraphs, work_groups, stderr):
    if passdef.mode == "paragraph":
        log(
            stderr,
            "zh2en: [%s] %d paragraph(s), one call per paragraph"
            % (passdef.name, len(work_groups)),
        )
        return

    log(
        stderr,
        "zh2en: [%s] %d paragraph(s) in %d chunk(s)"
        % (passdef.name, len(work_paragraphs), len(work_groups)),
    )


def enforce_pass_ascii(
    config,
    settings,
    passdef,
    text,
    source_paragraphs,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    usage,
    started_at,
    clock,
):
    log(stderr, "Starting ascii enforcement for [%s]..." % passdef.name)
    try:
        fixed, new_usage = ensure_ascii_output(
            config,
            settings,
            passdef.name,
            text,
            source_paragraphs,
            use_cache,
            cache_dir,
            verbose,
            stderr,
            usage,
        )
    except LLMError as e:
        return (
            text,
            usage,
            "zh2en: pass [%s] ascii enforcement failed: %s" % (passdef.name, e),
        )

    log_stage("Done", started_at, clock(), usage, new_usage, stderr)
    return fixed, new_usage, None


def run_analysis_pass(
    config,
    settings,
    passdef,
    state,
    started_at,
    source_paragraphs,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    clock,
):
    if verbose:
        log(
            stderr,
            "zh2en: [%s] whole-document analysis (%d characters, ~%d tokens)"
            % (passdef.name, len(state.text), estimate_tokens(state.text)),
        )

    try:
        analysis, new_usage = run_analysis_once(
            config,
            settings,
            passdef,
            state.text,
            use_cache,
            cache_dir,
            verbose,
            stderr,
            state.usage,
        )
    except LLMError as e:
        return PassResult(None, "zh2en: pass [%s] failed: %s" % (passdef.name, e))

    log_stage("Done", started_at, clock(), state.usage, new_usage, stderr)
    usage = new_usage
    if passdef.ascii:
        fixed, usage, error = enforce_pass_ascii(
            config,
            settings,
            passdef,
            analysis,
            source_paragraphs,
            use_cache,
            cache_dir,
            verbose,
            stderr,
            usage,
            clock(),
            clock,
        )
        if error is not None:
            return PassResult(None, error)

        analysis = fixed

    return PassResult(State(state.text, analysis, usage), None)


def run_text_pass(
    config,
    settings,
    passdef,
    state,
    started_at,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    clock,
):
    plan = paragraph_plan if passdef.mode == "paragraph" else chunk_plan
    work_paragraphs, _ = split_paragraphs(state.text)
    work_groups = resolve_work_groups(
        passdef.mode,
        work_paragraphs,
        plan,
        passdef.name,
        settings.chunk_budget_tokens,
        stderr,
    )
    if verbose:
        log_plan_info(passdef, work_paragraphs, work_groups, stderr)

    unit_seps = unit_separators(plan, separators)
    pass_salt = passdef.name + "\x00" + passdef.instruction
    outputs, usage, error = run_units(
        config,
        settings,
        passdef,
        work_groups,
        plan,
        unit_seps,
        pass_salt,
        state.analysis,
        state.usage,
        use_cache,
        cache_dir,
        verbose,
        stderr,
    )
    if error is not None:
        return PassResult(None, error)

    log_stage("Done", started_at, clock(), state.usage, usage, stderr)
    text = ensure_blank_line_separators("".join(outputs))
    if passdef.ascii:
        text, usage, error = enforce_pass_ascii(
            config,
            settings,
            passdef,
            text,
            source_paragraphs,
            use_cache,
            cache_dir,
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
    passdef,
    state,
    started_at,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    clock,
):
    if passdef.mode == "analysis":
        return run_analysis_pass(
            config,
            settings,
            passdef,
            state,
            started_at,
            source_paragraphs,
            use_cache,
            cache_dir,
            verbose,
            stderr,
            clock,
        )

    return run_text_pass(
        config,
        settings,
        passdef,
        state,
        started_at,
        source_paragraphs,
        separators,
        chunk_plan,
        paragraph_plan,
        use_cache,
        cache_dir,
        verbose,
        stderr,
        clock,
    )


def run_passes(
    config,
    settings,
    passdefs,
    state,
    source_paragraphs,
    separators,
    chunk_plan,
    paragraph_plan,
    use_cache,
    cache_dir,
    verbose,
    stderr,
    clock,
):
    stage_started = clock()
    for number, passdef in enumerate(passdefs, 1):
        log(
            stderr,
            "Starting pass %d/%d [%s]..." % (number, len(passdefs), passdef.name),
        )
        result = run_one_pass(
            config,
            settings,
            passdef,
            state,
            stage_started,
            source_paragraphs,
            separators,
            chunk_plan,
            paragraph_plan,
            use_cache,
            cache_dir,
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


def parse_args(argv):
    ap = argparse.ArgumentParser(
        prog="zh2en",
        description="Translate Chinese text from stdin to English on stdout.",
    )
    ap.add_argument(
        "passes",
        metavar="PASSES_INI",
        help="INI file defining the translation passes (in execution order)",
    )
    ap.add_argument(
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) and LLM reasoning "
        "traces to stderr",
    )
    return ap.parse_args(argv)


def main(argv, env, stdin, stdout, stderr, clock):
    args = parse_args(argv)
    text = stdin.read()
    if not text.strip():
        return 0

    started = clock()
    try:
        config = validate_config(build_config(env))
    except ConfigError as e:
        log(stderr, "zh2en: %s" % e)
        return 2

    try:
        passdefs, options = load_passes(args.passes)
    except PassError as e:
        log(stderr, "zh2en: %s" % e)
        return 2

    passdefs = apply_default_ascii(passdefs, options)
    settings = build_settings()
    cache_path = resolve_cache_dir(env)
    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(source_paragraphs, settings.chunk_budget_tokens)
    paragraph_plan = [[p] for p in source_paragraphs]
    state = State(text=text, analysis=None, usage=Usage())
    result = run_passes(
        config,
        settings,
        passdefs,
        state,
        source_paragraphs,
        separators,
        chunk_plan,
        paragraph_plan,
        not args.no_cache,
        cache_path,
        args.verbose,
        stderr,
        clock,
    )
    if result.error is not None:
        log(stderr, result.error)
        return 1

    write_output(stdout, result.state.text)
    total_elapsed = clock() - started
    if args.verbose:
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

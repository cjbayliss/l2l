#!/usr/bin/env python3

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
    params: dict


@dataclass(frozen=True)
class PassDefinition:
    name: str
    instruction: str
    mode: str
    strict_fidelity: bool
    params: dict
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


def default_api_settings():
    return {
        "base_url": "",
        "api_key": "",
        "model": "",
        "timeout": 120.0,
        "max_tokens": 100000,
        "params": {},
    }


def user_config_path(environment):
    base = environment.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "zh2en", "config.toml")


def load_toml(path, description):
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError("%s not found: %s" % (description, path)) from None
    except OSError as error:
        raise ConfigError("cannot read %s: %s" % (description, error)) from None
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            "cannot parse %s %s: %s" % (description, path, error)
        ) from None


def validate_document(path, document):
    if not document:
        return

    unknown = sorted(set(document) - {"api", "options", "pass"})
    if unknown:
        raise ConfigError(
            "%s: unknown top-level key(s): %s (expected [api], [options], [[pass]])"
            % (path, ", ".join(unknown))
        )


def document_api_settings(path, document):
    if "api" not in document:
        return {}

    table = document["api"]
    if not isinstance(table, dict):
        raise ConfigError("%s: [api] must be a table" % path)

    unknown = sorted(set(table) - set(API_SETTING_KEYS))
    if unknown:
        raise ConfigError("%s: [api]: unknown key(s): %s" % (path, ", ".join(unknown)))

    settings = {}
    for key in ("base_url", "api_key", "model"):
        if key in table:
            value = table[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(
                    "%s: [api] %s must be a non-empty string" % (path, key)
                )

            settings[key] = value.strip()

    if "timeout" in table:
        value = table["timeout"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ConfigError("%s: [api] timeout must be a positive number" % path)

        settings["timeout"] = float(value)

    if "max_tokens" in table:
        value = table["max_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigError("%s: [api] max_tokens must be a positive integer" % path)

        settings["max_tokens"] = value

    if "params" in table:
        value = table["params"]
        if not isinstance(value, dict):
            raise ConfigError("%s: [api] params must be a table" % path)

        settings["params"] = value

    return settings


def merge_api_settings(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if key == "params" and isinstance(merged.get("params"), dict):
            combined = dict(merged["params"])
            combined.update(value)
            merged["params"] = combined
        else:
            merged[key] = value

    return merged


def api_settings_from_environment(environment):
    settings = {}
    for variable, key in (
        ("TRANSLATE_BASE_URL", "base_url"),
        ("TRANSLATE_API_KEY", "api_key"),
        ("TRANSLATE_MODEL", "model"),
    ):
        value = environment.get(variable, "").strip()
        if value:
            settings[key] = value

    raw = environment.get("TRANSLATE_TIMEOUT", "").strip()
    if raw:
        try:
            timeout = float(raw)
        except ValueError:
            raise ConfigError(
                "TRANSLATE_TIMEOUT must be a number, got %r" % raw
            ) from None

        if timeout <= 0:
            raise ConfigError("TRANSLATE_TIMEOUT must be positive")

        settings["timeout"] = timeout

    raw = environment.get("TRANSLATE_MAX_TOKENS", "").strip()
    if raw:
        try:
            max_tokens = int(raw)
        except ValueError:
            raise ConfigError(
                "TRANSLATE_MAX_TOKENS must be an integer, got %r" % raw
            ) from None

        if max_tokens <= 0:
            raise ConfigError("TRANSLATE_MAX_TOKENS must be positive")

        settings["max_tokens"] = max_tokens

    return settings


def api_settings_from_arguments(arguments):
    settings = {}
    if arguments.base_url:
        settings["base_url"] = arguments.base_url

    if arguments.api_key:
        settings["api_key"] = arguments.api_key

    if arguments.model:
        settings["model"] = arguments.model

    if arguments.timeout is not None:
        settings["timeout"] = arguments.timeout

    if arguments.max_tokens is not None:
        settings["max_tokens"] = arguments.max_tokens

    return settings


def build_config(settings):
    config = Config(
        base_url=settings["base_url"].rstrip("/"),
        api_key=settings["api_key"],
        model=settings["model"],
        timeout=settings["timeout"],
        max_tokens=settings["max_tokens"],
        params=dict(settings["params"]),
    )
    missing = []
    if not config.base_url:
        missing.append("api.base_url (--base-url / TRANSLATE_BASE_URL)")

    if not config.api_key:
        missing.append("api.api_key (--api-key / TRANSLATE_API_KEY)")

    if not config.model:
        missing.append("api.model (--model / TRANSLATE_MODEL)")

    if missing:
        raise ConfigError("missing required API settings: " + ", ".join(missing))

    return config


def parse_options_table(path, table):
    if not isinstance(table, dict):
        raise PassError("%s: [options] must be a table" % path)

    unknown = sorted(set(table) - {"ascii"})
    if unknown:
        raise PassError(
            "%s: [options]: unknown key(s): %s" % (path, ", ".join(unknown))
        )

    ascii_value = table.get("ascii", False)
    if not isinstance(ascii_value, bool):
        raise PassError("%s: [options] ascii must be true or false" % path)

    return {"ascii": ascii_value}


def load_instruction_text(path, name, table, base_directory):
    has_file = "instruction_file" in table
    has_inline = "instruction" in table
    if has_file == has_inline:
        raise PassError(
            "%s: [[pass]] %s: exactly one of instruction_file or instruction "
            "is required" % (path, name)
        )

    if has_inline:
        instruction = table["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise PassError("%s: [[pass]] %s: instruction is empty" % (path, name))

        return instruction.strip()

    instruction_file = table["instruction_file"]
    if not isinstance(instruction_file, str) or not instruction_file.strip():
        raise PassError(
            "%s: [[pass]] %s: instruction_file must be a path" % (path, name)
        )

    instruction_path = os.path.join(base_directory, instruction_file)
    try:
        with open(instruction_path, encoding="utf-8") as instruction_handle:
            instruction = instruction_handle.read().strip()
    except OSError as error:
        raise PassError(
            "%s: [[pass]] %s: cannot read instruction file: %s" % (path, name, error)
        ) from None

    if not instruction:
        raise PassError("%s: [[pass]] %s: instruction file is empty" % (path, name))

    return instruction


def parse_pass_table(path, table, base_directory):
    if not isinstance(table, dict):
        raise PassError("%s: [[pass]] entries must be tables" % path)

    unknown = sorted(set(table) - set(PASS_KEYS))
    if unknown:
        raise PassError("%s: [[pass]]: unknown key(s): %s" % (path, ", ".join(unknown)))

    name = table.get("name")
    if not isinstance(name, str) or not name.strip():
        raise PassError("%s: [[pass]]: name must be a non-empty string" % path)

    name = name.strip()
    instruction = load_instruction_text(path, name, table, base_directory)
    mode = table.get("mode", "chunk")
    if mode not in ("analysis", "chunk", "paragraph"):
        raise PassError(
            "%s: [[pass]] %s: mode must be analysis, chunk, or paragraph" % (path, name)
        )

    strict_fidelity = table.get("strict_fidelity", False)
    if not isinstance(strict_fidelity, bool):
        raise PassError(
            "%s: [[pass]] %s: strict_fidelity must be true or false" % (path, name)
        )

    ascii_value = table.get("ascii")
    if ascii_value is not None and not isinstance(ascii_value, bool):
        raise PassError("%s: [[pass]] %s: ascii must be true or false" % (path, name))

    model = table.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise PassError(
            "%s: [[pass]] %s: model must be a non-empty string" % (path, name)
        )

    params = table.get("params", {})
    if not isinstance(params, dict):
        raise PassError("%s: [[pass]] %s: params must be a table" % (path, name))

    return PassDefinition(
        name=name,
        instruction=instruction,
        mode=mode,
        strict_fidelity=strict_fidelity,
        params=params,
        model=model.strip() if model else None,
        ascii=ascii_value,
    )


def document_passes(path, document):
    entries = document.get("pass")
    if entries is None:
        return None

    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        raise PassError("%s: [[pass]] must define one or more pass tables" % path)

    base_directory = os.path.dirname(os.path.abspath(path))
    return [parse_pass_table(path, entry, base_directory) for entry in entries]


def resolve_passes(user_path, user_document, selected_path, selected_document):
    sources = []
    if selected_document:
        sources.append((selected_path, selected_document))

    if user_document:
        sources.append((user_path, user_document))

    for path, document in sources:
        definitions = document_passes(path, document)
        if definitions is None:
            continue

        options = {}
        for option_path, option_document in sources:
            if "options" in option_document:
                options = parse_options_table(option_path, option_document["options"])
                break

        return definitions, options

    raise PassError(
        "no [[pass]] tables found; define at least one pass in %s"
        % (selected_path or user_path or "a config file (see --help)")
    )


def resolve_config_path(requested, environment):
    if requested:
        return requested

    configured = environment.get("TRANSLATE_CONFIG", "").strip()
    if configured:
        return configured

    local = os.path.join(os.getcwd(), "zh2en.toml")
    if os.path.exists(local):
        return local

    fallback = user_config_path(environment)
    if os.path.exists(fallback):
        return fallback

    return None


def load_setup(arguments, environment):
    user_path = user_config_path(environment)
    user_document = {}
    if os.path.exists(user_path):
        user_document = load_toml(user_path, "user config")
        validate_document(user_path, user_document)

    selected_path = resolve_config_path(arguments.config, environment)
    selected_document = {}
    if selected_path:
        selected_document = load_toml(selected_path, "config file")
        validate_document(selected_path, selected_document)

    settings = default_api_settings()
    settings = merge_api_settings(
        settings, document_api_settings(user_path, user_document)
    )
    settings = merge_api_settings(
        settings, document_api_settings(selected_path, selected_document)
    )
    settings = merge_api_settings(settings, api_settings_from_environment(environment))
    settings = merge_api_settings(settings, api_settings_from_arguments(arguments))
    config = build_config(settings)
    pass_definitions, options = resolve_passes(
        user_path, user_document, selected_path, selected_document
    )
    return config, apply_default_ascii(pass_definitions, options)


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
    StatusLine.for_stream(stderr).interrupt()
    print(message, file=stderr, flush=True)


class StatusLine:
    _instances = {}

    def __init__(self, stream):
        self.stream = stream
        try:
            self.live = stream.isatty()
        except (AttributeError, OSError, ValueError):
            self.live = False
        self.prefix = ""
        self.drawn = ""
        self.label = "Working"
        self.started = 0.0
        self.tokens = 0
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    @classmethod
    def for_stream(cls, stream):
        key = id(stream)
        instance = cls._instances.get(key)
        if instance is None:
            instance = cls(stream)
            cls._instances[key] = instance

        return instance

    def write_partial(self, text):
        with self._lock:
            self.prefix += text
            self.stream.write(text)
            self.stream.flush()

    def start(self, label):
        if not self.live:
            return

        with self._lock:
            self.label = label
            self.started = time.monotonic()
            self.tokens = 0
            self._render()

        self._stop.clear()
        self._thread = threading.Thread(target=self._tick, daemon=True)
        self._thread.start()

    def progress(self, label, count=1):
        if not self.live:
            return

        with self._lock:
            self.label = label
            self.tokens += count

    def stop(self):
        self._halt()
        with self._lock:
            if self._erase() and self.prefix:
                self.stream.write(self.prefix)
                self.drawn = self.prefix
            self.stream.flush()

    def interrupt(self):
        self._halt()
        with self._lock:
            had_drawn = self._erase()
            if self.prefix:
                if not (self.live and had_drawn):
                    self.stream.write("\n")

                self.prefix = ""

            self.stream.flush()

    def finish(self, text):
        self._halt()
        with self._lock:
            rewrote = self._erase()
            prefix = self.prefix if rewrote else ""
            self.stream.write(prefix + text + "\n")
            self.prefix = ""
            self.drawn = ""
            self.stream.flush()

    def _halt(self):
        if self._thread is not None and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=1.0)

    def _tick(self):
        while not self._stop.wait(0.1):
            with self._lock:
                self._render()

    def _render(self):
        line = "%s: time elapsed: %.2fs, tokens received: %d" % (
            self.label,
            time.monotonic() - self.started,
            self.tokens,
        )
        full = self.prefix + line if self.prefix else line
        padding = max(len(self.drawn) - len(full), 0)
        self.stream.write("\r" + full + " " * padding)
        self.stream.flush()
        self.drawn = full

    def _erase(self):
        if not self.drawn:
            return False

        self.stream.write("\r" + " " * len(self.drawn) + "\r")
        self.drawn = ""
        return True


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
    lone_newlines = text.count("\n") - sum(blank.count("\n") for blank in blanks)
    separator_pattern = r"(\n+)" if lone_newlines >= 3 * len(blanks) else r"(\n\s*\n)"
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


def resolve_call_settings(config, pass_definition):
    params = dict(config.params)
    params.update(pass_definition.params)
    model = pass_definition.model or config.model
    return model, params


def build_chat_payload(model, system, user, params):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    payload.update({key: value for key, value in params.items() if value is not None})
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


def extract_message(body):
    try:
        message = body["choices"][0]["message"]
        return message, message["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape: %s" % str(body)[:500]) from None


def collect_thinking_texts(part):
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
            thoughts.extend(collect_thinking_texts(part))
            continue

        if part.get("text"):
            texts.append(part["text"])

    return texts, thoughts


def flatten_content_parts(content):
    if not isinstance(content, list):
        return content, []

    texts, thoughts = collect_content_parts(content)
    return "".join(texts), thoughts


def message_reasoning_texts(message):
    for key in ("reasoning_content", "reasoning"):
        reasoning = message.get(key)
        if isinstance(reasoning, str) and reasoning.strip():
            return [reasoning.rstrip()]

    return []


def strip_think_tag(content):
    if not isinstance(content, str):
        return content, None

    match = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
    if not match:
        return content, None

    return content[match.end() :], match.group(1).strip()


def parse_stream_line(raw_line):
    line = raw_line.decode("utf-8", "replace").strip()
    if not line.startswith("data:"):
        return None

    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None

    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def stream_chat_chunks(config, payload):
    def open_stream(request_payload):
        request = urllib.request.Request(
            config.base_url + "/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": "Bearer " + config.api_key,
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=config.timeout)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise LLMError("HTTP %s from endpoint: %s" % (error.code, detail)) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMError("could not reach endpoint: %s" % error) from None

    try:
        response = open_stream(payload)
    except LLMError as error:
        if "stream_options" not in payload or "stream_options" not in str(error):
            raise
        # Endpoint rejected stream_options; retry without it.
        payload = {
            key: value for key, value in payload.items() if key != "stream_options"
        }
        response = open_stream(payload)

    with response:
        try:
            for raw_line in response:
                chunk = parse_stream_line(raw_line)
                if not isinstance(chunk, dict):
                    continue

                if isinstance(chunk.get("error"), dict):
                    raise LLMError(
                        "endpoint stream error: %s" % str(chunk["error"])[:500]
                    )

                yield chunk
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise LLMError("stream interrupted: %s" % error) from None


def chunk_delta(chunk):
    try:
        choices = chunk.get("choices")
        if not choices:
            return {}

        return choices[0].get("delta") or {}
    except (AttributeError, IndexError, TypeError):
        return {}


def delta_reasoning_text(delta):
    for key in ("reasoning_content", "reasoning"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            return value

    return None


def delta_text(delta):
    value = delta.get("content")
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))

    return ""


class ThinkTagTracker:
    def __init__(self):
        self.checking = True
        self.open = False
        self.tail = ""

    def feed(self, text):
        if not self.checking:
            return False

        searched = self.tail + text
        self.tail = searched[-8:]
        if self.open:
            if "</think>" in searched:
                self.checking = False
                self.open = False

            return self.open

        stripped = searched.lstrip()
        if stripped.startswith("<think>"):
            self.open = True
            if "</think>" in stripped[7:]:
                self.checking = False
                self.open = False

            return self.open

        if not stripped.startswith("<") or len(stripped) > 64:
            self.checking = False

        return False


def stream_chat(config, payload, status):
    payload = dict(payload)
    payload["stream"] = True
    if "stream_options" not in payload:
        payload["stream_options"] = {"include_usage": True}

    reasoning = []
    content_texts = []
    reported = None
    counted = 0
    think_tracker = ThinkTagTracker()
    content_started = False
    for chunk in stream_chat_chunks(config, payload):
        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            reported = chunk_usage

        delta = chunk_delta(chunk)
        reasoning_text = delta_reasoning_text(delta)
        text = delta_text(delta)
        if not reasoning_text and not text:
            continue

        if reasoning_text:
            reasoning.append(reasoning_text)
            counted += 1
            if not content_started:
                status.progress("Thinking")

        if text:
            content_started = True
            content_texts.append(text)
            counted += 1
            status.progress("Thinking" if think_tracker.feed(text) else "Working")

    return "".join(content_texts), reasoning, reported, counted


def chat(config, system, user, model, params, verbose, stderr, usage):
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > config.max_tokens:
        raise LLMError(
            "request is ~%d tokens, over the %d-token budget; raise "
            "api.max_tokens (--max-tokens / TRANSLATE_MAX_TOKENS) or "
            "shorten the input" % (estimated, config.max_tokens)
        )

    payload = build_chat_payload(model, system, user, params)
    status = StatusLine.for_stream(stderr)
    try:
        status.start("Working")
        if payload.get("stream", True):
            content, reasoning, reported, counted = stream_chat(config, payload, status)
        else:
            body = http_chat(config, payload)
            message, content = extract_message(body)
            reasoning = message_reasoning_texts(message)
            content, thoughts = flatten_content_parts(content)
            reasoning = reasoning + thoughts
            reported = body.get("usage") or {}
            counted = 0
    finally:
        status.stop()

    content, think_text = strip_think_tag(content)
    if think_text:
        reasoning.append(think_text)

    if verbose:
        for text in reasoning:
            if text.strip():
                log(stderr, text.rstrip())

    if reported:
        usage = add_usage(usage, reported)
    else:
        usage = add_usage(
            usage, {"prompt_tokens": estimated, "completion_tokens": counted}
        )

    if not isinstance(content, str):
        raise LLMError("unexpected content type: %s" % type(content).__name__)

    return content, usage


def run_analysis(config, settings, pass_definition, text, verbose, stderr, usage):
    user = settings.analysis_user_prefix + text
    model, params = resolve_call_settings(config, pass_definition)
    content, usage = chat(
        config,
        pass_definition.instruction,
        user,
        model,
        params,
        verbose,
        stderr,
        usage,
    )
    return content.strip(), usage


def run_merge(config, settings, pass_definition, briefs, verbose, stderr, usage):
    user = settings.merge_user_prefix + "\n\n".join(briefs)
    model, params = resolve_call_settings(config, pass_definition)
    content, usage = chat(
        config,
        settings.analysis_merge_instruction,
        user,
        model,
        params,
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
                "%d-token budget; raise api.max_tokens (--max-tokens / "
                "TRANSLATE_MAX_TOKENS) or shorten the input"
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
    model, params = resolve_call_settings(config, pass_definition)
    key = cache_key(full_text, model, salt, overrides=params)
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


def translate_chunk(
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
    model, params = resolve_call_settings(config, pass_definition)
    content, usage = chat(config, system, user, model, params, verbose, stderr, usage)
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
            config.model,
            config.params,
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
            + json.dumps(
                overrides, sort_keys=True, ensure_ascii=False, default=str
            ).encode("utf-8")
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
    model, params = resolve_call_settings(config, pass_definition)
    for index, work_group in enumerate(work_groups):
        source_chunk_text = "\n\n".join(plan[index]) if index < len(plan) else ""
        work_chunk_text = "\n\n".join(work_group)
        key = cache_key(
            source_chunk_text,
            model,
            pass_salt,
            work_chunk_text,
            overrides=params,
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
            result, total_usage = translate_chunk(
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
    status = StatusLine.for_stream(stderr)
    status.write_partial("Enforcing ASCII... ")
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
        status.interrupt()
        return (
            text,
            usage,
            "zh2en: pass [%s] ascii enforcement failed: %s"
            % (pass_definition.name, error),
        )

    prompt, completion, cost = usage_delta(usage, updated_usage)
    if prompt or completion or cost:
        status.finish(
            usage_line("Done", clock() - started_at, prompt, completion, cost)
        )
    else:
        status.finish("Done.")

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


def run_pass(
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
        result = run_pass(
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
    return parser.parse_args(arguments)


def main(arguments, environment, stdin, stdout, stderr, clock):
    parsed_arguments = parse_args(arguments)
    text = stdin.read()
    if not text.strip():
        return 0

    started = clock()
    try:
        config, pass_definitions = load_setup(parsed_arguments, environment)
    except (ConfigError, PassError) as error:
        log(stderr, "zh2en: %s" % error)
        return 2

    settings = build_settings()
    cache_directory = resolve_cache_dir(environment)
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
        cache_directory,
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

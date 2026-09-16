#!/usr/bin/env python3
"""zh2en — multi-pass Chinese-to-English translator.

Reads Chinese text from stdin, writes English translation to stdout.
Diagnostics go to stderr so the tool stays pipe-friendly.

Pipeline:
    stdin -> split into paragraphs
         -> passes 1..N from the passes INI file:
              mode = analysis    runs once over the whole document; its output
                                 is attached to every later call as context
                                 and never enters the translation chain
              mode = paragraph   one source paragraph per call
              mode = chunk       default; budget-packed chunks. Every
                                 translation call receives its source unit
                                 plus the previous pass's output for that unit
              after each pass whose `ascii` setting is true (the [options]
              `ascii` key is the default): each output paragraph is checked;
              non-ASCII ones get a mechanical conversion (punctuation,
              full-width forms, accents), and anything still non-ASCII is
              repaired by an LLM call that sees the source paragraph and
              must emit ASCII-only English
         -> stdout
"""

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

CHUNK_BUDGET = 3500  # max source characters per chunk

STRICT_FIDELITY_SUFFIX = (
    "\n- STRICT FIDELITY MODE: keep the exact number of paragraphs "
    "and the exact separators of the source. Do not merge "
    "or split paragraphs."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class Config:
    def __init__(self):
        self.base_url = os.environ.get("TRANSLATE_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("TRANSLATE_API_KEY", "")
        self.model = os.environ.get("TRANSLATE_MODEL", "gpt-4o-mini")
        self.timeout = float(os.environ.get("TRANSLATE_TIMEOUT", "120"))

    def validate(self):
        missing = []
        if not self.base_url:
            missing.append("TRANSLATE_BASE_URL")
        if not self.api_key:
            missing.append("TRANSLATE_API_KEY")
        if missing:
            raise ConfigError(
                "missing required environment variables: " + ", ".join(missing)
            )


class ConfigError(Exception):
    pass


class LLMError(Exception):
    pass


# ---------------------------------------------------------------------------
# Pass definitions (INI file)
# ---------------------------------------------------------------------------


class PassError(Exception):
    pass


class Pass:
    """One translation pass loaded from the passes INI file.

    `mode` selects the unit of work: "analysis" (whole document, output kept
    as context), "chunk" (default, budget-packed) or "paragraph" (one source
    paragraph per call). `ascii` enforces pure-ASCII output for this pass
    after it completes.

    Keys other than the reserved ones pass through into the chat-completions
    request payload as-is (with type inference), so API fields like
    `temperature`, `model`, `reasoning_effort`, `top_p` need no code here.
    """

    def __init__(
        self,
        name,
        instruction,
        mode="chunk",
        strict_fidelity=False,
        api_overrides=None,
        ascii=False,
    ):
        self.name = name
        self.instruction = instruction
        self.mode = mode
        self.strict_fidelity = strict_fidelity
        self.api_overrides = api_overrides or {}
        self.ascii = ascii


def _infer_type(value):
    v = value.strip()
    if v.lower() == "none":
        return None  # omit the field from the payload
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


OPTIONS_SECTION = "options"


def load_passes(path):
    """Load pass definitions from an INI file, in section order.

    Reserved keys: `instruction-file` (required; system-prompt text, resolved
    relative to the INI file), `mode` (`analysis`, `chunk` or `paragraph`;
    default `chunk`), `strict_fidelity` (bool) and `ascii` (bool; enforce
    pure-ASCII output for this pass). All other keys become chat-completions
    payload overrides.

    A special `[options]` section holds global script options instead of a
    pass; see load_options. Returns (passes, options).
    """
    parser = configparser.ConfigParser()
    try:
        with open(path, encoding="utf-8") as f:
            parser.read_file(f)
    except OSError as e:
        raise PassError("cannot read passes file: %s" % e) from None
    except configparser.Error as e:
        raise PassError("cannot parse passes file: %s" % e) from None

    if not parser.sections():
        raise PassError("passes file %s defines no passes" % path)

    base_dir = os.path.dirname(os.path.abspath(path))
    passes = []
    options = {}
    for section in parser.sections():
        opts = dict(parser.items(section))
        if section.lower() == OPTIONS_SECTION:
            options.update(load_options(section, opts))
            continue
        instruction_file = opts.pop("instruction-file", None)
        if not instruction_file:
            raise PassError(
                "pass [%s]: missing required key `instruction-file`" % section
            )
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

        strict_raw = opts.pop("strict_fidelity", "false")
        if strict_raw.strip().lower() not in ("true", "false"):
            raise PassError(
                "pass [%s]: strict_fidelity must be true or false" % section
            )
        strict_fidelity = strict_raw.strip().lower() == "true"

        ascii_raw = opts.pop("ascii", None)
        ascii_override = None
        if ascii_raw is not None:
            if ascii_raw.strip().lower() not in ("true", "false"):
                raise PassError("pass [%s]: ascii must be true or false" % section)
            ascii_override = ascii_raw.strip().lower() == "true"

        mode_raw = opts.pop("mode", "chunk").strip().lower()
        if mode_raw not in ("analysis", "chunk", "paragraph"):
            raise PassError(
                "pass [%s]: mode must be analysis, chunk, or paragraph" % section
            )

        api_overrides = {}
        for key, value in opts.items():
            parsed = _infer_type(value)
            if parsed is not None:
                api_overrides[key] = parsed
        passes.append(
            Pass(
                section,
                instruction,
                mode_raw,
                strict_fidelity,
                api_overrides,
                ascii=ascii_override,
            )
        )
    # Resolve the per-pass ascii default after the loop so it works no matter
    # where the [options] section sits in the file.
    default_ascii = options.get("ascii", False)
    for passdef in passes:
        if passdef.ascii is None:
            passdef.ascii = default_ascii
    return passes, options


def load_options(section, opts):
    """Parse the global `[options]` section. Currently supported:
    `ascii` (bool) — default for per-pass ASCII enforcement; any pass may
    override it with its own `ascii` key."""
    options = {}
    ascii_raw = opts.pop("ascii", "false").strip().lower()
    if ascii_raw not in ("true", "false"):
        raise PassError("[%s]: ascii must be true or false" % section)
    options["ascii"] = ascii_raw == "true"
    if opts:
        raise PassError(
            "[%s]: unknown option(s): %s" % (section, ", ".join(sorted(opts)))
        )
    return options


# ---------------------------------------------------------------------------
# Logging / usage stats (stderr only, so stdout stays pipe-friendly)
# ---------------------------------------------------------------------------


class UsageTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.reset_pass()

    def reset_pass(self):
        self.pass_started = time.time()
        self.pass_prompt = 0
        self.pass_completion = 0

    def add(self, usage):
        p = usage.get("prompt_tokens") or 0
        c = usage.get("completion_tokens") or 0
        self.prompt_tokens += p
        self.completion_tokens += c
        self.pass_prompt += p
        self.pass_completion += c

    def elapsed(self, since=None):
        return time.time() - (since if since is not None else self.pass_started)

    def log_pass(self):
        elapsed = self.elapsed()
        self.eprint(
            "Done: %s, prompt=%d, completion=%d, %.1f tok/s"
            % (
                fmt_duration(elapsed),
                self.pass_prompt,
                self.pass_completion,
                self.pass_completion / elapsed if elapsed else 0.0,
            )
        )
        self.reset_pass()

    def log_total(self, total_elapsed):
        self.eprint(
            "TOTAL: %s, prompt=%d, completion=%d, %.1f tok/s"
            % (
                fmt_duration(total_elapsed),
                self.prompt_tokens,
                self.completion_tokens,
                self.completion_tokens / total_elapsed if total_elapsed else 0.0,
            )
        )

    @staticmethod
    def eprint(msg):
        print(msg, file=sys.stderr, flush=True)


USAGE = UsageTracker()


def fmt_duration(seconds):
    if seconds < 60:
        return "%.1fs" % seconds
    return "%dm%ds" % (int(seconds // 60), int(seconds % 60))


# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible chat completions, stdlib only)
# ---------------------------------------------------------------------------


def chat(config, system, user, overrides=None, verbose=False):
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if overrides:
        payload.update({k: v for k, v in overrides.items() if v is not None})
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
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise LLMError("HTTP %s from endpoint: %s" % (e.code, detail)) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError("could not reach endpoint: %s" % e) from None
    USAGE.add(body.get("usage") or {})
    try:
        message = body["choices"][0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape: %s" % str(body)[:500]) from None
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if verbose and isinstance(reasoning, str) and reasoning.strip():
        print(reasoning.rstrip(), file=sys.stderr, flush=True)
    if isinstance(content, list):
        # Mistral reasoning models return content as a list of chunks:
        # {type: "thinking", thinking: [{type: "text", text}...]} for the
        # reasoning trace, {type: "text", text} for the answer itself.
        texts = []
        thoughts = []
        for part in content:
            if not isinstance(part, dict):
                texts.append(str(part))
            elif part.get("type") == "thinking":
                for sub in part.get("thinking") or []:
                    if isinstance(sub, dict) and sub.get("text"):
                        thoughts.append(sub["text"])
            elif part.get("text"):
                texts.append(part["text"])
        if verbose and thoughts:
            print("\n".join(thoughts).rstrip(), file=sys.stderr, flush=True)
        content = "".join(texts)
    if isinstance(content, str):
        # some open-weight deployments wrap the trace in <think> tags instead
        m = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
        if m:
            if verbose:
                print(m.group(1).strip(), file=sys.stderr, flush=True)
            content = content[m.end() :]
    if not isinstance(content, str):
        raise LLMError("unexpected content type: %s" % str(body)[:500])
    return content


# ---------------------------------------------------------------------------
# Chunking: split into paragraphs, pack them into budgeted chunks.
# ---------------------------------------------------------------------------

BLANK_LINE_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text):
    """Return (paragraphs, separators) preserving separator structure.

    Paragraphs are separated by blank lines, except that web-novel text
    usually puts one paragraph per line with no blank lines at all. When
    lone newlines (not part of a blank-line separator) outnumber the
    blank-line separators three to one, the text is treated as line-per-
    paragraph (with at most stray blank lines) and newline runs become the
    separators instead.
    """
    blanks = BLANK_LINE_RE.findall(text)
    lone = text.count("\n") - sum(s.count("\n") for s in blanks)
    sep_re = r"(\n+)" if lone >= 3 * len(blanks) else r"(\n\s*\n)"
    parts = re.split(sep_re, text)
    is_sep = re.compile(sep_re).fullmatch
    paragraphs = [p for p in parts if p and not is_sep(p)]
    separators = [s for s in parts if s and is_sep(s)]
    return paragraphs, separators


def make_chunks(paragraphs, budget=CHUNK_BUDGET):
    """Group paragraphs into chunks under `budget` source characters.
    A single paragraph longer than the budget becomes its own (oversized) chunk
    rather than being split mid-paragraph."""
    chunks = []
    current = []
    size = 0
    for p in paragraphs:
        if current and size + len(p) > budget:
            chunks.append(current)
            current = []
            size = 0
        current.append(p)
        size += len(p)
    if current:
        chunks.append(current)
    return chunks


def iter_units(plan, separators):
    """Yield (unit_paragraphs, separator_after_unit) for a unit plan (a list
    of paragraph lists) against the original text's paragraph separators.
    The final unit always gets no separator."""
    consumed = 0
    for i, unit in enumerate(plan):
        consumed += len(unit)
        sep = separators[consumed - 1] if consumed - 1 < len(separators) else ""
        yield unit, (sep if i < len(plan) - 1 else "")


def regroup_by_plan(paragraphs, plan):
    """Group `paragraphs` into lists sized like `plan` (a list of paragraph
    lists). Returns None when the paragraph count doesn't match the plan."""
    total = sum(len(c) for c in plan)
    if len(paragraphs) != total:
        return None
    groups = []
    i = 0
    for chunk in plan:
        groups.append(paragraphs[i : i + len(chunk)])
        i += len(chunk)
    return groups


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def parse_glossary_file(path):
    """Lines of `term -> translation` (also accepts `term = translation` or
    a single tab/comma). Blank lines and # comments ignored."""
    entries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.split(r"\s*(?:->|=>|=|\t|,\s)\s*", line, maxsplit=1)
            if len(m) == 2 and m[0] and m[1]:
                entries[m[0].strip()] = m[1].strip()
    return entries


def merge_glossaries(*dicts):
    """Later dicts win (user glossary is passed last)."""
    merged = {}
    for d in dicts:
        merged.update(d)
    return merged


def glossary_text(glossary):
    if not glossary:
        return "(none)"
    return "\n".join(
        "%s -> %s" % (term, trans) for term, trans in sorted(glossary.items())
    )


# ---------------------------------------------------------------------------
# Pass execution
# ---------------------------------------------------------------------------


def analysis_block(analysis):
    return analysis.strip() if analysis and analysis.strip() else "(none)"


def run_analysis(config, passdef, full_text, glossary, verbose=False):
    """Run an analysis pass once over the whole document."""
    user = (
        "Glossary (user-supplied; respect these renderings):\n%s\n\n"
        "Source text (full document, original Chinese):\n%s"
        % (glossary_text(glossary), full_text)
    )
    return chat(
        config,
        passdef.instruction,
        user,
        overrides=passdef.api_overrides,
        verbose=verbose,
    ).strip()


def run_analysis_once(config, passdef, full_text, glossary, use_cache, verbose):
    salt = passdef.name + "\x00" + passdef.instruction
    key = cache_key(full_text, config.model, glossary, salt)
    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            if verbose:
                eprint("zh2en: [%s] cache hit" % passdef.name)
            return cached
    result = run_analysis(config, passdef, full_text, glossary, verbose)
    if use_cache:
        cache_put(key, result)
    return result


def run_pass(
    config, passdef, source_chunk, work_chunk, glossary, analysis=None, verbose=False
):
    system = passdef.instruction
    if passdef.strict_fidelity:
        system += STRICT_FIDELITY_SUFFIX
    user = (
        "Glossary (use these renderings exactly):\n%s\n\n"
        "Preparation brief from a full read of the text "
        "(outline, names, hard-to-translate items):\n%s\n\n"
        "Source text (original Chinese):\n%s"
        % (glossary_text(glossary), analysis_block(analysis), source_chunk)
    )
    if work_chunk is not None and work_chunk != source_chunk:
        user += "\n\nCurrent draft from the previous pass:\n%s" % work_chunk
    return clean_translation(
        chat(config, system, user, overrides=passdef.api_overrides, verbose=verbose)
    )


def clean_translation(text):
    """Strip common LLM output wrapper artifacts."""
    t = text.strip()
    # strip markdown fences
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)
    # strip symmetric wrapping quotes
    if len(t) >= 2:
        pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        if t[0] in pairs and t[-1] == pairs[t[0]]:
            t = t[1:-1]
    # drop any leading "Translation:" label
    t = re.sub(r"^Translation:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


# ---------------------------------------------------------------------------
# ASCII enforcement (per-pass `ascii` setting; [options] ascii is the default)
# ---------------------------------------------------------------------------


ASCII_FIX_ATTEMPTS = 3

ASCII_FIX_INSTRUCTION = """
This paragraph failed to be fully translated or contains non-ASCII
characters. Please analyse it and only output a clean translation
without any non-ASCII.
"""

# Common typographic characters with no ASCII-compatible decomposition.
# Full-width forms, no-break spaces and combining accents are handled by
# NFKD normalization in to_ascii_mechanical().
ASCII_CHAR_MAP = {
    # spaces
    "\u00a0": " ",
    "\u2007": " ",
    "\u2009": " ",
    "\u202f": " ",
    # quotes
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
    # dashes
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    # ellipsis / dots
    "\u2026": "...",
    "\u2025": "..",
    # bullets / middle dot
    "\u2022": "*",
    "\u2023": "*",
    "\u2024": "*",
    "\u2219": "*",
    "\u00b7": " ",
    # arrows / comparisons
    "\u2190": "<-",
    "\u2192": "->",
    "\u2194": "<->",
    "\u2264": "<=",
    "\u2265": ">=",
    "\u2260": "!=",
    # CJK punctuation with no compatibility decomposition
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


def to_ascii_mechanical(text):
    """Best-effort ASCII conversion for common cases: a replacement map for
    typographic punctuation, then NFKD normalization with combining marks
    stripped (full-width forms, no-break spaces, accents). Anything left
    (Chinese characters, letters without ASCII decompositions, symbols)
    stays as-is; the caller treats that as the signal to ask the LLM."""
    t = text
    for src, repl in ASCII_CHAR_MAP.items():
        t = t.replace(src, repl)
    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def ascii_fix_llm(config, pass_name, source_para, out_para, use_cache, verbose):
    """Ask the LLM to rewrite a still-non-ASCII translated paragraph as
    pure ASCII English, given the source paragraph for context. Models
    sometimes parrot the paragraph back with the Chinese left in, so
    non-ASCII replies are retried with an escalating complaint, and only
    compliant replies are cached."""
    salt = "ascii-fix\x00" + pass_name + "\x00" + ASCII_FIX_INSTRUCTION
    key = cache_key(source_para, config.model, {}, salt, out_para)
    if use_cache:
        cached = cache_get(key)
        # a cached reply that is itself non-ASCII (from an older run or a
        # non-compliant model) is ignored so it can be regenerated
        if cached is not None and cached.isascii():
            if verbose:
                eprint("zh2en: ascii: cache hit")
            return cached
    user = (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Rewrite the translated paragraph as pure ASCII English."
        % (source_para, out_para)
    )
    result = ""
    for attempt in range(1, ASCII_FIX_ATTEMPTS + 1):
        result = clean_translation(
            chat(config, ASCII_FIX_INSTRUCTION, user, verbose=verbose)
        )
        if result.isascii():
            if use_cache:
                cache_put(key, result)
            return result
        if verbose:
            eprint(
                "zh2en: ascii: attempt %d/%d still non-ASCII; retrying"
                % (attempt, ASCII_FIX_ATTEMPTS)
            )
        user = (
            "Source paragraph (original language):\n%s\n\n"
            "Translated paragraph (must become pure ASCII English):\n%s\n\n"
            "Your previous reply still contained these non-ASCII "
            "characters: %s. Rewrite the translated paragraph again, "
            "inferring English for every one of them from the source and "
            "context. Reply with ASCII characters only."
            % (source_para, out_para, non_ascii_sample(result))
        )
    return result


def non_ascii_sample(text, limit=12):
    """Up to `limit` distinct non-ASCII characters in `text`, for use in
    diagnostics and retry prompts."""
    seen = []
    for c in text:
        if not c.isascii() and c not in seen:
            seen.append(c)
            if len(seen) >= limit:
                break
    return "".join(seen)


def ensure_ascii_output(config, pass_name, text, source_paragraphs, use_cache, verbose):
    """Enforce pure-ASCII output paragraph by paragraph. Non-ASCII
    paragraphs first get the mechanical conversion; whatever is still
    non-ASCII is repaired by the LLM using the matching source paragraph.
    If even the retried LLM replies keep non-ASCII characters, the strays
    are dropped as a last resort so the output stays ASCII."""
    paragraphs, separators = split_paragraphs(text)
    outputs = []
    for i, para in enumerate(paragraphs):
        sep = separators[i] if i < len(separators) else ""
        if para.isascii():
            outputs.append(para + sep)
            continue
        mechanical = to_ascii_mechanical(para)
        if mechanical.isascii():
            if verbose:
                eprint(
                    "zh2en: ascii: paragraph %d/%d converted mechanically"
                    % (i + 1, len(paragraphs))
                )
            outputs.append(mechanical + sep)
            continue
        if verbose:
            eprint(
                "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
                "LLM to repair it" % (i + 1, len(paragraphs))
            )
        source = source_paragraphs[i] if i < len(source_paragraphs) else "(unavailable)"
        repaired = ascii_fix_llm(config, pass_name, source, para, use_cache, verbose)
        if not repaired.isascii():
            fallback = to_ascii_mechanical(repaired)
            if fallback.isascii():
                repaired = fallback
        if not repaired.isascii():
            eprint(
                "zh2en: ascii: warning: paragraph %d still contained "
                "non-ASCII characters (%s) after %d LLM attempts; dropping "
                "them" % (i + 1, non_ascii_sample(repaired), ASCII_FIX_ATTEMPTS)
            )
            repaired = "".join(c for c in repaired if c.isascii())
            repaired = re.sub(r"  +", " ", repaired)
        outputs.append(repaired + sep)
    return "".join(outputs)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "zh2en")
    os.makedirs(d, exist_ok=True)
    return d


def cache_key(chunk_text, model, glossary, pass_salt="", work_text=""):
    h = hashlib.sha256()
    if pass_salt:
        h.update(pass_salt.encode("utf-8") + b"\x00")
    h.update(chunk_text.encode("utf-8"))
    if work_text:
        h.update(b"\x00work\x00" + work_text.encode("utf-8"))
    h.update(b"\x00" + model.encode("utf-8"))
    h.update(
        b"\x00"
        + json.dumps(glossary, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return h.hexdigest()


def cache_get(key):
    path = os.path.join(cache_dir(), key + ".txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def cache_put(key, value):
    path = os.path.join(cache_dir(), key + ".txt")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(value)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------


def parse_json_loose(raw):
    """Best-effort JSON extraction from an LLM reply."""
    t = raw.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def eprint(*args):
    print(*args, file=sys.stderr)


def main(argv=None):
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
        "--glossary",
        metavar="FILE",
        help="glossary file of `term -> rendering` lines, " "applied to every pass",
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
    args = ap.parse_args(argv)

    text = sys.stdin.read()
    if not text.strip():
        return 0

    try:
        config = Config()
        config.validate()
    except ConfigError as e:
        eprint("zh2en: %s" % e)
        return 2

    try:
        passes, _ = load_passes(args.passes)
    except PassError as e:
        eprint("zh2en: %s" % e)
        return 2

    user_glossary = {}
    if args.glossary:
        try:
            user_glossary = parse_glossary_file(args.glossary)
        except OSError as e:
            eprint("zh2en: cannot read glossary file: %s" % e)
            return 2
    glossary = user_glossary

    started = time.time()

    # Unit plans are computed once from the original source and reused for
    # every pass, so each pass sees the matching source unit alongside its
    # work. `separators` are the input's own blank-line separators.
    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(source_paragraphs)
    paragraph_plan = [[p] for p in source_paragraphs]

    analysis_text = None

    for pnum, passdef in enumerate(passes, 1):
        USAGE.eprint("Starting pass %d/%d [%s]..." % (pnum, len(passes), passdef.name))

        if passdef.mode == "analysis":
            if args.verbose:
                eprint(
                    "zh2en: [%s] whole-document analysis (%d characters)"
                    % (passdef.name, len(text))
                )
            try:
                analysis_text = run_analysis_once(
                    config, passdef, text, glossary, not args.no_cache, args.verbose
                )
            except LLMError as e:
                eprint("zh2en: pass [%s] failed: %s" % (passdef.name, e))
                return 1
            USAGE.log_pass()
            if passdef.ascii:
                USAGE.eprint(
                    "Starting ascii enforcement for [%s]..." % passdef.name
                )
                try:
                    analysis_text = ensure_ascii_output(
                        config,
                        passdef.name,
                        analysis_text,
                        source_paragraphs,
                        not args.no_cache,
                        args.verbose,
                    )
                except LLMError as e:
                    eprint(
                        "zh2en: pass [%s] ascii enforcement failed: %s"
                        % (passdef.name, e)
                    )
                    return 1
                USAGE.log_pass()
            continue

        plan = paragraph_plan if passdef.mode == "paragraph" else chunk_plan
        work_paragraphs, _ = split_paragraphs(text)
        work_groups = regroup_by_plan(work_paragraphs, plan)
        if work_groups is None:
            eprint(
                "zh2en: [%s] paragraph count changed by a previous pass; "
                "grouping working text independently" % passdef.name
            )
            if passdef.mode == "paragraph":
                work_groups = [[p] for p in work_paragraphs]
            else:
                work_groups = make_chunks(work_paragraphs)

        if args.verbose:
            if passdef.mode == "paragraph":
                eprint(
                    "zh2en: [%s] %d paragraph(s), one call per paragraph"
                    % (passdef.name, len(work_groups))
                )
            else:
                eprint(
                    "zh2en: [%s] %d paragraph(s) in %d chunk(s)"
                    % (passdef.name, len(work_paragraphs), len(work_groups))
                )

        unit_seps = [sep for _, sep in iter_units(plan, separators)]
        pass_salt = passdef.name + "\x00" + passdef.instruction
        outputs = []
        for i, work_group in enumerate(work_groups):
            source_chunk_text = "\n\n".join(plan[i]) if i < len(plan) else ""
            work_chunk_text = "\n\n".join(work_group)
            key = cache_key(
                source_chunk_text, config.model, glossary, pass_salt, work_chunk_text
            )
            trailing_sep = unit_seps[i] if i < len(unit_seps) else ""
            if not source_chunk_text:
                # No source paragraph pairs with this unit (the text drifted
                # out of alignment in an earlier pass). Reviewing or
                # translating without a source only invites guesses, so keep
                # the working text as-is.
                if args.verbose:
                    eprint(
                        "zh2en: [%s] unit %d/%d has no matching source; "
                        "passing it through unchanged"
                        % (passdef.name, i + 1, len(work_groups))
                    )
                outputs.append(work_chunk_text + trailing_sep)
                continue
            if not args.no_cache:
                cached = cache_get(key)
                if cached is not None:
                    outputs.append(cached + trailing_sep)
                    if args.verbose:
                        eprint(
                            "zh2en: [%s] unit %d/%d cache hit"
                            % (passdef.name, i + 1, len(work_groups))
                        )
                    continue
            try:
                result = run_pass(
                    config,
                    passdef,
                    source_chunk_text,
                    work_chunk_text,
                    glossary,
                    analysis_text,
                    args.verbose,
                )
            except LLMError as e:
                eprint(
                    "zh2en: pass [%s] failed on unit %d: %s" % (passdef.name, i + 1, e)
                )
                return 1
            if not args.no_cache:
                cache_put(key, result)
            outputs.append(result + trailing_sep)
            if args.verbose:
                eprint(
                    "zh2en: [%s] unit %d/%d done"
                    % (passdef.name, i + 1, len(work_groups))
                )

        USAGE.log_pass()
        text = "".join(outputs)
        if passdef.ascii:
            USAGE.eprint("Starting ascii enforcement for [%s]..." % passdef.name)
            try:
                text = ensure_ascii_output(
                    config,
                    passdef.name,
                    text,
                    source_paragraphs,
                    not args.no_cache,
                    args.verbose,
                )
            except LLMError as e:
                eprint(
                    "zh2en: pass [%s] ascii enforcement failed: %s"
                    % (passdef.name, e)
                )
                return 1
            USAGE.log_pass()

    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")

    if args.verbose:
        eprint("zh2en: done in %.1fs" % (time.time() - started))
    USAGE.log_total(time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())

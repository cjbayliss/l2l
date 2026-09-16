#!/usr/bin/env python3
"""zh2en — multi-pass Chinese-to-English translator.

Reads Chinese text from stdin, writes English translation to stdout.
Diagnostics go to stderr so the tool stays pipe-friendly.

Pipeline:
    stdin -> split into paragraphs
         -> passes 1..N from the passes INI file:
              mode = analysis    runs once over the whole document (split into
                                 parts and merged when it exceeds the request
                                 budget); its output is attached to every later
                                 call as context and never enters the
                                 translation chain
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

CHUNK_BUDGET_TOKENS = 3500
ANALYSIS_RESERVE_TOKENS = 128

STRICT_FIDELITY_SUFFIX = (
    "\n- STRICT FIDELITY MODE: keep the exact number of paragraphs "
    "and the exact separators of the source. Do not merge "
    "or split paragraphs."
)

ANALYSIS_MERGE_INSTRUCTION = """
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
"""


class Config:
    def __init__(self):
        self.base_url = os.environ.get("TRANSLATE_BASE_URL", "").rstrip("/")
        self.api_key = os.environ.get("TRANSLATE_API_KEY", "")
        self.model = os.environ.get("TRANSLATE_MODEL", "gpt-4o-mini")
        self.timeout = float(os.environ.get("TRANSLATE_TIMEOUT", "120"))
        try:
            self.max_tokens = int(os.environ.get("TRANSLATE_MAX_TOKENS", "100000"))
        except ValueError:
            self.max_tokens = 0

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

        if self.max_tokens <= 0:
            raise ConfigError("TRANSLATE_MAX_TOKENS must be a positive integer")


class ConfigError(Exception):
    pass


class LLMError(Exception):
    pass


class PassError(Exception):
    pass


class Pass:
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
        return "none"

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

    default_ascii = options.get("ascii", False)
    for passdef in passes:
        if passdef.ascii is None:
            passdef.ascii = default_ascii

    return passes, options


def load_options(section, opts):
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


def chat(config, system, user, overrides=None, verbose=False):
    estimated = estimate_tokens(system) + estimate_tokens(user)
    if estimated > config.max_tokens:
        raise LLMError(
            "request is ~%d tokens, over the %d-token budget; raise "
            "TRANSLATE_MAX_TOKENS or shorten the input" % (estimated, config.max_tokens)
        )

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
        m = re.match(r"\s*<think>(.*?)</think>", content, re.DOTALL)
        if m:
            if verbose:
                print(m.group(1).strip(), file=sys.stderr, flush=True)

            content = content[m.end() :]

    if not isinstance(content, str):
        raise LLMError("unexpected content type: %s" % str(body)[:500])

    return content


BLANK_LINE_RE = re.compile(r"\n\s*\n")


def split_paragraphs(text):
    blanks = BLANK_LINE_RE.findall(text)
    lone = text.count("\n") - sum(s.count("\n") for s in blanks)
    sep_re = r"(\n+)" if lone >= 3 * len(blanks) else r"(\n\s*\n)"
    parts = re.split(sep_re, text)
    is_sep = re.compile(sep_re).fullmatch
    paragraphs = [p for p in parts if p and not is_sep(p)]
    separators = [s for s in parts if s and is_sep(s)]

    return paragraphs, separators


def make_chunks(paragraphs, budget=CHUNK_BUDGET_TOKENS):
    chunks = []
    current = []
    size = 0
    for p in paragraphs:
        p_size = estimate_tokens(p)
        if current and size + p_size > budget:
            chunks.append(current)
            current = []
            size = 0

        current.append(p)
        size += p_size

    if current:
        chunks.append(current)

    return chunks


def _is_cjk(char):
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
    cjk = 0
    for char in text:
        if _is_cjk(char):
            cjk += 1

    return cjk + (len(text) - cjk + 3) // 4


SENTENCE_BOUNDARY_CHARS = "。！？!?；;\n"


def iter_sentences(text):
    buf = []
    for char in text:
        buf.append(char)
        if char in SENTENCE_BOUNDARY_CHARS:
            yield "".join(buf)
            buf = []

    if buf:
        yield "".join(buf)


def split_to_budget(text, budget):
    pieces = []
    buf = []
    size = 0
    for sentence in iter_sentences(text):
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


def iter_units(plan, separators):
    consumed = 0
    for i, unit in enumerate(plan):
        consumed += len(unit)
        sep = separators[consumed - 1] if consumed - 1 < len(separators) else ""
        yield unit, (sep if i < len(plan) - 1 else "")


def regroup_by_plan(paragraphs, plan):
    total = sum(len(c) for c in plan)
    if len(paragraphs) != total:
        return None

    groups = []
    i = 0
    for chunk in plan:
        groups.append(paragraphs[i : i + len(chunk)])
        i += len(chunk)

    return groups


def analysis_block(analysis):
    return analysis.strip() if analysis and analysis.strip() else "(none)"


def analysis_user_prefix():
    return "Source text (full document, original Chinese):\n"


def merge_user_prefix():
    return "Partial preparation briefs (parts of one document, in source order):\n\n"


def run_analysis(config, passdef, text, verbose=False):
    return chat(
        config,
        passdef.instruction,
        analysis_user_prefix() + text,
        overrides=passdef.api_overrides,
        verbose=verbose,
    ).strip()


def run_merge(config, passdef, briefs, verbose=False):
    return chat(
        config,
        ANALYSIS_MERGE_INSTRUCTION,
        merge_user_prefix() + "\n\n".join(briefs),
        overrides=passdef.api_overrides,
        verbose=verbose,
    ).strip()


def merge_analysis(config, passdef, briefs, verbose=False):
    while len(briefs) > 1:
        budget = (
            config.max_tokens
            - estimate_tokens(ANALYSIS_MERGE_INSTRUCTION)
            - estimate_tokens(merge_user_prefix())
            - ANALYSIS_RESERVE_TOKENS
        )
        groups = make_chunks(briefs, max(budget, 1))
        if len(groups) == len(briefs):
            raise LLMError(
                "partial analysis brief (~%d tokens) does not fit the "
                "%d-token budget; raise TRANSLATE_MAX_TOKENS or shorten "
                "the input"
                % (max(estimate_tokens(b) for b in briefs), config.max_tokens)
            )

        briefs = [
            (
                group[0]
                if len(group) == 1
                else run_merge(config, passdef, group, verbose)
            )
            for group in groups
        ]

    return briefs[0]


def analyze_document(config, passdef, full_text, verbose=False):
    overhead = estimate_tokens(passdef.instruction) + estimate_tokens(
        analysis_user_prefix()
    )
    budget = max(config.max_tokens - overhead - ANALYSIS_RESERVE_TOKENS, 1)
    paragraphs, _ = split_paragraphs(full_text)
    units = []
    for para in paragraphs:
        if estimate_tokens(para) <= budget:
            units.append(para)
        else:
            units.extend(split_to_budget(para, budget))

    chunks = make_chunks(units, budget)
    if len(chunks) <= 1:
        return run_analysis(config, passdef, full_text, verbose)

    if verbose:
        eprint(
            "zh2en: [%s] ~%d tokens over the %d-token budget; analysing in %d part(s)"
            % (passdef.name, estimate_tokens(full_text), config.max_tokens, len(chunks))
        )

    briefs = [
        run_analysis(config, passdef, "\n\n".join(chunk), verbose) for chunk in chunks
    ]
    return merge_analysis(config, passdef, briefs, verbose)


def run_analysis_once(config, passdef, full_text, use_cache, verbose):
    salt = passdef.name + "\x00" + passdef.instruction
    key = cache_key(
        full_text,
        config.model,
        salt,
        overrides=passdef.api_overrides,
    )
    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            if verbose:
                eprint("zh2en: [%s] cache hit" % passdef.name)

            return cached

    result = analyze_document(config, passdef, full_text, verbose)
    if use_cache:
        cache_put(key, result)

    return result


def run_pass(config, passdef, source_chunk, work_chunk, analysis=None, verbose=False):
    system = passdef.instruction
    if passdef.strict_fidelity:
        system += STRICT_FIDELITY_SUFFIX

    user = (
        "Preparation brief from a full read of the text "
        "(outline, names, hard-to-translate items):\n%s\n\n"
        "Source text (original Chinese):\n%s" % (analysis_block(analysis), source_chunk)
    )

    if work_chunk is not None and work_chunk != source_chunk:
        user += "\n\nCurrent draft from the previous pass:\n%s" % work_chunk

    return clean_translation(
        chat(config, system, user, overrides=passdef.api_overrides, verbose=verbose)
    )


def clean_translation(text):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t)

    if len(t) >= 2:
        pairs = {'"': '"', "'": "'", "“": "”", "‘": "’"}
        if t[0] in pairs and t[-1] == pairs[t[0]]:
            t = t[1:-1]

    t = re.sub(r"^Translation:\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


ASCII_FIX_ATTEMPTS = 3

ASCII_FIX_INSTRUCTION = """
This paragraph failed to be fully translated or contains non-ASCII
characters. Please analyse it and only output a clean translation
without any non-ASCII.
"""

ASCII_CHAR_MAP = {
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


def to_ascii_mechanical(text):
    t = text
    for src, repl in ASCII_CHAR_MAP.items():
        t = t.replace(src, repl)

    t = unicodedata.normalize("NFKD", t)
    return "".join(c for c in t if not unicodedata.combining(c))


def ascii_fix_llm(config, pass_name, source_para, out_para, use_cache, verbose):
    salt = "ascii-fix\x00" + pass_name + "\x00" + ASCII_FIX_INSTRUCTION
    key = cache_key(source_para, config.model, salt, out_para)
    if use_cache:
        cached = cache_get(key)
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
    seen = []
    for c in text:
        if not c.isascii() and c not in seen:
            seen.append(c)
            if len(seen) >= limit:
                break

    return "".join(seen)


def ensure_ascii_output(config, pass_name, text, source_paragraphs, use_cache, verbose):
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


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "zh2en")
    os.makedirs(d, exist_ok=True)
    return d


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

    started = time.time()

    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(source_paragraphs)
    paragraph_plan = [[p] for p in source_paragraphs]

    analysis_text = None

    for pnum, passdef in enumerate(passes, 1):
        USAGE.eprint("Starting pass %d/%d [%s]..." % (pnum, len(passes), passdef.name))
        if passdef.mode == "analysis":
            if args.verbose:
                eprint(
                    "zh2en: [%s] whole-document analysis (%d characters, ~%d tokens)"
                    % (passdef.name, len(text), estimate_tokens(text))
                )

            try:
                analysis_text = run_analysis_once(
                    config, passdef, text, not args.no_cache, args.verbose
                )
            except LLMError as e:
                eprint("zh2en: pass [%s] failed: %s" % (passdef.name, e))
                return 1

            USAGE.log_pass()
            if passdef.ascii:
                USAGE.eprint("Starting ascii enforcement for [%s]..." % passdef.name)
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
                source_chunk_text,
                config.model,
                pass_salt,
                work_chunk_text,
                overrides=passdef.api_overrides,
            )
            trailing_sep = unit_seps[i] if i < len(unit_seps) else ""
            if not source_chunk_text:
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
                    "zh2en: pass [%s] ascii enforcement failed: %s" % (passdef.name, e)
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

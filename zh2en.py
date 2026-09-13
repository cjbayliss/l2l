#!/usr/bin/env python3
"""zh2en — multi-pass Chinese-to-English translator.

Reads Chinese text from stdin, writes English translation to stdout.
Diagnostics go to stderr so the tool stays pipe-friendly.

Pipeline:
    stdin -> chunk by paragraphs
         -> Pass 1: analysis (glossary of names, idioms, terms, style notes)
         -> Pass 2: translation, with glossary context
         -> stitch -> stdout
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

CHUNK_BUDGET = 3500  # max source characters per chunk


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


def chat(config, system, user, temperature=0.2):
    payload = {
        "model": config.model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
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
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape: %s" % str(body)[:500]) from None


# ---------------------------------------------------------------------------
# Chunking: split on blank lines, pack paragraphs into budgeted chunks.
# ---------------------------------------------------------------------------


def split_paragraphs(text):
    """Return (paragraphs, separators) preserving blank-line structure."""
    parts = re.split(r"(\n\s*\n)", text)
    paragraphs = [p for p in parts if p and not re.fullmatch(r"\n\s*\n", p)]
    separators = [s for s in parts if s and re.fullmatch(r"\n\s*\n", s)]
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


def iter_chunk_items(paragraphs, budget=CHUNK_BUDGET):
    """Yield (chunk_paragraphs, separator_after_chunk) in input order.

    There is always exactly len(paragraphs)-1 separators; the separator that
    follows a chunk is the one at the index of its last paragraph.
    """
    _, separators = split_paragraphs("\n\n".join(paragraphs))
    chunks = make_chunks(paragraphs, budget)
    consumed = 0
    for i, chunk in enumerate(chunks):
        consumed += len(chunk)
        sep = separators[consumed - 1] if consumed - 1 < len(separators) else ""
        yield chunk, (sep if i < len(chunks) - 1 else "")


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
# Pass 1: analysis
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM = """\
You are a Chinese-to-English translation analyst. Examine the source text and \
return STRICT JSON only (no markdown fences, no commentary) with this shape:

{
  "names": {"source_term": "intended English rendering"},
  "idioms": {"chengyu or idiom": "plain-meaning gloss to guide translation"},
  "terms": {"technical/domain term": "preferred consistent English rendering"},
  "notes": ["short style/register/ambiguity notes for the translator"]
}

Judge from context whether a string of characters is a personal/place name \
rather than a common word. Only include items actually present in the text. \
Use empty objects/lists if nothing applies."""


def run_analysis(config, source_text):
    raw = chat(config, ANALYSIS_SYSTEM, source_text, temperature=0.0)
    parsed = parse_json_loose(raw)
    if not isinstance(parsed, dict):
        return {"names": {}, "idioms": {}, "terms": {}, "notes": []}
    out = {"names": {}, "idioms": {}, "terms": {}, "notes": []}
    for key in ("names", "idioms", "terms"):
        v = parsed.get(key)
        if isinstance(v, dict):
            out[key] = {str(k): str(v2) for k, v2 in v.items()}
    notes = parsed.get("notes")
    if isinstance(notes, list):
        out["notes"] = [str(n) for n in notes]
    return out


def flatten_glossary(analysis):
    """Merge analysis categories into one term->rendering map."""
    merged = {}
    merged.update(analysis.get("names", {}))
    merged.update(analysis.get("idioms", {}))
    merged.update(analysis.get("terms", {}))
    return merged


def notes_text(analysis):
    notes = analysis.get("notes", [])
    return "\n".join("- " + n for n in notes) if notes else "(none)"


# ---------------------------------------------------------------------------
# Pass 2: translation
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM = """\
You are an expert Chinese-to-English literary translator.

Rules:
- Output ONLY the English translation. No preamble, no explanations, no \
quotation marks around the whole output, no markdown fences.
- Preserve the paragraph structure of the source exactly: one source \
paragraph becomes one output paragraph, separated the same way.
- Use the provided glossary consistently for names and recurring terms.
- For untranslatable wordplay, translate the sense and add a brief \
translator's note in square brackets, e.g. [translator's note: pun on ...].
- Match the register and tone described in the style notes.
- Translate, never summarize or omit content."""


def run_translation(config, chunk_text, glossary, notes, strict=False):
    system = TRANSLATE_SYSTEM
    if strict:
        system += (
            "\n- STRICT FIDELITY MODE: keep the exact number of paragraphs "
            "and the exact blank-line separators of the source. Do not merge "
            "or split paragraphs."
        )
    user = (
        "Glossary (use these renderings exactly):\n%s\n\n"
        "Style and context notes from analysis:\n%s\n\n"
        "Source text:\n%s" % (glossary_text(glossary), notes_text(notes), chunk_text)
    )
    return clean_translation(chat(config, system, user, temperature=0.2))


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
# Caching
# ---------------------------------------------------------------------------


def cache_dir():
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    d = os.path.join(base, "zh2en")
    os.makedirs(d, exist_ok=True)
    return d


def cache_key(chunk_text, model, glossary):
    h = hashlib.sha256()
    h.update(chunk_text.encode("utf-8"))
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
        "--glossary",
        metavar="FILE",
        help="glossary file of `term -> rendering` lines; "
        "entries override the analysis pass",
    )
    ap.add_argument(
        "--analysis-only",
        action="store_true",
        help="print the analysis-pass glossary and notes, " "don't translate",
    )
    ap.add_argument(
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    ap.add_argument(
        "--fidelity",
        choices=["natural", "strict"],
        default="natural",
        help="strict preserves exact paragraph structure",
    )
    ap.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) to stderr",
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

    user_glossary = {}
    if args.glossary:
        try:
            user_glossary = parse_glossary_file(args.glossary)
        except OSError as e:
            eprint("zh2en: cannot read glossary file: %s" % e)
            return 2

    paragraphs, _ = split_paragraphs(text)
    chunk_items = list(iter_chunk_items(paragraphs))
    if args.verbose:
        eprint(
            "zh2en: %d paragraph(s) in %d chunk(s)"
            % (len(paragraphs), len(chunk_items))
        )

    started = time.time()

    # Pass 1: per-chunk analysis, merged into a global glossary.
    USAGE.eprint("Starting analysis...")
    glossary = {}
    all_notes = []
    for i, (chunk, _) in enumerate(chunk_items):
        chunk_text = "\n\n".join(chunk)
        try:
            analysis = run_analysis(config, chunk_text)
        except LLMError as e:
            eprint("zh2en: analysis pass failed on chunk %d: %s" % (i + 1, e))
            eprint("zh2en: continuing without analysis for this chunk")
            continue
        glossary = merge_glossaries(glossary, flatten_glossary(analysis))
        all_notes.extend(analysis.get("notes", []))
        if args.verbose:
            eprint("zh2en: analysis chunk %d/%d done" % (i + 1, len(chunk_items)))

    glossary = merge_glossaries(glossary, user_glossary)
    notes = {"notes": all_notes}
    USAGE.log_pass()

    if args.analysis_only:
        print("== Glossary ==")
        print(glossary_text(glossary))
        print("\n== Notes ==")
        print(notes_text(notes))
        USAGE.log_pass()
        return 0

    if args.verbose:
        eprint("zh2en: %d glossary entries, %d notes" % (len(glossary), len(all_notes)))

    # Pass 2: translate chunk by chunk with the global glossary.
    USAGE.eprint("Starting translation...")
    outputs = []
    cache_hits = 0
    for i, (chunk, sep) in enumerate(chunk_items):
        chunk_text = "\n\n".join(chunk)
        key = cache_key(chunk_text, config.model, glossary)
        if not args.no_cache:
            cached = cache_get(key)
            if cached is not None:
                outputs.append(cached + sep)
                cache_hits += 1
                if args.verbose:
                    eprint("zh2en: chunk %d/%d cache hit" % (i + 1, len(chunk_items)))
                continue
        try:
            result = run_translation(
                config,
                chunk_text,
                glossary,
                notes,
                strict=(args.fidelity == "strict"),
            )
        except LLMError as e:
            eprint("zh2en: translation failed on chunk %d: %s" % (i + 1, e))
            return 1
        if not args.no_cache:
            cache_put(key, result)
        outputs.append(result + sep)
        if args.verbose:
            eprint("zh2en: translated chunk %d/%d" % (i + 1, len(chunk_items)))

    USAGE.log_pass()

    sys.stdout.write("".join(outputs))
    if not outputs[-1].endswith("\n"):
        sys.stdout.write("\n")

    if args.verbose:
        eprint(
            "zh2en: done in %.1fs (%d/%d chunks cached)"
            % (time.time() - started, cache_hits, len(chunk_items))
        )
    USAGE.log_total(time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())

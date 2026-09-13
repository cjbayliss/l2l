#!/usr/bin/env python3
"""zh2en — multi-pass Chinese-to-English translator.

Reads Chinese text from stdin, writes English translation to stdout.
Diagnostics go to stderr so the tool stays pipe-friendly.

Pipeline:
    stdin -> chunk by paragraphs
         -> pass 1..N from the passes INI file (each pass receives the
            original source chunk plus the previous pass's output)
         -> stitch -> stdout
"""

import argparse
import configparser
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

CHUNK_BUDGET = 3500  # max source characters per chunk

STRICT_FIDELITY_SUFFIX = (
    "\n- STRICT FIDELITY MODE: keep the exact number of paragraphs "
    "and the exact blank-line separators of the source. Do not merge "
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

    Keys other than the reserved ones pass through into the chat-completions
    request payload as-is (with type inference), so API fields like
    `temperature`, `model`, `reasoning_effort`, `top_p` need no code here.
    """

    def __init__(self, name, instruction, strict_fidelity=False, api_overrides=None):
        self.name = name
        self.instruction = instruction
        self.strict_fidelity = strict_fidelity
        self.api_overrides = api_overrides or {}


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


def load_passes(path):
    """Load pass definitions from an INI file, in section order.

    Reserved keys: `instruction-file` (required; system-prompt text, resolved
    relative to the INI file) and `strict_fidelity` (bool). All other keys
    become chat-completions payload overrides.
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
    for section in parser.sections():
        opts = dict(parser.items(section))
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

        api_overrides = {}
        for key, value in opts.items():
            parsed = _infer_type(value)
            if parsed is not None:
                api_overrides[key] = parsed
        passes.append(Pass(section, instruction, strict_fidelity, api_overrides))
    return passes


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


def chat(config, system, user, temperature=0.2, overrides=None):
    payload = {
        "model": config.model,
        "temperature": temperature,
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
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise LLMError("unexpected response shape: %s" % str(body)[:500]) from None
    if isinstance(content, list):
        # some models return a list of content parts
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    if not isinstance(content, str):
        raise LLMError("unexpected content type: %s" % str(body)[:500])
    return content


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


def notes_text(notes):
    items = notes.get("notes", []) if isinstance(notes, dict) else []
    return "\n".join("- " + n for n in items) if items else "(none)"


def run_pass(config, passdef, source_chunk, work_chunk, glossary, notes):
    system = passdef.instruction
    if passdef.strict_fidelity:
        system += STRICT_FIDELITY_SUFFIX
    user = (
        "Glossary (use these renderings exactly):\n%s\n\n"
        "Style and context notes from analysis:\n%s\n\n"
        "Source text (original Chinese):\n%s"
        % (glossary_text(glossary), notes_text(notes), source_chunk)
    )
    if work_chunk is not None and work_chunk != source_chunk:
        user += "\n\nCurrent draft from the previous pass:\n%s" % work_chunk
    return clean_translation(
        chat(config, system, user, overrides=passdef.api_overrides)
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

    try:
        passes = load_passes(args.passes)
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
    notes = {"notes": []}

    started = time.time()

    # Chunks are planned once from the original source and reused for every
    # pass, so each pass sees the matching source chunk alongside its work.
    source_paragraphs, _ = split_paragraphs(text)
    chunk_items = list(iter_chunk_items(source_paragraphs))
    source_chunks = [c for c, _ in chunk_items]
    seps = [sep for _, sep in chunk_items]

    for pnum, passdef in enumerate(passes, 1):
        USAGE.eprint("Starting pass %d/%d [%s]..." % (pnum, len(passes), passdef.name))
        work_paragraphs, _ = split_paragraphs(text)
        work_groups = regroup_by_plan(work_paragraphs, source_chunks)
        if work_groups is None:
            eprint(
                "zh2en: [%s] paragraph count changed by a previous pass; "
                "re-chunking working text independently" % passdef.name
            )
            work_groups = make_chunks(work_paragraphs)

        if args.verbose:
            eprint(
                "zh2en: [%s] %d paragraph(s) in %d chunk(s)"
                % (passdef.name, len(work_paragraphs), len(work_groups))
            )

        pass_salt = passdef.name + "\x00" + passdef.instruction
        outputs = []
        for i, work_group in enumerate(work_groups):
            source_chunk_text = (
                "\n\n".join(source_chunks[i]) if i < len(source_chunks) else ""
            )
            work_chunk_text = "\n\n".join(work_group)
            key = cache_key(
                source_chunk_text, config.model, glossary, pass_salt, work_chunk_text
            )
            if not args.no_cache:
                cached = cache_get(key)
                if cached is not None:
                    outputs.append(cached + (seps[i] if i < len(seps) - 1 else ""))
                    if args.verbose:
                        eprint(
                            "zh2en: [%s] chunk %d/%d cache hit"
                            % (passdef.name, i + 1, len(work_groups))
                        )
                    continue
            try:
                result = run_pass(
                    config, passdef, source_chunk_text, work_chunk_text, glossary, notes
                )
            except LLMError as e:
                eprint(
                    "zh2en: pass [%s] failed on chunk %d: %s" % (passdef.name, i + 1, e)
                )
                return 1
            if not args.no_cache:
                cache_put(key, result)
            outputs.append(result + (seps[i] if i < len(seps) - 1 else ""))
            if args.verbose:
                eprint(
                    "zh2en: [%s] chunk %d/%d done"
                    % (passdef.name, i + 1, len(work_groups))
                )

        USAGE.log_pass()
        text = "".join(outputs)

    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")

    if args.verbose:
        eprint("zh2en: done in %.1fs" % (time.time() - started))
    USAGE.log_total(time.time() - started)
    return 0


if __name__ == "__main__":
    sys.exit(main())

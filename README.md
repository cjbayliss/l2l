# l2l

**NOTE:** This project is created using an LLM (mostly GLM 5.3 Flash,
Kimi K3, and Deepseek V4.1 Flash 1017).

l2l is a command-line translator: it reads text from stdin, translates
it with any OpenAI-compatible chat completions endpoint, and writes the
result to stdout.

There are no built-in language pairs - the instruction in your config
defines the direction, so one tool handles any pair. l2l splits text
into paragraphs and runs them through one or more passes, each a call
type with its own instruction, model, and parameters. Malformed replies
are re-asked with corrective feedback, and every result is cached on
disk so interrupted runs resume cheaply.

## Installation

```sh
uv tool install git+https://github.com/cjbayliss/l2l
```

This installs the `l2l` command. To run it without installing:

```sh
uv run l2l
```

## Requirements

- Python 3.15 or newer
- [uv](https://docs.astral.sh/uv/) for the install and test commands
  below
- libcurl: l2l uses `pycurl` for HTTP transport; on platforms without a
  `pycurl` wheel, install the libcurl development headers first so pip
  can build it
- An API key for any OpenAI-compatible chat completions endpoint

## Usage

```sh
l2l [options] < input.txt > output.txt
```

For example:

```sh
l2l -c zh2en.toml < chapter1.txt > chapter1.en.txt
```

Empty input succeeds without calling the endpoint. To get started:

1. Copy `docs/configs/zh2en.toml` (Chinese → English) or
   `docs/configs/ja2en.toml` (Japanese → English), plus the matching
   instruction files from `docs/prompts/`.
2. Put your API key in the config - or better, in the
   `TRANSLATE_API_KEY` environment variable.
3. Run `l2l --check-config` to verify your setup, then translate.

### Options

Short forms are listed before their long forms; long options are never
abbreviated.

Meta:

- `-h`, `--help` - show the help message and exit.
- `-v`, `--version` - print the version and exit.
- `-V`, `--verbose` - print chunking, cache, timing, and reasoning
  diagnostics to stderr.

Configuration and API:

- `-c`, `--config FILE` - path to a TOML config file; see
  [Config file resolution](#config-file-resolution).
- `-u`, `--base-url URL` - API endpoint; overrides `[api] base_url` and
  `TRANSLATE_BASE_URL`.
- `-k`, `--api-key KEY` - API key; overrides `[api] api_key` and
  `TRANSLATE_API_KEY`.
- `-m`, `--model MODEL` - default model; overrides `[api] model` and
  `TRANSLATE_MODEL`.
- `-t`, `--timeout SECONDS` - per-request timeout; overrides
  `[api] timeout` and `TRANSLATE_TIMEOUT`.
- `-T`, `--maximum-tokens N` - per-request token budget; overrides
  `[api] max_tokens` and `TRANSLATE_MAX_TOKENS`.
- `-s`, `--stream` / `--no-stream` - force streamed or plain responses;
  default follows `api.params.stream`.
- `--no-structured-output` - request glossary extraction without the
  JSON-schema response format, for endpoints that do not support
  structured output.

Translation behaviour:

- `-g`, `--glossary FILE` - JSON glossary file with mandatory term
  renderings; repeatable, later files override earlier entries, and
  generated or updated entries are written to the first file.
- `-l`, `--target-language LANG` - target language; required unless
  `target_language` is set in the config file. Drives
  `{target_language}` prompt substitution, cache separation, glossary
  extraction, and target script validation for every pass.
- `-P`, `--ensure-paragraphs` - check the paragraph count after each
  pass (see `ensure_paragraphs` under `[options]`).
- `-b`, `--best-effort` - keep the last reply and continue when a unit
  still fails validation after all repair attempts (default: fail the
  run with exit code 1). Unvalidated replies are never cached.

Cache and logs:

- `--cache-directory DIR` - cache directory (default:
  `$XDG_CACHE_HOME/l2l`, falling back to `~/.cache/l2l`).
- `--no-cache` - bypass the translation cache for this run.
- `--log-retention-days DAYS` - delete run logs older than DAYS days at
  startup (default 30; `0` keeps every log).
- `-p`, `--print-log-path` - print the run log's path to stderr at
  startup.

Modes:

- `--check-config` - print the resolved configuration and exit without
  translating.
- `-n`, `--dry-run` - print the per-pass call plan (units, token
  estimates, cache keys) and exit without calling the endpoint; warns
  when a paragraph cannot fit in a single request.
- `--extract-glossary FILE` - extract a starting glossary from the stdin
  text with one model call, write it to FILE, and exit. Existing
  `--glossary` entries are sent as already-known names and are never
  re-extracted; the model replies with only the new entries found, and
  the reply is merged into the existing entries. Pipeline passes are
  ignored in this mode.
- `--prune-cache DAYS` - delete cache entries older than DAYS days and
  exit.

The modes `--check-config`, `--prune-cache`, `--extract-glossary`, and
`--dry-run` are mutually exclusive: combining any two fails with a
configuration error (exit code 2) before any endpoint work starts.

Precedence, from weakest to strongest:

```text
defaults < user config < selected config < environment < command line
```

Config-file keys and `TRANSLATE_*` environment variables keep their
existing names even though some command-line overrides were renamed.

### Exit codes

- `0` - success (including empty input).
- `1` - pipeline error (endpoint, budget, or pass failure).
- `2` - configuration or argument error.
- `130` - interrupted with Ctrl-C.
- `141` - stdout closed early (SIGPIPE), e.g. piping into `head`.

Error messages number paragraphs, units, and calls from 1, matching
their position in the document (the first paragraph is paragraph 1); the
error types themselves carry 0-based indices and `describe` adds one
when rendering.

When stderr is a terminal, press **Tab** during a run to toggle verbose
output; the listener is off for pipes and CI.

Every run writes a log to `<cache-dir>/logs/<timestamp>-<pid>.log`, even
with `--no-cache`. It records each request payload and everything
received in reply; request headers - and therefore API keys - are never
logged. Pass `--print-log-path` to print the log's path at startup.

## Configuration

Configuration lives in a TOML file with a top-level `target_language`
and up to three kinds of table:

```toml
target_language = "English"

[api]      # endpoint, key, model, limits, extra request parameters
[[pass]]   # one or more passes, run in order
[options]  # global toggles
```

The top-level `target_language` is required unless `--target-language`
is passed. It names the language every pass produces, controls cache
identity, and supplies the expected script for output validation. Names
and BCP 47 tags are resolved through CLDR likely subtags, so `English`,
`en`, `eng`, `Chinese (Simplified)`, and `zh-Hant` all work. Japanese
and Korean accept both of their usual scripts (Hiragana, Katakana, and
Han; Hangul and Han). Names that CLDR does not know are allowed: they
disable script checks, fall back to the ASCII-letter heuristic, and
produce a warning. Script reports use Unicode Script names such as
`Latin`, `Han`, and `Cyrillic`.

A minimal example:

```toml
target_language = "English"

[api]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-..."
model = "z-ai/glm-5.3-flash"

[[pass]]
name = "translate"
instruction = "Translate Chinese fiction to natural {target_language}."
mode = "chunk"
```

Complete multi-pass examples live in `docs/configs/`, with their
instruction files in `docs/prompts/`.

### Config file resolution

1. The `--config FILE` option, if given.
2. The `TRANSLATE_CONFIG` environment variable.
3. `./l2l.toml` in the current directory.
4. `~/.config/l2l/config.toml` (or `$XDG_CONFIG_HOME/l2l/config.toml`).

Separately, that user config is always merged in as a base layer when it
exists; the selected config overrides it. Put your `base_url` and
`api_key` there, so per-project configs only need the passes.

### Environment variables

- `TRANSLATE_CONFIG` - fallback config path (see resolution order
  above).
- `TRANSLATE_BASE_URL`, `TRANSLATE_API_KEY`, `TRANSLATE_MODEL`,
  `TRANSLATE_TIMEOUT`, `TRANSLATE_MAX_TOKENS` - fallbacks for the
  matching `[api]` keys; the command-line options win over all of them.
- `XDG_CACHE_HOME`, `XDG_CONFIG_HOME` - relocate the cache and user
  config directories.

### `[api]` - endpoint settings

- `base_url` (string) - chat completions endpoint, e.g.
  `https://openrouter.ai/api/v1`; a trailing `/` is stripped.
- `api_key` (string) - sent as `Authorization: Bearer ...`.
- `model` (string) - default model for calls (can be overridden per
  pass).
- `timeout` (number, default `120.0`) - request timeout in seconds.
- `max_tokens` (integer, default `100000`) - token budget for requests.
- `params` (table, default `{}`) - extra keys merged into every request
  body, e.g. `stop = ["END"]`; must contain only JSON values.

`base_url`, `api_key`, and `model` have no built-in default but only
need to be set in one place: this table, the user config, the
environment, or the command line.

### `[[pass]]` - pipeline passes

Passes run in order; each pass's output is the next's input.

- `name` (required) - non-empty string, used in logs and cache keys.
- `instruction` or `instruction_file` (exactly one required, except for
  `mode = "update_glossary"` where both may be omitted) - the
  instruction as inline text, or a path to a text file (relative to the
  config file). The instruction defines the language pair and target
  style.
- `mode` (default `"chunk"`) - how the document is processed, see below.
- `model` (default `[api] model`) - model override for this pass's
  calls.
- `params` (default `{}`) - extra request-body keys for this pass's
  calls, merged over `[api] params`.
- `ascii`, `ensure_paragraphs`, `retranslate_untranslated` - per-pass
  overrides of the `[options]` toggles below.
- `target_language` - per-pass target language for every mode; falls
  back to the resolved global target language (config file or
  `--target-language`). It selects prompt substitution, the cache
  namespace, and script validation for that pass.
- `instruction` and `instruction_file` may contain `{target_language}`;
  it is replaced with the pass's effective target language. Other braces
  are left untouched, and no directive is appended automatically.
- `include_types` (default: all types) - `update_glossary` passes only:
  a list restricting which entry types the extraction may add, e.g.
  `include_types = ["character"]`. It applies only to what the model
  returns, never to entries already in the glossary, and filtered-out
  entries are neither written to the glossary file nor injected into
  later translation prompts. Allowed type names are `"character"`,
  `"place"`, and `"other"`.

Modes:

- `chunk` - translates paragraphs grouped into token-budgeted chunks.
  The default, and cheapest for long documents. Each pass derives its
  chunk size from the actual request budget (`max_tokens` minus
  instruction, context, and draft room), capped at 3,500 source tokens
  per call. A single paragraph too large for one request fails the run
  up front.
- `paragraph` - translates each paragraph with its own call, including
  neighbouring source paragraphs as read-only context to anchor short or
  ambiguous units (title-only lines, ellipses, ...).
- `analysis` - reads the whole document and stores a preparation brief
  (outline, names, hard-to-translate items) that later passes include
  for context. The document must fit in one call; otherwise increase
  `max_tokens`.
- `update_glossary` - makes one whole-document model call with the
  built-in glossary extraction prompt, parses the JSON reply, and
  appends only genuinely new entries to the glossary. The user message
  carries a plain list of the already-known names, so the model replies
  with only the entries it newly found instead of echoing the whole
  glossary back, which keeps the call cheaper and faster. The request
  uses a JSON-schema response format (`--no-structured-output` opts out
  for endpoints without structured output support). Requires at least
  one `--glossary FILE` on the command line; the first one is rewritten
  with the updated entries. Existing entries always win: an extracted
  entry whose `original` already exists is discarded (Chinese names have
  many valid renderings and pronouns are context-dependent), so place an
  `update_glossary` pass before the passes that translate. The extraction
  call is cached per document, independent of the current glossary
  contents, so re-running the same document reuses the cached reply
  instead of calling the model again. The updated
  glossary flows into the system prompts of all later passes in the same
  run. The pass's `instruction` is ignored; set `target_language` (or
  rely on the global `target_language` or `--target-language`) to name
  the target language for the extraction.

A typical multi-pass setup pairs an `analysis` pass with a `chunk` pass.

### Glossary file format

A glossary file is a JSON array of entry objects with fixed fields:

```json
[
  {
    "original": "林黛玉",
    "translated": "Lin Daiyu",
    "pronouns": "she/her",
    "type": "character"
  },
  {"original": "金陵", "translated": "Nanjing", "pronouns": null, "type": "place"}
]
```

- `original` (required) - the exact source name; must be a non-empty
  string.
- `translated` - the mandatory rendering in the target language, or
  `null` when unknown.
- `pronouns` - the pronouns for characters, or `null`.
- `type` (required) - one of `"character"`, `"place"`, `"other"`;
  omitted `type` fields in existing files are read as `"other"`.


### Target language and script checks

The required target language is used throughout the pipeline, not only
by glossary extraction:

- Every unit reply is checked for at least one letter in an expected
  target script. A reply that is empty, has the wrong paragraph count,
  exceeds the size ratio, or carries no expected-script letters is
  re-asked with corrective feedback. Output without letters is accepted,
  and Japanese/Korean multi-script output is not flagged.
- `--check-config` and the per-pass dry-run report show the target
  language and scripts for each pass. The pass line also shows whether
  its instruction uses `{target_language}`.
- Unknown language names, instructions that never mention the target
  language, and glossary renderings from another target language produce
  warnings at startup.
- The language is part of every cache key, so changing
  `--target-language` cannot reuse a translation made for another
  language. Older cache entries become unreachable after the salt
  version changes.

### `[options]` - global toggles

- `ascii` (default `false`) - enforce pure ASCII output. Configuration
  loading fails when a pass enables `ascii` and its resolved target
  language is known to use a non-Latin script (Russian, Greek, Japanese,
  Chinese, and so on); set `ascii = false` or choose a Latin-script
  target. Unknown target names still load.
- `ensure_paragraphs` (default `false`) - paragraph-count checking.
  `true` asks each pass's output to match the source paragraph count
  exactly, re-running a mismatching chunk-mode pass with one call per
  paragraph; a positive integer sets a tolerance (`2` accepts outputs
  within ±2 paragraphs); `false` disables the check. Replies whose count
  drifts beyond the tolerance are re-asked with corrective feedback; a
  count that still differs after the re-run is logged as a warning and
  the pass keeps its output - the run continues and still succeeds (exit
  code 0). `paragraph`-mode passes always use exactly one call per
  paragraph.
- `retranslate_untranslated` (default `false`) - after each pass, l2l
  flags output paragraphs that still read as their source (matching
  after mechanical ASCII folding, or carrying no letters of the
  configured target script while also carrying no numbers; a paragraph
  reduced to digits and punctuation, such as "2078!" for "2078年！", is
  considered translated) and re-asks each one with the pass's
  instruction plus neighbouring source paragraphs as context. When more
  than half the paragraphs are flagged in a chunk-mode pass, the whole
  pass is re-run once first. Retranslations are cached only when they no
  longer read as untranslated; a paragraph still untranslated afterwards
  fails the run with exit code 1 (the last reply is kept with a warning
  under `--best-effort`), and the failed repair is not cached.

## Testing

```sh
uv run pytest
```

The suite runs with branch coverage for `l2l` and `tools` and fails
below 99% (see `addopts` in `pyproject.toml`). Lint and types:

```sh
uv run ruff check
uv run mypy --strict
```

## Development tools

`tools/optimize` evolves a translation instruction through rounds of
pairwise judged challenges, or compares two instruction files directly;
see `tools/README.md`:

```sh
uv run python3 -m tools.optimize --help
```

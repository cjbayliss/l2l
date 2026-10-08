# l2l

**NOTE:** This project is created using an LLM (mostly GLM 5.3 Flash).

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

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/) for the install and test commands
  below
- libcurl: l2l uses `pycurl` for HTTP transport; on platforms without a
  `pycurl` wheel, install the libcurl development headers first so pip
  can build it
- An API key for any OpenAI-compatible chat completions endpoint

## Usage

```sh
l2l [options] [CONFIG] < input.txt > output.txt
```

For example:

```sh
l2l zh2en.toml < chapter1.txt > chapter1.en.txt
```

Empty input succeeds without calling the endpoint. To get started:

1. Copy `docs/configs/zh2en.toml` (Chinese → English) or
   `docs/configs/ja2en.toml` (Japanese → English), plus the matching
   instruction files from `docs/prompts/`.
2. Put your API key in the config - or better, in the
   `TRANSLATE_API_KEY` environment variable.
3. Run `l2l --check-config` to verify your setup, then translate.

### Options

- `CONFIG` - optional path to a TOML config file; see
  [Config file resolution](#config-file-resolution).
- `--base-url URL` - API endpoint; overrides `[api] base_url` and
  `TRANSLATE_BASE_URL`.
- `--api-key KEY` - API key; overrides `[api] api_key` and
  `TRANSLATE_API_KEY`.
- `--model MODEL` - default model; overrides `[api] model` and
  `TRANSLATE_MODEL`.
- `--timeout SECONDS` - per-request timeout; overrides `[api] timeout`
  and `TRANSLATE_TIMEOUT`.
- `--max-tokens N` - per-request token budget; overrides
  `[api] max_tokens` and `TRANSLATE_MAX_TOKENS`.
- `--cache-dir DIR` - cache directory (default: `$XDG_CACHE_HOME/l2l`,
  falling back to `~/.cache/l2l`).
- `--no-cache` - bypass the translation cache for this run.
- `--cache-prune DAYS` - delete cache entries older than DAYS days and
  exit.
- `--log-keep DAYS` - delete run logs older than DAYS days at startup
  (default 30; `0` keeps every log).
- `--ensure-paragraphs` - enforce the paragraph count after each pass
  (strict); equivalent to `ensure_paragraphs = true`, see `[options]`.
- `--best-effort` - keep the last reply and continue when a unit still
  fails validation after all repair attempts (default: fail the run with
  exit code 1). Unvalidated replies are never cached.
- `--verbose`, `-v` - print chunking, cache, timing, and reasoning
  diagnostics to stderr.
- `--show-log-path`, `-l` - print the run log's path to stderr at
  startup.
- `--stream` / `--no-stream` - force streamed or plain responses;
  default follows `api.params.stream`.
- `--check-config` - print the resolved configuration and exit without
  translating.
- `--dry-run` - print the per-pass call plan (units, token estimates,
  cache keys) and exit without calling the endpoint; warns when a
  paragraph cannot fit in a single request.
- `--version` - print the version and exit.

Precedence, from weakest to strongest:

```text
defaults < user config < selected config < environment < command line
```

### Exit codes

- `0` - success (including empty input).
- `1` - pipeline error (endpoint, budget, or pass failure).
- `2` - configuration or argument error.
- `130` - interrupted with Ctrl-C.
- `141` - stdout closed early (SIGPIPE), e.g. piping into `head`.

When stderr is a terminal, press **Tab** during a run to toggle verbose
output; the listener is off for pipes and CI.

Every run writes a log to `<cache-dir>/logs/<timestamp>-<pid>.log`, even
with `--no-cache`. It records each request payload and everything
received in reply; request headers - and therefore API keys - are never
logged. Pass `--show-log-path` to print the log's path at startup.

## Configuration

Configuration lives in a TOML file with up to three kinds of table:

```toml
[api]      # endpoint, key, model, limits, extra request parameters
[[pass]]   # one or more passes, run in order
[options]  # global toggles
```

A minimal example:

```toml
[api]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-..."
model = "z-ai/glm-5.3-flash"

[[pass]]
name = "translate"
instruction = "Translate Chinese fiction to natural English."
mode = "chunk"
```

Complete multi-pass examples live in `docs/configs/`, with their
instruction files in `docs/prompts/`.

### Config file resolution

1. The positional `CONFIG` argument, if given.
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
- `instruction` or `instruction_file` (exactly one required) - the
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

A typical multi-pass setup pairs an `analysis` pass with a `chunk` pass.

### `[options]` - global toggles

- `ascii` (default `false`) - enforce pure ASCII output. Assumes a
  Latin-script target language; leave it off for targets such as
  Russian, Greek, Japanese, or Chinese.
- `ensure_paragraphs` (default `false`) - paragraph-count enforcement.
  `true` requires each pass's output to match the source paragraph count
  exactly, re-running a mismatching chunk-mode pass with one call per
  paragraph; a positive integer sets a tolerance (`2` accepts outputs
  within ±2 paragraphs); `false` disables the check. Chunk replies whose
  count drifts beyond the tolerance are re-asked with corrective
  feedback; `paragraph`-mode passes always require exactly one paragraph
  per call.
- `retranslate_untranslated` (default `false`) - after each pass, l2l
  flags output paragraphs that still read as their source (matching
  after mechanical ASCII folding, or carrying no letters of the detected
  target script) and re-asks each one with the pass's instruction plus
  neighbouring source paragraphs as context. When more than half the
  paragraphs are flagged in a chunk-mode pass, the whole pass is re-run
  once first. Retranslations are cached only when they no longer read as
  untranslated; a paragraph still untranslated afterwards fails the run
  with exit code 1, and the failed repair is not cached.

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

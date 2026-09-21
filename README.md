# zh2en

Translate Chinese text from stdin to English on stdout using any
OpenAI-compatible chat completions endpoint (including OpenRouter).

Single-file, standard-library-only Python (3.14+).

## Install

```sh
pip install .
```

This provides the `zh2en` console script. You can also run it directly with
`python3 zh2en.py`.

## Usage

```sh
zh2en [CONFIG] [options] < input.txt > output.txt
```

- `CONFIG` is a TOML file. Resolution order: the positional argument, then
  `$TRANSLATE_CONFIG`, then `./zh2en.toml`, then
  `~/.config/zh2en/config.toml`. A user config at
  `~/.config/zh2en/config.toml` (or `$XDG_CONFIG_HOME`) is always merged in
  first when it exists; the selected config overrides it.
- Options: `--base-url`, `--api-key`, `--model`, `--timeout`, `--max-tokens`
  override the corresponding `[api]` setting and its `TRANSLATE_*`
  environment variable. `--cache-dir` overrides the cache location
  (default `$XDG_CACHE_HOME/zh2en`). `--no-cache` bypasses the cache.
  `--ensure-paragraphs` turns on the paragraph-count check described below.
  `--verbose` prints chunking, cache, timing, and reasoning diagnostics.
  `--show-log-path` prints the run log's path to stderr at startup.
  `--version` prints the version.

Precedence: defaults < user config < selected config < environment <
command line.

See `example.toml` for a starting point.

## Logging

Every run writes a log to `<cache-dir>/logs/<timestamp>-<pid>.log`
regardless of `--no-cache`. The log records each request payload sent to
the endpoint and everything received in reply: raw SSE lines for streamed
calls, response bodies for plain calls, and any transport or protocol
errors. Request headers (and therefore API keys) are never logged. With
`--show-log-path` (or `-l`) the log's path is printed to stderr at startup.

## Configuration

```toml
[api]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-..."
model = "z-ai/glm-5.3-flash"
timeout = 120.0
max_tokens = 100000

[api.params]
stop = ["END"]

[[pass]]
name = "translate"
mode = "chunk"
instruction_file = "translate.txt"
model = "z-ai/glm-5.3-flash"
[pass.params]
reasoning_effort = "high"

[options]
ascii = true
ensure_paragraphs = true
```

- `[[pass]]` entries run in order. Each defines `name`, exactly one of
  `instruction` (inline text) or `instruction_file` (path relative to the
  config file), and `mode`:
  - `analysis`: reads the whole document and stores a preparation brief
    (outline, names, hard-to-translate items) used by later passes. The
    document must fit in one call; otherwise raise `max_tokens`.
  - `chunk`: translates paragraphs grouped into token-budgeted chunks.
  - `paragraph`: translates each paragraph with its own call.
- `model` and `params` on a pass override the API-level values per call.
- `[options] ascii = true` (or `ascii = true` on a pass) enforces pure
  ASCII output: mechanical Unicode folding first, then LLM repair with
  retries, then character dropping as a last resort.
- `[options] ensure_paragraphs = true` (or the `--ensure-paragraphs` flag)
  checks each pass's output paragraph count against the source after the
  pass runs. On a mismatch the pass is re-run with one call per paragraph,
  which preserves the source's paragraph count; a warning is printed if the
  count still differs, and the output is emitted either way.
- Every pass reply is validated before it is used: it must be non-empty,
  contain exactly as many paragraphs as the source unit it was given, and
  stay within a plausible length of that source. A failed reply is
  re-requested with corrective feedback (two repairs by default); if it
  still fails, a warning is printed, the last reply is used, and it is not
  written to the cache. Paragraph-mode calls also include the neighbouring
  source paragraphs as read-only context, which anchors short or ambiguous
  units such as title-only or ellipsis-only lines.
- Passes are cached by content hash (source text, working text, model,
  params, instruction) under the cache directory, so re-runs after
  interruption are cheap.

## Development

```sh
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

## Design notes

- `Result[T, E]` (`Ok`/`Err`) for error handling without exceptions in the
  core; `IO[T]` thunks so the whole program is a composed value that runs
  exactly once at the entry point.
- Pure text machinery (paragraph splitting, token-budget chunking, cache
  keys, SSE and `<think>` tag state machines) is fully separated from
  effects, which live in the console/status line and the injected HTTP
  opener (`Context.open_http`).

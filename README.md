# zh2en

Translate Chinese text from stdin to English on stdout using any
OpenAI-compatible chat completions endpoint (including OpenRouter).

Standard-library-only Python (3.14+), organised as a small layered
package: pure machinery in the middle, effects only at the edges.

## Install

```sh
pip install .
```

This provides the `zh2en` console script. You can also run it directly with
`python3 -m zh2en`.

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
  `--cache-prune DAYS` deletes cache entries (including orphaned
  `.txt.tmp` files) older than DAYS days and exits. `--log-keep DAYS`
  deletes run logs older than DAYS days at startup (default 30; `0`
  keeps every log). `--ensure-paragraphs` turns on the paragraph-count
  check described below.
  `--verbose` prints chunking, cache, timing, and reasoning diagnostics.
  `--show-log-path` prints the run log's path to stderr at startup.
  `--stream` / `--no-stream` force streamed or plain responses (default
  follows `api.params.stream`). `--check-config` prints the resolved
  configuration and exits without translating. `--dry-run` prints the
  per-pass call plan (units, token estimates, cache keys) and exits without
  calling the endpoint. `--version` prints the version.

Precedence: defaults < user config < selected config < environment <
command line.

Exit codes: `0` success (including empty input), `1` a pipeline error
(endpoint, budget, or pass failure), `2` a configuration or argument
error, `130` on Ctrl-C, and `141` when stdout is closed early
(SIGPIPE).

See `example.toml` for a starting point.

## Interactive verbose toggle

When stderr is a terminal, zh2en listens for key presses on the
controlling TTY while the pipeline runs (stdin stays reserved for the
input text). Pressing **Tab** toggles verbose mode: the session's output
on stderr is erased (only the rows this run printed — your scrollback is
untouched) and re-rendered for the new mode, so `--verbose` diagnostics
and LLM reasoning traces can be switched on or off mid-run. History is
kept in memory, including reasoning captured while hidden; a toggle
reveals it retroactively. Taller-than-screen history scrolls into
scrollback and cannot be erased by the re-render. The listener is off
when stderr is not a TTY (pipes, CI).

## Logging

Every run writes a log to `<cache-dir>/logs/<timestamp>-<pid>.log`
regardless of `--no-cache`. Logs older than 30 days are deleted at
startup; `--log-keep DAYS` changes the horizon (`--log-keep 0` keeps
every log). The log records each request payload sent to
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
- `ascii = true` and `ensure_paragraphs = true` may also be set on a pass;
  an explicit per-pass value overrides `[options]` and the
  `--ensure-paragraphs` flag for that pass, otherwise the global setting
  applies.
- `[options] ascii = true` (or `ascii = true` on a pass) enforces pure
  ASCII output: mechanical Unicode folding first, then LLM repair with
  retries, then character dropping as a last resort.
- `[options] ensure_paragraphs = true` (or the `--ensure-paragraphs` flag,
  or `ensure_paragraphs = true` on a pass) checks each pass's output
  paragraph count against the source after the pass runs. On a mismatch the
  pass is re-run with one call per paragraph, which preserves the source's
  paragraph count; a warning is printed if the count still differs, and the
  output is emitted either way.
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
ruff format .
ruff check .
mypy
pytest
```

`pytest` reports branch coverage per module by default
(`pytest-cov`) and enforces a 90% floor. Lint groups include `C4`,
`PERF`, `FURB`, `SIM`, `RET`, and `UP`, which nudge toward functional
idioms (PEP 695 generics, comprehensions over accumulation).

## Design notes

- `Result[T, E]` (`Ok`/`Err`) and `Maybe[T]` (`Just`/`Nothing`) for error
  handling and optional values without exceptions in the core; the error
  channel is a typed ADT (`TranslationError`) built through `fail_*`
  constructors and rendered exactly once by `describe()`. Numeric parsing
  is algebraic too (`parse_float` matches with a grammar instead of
  catching `ValueError`). `IO[T]` thunks
  keep the whole program a composed value that runs exactly once at the
  entry point; `Ref[T]` is the sanctioned single-cell mutation primitive
  and is only ever read or written through its combinators
  (`read_ref`/`write_ref`/`modify_ref`), and `Cons[T]` gives O(1)-prepend
  accumulations for stream deltas and the session event log.
- `IO.run` may appear only in `cli.py` (the program edge) and `monads.py`
  (the combinator runners: `io_atomic`, `io_using`, `repeat_until`,
  `fold_io`, ...). Every other module only composes IO values; an AST
  test (`tests/test_architecture.py`) enforces this, along with the
  strictly-downward dependency layering.
- Exception handling follows one convention: small edge adapters that ARE
  the process boundary (`urllib_open`, `load_toml`, `read_text_file`,
  `open_tty`, the `io_catch*` combinators) may catch and translate into
  the error ADT; composing modules never try/except. `http.py` maps
  transport failures (`URLError`, timeouts, `OSError`,
  `http.client.HTTPException` — including mid-body and mid-stream
  disconnects) to `HttpError` so a broken connection surfaces as a
  pipeline error, never a traceback. An SSE data frame left unterminated
  at EOF is flushed and decoded rather than dropped.
- Pure text machinery (paragraph splitting, token-budget chunking, cache
  keys, SSE and `<think>` tag state machines) is fully separated from
  effects, which live in the console/status line and the injected HTTP
  opener (`Context.open_http`). Time is injected the same way: clocks
  (`Context.clock`) and sleeps (`Context.sleep`) both come from the caller,
  as does the HTTP opener, so retries and status lines are testable.
- Unit validation repair and ASCII LLM repair share one executor:
  `http.repaired_call` runs the chat-validate-rebuild-retry loop (counting
  repairs after the initial call) while callers supply the validator,
  retry-prompt builder, and loggers.
- Module map (dependencies point downward only):

  - `monads` — `Result`, `Maybe`, `IO`, `Ref`, the immutable `Cons` list,
    and their combinators (`fold_io`, lazy `fold_while`, `fold_io_lazy`,
    `io_traverse`, `io_when`, `io_pair`, `io_memoize`,
    `modify_ref_with`); no imports from the rest of the package.
  - `messages` — pure user-facing string builders; no imports.
  - `errors` — the error ADT (`ConfigError`, `HttpError`, `BudgetError`,
    pass/unit wrappers) with `fail_*` constructors and the single
    `describe()` renderer; depends only on `monads`.
  - `text` — pure string machinery, token estimates, ASCII folding,
    cache keys, usage arithmetic.
  - `effects` — the IO vocabulary: stdin/stdout, TOML and cache files
    (including pruning), the run log (with an injected clock).
  - `console` — the status line as a `Ref[StatusView]` plus pure render
    transitions, the session event log with verbose replay, and stderr
    reporting.
  - `keys` — the Tab-key listener on the controlling TTY (cbreak mode,
    restored on exit) that toggles verbose mode. Like `console`, it is a
    frozen dataclass over `Ref`s whose effects are IO values; the program
    edge runs the composed attach IO to obtain the shutdown IO.
  - `settings` — the frozen data vocabulary (`Config`, `Settings`,
    `PassDefinition`, `Arguments`, `Context`) with defaults and small
    pure accessors.
  - `config` — TOML/config/argument parsing and merging that builds the
    data in `settings`.
  - `plans` — pure pass planning: `UnitCall`/`plan_unit_calls`, prompt
    builders, validation (`unit_output_problem`), retry policies
    (`transient`, `plan_backoff`), and the report builders
    (`plan_report`, `setup_report`); data in, data out.
  - `cache` — the cache algebra: `cache_lookup` (with an
    `acceptable` filter, so stale or empty entries are recomputed),
    `cache_store`, and `cached_translation` (read, else compute, then
    store if acceptable).
  - `http` — request payloads, reply parsing, the SSE stream fold,
    retries, and `chat`; effects are IO values, never executed inline.
  - `ascii` — ASCII enforcement: mechanical folding, LLM repair with
    retries, and per-pass output checks.
  - `pipeline` — pass execution: IO executors (`run_units`, `run_pass`)
    that run the plans produced by `plans`.
  - `cli` — argument parsing, wiring, and the single entry point that
    runs the composed program.

# l2l

IMPORTANT: This project is created using an LLM (mostly GLM 5.3 Flash)

l2l is a command-line translator: it reads text from stdin, translates
it, and writes the result to stdout, using any OpenAI-compatible chat
completions endpoint.

Your config's instruction defines the translation direction — there
are no built-in language pairs, so one tool handles Chinese to English,
Japanese to English, English to German, or any other combination.

l2l splits text into paragraphs and processes units through one or
more *passes*, each a call type with its own instruction, model, and
parameters. It validates replies and re-asks with corrective feedback
when malformed, and caches every result on disk so interrupted runs
resume cheaply.

## Install

l2l is standard-library-only Python and requires **Python 3.14 or
newer**.

```sh
uv tool install .
```

This installs the `l2l` command. To run it without installing:

```sh
python3 -m l2l
```

## Usage

```sh
l2l [options] [CONFIG] < input.txt > output.txt
```

For example:

```sh
l2l zh2en.toml < chapter1.txt > chapter1.en.txt
```

Empty input succeeds without calling the endpoint.

Getting started:

1. Copy `examples/zh2en.toml` (Chinese → English) or
   `examples/ja2en.toml` (Japanese → English) and the matching
   `.txt` instruction file.
2. Put your API key in the config — or better, in the
   `TRANSLATE_API_KEY` environment variable.
3. Run `l2l --check-config` to verify your setup, then translate.

### Command-line options

| Option | Effect |
| --- | --- |
| `CONFIG` | Optional path to a TOML config file. See [Config file resolution](#config-file-resolution). |
| `--base-url URL` | API endpoint. Overrides `[api] base_url` and `TRANSLATE_BASE_URL`. |
| `--api-key KEY` | API key. Overrides `[api] api_key` and `TRANSLATE_API_KEY`. |
| `--model MODEL` | Default model. Overrides `[api] model` and `TRANSLATE_MODEL`. |
| `--timeout SECONDS` | Per-request timeout. Overrides `[api] timeout` and `TRANSLATE_TIMEOUT`. |
| `--max-tokens N` | Per-request token budget. Overrides `[api] max_tokens` and `TRANSLATE_MAX_TOKENS`. |
| `--cache-dir DIR` | Translation cache directory (default: `$XDG_CACHE_HOME/l2l`, falling back to `~/.cache/l2l`). |
| `--no-cache` | Bypass the translation cache for this run. |
| `--cache-prune DAYS` | Delete cache entries older than DAYS days and exit. |
| `--log-keep DAYS` | Delete run logs older than DAYS days at startup (default 30; `0` keeps every log). |
| `--ensure-paragraphs` | Enforce the paragraph count after each pass (strict). Equivalent to `ensure_paragraphs = true`; see `[options]` below. |
| `--verbose`, `-v` | Print chunking, cache, timing, and reasoning diagnostics to stderr. |
| `--show-log-path`, `-l` | Print the run log's path to stderr at startup. |
| `--stream` / `--no-stream` | Force streamed or plain responses. Default follows `api.params.stream`. |
| `--check-config` | Print the resolved configuration and exit without translating. |
| `--dry-run` | Print the per-pass call plan (units, token estimates, cache keys) and exit without calling the endpoint. |
| `--version` | Print the version and exit. |

Precedence, from weakest to strongest:

```
defaults < user config < selected config < environment < command line
```

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success (including empty input). |
| `1` | Pipeline error (endpoint, budget, or pass failure). |
| `2` | Configuration or argument error. |
| `130` | Interrupted with Ctrl-C. |
| `141` | stdout closed early (SIGPIPE), e.g. piping into `head`. |

### Interactive verbose toggle

When stderr is a terminal, press **Tab** during a run to toggle verbose
output. l2l re-renders the session's stderr for the new mode and
reveals reasoning captured while hidden. stdin stays reserved for the
input text; the listener is off when stderr is not a terminal (pipes,
CI).

### Logging

Every run writes a log to `<cache-dir>/logs/<timestamp>-<pid>.log`,
even with `--no-cache`. The log records each request payload and
everything received in reply: raw SSE lines for streamed calls,
response bodies for plain calls, and any transport or protocol errors.
Request headers — and therefore API keys — are never logged. Pass
`--show-log-path` (or `-l`) to print the log's path at startup, e.g.
when attaching a log to a bug report.

## Configuration

Configuration lives in a TOML file with up to three kinds of table:

```toml
[api]          # endpoint, key, model, limits, extra request parameters
[[pass]]       # one or more passes, run in order
[options]      # global toggles
```

A complete example (Chinese fiction to English — see `examples/` for
working copies):

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

### Config file resolution

l2l selects the config providing your passes in this order:

1. The positional `CONFIG` argument, if given.
2. The `$TRANSLATE_CONFIG` environment variable.
3. `./l2l.toml` in the current directory.
4. `~/.config/l2l/config.toml` (or `$XDG_CONFIG_HOME/l2l/config.toml`).

Separately, l2l always merges the user config at
`~/.config/l2l/config.toml` (or `$XDG_CONFIG_HOME/l2l/config.toml`) in
as a base layer when it exists; the selected config overrides it. Put
your `base_url` and `api_key` there, so per-project configs only need
the passes.

### `[api]` — endpoint settings

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `base_url` | string | — | Chat completions endpoint, e.g. `https://openrouter.ai/api/v1`. A trailing `/` is stripped. |
| `api_key` | string | — | Sent as `Authorization: Bearer …`. |
| `model` | string | — | Default model for calls (can be overridden per pass). |
| `timeout` | number | `120.0` | Request timeout in seconds. |
| `max_tokens` | integer | `100000` | Token budget for requests. |
| `params` | table | `{}` | Extra keys merged into every request body, e.g. `stop = ["END"]` or provider routing options. Must contain only JSON values. |

`base_url`, `api_key`, and `model` have no built-in default but only
need to be set in one place: this table, the user config, the
environment, or the matching command-line option, which always wins
(the table above names each setting's environment variable).

### `[[pass]]` — pipeline passes

`[[pass]]` entries run in order; each pass's output is the next's
input. Each pass requires:

| Key | Meaning |
| --- | --- |
| `name` | Non-empty string, used in logs and cache keys. |
| `instruction` *or* `instruction_file` | Exactly one: the instruction as inline text, or a path to a text file (relative to the config file). The instruction defines the language pair and target style. |

Optional per-pass keys:

| Key | Default | Meaning |
| --- | --- | --- |
| `mode` | `"chunk"` | How the document is processed (see below). |
| `model` | `[api] model` | Model override for this pass's calls. |
| `params` | `{}` | Extra request-body keys for this pass's calls, merged over `[api] params`. |
| `ascii` | `[options] ascii` | Per-pass ASCII enforcement override. |
| `ensure_paragraphs` | `[options] ensure_paragraphs` | Per-pass paragraph-count check override: `true` (strict), `false` (off), or a positive integer tolerance. |
| `retranslate_untranslated` | `[options] retranslate_untranslated` | Per-pass untranslated-paragraph check override (see below). |

Modes:

- **`chunk`** — translates paragraphs grouped into token-budgeted
  chunks. The default, and cheapest for long documents.
- **`paragraph`** — translates each paragraph with its own call,
  including neighbouring source paragraphs as read-only context to
  anchor short or ambiguous units (title-only lines, ellipses, …).
- **`analysis`** — reads the whole document and stores a preparation
  brief (outline, names, hard-to-translate items) that later passes
  include for context. The document must fit in one call; otherwise
  increase `max_tokens`.

A typical multi-pass setup pairs an `analysis` pass with a `chunk`
pass.

### `[options]` — global toggles

| Key | Default | Meaning |
| --- | --- | --- |
| `ascii` | `false` | Enforce pure ASCII output. Assumes a Latin-script target language; leave it off for targets such as Russian, Greek, Japanese, or Chinese. |
| `ensure_paragraphs` | `false` | Paragraph-count enforcement. `true` requires each pass's output to match the source paragraph count exactly, re-running a mismatching chunk-mode pass with one call per paragraph. A positive integer sets a tolerance: `ensure_paragraphs = 2` accepts outputs within ±2 paragraphs of the source and only re-runs beyond that. `false` disables the check entirely. While a pass runs, each chunk reply is also rejected (and re-asked with corrective feedback) when its paragraph count drifts beyond the tolerance; `paragraph`-mode passes always require exactly one paragraph per call. |
| `retranslate_untranslated` | `false` | After each pass, compare every output paragraph against its source paragraph (by index). A paragraph counts as untranslated when it matches the source after mechanical ASCII folding (echoes often differ only in punctuation and whitespace) or when it contains no ASCII letters at all; sources without letters (rules, numbers) are never flagged. Flagged paragraphs are re-asked with one call each, using the pass's own instruction plus neighbouring source paragraphs as context, and retranslation results are cached like any unit. A paragraph that is still untranslated after retranslation fails the run with exit code 1. |

All toggles can be overridden per pass (`ascii` / `ensure_paragraphs` /
`retranslate_untranslated` on a `[[pass]]` entry).


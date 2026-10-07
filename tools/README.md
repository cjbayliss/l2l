# tools

Development tools for l2l. Each script runs directly rather than being
installed, and composes the `l2l` package it ships with:

```sh
uv run python3 tools/optimize.py ...
```

- `optimize.py` — evolve a translation instruction through rounds of
  pairwise judged challenges, or compare two instruction files
  directly. Written in the same strict functional style as the
  package: effects are `IO` values, errors are `Result`s, and all data
  is immutable, so `tests/test_optimize.py` can drive whole runs with
  fake clocks, endpoints, and l2l subprocesses.
- `judge.txt` — prompt template for the judge calls (`optimize.py`'s
  default `--judge-template`).
- `rewrite.txt` — prompt template for the rewrite calls (`optimize.py`'s
  default `--rewrite-template`).

## optimize.py

`optimize.py` improves an l2l instruction file automatically. Each
round a *rewriter* model proposes a challenger instruction from the
judges' critiques of the previous round. The incumbent and the
challenger both translate every chapter in the workdir, and a *judge*
model scores the two translations in a double-blind pairing: every
judgment runs twice with the A/B sides swapped, so position cannot
decide the outcome. A challenger that wins a majority of chapters
becomes the incumbent; the loop stops early after `--stall` rounds
without a promotion, and the survivor is finally judged against the
seed on a held-out chapter.

With `--compare`, the rewrite loop is skipped and two given instruction
files are pit against each other directly over the chapters.

### Getting started

1. Create a workdir with a `chapters/` directory holding at least two
   source `.txt` files plus `holdout.txt`. The holdout stays out of the
   evolution rounds and is translated once, at the end.
2. Copy a `docs/configs/*.toml` config as the base config. It supplies the
   `[api]` table and the pass pipeline; the pass whose instruction
   evolves keeps its other settings, while its instruction is replaced
   per version.
3. Run:

```sh
python3 tools/optimize.py \
    --base-config zh2en.toml \
    --workdir exp-zh2en \
    --seed translate.txt \
    --judge-model openai/gpt-5 \
    --rewriter-model z-ai/glm-5.3-flash
```

`optimize.py` drives l2l as a subprocess, using `<python> -m l2l` from
the repository root by default; pass `--l2l "l2l"` to use an installed
`l2l` command instead. It requires the same Python 3.14 or newer as
l2l and needs the `l2l` package importable (run it through `uv run`
from the repository root, or install the project). Its endpoint calls
go through l2l's own HTTP transport, so proxy environment variables
behave exactly as they do for l2l.

Endpoint credentials resolve like l2l's, weakest to strongest:

```
user config < base config [api] < environment < command line
```

That is: `~/.config/l2l/config.toml`, then the base config's `[api]`
table, then `TRANSLATE_BASE_URL`/`TRANSLATE_API_KEY`, then
`--base-url`/`--api-key`.

### Command-line options

| Option | Effect |
| --- | --- |
| `--base-config PATH` | Required. l2l config supplying the `[api]` table and the pass pipeline. |
| `--workdir DIR` | Required. Experiment directory holding `chapters/`, `prompts/`, `gen/`, `out/`, `judge/`, and `cache/`. Created if missing. |
| `--seed PATH` | Instruction file copied to `prompts/v0.txt` when absent; required once per workdir. |
| `--rounds N` | Maximum number of evolution rounds (default 6). |
| `--stall N` | Stop after N consecutive rounds without a promotion (default 2). |
| `--judge-model MODEL` | Required. Model judging each pairing. |
| `--rewriter-model MODEL` | Model proposing challenger instructions. Required unless `--compare` is given. |
| `--translator-model MODEL` | Model override for the translated pass (default: the pass's configured model). |
| `--pass-name NAME` | Which pass's instruction to evolve. Needed when the base config has multiple passes. |
| `--compare PATH_A PATH_B` | Compare mode: pit two instruction files against each other instead of evolving. |
| `--holdout` | With `--compare`, judge only on `chapters/holdout.txt`. |
| `--chapters LIST` | With `--compare`, comma-separated chapter stems to judge, e.g. `ch1,ch3`. |
| `--base-url URL`, `--api-key KEY` | Endpoint and bearer token, overriding the environment and configs. |
| `--l2l COMMAND` | Command used to run l2l (default: `<python> -m l2l`). |
| `--judge-template PATH` | Judge prompt template (default: `tools/judge.txt`). |
| `--rewrite-template PATH` | Rewrite prompt template (default: `tools/rewrite.txt`). |
| `--judge-temperature F` | Sampling temperature for judge calls (default 0.0). |
| `--rewrite-temperature F` | Sampling temperature for rewrite calls (default 0.8). |
| `--translator-temperature F` | Temperature forced into the translated pass's params (default 0.0). |
| `--call-timeout SECONDS` | Per-request timeout (default 600). |
| `--call-max-tokens N` | Per-request token budget (default 32768). |
| `--judge-params JSON` | JSON object merged into judge request bodies, e.g. `'{"reasoning": {"effort": "low"}}'` for thinking models. |
| `--rewrite-params JSON` | JSON object merged into rewrite request bodies. |
| `--history-depth N` | How many recent history entries the rewriter sees (default 6). |

### The evolution loop

Each round:

1. The rewriter receives the current instruction, the recent round
   history, and the previous round's judge critiques, and replies with
   a new instruction. A challenger already on disk in `prompts/` is
   reused rather than rewritten.
2. A challenger whose text duplicates any earlier instruction is
   recorded as a duplicate and skipped without judging.
3. Both versions translate every chapter through l2l; finished
   translations are never recomputed, and l2l's own cache absorbs
   re-runs.
4. Every pairing is judged twice with the sides swapped; a chapter
   counts for the challenger only when it wins both orders. Ties and
   splits count for nobody.
5. The challenger needs a majority of chapters to be promoted. The
   ledger and history are appended either way, and the judges'
   critiques feed the next round's rewrite.

When the loop ends, the surviving incumbent is judged against the seed
on `holdout.txt` and the verdict is written to `judge/holdout.json`;
the check is skipped when nothing was ever promoted.

### Compare mode

```sh
python3 tools/optimize.py \
    --base-config zh2en.toml \
    --workdir exp-zh2en \
    --judge-model openai/gpt-5 \
    --compare prompts/v2.txt prompts/v4.txt
```

Chapters default to every file in `chapters/`; `--holdout` restricts
the run to `chapters/holdout.txt`, and `--chapters ch1,ch3` to the
named stems. Identical instruction files are rejected up front.
Verdicts land in `judge/compare/<hash-a>-vs-<hash-b>/`, one JSON file
per chapter plus a `summary.json` naming the overall winner.

### Workdir layout

| Path | Contents |
| --- | --- |
| `chapters/` | Your source `.txt` files (at least two) plus `holdout.txt`. |
| `prompts/` | `v0.txt` (the seed) and one `vN.txt` per challenger. |
| `gen/` | A generated l2l config per version: the base config with the target pass's instruction swapped and a pinned temperature. |
| `out/` | Translations, `out/<version>/<chapter>.en.txt`; failed runs keep their stderr next to the output as `.err`. |
| `judge/` | Per-round verdict JSON, per-round feedback text, `holdout.json`, and `compare/` results. |
| `cache/` | The l2l translation cache used for every run. |
| `ledger.jsonl` | One line per round: win counts and the decision (`promoted`, `kept`, or `duplicate`). |
| `history.jsonl` | One line per round, including the challenger's full text. |

Everything is resumable: a restart replays the ledger to recover the
incumbent and the stall count, reuses on-disk challenger prompts and
translations, and continues from the next round.

### Prompt templates

`judge.txt` is filled with `<<<SOURCE>>>`, `<<<TRANSLATION_A>>>`, and
`<<<TRANSLATION_B>>>` and must reply with a JSON object of the shape:

```json
{"winner": "A", "critique": "...", "evidence": ["..."]}
```

where `winner` is `A`, `B`, or `tie`. A judge reply without a usable
verdict is retried twice and then recorded as an `error` verdict rather
than aborting the run.

`rewrite.txt` is filled with `<<<CURRENT_PROMPT>>>`, `<<<HISTORY>>>`,
and `<<<CRITIQUES>>>` and must reply with the raw text of the new
instruction — no fences, title, or commentary.

Both templates can be swapped with `--judge-template` and
`--rewrite-template` as long as they keep their tokens; missing tokens
are rejected at startup.

### Endpoint calls and failures

An evolution round makes at most `1 + 3×chapters` endpoint calls: one
rewrite, up to one translation per chapter (the incumbent's are usually
cached after the first round), and two judgments per chapter. Compare
mode makes at most four calls per chapter: two translations and two
judgments. Each request retries up to four times with a 2s/4s/6s
backoff; a call that exhausts its retries aborts the run with the
endpoint error, and a failed l2l subprocess aborts with its stderr
preserved under `out/` as `.err` for inspection (successful runs leave
no `.err` behind).

Thinking models that spend the whole budget on reasoning return empty
content; raise `--call-max-tokens` or cap the reasoning, e.g.
`--judge-params '{"reasoning": {"effort": "low"}}'`.

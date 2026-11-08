# l2l

**NOTE:** This project was created with an LLM (mostly GLM 5.3 Flash,
Kimi K3, and Deepseek V4.1 Flash 1017).

l2l is a command-line translation tool for long-form text. It reads a
document from standard input, translates it with any OpenAI-compatible
chat completions endpoint, and writes the translation to standard
output.

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Configuration](#configuration)

## Installation

```sh
uv tool install git+https://github.com/cjbayliss/l2l
```

## Usage

```sh
l2l -c config.toml < chapter1.txt > chapter1.out.txt
```

Run `l2l --help` for more information.

## Requirements

- Python 3.15 or newer
- [uv](https://docs.astral.sh/uv/) for the install and test commands
  below
- libcurl: l2l uses `pycurl` for HTTP transport; on platforms without a
  `pycurl` wheel, install the libcurl development headers first so that
  pip can build it
- An API key for any OpenAI-compatible chat completions endpoint

## Configuration

l2l is configured with a TOML file. At minimum you need an `[api]` table
and one `[pass.<name>]` table:

```toml
target_language = "English"
source_language = "Japanese"

[api]
base_url = "https://openrouter.ai/api/v1"
api_key = "sk-..." # or set TRANSLATE_API_KEY
model = "z-ai/glm-5.3-flash"

[pass.translate]
```

Save it to `$XDG_CONFIG_HOME/l2l/config.toml`, or pass it explicitly
with `-c config.toml`. Multiple translation passes run in the order they
are defined.

### Top-level options

| Key               | Type                       | Description                                                                 |
| ----------------- | -------------------------- | --------------------------------------------------------------------------- |
| `target_language` | string                     | Target language name, for example `"English"`. Required.                    |
| `source_language` | string                     | Source language name. Defaults to auto-detection.                           |
| `glossary`        | list of strings            | Paths to glossary JSON files, relative to the config file.                  |
| `api`             | table                      | API connection settings; see the `[api]` section.                           |
| `options`         | table                      | Defaults applied to every pass; see the `[options]` section.                |
| `filter`          | table                      | Deterministic source preprocessing; see the `[filter]` section.             |
| `prompts`         | table                      | Global prompt section overrides; see the `[prompts]` section.               |
| `style`           | table                      | Global style policy; see the `[style]` section.                             |
| `output`          | table                      | Global output policy; see the `[output]` section.                           |
| `pass`            | table of named pass tables | Translation passes, run in document order; see the `[pass.<name>]` section. |

### `[api]`

| Key          | Type    | Default  | Description                                                                                                    |
| ------------ | ------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| `base_url`   | string  | —        | Root URL of an OpenAI-compatible chat completions endpoint. Required; alternatively, set `TRANSLATE_BASE_URL`. |
| `api_key`    | string  | —        | API key. Required; alternatively, set `TRANSLATE_API_KEY`.                                                     |
| `model`      | string  | —        | Default model for passes that do not set their own. Required; alternatively, set `TRANSLATE_MODEL`.            |
| `timeout`    | number  | `120`    | Request timeout in seconds (`TRANSLATE_TIMEOUT`).                                                              |
| `max_tokens` | integer | `100000` | Maximum tokens per request (`TRANSLATE_MAX_TOKENS`).                                                           |
| `params`     | table   | `{}`     | Extra JSON values merged into every request body, for example `stop = ["END"]`.                                |

Environment variables override config files.

### `[options]`

Defaults for every pass; each pass can override them.

| Key                        | Type                        | Default | Description                                                                                           |
| -------------------------- | --------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `ascii`                    | boolean                     | `false` | Restrict output to ASCII. The target language must use a Latin script.                                |
| `ensure_paragraphs`        | boolean or positive integer | `false` | Ask the model to preserve paragraph structure; an integer sets the allowed paragraph-count tolerance. |
| `retranslate_untranslated` | boolean                     | `false` | Re-translate chunks the model left untranslated instead of keeping them.                              |

### `[filter]`

Deterministic preprocessing applied before the passes run; the filtered
text becomes the source text for every pass. At least one key is
required.

| Key       | Type                       | Default | Description                                                                                                                                                                 |
| --------- | -------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `script`  | string or string array     | —       | External filter command (prefer the array form; a string is split shell-style, never run through a shell), given UTF-8 source on stdin, returning filtered UTF-8 on stdout. |
| `filters` | array of strings or tables | —       | Built-in paragraph filters, applied to the script output in list order.                                                                                                     |

- `"duplicate_paragraphs"`: removes repeated paragraphs after
  normalizing whitespace, keeping the first occurrence.
- `{ kind = "pattern", pattern = "..." }`: removes paragraphs matching
  the regular expression.
- `{ kind = "short_paragraphs", minimum_characters = 4 }`: removes
  paragraphs shorter than `minimum_characters`, measured after
  whitespace normalization.

```toml
[filter]
script = ["python3", "scripts/clean_chapter.py"]
filters = [
  "duplicate_paragraphs",
  { kind = "pattern", pattern = '下一章|advertisement' },
  { kind = "short_paragraphs", minimum_characters = 4 },
]
```

### `[pass.<name>]`

Example:

```toml
[pass.glossary]
mode = "update_glossary"

[pass.translate]
```

| Key                        | Type                        | Default       | Description                                                                                                                                                                                   |
| -------------------------- | --------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mode`                     | string                      | `"translate"` | One of `analysis` (extract guidance from the source), `translate` (translate the text), or `update_glossary` (extract or refresh glossary entries).                                           |
| `paragraphs_per_call`      | non-negative integer        | `0`           | For `translate` passes: number of paragraphs sent per model call. `0` groups paragraphs by token budget; `N >= 1` sends N paragraphs per call, splitting oversized batches to fit the budget. |
| `instruction`              | string                      | —             | Inline instruction replacing the built-in prompt. Exactly one of `instruction` / `instruction_file`; optional for built-in modes.                                                             |
| `instruction_file`         | string                      | —             | Path to an instruction file, relative to the config file.                                                                                                                                     |
| `model`                    | string                      | `[api] model` | Model override for this pass.                                                                                                                                                                 |
| `params`                   | table                       | `{}`          | Extra request-body values for this pass.                                                                                                                                                      |
| `ascii`                    | boolean                     | `[options]`   | Per-pass ASCII restriction.                                                                                                                                                                   |
| `ensure_paragraphs`        | boolean or positive integer | `[options]`   | Per-pass paragraph preservation.                                                                                                                                                              |
| `retranslate_untranslated` | boolean                     | `[options]`   | Per-pass retranslation of untranslated chunks.                                                                                                                                                |
| `target_language`          | string                      | global        | Per-pass target language override.                                                                                                                                                            |
| `include_types`            | list of strings             | all           | For `update_glossary` passes: entry types to include, from `character`, `place`, `other`.                                                                                                     |
| `prompts`                  | table                       | —             | Per-pass prompt section overrides; built-in rendered `translate`/`analysis` passes only.                                                                                                      |
| `style`                    | table                       | —             | Per-pass style policy; built-in rendered `translate` passes only.                                                                                                                             |

### `[prompts]`

Overrides individual sections of the built-in prompts. Each section is
set to inline text, or `{ file = "path" }` to load it from a file
relative to the config file. Setting a section to `""` removes it.
Available sections: `role`, `task`, `languages`, `source_guidance`,
`target_guidance`, `accuracy`, `fluency`, `style`, `context_handling`,
`output_rules`.

```toml
[prompts]
role = "You are a translator of Victorian novels."
style = { file = "style-section.txt" }
```

### `[style]`

| Key                 | Values                           | Default    | Description                                              |
| ------------------- | -------------------------------- | ---------- | -------------------------------------------------------- |
| `formality`         | `preserve`, `formal`, `informal` | `preserve` | Formality of the translation.                            |
| `honorifics`        | `keep`, `drop`, `titles`         | `keep`     | Honorific handling for Japanese and Korean sources only. |
| `register_fidelity` | `preserve`, `soften`             | `preserve` | Whether to keep or soften coarse language.               |

### `[output]`

| Key                  | Type                         | Default   | Description                                                            |
| -------------------- | ---------------------------- | --------- | ---------------------------------------------------------------------- |
| `paragraphs`         | `"exact"` or `"approximate"` | `"exact"` | Whether the output must have exactly as many paragraphs as the source. |
| `maximum_characters` | positive integer             | —         | Cap on the length of each output chunk.                                |

### Glossary files

Glossaries are JSON files listing terms to translate consistently. Each
entry is typed as `character`, `place`, or `other`. Paths are relative
to the config file; more can be added with `--glossary`.

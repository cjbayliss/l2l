# l2l

**NOTE:** This project was created with an LLM (mostly GLM 5.3 Flash,
Kimi K3, and Deepseek V4.1 Flash 1017).

l2l is a command-line translation tool for long-form text. It reads a
document from standard input, translates it with any OpenAI-compatible
chat completions endpoint, and writes the translation to standard
output.

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

For a detailed example of the options, see `docs/example-config.toml`.

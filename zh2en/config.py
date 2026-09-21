from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any, Literal

from zh2en.console import Console
from zh2en.effects import (
    Clock,
    RunLog,
    cwd,
    load_toml,
    path_exists,
    user_config_path,
)
from zh2en.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_sequence,
    result_bind,
    result_bind_io,
    result_map,
    results_sequence,
)
from zh2en.text import CACHE_SALT_VERSION


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    params: Mapping[str, Any]


PassMode = Literal["analysis", "chunk", "paragraph"]


@dataclass(frozen=True)
class PassDefinition:
    name: str
    instruction: str
    mode: PassMode
    params: Mapping[str, Any]
    model: str | None
    ascii: bool | None


@dataclass(frozen=True)
class Settings:
    chunk_budget_tokens: int
    analysis_reserve_tokens: int
    ascii_fix_attempts: int
    sentence_boundary_characters: str
    ascii_fix_instruction: str
    ascii_character_map: Mapping[str, str]
    unit_fix_attempts: int
    unit_output_max_ratio: float


@dataclass(frozen=True)
class Arguments:
    config: str | None
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout: float | None
    max_tokens: int | None
    no_cache: bool
    ensure_paragraphs: bool
    verbose: bool
    show_log_path: bool
    cache_dir: str | None


@dataclass(frozen=True)
class Setup:
    config: Config
    passes: tuple[PassDefinition, ...]
    ensure_paragraphs: bool


OpenHTTP = Callable[[Any, float], Result[Any, str]]


@dataclass(frozen=True)
class Context:
    config: Config
    settings: Settings
    use_cache: bool
    cache_directory: str
    verbose: bool
    ensure_paragraphs: bool
    console: Console
    open_http: OpenHTTP
    log: RunLog
    clock: Clock


def build_settings() -> Settings:
    return Settings(
        chunk_budget_tokens=3500,
        analysis_reserve_tokens=128,
        ascii_fix_attempts=3,
        unit_fix_attempts=2,
        unit_output_max_ratio=6.0,
        sentence_boundary_characters="。！？!?；;\n",
        ascii_fix_instruction="""
This paragraph failed to be fully translated or contains non-ASCII
characters. Please analyse it and only output a clean translation
without any non-ASCII.
""",
        ascii_character_map=MappingProxyType(
            {
                "\u00a0": " ",
                "\u2007": " ",
                "\u2009": " ",
                "\u202f": " ",
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": ",",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
                "\u2032": "'",
                "\u2033": '"',
                "\u2039": "'",
                "\u203a": "'",
                "\u00ab": '"',
                "\u00bb": '"',
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
                "\u2026": "...",
                "\u2025": "..",
                "\u2022": "*",
                "\u2023": "*",
                "\u2024": "*",
                "\u2219": "*",
                "\u00b7": " ",
                "\u2190": "<-",
                "\u2192": "->",
                "\u2194": "<->",
                "\u2264": "<=",
                "\u2265": ">=",
                "\u2260": "!=",
                "\u3002": ".",
                "\u3001": ",",
                "\u300c": '"',
                "\u300d": '"',
                "\u300e": '"',
                "\u300f": '"',
                "\u3008": "<",
                "\u3009": ">",
                "\u300a": '"',
                "\u300b": '"',
                "\u3010": "[",
                "\u3011": "]",
            }
        ),
    )


API_SETTING_KEYS = ("base_url", "api_key", "model", "timeout", "max_tokens", "params")
PASS_KEYS = (
    "name",
    "instruction",
    "instruction_file",
    "mode",
    "ascii",
    "model",
    "params",
)


DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 100000


@dataclass(frozen=True)
class PartialApiSettings:
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float | None = None
    max_tokens: int | None = None
    params: Mapping[str, Any] | None = None

    def merge(self, extra: PartialApiSettings) -> PartialApiSettings:
        if self.params is not None and extra.params is not None:
            params: Mapping[str, Any] | None = {**self.params, **extra.params}
        elif extra.params is None:
            params = self.params
        else:
            params = extra.params

        return PartialApiSettings(
            base_url=extra.base_url if extra.base_url is not None else self.base_url,
            api_key=extra.api_key if extra.api_key is not None else self.api_key,
            model=extra.model if extra.model is not None else self.model,
            timeout=extra.timeout if extra.timeout is not None else self.timeout,
            max_tokens=extra.max_tokens
            if extra.max_tokens is not None
            else self.max_tokens,
            params=params,
        )


DEFAULT_API_SETTINGS = PartialApiSettings(
    base_url="",
    api_key="",
    model="",
    timeout=DEFAULT_TIMEOUT,
    max_tokens=DEFAULT_MAX_TOKENS,
    params=MappingProxyType({}),
)


def validate_document(path: str, document: dict[str, Any]) -> Result[None, str]:
    if not document:
        return Ok(None)

    unknown = sorted(set(document) - {"api", "options", "pass"})
    if unknown:
        return Err(
            "%s: unknown top-level key(s): %s (expected [api], [options], [[pass]])"
            % (path, ", ".join(unknown))
        )

    return Ok(None)


def string_api_settings(
    path: str, table: Mapping[str, Any]
) -> Result[PartialApiSettings, str]:
    def updated(
        partial: PartialApiSettings, key: str, value: str
    ) -> PartialApiSettings:
        if key == "base_url":
            return replace(partial, base_url=value)

        if key == "api_key":
            return replace(partial, api_key=value)

        return replace(partial, model=value)

    def add(partial: PartialApiSettings, key: str) -> Result[PartialApiSettings, str]:
        if key not in table:
            return Ok(partial)

        value = table[key]
        if not isinstance(value, str) or not value.strip():
            return Err("%s: [api] %s must be a non-empty string" % (path, key))

        return Ok(updated(partial, key, value.strip()))

    def step(
        partial_result: Result[PartialApiSettings, str], key: str
    ) -> Result[PartialApiSettings, str]:
        return result_bind(partial_result, lambda partial: add(partial, key))

    initial: Result[PartialApiSettings, str] = Ok(PartialApiSettings())
    return reduce(step, ("base_url", "api_key", "model"), initial)


def timeout_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, str]:
    if "timeout" not in table:
        return Ok(partial)

    value = table["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return Err("%s: [api] timeout must be a positive number" % path)

    return Ok(replace(partial, timeout=float(value)))


def max_tokens_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, str]:
    if "max_tokens" not in table:
        return Ok(partial)

    value = table["max_tokens"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return Err("%s: [api] max_tokens must be a positive integer" % path)

    return Ok(replace(partial, max_tokens=value))


def params_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, str]:
    if "params" not in table:
        return Ok(partial)

    value = table["params"]
    if not isinstance(value, dict):
        return Err("%s: [api] params must be a table" % path)

    return Ok(replace(partial, params=value))


def document_api_settings(
    path: str, document: dict[str, Any]
) -> Result[PartialApiSettings, str]:
    if "api" not in document:
        return Ok(PartialApiSettings())

    table = document["api"]
    if not isinstance(table, dict):
        return Err("%s: [api] must be a table" % path)

    unknown = sorted(set(table) - set(API_SETTING_KEYS))
    if unknown:
        return Err("%s: [api]: unknown key(s): %s" % (path, ", ".join(unknown)))

    return result_bind(
        string_api_settings(path, table),
        lambda partial: result_bind(
            timeout_api_setting(path, partial, table),
            lambda partial: result_bind(
                max_tokens_api_setting(path, partial, table),
                lambda partial: params_api_setting(path, partial, table),
            ),
        ),
    )


def merge_api_settings(
    base: PartialApiSettings, extra: PartialApiSettings
) -> PartialApiSettings:
    return base.merge(extra)


def env_string_settings(environment: Mapping[str, str]) -> PartialApiSettings:
    def value(variable: str) -> str | None:
        raw = environment.get(variable, "").strip()
        return raw or None

    return PartialApiSettings(
        base_url=value("TRANSLATE_BASE_URL"),
        api_key=value("TRANSLATE_API_KEY"),
        model=value("TRANSLATE_MODEL"),
    )


def env_timeout_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, str]:
    raw = environment.get("TRANSLATE_TIMEOUT", "").strip()
    if not raw:
        return Ok(partial)

    try:
        timeout = float(raw)
    except ValueError:
        return Err("TRANSLATE_TIMEOUT must be a number, got %r" % raw)

    if timeout <= 0:
        return Err("TRANSLATE_TIMEOUT must be positive")

    return Ok(replace(partial, timeout=timeout))


def env_max_tokens_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, str]:
    raw = environment.get("TRANSLATE_MAX_TOKENS", "").strip()
    if not raw:
        return Ok(partial)

    try:
        max_tokens = int(raw)
    except ValueError:
        return Err("TRANSLATE_MAX_TOKENS must be an integer, got %r" % raw)

    if max_tokens <= 0:
        return Err("TRANSLATE_MAX_TOKENS must be positive")

    return Ok(replace(partial, max_tokens=max_tokens))


def api_settings_from_environment(
    environment: Mapping[str, str],
) -> Result[PartialApiSettings, str]:
    return result_bind(
        env_timeout_setting(environment, env_string_settings(environment)),
        lambda partial: env_max_tokens_setting(environment, partial),
    )


def api_settings_from_arguments(arguments: Arguments) -> PartialApiSettings:
    return PartialApiSettings(
        base_url=arguments.base_url,
        api_key=arguments.api_key,
        model=arguments.model,
        timeout=arguments.timeout,
        max_tokens=arguments.max_tokens,
    )


def missing_api_settings(partial: PartialApiSettings) -> tuple[str, ...]:
    return tuple(
        description
        for value, description in (
            (partial.base_url, "api.base_url (--base-url / TRANSLATE_BASE_URL)"),
            (partial.api_key, "api.api_key (--api-key / TRANSLATE_API_KEY)"),
            (partial.model, "api.model (--model / TRANSLATE_MODEL)"),
        )
        if not value
    )


def build_config(partial: PartialApiSettings) -> Result[Config, str]:
    missing = missing_api_settings(partial)
    if missing:
        return Err("missing required API settings: " + ", ".join(missing))

    return Ok(
        Config(
            base_url=str(partial.base_url).rstrip("/"),
            api_key=str(partial.api_key),
            model=str(partial.model),
            timeout=partial.timeout
            if partial.timeout is not None
            else DEFAULT_TIMEOUT,
            max_tokens=partial.max_tokens
            if partial.max_tokens is not None
            else DEFAULT_MAX_TOKENS,
            params=partial.params
            if partial.params is not None
            else MappingProxyType({}),
        )
    )


def merged_api_settings(
    arguments: Arguments,
    environment: Mapping[str, str],
    user_path: str,
    user_document_result: Result[dict[str, Any], str],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], str],
) -> Result[Config, str]:
    def merge_document(
        path: str | None,
        document_result: Result[dict[str, Any], str],
        partial: PartialApiSettings,
    ) -> Result[PartialApiSettings, str]:
        return result_bind(
            document_result,
            lambda document: result_map(
                document_api_settings(path or "", document),
                lambda extra: merge_api_settings(partial, extra),
            ),
        )

    layer: Result[PartialApiSettings, str] = Ok(DEFAULT_API_SETTINGS)
    layer = result_bind(
        layer,
        lambda partial: merge_document(user_path, user_document_result, partial),
    )
    layer = result_bind(
        layer,
        lambda partial: merge_document(
            selected_path, selected_document_result, partial
        ),
    )
    layer = result_bind(
        layer,
        lambda partial: result_map(
            api_settings_from_environment(environment),
            lambda extra: merge_api_settings(partial, extra),
        ),
    )
    layer = result_map(
        layer,
        lambda partial: merge_api_settings(
            partial, api_settings_from_arguments(arguments)
        ),
    )
    return result_bind(layer, build_config)


def parse_options_table(path: str, table: Any) -> Result[dict[str, bool], str]:
    if not isinstance(table, dict):
        return Err("%s: [options] must be a table" % path)

    unknown = sorted(set(table) - {"ascii", "ensure_paragraphs"})
    if unknown:
        return Err("%s: [options]: unknown key(s): %s" % (path, ", ".join(unknown)))

    ascii_value = table.get("ascii", False)
    if not isinstance(ascii_value, bool):
        return Err("%s: [options] ascii must be true or false" % path)

    ensure_paragraphs = table.get("ensure_paragraphs", False)
    if not isinstance(ensure_paragraphs, bool):
        return Err("%s: [options] ensure_paragraphs must be true or false" % path)

    return Ok({"ascii": ascii_value, "ensure_paragraphs": ensure_paragraphs})


def pass_definition_from(
    path: str, name: str, table: dict[str, Any], instruction: str
) -> Result[PassDefinition, str]:
    mode = table.get("mode", "chunk")
    if mode not in ("analysis", "chunk", "paragraph"):
        return Err(
            "%s: [[pass]] %s: mode must be analysis, chunk, or paragraph" % (path, name)
        )

    ascii_value = table.get("ascii")
    if ascii_value is not None and not isinstance(ascii_value, bool):
        return Err("%s: [[pass]] %s: ascii must be true or false" % (path, name))

    model = table.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return Err("%s: [[pass]] %s: model must be a non-empty string" % (path, name))

    params = table.get("params", {})
    if not isinstance(params, dict):
        return Err("%s: [[pass]] %s: params must be a table" % (path, name))

    return Ok(
        PassDefinition(
            name=name,
            instruction=instruction,
            mode=mode,
            params=MappingProxyType(dict(params)),
            model=model.strip() if model else None,
            ascii=ascii_value,
        )
    )


def apply_default_ascii(
    pass_definitions: tuple[PassDefinition, ...], options: Mapping[str, bool]
) -> tuple[PassDefinition, ...]:
    return tuple(
        (
            replace(pass_definition, ascii=options.get("ascii", False))
            if pass_definition.ascii is None
            else pass_definition
        )
        for pass_definition in pass_definitions
    )


def resolve_call_settings(
    config: Config, pass_definition: PassDefinition
) -> tuple[str, dict[str, Any]]:
    return (pass_definition.model or config.model), {
        **dict(config.params),
        **dict(pass_definition.params),
    }


def pass_salt(pass_definition: PassDefinition) -> str:
    return "\x00".join(
        (CACHE_SALT_VERSION, pass_definition.name, pass_definition.instruction)
    )


def load_instruction_text(
    path: str, name: str, table: dict[str, Any], base_directory: str
) -> IO[Result[str, str]]:
    has_inline = "instruction" in table
    if ("instruction_file" in table) == has_inline:
        return io_result(
            Err(
                "%s: [[pass]] %s: exactly one of instruction_file or instruction "
                "is required" % (path, name)
            )
        )

    if has_inline:
        instruction = table["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            return io_result(
                Err("%s: [[pass]] %s: instruction is empty" % (path, name))
            )

        return io_result(Ok(instruction.strip()))

    instruction_file = table["instruction_file"]
    if not isinstance(instruction_file, str) or not instruction_file.strip():
        return io_result(
            Err("%s: [[pass]] %s: instruction_file must be a path" % (path, name))
        )

    def thunk() -> Result[str, str]:
        try:
            with open(
                os.path.join(base_directory, instruction_file), encoding="utf-8"
            ) as handle:
                instruction = handle.read().strip()
        except OSError as error:
            return Err(
                "%s: [[pass]] %s: cannot read instruction file: %s"
                % (path, name, error)
            )

        if not instruction:
            return Err("%s: [[pass]] %s: instruction file is empty" % (path, name))

        return Ok(instruction)

    return IO(thunk)


def parse_pass_table(
    path: str, table: Any, base_directory: str
) -> IO[Result[PassDefinition, str]]:
    if not isinstance(table, dict):
        return io_result(Err("%s: [[pass]] entries must be tables" % path))

    unknown = sorted(set(table) - set(PASS_KEYS))
    if unknown:
        return io_result(
            Err("%s: [[pass]]: unknown key(s): %s" % (path, ", ".join(unknown)))
        )

    name = table.get("name")
    if not isinstance(name, str) or not name.strip():
        return io_result(Err("%s: [[pass]]: name must be a non-empty string" % path))

    return io_bind(
        load_instruction_text(path, name.strip(), table, base_directory),
        lambda instruction_result: io_result(
            result_bind(
                instruction_result,
                lambda instruction: pass_definition_from(
                    path, name.strip(), table, instruction
                ),
            )
        ),
    )


def document_passes(
    path: str, document: dict[str, Any]
) -> IO[Result[tuple[PassDefinition, ...], str]]:
    entries = document.get("pass")
    if entries is None:
        return io_result(Ok(()))

    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        return io_result(Err("%s: [[pass]] must define one or more pass tables" % path))

    return io_map(
        io_sequence(
            parse_pass_table(path, entry, os.path.dirname(os.path.abspath(path)))
            for entry in entries
        ),
        results_sequence,
    )


def resolve_passes(
    user_path: str,
    user_document_result: Result[dict[str, Any], str],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], str],
) -> IO[Result[tuple[dict[str, bool], tuple[PassDefinition, ...]], str]]:
    pair_type = tuple[dict[str, bool], tuple[PassDefinition, ...]]

    def resolve(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> IO[Result[pair_type, str]]:
        user_document, selected_document = documents
        sources: tuple[tuple[str | None, dict[str, Any]], ...] = ()
        if selected_document:
            sources += ((selected_path, selected_document),)

        if user_document:
            sources += ((user_path, user_document),)

        candidates = tuple(
            (path, document) for path, document in sources if "pass" in document
        )
        if not candidates:
            return io_result(
                Err(
                    "no [[pass]] tables found; define at least one pass in %s"
                    % (selected_path or user_path or "a config file (see --help)")
                )
            )

        option_sources = tuple(
            (path, document) for path, document in sources if "options" in document
        )
        options: Result[dict[str, bool], str]
        if option_sources:
            option_path, option_document = option_sources[0]
            options = parse_options_table(option_path or "", option_document["options"])
        else:
            options = Ok({})

        pass_path, pass_document = candidates[0]

        def combine(
            result: Result[tuple[PassDefinition, ...], str],
        ) -> Result[pair_type, str]:
            return result_bind(
                options,
                lambda option_values: result_map(
                    result, lambda passes: (option_values, passes)
                ),
            )

        return io_map(document_passes(pass_path or "", pass_document), combine)

    return result_bind_io(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        resolve,
    )


def read_document(
    path: str | None, description: str, exists: bool
) -> IO[Result[dict[str, Any], str]]:
    if not exists:
        return io_result(Ok({}))

    return io_map(
        load_toml(path or "", description),
        lambda result: result_bind(
            result,
            lambda document: result_map(
                validate_document(path or "", document), lambda _: document
            ),
        ),
    )


def resolve_config_path(
    requested: str | None, environment: Mapping[str, str]
) -> IO[str | None]:
    if requested:
        return io_pure(requested)

    configured = environment.get("TRANSLATE_CONFIG", "").strip()
    if configured:
        return io_pure(configured)

    def pick(cwd_value: str, user_path: str) -> str | None:
        local = os.path.join(cwd_value, "zh2en.toml")
        if os.path.exists(local):
            return local

        return user_path if os.path.exists(user_path) else None

    return io_bind(
        cwd(),
        lambda cwd_value: io_bind(
            user_config_path(environment),
            lambda user_path: IO(lambda: pick(cwd_value, user_path)),
        ),
    )


DocumentResult = Result[dict[str, Any], str]
ResolvedPasses = tuple[dict[str, bool], tuple[PassDefinition, ...]]
SetupResult = Result[Setup, str]


def build_setup(
    config: Config, pair: Result[ResolvedPasses, str]
) -> Result[Setup, str]:
    return result_bind(
        pair,
        lambda resolved: Ok(
            Setup(
                config=config,
                passes=apply_default_ascii(resolved[1], resolved[0]),
                ensure_paragraphs=resolved[0].get("ensure_paragraphs", False),
            )
        ),
    )


def load_setup(
    arguments: Arguments, environment: Mapping[str, str]
) -> IO[SetupResult]:
    def after_user(
        user_path: str, selected_path: str | None
    ) -> Callable[[bool], IO[tuple[DocumentResult, DocumentResult]]]:
        def read_both(user_exists: bool) -> IO[tuple[DocumentResult, DocumentResult]]:
            return io_bind(
                read_document(user_path, "user config", user_exists),
                lambda user_document: io_map(
                    read_document(selected_path, "config file", bool(selected_path)),
                    lambda selected_document: (user_document, selected_document),
                ),
            )

        return read_both

    def with_paths(user_path: str, selected_path: str | None) -> IO[SetupResult]:
        def after_documents(
            documents: tuple[DocumentResult, DocumentResult],
        ) -> IO[tuple[DocumentResult, DocumentResult, Result[ResolvedPasses, str]]]:
            return io_map(
                resolve_passes(
                    user_path, documents[0], selected_path, documents[1]
                ),
                lambda passes_result: (documents[0], documents[1], passes_result),
            )

        def assemble(
            resolved: tuple[
                DocumentResult,
                DocumentResult,
                Result[ResolvedPasses, str],
            ],
        ) -> SetupResult:
            return result_bind(
                merged_api_settings(
                    arguments,
                    environment,
                    user_path,
                    resolved[0],
                    selected_path,
                    resolved[1],
                ),
                lambda config: build_setup(config, resolved[2]),
            )

        return io_map(
            io_bind(
                io_bind(path_exists(user_path), after_user(user_path, selected_path)),
                after_documents,
            ),
            assemble,
        )

    return io_bind(
        user_config_path(environment),
        lambda user_path: io_bind(
            resolve_config_path(arguments.config, environment),
            lambda selected_path: with_paths(user_path, selected_path),
        ),
    )

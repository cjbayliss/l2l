from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any

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


@dataclass(frozen=True)
class PassDefinition:
    name: str
    instruction: str
    mode: str
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
ProgressCallback = Callable[[str, int], None]


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


def default_api_settings() -> dict[str, Any]:
    return {
        "base_url": "",
        "api_key": "",
        "model": "",
        "timeout": 120.0,
        "max_tokens": 100000,
        "params": {},
    }


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
    path: str, table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    def add(settings: dict[str, Any], key: str) -> Result[dict[str, Any], str]:
        if key not in table:
            return Ok(settings)

        value = table[key]
        if not isinstance(value, str) or not value.strip():
            return Err("%s: [api] %s must be a non-empty string" % (path, key))

        return Ok({**settings, key: value.strip()})

    def step(
        settings_result: Result[dict[str, Any], str], key: str
    ) -> Result[dict[str, Any], str]:
        return result_bind(settings_result, lambda settings: add(settings, key))

    initial: Result[dict[str, Any], str] = Ok({})
    return reduce(step, ("base_url", "api_key", "model"), initial)


def timeout_api_setting(
    path: str, settings: dict[str, Any], table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "timeout" not in table:
        return Ok(settings)

    value = table["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return Err("%s: [api] timeout must be a positive number" % path)

    return Ok({**settings, "timeout": float(value)})


def max_tokens_api_setting(
    path: str, settings: dict[str, Any], table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "max_tokens" not in table:
        return Ok(settings)

    value = table["max_tokens"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return Err("%s: [api] max_tokens must be a positive integer" % path)

    return Ok({**settings, "max_tokens": value})


def params_api_setting(
    path: str, settings: dict[str, Any], table: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "params" not in table:
        return Ok(settings)

    value = table["params"]
    if not isinstance(value, dict):
        return Err("%s: [api] params must be a table" % path)

    return Ok({**settings, "params": value})


def document_api_settings(
    path: str, document: dict[str, Any]
) -> Result[dict[str, Any], str]:
    if "api" not in document:
        return Ok({})

    table = document["api"]
    if not isinstance(table, dict):
        return Err("%s: [api] must be a table" % path)

    unknown = sorted(set(table) - set(API_SETTING_KEYS))
    if unknown:
        return Err("%s: [api]: unknown key(s): %s" % (path, ", ".join(unknown)))

    return result_bind(
        string_api_settings(path, table),
        lambda settings: result_bind(
            timeout_api_setting(path, settings, table),
            lambda settings: result_bind(
                max_tokens_api_setting(path, settings, table),
                lambda settings: params_api_setting(path, settings, table),
            ),
        ),
    )


def merge_api_settings(
    base: Mapping[str, Any], extra: Mapping[str, Any]
) -> dict[str, Any]:
    def merge_one(settings: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
        if key == "params" and isinstance(settings.get("params"), dict):
            return {**settings, "params": {**settings["params"], **value}}

        return {**settings, key: value}

    return reduce(
        lambda settings, entry: merge_one(settings, entry[0], entry[1]),
        extra.items(),
        dict(base),
    )


def parse_number_setting(
    environment: Mapping[str, str],
    settings: dict[str, Any],
    variable: str,
    key: str,
    convert: Callable[[str], Any],
    invalid_label: str,
) -> Result[dict[str, Any], str]:
    raw = environment.get(variable, "").strip()
    if not raw:
        return Ok(settings)

    try:
        value = convert(raw)
    except ValueError:
        return Err("%s must be %s, got %r" % (variable, invalid_label, raw))

    if value <= 0:
        return Err("%s must be positive" % variable)

    return Ok({**settings, key: value})


def api_settings_from_environment(
    environment: Mapping[str, str],
) -> Result[dict[str, Any], str]:
    def string_step(
        settings_result: Result[dict[str, Any], str], binding: tuple[str, str]
    ) -> Result[dict[str, Any], str]:
        variable, key = binding
        value = environment.get(variable, "").strip()
        return result_bind(
            settings_result,
            lambda settings: Ok({**settings, key: value}) if value else Ok(settings),
        )

    initial: Result[dict[str, Any], str] = Ok({})
    return result_bind(
        reduce(
            string_step,
            (
                ("TRANSLATE_BASE_URL", "base_url"),
                ("TRANSLATE_API_KEY", "api_key"),
                ("TRANSLATE_MODEL", "model"),
            ),
            initial,
        ),
        lambda settings: result_bind(
            parse_number_setting(
                environment, settings, "TRANSLATE_TIMEOUT", "timeout", float, "a number"
            ),
            lambda settings: parse_number_setting(
                environment,
                settings,
                "TRANSLATE_MAX_TOKENS",
                "max_tokens",
                int,
                "an integer",
            ),
        ),
    )


def api_settings_from_arguments(arguments: Arguments) -> dict[str, Any]:
    def step(settings: dict[str, Any], binding: tuple[Any, str]) -> dict[str, Any]:
        value, key = binding
        return {**settings, key: value} if value is not None else settings

    return reduce(
        step,
        (
            (arguments.base_url, "base_url"),
            (arguments.api_key, "api_key"),
            (arguments.model, "model"),
            (arguments.timeout, "timeout"),
            (arguments.max_tokens, "max_tokens"),
        ),
        {},
    )


def missing_api_settings(config: Config) -> tuple[str, ...]:
    return tuple(
        description
        for value, description in (
            (config.base_url, "api.base_url (--base-url / TRANSLATE_BASE_URL)"),
            (config.api_key, "api.api_key (--api-key / TRANSLATE_API_KEY)"),
            (config.model, "api.model (--model / TRANSLATE_MODEL)"),
        )
        if not value
    )


def build_config(settings: Mapping[str, Any]) -> Result[Config, str]:
    config = Config(
        base_url=settings["base_url"].rstrip("/"),
        api_key=settings["api_key"],
        model=settings["model"],
        timeout=settings["timeout"],
        max_tokens=settings["max_tokens"],
        params=MappingProxyType(dict(settings["params"])),
    )
    missing = missing_api_settings(config)
    if missing:
        return Err("missing required API settings: " + ", ".join(missing))

    return Ok(config)


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
        settings: dict[str, Any],
    ) -> Result[dict[str, Any], str]:
        return result_bind(
            document_result,
            lambda document: result_map(
                document_api_settings(path or "", document),
                lambda extra: merge_api_settings(settings, extra),
            ),
        )

    pipeline: Result[dict[str, Any], str] = Ok(default_api_settings())
    pipeline = result_bind(
        pipeline,
        lambda settings: merge_document(user_path, user_document_result, settings),
    )
    pipeline = result_bind(
        pipeline,
        lambda settings: merge_document(
            selected_path, selected_document_result, settings
        ),
    )
    pipeline = result_bind(
        pipeline,
        lambda settings: result_map(
            api_settings_from_environment(environment),
            lambda extra: merge_api_settings(settings, extra),
        ),
    )
    pipeline = result_bind(
        pipeline,
        lambda settings: Ok(
            merge_api_settings(settings, api_settings_from_arguments(arguments))
        ),
    )
    return result_bind(pipeline, build_config)


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


def load_setup(
    arguments: Arguments, environment: Mapping[str, str]
) -> IO[Result[Setup, str]]:
    def from_paths(user_path: str, selected_path: str | None) -> IO[Result[Setup, str]]:
        def after_user(user_exists: bool) -> IO[Result[Setup, str]]:
            def after_user_document(
                user_document_result: Result[dict[str, Any], str],
            ) -> IO[Result[Setup, str]]:
                def after_selected_document(
                    selected_document_result: Result[dict[str, Any], str],
                ) -> IO[Result[Setup, str]]:
                    def after_passes(
                        passes_result: Result[
                            tuple[dict[str, bool], tuple[PassDefinition, ...]], str
                        ],
                    ) -> IO[Result[Setup, str]]:
                        return io_result(
                            result_bind(
                                merged_api_settings(
                                    arguments,
                                    environment,
                                    user_path,
                                    user_document_result,
                                    selected_path,
                                    selected_document_result,
                                ),
                                lambda config: result_bind(
                                    passes_result,
                                    lambda pair: Ok(
                                        Setup(
                                            config=config,
                                            passes=apply_default_ascii(
                                                pair[1], pair[0]
                                            ),
                                            ensure_paragraphs=pair[0].get(
                                                "ensure_paragraphs", False
                                            ),
                                        )
                                    ),
                                ),
                            )
                        )

                    return io_bind(
                        resolve_passes(
                            user_path,
                            user_document_result,
                            selected_path,
                            selected_document_result,
                        ),
                        after_passes,
                    )

                return io_bind(
                    read_document(selected_path, "config file", bool(selected_path)),
                    after_selected_document,
                )

            return io_bind(
                read_document(user_path, "user config", user_exists),
                after_user_document,
            )

        return io_bind(path_exists(user_path), after_user)

    return io_bind(
        user_config_path(environment),
        lambda user_path: io_bind(
            resolve_config_path(arguments.config, environment),
            lambda selected_path: from_paths(user_path, selected_path),
        ),
    )

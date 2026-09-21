from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import reduce
from types import MappingProxyType
from typing import Any, Literal, assert_never

from zh2en.effects import (
    cwd,
    load_toml,
    path_exists,
    user_config_path,
)
from zh2en.errors import TranslationError, fail_config, fail_missing_settings
from zh2en.monads import (
    IO,
    Ok,
    Result,
    io_bind,
    io_map,
    io_pair,
    io_pure,
    io_result,
    io_sequence,
    result_bind,
    result_bind_io,
    result_map,
    results_sequence,
)
from zh2en.settings import (
    API_SETTING_KEYS,
    DEFAULT_API_SETTINGS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT,
    PASS_KEYS,
    Arguments,
    Config,
    PartialApiSettings,
    PassDefinition,
    Setup,
)


def validate_document(
    path: str, document: dict[str, Any]
) -> Result[None, TranslationError]:
    if not document:
        return Ok(None)

    unknown = sorted(set(document) - {"api", "options", "pass"})
    if unknown:
        return fail_config(
            "%s: unknown top-level key(s): %s (expected [api], [options], [[pass]])"
            % (path, ", ".join(unknown))
        )

    return Ok(None)


StringApiKey = Literal["base_url", "api_key", "model"]

STRING_API_KEYS: tuple[StringApiKey, ...] = ("base_url", "api_key", "model")


def string_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    def set_field(
        partial: PartialApiSettings, key: StringApiKey, value: str
    ) -> PartialApiSettings:
        match key:
            case "base_url":
                return replace(partial, base_url=value)

            case "api_key":
                return replace(partial, api_key=value)

            case "model":
                return replace(partial, model=value)

            case other:
                assert_never(other)

    def set_field_checked(
        partial: PartialApiSettings, key: StringApiKey
    ) -> Result[PartialApiSettings, TranslationError]:
        if key not in table:
            return Ok(partial)

        value = table[key]
        if not isinstance(value, str) or not value.strip():
            return fail_config("%s: [api] %s must be a non-empty string" % (path, key))

        return Ok(set_field(partial, key, value.strip()))

    def step(
        partial_result: Result[PartialApiSettings, TranslationError], key: StringApiKey
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(
            partial_result, lambda partial: set_field_checked(partial, key)
        )

    initial: Result[PartialApiSettings, TranslationError] = Ok(partial)
    return reduce(step, STRING_API_KEYS, initial)


def timeout_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    if "timeout" not in table:
        return Ok(partial)

    value = table["timeout"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return fail_config("%s: [api] timeout must be a positive number" % path)

    return Ok(replace(partial, timeout=float(value)))


def max_tokens_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    if "max_tokens" not in table:
        return Ok(partial)

    value = table["max_tokens"]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return fail_config("%s: [api] max_tokens must be a positive integer" % path)

    return Ok(replace(partial, max_tokens=value))


def params_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    if "params" not in table:
        return Ok(partial)

    value = table["params"]
    if not isinstance(value, dict):
        return fail_config("%s: [api] params must be a table" % path)

    return Ok(replace(partial, params=value))


API_FIELD_PARSERS = (
    string_api_setting,
    timeout_api_setting,
    max_tokens_api_setting,
    params_api_setting,
)


def document_api_settings(
    path: str, document: dict[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    if "api" not in document:
        return Ok(PartialApiSettings())

    table = document["api"]
    if not isinstance(table, dict):
        return fail_config("%s: [api] must be a table" % path)

    unknown = sorted(set(table) - set(API_SETTING_KEYS))
    if unknown:
        return fail_config("%s: [api]: unknown key(s): %s" % (path, ", ".join(unknown)))

    def step(
        partial_result: Result[PartialApiSettings, TranslationError],
        parse: Callable[
            [str, PartialApiSettings, Mapping[str, Any]],
            Result[PartialApiSettings, TranslationError],
        ],
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(partial_result, lambda partial: parse(path, partial, table))

    initial: Result[PartialApiSettings, TranslationError] = Ok(PartialApiSettings())
    return reduce(step, API_FIELD_PARSERS, initial)


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
) -> Result[PartialApiSettings, TranslationError]:
    raw = environment.get("TRANSLATE_TIMEOUT", "").strip()
    if not raw:
        return Ok(partial)

    try:
        timeout = float(raw)
    except ValueError:
        return fail_config("TRANSLATE_TIMEOUT must be a number, got %r" % raw)

    if timeout <= 0:
        return fail_config("TRANSLATE_TIMEOUT must be positive")

    return Ok(replace(partial, timeout=timeout))


def env_max_tokens_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, TranslationError]:
    raw = environment.get("TRANSLATE_MAX_TOKENS", "").strip()
    if not raw:
        return Ok(partial)

    try:
        max_tokens = int(raw)
    except ValueError:
        return fail_config("TRANSLATE_MAX_TOKENS must be an integer, got %r" % raw)

    if max_tokens <= 0:
        return fail_config("TRANSLATE_MAX_TOKENS must be positive")

    return Ok(replace(partial, max_tokens=max_tokens))


def api_settings_from_environment(
    environment: Mapping[str, str],
) -> Result[PartialApiSettings, TranslationError]:
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


def build_config(partial: PartialApiSettings) -> Result[Config, TranslationError]:
    missing = missing_api_settings(partial)
    if missing:
        return fail_missing_settings(missing)

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
    user_document_result: Result[dict[str, Any], TranslationError],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], TranslationError],
) -> Result[Config, TranslationError]:
    def merge_document(
        path: str | None,
        document_result: Result[dict[str, Any], TranslationError],
        partial: PartialApiSettings,
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(
            document_result,
            lambda document: result_map(
                document_api_settings(path or "", document),
                lambda extra: merge_api_settings(partial, extra),
            ),
        )

    layer: Result[PartialApiSettings, TranslationError] = Ok(DEFAULT_API_SETTINGS)
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


def parse_options_table(
    path: str, table: Any
) -> Result[dict[str, bool], TranslationError]:
    if not isinstance(table, dict):
        return fail_config("%s: [options] must be a table" % path)

    unknown = sorted(set(table) - {"ascii", "ensure_paragraphs"})
    if unknown:
        return fail_config(
            "%s: [options]: unknown key(s): %s" % (path, ", ".join(unknown))
        )

    ascii_value = table.get("ascii", False)
    if not isinstance(ascii_value, bool):
        return fail_config("%s: [options] ascii must be true or false" % path)

    ensure_paragraphs = table.get("ensure_paragraphs", False)
    if not isinstance(ensure_paragraphs, bool):
        return fail_config(
            "%s: [options] ensure_paragraphs must be true or false" % path
        )

    return Ok({"ascii": ascii_value, "ensure_paragraphs": ensure_paragraphs})


def pass_definition_from(
    path: str, name: str, table: dict[str, Any], instruction: str
) -> Result[PassDefinition, TranslationError]:
    mode = table.get("mode", "chunk")
    if mode not in ("analysis", "chunk", "paragraph"):
        return fail_config(
            "%s: [[pass]] %s: mode must be analysis, chunk, or paragraph" % (path, name)
        )

    ascii_value = table.get("ascii")
    if ascii_value is not None and not isinstance(ascii_value, bool):
        return fail_config(
            "%s: [[pass]] %s: ascii must be true or false" % (path, name)
        )

    model = table.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return fail_config(
            "%s: [[pass]] %s: model must be a non-empty string" % (path, name)
        )

    params = table.get("params", {})
    if not isinstance(params, dict):
        return fail_config("%s: [[pass]] %s: params must be a table" % (path, name))

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


def load_instruction_text(
    path: str, name: str, table: dict[str, Any], base_directory: str
) -> IO[Result[str, TranslationError]]:
    has_inline = "instruction" in table
    if ("instruction_file" in table) == has_inline:
        return io_result(
            fail_config(
                "%s: [[pass]] %s: exactly one of instruction_file or instruction "
                "is required" % (path, name)
            )
        )

    if has_inline:
        instruction = table["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            return io_result(
                fail_config("%s: [[pass]] %s: instruction is empty" % (path, name))
            )

        return io_result(Ok(instruction.strip()))

    instruction_file = table["instruction_file"]
    if not isinstance(instruction_file, str) or not instruction_file.strip():
        return io_result(
            fail_config(
                "%s: [[pass]] %s: instruction_file must be a path" % (path, name)
            )
        )

    def thunk() -> Result[str, TranslationError]:
        try:
            with open(
                os.path.join(base_directory, instruction_file), encoding="utf-8"
            ) as handle:
                instruction = handle.read().strip()
        except OSError as error:
            return fail_config(
                "%s: [[pass]] %s: cannot read instruction file: %s"
                % (path, name, error)
            )

        if not instruction:
            return fail_config(
                "%s: [[pass]] %s: instruction file is empty" % (path, name)
            )

        return Ok(instruction)

    return IO(thunk)


def parse_pass_table(
    path: str, table: Any, base_directory: str
) -> IO[Result[PassDefinition, TranslationError]]:
    if not isinstance(table, dict):
        return io_result(fail_config("%s: [[pass]] entries must be tables" % path))

    unknown = sorted(set(table) - set(PASS_KEYS))
    if unknown:
        return io_result(
            fail_config("%s: [[pass]]: unknown key(s): %s" % (path, ", ".join(unknown)))
        )

    name = table.get("name")
    if not isinstance(name, str) or not name.strip():
        return io_result(
            fail_config("%s: [[pass]]: name must be a non-empty string" % path)
        )

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
) -> IO[Result[tuple[PassDefinition, ...], TranslationError]]:
    entries = document.get("pass")
    if entries is None:
        return io_result(Ok(()))

    if (
        not isinstance(entries, list)
        or not entries
        or not all(isinstance(entry, dict) for entry in entries)
    ):
        return io_result(
            fail_config(
                "%s: [[pass]] must define one or more pass tables" % path
            )
        )

    return io_map(
        io_sequence(
            parse_pass_table(path, entry, os.path.dirname(os.path.abspath(path)))
            for entry in entries
        ),
        results_sequence,
    )


def document_layers(
    user_path: str,
    user_document: dict[str, Any],
    selected_path: str | None,
    selected_document: dict[str, Any],
) -> tuple[tuple[str | None, dict[str, Any]], ...]:
    layers: tuple[tuple[str | None, dict[str, Any]], ...] = ()
    if selected_document:
        layers += ((selected_path, selected_document),)

    if user_document:
        layers += ((user_path, user_document),)

    return layers


def resolve_passes(
    user_path: str,
    user_document_result: Result[dict[str, Any], TranslationError],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], TranslationError],
) -> IO[Result[tuple[dict[str, bool], tuple[PassDefinition, ...]], TranslationError]]:
    pair_type = tuple[dict[str, bool], tuple[PassDefinition, ...]]

    def resolve(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> IO[Result[pair_type, TranslationError]]:
        layers = document_layers(
            user_path, documents[0], selected_path, documents[1]
        )
        pass_sources = tuple(layer for layer in layers if "pass" in layer[1])
        if not pass_sources:
            return io_result(
                fail_config(
                    "no [[pass]] tables found; define at least one pass in %s"
                    % (selected_path or user_path or "a config file (see --help)")
                )
            )

        option_sources = tuple(layer for layer in layers if "options" in layer[1])
        options: Result[dict[str, bool], TranslationError]
        if option_sources:
            option_path, option_document = option_sources[0]
            options = parse_options_table(
                option_path or "", option_document["options"]
            )
        else:
            options = Ok({})

        pass_path, pass_document = pass_sources[0]

        def combine(
            resolved_passes: Result[tuple[PassDefinition, ...], TranslationError],
        ) -> Result[pair_type, TranslationError]:
            return result_bind(
                options,
                lambda option_values: result_map(
                    resolved_passes, lambda passes: (option_values, passes)
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
) -> IO[Result[dict[str, Any], TranslationError]]:
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


DocumentResult = Result[dict[str, Any], TranslationError]
ResolvedPasses = tuple[dict[str, bool], tuple[PassDefinition, ...]]
SetupResult = Result[Setup, TranslationError]


def build_setup(
    config: Config, pair: Result[ResolvedPasses, TranslationError]
) -> Result[Setup, TranslationError]:
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
    def with_paths(user_path: str, selected_path: str | None) -> IO[SetupResult]:
        def read_documents(
            user_exists: bool,
        ) -> IO[tuple[DocumentResult, DocumentResult]]:
            return io_pair(
                read_document(user_path, "user config", user_exists),
                read_document(selected_path, "config file", bool(selected_path)),
            )

        def assemble(
            documents: tuple[DocumentResult, DocumentResult],
        ) -> IO[SetupResult]:
            def setup_with(
                resolved: Result[ResolvedPasses, TranslationError],
            ) -> SetupResult:
                return result_bind(
                    merged_api_settings(
                        arguments,
                        environment,
                        user_path,
                        documents[0],
                        selected_path,
                        documents[1],
                    ),
                    lambda config: build_setup(config, resolved),
                )

            return io_map(
                resolve_passes(user_path, documents[0], selected_path, documents[1]),
                setup_with,
            )

        return io_bind(
            io_bind(path_exists(user_path), read_documents),
            assemble,
        )

    return io_bind(
        user_config_path(environment),
        lambda user_path: io_bind(
            resolve_config_path(arguments.config, environment),
            lambda selected_path: with_paths(user_path, selected_path),
        ),
    )


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "***"

    return api_key[:4] + "..." + api_key[-2:]


def params_text(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)


def setup_report(setup: Setup, effective_ensure_paragraphs: bool) -> str:
    config = setup.config
    api_lines: tuple[str, ...] = (
        "api.base_url: %s" % config.base_url,
        "api.model: %s" % config.model,
        "api.timeout: %g" % config.timeout,
        "api.max_tokens: %d" % config.max_tokens,
        "api.api_key: %s" % mask_api_key(config.api_key),
        *(("api.params: %s" % params_text(config.params),) if config.params else ()),
    )
    pass_lines = tuple(
        line
        for number, pass_definition in enumerate(setup.passes, 1)
        for line in (
            (
                "pass %d/%d [%s]: mode=%s ascii=%s model=%s instruction=%d chars"
                % (
                    number,
                    len(setup.passes),
                    pass_definition.name,
                    pass_definition.mode,
                    pass_definition.ascii,
                    pass_definition.model or "<default>",
                    len(pass_definition.instruction),
                ),
                *(
                    ("  params: %s" % params_text(pass_definition.params),)
                    if pass_definition.params
                    else ()
                ),
            )
        )
    )
    return "\n".join(
        api_lines
        + pass_lines
        + ("options.ensure_paragraphs: %s" % effective_ensure_paragraphs,)
    )

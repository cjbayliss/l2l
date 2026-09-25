from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import reduce
from types import MappingProxyType
from typing import Any, Literal, assert_never

from l2l.effects import (
    cwd,
    load_toml,
    path_exists,
    read_text_file,
    user_config_path,
)
from l2l.errors import TranslationError, fail_config, fail_missing_settings
from l2l.monads import (
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
    result_sequence,
)
from l2l.settings import (
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
            "%s: unknown top-level key(s): %s (expected [api], [options], "
            "[[pass]])" % (path, ", ".join(unknown))
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


NumericApiKey = Literal["timeout", "max_tokens"]


def positive_number_setting(
    path: str,
    partial: PartialApiSettings,
    table: Mapping[str, Any],
    key: NumericApiKey,
    whole: bool,
) -> Result[PartialApiSettings, TranslationError]:
    if key not in table:
        return Ok(partial)

    value = table[key]
    valid = (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and value > 0
        and (isinstance(value, int) or not whole)
    )
    if not valid:
        expected = "a positive integer" if whole else "a positive number"
        return fail_config("%s: [api] %s must be %s" % (path, key, expected))

    if whole:
        return Ok(replace(partial, max_tokens=int(value)))

    return Ok(replace(partial, timeout=float(value)))


def timeout_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    return positive_number_setting(path, partial, table, "timeout", whole=False)


def max_tokens_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    return positive_number_setting(path, partial, table, "max_tokens", whole=True)


def params_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    if "params" not in table:
        return Ok(partial)

    value = table["params"]
    if not isinstance(value, dict):
        return fail_config("%s: [api] params must be a table" % path)

    if not is_json_value(value):
        return fail_config("%s: [api] params must contain only JSON values" % path)

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


def env_positive_setting(
    environment: Mapping[str, str],
    variable: str,
    parse: Callable[[str], float],
    kind: str,
) -> Result[float | None, TranslationError]:
    raw = environment.get(variable, "").strip()
    if not raw:
        return Ok(None)

    try:
        parsed = parse(raw)
    except ValueError:
        return fail_config("%s must be %s, got %r" % (variable, kind, raw))

    if parsed <= 0:
        return fail_config("%s must be positive" % variable)

    return Ok(parsed)


def env_timeout_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, TranslationError]:
    parsed = env_positive_setting(environment, "TRANSLATE_TIMEOUT", float, "a number")
    return result_map(
        parsed,
        lambda value: partial if value is None else replace(partial, timeout=value),
    )


def env_max_tokens_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, TranslationError]:
    parsed = env_positive_setting(
        environment, "TRANSLATE_MAX_TOKENS", int, "an integer"
    )
    return result_map(
        parsed,
        lambda value: (
            partial if value is None else replace(partial, max_tokens=int(value))
        ),
    )


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
            timeout=partial.timeout if partial.timeout is not None else DEFAULT_TIMEOUT,
            max_tokens=(
                partial.max_tokens
                if partial.max_tokens is not None
                else DEFAULT_MAX_TOKENS
            ),
            params=(
                partial.params if partial.params is not None else MappingProxyType({})
            ),
        )
    )


def is_json_value(value: Any) -> bool:
    match value:
        case bool() | int() | float() | str() | None:
            return True

        case list():
            return all(is_json_value(item) for item in value)

        case dict():
            return all(is_json_value(item) for item in value.values())

        case _:
            return False


def is_paragraph_tri_state(value: Any) -> bool:
    match value:
        case bool():
            return True

        case int():
            return value > 0

        case _:
            return False


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

    def environment_layer(
        partial: PartialApiSettings,
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_map(
            api_settings_from_environment(environment),
            lambda extra: merge_api_settings(partial, extra),
        )

    def arguments_layer(
        partial: PartialApiSettings,
    ) -> Result[PartialApiSettings, TranslationError]:
        return Ok(merge_api_settings(partial, api_settings_from_arguments(arguments)))

    def step(
        layer_result: Result[PartialApiSettings, TranslationError],
        merge_layer: Callable[
            [PartialApiSettings], Result[PartialApiSettings, TranslationError]
        ],
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(layer_result, merge_layer)

    layers = (
        lambda partial: merge_document(user_path, user_document_result, partial),
        lambda partial: merge_document(
            selected_path, selected_document_result, partial
        ),
        environment_layer,
        arguments_layer,
    )
    initial: Result[PartialApiSettings, TranslationError] = Ok(DEFAULT_API_SETTINGS)
    return result_bind(reduce(step, layers, initial), build_config)


def parse_options_table(
    path: str, table: Any
) -> Result[dict[str, bool | int], TranslationError]:
    if not isinstance(table, dict):
        return fail_config("%s: [options] must be a table" % path)

    unknown = sorted(
        set(table) - {"ascii", "ensure_paragraphs", "retranslate_untranslated"}
    )
    if unknown:
        return fail_config(
            "%s: [options]: unknown key(s): %s" % (path, ", ".join(unknown))
        )

    ascii_value = table.get("ascii", False)
    if not isinstance(ascii_value, bool):
        return fail_config("%s: [options] ascii must be true or false" % path)

    ensure_paragraphs = table.get("ensure_paragraphs", False)
    if not is_paragraph_tri_state(ensure_paragraphs):
        return fail_config(
            "%s: [options] ensure_paragraphs must be true, false, "
            "or a positive integer" % path
        )

    retranslate_untranslated = table.get("retranslate_untranslated", False)
    if not isinstance(retranslate_untranslated, bool):
        return fail_config(
            "%s: [options] retranslate_untranslated must be true or false" % path
        )

    return Ok(
        {
            "ascii": ascii_value,
            "ensure_paragraphs": ensure_paragraphs,
            "retranslate_untranslated": retranslate_untranslated,
        }
    )


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

    ensure_paragraphs_value = table.get("ensure_paragraphs")
    if ensure_paragraphs_value is not None and not is_paragraph_tri_state(
        ensure_paragraphs_value
    ):
        return fail_config(
            "%s: [[pass]] %s: ensure_paragraphs must be true, false, "
            "or a positive integer" % (path, name)
        )

    retranslate_value = table.get("retranslate_untranslated")
    if retranslate_value is not None and not isinstance(retranslate_value, bool):
        return fail_config(
            "%s: [[pass]] %s: retranslate_untranslated must be true or false"
            % (path, name)
        )

    model = table.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        return fail_config(
            "%s: [[pass]] %s: model must be a non-empty string" % (path, name)
        )

    params = table.get("params", {})
    if not isinstance(params, dict):
        return fail_config("%s: [[pass]] %s: params must be a table" % (path, name))

    if not is_json_value(params):
        return fail_config(
            "%s: [[pass]] %s: params must contain only JSON values" % (path, name)
        )

    return Ok(
        PassDefinition(
            name=name,
            instruction=instruction,
            mode=mode,
            params=MappingProxyType(dict(params)),
            model=model.strip() if model else None,
            ascii=ascii_value,
            ensure_paragraphs=ensure_paragraphs_value,
            retranslate_untranslated=retranslate_value,
        )
    )


def apply_default_options(
    pass_definitions: tuple[PassDefinition, ...],
    options: Mapping[str, bool | int],
    ensure_paragraphs_flag: bool | int,
) -> tuple[PassDefinition, ...]:
    ascii_option = options.get("ascii", False)
    ascii_default = ascii_option if isinstance(ascii_option, bool) else False
    retranslate_option = options.get("retranslate_untranslated", False)
    retranslate_default = (
        retranslate_option if isinstance(retranslate_option, bool) else False
    )
    ensure_default = options.get("ensure_paragraphs", False) or ensure_paragraphs_flag
    return tuple(
        replace(
            pass_definition,
            ascii=(
                ascii_default
                if pass_definition.ascii is None
                else pass_definition.ascii
            ),
            ensure_paragraphs=(
                ensure_default
                if pass_definition.ensure_paragraphs is None
                else pass_definition.ensure_paragraphs
            ),
            retranslate_untranslated=(
                retranslate_default
                if pass_definition.retranslate_untranslated is None
                else pass_definition.retranslate_untranslated
            ),
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
                "%s: [[pass]] %s: exactly one of instruction_file "
                "or instruction is required" % (path, name)
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

    description = "%s: [[pass]] %s: instruction file" % (path, name)
    file_path = os.path.join(base_directory, instruction_file)

    def checked(
        text_result: Result[str, TranslationError],
    ) -> Result[str, TranslationError]:
        return result_bind(
            text_result,
            lambda text: (
                Ok(text.strip())
                if text.strip()
                else fail_config(
                    "%s: [[pass]] %s: instruction file is empty" % (path, name)
                )
            ),
        )

    return io_map(read_text_file(file_path, description), checked)


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
            fail_config("%s: [[pass]] must define one or more pass tables" % path)
        )

    return io_map(
        io_sequence(
            parse_pass_table(path, entry, os.path.dirname(os.path.abspath(path)))
            for entry in entries
        ),
        result_sequence,
    )


def document_layers(
    user_path: str,
    user_document: dict[str, Any],
    selected_path: str | None,
    selected_document: dict[str, Any],
) -> tuple[tuple[str | None, dict[str, Any]], ...]:
    return tuple(
        (path, document)
        for path, document, present in (
            (selected_path, selected_document, bool(selected_document)),
            (user_path, user_document, bool(user_document)),
        )
        if present
    )


def resolve_passes(
    user_path: str,
    user_document_result: Result[dict[str, Any], TranslationError],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], TranslationError],
) -> IO[
    Result[tuple[dict[str, bool | int], tuple[PassDefinition, ...]], TranslationError]
]:
    pair_type = tuple[dict[str, bool | int], tuple[PassDefinition, ...]]

    def resolve(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> IO[Result[pair_type, TranslationError]]:
        layers = document_layers(user_path, documents[0], selected_path, documents[1])
        pass_sources = tuple(layer for layer in layers if "pass" in layer[1])
        if not pass_sources:
            return io_result(
                fail_config(
                    "no [[pass]] tables found; define at least one pass in %s"
                    % (selected_path or user_path or "a config file (see --help)")
                )
            )

        option_sources = tuple(layer for layer in layers if "options" in layer[1])
        options: Result[dict[str, bool | int], TranslationError]
        if option_sources:
            option_path, option_document = option_sources[0]
            options = parse_options_table(option_path or "", option_document["options"])
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
    requested: str | None,
    environment: Mapping[str, str],
    path_exists: Callable[[str], bool] = os.path.exists,
    current_directory: IO[str] | None = None,
) -> IO[str | None]:
    if requested:
        return io_pure(requested)

    configured = environment.get("TRANSLATE_CONFIG", "").strip()
    if configured:
        return io_pure(configured)

    def pick(cwd_value: str, user_path: str) -> str | None:
        local = os.path.join(cwd_value, "l2l.toml")
        if path_exists(local):
            return local

        return user_path if path_exists(user_path) else None

    return io_bind(
        current_directory if current_directory is not None else cwd(),
        lambda cwd_value: io_bind(
            user_config_path(environment),
            lambda user_path: IO(lambda: pick(cwd_value, user_path)),
        ),
    )


DocumentResult = Result[dict[str, Any], TranslationError]
ResolvedPasses = tuple[dict[str, bool | int], tuple[PassDefinition, ...]]
SetupResult = Result[Setup, TranslationError]


def build_setup(
    config: Config,
    pair: Result[ResolvedPasses, TranslationError],
    ensure_paragraphs_flag: bool = False,
) -> Result[Setup, TranslationError]:
    def with_options(resolved: ResolvedPasses) -> Setup:
        effective = (
            resolved[0].get("ensure_paragraphs", False) or ensure_paragraphs_flag
        )
        return Setup(
            config=config,
            passes=apply_default_options(resolved[1], resolved[0], effective),
            ensure_paragraphs=effective,
        )

    return result_bind(pair, lambda resolved: Ok(with_options(resolved)))


def load_setup(arguments: Arguments, environment: Mapping[str, str]) -> IO[SetupResult]:
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
                    lambda config: build_setup(
                        config, resolved, arguments.ensure_paragraphs
                    ),
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

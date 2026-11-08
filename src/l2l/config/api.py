from __future__ import annotations

from dataclasses import replace
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, assert_never

from l2l.errors import TranslationError, fail_config, fail_missing_settings
from l2l.json_values import is_json_object, is_non_empty_string, is_string_keyed_table
from l2l.monads import Ok, Result, result_bind, result_map, result_sequence
from l2l.settings import (
    API_SETTING_KEYS,
    DEFAULT_API_SETTINGS,
    DEFAULT_MAXIMUM_TOKENS,
    DEFAULT_TIMEOUT,
    UNSET,
    Config,
    PartialApiSettings,
    Unset,
    value_if_set,
)
lazy from l2l.config.models import DocumentResult

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

StringApiKey = Literal["base_url", "api_key", "model"]

STRING_API_KEYS: Final[tuple[StringApiKey, ...]] = ("base_url", "api_key", "model")


def string_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, object]
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

            case _:
                assert_never(key)

    def set_field_checked(
        partial: PartialApiSettings, key: StringApiKey
    ) -> Result[PartialApiSettings, TranslationError]:
        if key not in table:
            return Ok(partial)

        value = table[key]
        if not is_non_empty_string(value):
            return fail_config(f"{path}: [api] {key} must be a non-empty string")

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
    table: Mapping[str, object],
    key: NumericApiKey,
    whole: bool,
) -> Result[PartialApiSettings, TranslationError]:
    if key not in table:
        return Ok(partial)

    value = table[key]
    if whole:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return fail_config(f"{path}: [api] {key} must be a positive integer")

        return Ok(replace(partial, maximum_tokens=value))

    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return fail_config(f"{path}: [api] {key} must be a positive number")

    return Ok(replace(partial, timeout=float(value)))


def timeout_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, object]
) -> Result[PartialApiSettings, TranslationError]:
    return positive_number_setting(path, partial, table, "timeout", whole=False)


def maximum_tokens_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, object]
) -> Result[PartialApiSettings, TranslationError]:
    return positive_number_setting(path, partial, table, "max_tokens", whole=True)


def parameters_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, object]
) -> Result[PartialApiSettings, TranslationError]:
    if "params" not in table:
        return Ok(partial)

    value = table["params"]
    if not is_string_keyed_table(value):
        return fail_config(f"{path}: [api] params must be a table")
    if not is_json_object(value):
        return fail_config(f"{path}: [api] params must contain only JSON values")

    return Ok(replace(partial, parameters=value))


API_FIELD_PARSERS: Final = (
    string_api_setting,
    timeout_api_setting,
    maximum_tokens_api_setting,
    parameters_api_setting,
)


def document_api_settings(
    path: str, document: Mapping[str, object]
) -> Result[PartialApiSettings, TranslationError]:
    if "api" not in document:
        return Ok(PartialApiSettings())

    table = document["api"]
    if not is_string_keyed_table(table):
        return fail_config(f"{path}: [api] must be a table")

    unknown = sorted(set(table) - set(API_SETTING_KEYS))
    if unknown:
        return fail_config(
            "{}: [api]: unknown key(s): {}".format(path, ", ".join(unknown))
        )

    def step(
        partial_result: Result[PartialApiSettings, TranslationError],
        parse: Callable[
            [str, PartialApiSettings, Mapping[str, object]],
            Result[PartialApiSettings, TranslationError],
        ],
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(partial_result, lambda partial: parse(path, partial, table))

    initial: Result[PartialApiSettings, TranslationError] = Ok(PartialApiSettings())
    return reduce(step, API_FIELD_PARSERS, initial)


def env_string_settings(environment: Mapping[str, str]) -> PartialApiSettings:
    def value(variable: str) -> str | Unset:
        raw = environment.get(variable, "").strip()
        return raw or UNSET

    return PartialApiSettings(
        base_url=value("TRANSLATE_BASE_URL"),
        api_key=value("TRANSLATE_API_KEY"),
        model=value("TRANSLATE_MODEL"),
    )


def optional_positive_setting(
    value: float, description: str
) -> Result[float, TranslationError]:
    if value <= 0:
        return fail_config(f"{description} must be positive")
    return Ok(value)


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
        return fail_config(f"{variable} must be {kind}, got {raw!r}")

    return result_map(
        optional_positive_setting(parsed, variable),
        lambda value: value or None,
    )


def env_timeout_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, TranslationError]:
    parsed = env_positive_setting(environment, "TRANSLATE_TIMEOUT", float, "a number")
    return result_map(
        parsed,
        lambda value: partial if value is None else replace(partial, timeout=value),
    )


def environment_maximum_tokens_setting(
    environment: Mapping[str, str], partial: PartialApiSettings
) -> Result[PartialApiSettings, TranslationError]:
    parsed = env_positive_setting(
        environment, "TRANSLATE_MAX_TOKENS", int, "an integer"
    )
    return result_map(
        parsed,
        lambda value: (
            partial if value is None else replace(partial, maximum_tokens=int(value))
        ),
    )


def api_settings_from_environment(
    environment: Mapping[str, str],
) -> Result[PartialApiSettings, TranslationError]:
    return result_bind(
        env_timeout_setting(environment, env_string_settings(environment)),
        lambda partial: environment_maximum_tokens_setting(environment, partial),
    )


def missing_api_settings(partial: PartialApiSettings) -> tuple[str, ...]:
    return tuple(
        description
        for value, description in (
            (value_if_set(partial.base_url), "api.base_url (TRANSLATE_BASE_URL)"),
            (value_if_set(partial.api_key), "api.api_key (TRANSLATE_API_KEY)"),
            (value_if_set(partial.model), "api.model (TRANSLATE_MODEL)"),
        )
        if not value
    )


def build_config(partial: PartialApiSettings) -> Result[Config, TranslationError]:
    missing = missing_api_settings(partial)
    if missing:
        return fail_missing_settings(missing)

    timeout = value_if_set(partial.timeout)
    maximum_tokens = value_if_set(partial.maximum_tokens)
    parameters = value_if_set(partial.parameters)
    return Ok(
        Config(
            base_url=str(value_if_set(partial.base_url) or "").rstrip("/"),
            api_key=str(value_if_set(partial.api_key) or ""),
            model=str(value_if_set(partial.model) or ""),
            timeout=timeout if timeout is not None else DEFAULT_TIMEOUT,
            maximum_tokens=(
                maximum_tokens if maximum_tokens is not None else DEFAULT_MAXIMUM_TOKENS
            ),
            parameters=MappingProxyType(dict(parameters or {})),
        )
    )


def merged_api_settings(
    environment: Mapping[str, str],
    path: str,
    document_result: DocumentResult,
) -> Result[Config, TranslationError]:
    def document_settings(
        document_result: DocumentResult,
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(
            document_result,
            lambda document: document_api_settings(path, document),
        )

    layers = (
        document_settings(document_result),
        api_settings_from_environment(environment),
    )
    return result_bind(
        result_sequence(layers),
        lambda extras: build_config(
            reduce(PartialApiSettings.merge, extras, DEFAULT_API_SETTINGS)
        ),
    )

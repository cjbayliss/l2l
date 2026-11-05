from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from types import MappingProxyType
from typing import Any, Literal, assert_never

from l2l.effects import (
    current_working_directory,
    load_json,
    load_toml,
    path_exists,
    read_text_file,
    user_config_path,
)
from l2l.errors import TranslationError, fail_config, fail_missing_settings
from l2l.glossary import (
    GLOSSARY_ENTRY_TYPES,
    GlossaryEntry,
    merge_glossary_entries,
    parse_glossary_document,
)
from l2l.languages import (
    expected_target_scripts,
    language_base_and_region,
    normalized_source_language,
    scripts_description,
)
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_bind,
    io_map,
    io_pair,
    io_pure,
    io_result,
    io_result_bind,
    io_traverse,
    result_bind,
    result_bind_io,
    result_map,
    result_sequence,
)
from l2l.prompts import (
    JAPANESE_KOREAN_SOURCE_BASES,
    PROMPT_SECTION_NAMES,
    OutputParagraphs,
    OutputPolicy,
    OutputPolicyPatch,
    PassPromptRequest,
    PromptRole,
    PromptSectionName,
    PromptSectionOrigin,
    PromptSectionOverride,
    StylePolicy,
    StylePolicyPatch,
    render_pass_prompt,
    role_sections,
)
from l2l.settings import (
    API_SETTING_KEYS,
    DEFAULT_API_SETTINGS,
    DEFAULT_MAXIMUM_TOKENS,
    DEFAULT_TIMEOUT,
    PASS_KEYS,
    PASS_MODES,
    Arguments,
    Config,
    OptionDefaults,
    PartialApiSettings,
    PassDefinition,
    PassDraft,
    Setup,
    effective_target_language,
    paragraph_tolerance,
)

EMPTY_STYLE_PATCH = StylePolicyPatch()
EMPTY_OUTPUT_PATCH = OutputPolicyPatch()


def validate_document(
    path: str, document: dict[str, Any]
) -> Result[None, TranslationError]:
    if not document:
        return Ok(None)

    unknown = sorted(
        set(document)
        - {
            "api",
            "options",
            "pass",
            "glossary",
            "target_language",
            "source_language",
            "prompts",
            "style",
            "output",
        }
    )
    if unknown:
        return fail_config(
            "%s: unknown top-level key(s): %s (expected [api], [options], "
            "glossary, target_language, source_language, [prompts], [style], "
            "[output], "
            "[[pass]])" % (path, ", ".join(unknown))
        )

    glossary = document.get("glossary")
    if glossary is not None and (
        not isinstance(glossary, list)
        or not all(isinstance(item, str) and item.strip() for item in glossary)
    ):
        return fail_config("%s: glossary must be a list of file paths" % path)

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
        return Ok(replace(partial, maximum_tokens=int(value)))

    return Ok(replace(partial, timeout=float(value)))


def timeout_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    return positive_number_setting(path, partial, table, "timeout", whole=False)


def maximum_tokens_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    return positive_number_setting(path, partial, table, "max_tokens", whole=True)


def parameters_api_setting(
    path: str, partial: PartialApiSettings, table: Mapping[str, Any]
) -> Result[PartialApiSettings, TranslationError]:
    if "params" not in table:
        return Ok(partial)

    value = table["params"]
    if not isinstance(value, dict):
        return fail_config("%s: [api] params must be a table" % path)

    if not is_json_value(value):
        return fail_config("%s: [api] params must contain only JSON values" % path)

    return Ok(replace(partial, parameters=value))


API_FIELD_PARSERS = (
    string_api_setting,
    timeout_api_setting,
    maximum_tokens_api_setting,
    parameters_api_setting,
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


def env_string_settings(environment: Mapping[str, str]) -> PartialApiSettings:
    def value(variable: str) -> str | None:
        raw = environment.get(variable, "").strip()
        return raw or None

    return PartialApiSettings(
        base_url=value("TRANSLATE_BASE_URL"),
        api_key=value("TRANSLATE_API_KEY"),
        model=value("TRANSLATE_MODEL"),
    )


def optional_positive_setting(
    value: float | None, description: str
) -> Result[float | None, TranslationError]:
    if value is None:
        return Ok(None)
    if value <= 0:
        return fail_config("%s must be positive" % description)
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
        return fail_config("%s must be %s, got %r" % (variable, kind, raw))

    return optional_positive_setting(parsed, variable)


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


def api_settings_from_arguments(
    arguments: Arguments,
) -> Result[PartialApiSettings, TranslationError]:
    timeout = optional_positive_setting(arguments.timeout, "--timeout")
    maximum_tokens = optional_positive_setting(
        arguments.maximum_tokens, "--maximum-tokens"
    )
    return result_map(
        result_sequence((timeout, maximum_tokens)),
        lambda values: PartialApiSettings(
            base_url=arguments.base_url,
            api_key=arguments.api_key,
            model=arguments.model,
            timeout=values[0],
            maximum_tokens=(int(values[1]) if values[1] is not None else None),
        ),
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
            maximum_tokens=(
                partial.maximum_tokens
                if partial.maximum_tokens is not None
                else DEFAULT_MAXIMUM_TOKENS
            ),
            parameters=MappingProxyType(dict(partial.parameters or {})),
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
    def document_settings(
        path: str | None,
        document_result: Result[dict[str, Any], TranslationError],
    ) -> Result[PartialApiSettings, TranslationError]:
        return result_bind(
            document_result,
            lambda document: document_api_settings(path or "", document),
        )

    layers = (
        document_settings(user_path, user_document_result),
        document_settings(selected_path, selected_document_result),
        api_settings_from_environment(environment),
        api_settings_from_arguments(arguments),
    )
    return result_bind(
        result_sequence(layers),
        lambda extras: build_config(
            reduce(PartialApiSettings.merge, extras, DEFAULT_API_SETTINGS)
        ),
    )


def parse_options_table(
    path: str, table: Any
) -> Result[OptionDefaults, TranslationError]:
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
        OptionDefaults(
            ascii=ascii_value,
            ensure_paragraphs=ensure_paragraphs,
            retranslate_untranslated=retranslate_untranslated,
            ensure_paragraphs_explicit="ensure_paragraphs" in table,
        )
    )


def pass_draft_from(
    path: str,
    name: str,
    table: dict[str, Any],
    instruction: str | None,
    prompts: Mapping[PromptSectionName, PromptSectionOverride] = MappingProxyType({}),
    style_patch: StylePolicyPatch = EMPTY_STYLE_PATCH,
    output_patch: OutputPolicyPatch = EMPTY_OUTPUT_PATCH,
) -> Result[PassDraft, TranslationError]:
    mode = table.get("mode", "chunk")
    if mode not in PASS_MODES:
        return fail_config(
            "%s: [[pass]] %s: mode must be analysis, chunk, paragraph, or "
            "update_glossary" % (path, name)
        )

    if prompts and (instruction is not None or mode == "update_glossary"):
        return fail_config(
            "%s: [[pass]] %s: per-pass prompt overrides apply only to "
            "built-in rendered chunk, paragraph, or analysis passes" % (path, name)
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

    target_language = table.get("target_language")
    if target_language is not None and (
        not isinstance(target_language, str) or not target_language.strip()
    ):
        return fail_config(
            "%s: [[pass]] %s: target_language must be a non-empty string" % (path, name)
        )

    if "include_types" in table and mode != "update_glossary":
        return fail_config(
            "%s: [[pass]] %s: include_types applies only to update_glossary passes"
            % (path, name)
        )

    include_types_value = table.get("include_types", [])
    if not isinstance(include_types_value, list):
        return fail_config(
            "%s: [[pass]] %s: include_types must be a list of strings" % (path, name)
        )

    if any(value not in GLOSSARY_ENTRY_TYPES for value in include_types_value):
        return fail_config(
            "%s: [[pass]] %s: include_types entries must be chosen from: %s"
            % (path, name, ", ".join(GLOSSARY_ENTRY_TYPES))
        )

    parameters = table.get("params", {})
    if not isinstance(parameters, dict):
        return fail_config("%s: [[pass]] %s: params must be a table" % (path, name))

    if not is_json_value(parameters):
        return fail_config(
            "%s: [[pass]] %s: params must contain only JSON values" % (path, name)
        )

    return Ok(
        PassDraft(
            name=name,
            mode=mode,
            instruction=instruction,
            parameters=MappingProxyType(dict(parameters)),
            model=model.strip() if model else None,
            ascii=ascii_value,
            ensure_paragraphs=ensure_paragraphs_value,
            retranslate_untranslated=retranslate_value,
            target_language=(
                target_language.strip() if target_language is not None else None
            ),
            include_types=tuple(include_types_value),
            prompts=prompts,
            style_patch=style_patch,
            output_patch=output_patch,
            ensure_paragraphs_explicit=table.get("ensure_paragraphs"),
        )
    )


def apply_default_options(
    pass_drafts: tuple[PassDraft, ...],
    options: OptionDefaults,
    ensure_paragraphs_flag: bool | int,
) -> tuple[PassDraft, ...]:
    ensure_default = options.ensure_paragraphs or ensure_paragraphs_flag
    return tuple(
        replace(
            pass_draft,
            ascii=(options.ascii if pass_draft.ascii is None else pass_draft.ascii),
            ensure_paragraphs=(
                ensure_default
                if pass_draft.ensure_paragraphs is None
                else pass_draft.ensure_paragraphs
            ),
            retranslate_untranslated=(
                options.retranslate_untranslated
                if pass_draft.retranslate_untranslated is None
                else pass_draft.retranslate_untranslated
            ),
        )
        for pass_draft in pass_drafts
    )


def load_instruction_text(
    path: str, name: str, table: dict[str, Any], base_directory: str
) -> IO[Result[str | None, TranslationError]]:
    has_inline = "instruction" in table
    has_file = "instruction_file" in table
    mode = table.get("mode", "chunk")
    if not has_inline and not has_file:
        if mode in ("chunk", "paragraph", "update_glossary", "analysis"):
            return io_result(Ok(None))

        return io_result(
            fail_config(
                "%s: [[pass]] %s: exactly one of instruction_file "
                "or instruction is required" % (path, name)
            )
        )

    if has_file == has_inline:
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


def prompt_override_from_value(
    path: str,
    description: str,
    section: str,
    value: Any,
    inline_origin: PromptSectionOrigin,
) -> Result[PromptSectionOverride | None, TranslationError]:
    if isinstance(value, str):
        if not value.strip():
            if value == "":
                return Ok(PromptSectionOverride("", inline_origin))

            return fail_config(
                '%s: %s %s must be empty ("") to remove a section, or '
                "contain non-whitespace text" % (path, description, section)
            )

        return Ok(PromptSectionOverride(value.strip(), inline_origin))

    if not isinstance(value, dict):
        return fail_config(
            "%s: %s %s must be inline text or a table with a single "
            "file key" % (path, description, section)
        )

    if set(value) != {"file"}:
        return fail_config(
            "%s: %s %s must be a table with exactly one key, file"
            % (path, description, section)
        )

    file_value = value["file"]
    if not isinstance(file_value, str) or not file_value.strip():
        return fail_config(
            "%s: %s %s: file must be a non-empty path string"
            % (path, description, section)
        )

    return Ok(None)


def parse_prompt_overrides(
    path: str,
    table: Any,
    base_directory: str,
    description: str,
    inline_origin: PromptSectionOrigin,
    file_origin: PromptSectionOrigin,
) -> IO[Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]]:
    if not isinstance(table, dict):
        return io_result(fail_config("%s: %s must be a table" % (path, description)))

    unknown = sorted(set(table) - set(PROMPT_SECTION_NAMES))
    if unknown:
        return io_result(
            fail_config(
                "%s: %s: unknown prompt section(s): %s (expected: %s)"
                % (
                    path,
                    description,
                    ", ".join(unknown),
                    ", ".join(PROMPT_SECTION_NAMES),
                )
            )
        )

    def parse_section(
        section: str,
    ) -> IO[Result[PromptSectionOverride | None, TranslationError]]:
        value = table[section]
        shape_result = prompt_override_from_value(
            path, description, section, value, inline_origin
        )
        if isinstance(shape_result, Err) or shape_result.value is not None:
            return io_result(shape_result)

        file_path = os.path.join(base_directory, table[section]["file"])
        file_description = "%s %s: prompt file" % (description, section)

        def checked(
            text_result: Result[str, TranslationError],
        ) -> Result[PromptSectionOverride | None, TranslationError]:
            return result_bind(
                text_result,
                lambda text: (
                    Ok(PromptSectionOverride(text.strip(), file_origin))
                    if text.strip()
                    else fail_config(
                        "%s: %s %s: prompt file is empty" % (path, description, section)
                    )
                ),
            )

        return io_map(read_text_file(file_path, file_description), checked)

    def combine(
        parsed_result: Result[
            tuple[PromptSectionOverride | None, ...], TranslationError
        ],
    ) -> Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]:
        return result_bind(
            parsed_result,
            lambda parsed: Ok(
                MappingProxyType(
                    {
                        section: override
                        for section, override in zip(sorted(table), parsed, strict=True)
                        if override is not None
                    }
                )
            ),
        )

    return io_map(
        io_traverse(sorted(table), parse_section),
        combine,
    )


STYLE_KEYS = ("formality", "honorifics", "register_fidelity")

STYLE_VALUES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "formality": ("preserve", "formal", "informal"),
        "honorifics": ("keep", "drop", "titles"),
        "register_fidelity": ("preserve", "soften"),
    }
)


def style_field_value(
    path: str,
    description: str,
    key: str,
    value: Any,
) -> Result[Any, TranslationError]:
    allowed = STYLE_VALUES[key]
    if not isinstance(value, str) or value not in allowed:
        return fail_config(
            "%s: %s %s must be one of: %s"
            % (path, description, key, ", ".join(allowed))
        )

    return Ok(value)


def parse_style_table(
    path: str,
    table: Any,
    description: str,
) -> Result[StylePolicyPatch, TranslationError]:
    if not isinstance(table, dict):
        return fail_config("%s: %s must be a table" % (path, description))

    unknown = sorted(set(table) - set(STYLE_KEYS))
    if unknown:
        return fail_config(
            "%s: %s: unknown style key(s): %s (expected: %s)"
            % (path, description, ", ".join(unknown), ", ".join(STYLE_KEYS))
        )

    present_keys = tuple(key for key in STYLE_KEYS if key in table)
    return result_map(
        result_sequence(
            tuple(
                style_field_value(path, description, key, table[key])
                for key in present_keys
            )
        ),
        lambda values: StylePolicyPatch(
            **dict(zip(present_keys, values, strict=True))
        ),
    )


def document_style_patch(
    path: str | None, document: dict[str, Any]
) -> Result[StylePolicyPatch, TranslationError]:
    if "style" not in document:
        return Ok(StylePolicyPatch())

    return parse_style_table(path or "", document["style"], "[style]")


def pass_style_patch(
    path: str,
    name: str,
    table: dict[str, Any],
    instruction: str | None,
    mode: str,
) -> Result[StylePolicyPatch, TranslationError]:
    if "style" not in table:
        return Ok(StylePolicyPatch())

    if instruction is not None or mode not in ("chunk", "paragraph"):
        return fail_config(
            "%s: [[pass]] %s: per-pass style applies only to built-in "
            "rendered chunk or paragraph passes" % (path, name)
        )

    return parse_style_table(
        path,
        table["style"],
        "[[pass]] %s: [pass.style]" % name,
    )


OUTPUT_KEYS = ("paragraphs", "maximum_characters")


def parse_output_table(
    path: str,
    table: Any,
    description: str,
) -> Result[OutputPolicyPatch, TranslationError]:
    if not isinstance(table, dict):
        return fail_config("%s: %s must be a table" % (path, description))

    unknown = sorted(set(table) - set(OUTPUT_KEYS))
    if unknown:
        return fail_config(
            "%s: %s: unknown output key(s): %s (expected: %s)"
            % (path, description, ", ".join(unknown), ", ".join(OUTPUT_KEYS))
        )

    paragraphs: Result[Any, TranslationError] = Ok(None)
    if "paragraphs" in table:
        value = table["paragraphs"]
        if value not in ("exact", "approximate"):
            return fail_config(
                '%s: %s: paragraphs must be "exact" or "approximate"'
                % (path, description)
            )

        paragraphs = Ok(value)

    maximum_characters: Result[Any, TranslationError] = Ok(None)
    if "maximum_characters" in table:
        value = table["maximum_characters"]
        valid = not isinstance(value, bool) and isinstance(value, int) and value > 0
        if not valid:
            return fail_config(
                "%s: %s maximum_characters must be a positive integer"
                % (path, description)
            )

        maximum_characters = Ok(value)

    return result_map(
        result_sequence((paragraphs, maximum_characters)),
        lambda values: OutputPolicyPatch(
            paragraphs=values[0], maximum_characters=values[1]
        ),
    )


def document_output_patch(
    path: str | None, document: dict[str, Any]
) -> Result[OutputPolicyPatch, TranslationError]:
    if "output" not in document:
        return Ok(OutputPolicyPatch())

    return parse_output_table(path or "", document["output"], "[output]")


def pass_output_patch(
    path: str,
    name: str,
    table: dict[str, Any],
    instruction: str | None,
    mode: str,
) -> Result[OutputPolicyPatch, TranslationError]:
    if "output" not in table:
        return Ok(OutputPolicyPatch())

    if instruction is not None or mode not in ("chunk", "paragraph"):
        return fail_config(
            "%s: [[pass]] %s: per-pass output applies only to built-in "
            "rendered chunk or paragraph passes" % (path, name)
        )

    return parse_output_table(
        path,
        table["output"],
        "[[pass]] %s: [pass.output]" % name,
    )


def document_prompt_overrides(
    path: str | None, document: dict[str, Any]
) -> IO[Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]]:
    table = document.get("prompts")
    if table is None:
        return io_result(Ok(MappingProxyType({})))

    base_directory = os.path.dirname(os.path.abspath(path or "."))
    return parse_prompt_overrides(
        path or "",
        table,
        base_directory,
        "[prompts]",
        "global-inline",
        "global-file",
    )


def pass_prompt_overrides(
    path: str, name: str, table: dict[str, Any], base_directory: str
) -> IO[Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]]:
    if "prompts" not in table:
        return io_result(Ok(MappingProxyType({})))

    return parse_prompt_overrides(
        path,
        table["prompts"],
        base_directory,
        "[[pass]] %s: [pass.prompts]" % name,
        "pass-inline",
        "pass-file",
    )


def parse_pass_table(
    path: str, table: Any, base_directory: str
) -> IO[Result[PassDraft, TranslationError]]:
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

    name = name.strip()
    mode = table.get("mode", "chunk")

    return io_bind(
        load_instruction_text(path, name, table, base_directory),
        lambda instruction_result: io_bind(
            pass_prompt_overrides(path, name, table, base_directory),
            lambda prompts_result: io_result(
                result_bind(
                    instruction_result,
                    lambda instruction: result_bind(
                        prompts_result,
                        lambda prompts: result_bind(
                            pass_style_patch(path, name, table, instruction, mode),
                            lambda style_patch: result_bind(
                                pass_output_patch(path, name, table, instruction, mode),
                                lambda output_patch: pass_draft_from(
                                    path,
                                    name,
                                    table,
                                    instruction,
                                    prompts,
                                    style_patch,
                                    output_patch,
                                ),
                            ),
                        ),
                    ),
                )
            ),
        ),
    )


def document_passes(
    path: str, document: dict[str, Any]
) -> IO[Result[tuple[PassDraft, ...], TranslationError]]:
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

    return io_traverse(
        entries,
        lambda entry: parse_pass_table(
            path, entry, os.path.dirname(os.path.abspath(path))
        ),
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
) -> IO[Result[ResolvedPasses, TranslationError]]:
    def resolve(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> IO[Result[ResolvedPasses, TranslationError]]:
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
        options: Result[OptionDefaults, TranslationError]
        if option_sources:
            option_path, option_document = option_sources[0]
            options = parse_options_table(option_path or "", option_document["options"])
        else:
            options = Ok(OptionDefaults())

        def merged_global_prompts(
            layer_prompts: tuple[
                Mapping[PromptSectionName, PromptSectionOverride], ...
            ],
        ) -> Mapping[PromptSectionName, PromptSectionOverride]:
            return MappingProxyType(
                {
                    name: override
                    for layer in reversed(layer_prompts)
                    for name, override in layer.items()
                }
            )

        def merged_style_patch(
            layer_patches: tuple[StylePolicyPatch, ...],
        ) -> StylePolicyPatch:
            return reduce(StylePolicyPatch.merge, layer_patches, StylePolicyPatch())

        def merged_output_patch(
            layer_patches: tuple[OutputPolicyPatch, ...],
        ) -> OutputPolicyPatch:
            return reduce(OutputPolicyPatch.merge, layer_patches, OutputPolicyPatch())

        def combine(
            resolved: tuple[
                tuple[
                    Result[tuple[PassDraft, ...], TranslationError],
                    Result[
                        tuple[Mapping[PromptSectionName, PromptSectionOverride], ...],
                        TranslationError,
                    ],
                ],
                tuple[
                    Result[tuple[StylePolicyPatch, ...], TranslationError],
                    Result[tuple[OutputPolicyPatch, ...], TranslationError],
                ],
            ],
        ) -> IO[Result[ResolvedPasses, TranslationError]]:
            ((drafts_result, layer_prompts_result), styles_and_outputs) = resolved
            layer_styles_result, layer_outputs_result = styles_and_outputs
            return io_result(
                result_bind(
                    drafts_result,
                    lambda drafts: result_bind(
                        layer_prompts_result,
                        lambda layer_prompts: result_bind(
                            layer_styles_result,
                            lambda layer_styles: result_bind(
                                layer_outputs_result,
                                lambda layer_outputs: result_map(
                                    options,
                                    lambda option_values: (
                                        option_values,
                                        merged_global_prompts(layer_prompts),
                                        merged_style_patch(layer_styles),
                                        merged_output_patch(layer_outputs),
                                        drafts,
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )

        pass_path, pass_document = pass_sources[0]
        return io_bind(
            io_pair(
                io_pair(
                    document_passes(pass_path or "", pass_document),
                    io_traverse(
                        layers,
                        lambda layer: document_prompt_overrides(layer[0], layer[1]),
                    ),
                ),
                io_pair(
                    io_traverse(
                        layers,
                        lambda layer: io_result(
                            document_style_patch(layer[0], layer[1])
                        ),
                    ),
                    io_traverse(
                        layers,
                        lambda layer: io_result(
                            document_output_patch(layer[0], layer[1])
                        ),
                    ),
                ),
            ),
            combine,
        )

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

    return io_result_bind(
        load_toml(path or "", description),
        lambda document: io_result(
            result_map(validate_document(path or "", document), lambda _: document)
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

    def pick(current_working_directory_value: str, user_path: str) -> str | None:
        local = os.path.join(current_working_directory_value, "l2l.toml")
        if path_exists(local):
            return local

        return user_path if path_exists(user_path) else None

    return io_bind(
        current_directory
        if current_directory is not None
        else current_working_directory(),
        lambda current_working_directory_value: io_bind(
            user_config_path(environment),
            lambda user_path: IO(
                lambda: pick(current_working_directory_value, user_path)
            ),
        ),
    )


DocumentResult = Result[dict[str, Any], TranslationError]
ResolvedPasses = tuple[
    OptionDefaults,
    Mapping[PromptSectionName, PromptSectionOverride],
    StylePolicyPatch,
    OutputPolicyPatch,
    tuple[PassDraft, ...],
]
Glossaries = tuple[tuple[str, ...], tuple[GlossaryEntry, ...]]
SetupResult = Result[Setup, TranslationError]


def document_glossary_paths(
    path: str | None, document: dict[str, Any]
) -> Result[tuple[str, ...], TranslationError]:
    value = document.get("glossary")
    if value is None:
        return Ok(())

    if not path:
        return fail_config(
            "glossary configured in a config document without a file path"
        )

    base_directory = os.path.dirname(os.path.abspath(path))
    return Ok(tuple(os.path.join(base_directory, item) for item in value))


def glossary_paths(
    user_path: str,
    user_document: dict[str, Any],
    selected_path: str | None,
    selected_document: dict[str, Any],
    command_line_paths: tuple[str, ...],
) -> Result[tuple[str, ...], TranslationError]:
    return result_map(
        result_sequence(
            document_glossary_paths(path, document)
            for path, document in (
                (user_path, user_document),
                (selected_path, selected_document),
            )
            if document
        ),
        lambda groups: (
            tuple(path for group in groups for path in group)
            + tuple(command_line_paths)
        ),
    )


def load_glossaries(
    paths: tuple[str, ...],
) -> IO[Result[Glossaries, TranslationError]]:
    def parsed(
        loaded_result: Result[tuple[Any, ...], TranslationError],
    ) -> Result[Glossaries, TranslationError]:
        return result_bind(
            loaded_result,
            lambda loaded: result_map(
                result_sequence(
                    parse_glossary_document(item_path, document)
                    for item_path, document in zip(paths, loaded, strict=True)
                ),
                lambda groups: (paths, merge_glossary_entries(groups)),
            ),
        )

    return io_map(
        io_traverse(paths, lambda item_path: load_json(item_path, "glossary file")),
        parsed,
    )


def resolve_glossaries(
    user_path: str,
    user_document_result: Result[dict[str, Any], TranslationError],
    selected_path: str | None,
    selected_document_result: Result[dict[str, Any], TranslationError],
    command_line_paths: tuple[str, ...],
) -> IO[Result[Glossaries, TranslationError]]:
    def with_documents(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> IO[Result[Glossaries, TranslationError]]:
        return result_bind_io(
            glossary_paths(
                user_path,
                documents[0],
                selected_path,
                documents[1],
                command_line_paths,
            ),
            load_glossaries,
        )

    return result_bind_io(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        with_documents,
    )


def document_target_language(
    path: str | None, document: dict[str, Any]
) -> Result[str | None, TranslationError]:
    target_language = document.get("target_language")
    if target_language is None:
        return Ok(None)

    if not isinstance(target_language, str) or not target_language.strip():
        return fail_config(
            "%s: target_language must be a non-empty string" % (path or "config")
        )

    return Ok(target_language.strip())


def command_line_target_language(
    target_language: str | None,
) -> Result[str | None, TranslationError]:
    if target_language is None:
        return Ok(None)

    if not target_language.strip():
        return fail_config("--target-language must be a non-empty string")

    return Ok(target_language.strip())


def document_source_language(
    path: str | None, document: dict[str, Any]
) -> Result[str | None, TranslationError]:
    source_language = document.get("source_language")
    if source_language is None:
        return Ok(None)

    if not isinstance(source_language, str) or not source_language.strip():
        return fail_config(
            "%s: source_language must be a non-empty string" % (path or "config")
        )

    return Ok(source_language.strip())


def command_line_source_language(
    source_language: str | None,
) -> Result[str | None, TranslationError]:
    if source_language is None:
        return Ok(None)

    if not source_language.strip():
        return fail_config("--source-language must be a non-empty string")

    return Ok(source_language.strip())


def require_target_language(
    target_language: str | None,
) -> Result[str, TranslationError]:
    if target_language is None:
        return fail_config(
            "target language is required; pass --target-language or set "
            "target_language in a config file"
        )

    return Ok(target_language)


def resolve_target_language(
    command_line_value: str | None,
    user_path: str,
    user_document_result: DocumentResult,
    selected_path: str | None,
    selected_document_result: DocumentResult,
) -> Result[str, TranslationError]:
    def configured(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> Result[str, TranslationError]:
        layers = tuple(
            (path, document)
            for path, document in (
                (selected_path, documents[1]),
                (user_path, documents[0]),
            )
            if document
        )
        resolved = reduce(
            lambda found, layer: result_bind(
                found,
                lambda value: (
                    Ok(value)
                    if value is not None
                    else document_target_language(layer[0], layer[1])
                ),
            ),
            layers,
            command_line_target_language(command_line_value),
        )
        return result_bind(resolved, require_target_language)

    return result_bind(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        configured,
    )


def resolve_source_language(
    command_line_value: str | None,
    user_path: str,
    user_document_result: DocumentResult,
    selected_path: str | None,
    selected_document_result: DocumentResult,
) -> Result[str, TranslationError]:
    def configured(
        documents: tuple[dict[str, Any], dict[str, Any]],
    ) -> Result[str | None, TranslationError]:
        layers = tuple(
            (path, document)
            for path, document in (
                (selected_path, documents[1]),
                (user_path, documents[0]),
            )
            if document
        )
        return reduce(
            lambda found, layer: result_bind(
                found,
                lambda value: (
                    Ok(value)
                    if value is not None
                    else document_source_language(layer[0], layer[1])
                ),
            ),
            layers,
            command_line_source_language(command_line_value),
        )

    def normalized(value: str | None) -> Result[str, TranslationError]:
        return Ok(normalized_source_language(value or "auto"))

    return result_bind(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        lambda documents: result_bind(configured(documents), normalized),
    )


def setup_with_glossary(setup: Setup, glossaries: Glossaries) -> Setup:
    paths, entries = glossaries
    return replace(
        setup,
        glossary_paths=paths,
        glossary_entries=entries,
    )


def require_glossary_for_update_passes(
    command_line_paths: tuple[str, ...], setup: Setup
) -> Result[Setup, TranslationError]:
    if command_line_paths or not any(
        pass_definition.mode == "update_glossary" for pass_definition in setup.passes
    ):
        return Ok(setup)

    return fail_config(
        "an update_glossary pass requires at least one --glossary file "
        "on the command line"
    )


def selected_pass_roles(
    pass_drafts: tuple[PassDraft, ...],
) -> tuple[PromptRole | None, ...]:
    roles: list[PromptRole | None] = []
    seen_text_pass = False
    for pass_draft in pass_drafts:
        if pass_draft.mode in ("chunk", "paragraph"):
            roles.append("reviewer" if seen_text_pass else "translator")
            seen_text_pass = True
        elif pass_draft.mode == "analysis":
            roles.append("analyser")
        else:
            roles.append(None)

    return tuple(roles)


def finalized_pass_definition(
    pass_draft: PassDraft,
    instruction: str,
    prompt_role: PromptRole | None,
    prompt_sections: tuple[PromptSectionName, ...],
    prompt_omitted_sections: tuple[PromptSectionName, ...],
    prompt_section_origins: Mapping[PromptSectionName, PromptSectionOrigin],
    style_policy: StylePolicy,
    output_policy: OutputPolicy,
    unit_tolerance: int | None,
    pass_tolerance: int | None,
) -> PassDefinition:
    return PassDefinition(
        name=pass_draft.name,
        instruction=instruction,
        mode=pass_draft.mode,
        parameters=pass_draft.parameters,
        model=pass_draft.model,
        ascii=pass_draft.ascii,
        ensure_paragraphs=pass_draft.ensure_paragraphs,
        retranslate_untranslated=pass_draft.retranslate_untranslated,
        target_language=pass_draft.target_language,
        include_types=pass_draft.include_types,
        style=style_policy,
        output=output_policy,
        unit_paragraph_tolerance=unit_tolerance,
        pass_paragraph_tolerance=pass_tolerance,
        prompt_role=prompt_role,
        prompt_sections=prompt_sections,
        prompt_omitted_sections=prompt_omitted_sections,
        prompt_section_origins=prompt_section_origins,
    )


@dataclass(frozen=True)
class FinalizedPasses:
    passes: tuple[PassDefinition, ...]
    warnings: tuple[str, ...]


def resolved_paragraph_tolerances(
    mode: str,
    explicit: bool | int | None,
    output_policy: OutputPolicy,
) -> tuple[int | None, int | None, OutputParagraphs]:
    if explicit is not None:
        tolerance = paragraph_tolerance(explicit)
        effective: OutputParagraphs = (
            "approximate" if tolerance is not None and tolerance > 0 else "exact"
        )
        return (
            (0 if mode == "paragraph" else tolerance),
            tolerance,
            effective,
        )

    if output_policy.paragraphs == "approximate":
        return (1, 1, "approximate")

    return (0 if mode == "paragraph" else None, None, "exact")


def finalize_pass_draft(
    pass_draft: PassDraft,
    role: PromptRole | None,
    global_target_language: str,
    global_source_language: str,
    global_prompts: Mapping[PromptSectionName, PromptSectionOverride],
    global_style_patch: StylePolicyPatch,
    global_output_patch: OutputPolicyPatch,
    global_ensure_paragraphs: bool | int | None,
) -> Result[tuple[PassDefinition, tuple[str, ...]], TranslationError]:
    explicit = (
        pass_draft.ensure_paragraphs_explicit
        if pass_draft.ensure_paragraphs_explicit is not None
        else global_ensure_paragraphs
    )
    if pass_draft.instruction is not None or pass_draft.mode == "update_glossary":
        if (
            pass_draft.prompts
            or pass_draft.style_patch != StylePolicyPatch()
            or pass_draft.output_patch != OutputPolicyPatch()
        ):
            return fail_config(
                "pass [%s]: per-pass prompt, style, or output settings apply "
                "only to built-in rendered chunk, paragraph, or analysis passes"
                % pass_draft.name
            )

        warnings: list[str] = []
        if global_prompts and pass_draft.instruction is not None:
            warnings.append(
                "global prompt section override(s) %s do not apply to the "
                "legacy pass [%s]"
                % (", ".join(sorted(global_prompts)), pass_draft.name)
            )
        if global_style_patch != StylePolicyPatch():
            warnings.append(
                "global style settings do not apply to the pass [%s]" % pass_draft.name
            )
        if global_output_patch != OutputPolicyPatch():
            warnings.append(
                "global output settings do not apply to the pass [%s]" % pass_draft.name
            )
        instruction = (
            pass_draft.instruction if pass_draft.instruction is not None else ""
        )
        unit_tolerance, pass_tolerance, _ = resolved_paragraph_tolerances(
            pass_draft.mode, explicit, OutputPolicy()
        )
        return Ok(
            (
                finalized_pass_definition(
                    pass_draft,
                    instruction,
                    None,
                    (),
                    (),
                    {},
                    StylePolicy(),
                    OutputPolicy(),
                    unit_tolerance,
                    pass_tolerance,
                ),
                tuple(warnings),
            )
        )

    merged_prompts: dict[PromptSectionName, PromptSectionOverride] = dict(
        global_prompts
    )
    merged_prompts.update(pass_draft.prompts)
    overrides = MappingProxyType(
        {name: override.text for name, override in merged_prompts.items()}
    )
    origins = MappingProxyType(
        {name: override.origin for name, override in merged_prompts.items()}
    )
    style_policy = global_style_patch.merge(pass_draft.style_patch).resolved()
    output_policy = global_output_patch.merge(pass_draft.output_patch).resolved()
    if pass_draft.mode == "analysis":
        unit_tolerance = None
        pass_tolerance = None
        effective_output_policy = OutputPolicy()
    else:
        unit_tolerance, pass_tolerance, effective_paragraphs = (
            resolved_paragraph_tolerances(pass_draft.mode, explicit, output_policy)
        )
        effective_output_policy = OutputPolicy(
            paragraphs=effective_paragraphs,
            maximum_characters=output_policy.maximum_characters,
        )
    rendered = render_pass_prompt(
        PassPromptRequest(
            role=role if role is not None else "translator",
            source_language=global_source_language,
            target_language=effective_target_language(
                global_target_language, pass_draft.target_language
            ),
            output_policy=effective_output_policy,
            style_policy=style_policy,
            overrides=overrides,
            override_origins=origins,
        )
    )
    honorifics_warnings: tuple[str, ...] = (
        (
            "pass [%s]: honorifics = %s applies only to Japanese or Korean "
            "sources; the resolved source language is %s"
            % (pass_draft.name, style_policy.honorifics, global_source_language),
        )
        if style_policy.honorifics != "keep"
        and not uses_japanese_or_korean_honorifics(global_source_language)
        else ()
    )
    no_op_warnings = tuple(
        "pass [%s]: the empty prompt override for section %s removes "
        "nothing: the built-in %s section has no static text"
        % (pass_draft.name, name, name)
        for name, override in merged_prompts.items()
        if override.text == "" and not role_sections(rendered.role)[name].strip()
    )
    return Ok(
        (
            finalized_pass_definition(
                pass_draft,
                rendered.instruction,
                rendered.role,
                rendered.included_sections,
                rendered.omitted_sections,
                rendered.section_origins,
                style_policy,
                effective_output_policy,
                unit_tolerance,
                pass_tolerance,
            ),
            no_op_warnings + honorifics_warnings,
        )
    )


def uses_japanese_or_korean_honorifics(source_language: str) -> bool:
    components = language_base_and_region(normalized_source_language(source_language))
    return components is not None and components[0] in JAPANESE_KOREAN_SOURCE_BASES


def finalize_passes(
    pass_drafts: tuple[PassDraft, ...],
    global_target_language: str,
    global_source_language: str,
    global_prompts: Mapping[PromptSectionName, PromptSectionOverride],
    global_style_patch: StylePolicyPatch,
    global_output_patch: OutputPolicyPatch,
    global_ensure_paragraphs: bool | int | None,
) -> Result[FinalizedPasses, TranslationError]:
    roles = selected_pass_roles(pass_drafts)

    per_pass = result_sequence(
        tuple(
            finalize_pass_draft(
                pass_draft,
                role,
                global_target_language,
                global_source_language,
                global_prompts,
                global_style_patch,
                global_output_patch,
                global_ensure_paragraphs,
            )
            for pass_draft, role in zip(pass_drafts, roles, strict=True)
        )
    )
    return result_map(
        per_pass,
        lambda pairs: FinalizedPasses(
            passes=tuple(pass_definition for pass_definition, _ in pairs),
            warnings=tuple(warning for _, warnings in pairs for warning in warnings),
        ),
    )


def validate_ascii_target_language(
    setup: Setup,
) -> Result[Setup, TranslationError]:
    conflicting = tuple(
        (
            pass_definition,
            effective_target_language(
                setup.target_language, pass_definition.target_language
            ),
        )
        for pass_definition in setup.passes
        if pass_definition.ascii is True
    )

    def problem(
        indexed: tuple[int, tuple[PassDefinition, str]],
    ) -> str | None:
        position, (pass_definition, language) = indexed
        scripts = expected_target_scripts(language)
        if not scripts or "Latn" in scripts:
            return None

        return (
            "pass %d [%s]: ascii = true but target language %s uses the "
            "%s script; set ascii = false or choose a Latin-script target"
            % (position, pass_definition.name, language, scripts_description(scripts))
        )

    problems = tuple(
        message
        for message in map(problem, enumerate(conflicting, 1))
        if message is not None
    )
    if problems:
        return fail_config(problems[0])

    return Ok(setup)


def build_setup(
    config: Config,
    pair: Result[ResolvedPasses, TranslationError],
    target_language: str,
    source_language: str,
    ensure_paragraphs_flag: bool = False,
) -> Result[Setup, TranslationError]:
    def with_options(resolved: ResolvedPasses) -> Result[Setup, TranslationError]:
        (
            option_defaults,
            global_prompts,
            global_style_patch,
            global_output_patch,
            pass_drafts,
        ) = resolved
        effective = option_defaults.ensure_paragraphs or ensure_paragraphs_flag
        global_ensure = (
            effective
            if option_defaults.ensure_paragraphs_explicit or ensure_paragraphs_flag
            else None
        )
        return result_bind(
            finalize_passes(
                apply_default_options(pass_drafts, option_defaults, effective),
                target_language,
                source_language,
                global_prompts,
                global_style_patch,
                global_output_patch,
                global_ensure,
            ),
            lambda finalized: validate_ascii_target_language(
                Setup(
                    config=config,
                    passes=finalized.passes,
                    ensure_paragraphs=effective,
                    target_language=target_language,
                    source_language=source_language,
                    config_warnings=finalized.warnings,
                )
            ),
        )

    return result_bind(pair, with_options)


def read_setup_documents(
    user_path: str, selected_path: str | None
) -> IO[tuple[DocumentResult, DocumentResult]]:
    def with_user_exists(
        user_exists: bool,
    ) -> IO[tuple[DocumentResult, DocumentResult]]:
        return io_pair(
            read_document(user_path, "user config", user_exists),
            read_document(selected_path, "config file", bool(selected_path)),
        )

    return io_bind(path_exists(user_path), with_user_exists)


def assemble_setup(
    arguments: Arguments,
    environment: Mapping[str, str],
    user_path: str,
    selected_path: str | None,
    documents: tuple[DocumentResult, DocumentResult],
    resolve_pipeline_passes: bool,
) -> IO[SetupResult]:
    user_document, selected_document = documents

    def setup_with(
        pair: tuple[
            Result[ResolvedPasses, TranslationError],
            Result[Glossaries, TranslationError],
        ],
    ) -> SetupResult:
        resolved, glossaries = pair
        return result_bind(
            merged_api_settings(
                arguments,
                environment,
                user_path,
                user_document,
                selected_path,
                selected_document,
            ),
            lambda config: result_bind(
                resolve_target_language(
                    arguments.target_language,
                    user_path,
                    user_document,
                    selected_path,
                    selected_document,
                ),
                lambda target_language: result_bind(
                    resolve_source_language(
                        arguments.source_language,
                        user_path,
                        user_document,
                        selected_path,
                        selected_document,
                    ),
                    lambda source_language: result_bind(
                        build_setup(
                            config,
                            resolved,
                            target_language,
                            source_language,
                            arguments.ensure_paragraphs,
                        ),
                        lambda setup: result_bind(
                            glossaries,
                            lambda resolved_glossaries: (
                                require_glossary_for_update_passes(
                                    arguments.glossary,
                                    setup_with_glossary(setup, resolved_glossaries),
                                )
                            ),
                        ),
                    ),
                ),
            ),
        )

    return io_map(
        io_pair(
            resolve_passes(user_path, user_document, selected_path, selected_document)
            if resolve_pipeline_passes
            else io_result(
                Ok(
                    (
                        OptionDefaults(),
                        MappingProxyType({}),
                        StylePolicyPatch(),
                        OutputPolicyPatch(),
                        (),
                    )
                )
            ),
            resolve_glossaries(
                user_path,
                user_document,
                selected_path,
                selected_document,
                arguments.glossary,
            ),
        ),
        setup_with,
    )


def load_setup(
    arguments: Arguments,
    environment: Mapping[str, str],
    resolve_pipeline_passes: bool = True,
) -> IO[SetupResult]:
    def with_selected_path(
        user_path: str, selected_path: str | None
    ) -> IO[SetupResult]:
        return io_bind(
            read_setup_documents(user_path, selected_path),
            lambda documents: assemble_setup(
                arguments,
                environment,
                user_path,
                selected_path,
                documents,
                resolve_pipeline_passes,
            ),
        )

    return io_bind(
        user_config_path(environment),
        lambda user_path: io_bind(
            resolve_config_path(arguments.config, environment),
            lambda selected_path: with_selected_path(user_path, selected_path),
        ),
    )

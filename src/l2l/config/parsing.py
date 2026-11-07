from __future__ import annotations

import os
from dataclasses import replace
from types import MappingProxyType
from typing import Final
lazy from collections.abc import Mapping

from l2l.config.values import is_paragraph_tri_state
from l2l.effects import read_text_file
from l2l.errors import TranslationError, fail_config
from l2l.glossary import GLOSSARY_ENTRY_TYPES
from l2l.json_values import (
    is_json_object,
    is_non_empty_string,
    is_non_empty_table_array,
    is_string_array,
    is_string_keyed_table,
)
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_bind,
    io_map,
    io_result,
    io_traverse,
    result_bind,
)
from l2l.prompts import (
    PROMPT_SECTION_NAMES,
    OutputParagraphs,
    OutputPolicyPatch,
    PromptSectionName,
    PromptSectionOrigin,
    PromptSectionOverride,
    StyleFormality,
    StyleHonorifics,
    StylePolicyPatch,
    StyleRegisterFidelity,
)
from l2l.settings import (
    PASS_KEYS,
    UNSET,
    OptionDefaults,
    ParagraphTriState,
    PassDraft,
    Unset,
    parse_pass_mode,
)

EMPTY_STYLE_PATCH: Final = StylePolicyPatch()

EMPTY_OUTPUT_PATCH: Final = OutputPolicyPatch()

EMPTY_PROMPT_SECTION_OVERRIDES: Final[
    Mapping[PromptSectionName, PromptSectionOverride]
] = MappingProxyType({})


def parse_options_table(
    path: str, table: object
) -> Result[OptionDefaults, TranslationError]:
    if not is_string_keyed_table(table):
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
    table: Mapping[str, object],
    instruction: str | None,
    prompts: Mapping[PromptSectionName, PromptSectionOverride] = (
        EMPTY_PROMPT_SECTION_OVERRIDES
    ),
    style_patch: StylePolicyPatch = EMPTY_STYLE_PATCH,
    output_patch: OutputPolicyPatch = EMPTY_OUTPUT_PATCH,
) -> Result[PassDraft, TranslationError]:
    mode = parse_pass_mode(table.get("mode", "chunk"))
    if mode is None:
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
    if ascii_value is None:
        ascii_setting: bool | Unset = UNSET
    elif isinstance(ascii_value, bool):
        ascii_setting = ascii_value
    else:
        return fail_config(
            "%s: [[pass]] %s: ascii must be true or false" % (path, name)
        )

    ensure_paragraphs_value = table.get("ensure_paragraphs")
    if ensure_paragraphs_value is None:
        ensure_paragraphs_setting: ParagraphTriState | Unset = UNSET
    elif is_paragraph_tri_state(ensure_paragraphs_value):
        ensure_paragraphs_setting = ensure_paragraphs_value
    else:
        return fail_config(
            "%s: [[pass]] %s: ensure_paragraphs must be true, false, "
            "or a positive integer" % (path, name)
        )

    retranslate_value = table.get("retranslate_untranslated")
    if retranslate_value is None:
        retranslate_setting: bool | Unset = UNSET
    elif isinstance(retranslate_value, bool):
        retranslate_setting = retranslate_value
    else:
        return fail_config(
            "%s: [[pass]] %s: retranslate_untranslated must be true or false"
            % (path, name)
        )

    model_value = table.get("model")
    if model_value is None:
        model: str | Unset = UNSET
    elif is_non_empty_string(model_value):
        model = model_value.strip()
    else:
        return fail_config(
            "%s: [[pass]] %s: model must be a non-empty string" % (path, name)
        )

    target_language_value = table.get("target_language")
    if target_language_value is None:
        target_language: str | Unset = UNSET
    elif is_non_empty_string(target_language_value):
        target_language = target_language_value.strip()
    else:
        return fail_config(
            "%s: [[pass]] %s: target_language must be a non-empty string" % (path, name)
        )

    if "include_types" in table and mode != "update_glossary":
        return fail_config(
            "%s: [[pass]] %s: include_types applies only to update_glossary passes"
            % (path, name)
        )

    include_types_value = table.get("include_types", [])
    if not is_string_array(include_types_value):
        return fail_config(
            "%s: [[pass]] %s: include_types must be a list of strings" % (path, name)
        )

    if any(value not in GLOSSARY_ENTRY_TYPES for value in include_types_value):
        return fail_config(
            "%s: [[pass]] %s: include_types entries must be chosen from: %s"
            % (path, name, ", ".join(GLOSSARY_ENTRY_TYPES))
        )

    parameters_value = table.get("params", {})
    if not is_string_keyed_table(parameters_value):
        return fail_config("%s: [[pass]] %s: params must be a table" % (path, name))
    if not is_json_object(parameters_value):
        return fail_config(
            "%s: [[pass]] %s: params must contain only JSON values" % (path, name)
        )

    return Ok(
        PassDraft(
            name=name,
            mode=mode,
            instruction=instruction,
            parameters=MappingProxyType(dict(parameters_value)),
            model=model,
            ascii=ascii_setting,
            ensure_paragraphs=ensure_paragraphs_setting,
            retranslate_untranslated=retranslate_setting,
            target_language=target_language,
            include_types=tuple(include_types_value),
            prompts=prompts,
            style_patch=style_patch,
            output_patch=output_patch,
            ensure_paragraphs_explicit=ensure_paragraphs_setting,
        )
    )


def apply_default_options(
    pass_drafts: tuple[PassDraft, ...],
    options: OptionDefaults,
    ensure_paragraphs_flag: ParagraphTriState,
) -> tuple[PassDraft, ...]:
    ensure_default = options.ensure_paragraphs or ensure_paragraphs_flag
    return tuple(
        replace(
            pass_draft,
            ascii=(
                options.ascii
                if isinstance(pass_draft.ascii, Unset)
                else pass_draft.ascii
            ),
            ensure_paragraphs=(
                ensure_default
                if isinstance(pass_draft.ensure_paragraphs, Unset)
                else pass_draft.ensure_paragraphs
            ),
            retranslate_untranslated=(
                options.retranslate_untranslated
                if isinstance(pass_draft.retranslate_untranslated, Unset)
                else pass_draft.retranslate_untranslated
            ),
        )
        for pass_draft in pass_drafts
    )


def load_instruction_text(
    path: str, name: str, table: Mapping[str, object], base_directory: str
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
        instruction_value = table["instruction"]
        if not is_non_empty_string(instruction_value):
            return io_result(
                fail_config("%s: [[pass]] %s: instruction is empty" % (path, name))
            )

        return io_result(Ok(instruction_value.strip()))

    instruction_file_value = table["instruction_file"]
    if not is_non_empty_string(instruction_file_value):
        return io_result(
            fail_config(
                "%s: [[pass]] %s: instruction_file must be a path" % (path, name)
            )
        )

    description = "%s: [[pass]] %s: instruction file" % (path, name)
    file_path = os.path.join(base_directory, instruction_file_value)

    def checked(
        text_result: Result[str, TranslationError],
    ) -> Result[str | None, TranslationError]:
        def check(text: str) -> Result[str | None, TranslationError]:
            stripped = text.strip()
            if stripped:
                return Ok(stripped)

            return fail_config(
                "%s: [[pass]] %s: instruction file is empty" % (path, name)
            )

        return result_bind(text_result, check)

    return io_map(read_text_file(file_path, description), checked)


def prompt_override_inline_text(
    path: str,
    description: str,
    section: str,
    value: object,
    inline_origin: PromptSectionOrigin,
) -> Result[PromptSectionOverride | None, TranslationError]:
    if not isinstance(value, str):
        return Ok(None)

    if not value.strip():
        if value == "":
            return Ok(PromptSectionOverride("", inline_origin))

        return fail_config(
            '%s: %s %s must be empty ("") to remove a section, or '
            "contain non-whitespace text" % (path, description, section)
        )

    return Ok(PromptSectionOverride(value.strip(), inline_origin))


def prompt_override_file_path(
    path: str, description: str, section: str, value: object
) -> Result[str, TranslationError]:
    if not is_json_object(value):
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
    if not is_non_empty_string(file_value):
        return fail_config(
            "%s: %s %s: file must be a non-empty path string"
            % (path, description, section)
        )

    return Ok(file_value)


def sorted_prompt_section_names(
    table: Mapping[str, object],
) -> tuple[PromptSectionName, ...]:
    return tuple(
        section_name
        for section_name in sorted(PROMPT_SECTION_NAMES)
        if section_name in table
    )


def parse_prompt_overrides(
    path: str,
    table: object,
    base_directory: str,
    description: str,
    inline_origin: PromptSectionOrigin,
    file_origin: PromptSectionOrigin,
) -> IO[Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]]:
    if not is_string_keyed_table(table):
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
        inline_result = prompt_override_inline_text(
            path, description, section, value, inline_origin
        )
        if isinstance(inline_result, Err) or inline_result.value is not None:
            return io_result(inline_result)

        file_path_result = prompt_override_file_path(path, description, section, value)
        if isinstance(file_path_result, Err):
            return io_result(file_path_result)

        file_path = os.path.join(base_directory, file_path_result.value)
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

    def build(
        parsed: tuple[PromptSectionOverride | None, ...],
    ) -> Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]:
        overrides: dict[PromptSectionName, PromptSectionOverride] = {
            section_name: override
            for section_name, override in zip(
                sorted_prompt_section_names(table), parsed, strict=True
            )
            if override is not None
        }
        return Ok(MappingProxyType(overrides))

    def combine(
        parsed_result: Result[
            tuple[PromptSectionOverride | None, ...], TranslationError
        ],
    ) -> Result[Mapping[PromptSectionName, PromptSectionOverride], TranslationError]:
        return result_bind(parsed_result, build)

    return io_map(
        io_traverse(sorted(table), parse_section),
        combine,
    )


STYLE_KEYS: Final = ("formality", "honorifics", "register_fidelity")

STYLE_VALUES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "formality": ("preserve", "formal", "informal"),
        "honorifics": ("keep", "drop", "titles"),
        "register_fidelity": ("preserve", "soften"),
    }
)


def style_formality_field(
    path: str, description: str, table: Mapping[str, object]
) -> Result[StyleFormality | None, TranslationError]:
    match table.get("formality"):
        case None:
            return Ok(None)

        case "preserve":
            return Ok("preserve")

        case "formal":
            return Ok("formal")

        case "informal":
            return Ok("informal")

        case _:
            return fail_config(
                "%s: %s formality must be one of: %s"
                % (path, description, ", ".join(STYLE_VALUES["formality"]))
            )


def style_honorifics_field(
    path: str, description: str, table: Mapping[str, object]
) -> Result[StyleHonorifics | None, TranslationError]:
    match table.get("honorifics"):
        case None:
            return Ok(None)

        case "keep":
            return Ok("keep")

        case "drop":
            return Ok("drop")

        case "titles":
            return Ok("titles")

        case _:
            return fail_config(
                "%s: %s honorifics must be one of: %s"
                % (path, description, ", ".join(STYLE_VALUES["honorifics"]))
            )


def style_register_fidelity_field(
    path: str, description: str, table: Mapping[str, object]
) -> Result[StyleRegisterFidelity | None, TranslationError]:
    match table.get("register_fidelity"):
        case None:
            return Ok(None)

        case "preserve":
            return Ok("preserve")

        case "soften":
            return Ok("soften")

        case _:
            return fail_config(
                "%s: %s register_fidelity must be one of: %s"
                % (path, description, ", ".join(STYLE_VALUES["register_fidelity"]))
            )


def parse_style_table(
    path: str,
    table: object,
    description: str,
) -> Result[StylePolicyPatch, TranslationError]:
    if not is_string_keyed_table(table):
        return fail_config("%s: %s must be a table" % (path, description))

    unknown = sorted(set(table) - set(STYLE_KEYS))
    if unknown:
        return fail_config(
            "%s: %s: unknown style key(s): %s (expected: %s)"
            % (path, description, ", ".join(unknown), ", ".join(STYLE_KEYS))
        )

    formality_result = style_formality_field(path, description, table)
    if isinstance(formality_result, Err):
        return formality_result

    honorifics_result = style_honorifics_field(path, description, table)
    if isinstance(honorifics_result, Err):
        return honorifics_result

    register_fidelity_result = style_register_fidelity_field(path, description, table)
    if isinstance(register_fidelity_result, Err):
        return register_fidelity_result

    return Ok(
        StylePolicyPatch(
            formality=formality_result.value,
            honorifics=honorifics_result.value,
            register_fidelity=register_fidelity_result.value,
        )
    )


def document_style_patch(
    path: str | None, document: Mapping[str, object]
) -> Result[StylePolicyPatch, TranslationError]:
    if "style" not in document:
        return Ok(StylePolicyPatch())

    return parse_style_table(path or "", document["style"], "[style]")


def pass_style_patch(
    path: str,
    name: str,
    table: Mapping[str, object],
    instruction: str | None,
    mode: object,
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


OUTPUT_KEYS: Final = ("paragraphs", "maximum_characters")


def parse_output_table(
    path: str,
    table: object,
    description: str,
) -> Result[OutputPolicyPatch, TranslationError]:
    if not is_string_keyed_table(table):
        return fail_config("%s: %s must be a table" % (path, description))

    unknown = sorted(set(table) - set(OUTPUT_KEYS))
    if unknown:
        return fail_config(
            "%s: %s: unknown output key(s): %s (expected: %s)"
            % (path, description, ", ".join(unknown), ", ".join(OUTPUT_KEYS))
        )

    paragraphs_value = table.get("paragraphs")
    if paragraphs_value is None:
        paragraphs_setting: OutputParagraphs | None = None
    elif paragraphs_value == "exact":
        paragraphs_setting = "exact"
    elif paragraphs_value == "approximate":
        paragraphs_setting = "approximate"
    else:
        return fail_config(
            '%s: %s: paragraphs must be "exact" or "approximate"' % (path, description)
        )

    maximum_characters_value = table.get("maximum_characters")
    if maximum_characters_value is None:
        maximum_characters_setting = None
    elif (
        not isinstance(maximum_characters_value, bool)
        and isinstance(maximum_characters_value, int)
        and maximum_characters_value > 0
    ):
        maximum_characters_setting = maximum_characters_value
    else:
        return fail_config(
            "%s: %s maximum_characters must be a positive integer" % (path, description)
        )

    return Ok(
        OutputPolicyPatch(
            paragraphs=paragraphs_setting,
            maximum_characters=maximum_characters_setting,
        )
    )


def document_output_patch(
    path: str | None, document: Mapping[str, object]
) -> Result[OutputPolicyPatch, TranslationError]:
    if "output" not in document:
        return Ok(OutputPolicyPatch())

    return parse_output_table(path or "", document["output"], "[output]")


def pass_output_patch(
    path: str,
    name: str,
    table: Mapping[str, object],
    instruction: str | None,
    mode: object,
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
    path: str | None, document: Mapping[str, object]
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
    path: str, name: str, table: Mapping[str, object], base_directory: str
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
    path: str, table: object, base_directory: str
) -> IO[Result[PassDraft, TranslationError]]:
    if not is_string_keyed_table(table):
        return io_result(fail_config("%s: [[pass]] entries must be tables" % path))

    unknown = sorted(set(table) - set(PASS_KEYS))
    if unknown:
        return io_result(
            fail_config("%s: [[pass]]: unknown key(s): %s" % (path, ", ".join(unknown)))
        )

    name_value = table.get("name")
    if not is_non_empty_string(name_value):
        return io_result(
            fail_config("%s: [[pass]]: name must be a non-empty string" % path)
        )

    name = name_value.strip()
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
    path: str, document: Mapping[str, object]
) -> IO[Result[tuple[PassDraft, ...], TranslationError]]:
    entries = document.get("pass")
    if entries is None:
        return io_result(Ok(()))

    if not is_non_empty_table_array(entries):
        return io_result(
            fail_config("%s: [[pass]] must define one or more pass tables" % path)
        )

    return io_traverse(
        entries,
        lambda entry: parse_pass_table(
            path, entry, os.path.dirname(os.path.abspath(path))
        ),
    )

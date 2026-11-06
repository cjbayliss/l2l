from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Any

from l2l.config.values import is_json_value, is_paragraph_tri_state
from l2l.effects import read_text_file
from l2l.errors import TranslationError, fail_config
from l2l.glossary import GLOSSARY_ENTRY_TYPES
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
    result_map,
    result_sequence,
)
from l2l.prompts import (
    PROMPT_SECTION_NAMES,
    OutputPolicyPatch,
    PromptSectionName,
    PromptSectionOrigin,
    PromptSectionOverride,
    StylePolicyPatch,
)
from l2l.settings import (
    PASS_KEYS,
    PASS_MODES,
    OptionDefaults,
    PassDraft,
)

EMPTY_STYLE_PATCH = StylePolicyPatch()

EMPTY_OUTPUT_PATCH = OutputPolicyPatch()


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
        lambda values: StylePolicyPatch(**dict(zip(present_keys, values, strict=True))),
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

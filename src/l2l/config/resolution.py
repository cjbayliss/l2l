from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import final
lazy from collections.abc import Mapping

from l2l.config.models import DocumentResult, ResolvedPasses
from l2l.config.parsing import (
    document_output_patch,
    document_passes,
    document_prompt_overrides,
    document_style_patch,
    parse_filter_table,
    parse_options_table,
)
from l2l.errors import TranslationError, fail_config
from l2l.languages import language_base_and_region, normalized_source_language
from l2l.monads import (
    IO,
    Ok,
    Result,
    io_bind,
    io_pair,
    io_result,
    result_bind,
    result_bind_io,
    result_map,
    result_sequence,
)
from l2l.prompts import (
    JAPANESE_KOREAN_SOURCE_BASES,
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
    OptionDefaults,
    PassDefinition,
    PassDraft,
    Unset,
    effective_target_language,
    paragraph_tolerance,
    value_if_set,
)
lazy from l2l.filtering import FilterSettings
lazy from l2l.json_values import JsonObject


def resolve_filter_settings(
    path: str,
    document_result: DocumentResult,
) -> Result[FilterSettings | None, TranslationError]:
    def resolve(
        document: JsonObject,
    ) -> Result[FilterSettings | None, TranslationError]:
        if "filter" not in document:
            return Ok(None)

        # result_map widens Ok[FilterSettings] to the optional settings type.
        return result_map(
            parse_filter_table(path, document["filter"]),
            lambda settings: settings,
        )

    return result_bind(document_result, resolve)


def resolve_passes(
    path: str,
    document_result: DocumentResult,
) -> IO[Result[ResolvedPasses, TranslationError]]:
    def resolve(
        document: JsonObject,
    ) -> IO[Result[ResolvedPasses, TranslationError]]:
        if "pass" not in document:
            return io_result(
                fail_config(
                    f"no [pass.<name>] tables found; define at least one pass in {path}"
                )
            )

        options: Result[OptionDefaults, TranslationError] = (
            Ok(OptionDefaults())
            if "options" not in document
            else parse_options_table(path, document["options"])
        )

        def combine(
            resolved: tuple[
                tuple[
                    Result[tuple[PassDraft, ...], TranslationError],
                    Result[
                        Mapping[PromptSectionName, PromptSectionOverride],
                        TranslationError,
                    ],
                ],
                tuple[
                    Result[StylePolicyPatch, TranslationError],
                    Result[OutputPolicyPatch, TranslationError],
                ],
            ],
        ) -> IO[Result[ResolvedPasses, TranslationError]]:
            ((drafts_result, prompts_result), styles_and_outputs) = resolved
            styles_result, outputs_result = styles_and_outputs
            return io_result(
                result_bind(
                    drafts_result,
                    lambda drafts: result_bind(
                        prompts_result,
                        lambda prompts: result_bind(
                            styles_result,
                            lambda styles: result_bind(
                                outputs_result,
                                lambda outputs: result_map(
                                    options,
                                    lambda option_values: ResolvedPasses(
                                        option_defaults=option_values,
                                        global_prompts=prompts,
                                        global_style_patch=styles,
                                        global_output_patch=outputs,
                                        pass_drafts=drafts,
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )

        return io_bind(
            io_pair(
                io_pair(
                    document_passes(path, document),
                    document_prompt_overrides(path, document),
                ),
                io_pair(
                    io_result(document_style_patch(path, document)),
                    io_result(document_output_patch(path, document)),
                ),
            ),
            combine,
        )

    return result_bind_io(document_result, resolve)


def selected_pass_role(
    pass_draft: PassDraft, seen_text_pass: bool
) -> PromptRole | None:
    match pass_draft.mode:
        case "chunk" | "paragraph":
            return "reviewer" if seen_text_pass else "translator"

        case "analysis":
            return "analyser"

        case "update_glossary":
            return None


def selected_pass_roles(
    pass_drafts: tuple[PassDraft, ...],
) -> tuple[PromptRole | None, ...]:
    def select_remaining(
        remaining: tuple[PassDraft, ...], seen_text_pass: bool = False
    ) -> tuple[PromptRole | None, ...]:
        if not remaining:
            return ()

        current, *rest = remaining
        return (
            selected_pass_role(current, seen_text_pass),
            *select_remaining(
                tuple(rest), seen_text_pass or current.mode in ("chunk", "paragraph")
            ),
        )

    return select_remaining(pass_drafts)


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
        model=value_if_set(pass_draft.model),
        ascii=value_if_set(pass_draft.ascii),
        ensure_paragraphs=value_if_set(pass_draft.ensure_paragraphs),
        retranslate_untranslated=value_if_set(pass_draft.retranslate_untranslated),
        target_language=value_if_set(pass_draft.target_language),
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


@final
@dataclass(frozen=True, slots=True)
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
        if not isinstance(pass_draft.ensure_paragraphs_explicit, Unset)
        else global_ensure_paragraphs
    )
    if pass_draft.instruction is not None or pass_draft.mode == "update_glossary":
        if (
            pass_draft.prompts
            or pass_draft.style_patch != StylePolicyPatch()
            or pass_draft.output_patch != OutputPolicyPatch()
        ):
            return fail_config(
                f"pass [{pass_draft.name}]: per-pass prompt, style, or output "
                "settings apply only to built-in rendered chunk, paragraph, "
                "or analysis passes"
            )

        warnings: list[str] = []
        if global_prompts and pass_draft.instruction is not None:
            warnings.append(
                "global prompt section override(s) {} do not apply to the "
                "legacy pass [{}]".format(
                    ", ".join(sorted(global_prompts)), pass_draft.name
                )
            )
        if global_style_patch != StylePolicyPatch():
            warnings.append(
                f"global style settings do not apply to the pass [{pass_draft.name}]"
            )
        if global_output_patch != OutputPolicyPatch():
            warnings.append(
                f"global output settings do not apply to the pass [{pass_draft.name}]"
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

    merged_prompts: Mapping[PromptSectionName, PromptSectionOverride] = {
        **global_prompts,
        **pass_draft.prompts,
    }
    override_texts: dict[PromptSectionName, str] = {
        name: override.text for name, override in merged_prompts.items()
    }
    override_origins: dict[PromptSectionName, PromptSectionOrigin] = {
        name: override.origin for name, override in merged_prompts.items()
    }
    overrides = MappingProxyType(override_texts)
    origins = MappingProxyType(override_origins)
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
                global_target_language, value_if_set(pass_draft.target_language)
            ),
            output_policy=effective_output_policy,
            style_policy=style_policy,
            overrides=overrides,
            override_origins=origins,
        )
    )
    honorifics_warnings: tuple[str, ...] = (
        (
            f"pass [{pass_draft.name}]: honorifics = {style_policy.honorifics} "
            f"applies only to Japanese or Korean sources; the resolved source "
            f"language is {global_source_language}",
        )
        if style_policy.honorifics != "keep"
        and not uses_japanese_or_korean_honorifics(global_source_language)
        else ()
    )
    no_op_warnings = tuple(
        f"pass [{pass_draft.name}]: the empty prompt override for section {name} "
        f"removes nothing: the built-in {name} section has no static text"
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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Any

from l2l.config.models import ResolvedPasses
from l2l.config.parsing import (
    document_output_patch,
    document_passes,
    document_prompt_overrides,
    document_style_patch,
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
    io_traverse,
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
    effective_target_language,
    paragraph_tolerance,
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
                                    lambda option_values: ResolvedPasses(
                                        option_defaults=option_values,
                                        global_prompts=merged_global_prompts(
                                            layer_prompts
                                        ),
                                        global_style_patch=merged_style_patch(
                                            layer_styles
                                        ),
                                        global_output_patch=merged_output_patch(
                                            layer_outputs
                                        ),
                                        pass_drafts=drafts,
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


def selected_pass_role(
    pass_draft: PassDraft, seen_text_pass: bool
) -> PromptRole | None:
    match pass_draft.mode:
        case "chunk" | "paragraph":
            return "reviewer" if seen_text_pass else "translator"

        case "analysis":
            return "analyser"

        case _:
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

    merged_prompts: Mapping[PromptSectionName, PromptSectionOverride] = {
        **global_prompts,
        **pass_draft.prompts,
    }
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

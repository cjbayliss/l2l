from __future__ import annotations

import json
lazy from collections.abc import Mapping

from l2l.languages import (
    expected_target_scripts,
    instruction_uses_target_language,
    scripts_description,
)
from l2l.messages import chunk_budget_violation_message
from l2l.monads import Just, Maybe
from l2l.plans import (
    UnitCall,
    oversized_paragraph_sizes,
    plan_pass_chunks,
    plan_unit_calls,
    preceding_modes,
    resolve_work_groups,
)
from l2l.settings import (
    Config,
    PassDefinition,
    PassMode,
    Setup,
    effective_target_language,
)
from l2l.text import (
    dominant_script,
    estimate_tokens,
    split_paragraphs,
    unit_separators,
)
lazy from l2l.context import Context
lazy from l2l.json_values import JsonValue


def source_report_line(text: str, source_paragraphs: tuple[str, ...]) -> str:
    return (
        f"source: {len(text)} character(s), {len(source_paragraphs)} paragraph(s), "
        f"~{estimate_tokens(text)} tokens"
    )


def target_language_report_note(
    run_context: Context, pass_definition: PassDefinition
) -> str:
    language = effective_target_language(
        run_context.target_language, pass_definition.target_language
    )
    scripts = expected_target_scripts(language)
    return f"target language {language} (scripts: {scripts_description(scripts)})"


def pass_report_header(number: int, total: int, pass_definition: PassDefinition) -> str:
    return f"pass {number}/{total} [{pass_definition.name}]"


def whole_document_pass_lines(
    header: str,
    language_note: str,
    pass_definition: PassDefinition,
) -> tuple[str, ...]:
    return (
        f"{header}: {language_note}, mode={pass_definition.mode}, "
        "1 call with the whole document",
    )


def unit_call_report_lines(calls: tuple[UnitCall, ...]) -> tuple[str, ...]:
    return tuple(
        f"  unit {call.index + 1}/{call.total}: "
        f"~{estimate_tokens(call.source_chunk)} source tokens, "
        f"cache key {call.key[:12]}..."
        for call in calls
    )


def work_group_warning_lines(warning: Maybe[str]) -> tuple[str, ...]:
    return (f"  warning: {warning.value}",) if isinstance(warning, Just) else ()


def budget_violation_lines(
    pass_definition: PassDefinition,
    violations: tuple[tuple[int, int], ...],
    source_budget: int,
) -> tuple[str, ...]:
    return tuple(
        "  warning: {}".format(
            chunk_budget_violation_message(
                pass_definition.name,
                paragraph_index,
                paragraph_tokens,
                source_budget,
            )
        )
        for paragraph_index, paragraph_tokens in violations
    )


def text_pass_report_lines(
    run_context: Context,
    pass_definition: PassDefinition,
    earlier_pass_modes: frozenset[PassMode],
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
    header: str,
    language_note: str,
) -> tuple[str, ...]:
    pass_plan = plan_pass_chunks(
        run_context, pass_definition, earlier_pass_modes, source_paragraphs
    )
    plan = paragraph_plan if pass_definition.mode == "paragraph" else pass_plan.chunks
    work_groups, warning = resolve_work_groups(
        pass_definition.mode,
        source_paragraphs,
        plan,
        pass_definition.name,
        pass_plan.chunk_budget,
    )
    calls = plan_unit_calls(
        run_context,
        pass_definition,
        plan,
        work_groups,
        unit_separators(plan, separators),
        None,
        glossary=run_context.glossary_entries,
        earlier_pass_modes=earlier_pass_modes,
    )
    is_chunk = pass_definition.mode == "chunk"
    budget_note = (
        f", source budget ~{pass_plan.chunk_budget} tokens" if is_chunk else ""
    )
    violations = (
        oversized_paragraph_sizes(source_paragraphs, pass_plan.source_budget)
        if is_chunk
        else ()
    )
    return (
        f"{header}: {language_note}, mode={pass_definition.mode}, "
        f"{len(work_groups)} unit(s){budget_note}",
        *unit_call_report_lines(calls),
        *work_group_warning_lines(warning),
        *budget_violation_lines(pass_definition, violations, pass_plan.source_budget),
    )


def report_pass(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    earlier_pass_modes: frozenset[PassMode],
    paragraph_plan: tuple[tuple[str, ...], ...],
    indexed: tuple[int, PassDefinition],
) -> tuple[str, ...]:
    number, pass_definition = indexed
    language_note = target_language_report_note(run_context, pass_definition)
    header = pass_report_header(number, len(pass_definitions), pass_definition)
    match pass_definition.mode:
        case "analysis" | "update_glossary":
            return whole_document_pass_lines(header, language_note, pass_definition)

        case _:
            return text_pass_report_lines(
                run_context,
                pass_definition,
                earlier_pass_modes,
                source_paragraphs,
                separators,
                paragraph_plan,
                header,
                language_note,
            )


def plan_report(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
) -> str:
    source_paragraphs, separators = split_paragraphs(text)
    earlier_pass_modes = preceding_modes(pass_definitions)
    paragraph_plan = tuple((paragraph,) for paragraph in source_paragraphs)
    return "\n".join(
        (
            source_report_line(text, source_paragraphs),
            *(
                line
                for earlier_modes, indexed in zip(
                    earlier_pass_modes, enumerate(pass_definitions, 1), strict=True
                )
                for line in report_pass(
                    run_context,
                    pass_definitions,
                    source_paragraphs,
                    separators,
                    earlier_modes,
                    paragraph_plan,
                    indexed,
                )
            ),
        )
    )


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "***"

    return api_key[:4] + "..." + api_key[-2:]


def parameters_text(parameters: Mapping[str, JsonValue]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, ensure_ascii=False, default=str)


def setup_warnings(setup: Setup) -> tuple[str, ...]:
    target_scripts = expected_target_scripts(setup.target_language)
    unknown_language_warnings: tuple[str, ...] = (
        (
            f"target language {setup.target_language} is not recognized; "
            "script checks fall back to the ASCII-letter heuristic",
        )
        if not target_scripts
        else ()
    )
    instruction_warnings = tuple(
        f"pass {number} [{pass_definition.name}]: instruction does not mention "
        f"the target language {language}; add {{target_language}} to the instruction"
        for number, pass_definition in enumerate(setup.passes, 1)
        for language in (
            effective_target_language(
                setup.target_language, pass_definition.target_language
            ),
        )
        if pass_definition.mode != "update_glossary"
        and pass_definition.instruction.strip()
        and not instruction_uses_target_language(pass_definition.instruction, language)
    )
    glossary_warnings = tuple(
        f"glossary entry {entry.original!r} renders as {rendering!r}, which is "
        f"not in the target script(s) {scripts_description(target_scripts)}; "
        "the glossary may come from another target language"
        for entry in setup.glossary_entries
        for rendering in (entry.translated or "",)
        for script in (dominant_script(rendering),)
        if rendering.strip()
        and target_scripts
        and script
        and script not in target_scripts
    )
    return setup.config_warnings + (
        unknown_language_warnings + instruction_warnings + glossary_warnings
    )


def api_setup_lines(config: Config) -> tuple[str, ...]:
    return (
        f"api.base_url: {config.base_url}",
        f"api.model: {config.model}",
        f"api.timeout: {config.timeout:g}",
        f"api.max_tokens: {config.maximum_tokens}",
        f"api.api_key: {mask_api_key(config.api_key)}",
        *(
            (f"api.params: {parameters_text(config.parameters)}",)
            if config.parameters
            else ()
        ),
    )


def prompt_label(pass_definition: PassDefinition) -> str:
    return (
        f"built-in role={pass_definition.prompt_role}"
        if pass_definition.prompt_role is not None
        else "legacy instruction"
    )


def prompt_metadata_lines(pass_definition: PassDefinition) -> tuple[str, ...]:
    if pass_definition.prompt_role is None:
        return ()

    included = ", ".join(
        "{}={}".format(
            name, pass_definition.prompt_section_origins.get(name, "built-in")
        )
        for name in pass_definition.prompt_sections
    )
    omitted = (
        ", ".join(pass_definition.prompt_omitted_sections)
        if pass_definition.prompt_omitted_sections
        else "none"
    )
    return (
        f"  prompt sections: {included}",
        f"  prompt omitted sections: {omitted}",
        "  instruction text:",
        *(f"    {line}" for line in pass_definition.instruction.splitlines()),
    )


def pass_setup_lines(
    setup: Setup, number: int, pass_definition: PassDefinition
) -> tuple[str, ...]:
    language = effective_target_language(
        setup.target_language, pass_definition.target_language
    )
    uses_placeholder = "{target_language}" in pass_definition.instruction
    summary = (
        f"pass {number}/{len(setup.passes)} [{pass_definition.name}]: "
        f"mode={pass_definition.mode} ascii={pass_definition.ascii} "
        f"ensure_paragraphs={pass_definition.ensure_paragraphs} "
        f"retranslate_untranslated={pass_definition.retranslate_untranslated} "
        f"model={pass_definition.model or '<default>'} "
        f"instruction={len(pass_definition.instruction)} chars "
        f"target_language={language} "
        f"scripts={scripts_description(expected_target_scripts(language))} "
        f"uses_target_language_placeholder={uses_placeholder} "
        f"style=formality:{pass_definition.style.formality} "
        f"honorifics:{pass_definition.style.honorifics} "
        f"register_fidelity:{pass_definition.style.register_fidelity} "
        f"prompt={prompt_label(pass_definition)}"
    )
    parameters = (
        (f"  params: {parameters_text(pass_definition.parameters)}",)
        if pass_definition.parameters
        else ()
    )
    output_values = (
        ("paragraphs", pass_definition.output.paragraphs),
        ("maximum_characters", pass_definition.output.maximum_characters),
        ("unit_tolerance", pass_definition.unit_paragraph_tolerance),
        ("pass_tolerance", pass_definition.pass_paragraph_tolerance),
    )
    output_line = "  output: {}".format(
        " ".join(
            "{}={}".format(label, "unset" if value is None else value)
            for label, value in output_values
        )
    )
    return (summary, *parameters, output_line, *prompt_metadata_lines(pass_definition))


def glossary_setup_lines(setup: Setup) -> tuple[str, ...]:
    if not setup.glossary_paths:
        return ()

    return (
        f"glossary: {len(setup.glossary_paths)} file(s), "
        f"{len(setup.glossary_entries)} entries",
        *((f"  - {path}") for path in setup.glossary_paths),
    )


def setup_report(setup: Setup, effective_ensure_paragraphs: bool | int) -> str:
    global_scripts = expected_target_scripts(setup.target_language)
    return "\n".join(
        (
            *api_setup_lines(setup.config),
            f"target_language: {setup.target_language} "
            f"(scripts: {scripts_description(global_scripts)})",
            f"source_language: {setup.source_language}",
            *tuple(
                line
                for number, pass_definition in enumerate(setup.passes, 1)
                for line in pass_setup_lines(setup, number, pass_definition)
            ),
            *glossary_setup_lines(setup),
            f"options.ensure_paragraphs: {effective_ensure_paragraphs}",
        )
    )

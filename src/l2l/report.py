from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from l2l.context import Context
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


def source_report_line(text: str, source_paragraphs: tuple[str, ...]) -> str:
    return "source: %d character(s), %d paragraph(s), ~%d tokens" % (
        len(text),
        len(source_paragraphs),
        estimate_tokens(text),
    )


def target_language_report_note(
    run_context: Context, pass_definition: PassDefinition
) -> str:
    language = effective_target_language(run_context.target_language, pass_definition)
    scripts = expected_target_scripts(language)
    return "target language %s (scripts: %s)" % (
        language,
        scripts_description(scripts),
    )


def pass_report_header(number: int, total: int, pass_definition: PassDefinition) -> str:
    return "pass %d/%d [%s]" % (number, total, pass_definition.name)


def whole_document_pass_lines(
    header: str,
    language_note: str,
    pass_definition: PassDefinition,
) -> tuple[str, ...]:
    return (
        "%s: %s, mode=%s, 1 call with the whole document"
        % (header, language_note, pass_definition.mode),
    )


def unit_call_report_lines(calls: tuple[UnitCall, ...]) -> tuple[str, ...]:
    return tuple(
        "  unit %d/%d: ~%d source tokens, cache key %s..."
        % (
            call.index + 1,
            call.total,
            estimate_tokens(call.source_chunk),
            call.key[:12],
        )
        for call in calls
    )


def work_group_warning_lines(warning: Maybe[str]) -> tuple[str, ...]:
    return ("  warning: %s" % warning.value,) if isinstance(warning, Just) else ()


def budget_violation_lines(
    pass_definition: PassDefinition,
    violations: tuple[tuple[int, int], ...],
    source_budget: int,
) -> tuple[str, ...]:
    return tuple(
        "  warning: %s"
        % chunk_budget_violation_message(
            pass_definition.name,
            paragraph_index,
            paragraph_tokens,
            source_budget,
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
    )
    is_chunk = pass_definition.mode == "chunk"
    budget_note = (
        ", source budget ~%d tokens" % pass_plan.chunk_budget if is_chunk else ""
    )
    violations = (
        oversized_paragraph_sizes(source_paragraphs, pass_plan.source_budget)
        if is_chunk
        else ()
    )
    return (
        "%s: %s, mode=%s, %d unit(s)%s"
        % (
            header,
            language_note,
            pass_definition.mode,
            len(work_groups),
            budget_note,
        ),
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


def parameters_text(parameters: Mapping[str, Any]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, ensure_ascii=False, default=str)


def setup_warnings(setup: Setup) -> tuple[str, ...]:
    target_scripts = expected_target_scripts(setup.target_language)
    unknown_language_warnings: tuple[str, ...] = (
        (
            "target language %s is not recognized; script checks fall back to "
            "the ASCII-letter heuristic" % setup.target_language,
        )
        if not target_scripts
        else ()
    )
    instruction_warnings = tuple(
        "pass %d [%s]: instruction does not mention the target language %s; "
        "add {target_language} to the instruction"
        % (
            number,
            pass_definition.name,
            language,
        )
        for number, pass_definition in enumerate(setup.passes, 1)
        for language in (
            effective_target_language(setup.target_language, pass_definition),
        )
        if pass_definition.mode != "update_glossary"
        and pass_definition.instruction.strip()
        and not instruction_uses_target_language(pass_definition.instruction, language)
    )
    glossary_warnings = tuple(
        "glossary entry %r renders as %r, which is not in the target "
        "script(s) %s; the glossary may come from another target language"
        % (
            entry.original,
            rendering,
            scripts_description(target_scripts),
        )
        for entry in setup.glossary_entries
        for rendering in (entry.attributes.get("translated", ""),)
        for script in (dominant_script(rendering),)
        if rendering.strip()
        and target_scripts
        and script
        and script not in target_scripts
    )
    return unknown_language_warnings + instruction_warnings + glossary_warnings


def api_setup_lines(config: Config) -> tuple[str, ...]:
    return (
        "api.base_url: %s" % config.base_url,
        "api.model: %s" % config.model,
        "api.timeout: %g" % config.timeout,
        "api.max_tokens: %d" % config.maximum_tokens,
        "api.api_key: %s" % mask_api_key(config.api_key),
        *(
            ("api.params: %s" % parameters_text(config.parameters),)
            if config.parameters
            else ()
        ),
    )


def pass_setup_lines(
    setup: Setup, number: int, pass_definition: PassDefinition
) -> tuple[str, ...]:
    language = effective_target_language(setup.target_language, pass_definition)
    summary = (
        "pass %d/%d [%s]: mode=%s ascii=%s ensure_paragraphs=%s "
        "retranslate_untranslated=%s model=%s instruction=%d chars "
        "target_language=%s scripts=%s uses_target_language_placeholder=%s"
        % (
            number,
            len(setup.passes),
            pass_definition.name,
            pass_definition.mode,
            pass_definition.ascii,
            pass_definition.ensure_paragraphs,
            pass_definition.retranslate_untranslated,
            pass_definition.model or "<default>",
            len(pass_definition.instruction),
            language,
            scripts_description(expected_target_scripts(language)),
            "{target_language}" in pass_definition.instruction,
        )
    )
    parameters = (
        ("  params: %s" % parameters_text(pass_definition.parameters),)
        if pass_definition.parameters
        else ()
    )
    return (summary, *parameters)


def glossary_setup_lines(setup: Setup) -> tuple[str, ...]:
    if not setup.glossary_paths:
        return ()

    return (
        "glossary: %d file(s), %d entries"
        % (len(setup.glossary_paths), len(setup.glossary_entries)),
        *(("  - %s" % path) for path in setup.glossary_paths),
    )


def setup_report(setup: Setup, effective_ensure_paragraphs: bool | int) -> str:
    global_scripts = expected_target_scripts(setup.target_language)
    return "\n".join(
        api_setup_lines(setup.config)
        + (
            "target_language: %s (scripts: %s)"
            % (setup.target_language, scripts_description(global_scripts)),
        )
        + tuple(
            line
            for number, pass_definition in enumerate(setup.passes, 1)
            for line in pass_setup_lines(setup, number, pass_definition)
        )
        + glossary_setup_lines(setup)
        + ("options.ensure_paragraphs: %s" % effective_ensure_paragraphs,)
    )

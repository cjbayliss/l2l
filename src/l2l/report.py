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
from l2l.monads import Just
from l2l.plans import (
    oversized_paragraph_sizes,
    plan_pass_chunks,
    plan_unit_calls,
    preceding_modes,
    resolve_work_groups,
)
from l2l.settings import PassDefinition, Setup, effective_target_language
from l2l.text import (
    dominant_script,
    estimate_tokens,
    split_paragraphs,
    unit_separators,
)


def plan_report(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
) -> str:
    source_paragraphs, separators = split_paragraphs(text)
    earlier_pass_modes = preceding_modes(pass_definitions)
    paragraph_plan = tuple((paragraph,) for paragraph in source_paragraphs)

    def report_pass(indexed: tuple[int, PassDefinition]) -> tuple[str, ...]:
        number, pass_definition = indexed
        language = effective_target_language(
            run_context.target_language, pass_definition
        )
        scripts = expected_target_scripts(language)
        language_note = "target language %s (scripts: %s)" % (
            language,
            scripts_description(scripts),
        )
        header = "pass %d/%d [%s]" % (
            number,
            len(pass_definitions),
            pass_definition.name,
        )
        match pass_definition.mode:
            case "analysis":
                return (
                    "%s: %s, mode=analysis, 1 call with the whole document"
                    % (header, language_note),
                )

            case "update_glossary":
                return (
                    "%s: %s, mode=update_glossary, 1 call with the whole document"
                    % (header, language_note),
                )

            case _:
                pass_plan = plan_pass_chunks(
                    run_context,
                    pass_definition,
                    earlier_pass_modes[number - 1],
                    source_paragraphs,
                )
                plan = (
                    paragraph_plan
                    if pass_definition.mode == "paragraph"
                    else pass_plan.chunks
                )
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
                budget_note = (
                    ", source budget ~%d tokens" % pass_plan.chunk_budget
                    if pass_definition.mode == "chunk"
                    else ""
                )
                violations = (
                    oversized_paragraph_sizes(
                        source_paragraphs, pass_plan.source_budget
                    )
                    if pass_definition.mode == "chunk"
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
                    *(
                        "  unit %d/%d: ~%d source tokens, cache key %s..."
                        % (
                            call.index + 1,
                            call.total,
                            estimate_tokens(call.source_chunk),
                            call.key[:12],
                        )
                        for call in calls
                    ),
                    *(
                        ("  warning: %s" % warning.value,)
                        if isinstance(warning, Just)
                        else ()
                    ),
                    *(
                        "  warning: %s"
                        % chunk_budget_violation_message(
                            pass_definition.name,
                            paragraph_index,
                            paragraph_tokens,
                            pass_plan.source_budget,
                        )
                        for paragraph_index, paragraph_tokens in violations
                    ),
                )

    lines = (
        "source: %d character(s), %d paragraph(s), ~%d tokens"
        % (len(text), len(source_paragraphs), estimate_tokens(text)),
        *(
            line
            for indexed in enumerate(pass_definitions, 1)
            for line in report_pass(indexed)
        ),
    )
    return "\n".join(lines)


def mask_api_key(api_key: str) -> str:
    if len(api_key) <= 8:
        return "***"

    return api_key[:4] + "..." + api_key[-2:]


def parameters_text(parameters: Mapping[str, Any]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, ensure_ascii=False, default=str)


def setup_warnings(setup: Setup) -> tuple[str, ...]:
    unknown_language_warnings: tuple[str, ...] = (
        (
            "target language %s is not recognized; script checks fall back to "
            "the ASCII-letter heuristic" % setup.target_language,
        )
        if not expected_target_scripts(setup.target_language)
        else ()
    )
    instruction_warnings = tuple(
        "pass %d [%s]: instruction does not mention the target language %s; "
        "add {target_language} to the instruction"
        % (
            number,
            pass_definition.name,
            effective_target_language(setup.target_language, pass_definition),
        )
        for number, pass_definition in enumerate(setup.passes, 1)
        if pass_definition.mode != "update_glossary"
        and pass_definition.instruction.strip()
        and not instruction_uses_target_language(
            pass_definition.instruction,
            effective_target_language(setup.target_language, pass_definition),
        )
    )
    glossary_warnings = tuple(
        "glossary entry %r renders as %r, which is not in the target "
        "script(s) %s; the glossary may come from another target language"
        % (
            entry.original,
            rendering,
            scripts_description(expected_target_scripts(setup.target_language)),
        )
        for entry in setup.glossary_entries
        for rendering in (entry.attributes.get("translated", ""),)
        if rendering.strip()
        and expected_target_scripts(setup.target_language)
        and dominant_script(rendering)
        and dominant_script(rendering)
        not in expected_target_scripts(setup.target_language)
    )
    return unknown_language_warnings + instruction_warnings + glossary_warnings


def setup_report(setup: Setup, effective_ensure_paragraphs: bool | int) -> str:
    config = setup.config
    api_lines: tuple[str, ...] = (
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
    global_scripts = expected_target_scripts(setup.target_language)
    pass_lines = tuple(
        line
        for number, pass_definition in enumerate(setup.passes, 1)
        for line in (
            (
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
                    effective_target_language(setup.target_language, pass_definition),
                    scripts_description(
                        expected_target_scripts(
                            effective_target_language(
                                setup.target_language, pass_definition
                            )
                        )
                    ),
                    "{target_language}" in pass_definition.instruction,
                ),
                *(
                    ("  params: %s" % parameters_text(pass_definition.parameters),)
                    if pass_definition.parameters
                    else ()
                ),
            )
        )
    )
    glossary_lines: tuple[str, ...] = (
        (
            "glossary: %d file(s), %d entries"
            % (len(setup.glossary_paths), len(setup.glossary_entries)),
        )
        + tuple("  - %s" % path for path in setup.glossary_paths)
        if setup.glossary_paths
        else ()
    )
    return "\n".join(
        api_lines
        + (
            "target_language: %s (scripts: %s)"
            % (setup.target_language, scripts_description(global_scripts)),
        )
        + pass_lines
        + glossary_lines
        + ("options.ensure_paragraphs: %s" % effective_ensure_paragraphs,)
    )

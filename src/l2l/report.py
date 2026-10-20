from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from l2l.context import Context
from l2l.messages import chunk_budget_violation_message
from l2l.monads import Just
from l2l.plans import (
    oversized_paragraph_sizes,
    plan_pass_chunks,
    plan_unit_calls,
    preceding_modes,
    resolve_work_groups,
)
from l2l.settings import PassDefinition, Setup
from l2l.text import estimate_tokens, split_paragraphs, unit_separators


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
        header = "pass %d/%d [%s]" % (
            number,
            len(pass_definitions),
            pass_definition.name,
        )
        match pass_definition.mode:
            case "analysis":
                return ("%s: mode=analysis, 1 call with the whole document" % header,)

            case "update_glossary":
                return (
                    "%s: mode=update_glossary, 1 call with the whole document, "
                    "target language %s"
                    % (
                        header,
                        pass_definition.target_language or run_context.target_language,
                    ),
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
                    "%s: mode=%s, %d unit(s)%s"
                    % (
                        header,
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
    pass_lines = tuple(
        line
        for number, pass_definition in enumerate(setup.passes, 1)
        for line in (
            (
                "pass %d/%d [%s]: mode=%s ascii=%s ensure_paragraphs=%s "
                "retranslate_untranslated=%s model=%s instruction=%d chars"
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
        + pass_lines
        + glossary_lines
        + ("options.ensure_paragraphs: %s" % effective_ensure_paragraphs,)
    )

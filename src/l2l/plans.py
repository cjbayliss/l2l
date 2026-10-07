from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate, chain
from typing import Any

from l2l.errors import HttpError, TranslationError
from l2l.monads import IO, NOTHING, Just, Maybe
from l2l.settings import (
    Context,
    PassDefinition,
    PassMode,
    Settings,
    Setup,
    pass_salt,
    resolve_call_settings,
    retranslation_salt,
)
from l2l.text import (
    AsciiDrop,
    cache_key,
    count_paragraphs,
    dominant_script,
    estimate_tokens,
    make_chunks,
    non_ascii_sample,
    regroup_by_plan,
    split_paragraphs,
    unit_separators,
)


def verbose_log(run_context: Context, message: str) -> IO[None]:
    return run_context.console.log_verbose(message)


def transient(error: TranslationError) -> bool:
    match error:
        case HttpError(kind="unreachable"):
            return True

        case HttpError(kind="status", status=int(code)):
            return code == 429 or code >= 500

        case _:
            return False


def plan_backoff(base_delay: float, cap: float, attempts: int) -> tuple[float, ...]:
    def delay(index: int) -> float:
        exponential: float = base_delay * 2.0**index
        return min(exponential, cap)

    return tuple(delay(index) for index in range(max(attempts, 0)))


def retry_delay(error: TranslationError, planned: float) -> float:
    match error:
        case HttpError(retry_after=float(seconds)):
            return max(planned, seconds)

        case _:
            return planned


def unit_output_problem(
    source_text: str, output: str, settings: Settings, tolerance: int | None
) -> Maybe[str]:
    if not output.strip():
        return Just("the reply was empty")

    if tolerance is not None:
        expected = count_paragraphs(source_text)
        found = count_paragraphs(output)
        if abs(found - expected) > tolerance:
            allowed = "" if tolerance <= 0 else " (allowed ±%d)" % tolerance
            return Just(
                "the reply has %d paragraph(s) but the source has %d%s"
                % (found, expected, allowed)
            )

    source_estimate = estimate_tokens(source_text)
    output_estimate = estimate_tokens(output)
    if output_estimate > settings.unit_output_max_ratio * max(source_estimate, 1):
        return Just(
            "the reply is ~%d tokens against a source of ~%d tokens "
            "(limit %.0fx)"
            % (
                output_estimate,
                source_estimate,
                settings.unit_output_max_ratio,
            )
        )

    return NOTHING


def script_mismatch_problem(
    source_text: str, work_text: str, output: str
) -> Maybe[str]:
    source_script = dominant_script(source_text)
    if not source_script or source_script == dominant_script(work_text):
        return NOTHING

    if dominant_script(output) != source_script:
        return NOTHING

    return Just(
        "the reply is in the source language (%s script) instead of the "
        "draft's language" % source_script
    )


def context_parts(context: tuple[str, ...]) -> tuple[str, ...]:
    if not context:
        return ()

    return (
        "Context paragraphs (reference only — do not translate them, do not "
        "continue the story from them, and do not include them in the "
        "output):",
        *context,
        "Translate only the paragraph that follows, as exactly one paragraph:",
    )


def build_pass_user(
    source_chunk: str,
    work_chunk: str | None,
    analysis: str | None,
    context: tuple[str, ...] = (),
) -> str:
    has_draft = work_chunk is not None and work_chunk != source_chunk
    source_block = ("Source text:\n%s" % source_chunk) if has_draft else source_chunk
    draft_block = ("Current draft:\n%s" % work_chunk) if has_draft else ""
    parts = (
        analysis.strip() if analysis else "",
        *context_parts(context),
        source_block,
        draft_block,
    )
    return "\n\n".join(part for part in parts if part)


def plan_information_message(
    pass_definition: PassDefinition,
    work_paragraphs: tuple[str, ...],
    work_groups: tuple[tuple[str, ...], ...],
) -> str:
    match pass_definition.mode:
        case "paragraph":
            return "l2l: [%s] %d paragraph(s), one call per paragraph" % (
                pass_definition.name,
                len(work_groups),
            )

        case _:
            return "l2l: [%s] %d paragraph(s) in %d chunk(s)" % (
                pass_definition.name,
                len(work_paragraphs),
                len(work_groups),
            )


def resolve_work_groups(
    mode: PassMode,
    work_paragraphs: tuple[str, ...],
    plan: tuple[tuple[str, ...], ...],
    pass_name: str,
    budget: int,
) -> tuple[tuple[tuple[str, ...], ...], Maybe[str]]:
    grouped = regroup_by_plan(work_paragraphs, plan)
    if isinstance(grouped, Just):
        return grouped.value, NOTHING

    warning: Maybe[str] = Just(
        "l2l: [%s] paragraph count changed by a previous pass; "
        "grouping working text independently" % pass_name
    )
    match mode:
        case "paragraph":
            return tuple((paragraph,) for paragraph in work_paragraphs), warning

        case _:
            return make_chunks(work_paragraphs, budget), warning


def build_ascii_fix_user(source_paragraph: str, output_paragraph: str) -> str:
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must use ASCII characters only):\n"
        "%s\n\n"
        "Rewrite the translated paragraph using ASCII characters only, "
        "preserving its meaning, register, and language."
        % (source_paragraph, output_paragraph)
    )


def build_ascii_retry_user(
    source_paragraph: str, output_paragraph: str, result: str
) -> str:
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must use ASCII characters only):\n"
        "%s\n\n"
        "Your previous reply still contained these non-ASCII "
        "characters: %s. Rewrite the translated "
        "paragraph again, finding an equivalent ASCII formulation for "
        "every one of them from the source and context. Reply with ASCII "
        "characters only."
        % (source_paragraph, output_paragraph, non_ascii_sample(result))
    )


def build_unit_retry_user(
    source_chunk: str,
    work_chunk: str | None,
    context: tuple[str, ...],
    bad_output: str,
    problem: str,
) -> str:
    return "\n\n".join(
        (
            "Your previous reply below does not satisfy the output rules: %s."
            % problem,
            "Previous reply:\n%s" % bad_output,
            build_pass_user(source_chunk, work_chunk, None, context),
            "Translate the source text again, fixing the problem; output only "
            "the translation.",
        )
    )


def ascii_drop_warning(drop: AsciiDrop) -> str:
    return (
        "l2l: ascii: warning: paragraph %d still contained "
        "non-ASCII characters (%s) after %d LLM attempts; dropping "
        "them" % (drop.index + 1, drop.sample, drop.attempts)
    )


@dataclass(frozen=True)
class UnitCall:
    index: int
    total: int
    source_chunk: str
    work_chunk: str
    context: tuple[str, ...]
    analysis: str | None
    key: str
    trailing_separator: str
    model: str
    parameters: Mapping[str, Any]


def build_unit_call(
    model: str,
    parameters: Mapping[str, Any],
    salt: str,
    separators: tuple[str, ...],
    index: int,
    separator_index: int,
    total: int,
    source_chunk: str,
    work_chunk: str,
    context: tuple[str, ...],
    analysis: str | None,
) -> UnitCall:
    return UnitCall(
        index=index,
        total=total,
        source_chunk=source_chunk,
        work_chunk=work_chunk,
        context=context,
        analysis=analysis,
        key=cache_key(
            source_chunk,
            model,
            salt,
            work_chunk,
            overrides=parameters,
            context="\n\n".join(context),
            analysis=analysis or "",
        ),
        trailing_separator=(
            separators[separator_index] if separator_index < len(separators) else ""
        ),
        model=model,
        parameters=parameters,
    )


def plan_unit_calls(
    run_context: Context,
    pass_definition: PassDefinition,
    plan: tuple[tuple[str, ...], ...],
    work_groups: tuple[tuple[str, ...], ...],
    trailing_separators: tuple[str, ...],
    analysis: str | None,
    retry_attempt: int = 0,
) -> tuple[UnitCall, ...]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    flat_plan = tuple(chain.from_iterable(plan))
    plan_starts = tuple(accumulate(map(len, plan), initial=0))

    def neighbour_context(index: int) -> tuple[str, ...]:
        match pass_definition.mode:
            case "paragraph":
                if index >= len(plan):
                    return ()

                start = plan_starts[index]
                end = plan_starts[index + 1]
                return (flat_plan[start - 1 : start] if start > 0 else ()) + (
                    flat_plan[end : end + 1] if end < len(flat_plan) else ()
                )

            case _:
                return ()

    def call(index: int, work_group: tuple[str, ...]) -> UnitCall:
        return build_unit_call(
            model,
            parameters,
            pass_salt(pass_definition, retry_attempt),
            trailing_separators,
            index,
            index,
            len(work_groups),
            "\n\n".join(plan[index]) if index < len(plan) else "",
            "\n\n".join(work_group),
            neighbour_context(index),
            analysis,
        )

    return tuple(call(index, group) for index, group in enumerate(work_groups))


def plan_retranslation_calls(
    run_context: Context,
    pass_definition: PassDefinition,
    source_paragraphs: tuple[str, ...],
    work_paragraphs: tuple[str, ...],
    flagged: tuple[int, ...],
    separators: tuple[str, ...],
    analysis: str | None,
) -> tuple[UnitCall, ...]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)

    def neighbour_context(index: int) -> tuple[str, ...]:
        context: tuple[str, ...] = ()
        if index > 0:
            context += (source_paragraphs[index - 1],)

        if index + 1 < len(source_paragraphs):
            context += (source_paragraphs[index + 1],)

        return context

    def call(position: int, index: int) -> UnitCall:
        return build_unit_call(
            model,
            parameters,
            retranslation_salt(pass_definition),
            separators,
            position,
            index,
            len(flagged),
            source_paragraphs[index],
            work_paragraphs[index],
            neighbour_context(index),
            analysis,
        )

    return tuple(call(position, index) for position, index in enumerate(flagged))


def plan_report(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
) -> str:
    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(
        source_paragraphs, run_context.settings.chunk_budget_tokens
    )
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

            case _:
                plan = (
                    paragraph_plan
                    if pass_definition.mode == "paragraph"
                    else chunk_plan
                )
                work_groups, warning = resolve_work_groups(
                    pass_definition.mode,
                    source_paragraphs,
                    plan,
                    pass_definition.name,
                    run_context.settings.chunk_budget_tokens,
                )
                calls = plan_unit_calls(
                    run_context,
                    pass_definition,
                    plan,
                    work_groups,
                    unit_separators(plan, separators),
                    None,
                )
                return (
                    "%s: mode=%s, %d unit(s)"
                    % (header, pass_definition.mode, len(work_groups)),
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
    return "\n".join(
        api_lines
        + pass_lines
        + ("options.ensure_paragraphs: %s" % effective_ensure_paragraphs,)
    )

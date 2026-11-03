from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import accumulate, chain, takewhile
from typing import Any

from l2l.context import Context
from l2l.errors import TranslationError, fail_budget, fail_pass
from l2l.glossary import GlossaryEntry, render_glossary_block
from l2l.languages import scripts_description
from l2l.monads import IO, NOTHING, Err, Just, Maybe, Ok, Result
from l2l.settings import (
    PassDefinition,
    PassMode,
    Settings,
    effective_target_language,
    pass_salt,
    resolve_call_settings,
    retranslation_salt,
)
from l2l.text import (
    AsciiDrop,
    cache_key,
    character_scripts,
    count_paragraphs,
    dominant_script,
    estimate_tokens,
    make_chunks,
    non_ascii_sample,
    regroup_by_plan,
    separator_at,
)


def verbose_log(run_context: Context, message: str) -> IO[None]:
    return run_context.console.log_verbose(message)


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
    if not re.search(r"\w", source_text):
        return NOTHING

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
        "draft's language" % scripts_description((source_script,))
    )


def instruction_with_target_language(instruction: str, target_language: str) -> str:
    return instruction.replace("{target_language}", target_language)


def target_script_problem(
    output: str,
    target_language: str,
    target_scripts: tuple[str, ...],
) -> Maybe[str]:
    if not target_scripts:
        return NOTHING

    scripts = tuple(
        script for character in output for script in character_scripts(character)
    )
    if not scripts:
        return NOTHING

    if any(script in target_scripts for script in scripts):
        return NOTHING

    return Just(
        "the reply is not in the target language (%s: %s script)"
        % (target_language, scripts_description(target_scripts))
    )


def translated_context_parts(translated_context: tuple[str, ...]) -> tuple[str, ...]:
    if not translated_context:
        return ()

    return (
        "Already-translated context (reference only — do not translate it, "
        "do not continue the story from it, and do not include it in the "
        "output):",
        *translated_context,
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
    translated_context: tuple[str, ...] = (),
) -> str:
    has_draft = work_chunk is not None and work_chunk != source_chunk
    source_block = ("Source text:\n%s" % source_chunk) if has_draft else source_chunk
    draft_block = ("Current draft:\n%s" % work_chunk) if has_draft else ""
    parts = (
        analysis.strip() if analysis else "",
        *translated_context_parts(translated_context),
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


type ChunkPlan = tuple[tuple[str, ...], ...]

PROMPT_SCAFFOLDING_TOKENS = 16
TRANSLATED_CONTEXT_PARAGRAPH_COUNT = 2
TRANSLATED_CONTEXT_TOKEN_BUDGET = 400
NON_TEXT_PASS_MODES = frozenset[PassMode](("analysis", "update_glossary"))


def earlier_passes_translated_text(earlier_pass_modes: PassModeSet) -> bool:
    return bool(earlier_pass_modes - NON_TEXT_PASS_MODES)


def suffix_within_token_budget(
    paragraphs: tuple[str, ...], budget_tokens: int
) -> tuple[str, ...]:
    cumulative_sizes = accumulate(
        estimate_tokens(paragraph) for paragraph in reversed(paragraphs)
    )
    affordable_paragraph_count = len(
        tuple(takewhile(lambda total: total <= budget_tokens, cumulative_sizes))
    )
    return paragraphs[len(paragraphs) - affordable_paragraph_count :]


type PassModeSet = frozenset[PassMode]


@dataclass(frozen=True, kw_only=True)
class PromptAllowance:
    instruction_tokens: int
    analysis_tokens: int
    scaffolding_tokens: int
    includes_draft: bool

    def request_tokens(self, source_tokens: int) -> int:
        draft_tokens = source_tokens if self.includes_draft else 0
        return (
            self.instruction_tokens
            + self.analysis_tokens
            + self.scaffolding_tokens
            + source_tokens
            + draft_tokens
        )


@dataclass(frozen=True, kw_only=True)
class PassPlan:
    chunks: ChunkPlan
    chunk_budget: int
    source_budget: int


def preceding_modes(
    pass_definitions: tuple[PassDefinition, ...],
) -> tuple[PassModeSet, ...]:
    accumulated_modes = accumulate(
        (pass_definition.mode for pass_definition in pass_definitions),
        lambda modes, mode: modes | {mode},
        initial=frozenset[PassMode](),
    )
    return tuple(accumulated_modes)[:-1]


def prompt_allowance(
    run_context: Context,
    pass_definition: PassDefinition,
    earlier_pass_modes: PassModeSet,
) -> PromptAllowance:
    glossary_tokens = (
        estimate_tokens(render_glossary_block(run_context.glossary_entries))
        if run_context.glossary_entries
        else 0
    )
    return PromptAllowance(
        instruction_tokens=estimate_tokens(pass_definition.instruction)
        + glossary_tokens,
        analysis_tokens=(
            run_context.settings.analysis_reserve_tokens
            if "analysis" in earlier_pass_modes
            else 0
        ),
        scaffolding_tokens=PROMPT_SCAFFOLDING_TOKENS,
        includes_draft=earlier_passes_translated_text(earlier_pass_modes),
    )


def pass_source_budget(allowance: PromptAllowance, maximum_tokens: int) -> int:
    reserved_tokens = (
        allowance.instruction_tokens
        + allowance.analysis_tokens
        + allowance.scaffolding_tokens
    )
    available_tokens = max(maximum_tokens - reserved_tokens, 0)
    source_share_count = 2 if allowance.includes_draft else 1
    return max(available_tokens // source_share_count, 1)


def plan_pass_chunks(
    run_context: Context,
    pass_definition: PassDefinition,
    earlier_pass_modes: PassModeSet,
    source_paragraphs: tuple[str, ...],
) -> PassPlan:
    allowance = prompt_allowance(run_context, pass_definition, earlier_pass_modes)
    source_budget = pass_source_budget(allowance, run_context.config.maximum_tokens)
    chunk_budget = min(run_context.settings.chunk_budget_tokens, source_budget)
    return PassPlan(
        chunks=make_chunks(source_paragraphs, chunk_budget),
        chunk_budget=chunk_budget,
        source_budget=source_budget,
    )


def oversized_paragraph_sizes(
    source_paragraphs: tuple[str, ...], source_budget: int
) -> tuple[tuple[int, int], ...]:
    sizes = tuple(
        (index, estimate_tokens(paragraph))
        for index, paragraph in enumerate(source_paragraphs)
    )
    return tuple((index, size) for index, size in sizes if size > source_budget)


def plan_violation(
    run_context: Context,
    pass_definition: PassDefinition,
    earlier_pass_modes: PassModeSet,
    source_paragraphs: tuple[str, ...],
) -> TranslationError | None:
    match pass_definition.mode:
        case "analysis":
            return None

        case "update_glossary":
            return None

        case _:
            allowance = prompt_allowance(
                run_context, pass_definition, earlier_pass_modes
            )
            source_budget = pass_source_budget(
                allowance, run_context.config.maximum_tokens
            )
            oversized = oversized_paragraph_sizes(source_paragraphs, source_budget)
            if not oversized:
                return None

            _, source_tokens = oversized[0]
            return fail_pass(
                pass_definition.name,
                fail_budget(
                    allowance.request_tokens(source_tokens),
                    run_context.config.maximum_tokens,
                ).error,
            ).error


def pass_chunk_plans(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    source_paragraphs: tuple[str, ...],
) -> Result[tuple[PassPlan, ...], TranslationError]:
    earlier_pass_modes = preceding_modes(pass_definitions)

    def indexed_violation(
        indexed: tuple[int, PassDefinition],
    ) -> TranslationError | None:
        index, pass_definition = indexed
        return plan_violation(
            run_context,
            pass_definition,
            earlier_pass_modes[index],
            source_paragraphs,
        )

    first_violation = next(
        (
            violation
            for violation in map(indexed_violation, enumerate(pass_definitions))
            if violation is not None
        ),
        None,
    )
    if first_violation is not None:
        return Err(first_violation)

    return Ok(
        tuple(
            plan_pass_chunks(
                run_context,
                pass_definition,
                earlier_pass_modes[index],
                source_paragraphs,
            )
            for index, pass_definition in enumerate(pass_definitions)
        )
    )


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
    translated_context: tuple[str, ...] = (),
) -> str:
    has_draft = work_chunk is not None and work_chunk != source_chunk
    draft_part = (
        "Current draft (correct this where the problem requires it):\n%s" % work_chunk
        if has_draft
        else ""
    )
    return "\n\n".join(
        (
            "Your previous reply below does not satisfy the output rules: %s."
            % problem,
            *translated_context_parts(translated_context),
            *context_parts(context),
            "Translate only the source text between the <<<SOURCE and SOURCE>>> "
            "markers, as exactly one paragraph; do not translate the context "
            "paragraphs and do not repeat the rejected reply.",
            "<<<SOURCE\n%s\nSOURCE>>>" % source_chunk,
            draft_part,
            "Your previous rejected reply (do not copy it):\n<<<\n%s\n>>>" % bad_output,
            "Translate the source text again, fixing the problem; output only "
            "the translation of the source text.",
        )
    )


def ascii_drop_warning(drop: AsciiDrop) -> str:
    return (
        "l2l: ascii: warning: paragraph %d still contained "
        "non-ASCII characters (%s) after %d LLM attempts; dropping "
        "them" % (drop.index + 1, drop.sample, drop.attempts)
    )


@dataclass(frozen=True, kw_only=True)
class UnitCall:
    index: int
    total: int
    source_chunk: str
    work_chunk: str
    context: tuple[str, ...]
    translated_context: tuple[str, ...]
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
    translated_context: tuple[str, ...] = (),
) -> UnitCall:
    return UnitCall(
        index=index,
        total=total,
        source_chunk=source_chunk,
        work_chunk=work_chunk,
        context=context,
        translated_context=translated_context,
        analysis=analysis,
        key=cache_key(
            source_chunk,
            model,
            salt,
            work_chunk,
            overrides=parameters,
            context="\n\n".join(context),
            translated_context="\n\n".join(translated_context),
            analysis=analysis or "",
        ),
        trailing_separator=separator_at(separators, separator_index),
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
    glossary: tuple[GlossaryEntry, ...] = (),
    earlier_pass_modes: PassModeSet = frozenset[PassMode](),
) -> tuple[UnitCall, ...]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    call_salt = pass_salt(
        pass_definition,
        glossary,
        retry_attempt,
        effective_target_language(run_context.target_language, pass_definition),
    )
    flat_plan = tuple(chain.from_iterable(plan))
    plan_starts = tuple(accumulate(map(len, plan), initial=0))
    working_text_is_translated = earlier_passes_translated_text(earlier_pass_modes)

    def rolling_translated_context(index: int) -> tuple[str, ...]:
        if index == 0 or not working_text_is_translated:
            return ()

        previous_group = work_groups[index - 1]
        match pass_definition.mode:
            case "paragraph":
                return suffix_within_token_budget(
                    previous_group[-1:], TRANSLATED_CONTEXT_TOKEN_BUDGET
                )

            case "chunk":
                return suffix_within_token_budget(
                    previous_group[-TRANSLATED_CONTEXT_PARAGRAPH_COUNT:],
                    TRANSLATED_CONTEXT_TOKEN_BUDGET,
                )

            case _:
                return ()

    def neighbour_context(
        index: int, translated_context: tuple[str, ...]
    ) -> tuple[str, ...]:
        match pass_definition.mode:
            case "paragraph":
                if index >= len(plan):
                    return ()

                start = plan_starts[index]
                end = plan_starts[index + 1]
                preceding_source = (
                    flat_plan[start - 1 : start]
                    if start > 0 and not translated_context
                    else ()
                )
                return preceding_source + (
                    flat_plan[end : end + 1] if end < len(flat_plan) else ()
                )

            case _:
                return ()

    def call(index: int, work_group: tuple[str, ...]) -> UnitCall:
        translated_context = rolling_translated_context(index)
        return build_unit_call(
            model,
            parameters,
            call_salt,
            trailing_separators,
            index,
            index,
            len(work_groups),
            "\n\n".join(plan[index]) if index < len(plan) else "",
            "\n\n".join(work_group),
            neighbour_context(index, translated_context),
            analysis,
            translated_context,
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
    glossary: tuple[GlossaryEntry, ...] = (),
) -> tuple[UnitCall, ...]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    call_salt = retranslation_salt(
        pass_definition,
        glossary,
        effective_target_language(run_context.target_language, pass_definition),
    )

    def neighbour_context(index: int) -> tuple[str, ...]:
        neighbouring_indices = (index - 1, index + 1)
        return tuple(
            source_paragraphs[neighbour_index]
            for neighbour_index in neighbouring_indices
            if 0 <= neighbour_index < len(source_paragraphs)
        )

    def call(position: int, index: int) -> UnitCall:
        return build_unit_call(
            model,
            parameters,
            call_salt,
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

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import accumulate, takewhile
from typing import Final, final
lazy from collections.abc import Mapping

from l2l.errors import TranslationError, fail_budget, fail_pass
from l2l.glossary import (
    GlossaryEntry,
    render_glossary_block,
    untranslated_glossary_entries,
)
from l2l.languages import expected_source_scripts, scripts_description
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
    paragraph_groups,
    regroup_by_plan,
    separator_at,
    split_paragraphs,
)
lazy from l2l.context import Context
lazy from l2l.json_values import JsonValue


def verbose_log(run_context: Context, message: str) -> IO[None]:
    return run_context.console.log_verbose(message)


def unit_output_problem(
    source_text: str,
    output: str,
    settings: Settings,
    tolerance: int | None,
    maximum_characters: int | None = None,
) -> Maybe[str]:
    if not output.strip():
        return Just("the reply was empty")

    if maximum_characters is not None:
        for index, paragraph in enumerate(split_paragraphs(output)[0], start=1):
            if len(paragraph) > maximum_characters:
                return Just(
                    f"paragraph {index} has {len(paragraph)} characters but the "
                    f"limit is at most {maximum_characters}"
                )

    if tolerance is not None:
        expected = count_paragraphs(source_text)
        found = count_paragraphs(output)
        if abs(found - expected) > tolerance:
            allowed = "" if tolerance <= 0 else f" (allowed ±{tolerance})"
            return Just(
                f"the reply has {found} paragraph(s) but the source has {expected}"
                f"{allowed}"
            )

    source_estimate = estimate_tokens(source_text)
    output_estimate = estimate_tokens(output)
    if not re.search(r"\w", source_text):
        return NOTHING

    if output_estimate > settings.unit_output_max_ratio * max(source_estimate, 1):
        return Just(
            f"the reply is ~{output_estimate} tokens against a source of "
            f"~{source_estimate} tokens (limit {settings.unit_output_max_ratio:.0f}x)"
        )

    return NOTHING


UNTRANSLATED_GLOSSARY_TERM_SAMPLE_COUNT: Final = 3


def glossary_term_problem(
    output: str, entries: tuple[GlossaryEntry, ...]
) -> Maybe[str]:
    untranslated = untranslated_glossary_entries(entries, output)
    if not untranslated:
        return NOTHING

    return Just(
        "the reply contains untranslated glossary term(s) {}; use their "
        "mandated renderings".format(
            ", ".join(
                f'"{entry.original}" (use "{entry.translated}")'
                for entry in untranslated[:UNTRANSLATED_GLOSSARY_TERM_SAMPLE_COUNT]
            )
        )
    )


def script_mismatch_problem(
    source_text: str,
    work_text: str,
    output: str,
    source_language: str,
) -> Maybe[str]:
    observed_source_script = dominant_script(source_text)
    candidate_scripts = expected_source_scripts(source_language) or (
        (observed_source_script,) if observed_source_script else ()
    )
    if not candidate_scripts:
        return NOTHING

    if dominant_script(work_text) in candidate_scripts:
        return NOTHING

    output_script = dominant_script(output)
    if output_script not in candidate_scripts:
        return NOTHING

    return Just(
        f"the reply is in the source language "
        f"({scripts_description((output_script,))} script) instead of "
        "the draft's language"
    )


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
        f"the reply is not in the target language ({target_language}: "
        f"{scripts_description(target_scripts)} script)"
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
    source_block = (f"Source text:\n{source_chunk}") if has_draft else source_chunk
    draft_block = (f"Current draft:\n{work_chunk}") if has_draft else ""
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
    per_call_note = (
        f", up to {pass_definition.paragraphs_per_call} paragraph(s) per call"
        if pass_definition.paragraphs_per_call >= 1
        else ""
    )
    return (
        f"l2l: [{pass_definition.name}] {len(work_paragraphs)} paragraph(s) "
        f"in {len(work_groups)} call(s){per_call_note}"
    )


def resolve_work_groups(
    paragraphs_per_call: int,
    work_paragraphs: tuple[str, ...],
    plan: tuple[tuple[str, ...], ...],
    pass_name: str,
    budget: int,
) -> tuple[tuple[tuple[str, ...], ...], Maybe[str]]:
    grouped = regroup_by_plan(work_paragraphs, plan)
    if isinstance(grouped, Just):
        return grouped.value, NOTHING

    warning: Maybe[str] = Just(
        f"l2l: [{pass_name}] paragraph count changed by a previous pass; "
        "grouping working text independently"
    )
    return paragraph_groups(work_paragraphs, paragraphs_per_call, budget), warning


type ChunkPlan = tuple[tuple[str, ...], ...]

PROMPT_SCAFFOLDING_TOKENS: Final = 16
TRANSLATED_CONTEXT_PARAGRAPH_COUNT: Final = 2
TRANSLATED_CONTEXT_TOKEN_BUDGET: Final = 400
NON_TEXT_PASS_MODES: Final = frozenset[PassMode](("analysis", "update_glossary"))


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

EMPTY_PASS_MODE_SET: Final[PassModeSet] = frozenset[PassMode]()


@final
@dataclass(frozen=True, kw_only=True, slots=True)
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


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class PassPlan:
    chunks: ChunkPlan
    chunk_budget: int
    source_budget: int


def merge_mode(modes: PassModeSet, mode: PassMode) -> PassModeSet:
    return modes | frozenset((mode,))


def preceding_modes(
    pass_definitions: tuple[PassDefinition, ...],
) -> tuple[PassModeSet, ...]:
    modes: tuple[PassMode, ...] = tuple(
        pass_definition.mode for pass_definition in pass_definitions
    )
    accumulated_modes = accumulate(modes, merge_mode, initial=frozenset[PassMode]())
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

        case "translate":
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
        f"Source paragraph (original language):\n{source_paragraph}\n\n"
        "Translated paragraph (must use ASCII characters only):\n"
        f"{output_paragraph}\n\n"
        "Rewrite the translated paragraph using ASCII characters only, "
        "preserving its meaning, register, and language."
    )


def build_ascii_retry_user(
    source_paragraph: str, output_paragraph: str, result: str
) -> str:
    return (
        f"Source paragraph (original language):\n{source_paragraph}\n\n"
        "Translated paragraph (must use ASCII characters only):\n"
        f"{output_paragraph}\n\n"
        "Your previous reply still contained these non-ASCII "
        f"characters: {non_ascii_sample(result)}. Rewrite the translated "
        "paragraph again, finding an equivalent ASCII formulation for "
        "every one of them from the source and context. Reply with ASCII "
        "characters only."
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
        f"Current draft (correct this where the problem requires it):\n{work_chunk}"
        if has_draft
        else ""
    )
    return "\n\n".join(
        (
            f"Your previous reply below does not satisfy the output rules: {problem}.",
            *translated_context_parts(translated_context),
            *context_parts(context),
            "Translate only the source text between the <<<SOURCE and SOURCE>>> "
            "markers, as exactly one paragraph; do not translate the context "
            "paragraphs and do not repeat the rejected reply.",
            f"<<<SOURCE\n{source_chunk}\nSOURCE>>>",
            draft_part,
            f"Your previous rejected reply (do not copy it):\n<<<\n{bad_output}\n>>>",
            "Translate the source text again, fixing the problem; output only "
            "the translation of the source text.",
        )
    )


def ascii_drop_warning(drop: AsciiDrop) -> str:
    return (
        f"l2l: ascii: warning: paragraph {drop.index + 1} still contained "
        f"non-ASCII characters ({drop.sample}) after {drop.attempts} LLM "
        "attempts; dropping them"
    )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
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
    parameters: Mapping[str, JsonValue]


def build_unit_call(
    model: str,
    parameters: Mapping[str, JsonValue],
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
    earlier_pass_modes: PassModeSet = EMPTY_PASS_MODE_SET,
) -> tuple[UnitCall, ...]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    call_salt = pass_salt(
        pass_definition,
        glossary,
        retry_attempt,
        effective_target_language(
            run_context.target_language, pass_definition.target_language
        ),
        run_context.source_language,
    )
    working_text_is_translated = earlier_passes_translated_text(earlier_pass_modes)

    def rolling_translated_context(index: int) -> tuple[str, ...]:
        if index == 0 or not working_text_is_translated:
            return ()

        previous_group = work_groups[index - 1]
        match pass_definition.mode:
            case "translate":
                return suffix_within_token_budget(
                    previous_group[-TRANSLATED_CONTEXT_PARAGRAPH_COUNT:],
                    TRANSLATED_CONTEXT_TOKEN_BUDGET,
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
            (),
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
        effective_target_language(
            run_context.target_language, pass_definition.target_language
        ),
        run_context.source_language,
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

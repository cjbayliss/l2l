from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from typing import TextIO

from l2l.ascii import enforce_pass_ascii
from l2l.cache import (
    always_acceptable,
    cache_lookup,
    cached_translation,
    non_empty,
    store_translation,
)
from l2l.console import Console
from l2l.context import Context
from l2l.effects import now, write_stdout, write_text_file
from l2l.errors import (
    TranslationError,
    describe,
    fail_budget,
    fail_config,
    fail_invalid_output,
    fail_pass,
    fail_unit,
    fail_untranslated,
)
from l2l.glossary import (
    GlossaryEntry,
    append_glossary_entries,
    entries_to_json_text,
    extraction_user_message,
    instruction_with_glossary,
    parse_model_glossary_reply,
)
from l2l.http import RepairOutcome, RepairPolicy, chat, repaired_call
from l2l.languages import (
    expected_target_scripts,
    script_description,
    scripts_description,
)
from l2l.messages import (
    analysis_information_message,
    done_in_message,
    glossary_update_information_message,
    paragraph_mismatch_message,
    paragraph_still_differs_message,
    pass_cache_hit_message,
    pass_started_message,
    retranslate_information_message,
    retranslate_majority_message,
    retranslate_skipped_message,
    retranslate_unit_message,
    retranslation_check_message,
    retranslation_target_script_message,
    stage_done_line,
    unit_cache_hit_message,
    unit_done_message,
    unit_failed_validation_attempt,
    unit_failed_validation_final,
    unit_no_source_message,
    untranslated_kept_message,
    untranslated_problem,
    usage_line,
)
from l2l.monads import (
    IO,
    Err,
    Just,
    Maybe,
    Ok,
    Result,
    fold_io,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_result_bind,
    io_result_map,
    io_traverse,
    maybe_either,
    maybe_or,
    result_map,
)
from l2l.plans import (
    PassPlan,
    UnitCall,
    build_pass_user,
    build_unit_retry_user,
    instruction_with_target_language,
    pass_chunk_plans,
    plan_information_message,
    plan_retranslation_calls,
    plan_unit_calls,
    resolve_work_groups,
    script_mismatch_problem,
    target_script_problem,
    unit_output_problem,
    verbose_log,
)
from l2l.settings import (
    PassDefinition,
    effective_target_language,
    paragraph_tolerance,
    pass_salt,
    resolve_call_settings,
)
from l2l.text import (
    Translated,
    Usage,
    cache_key,
    count_paragraphs,
    ensure_blank_line_separators,
    estimate_tokens,
    excerpt,
    make_chunks,
    paragraph_is_in_target_script,
    separator_at,
    source_script,
    split_paragraphs,
    split_units_to_budget,
    unit_separators,
    untranslated_paragraph,
    usage_add,
    usage_delta,
)


@dataclass(frozen=True, kw_only=True)
class State:
    text: str
    analysis: str | None
    usage: Usage
    validated: bool
    problem: str | None
    glossary: tuple[GlossaryEntry, ...]


@dataclass(frozen=True, kw_only=True)
class UnitResult:
    text: str
    validated: bool
    usage: Usage
    problem: str | None = None


@dataclass(frozen=True, kw_only=True)
class UnitsSoFar:
    outputs: tuple[str, ...]
    usage: Usage
    validated: bool
    problem: str | None


@dataclass(frozen=True, kw_only=True)
class PassExecution:
    run_context: Context
    pass_definition: PassDefinition
    source_paragraphs: tuple[str, ...]
    separators: tuple[str, ...]
    chunk_plan: PassPlan
    paragraph_plan: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, kw_only=True)
class CallEnvironment:
    run_context: Context
    pass_definition: PassDefinition
    glossary: tuple[GlossaryEntry, ...]


StateResult = Result[State, TranslationError]

PassRetry = Callable[[Usage, float], IO[StateResult]]


def state_with_output(
    state: State,
    text: str,
    usage: Usage,
    validated: bool,
    problem: str | None,
) -> State:
    return replace(
        state,
        text=text,
        usage=usage,
        validated=state.validated and validated,
        problem=problem if problem is not None else state.problem,
    )


def untranslated_majority(flagged: int, total: int) -> bool:
    return flagged * 2 > total


def run_analysis(
    run_context: Context,
    pass_definition: PassDefinition,
    glossary: tuple[GlossaryEntry, ...],
    text: str,
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    instruction = instruction_with_target_language(
        pass_definition.instruction,
        effective_target_language(run_context.target_language, pass_definition),
    )
    return io_result_map(
        chat(
            run_context,
            instruction_with_glossary(instruction, glossary),
            text,
            model,
            parameters,
            usage,
        ),
        lambda translated: Translated(translated.text.strip(), translated.usage),
    )


def analyze_document(
    run_context: Context,
    pass_definition: PassDefinition,
    glossary: tuple[GlossaryEntry, ...],
    full_text: str,
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    instruction = instruction_with_target_language(
        pass_definition.instruction,
        effective_target_language(run_context.target_language, pass_definition),
    )
    budget = max(
        run_context.config.maximum_tokens
        - estimate_tokens(instruction_with_glossary(instruction, glossary))
        - run_context.settings.analysis_reserve_tokens,
        1,
    )
    chunks = make_chunks(
        split_units_to_budget(
            split_paragraphs(full_text)[0],
            budget,
            run_context.settings.sentence_boundary_characters,
        ),
        budget,
    )
    if len(chunks) > 1:
        return io_result(
            fail_budget(
                estimate_tokens(full_text),
                run_context.config.maximum_tokens,
                len(chunks),
            )
        )

    return run_analysis(run_context, pass_definition, glossary, full_text, usage)


def run_analysis_once(
    run_context: Context,
    pass_definition: PassDefinition,
    glossary: tuple[GlossaryEntry, ...],
    full_text: str,
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    key = cache_key(
        full_text,
        model,
        pass_salt(
            pass_definition,
            glossary,
            effective_language=effective_target_language(
                run_context.target_language, pass_definition
            ),
        ),
        overrides=parameters,
    )
    return cached_translation(
        run_context,
        key,
        usage,
        compute=lambda: analyze_document(
            run_context, pass_definition, glossary, full_text, usage
        ),
        hit_log=verbose_log(run_context, pass_cache_hit_message(pass_definition.name)),
    )


def log_stage(
    console: Console,
    label: str,
    started_at: float,
    ended_at: float,
    start_usage: Usage,
    end_usage: Usage,
) -> IO[None]:
    prompt, completion, cost = usage_delta(start_usage, end_usage)
    return console.log(
        usage_line(label, ended_at - started_at, prompt, completion, cost)
    )


def finish_pass_stage(
    run_context: Context,
    pass_definition: PassDefinition,
    previous_usage: Usage,
    next_state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    retry_same_mode: PassRetry | None = None,
) -> IO[StateResult]:
    def after_stage(ended_at: float) -> IO[StateResult]:
        def concluded(_: None) -> IO[StateResult]:
            return conclude_state(
                run_context,
                pass_definition,
                next_state,
                source_paragraphs,
                retry_same_mode,
            )

        return io_bind(
            log_stage(
                run_context.console,
                "Done",
                started_at,
                ended_at,
                previous_usage,
                next_state.usage,
            ),
            concluded,
        )

    return io_bind(now(run_context.clock), after_stage)


def enforce_ascii_state(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs: tuple[str, ...],
) -> IO[StateResult]:
    if not pass_definition.ascii:
        return io_result(Ok(state))

    def enforce(ascii_started: float) -> IO[StateResult]:
        return io_result_map(
            enforce_pass_ascii(
                run_context,
                pass_definition,
                state.text,
                source_paragraphs,
                state.usage,
                ascii_started,
            ),
            lambda translated: replace(
                state,
                text=translated.text,
                usage=translated.usage,
            ),
        )

    return io_bind(now(run_context.clock), enforce)


def retranslate_state(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs: tuple[str, ...],
    retry_same_mode: PassRetry | None,
) -> IO[StateResult]:
    def after_retranslation(
        result: Result[Translated, TranslationError],
    ) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(Err(result.error))

        return enforce_ascii_state(
            run_context,
            pass_definition,
            state_with_output(
                state,
                result.value.text,
                result.value.usage,
                result.value.validated,
                result.value.problem,
            ),
            source_paragraphs,
        )

    def retranslation_stage(retranslation_started: float) -> IO[StateResult]:
        return io_bind(
            retranslate_untranslated(
                run_context,
                pass_definition,
                state.text,
                source_paragraphs,
                state.analysis,
                state.usage,
                retranslation_started,
                retry_same_mode,
                state.glossary,
            ),
            after_retranslation,
        )

    return io_bind(now(run_context.clock), retranslation_stage)


def conclude_state(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs: tuple[str, ...],
    retry_same_mode: PassRetry | None = None,
) -> IO[StateResult]:
    match pass_definition.mode:
        case "analysis" | "update_glossary":
            return io_result(Ok(state))

        case _:
            pass

    if pass_definition.retranslate_untranslated:
        return retranslate_state(
            run_context, pass_definition, state, source_paragraphs, retry_same_mode
        )

    return enforce_ascii_state(run_context, pass_definition, state, source_paragraphs)


@dataclass(frozen=True, kw_only=True)
class RetranslationCheck:
    has_matching_paragraph_count: bool
    work_paragraphs: tuple[str, ...]
    separators: tuple[str, ...]
    flagged: tuple[int, ...]
    target_language: str
    target_scripts: tuple[str, ...]
    source_script_display: str
    target_scripts_display: str


@dataclass(frozen=True, kw_only=True)
class RetranslationRun:
    environment: CallEnvironment
    check: RetranslationCheck
    source_paragraphs: tuple[str, ...]
    calls: tuple[UnitCall, ...]
    usage: Usage
    started_at: float


@dataclass(frozen=True, kw_only=True)
class RetranslationEvaluation:
    outcome: Result[Translated, TranslationError]
    target_script_kept: tuple[tuple[int, str], ...]
    best_effort_kept: tuple[int, str] | None


def untranslated_paragraph_indices(
    source_paragraphs: tuple[str, ...],
    work_paragraphs: tuple[str, ...],
    target_scripts: tuple[str, ...],
    character_map: Mapping[str, str],
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (source, output) in enumerate(
            zip(source_paragraphs, work_paragraphs, strict=True)
        )
        if untranslated_paragraph(source, output, target_scripts, character_map)
    )


def retranslation_check(
    run_context: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
) -> RetranslationCheck:
    work_paragraphs, separators = split_paragraphs(text)
    target_language = effective_target_language(
        run_context.target_language, pass_definition
    )
    target_scripts = expected_target_scripts(target_language)
    source_script_name = source_script(source_paragraphs)
    has_matching_paragraph_count = len(work_paragraphs) == len(source_paragraphs)
    return RetranslationCheck(
        has_matching_paragraph_count=has_matching_paragraph_count,
        work_paragraphs=work_paragraphs,
        separators=separators,
        flagged=(
            untranslated_paragraph_indices(
                source_paragraphs,
                work_paragraphs,
                target_scripts,
                run_context.settings.ascii_character_map,
            )
            if has_matching_paragraph_count
            else ()
        ),
        target_language=target_language,
        target_scripts=target_scripts,
        source_script_display=(
            script_description(source_script_name) if source_script_name else ""
        ),
        target_scripts_display=(
            scripts_description(target_scripts) if target_scripts else ""
        ),
    )


def retranslation_cacheable(
    check: RetranslationCheck,
    source_paragraphs: tuple[str, ...],
    character_map: Mapping[str, str],
    index: int,
) -> Callable[[str], bool]:
    def cacheable(text: str) -> bool:
        translated = not untranslated_paragraph(
            source_paragraphs[index], text, check.target_scripts, character_map
        )
        return translated or paragraph_is_in_target_script(text, check.target_scripts)

    return cacheable


def remaining_untranslated_paragraphs(
    run_context: Context,
    check: RetranslationCheck,
    source_paragraphs: tuple[str, ...],
    texts: tuple[str, ...],
) -> tuple[tuple[int, str], ...]:
    bodies = tuple(
        (
            index,
            texts[index].removesuffix(separator_at(check.separators, index)),
        )
        for index in check.flagged
    )
    return tuple(
        (index, body)
        for index, body in bodies
        if untranslated_paragraph(
            source_paragraphs[index],
            body,
            check.target_scripts,
            run_context.settings.ascii_character_map,
        )
    )


def evaluate_retranslation(
    run_context: Context,
    pass_definition: PassDefinition,
    check: RetranslationCheck,
    source_paragraphs: tuple[str, ...],
    units: tuple[UnitResult, ...],
    usage: Usage,
) -> RetranslationEvaluation:
    texts = tuple(unit.text for unit in units)
    untranslated = remaining_untranslated_paragraphs(
        run_context, check, source_paragraphs, texts
    )
    target_script_kept = tuple(
        (index, body)
        for index, body in untranslated
        if paragraph_is_in_target_script(body, check.target_scripts)
    )
    unresolved = tuple(
        (index, body)
        for index, body in untranslated
        if not paragraph_is_in_target_script(body, check.target_scripts)
    )
    total_usage = reduce(usage_add, (unit.usage for unit in units), usage)
    joined_text = "".join(texts)

    if unresolved and not run_context.best_effort:
        index, body = unresolved[0]
        return RetranslationEvaluation(
            outcome=fail_untranslated(pass_definition.name, index, excerpt(body)),
            target_script_kept=target_script_kept,
            best_effort_kept=None,
        )

    if not unresolved:
        return RetranslationEvaluation(
            outcome=Ok(Translated(joined_text, total_usage)),
            target_script_kept=target_script_kept,
            best_effort_kept=None,
        )

    index, body = unresolved[0]
    return RetranslationEvaluation(
        outcome=Ok(
            Translated(
                joined_text,
                total_usage,
                False,
                untranslated_problem(index, excerpt(body)),
            )
        ),
        target_script_kept=target_script_kept,
        best_effort_kept=(index, body),
    )


def announce_retranslation_evaluation(
    run_context: Context,
    pass_definition: PassDefinition,
    total_paragraphs: int,
    evaluation: RetranslationEvaluation,
) -> IO[None]:
    verify_kept = reduce(
        io_and_then,
        (
            verbose_log(
                run_context,
                retranslation_target_script_message(
                    pass_definition.name,
                    index,
                    total_paragraphs,
                    excerpt(body),
                ),
            )
            for index, body in evaluation.target_script_kept
        ),
        io_pure(None),
    )

    def announce_kept(_: None) -> IO[None]:
        match evaluation.best_effort_kept:
            case None:
                return io_pure(None)

            case (index, body):
                return run_context.console.log(
                    untranslated_kept_message(
                        pass_definition.name,
                        index,
                        total_paragraphs,
                        excerpt(body),
                    )
                )

    return io_bind(verify_kept, announce_kept)


def combine_retranslated_units(
    run_context: Context,
    pass_definition: PassDefinition,
    check: RetranslationCheck,
    source_paragraphs: tuple[str, ...],
    units: tuple[UnitResult, ...],
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    evaluation = evaluate_retranslation(
        run_context, pass_definition, check, source_paragraphs, units, usage
    )
    return io_map(
        announce_retranslation_evaluation(
            run_context,
            pass_definition,
            len(check.work_paragraphs),
            evaluation,
        ),
        lambda _: evaluation.outcome,
    )


def retranslate_indexed_paragraph(
    run: RetranslationRun,
    calls_by_index: Mapping[int, UnitCall],
    indexed: tuple[int, str],
) -> IO[Result[UnitResult, TranslationError]]:
    index, paragraph = indexed
    check = run.check
    call = calls_by_index.get(index)
    if call is None:
        return io_result(
            Ok(
                UnitResult(
                    text=paragraph + separator_at(check.separators, index),
                    validated=True,
                    usage=Usage(),
                    problem=None,
                )
            )
        )

    return io_and_then(
        verbose_log(
            run.environment.run_context,
            retranslate_unit_message(
                run.environment.pass_definition.name,
                call.index + 1,
                call.total,
            ),
        ),
        run_single_call(
            run.environment,
            call,
            run.usage,
            retranslation_cacheable(
                check,
                run.source_paragraphs,
                run.environment.run_context.settings.ascii_character_map,
                index,
            ),
        ),
    )


def conclude_retranslation(
    run_context: Context,
    result: Result[Translated, TranslationError],
    usage: Usage,
    started_at: float,
) -> IO[Result[Translated, TranslationError]]:
    if isinstance(result, Err):
        return io_map(run_context.console.interrupt(), lambda _: result)

    prompt, completion, cost = usage_delta(usage, result.value.usage)

    def report(ended_at: float) -> IO[Result[Translated, TranslationError]]:
        return io_map(
            run_context.console.finish(
                stage_done_line(ended_at - started_at, prompt, completion, cost)
            ),
            lambda _: result,
        )

    return io_bind(now(run_context.clock), report)


def run_retranslation(
    run: RetranslationRun,
) -> IO[Result[Translated, TranslationError]]:
    run_context = run.environment.run_context
    pass_definition = run.environment.pass_definition

    def units_result(
        units: tuple[UnitResult, ...],
    ) -> IO[Result[Translated, TranslationError]]:
        return combine_retranslated_units(
            run_context,
            pass_definition,
            run.check,
            run.source_paragraphs,
            units,
            run.usage,
        )

    calls_by_index = dict(zip(run.check.flagged, run.calls, strict=True))
    return io_bind(
        io_result_bind(
            io_traverse(
                enumerate(run.check.work_paragraphs),
                lambda item: retranslate_indexed_paragraph(run, calls_by_index, item),
            ),
            units_result,
        ),
        lambda result: conclude_retranslation(
            run_context, result, run.usage, run.started_at
        ),
    )


def retranslation_majority_retry(
    run_context: Context,
    pass_definition: PassDefinition,
    check: RetranslationCheck,
    usage: Usage,
    retry_same_mode: PassRetry,
) -> IO[Result[Translated, TranslationError]]:
    def escalate(
        retry_started: float,
    ) -> IO[Result[Translated, TranslationError]]:
        def lifted(outcome: StateResult) -> Result[Translated, TranslationError]:
            if isinstance(outcome, Err):
                return Err(outcome.error)

            return Ok(
                Translated(
                    outcome.value.text,
                    outcome.value.usage,
                    outcome.value.validated,
                    outcome.value.problem,
                )
            )

        return io_map(retry_same_mode(usage, retry_started), lifted)

    return io_and_then(
        run_context.console.log(
            retranslate_majority_message(
                pass_definition.name,
                len(check.flagged),
                len(check.work_paragraphs),
            )
        ),
        io_bind(now(run_context.clock), escalate),
    )


def retranslate_untranslated(
    run_context: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    analysis: str | None,
    usage: Usage,
    started_at: float,
    retry_same_mode: PassRetry | None = None,
    glossary: tuple[GlossaryEntry, ...] = (),
) -> IO[Result[Translated, TranslationError]]:
    check = retranslation_check(run_context, pass_definition, text, source_paragraphs)

    if not check.has_matching_paragraph_count:
        return io_bind(
            verbose_log(
                run_context,
                retranslate_skipped_message(
                    pass_definition.name,
                    len(check.work_paragraphs),
                    len(source_paragraphs),
                ),
            ),
            lambda _: io_result(Ok(Translated(text, usage))),
        )

    if not check.flagged:
        return io_result(Ok(Translated(text, usage)))

    if retry_same_mode is not None and untranslated_majority(
        len(check.flagged), len(check.work_paragraphs)
    ):
        return retranslation_majority_retry(
            run_context, pass_definition, check, usage, retry_same_mode
        )

    run = RetranslationRun(
        environment=CallEnvironment(
            run_context=run_context,
            pass_definition=pass_definition,
            glossary=glossary,
        ),
        check=check,
        source_paragraphs=source_paragraphs,
        calls=plan_retranslation_calls(
            run_context,
            pass_definition,
            source_paragraphs,
            check.work_paragraphs,
            check.flagged,
            check.separators,
            analysis,
            glossary,
        ),
        usage=usage,
        started_at=started_at,
    )
    return io_and_then(
        run_context.console.log(
            retranslate_information_message(
                pass_definition.name,
                len(check.flagged),
                len(check.work_paragraphs),
            )
        ),
        io_bind(
            run_context.console.log(
                retranslation_check_message(
                    check.target_language,
                    check.source_script_display,
                    check.target_scripts_display,
                )
            ),
            lambda _: run_retranslation(run),
        ),
    )


def run_analysis_pass(
    execution: PassExecution,
    state: State,
    started_at: float,
) -> IO[StateResult]:
    run_context = execution.run_context
    pass_definition = execution.pass_definition

    def after_analysis(
        analysis_result: Result[Translated, TranslationError],
    ) -> IO[StateResult]:
        if isinstance(analysis_result, Err):
            return io_result(fail_pass(pass_definition.name, analysis_result.error))

        outcome = analysis_result.value
        return finish_pass_stage(
            run_context,
            pass_definition,
            state.usage,
            replace(
                state,
                analysis=outcome.text,
                usage=outcome.usage,
            ),
            started_at,
            execution.source_paragraphs,
        )

    return io_bind(
        verbose_log(
            run_context,
            analysis_information_message(
                pass_definition.name, len(state.text), estimate_tokens(state.text)
            ),
        ),
        lambda _: io_bind(
            run_analysis_once(
                run_context, pass_definition, state.glossary, state.text, state.usage
            ),
            after_analysis,
        ),
    )


def run_update_glossary_once(
    run_context: Context,
    pass_definition: PassDefinition,
    target_language: str,
    state: State,
) -> IO[Result[Translated, TranslationError]]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    key = cache_key(
        state.text,
        model,
        pass_salt(pass_definition, effective_language=target_language),
        overrides=parameters,
    )

    def reply_is_parseable_glossary(reply_text: str) -> bool:
        return isinstance(
            parse_model_glossary_reply(
                reply_text, run_context.glossary_update_path or "glossary"
            ),
            Ok,
        )

    return cached_translation(
        run_context,
        key,
        state.usage,
        compute=lambda: chat(
            run_context,
            run_context.settings.glossary_extraction_instruction,
            extraction_user_message(target_language, state.glossary, state.text),
            model,
            parameters,
            state.usage,
        ),
        hit_log=verbose_log(run_context, pass_cache_hit_message(pass_definition.name)),
        acceptable=reply_is_parseable_glossary,
    )


def apply_glossary_update(
    execution: PassExecution,
    state: State,
    started_at: float,
    update_path: str,
    reply_result: Result[Translated, TranslationError],
) -> IO[StateResult]:
    run_context = execution.run_context
    pass_definition = execution.pass_definition
    if isinstance(reply_result, Err):
        return io_result(fail_pass(pass_definition.name, reply_result.error))

    reply = reply_result.value
    reply_entries = parse_model_glossary_reply(reply.text, update_path)
    if isinstance(reply_entries, Err):
        return io_result(fail_pass(pass_definition.name, reply_entries.error))

    merged = append_glossary_entries(state.glossary, reply_entries.value)

    def after_write(written: Result[None, TranslationError]) -> IO[StateResult]:
        if isinstance(written, Err):
            return io_result(fail_pass(pass_definition.name, written.error))

        return finish_pass_stage(
            run_context,
            pass_definition,
            state.usage,
            replace(
                state,
                usage=reply.usage,
                glossary=merged,
            ),
            started_at,
            execution.source_paragraphs,
        )

    return io_bind(
        write_text_file(update_path, entries_to_json_text(merged), "glossary file"),
        after_write,
    )


def run_update_glossary_pass(
    execution: PassExecution,
    state: State,
    started_at: float,
) -> IO[StateResult]:
    run_context = execution.run_context
    pass_definition = execution.pass_definition
    update_path = run_context.glossary_update_path
    if update_path is None:
        return io_result(
            fail_pass(
                pass_definition.name,
                fail_config(
                    "an update_glossary pass requires at least one --glossary "
                    "file on the command line"
                ).error,
            )
        )

    target_language = effective_target_language(
        run_context.target_language, pass_definition
    )
    return io_bind(
        verbose_log(
            run_context,
            glossary_update_information_message(
                pass_definition.name,
                target_language,
                len(state.text),
                estimate_tokens(state.text),
            ),
        ),
        lambda _: io_bind(
            run_update_glossary_once(
                run_context, pass_definition, target_language, state
            ),
            lambda reply: apply_glossary_update(
                execution, state, started_at, update_path, reply
            ),
        ),
    )


def unit_paragraph_tolerance(pass_definition: PassDefinition) -> int | None:
    match pass_definition.mode:
        case "paragraph":
            return 0

        case _:
            return paragraph_tolerance(pass_definition.ensure_paragraphs)


def unit_result_from_repair(
    outcome: Result[RepairOutcome, TranslationError],
) -> Result[UnitResult, TranslationError]:
    return result_map(
        outcome,
        lambda repaired: UnitResult(
            text=repaired.text,
            validated=repaired.validated,
            usage=repaired.usage,
            problem=repaired.problem,
        ),
    )


def run_unit(
    environment: CallEnvironment,
    call: UnitCall,
    usage: Usage,
) -> IO[Result[UnitResult, TranslationError]]:
    run_context = environment.run_context
    pass_definition = environment.pass_definition
    target_language = effective_target_language(
        run_context.target_language, pass_definition
    )
    target_scripts = expected_target_scripts(target_language)

    def validate(output: str) -> Maybe[str]:
        return maybe_or(
            unit_output_problem(
                call.source_chunk,
                output,
                run_context.settings,
                unit_paragraph_tolerance(pass_definition),
            ),
            maybe_or(
                target_script_problem(output, target_language, target_scripts),
                script_mismatch_problem(call.source_chunk, call.work_chunk, output),
            ),
        )

    def build_retry_user(bad_output: str, problem: str) -> str:
        return build_unit_retry_user(
            call.source_chunk, call.work_chunk, call.context, bad_output, problem
        )

    def on_repair(failed: int, problem: str) -> IO[None]:
        return verbose_log(
            run_context,
            unit_failed_validation_attempt(
                pass_definition.name,
                call.index + 1,
                call.total,
                problem,
                failed,
                run_context.settings.unit_fix_attempts,
            ),
        )

    def on_exhausted(bad_output: str, problem: str) -> IO[None]:
        return run_context.console.log(
            unit_failed_validation_final(
                pass_definition.name,
                call.index + 1,
                call.total,
                run_context.settings.unit_fix_attempts,
                problem,
                run_context.best_effort,
            )
        )

    return io_map(
        repaired_call(
            run_context,
            call.model,
            call.parameters,
            instruction_with_glossary(
                instruction_with_target_language(
                    pass_definition.instruction,
                    target_language,
                ),
                environment.glossary,
            ),
            build_pass_user(
                call.source_chunk, call.work_chunk, call.analysis, call.context
            ),
            RepairPolicy(
                validate=validate,
                build_retry_user=build_retry_user,
                maximum_repairs=run_context.settings.unit_fix_attempts,
                on_repair=on_repair,
                on_exhausted=on_exhausted,
            ),
        )(usage),
        unit_result_from_repair,
    )


def run_single_call(
    environment: CallEnvironment,
    call: UnitCall,
    usage: Usage,
    cacheable: Callable[[str], bool] = always_acceptable,
) -> IO[Result[UnitResult, TranslationError]]:
    run_context = environment.run_context
    pass_definition = environment.pass_definition

    def store_unit_result(
        result: Result[UnitResult, TranslationError],
    ) -> IO[Result[UnitResult, TranslationError]]:
        if isinstance(result, Err):
            return io_result(fail_unit(pass_definition.name, call.index, result.error))

        outcome = result.value
        stored = io_and_then(
            store_translation(
                run_context,
                call.key,
                result_map(
                    result,
                    lambda unit: Translated(
                        unit.text, unit.usage, unit.validated, unit.problem
                    ),
                ),
                cacheable,
            ),
            verbose_log(
                run_context,
                unit_done_message(pass_definition.name, call.index + 1, call.total),
            ),
        )
        return io_map(
            stored,
            lambda _: Ok(
                UnitResult(
                    text=outcome.text + call.trailing_separator,
                    validated=outcome.validated,
                    usage=Usage(*usage_delta(usage, outcome.usage)),
                    problem=outcome.problem,
                )
            ),
        )

    def continue_after_cache_lookup(
        cached: Maybe[str],
    ) -> IO[Result[UnitResult, TranslationError]]:
        if isinstance(cached, Just):
            return io_map(
                verbose_log(
                    run_context,
                    unit_cache_hit_message(
                        pass_definition.name, call.index + 1, call.total
                    ),
                ),
                lambda _: Ok(
                    UnitResult(
                        text=cached.value + call.trailing_separator,
                        validated=True,
                        usage=Usage(),
                        problem=None,
                    )
                ),
            )

        return io_bind(run_unit(environment, call, usage), store_unit_result)

    return io_bind(
        cache_lookup(run_context, call.key, acceptable=non_empty),
        continue_after_cache_lookup,
    )


def units_so_far(units: tuple[UnitResult, ...], usage: Usage) -> UnitsSoFar:
    first_problem = next(
        (
            unit.problem
            for unit in units
            if not unit.validated and unit.problem is not None
        ),
        None,
    )
    return UnitsSoFar(
        outputs=tuple(unit.text for unit in units),
        usage=reduce(usage_add, (unit.usage for unit in units), usage),
        validated=all(unit.validated for unit in units),
        problem=first_problem,
    )


def run_units(
    environment: CallEnvironment,
    work_groups: tuple[tuple[str, ...], ...],
    plan: tuple[tuple[str, ...], ...],
    trailing_separators: tuple[str, ...],
    analysis: str | None,
    usage: Usage,
    retry_attempt: int = 0,
) -> IO[Result[UnitsSoFar, TranslationError]]:
    run_context = environment.run_context
    pass_definition = environment.pass_definition
    calls = plan_unit_calls(
        run_context,
        pass_definition,
        plan,
        work_groups,
        trailing_separators,
        analysis,
        retry_attempt,
        environment.glossary,
    )

    def unit_part(
        call: UnitCall,
    ) -> IO[Result[UnitResult, TranslationError]]:
        if call.source_chunk:
            return run_single_call(environment, call, usage)

        return io_map(
            verbose_log(
                run_context,
                unit_no_source_message(
                    pass_definition.name, call.index + 1, call.total
                ),
            ),
            lambda _: Ok(
                UnitResult(
                    text=call.work_chunk + call.trailing_separator,
                    validated=True,
                    usage=Usage(),
                    problem=None,
                )
            ),
        )

    return io_result_map(
        io_traverse(calls, unit_part),
        lambda units: units_so_far(units, usage),
    )


def text_pass_unit_plan(
    pass_definition: PassDefinition,
    chunk_plan: PassPlan,
    paragraph_plan: tuple[tuple[str, ...], ...],
) -> tuple[tuple[str, ...], ...]:
    match pass_definition.mode:
        case "paragraph":
            return paragraph_plan

        case _:
            return chunk_plan.chunks


def state_after_units(state: State, units: UnitsSoFar) -> State:
    return state_with_output(
        state,
        ensure_blank_line_separators("".join(units.outputs)),
        units.usage,
        units.validated,
        units.problem,
    )


def run_text_pass_once(
    execution: PassExecution,
    state: State,
    started_at: float,
    retry_attempt: int = 0,
    retry_same_mode: PassRetry | None = None,
) -> IO[StateResult]:
    run_context = execution.run_context
    pass_definition = execution.pass_definition
    plan = text_pass_unit_plan(
        pass_definition, execution.chunk_plan, execution.paragraph_plan
    )

    work_paragraphs, _ = split_paragraphs(state.text)
    work_groups, warning = resolve_work_groups(
        pass_definition.mode,
        work_paragraphs,
        plan,
        pass_definition.name,
        execution.chunk_plan.chunk_budget,
    )

    def run_units_after_warning(_: None) -> IO[StateResult]:
        def after_units(
            units_result: Result[UnitsSoFar, TranslationError],
        ) -> IO[StateResult]:
            if isinstance(units_result, Err):
                return io_result(units_result)

            return finish_pass_stage(
                run_context,
                pass_definition,
                state.usage,
                state_after_units(state, units_result.value),
                started_at,
                execution.source_paragraphs,
                retry_same_mode,
            )

        environment = CallEnvironment(
            run_context=run_context,
            pass_definition=pass_definition,
            glossary=state.glossary,
        )
        return io_and_then(
            verbose_log(
                run_context,
                plan_information_message(pass_definition, work_paragraphs, work_groups),
            ),
            io_bind(
                run_units(
                    environment,
                    work_groups,
                    plan,
                    unit_separators(plan, execution.separators),
                    state.analysis,
                    state.usage,
                    retry_attempt,
                ),
                after_units,
            ),
        )

    reported_warning: IO[None] = maybe_either(
        warning, run_context.console.log, lambda: io_pure(None)
    )
    return io_bind(reported_warning, run_units_after_warning)


def run_pass_attempt(
    execution: PassExecution,
    state: State,
    started_at: float,
    escalate: bool,
    retry_attempt: int = 0,
) -> IO[StateResult]:
    pass_definition = execution.pass_definition
    retry: PassRetry | None = None
    if escalate and pass_definition.mode == "chunk":
        retry_state = state

        def retry(accumulated: Usage, retry_started: float) -> IO[StateResult]:
            return run_pass_attempt(
                execution,
                replace(retry_state, usage=accumulated),
                retry_started,
                escalate=False,
                retry_attempt=1,
            )

    return run_text_pass_once(
        execution,
        state,
        started_at,
        retry_attempt,
        retry,
    )


def paragraph_count_is_accepted(count: int, source_count: int, tolerance: int) -> bool:
    return abs(count - source_count) <= tolerance


def after_paragraph_retry(
    run_context: Context,
    pass_definition: PassDefinition,
    source_count: int,
    tolerance: int,
    result: StateResult,
) -> IO[StateResult]:
    if isinstance(result, Err):
        return io_result(result)

    count = count_paragraphs(result.value.text)
    if paragraph_count_is_accepted(count, source_count, tolerance):
        return io_result(result)

    return io_map(
        run_context.console.log(
            paragraph_still_differs_message(pass_definition.name, count, source_count)
        ),
        lambda _: result,
    )


def verify_paragraph_count(
    execution: PassExecution,
    state: State,
    tolerance: int,
    result: StateResult,
) -> IO[StateResult]:
    run_context = execution.run_context
    pass_definition = execution.pass_definition
    source_count = len(execution.source_paragraphs)

    def mismatch_message(count: int, action: str) -> str:
        return paragraph_mismatch_message(
            pass_definition.name, count, source_count, tolerance, action
        )

    if isinstance(result, Err):
        return io_result(result)

    count = count_paragraphs(result.value.text)
    if paragraph_count_is_accepted(count, source_count, tolerance):
        return io_result(result)

    match pass_definition.mode:
        case "chunk":
            pass

        case _:
            return io_map(
                run_context.console.log(mismatch_message(count, "continuing")),
                lambda _: result,
            )

    def retry_at(retry_started: float) -> IO[StateResult]:
        paragraph_execution = replace(
            execution,
            pass_definition=replace(pass_definition, mode="paragraph"),
        )
        return io_bind(
            run_pass_attempt(
                paragraph_execution,
                replace(state, usage=result.value.usage),
                retry_started,
                escalate=True,
            ),
            lambda retried: after_paragraph_retry(
                run_context, pass_definition, source_count, tolerance, retried
            ),
        )

    return io_and_then(
        run_context.console.log(
            mismatch_message(count, "re-running the pass with one call per paragraph")
        ),
        io_bind(now(run_context.clock), retry_at),
    )


def run_text_pass(
    execution: PassExecution,
    state: State,
    started_at: float,
) -> IO[StateResult]:
    tolerance = paragraph_tolerance(execution.pass_definition.ensure_paragraphs)
    if tolerance is None:
        return run_pass_attempt(execution, state, started_at, escalate=True)

    return io_bind(
        run_pass_attempt(execution, state, started_at, escalate=True),
        lambda result: verify_paragraph_count(execution, state, tolerance, result),
    )


def run_pass(
    execution: PassExecution,
    state: State,
    started_at: float,
) -> IO[StateResult]:
    match execution.pass_definition.mode:
        case "analysis":
            return run_analysis_pass(execution, state, started_at)

        case "update_glossary":
            return run_update_glossary_pass(execution, state, started_at)

        case _:
            return run_text_pass(execution, state, started_at)


def conclude_pass_result(
    run_context: Context,
    pass_definition: PassDefinition,
    result: StateResult,
) -> StateResult:
    if isinstance(result, Err):
        return result

    if result.value.validated or run_context.best_effort:
        return result

    return fail_invalid_output(
        pass_definition.name,
        run_context.settings.unit_fix_attempts,
        result.value.problem or "the output failed validation",
    )


def run_passes(
    executions: tuple[PassExecution, ...],
    state: State,
) -> IO[StateResult]:
    def step(
        current_state: State, indexed: tuple[int, PassExecution]
    ) -> IO[StateResult]:
        number, execution = indexed

        def begin_pass_at(stage_started: float) -> IO[StateResult]:
            return io_and_then(
                execution.run_context.console.log(
                    pass_started_message(
                        number,
                        len(executions),
                        execution.pass_definition.name,
                    )
                ),
                io_map(
                    run_pass(execution, current_state, stage_started),
                    lambda result: conclude_pass_result(
                        execution.run_context, execution.pass_definition, result
                    ),
                ),
            )

        return io_bind(now(execution.run_context.clock), begin_pass_at)

    return fold_io(enumerate(executions, 1), step, Ok(state))


def build_pass_executions(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plans: tuple[PassPlan, ...],
) -> tuple[PassExecution, ...]:
    paragraph_plan = tuple((paragraph,) for paragraph in source_paragraphs)
    return tuple(
        PassExecution(
            run_context=run_context,
            pass_definition=pass_definition,
            source_paragraphs=source_paragraphs,
            separators=separators,
            chunk_plan=chunk_plan,
            paragraph_plan=paragraph_plan,
        )
        for pass_definition, chunk_plan in zip(
            pass_definitions, chunk_plans, strict=True
        )
    )


def run_pipeline(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
    started: float,
    stdout: TextIO,
) -> IO[int]:
    source_paragraphs, separators = split_paragraphs(text)

    def conclude(result: StateResult) -> IO[int]:
        if isinstance(result, Err):
            return io_map(run_context.console.log(describe(result.error)), lambda _: 1)

        return finish_output(run_context, result.value, started, stdout)

    def run_all_passes(chunk_plans: tuple[PassPlan, ...]) -> IO[StateResult]:
        return run_passes(
            build_pass_executions(
                run_context,
                pass_definitions,
                source_paragraphs,
                separators,
                chunk_plans,
            ),
            State(
                text=text,
                analysis=None,
                usage=Usage(),
                validated=True,
                problem=None,
                glossary=run_context.glossary_entries,
            ),
        )

    return io_bind(
        io_result_bind(
            io_result(
                pass_chunk_plans(run_context, pass_definitions, source_paragraphs)
            ),
            run_all_passes,
        ),
        conclude,
    )


def finish_output(
    run_context: Context,
    state: State,
    started: float,
    stdout: TextIO,
) -> IO[int]:
    def after_write(_: None) -> IO[int]:
        def report_total(ended_at: float) -> IO[int]:
            elapsed = ended_at - started

            def after_done(_: None) -> IO[int]:
                return io_map(
                    run_context.console.log(
                        usage_line(
                            "TOTAL",
                            elapsed,
                            state.usage.prompt_tokens,
                            state.usage.completion_tokens,
                            state.usage.cost,
                        )
                    ),
                    lambda _: 0,
                )

            return io_bind(
                verbose_log(run_context, done_in_message(elapsed)),
                after_done,
            )

        return io_bind(now(run_context.clock), report_total)

    return io_bind(write_stdout(stdout, state.text), after_write)

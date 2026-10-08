from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import reduce
from typing import TextIO

from l2l.ascii import enforce_pass_ascii
from l2l.cache import (
    always_acceptable,
    cache_lookup,
    cache_store,
    cached_translation,
    non_empty,
)
from l2l.console import Console
from l2l.effects import now, write_stdout
from l2l.errors import (
    TranslationError,
    describe,
    fail_budget,
    fail_invalid_output,
    fail_pass,
    fail_unit,
    fail_untranslated,
)
from l2l.http import RepairOutcome, chat, repaired_call
from l2l.messages import (
    analysis_information_message,
    done_in_message,
    paragraph_mismatch_message,
    paragraph_still_differs_message,
    pass_cache_hit_message,
    pass_started_message,
    retranslate_information_message,
    retranslate_majority_message,
    retranslate_pair_message,
    retranslate_skipped_message,
    retranslate_unit_message,
    stage_done_line,
    unit_cache_hit_message,
    unit_done_message,
    unit_failed_validation_attempt,
    unit_failed_validation_final,
    unit_no_source_message,
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
    io_traverse,
    maybe_either,
    maybe_or,
    result_bind,
    result_map,
)
from l2l.plans import (
    UnitCall,
    build_pass_user,
    build_unit_retry_user,
    plan_information_message,
    plan_retranslation_calls,
    plan_unit_calls,
    resolve_work_groups,
    script_mismatch_problem,
    unit_output_problem,
    verbose_log,
)
from l2l.settings import (
    Context,
    PassDefinition,
    paragraph_tolerance,
    pass_salt,
    resolve_call_settings,
)
from l2l.text import (
    Translated,
    Usage,
    cache_key,
    count_paragraphs,
    detect_language_pair,
    ensure_blank_line_separators,
    estimate_tokens,
    excerpt,
    make_chunks,
    split_paragraphs,
    split_units_to_budget,
    unit_separators,
    untranslated_paragraph,
    usage_add,
    usage_delta,
)


@dataclass(frozen=True)
class State:
    text: str
    analysis: str | None
    usage: Usage
    validated: bool = True
    problem: str | None = None


@dataclass(frozen=True)
class UnitResult:
    text: str
    validated: bool
    usage: Usage
    problem: str | None = None


@dataclass(frozen=True)
class UnitsSoFar:
    outputs: tuple[str, ...]
    usage: Usage
    validated: bool
    problem: str | None


StateResult = Result[State, TranslationError]

PassRetry = Callable[[Usage, float], IO[StateResult]]


def untranslated_majority(flagged: int, total: int) -> bool:
    return flagged * 2 > total


def run_analysis(
    run_context: Context,
    pass_definition: PassDefinition,
    text: str,
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    return io_map(
        chat(run_context, pass_definition.instruction, text, model, parameters, usage),
        lambda result: result_map(
            result,
            lambda translated: Translated(translated.text.strip(), translated.usage),
        ),
    )


def analyze_document(
    run_context: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[Translated, TranslationError]]:
    budget = max(
        run_context.config.maximum_tokens
        - estimate_tokens(pass_definition.instruction)
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

    return run_analysis(run_context, pass_definition, full_text, usage)


def run_analysis_once(
    run_context: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[Translated, TranslationError]]:
    model, parameters = resolve_call_settings(run_context.config, pass_definition)
    key = cache_key(full_text, model, pass_salt(pass_definition), overrides=parameters)
    return cached_translation(
        run_context,
        key,
        usage,
        compute=lambda: analyze_document(
            run_context, pass_definition, full_text, usage
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


def conclude_state(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs: tuple[str, ...],
    retry_same_mode: PassRetry | None = None,
) -> IO[StateResult]:
    if pass_definition.mode == "analysis":
        return io_result(Ok(state))

    def ascii_stage(current: State) -> IO[StateResult]:
        if not pass_definition.ascii:
            return io_result(Ok(current))

        def enforce(ascii_started: float) -> IO[StateResult]:
            return io_map(
                enforce_pass_ascii(
                    run_context,
                    pass_definition,
                    current.text,
                    source_paragraphs,
                    current.usage,
                    ascii_started,
                ),
                lambda result: result_map(
                    result,
                    lambda translated: State(
                        translated.text,
                        state.analysis,
                        translated.usage,
                        state.validated,
                        state.problem,
                    ),
                ),
            )

        return io_bind(now(run_context.clock), enforce)

    if not pass_definition.retranslate_untranslated:
        return ascii_stage(state)

    def after_retranslation(
        result: Result[Translated, TranslationError],
    ) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(Err(result.error))

        return ascii_stage(
            State(
                result.value.text,
                state.analysis,
                result.value.usage,
                state.validated,
                state.problem,
            )
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
            ),
            after_retranslation,
        )

    return io_bind(now(run_context.clock), retranslation_stage)


def retranslate_untranslated(
    run_context: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    analysis: str | None,
    usage: Usage,
    started_at: float,
    retry_same_mode: PassRetry | None = None,
) -> IO[Result[Translated, TranslationError]]:
    paragraphs, separators = split_paragraphs(text)
    if len(paragraphs) != len(source_paragraphs):
        return io_bind(
            verbose_log(
                run_context,
                retranslate_skipped_message(
                    pass_definition.name, len(paragraphs), len(source_paragraphs)
                ),
            ),
            lambda _: io_result(Ok(Translated(text, usage))),
        )

    pair = detect_language_pair(source_paragraphs, paragraphs)
    flagged = tuple(
        index
        for index, (source, output) in enumerate(
            zip(source_paragraphs, paragraphs, strict=True)
        )
        if untranslated_paragraph(
            source, output, pair, run_context.settings.ascii_character_map
        )
    )
    if not flagged:
        return io_result(Ok(Translated(text, usage)))

    if retry_same_mode is not None and untranslated_majority(
        len(flagged), len(paragraphs)
    ):
        retry = retry_same_mode

        def escalate(
            retry_started: float,
        ) -> IO[Result[Translated, TranslationError]]:
            def lifted(outcome: StateResult) -> Result[Translated, TranslationError]:
                if isinstance(outcome, Err):
                    return Err(outcome.error)

                return Ok(Translated(outcome.value.text, outcome.value.usage))

            return io_map(retry(usage, retry_started), lifted)

        return io_and_then(
            run_context.console.log(
                retranslate_majority_message(
                    pass_definition.name, len(flagged), len(paragraphs)
                )
            ),
            io_bind(now(run_context.clock), escalate),
        )

    def cacheable_retranslation(index: int) -> Callable[[str], bool]:
        def cacheable(text: str) -> bool:
            return not untranslated_paragraph(
                source_paragraphs[index],
                text,
                pair,
                run_context.settings.ascii_character_map,
            )

        return cacheable

    calls = plan_retranslation_calls(
        run_context,
        pass_definition,
        source_paragraphs,
        paragraphs,
        flagged,
        separators,
        analysis,
    )
    call_for = dict(zip(flagged, calls, strict=True))

    def part(
        indexed: tuple[int, str],
    ) -> IO[Result[UnitResult, TranslationError]]:
        index, paragraph = indexed
        call = call_for.get(index)
        if call is None:
            separator = separators[index] if index < len(separators) else ""
            return io_result(Ok(UnitResult(paragraph + separator, True, Usage())))

        return io_and_then(
            verbose_log(
                run_context,
                retranslate_unit_message(
                    pass_definition.name, call.index + 1, call.total
                ),
            ),
            run_single_call(
                run_context,
                pass_definition,
                call,
                usage,
                cacheable_retranslation(index),
            ),
        )

    def collect(
        units: tuple[UnitResult, ...],
    ) -> Result[Translated, TranslationError]:
        texts = tuple(unit.text for unit in units)
        for index in flagged:
            separator = separators[index] if index < len(separators) else ""
            body = texts[index].removesuffix(separator)
            if untranslated_paragraph(
                source_paragraphs[index],
                body,
                pair,
                run_context.settings.ascii_character_map,
            ):
                return fail_untranslated(pass_definition.name, index, excerpt(body))

        return Ok(
            Translated(
                "".join(texts),
                reduce(usage_add, (unit.usage for unit in units), usage),
            )
        )

    def conclude(
        result: Result[Translated, TranslationError],
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

    def start(_: None) -> IO[Result[Translated, TranslationError]]:
        return io_map(
            io_traverse(enumerate(paragraphs), part),
            lambda result: result_bind(result, collect),
        )

    return io_and_then(
        run_context.console.log(
            retranslate_information_message(
                pass_definition.name, len(flagged), len(paragraphs)
            )
        ),
        io_bind(
            run_context.console.log(retranslate_pair_message(pair)),
            lambda _: io_bind(start(None), conclude),
        ),
    )


def run_analysis_pass(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
) -> IO[StateResult]:
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
            State(
                text=state.text,
                analysis=outcome.text,
                usage=outcome.usage,
                validated=state.validated,
                problem=state.problem,
            ),
            started_at,
            source_paragraphs,
        )

    return io_bind(
        verbose_log(
            run_context,
            analysis_information_message(
                pass_definition.name, len(state.text), estimate_tokens(state.text)
            ),
        ),
        lambda _: io_bind(
            run_analysis_once(run_context, pass_definition, state.text, state.usage),
            after_analysis,
        ),
    )


def run_unit(
    run_context: Context,
    pass_definition: PassDefinition,
    call: UnitCall,
    usage: Usage,
) -> IO[Result[UnitResult, TranslationError]]:
    def unit_tolerance() -> int | None:
        match pass_definition.mode:
            case "paragraph":
                return 0

            case _:
                return paragraph_tolerance(pass_definition.ensure_paragraphs)

    def validate(output: str) -> Maybe[str]:
        return maybe_or(
            unit_output_problem(
                call.source_chunk, output, run_context.settings, unit_tolerance()
            ),
            script_mismatch_problem(call.source_chunk, call.work_chunk, output),
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

    def assessed(
        outcome: Result[RepairOutcome, TranslationError],
    ) -> Result[UnitResult, TranslationError]:
        return result_map(
            outcome,
            lambda repaired: UnitResult(
                repaired.text,
                repaired.validated,
                repaired.usage,
                repaired.problem,
            ),
        )

    return io_map(
        repaired_call(
            run_context,
            call.model,
            call.parameters,
            pass_definition.instruction,
            build_pass_user(
                call.source_chunk, call.work_chunk, call.analysis, call.context
            ),
            validate,
            build_retry_user,
            run_context.settings.unit_fix_attempts,
            on_repair,
            on_exhausted,
        )(usage),
        assessed,
    )


def run_single_call(
    run_context: Context,
    pass_definition: PassDefinition,
    call: UnitCall,
    usage: Usage,
    cacheable: Callable[[str], bool] = always_acceptable,
) -> IO[Result[UnitResult, TranslationError]]:
    def store(
        result: Result[UnitResult, TranslationError],
    ) -> IO[Result[UnitResult, TranslationError]]:
        if isinstance(result, Err):
            return io_result(
                fail_unit(pass_definition.name, call.index + 1, result.error)
            )

        outcome = result.value
        stored = io_map(
            cache_store(
                run_context,
                call.key,
                outcome.text,
                outcome.validated and cacheable(outcome.text),
            ),
            lambda _: verbose_log(
                run_context,
                unit_done_message(pass_definition.name, call.index + 1, call.total),
            ),
        )
        return io_map(
            stored,
            lambda _: Ok(
                UnitResult(
                    outcome.text + call.trailing_separator,
                    outcome.validated,
                    Usage(*usage_delta(usage, outcome.usage)),
                    outcome.problem,
                )
            ),
        )

    def proceed(
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
                    UnitResult(cached.value + call.trailing_separator, True, Usage())
                ),
            )

        return io_bind(
            run_unit(run_context, pass_definition, call, usage),
            store,
        )

    return io_bind(cache_lookup(run_context, call.key, acceptable=non_empty), proceed)


def run_units(
    run_context: Context,
    pass_definition: PassDefinition,
    work_groups: tuple[tuple[str, ...], ...],
    plan: tuple[tuple[str, ...], ...],
    trailing_separators: tuple[str, ...],
    analysis: str | None,
    usage: Usage,
    retry_attempt: int = 0,
) -> IO[Result[UnitsSoFar, TranslationError]]:
    calls = plan_unit_calls(
        run_context,
        pass_definition,
        plan,
        work_groups,
        trailing_separators,
        analysis,
        retry_attempt,
    )

    def unit_part(
        call: UnitCall,
    ) -> IO[Result[UnitResult, TranslationError]]:
        if not call.source_chunk:
            return io_map(
                verbose_log(
                    run_context,
                    unit_no_source_message(
                        pass_definition.name, call.index + 1, call.total
                    ),
                ),
                lambda _: Ok(
                    UnitResult(call.work_chunk + call.trailing_separator, True, Usage())
                ),
            )

        return run_single_call(run_context, pass_definition, call, usage)

    def first_unvalidated_problem(
        units: tuple[UnitResult, ...],
    ) -> str | None:
        return next(
            (
                unit.problem
                for unit in units
                if not unit.validated and unit.problem is not None
            ),
            None,
        )

    def collect(units: tuple[UnitResult, ...]) -> UnitsSoFar:
        return UnitsSoFar(
            tuple(unit.text for unit in units),
            reduce(usage_add, (unit.usage for unit in units), usage),
            all(unit.validated for unit in units),
            first_unvalidated_problem(units),
        )

    return io_map(
        io_traverse(calls, unit_part),
        lambda result: result_map(result, collect),
    )


def run_text_pass_once(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
    retry_attempt: int = 0,
    retry_same_mode: PassRetry | None = None,
) -> IO[StateResult]:
    match pass_definition.mode:
        case "paragraph":
            plan = paragraph_plan

        case _:
            plan = chunk_plan

    work_paragraphs, _ = split_paragraphs(state.text)
    work_groups, warning = resolve_work_groups(
        pass_definition.mode,
        work_paragraphs,
        plan,
        pass_definition.name,
        run_context.settings.chunk_budget_tokens,
    )

    def proceed(_: None) -> IO[StateResult]:
        def after_units(
            units_result: Result[UnitsSoFar, TranslationError],
        ) -> IO[StateResult]:
            if isinstance(units_result, Err):
                return io_result(units_result)

            next_state = State(
                text=ensure_blank_line_separators("".join(units_result.value.outputs)),
                analysis=state.analysis,
                usage=units_result.value.usage,
                validated=state.validated and units_result.value.validated,
                problem=(
                    units_result.value.problem
                    if units_result.value.problem is not None
                    else state.problem
                ),
            )
            return finish_pass_stage(
                run_context,
                pass_definition,
                state.usage,
                next_state,
                started_at,
                source_paragraphs,
                retry_same_mode,
            )

        return io_and_then(
            verbose_log(
                run_context,
                plan_information_message(pass_definition, work_paragraphs, work_groups),
            ),
            io_bind(
                run_units(
                    run_context,
                    pass_definition,
                    work_groups,
                    plan,
                    unit_separators(plan, separators),
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
    return io_bind(reported_warning, proceed)


def run_text_pass(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
) -> IO[StateResult]:
    def attempt(
        pass_to_run: PassDefinition,
        current_state: State,
        attempt_started: float,
        escalate: bool,
        retry_attempt: int = 0,
    ) -> IO[StateResult]:
        retry: PassRetry | None = None
        if escalate and pass_to_run.mode == "chunk":
            retry_state = current_state

            def retry(accumulated: Usage, retry_started: float) -> IO[StateResult]:
                return attempt(
                    pass_to_run,
                    replace(retry_state, usage=accumulated),
                    retry_started,
                    escalate=False,
                    retry_attempt=1,
                )

        return run_text_pass_once(
            run_context,
            pass_to_run,
            current_state,
            attempt_started,
            source_paragraphs,
            separators,
            chunk_plan,
            paragraph_plan,
            retry_attempt,
            retry,
        )

    tolerance = paragraph_tolerance(pass_definition.ensure_paragraphs)

    if tolerance is None:
        return attempt(pass_definition, state, started_at, escalate=True)

    def mismatch_message(count: int, action: str) -> str:
        return paragraph_mismatch_message(
            pass_definition.name, count, len(source_paragraphs), tolerance, action
        )

    def accepted(count: int) -> bool:
        return abs(count - len(source_paragraphs)) <= tolerance

    def after_retry(result: StateResult) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if accepted(count):
            return io_result(result)

        return io_map(
            run_context.console.log(
                paragraph_still_differs_message(
                    pass_definition.name, count, len(source_paragraphs)
                )
            ),
            lambda _: result,
        )

    def check(result: StateResult) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if accepted(count):
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
            return io_bind(
                attempt(
                    replace(pass_definition, mode="paragraph"),
                    replace(state, usage=result.value.usage),
                    retry_started,
                    escalate=True,
                ),
                after_retry,
            )

        return io_and_then(
            run_context.console.log(
                mismatch_message(
                    count, "re-running the pass with one call per paragraph"
                )
            ),
            io_bind(now(run_context.clock), retry_at),
        )

    return io_bind(attempt(pass_definition, state, started_at, escalate=True), check)


def run_pass(
    run_context: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
) -> IO[StateResult]:
    match pass_definition.mode:
        case "analysis":
            return run_analysis_pass(
                run_context, pass_definition, state, started_at, source_paragraphs
            )

        case _:
            return run_text_pass(
                run_context,
                pass_definition,
                state,
                started_at,
                source_paragraphs,
                separators,
                chunk_plan,
                paragraph_plan,
            )


def conclude_pass_result(
    run_context: Context,
    pass_definition: PassDefinition,
    result: StateResult,
) -> StateResult:
    if isinstance(result, Err):
        return result

    if result.value.validated or run_context.best_effort:
        return result

    return fail_pass(
        pass_definition.name,
        fail_invalid_output(
            run_context.settings.unit_fix_attempts,
            result.value.problem or "the output failed validation",
        ).error,
    )


def run_passes(
    run_context: Context,
    pass_definitions: tuple[PassDefinition, ...],
    state: State,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
) -> IO[StateResult]:
    def step(
        current_state: State, indexed: tuple[int, PassDefinition]
    ) -> IO[StateResult]:
        number, pass_definition = indexed

        def launch(stage_started: float) -> IO[StateResult]:
            return io_and_then(
                run_context.console.log(
                    pass_started_message(
                        number, len(pass_definitions), pass_definition.name
                    )
                ),
                io_map(
                    run_pass(
                        run_context,
                        pass_definition,
                        current_state,
                        stage_started,
                        source_paragraphs,
                        separators,
                        chunk_plan,
                        paragraph_plan,
                    ),
                    lambda result: conclude_pass_result(
                        run_context, pass_definition, result
                    ),
                ),
            )

        return io_bind(now(run_context.clock), launch)

    return fold_io(enumerate(pass_definitions, 1), step, Ok(state))


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

    return io_bind(
        run_passes(
            run_context,
            pass_definitions,
            State(text=text, analysis=None, usage=Usage()),
            source_paragraphs,
            separators,
            make_chunks(source_paragraphs, run_context.settings.chunk_budget_tokens),
            tuple((paragraph,) for paragraph in source_paragraphs),
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

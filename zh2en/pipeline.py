from __future__ import annotations

from dataclasses import dataclass, replace
from functools import reduce
from typing import TextIO

from zh2en.ascii import enforce_pass_ascii
from zh2en.cache import cache_lookup, cache_store, cached_translation, non_empty
from zh2en.console import Console
from zh2en.effects import now, write_stdout
from zh2en.errors import (
    TranslationError,
    describe,
    fail_budget,
    fail_pass,
    fail_unit,
)
from zh2en.http import RepairOutcome, chat, repaired_call
from zh2en.messages import (
    analysis_info_message,
    done_in_message,
    paragraph_mismatch_message,
    paragraph_still_differs_message,
    pass_cache_hit_message,
    pass_started_message,
    unit_cache_hit_message,
    unit_done_message,
    unit_failed_validation_attempt,
    unit_failed_validation_final,
    unit_no_source_message,
    usage_line,
)
from zh2en.monads import (
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
    result_map,
)
from zh2en.plans import (
    UnitCall,
    build_pass_user,
    build_unit_retry_user,
    plan_info_message,
    plan_unit_calls,
    resolve_work_groups,
    unit_output_problem,
    verbose_log,
)
from zh2en.settings import (
    Context,
    PassDefinition,
    pass_salt,
    resolve_call_settings,
)
from zh2en.text import (
    Translated,
    Usage,
    cache_key,
    count_paragraphs,
    ensure_blank_line_separators,
    estimate_tokens,
    make_chunks,
    split_paragraphs,
    split_units_to_budget,
    unit_separators,
    usage_add,
    usage_delta,
)


@dataclass(frozen=True)
class State:
    text: str
    analysis: str | None
    usage: Usage


@dataclass(frozen=True)
class UnitResult:
    text: str
    validated: bool
    usage: Usage


@dataclass(frozen=True)
class UnitsSoFar:
    outputs: tuple[str, ...]
    usage: Usage


StateResult = Result[State, TranslationError]


def run_analysis(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return io_map(
        chat(ctx, pass_definition.instruction, text, model, params, usage),
        lambda result: result_map(
            result,
            lambda translated: Translated(translated.text.strip(), translated.usage),
        ),
    )


def analyze_document(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[Translated, TranslationError]]:
    budget = max(
        ctx.config.max_tokens
        - estimate_tokens(pass_definition.instruction)
        - ctx.settings.analysis_reserve_tokens,
        1,
    )
    chunks = make_chunks(
        split_units_to_budget(
            split_paragraphs(full_text)[0],
            budget,
            ctx.settings.sentence_boundary_characters,
        ),
        budget,
    )
    if len(chunks) > 1:
        return io_result(
            fail_budget(estimate_tokens(full_text), ctx.config.max_tokens, len(chunks))
        )

    return run_analysis(ctx, pass_definition, full_text, usage)


def run_analysis_once(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[Translated, TranslationError]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    key = cache_key(full_text, model, pass_salt(pass_definition), overrides=params)
    return cached_translation(
        ctx,
        key,
        usage,
        compute=lambda: analyze_document(ctx, pass_definition, full_text, usage),
        hit_log=verbose_log(ctx, pass_cache_hit_message(pass_definition.name)),
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
    ctx: Context,
    pass_definition: PassDefinition,
    previous_usage: Usage,
    next_state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
) -> IO[StateResult]:
    """Log the stage's usage line, then apply the pass's conclusion."""

    def after_stage(ended_at: float) -> IO[StateResult]:
        def concluded(_: None) -> IO[StateResult]:
            return conclude_state(ctx, pass_definition, next_state, source_paragraphs)

        return io_bind(
            log_stage(
                ctx.console,
                "Done",
                started_at,
                ended_at,
                previous_usage,
                next_state.usage,
            ),
            concluded,
        )

    return io_bind(now(ctx.clock), after_stage)


def conclude_state(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    source_paragraphs: tuple[str, ...],
) -> IO[StateResult]:
    if not pass_definition.ascii:
        return io_result(Ok(state))

    def enforce(ascii_started: float) -> IO[StateResult]:
        return io_map(
            enforce_pass_ascii(
                ctx,
                pass_definition,
                state.text,
                source_paragraphs,
                state.usage,
                ascii_started,
            ),
            lambda result: result_map(
                result,
                lambda translated: State(
                    translated.text, state.analysis, translated.usage
                ),
            ),
        )

    return io_bind(now(ctx.clock), enforce)


def run_analysis_pass(
    ctx: Context,
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
            ctx,
            pass_definition,
            state.usage,
            State(text=state.text, analysis=outcome.text, usage=outcome.usage),
            started_at,
            source_paragraphs,
        )

    return io_bind(
        verbose_log(
            ctx,
            analysis_info_message(
                pass_definition.name, len(state.text), estimate_tokens(state.text)
            ),
        ),
        lambda _: io_bind(
            run_analysis_once(ctx, pass_definition, state.text, state.usage),
            after_analysis,
        ),
    )


def run_unit(
    ctx: Context,
    pass_definition: PassDefinition,
    call: UnitCall,
    analysis: str | None,
    usage: Usage,
) -> IO[Result[UnitResult, TranslationError]]:
    def validate(output: str) -> Maybe[str]:
        return unit_output_problem(call.source_chunk, output, ctx.settings)

    def build_retry_user(bad_output: str, problem: str) -> str:
        return build_unit_retry_user(
            call.source_chunk, call.context, bad_output, problem
        )

    def on_repair(failed: int, problem: str) -> IO[None]:
        return verbose_log(
            ctx,
            unit_failed_validation_attempt(
                pass_definition.name,
                call.index + 1,
                call.total,
                problem,
                failed,
                ctx.settings.unit_fix_attempts,
            ),
        )

    def on_exhausted(bad_output: str, problem: str) -> IO[None]:
        return ctx.console.log(
            unit_failed_validation_final(
                pass_definition.name,
                call.index + 1,
                call.total,
                ctx.settings.unit_fix_attempts,
                problem,
            )
        )

    def assessed(
        outcome: Result[RepairOutcome, TranslationError],
    ) -> Result[UnitResult, TranslationError]:
        return result_map(
            outcome, lambda repaired: UnitResult(repaired[0], repaired[2], repaired[1])
        )

    return io_map(
        repaired_call(
            ctx,
            call.model,
            call.params,
            pass_definition.instruction,
            build_pass_user(call.source_chunk, call.work_chunk, analysis, call.context),
            validate,
            build_retry_user,
            ctx.settings.unit_fix_attempts,
            on_repair,
            on_exhausted,
        )(usage),
        assessed,
    )


def run_units(
    ctx: Context,
    pass_definition: PassDefinition,
    work_groups: tuple[tuple[str, ...], ...],
    plan: tuple[tuple[str, ...], ...],
    trailing_separators: tuple[str, ...],
    analysis: str | None,
    usage: Usage,
) -> IO[Result[UnitsSoFar, TranslationError]]:
    calls = plan_unit_calls(
        ctx, pass_definition, plan, work_groups, trailing_separators
    )

    def unit_part(
        call: UnitCall,
    ) -> IO[Result[tuple[str, Usage], TranslationError]]:
        if not call.source_chunk:
            return io_map(
                verbose_log(
                    ctx,
                    unit_no_source_message(
                        pass_definition.name, call.index + 1, call.total
                    ),
                ),
                lambda _: Ok((call.work_chunk + call.trailing_separator, Usage())),
            )

        def store(
            result: Result[UnitResult, TranslationError],
        ) -> IO[Result[tuple[str, Usage], TranslationError]]:
            if isinstance(result, Err):
                return io_result(
                    fail_unit(pass_definition.name, call.index + 1, result.error)
                )

            outcome = result.value
            stored = io_map(
                cache_store(ctx, call.key, outcome.text, outcome.validated),
                lambda _: verbose_log(
                    ctx,
                    unit_done_message(pass_definition.name, call.index + 1, call.total),
                ),
            )
            return io_map(
                stored,
                lambda _: Ok(
                    (
                        outcome.text + call.trailing_separator,
                        Usage(*usage_delta(usage, outcome.usage)),
                    )
                ),
            )

        def proceed(
            cached: Maybe[str],
        ) -> IO[Result[tuple[str, Usage], TranslationError]]:
            if isinstance(cached, Just):
                return io_map(
                    verbose_log(
                        ctx,
                        unit_cache_hit_message(
                            pass_definition.name, call.index + 1, call.total
                        ),
                    ),
                    lambda _: Ok((cached.value + call.trailing_separator, Usage())),
                )

            return io_bind(
                run_unit(ctx, pass_definition, call, analysis, usage),
                store,
            )

        return io_bind(cache_lookup(ctx, call.key, acceptable=non_empty), proceed)

    def collect(parts: tuple[tuple[str, Usage], ...]) -> UnitsSoFar:
        return UnitsSoFar(
            tuple(text for text, _ in parts),
            reduce(usage_add, (delta for _, delta in parts), usage),
        )

    return io_map(
        io_traverse(calls, unit_part),
        lambda result: result_map(result, collect),
    )


def run_text_pass_once(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
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
        ctx.settings.chunk_budget_tokens,
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
            )
            return finish_pass_stage(
                ctx,
                pass_definition,
                state.usage,
                next_state,
                started_at,
                source_paragraphs,
            )

        return io_and_then(
            verbose_log(
                ctx, plan_info_message(pass_definition, work_paragraphs, work_groups)
            ),
            io_bind(
                run_units(
                    ctx,
                    pass_definition,
                    work_groups,
                    plan,
                    unit_separators(plan, separators),
                    state.analysis,
                    state.usage,
                ),
                after_units,
            ),
        )

    reported_warning: IO[None] = maybe_either(
        warning, ctx.console.log, lambda: io_pure(None)
    )
    return io_bind(reported_warning, proceed)


def run_text_pass(
    ctx: Context,
    pass_definition: PassDefinition,
    state: State,
    started_at: float,
    source_paragraphs: tuple[str, ...],
    separators: tuple[str, ...],
    chunk_plan: tuple[tuple[str, ...], ...],
    paragraph_plan: tuple[tuple[str, ...], ...],
) -> IO[StateResult]:
    def attempt(
        pass_to_run: PassDefinition, current_state: State, attempt_started: float
    ) -> IO[StateResult]:
        return run_text_pass_once(
            ctx,
            pass_to_run,
            current_state,
            attempt_started,
            source_paragraphs,
            separators,
            chunk_plan,
            paragraph_plan,
        )

    if not ctx.ensure_paragraphs:
        return attempt(pass_definition, state, started_at)

    def mismatch_message(count: int, action: str) -> str:
        return paragraph_mismatch_message(
            pass_definition.name, count, len(source_paragraphs), action
        )

    def after_retry(result: StateResult) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if count == len(source_paragraphs):
            return io_result(result)

        return io_map(
            ctx.console.log(
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
        if count == len(source_paragraphs):
            return io_result(result)

        match pass_definition.mode:
            case "chunk":
                pass

            case _:
                return io_map(
                    ctx.console.log(mismatch_message(count, "continuing")),
                    lambda _: result,
                )

        def retry_at(retry_started: float) -> IO[StateResult]:
            return io_bind(
                attempt(
                    replace(pass_definition, mode="paragraph"),
                    replace(state, usage=result.value.usage),
                    retry_started,
                ),
                after_retry,
            )

        return io_and_then(
            ctx.console.log(
                mismatch_message(
                    count, "re-running the pass with one call per paragraph"
                )
            ),
            io_bind(now(ctx.clock), retry_at),
        )

    return io_bind(attempt(pass_definition, state, started_at), check)


def run_pass(
    ctx: Context,
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
                ctx, pass_definition, state, started_at, source_paragraphs
            )

        case _:
            return run_text_pass(
                ctx,
                pass_definition,
                state,
                started_at,
                source_paragraphs,
                separators,
                chunk_plan,
                paragraph_plan,
            )


def run_passes(
    ctx: Context,
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
                ctx.console.log(
                    pass_started_message(
                        number, len(pass_definitions), pass_definition.name
                    )
                ),
                run_pass(
                    ctx,
                    pass_definition,
                    current_state,
                    stage_started,
                    source_paragraphs,
                    separators,
                    chunk_plan,
                    paragraph_plan,
                ),
            )

        return io_bind(now(ctx.clock), launch)

    return fold_io(enumerate(pass_definitions, 1), step, Ok(state))


def run_pipeline(
    ctx: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
    started: float,
    stdout: TextIO,
) -> IO[int]:
    source_paragraphs, separators = split_paragraphs(text)

    def conclude(result: StateResult) -> IO[int]:
        if isinstance(result, Err):
            return io_map(ctx.console.log(describe(result.error)), lambda _: 1)

        return finish_output(ctx, result.value, started, stdout)

    return io_bind(
        run_passes(
            ctx,
            pass_definitions,
            State(text=text, analysis=None, usage=Usage()),
            source_paragraphs,
            separators,
            make_chunks(source_paragraphs, ctx.settings.chunk_budget_tokens),
            tuple((paragraph,) for paragraph in source_paragraphs),
        ),
        conclude,
    )


def finish_output(
    ctx: Context,
    state: State,
    started: float,
    stdout: TextIO,
) -> IO[int]:
    def after_write(_: None) -> IO[int]:
        def report_total(total_elapsed: float) -> IO[int]:
            def after_done(_: None) -> IO[int]:
                return io_map(
                    ctx.console.log(
                        usage_line(
                            "TOTAL",
                            total_elapsed,
                            state.usage.prompt_tokens,
                            state.usage.completion_tokens,
                            state.usage.cost,
                        )
                    ),
                    lambda _: 0,
                )

            return io_bind(
                verbose_log(ctx, done_in_message(total_elapsed)),
                after_done,
            )

        return io_bind(now(ctx.clock), report_total)

    return io_bind(write_stdout(stdout, state.text), after_write)

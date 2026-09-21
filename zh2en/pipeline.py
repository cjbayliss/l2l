from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TextIO

from zh2en.config import (
    Context,
    PassDefinition,
    pass_salt,
    resolve_call_settings,
)
from zh2en.console import Console
from zh2en.effects import cache_read, cache_write, now, write_stdout
from zh2en.errors import (
    TranslationError,
    describe,
    fail_ascii,
    fail_budget,
    fail_pass,
    fail_unit,
)
from zh2en.http import chat
from zh2en.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_when,
    result_bind_io,
    result_map,
)
from zh2en.plans import (
    UnitCall,
    ascii_drop_warning,
    build_ascii_fix_user,
    build_ascii_retry_user,
    build_pass_user,
    build_unit_retry_user,
    plan_info_message,
    plan_unit_calls,
    resolve_work_groups,
    unit_output_problem,
)
from zh2en.text import (
    CACHE_SALT_VERSION,
    Translated,
    Usage,
    cache_key,
    count_paragraphs,
    drop_non_ascii,
    ensure_blank_line_separators,
    estimate_tokens,
    make_chunks,
    split_paragraphs,
    split_units_to_budget,
    to_ascii_mechanical,
    unit_separators,
    usage_delta,
    usage_line,
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


def verbose_log(ctx: Context, message: str) -> IO[None]:
    return io_when(ctx.verbose, ctx.console.log(message))


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
            lambda translated: Translated(
                translated.text.strip(), translated.usage
            ),
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

    def compute(current_usage: Usage) -> IO[Result[Translated, TranslationError]]:
        def store(
            result: Result[Translated, TranslationError],
        ) -> IO[Result[Translated, TranslationError]]:
            if isinstance(result, Err):
                return io_result(result)

            return io_map(
                io_when(
                    ctx.use_cache,
                    cache_write(ctx.cache_directory, key, result.value.text),
                ),
                lambda _: result,
            )

        return io_bind(
            analyze_document(ctx, pass_definition, full_text, current_usage), store
        )

    if not ctx.use_cache:
        return compute(usage)

    def use_cached(cached: str | None) -> IO[Result[Translated, TranslationError]]:
        if cached is None:
            return compute(usage)

        return io_map(
            verbose_log(ctx, "zh2en: [%s] cache hit" % pass_definition.name),
            lambda _: Ok(Translated(cached, usage)),
        )

    return io_bind(cache_read(ctx.cache_directory, key), use_cached)


def translate_chunk(
    ctx: Context,
    pass_definition: PassDefinition,
    source_chunk: str,
    work_chunk: str | None,
    analysis: str | None,
    usage: Usage,
    context: tuple[str, ...] = (),
) -> IO[Result[Translated, TranslationError]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return chat(
        ctx,
        pass_definition.instruction,
        build_pass_user(source_chunk, work_chunk, analysis, context),
        model,
        params,
        usage,
    )


def ascii_fix_llm(
    ctx: Context,
    pass_definition: PassDefinition,
    source_paragraph: str,
    output_paragraph: str,
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    key = cache_key(
        source_paragraph,
        model,
        "\x00".join(
            (
                CACHE_SALT_VERSION,
                "ascii-fix",
                pass_definition.name,
                ctx.settings.ascii_fix_instruction,
            )
        ),
        output_paragraph,
        overrides=params,
    )

    def attempt(
        index: int, user: str, last_result: str, current_usage: Usage
    ) -> IO[Result[Translated, TranslationError]]:
        if index > ctx.settings.ascii_fix_attempts:
            return io_result(Ok(Translated(last_result, current_usage)))

        return io_bind(
            chat(
                ctx,
                ctx.settings.ascii_fix_instruction,
                user,
                model,
                params,
                current_usage,
            ),
            lambda result: result_bind_io(
                result,
                lambda translated: ascii_outcome(
                    index, translated.text, translated.usage
                ),
            ),
        )

    def ascii_outcome(
        index: int, result: str, current_usage: Usage
    ) -> IO[Result[Translated, TranslationError]]:
        if result.isascii():
            return io_map(
                io_when(ctx.use_cache, cache_write(ctx.cache_directory, key, result)),
                lambda _: Ok(Translated(result, current_usage)),
            )

        return io_bind(
            verbose_log(
                ctx,
                "zh2en: ascii: attempt %d/%d still non-ASCII; retrying"
                % (index, ctx.settings.ascii_fix_attempts),
            ),
            lambda _: attempt(
                index + 1,
                build_ascii_retry_user(source_paragraph, output_paragraph, result),
                result,
                current_usage,
            ),
        )

    def start(current_usage: Usage) -> IO[Result[Translated, TranslationError]]:
        return attempt(
            1,
            build_ascii_fix_user(source_paragraph, output_paragraph),
            "",
            current_usage,
        )

    if not ctx.use_cache:
        return start(usage)

    def use_cached(cached: str | None) -> IO[Result[Translated, TranslationError]]:
        if cached is None or not cached.isascii():
            return start(usage)

        return io_map(
            verbose_log(ctx, "zh2en: ascii: cache hit"),
            lambda _: Ok(Translated(cached, usage)),
        )

    return io_bind(cache_read(ctx.cache_directory, key), use_cached)


def repair_paragraph(
    ctx: Context,
    pass_definition: PassDefinition,
    paragraph: str,
    separator: str,
    index: int,
    total: int,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    mechanical = to_ascii_mechanical(paragraph, ctx.settings.ascii_character_map)
    if mechanical.isascii():
        return io_map(
            verbose_log(
                ctx,
                "zh2en: ascii: paragraph %d/%d converted mechanically"
                % (index + 1, total),
            ),
            lambda _: Ok(Translated(mechanical + separator, usage)),
        )

    def repaired(
        result: Result[Translated, TranslationError],
    ) -> IO[Result[Translated, TranslationError]]:
        if isinstance(result, Err):
            return io_result(result)

        final, drop = drop_non_ascii(
            result.value.text,
            ctx.settings.ascii_character_map,
            ctx.settings.ascii_fix_attempts,
            index,
        )

        def emit(_: None) -> Result[Translated, TranslationError]:
            return Ok(Translated(final + separator, result.value.usage))

        return (
            io_map(ctx.console.log(ascii_drop_warning(drop)), emit)
            if drop is not None
            else io_result(emit(None))
        )

    return io_and_then(
        verbose_log(
            ctx,
            "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
            "LLM to repair it" % (index + 1, total),
        ),
        io_bind(
            ascii_fix_llm(
                ctx,
                pass_definition,
                (
                    source_paragraphs[index]
                    if index < len(source_paragraphs)
                    else "(unavailable)"
                ),
                paragraph,
                usage,
            ),
            repaired,
        ),
    )


def ensure_ascii_output(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
) -> IO[Result[Translated, TranslationError]]:
    paragraphs, separators = split_paragraphs(text)

    def step(
        accumulated: UnitsSoFar, indexed: tuple[int, str]
    ) -> IO[Result[UnitsSoFar, TranslationError]]:
        index, paragraph = indexed
        separator = separators[index] if index < len(separators) else ""
        if paragraph.isascii():
            return io_result(
                Ok(
                    UnitsSoFar(
                        accumulated.outputs + (paragraph + separator,),
                        accumulated.usage,
                    )
                )
            )

        return io_map(
            repair_paragraph(
                ctx,
                pass_definition,
                paragraph,
                separator,
                index,
                len(paragraphs),
                source_paragraphs,
                accumulated.usage,
            ),
            lambda result: result_map(
                result,
                lambda translated: UnitsSoFar(
                    accumulated.outputs + (translated.text,), translated.usage
                ),
            ),
        )

    return io_map(
        fold_io(enumerate(paragraphs), step, Ok(UnitsSoFar((), usage))),
        lambda result: result_map(
            result,
            lambda collected: Translated(
                "".join(collected.outputs), collected.usage
            ),
        ),
    )


def enforce_pass_ascii(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
    started_at: float,
) -> IO[Result[Translated, TranslationError]]:
    def conclude(
        result: Result[Translated, TranslationError],
    ) -> IO[Result[Translated, TranslationError]]:
        if isinstance(result, Err):
            return io_map(
                ctx.console.interrupt(),
                lambda _: fail_ascii(pass_definition.name, result.error),
            )

        prompt, completion, cost = usage_delta(usage, result.value.usage)

        def report(ended_at: float) -> IO[Result[Translated, TranslationError]]:
            return io_map(
                ctx.console.finish(
                    usage_line("Done", ended_at - started_at, prompt, completion, cost)
                    if prompt or completion or cost
                    else "Done."
                ),
                lambda _: result,
            )

        return io_bind(now(ctx.clock), report)

    return io_and_then(
        ctx.console.write_partial("Enforcing ASCII... "),
        io_bind(
            ensure_ascii_output(ctx, pass_definition, text, source_paragraphs, usage),
            conclude,
        ),
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

        def after_stage(ended_at: float) -> IO[StateResult]:
            return io_bind(
                log_stage(
                    ctx.console,
                    "Done",
                    started_at,
                    ended_at,
                    state.usage,
                    outcome.usage,
                ),
                lambda _: conclude_state(
                    ctx,
                    pass_definition,
                    State(text=state.text, analysis=outcome.text, usage=outcome.usage),
                    source_paragraphs,
                ),
            )

        return io_bind(now(ctx.clock), after_stage)

    return io_bind(
        verbose_log(
            ctx,
            "zh2en: [%s] whole-document analysis (%d characters, ~%d tokens)"
            % (pass_definition.name, len(state.text), estimate_tokens(state.text)),
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
    model, params = resolve_call_settings(ctx.config, pass_definition)

    def assess_initial(
        translated: str, unit_usage: Usage
    ) -> IO[Result[UnitResult, TranslationError]]:
        problem = unit_output_problem(call.source_chunk, translated, ctx.settings)
        if problem is None:
            return io_result(Ok(UnitResult(translated, True, unit_usage)))

        return repair(1, translated, unit_usage, problem)

    def repair(
        attempt_index: int,
        bad_output: str,
        unit_usage: Usage,
        problem: str,
    ) -> IO[Result[UnitResult, TranslationError]]:
        if attempt_index > ctx.settings.unit_fix_attempts:
            return io_map(
                ctx.console.log(
                    "zh2en: [%s] unit %d/%d failed validation %d time(s); "
                    "last problem: %s. Keeping the last reply, uncached"
                    % (
                        pass_definition.name,
                        call.index + 1,
                        call.total,
                        ctx.settings.unit_fix_attempts,
                        problem,
                    )
                ),
                lambda _: Ok(UnitResult(bad_output, False, unit_usage)),
            )

        return io_bind(
            verbose_log(
                ctx,
                "zh2en: [%s] unit %d/%d failed validation (%s); repair "
                "attempt %d/%d"
                % (
                    pass_definition.name,
                    call.index + 1,
                    call.total,
                    problem,
                    attempt_index,
                    ctx.settings.unit_fix_attempts,
                ),
            ),
            lambda _: io_bind(
                chat(
                    ctx,
                    pass_definition.instruction,
                    build_unit_retry_user(
                        call.source_chunk, call.context, bad_output, problem
                    ),
                    model,
                    params,
                    unit_usage,
                ),
                settled(attempt_index),
            ),
        )

    def settled(
        attempt_index: int,
    ) -> Callable[
        [Result[Translated, TranslationError]], IO[Result[UnitResult, TranslationError]]
    ]:
        def continue_after(
            reply_result: Result[Translated, TranslationError],
        ) -> IO[Result[UnitResult, TranslationError]]:
            if isinstance(reply_result, Err):
                return io_result(reply_result)

            translated = reply_result.value
            problem = unit_output_problem(
                call.source_chunk, translated.text, ctx.settings
            )
            if problem is None:
                return io_result(
                    Ok(UnitResult(translated.text, True, translated.usage))
                )

            return repair(
                attempt_index + 1, translated.text, translated.usage, problem
            )

        return continue_after

    def assessed(
        reply_result: Result[Translated, TranslationError],
    ) -> IO[Result[UnitResult, TranslationError]]:
        if isinstance(reply_result, Err):
            return io_result(reply_result)

        return assess_initial(reply_result.value.text, reply_result.value.usage)

    return io_bind(
        translate_chunk(
            ctx,
            pass_definition,
            call.source_chunk,
            call.work_chunk,
            analysis,
            usage,
            call.context,
        ),
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

    def step(
        accumulator: UnitsSoFar, call: UnitCall
    ) -> IO[Result[UnitsSoFar, TranslationError]]:
        if not call.source_chunk:
            return io_map(
                verbose_log(
                    ctx,
                    "zh2en: [%s] unit %d/%d has no matching source; "
                    "passing it through unchanged"
                    % (pass_definition.name, call.index + 1, call.total),
                ),
                lambda _: Ok(
                    UnitsSoFar(
                        accumulator.outputs
                        + (call.work_chunk + call.trailing_separator,),
                        accumulator.usage,
                    )
                ),
            )

        def store(
            result: Result[UnitResult, TranslationError],
        ) -> IO[Result[UnitsSoFar, TranslationError]]:
            if isinstance(result, Err):
                return io_result(
                    fail_unit(pass_definition.name, call.index + 1, result.error)
                )

            outcome = result.value
            stored = io_map(
                io_when(
                    outcome.validated and ctx.use_cache,
                    cache_write(ctx.cache_directory, call.key, outcome.text),
                ),
                lambda _: verbose_log(
                    ctx,
                    "zh2en: [%s] unit %d/%d done"
                    % (pass_definition.name, call.index + 1, call.total),
                ),
            )
            return io_map(
                stored,
                lambda _: Ok(
                    UnitsSoFar(
                        accumulator.outputs
                        + (outcome.text + call.trailing_separator,),
                        outcome.usage,
                    )
                ),
            )

        def proceed(cached: str | None) -> IO[Result[UnitsSoFar, TranslationError]]:
            if cached is not None:
                return io_map(
                    verbose_log(
                        ctx,
                        "zh2en: [%s] unit %d/%d cache hit"
                        % (pass_definition.name, call.index + 1, call.total),
                    ),
                    lambda _: Ok(
                        UnitsSoFar(
                            accumulator.outputs
                            + (cached + call.trailing_separator,),
                            accumulator.usage,
                        )
                    ),
                )

            return io_bind(
                run_unit(ctx, pass_definition, call, analysis, accumulator.usage),
                store,
            )

        looked_up: IO[str | None] = (
            cache_read(ctx.cache_directory, call.key)
            if ctx.use_cache
            else io_pure(None)
        )
        return io_bind(looked_up, proceed)

    return fold_io(calls, step, Ok(UnitsSoFar((), usage)))


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
    plan = paragraph_plan if pass_definition.mode == "paragraph" else chunk_plan
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
                text=ensure_blank_line_separators(
                    "".join(units_result.value.outputs)
                ),
                analysis=state.analysis,
                usage=units_result.value.usage,
            )

            def after_stage(ended_at: float) -> IO[StateResult]:
                return io_bind(
                    log_stage(
                        ctx.console,
                        "Done",
                        started_at,
                        ended_at,
                        state.usage,
                        next_state.usage,
                    ),
                    lambda _: conclude_state(
                        ctx, pass_definition, next_state, source_paragraphs
                    ),
                )

            return io_bind(now(ctx.clock), after_stage)

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

    reported_warning = io_pure(None) if warning is None else ctx.console.log(warning)
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
        return "zh2en: [%s] output has %d paragraph(s), source has %d; %s" % (
            pass_definition.name,
            count,
            len(source_paragraphs),
            action,
        )

    def after_retry(result: StateResult) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if count == len(source_paragraphs):
            return io_result(result)

        return io_map(
            ctx.console.log(
                "zh2en: [%s] paragraph count still differs (%d vs %d); continuing"
                % (pass_definition.name, count, len(source_paragraphs))
            ),
            lambda _: result,
        )

    def check(result: StateResult) -> IO[StateResult]:
        if isinstance(result, Err):
            return io_result(result)

        count = count_paragraphs(result.value.text)
        if count == len(source_paragraphs):
            return io_result(result)

        if pass_definition.mode != "chunk":
            return io_map(
                ctx.console.log(mismatch_message(count, "continuing")),
                lambda _: result,
            )

        retried = io_bind(
            attempt(
                replace(pass_definition, mode="paragraph"),
                replace(state, usage=result.value.usage),
                ctx.clock(),
            ),
            after_retry,
        )
        return io_and_then(
            ctx.console.log(
                mismatch_message(
                    count, "re-running the pass with one call per paragraph"
                )
            ),
            retried,
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
    if pass_definition.mode == "analysis":
        return run_analysis_pass(
            ctx, pass_definition, state, started_at, source_paragraphs
        )

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
                    "Starting pass %d/%d [%s]..."
                    % (number, len(pass_definitions), pass_definition.name)
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
        total_elapsed = ctx.clock() - started

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
            verbose_log(ctx, "zh2en: done in %.1fs" % total_elapsed),
            after_done,
        )

    return io_bind(write_stdout(stdout, state.text), after_write)

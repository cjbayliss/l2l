from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import accumulate, chain
from typing import TextIO

from zh2en.config import (
    Context,
    PassDefinition,
    Settings,
    pass_salt,
    resolve_call_settings,
)
from zh2en.console import Console
from zh2en.effects import cache_read, cache_write, now, write_stdout
from zh2en.http import chat
from zh2en.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_bind,
    io_map,
    io_pure,
    io_result,
    result_bind_io,
    result_map,
)
from zh2en.text import (
    CACHE_SALT_VERSION,
    ChatOutcome,
    Usage,
    cache_key,
    count_paragraphs,
    drop_non_ascii,
    ensure_blank_line_separators,
    estimate_tokens,
    make_chunks,
    non_ascii_sample,
    regroup_by_plan,
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


StateResult = Result[State, str]
UnitOutcome = tuple[str, bool, Usage]
UnitAccumulator = tuple[tuple[str, ...], Usage]


def verbose_log(ctx: Context, message: str) -> IO[None]:
    return ctx.console.log(message) if ctx.verbose else io_pure(None)


def unit_output_problem(
    source_text: str, output: str, settings: Settings
) -> str | None:
    if not output.strip():
        return "the reply was empty"

    expected = count_paragraphs(source_text)
    found = count_paragraphs(output)
    if found != expected:
        return "the reply has %d paragraph(s) but the source has %d" % (
            found,
            expected,
        )

    source_estimate = estimate_tokens(source_text)
    output_estimate = estimate_tokens(output)
    if output_estimate > settings.unit_output_max_ratio * max(source_estimate, 1):
        return (
            "the reply is ~%d tokens against a source of ~%d tokens "
            "(limit %.0fx)"
            % (
                output_estimate,
                source_estimate,
                settings.unit_output_max_ratio,
            )
        )

    return None


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
    parts = [
        analysis.strip() if analysis else "",
        *context_parts(context),
        source_chunk,
    ]
    if work_chunk is not None and work_chunk != source_chunk:
        parts.append(work_chunk)
    return "\n\n".join(part for part in parts if part)


def plan_info_message(
    pass_definition: PassDefinition,
    work_paragraphs: tuple[str, ...],
    work_groups: tuple[tuple[str, ...], ...],
) -> str:
    if pass_definition.mode == "paragraph":
        return "zh2en: [%s] %d paragraph(s), one call per paragraph" % (
            pass_definition.name,
            len(work_groups),
        )

    return "zh2en: [%s] %d paragraph(s) in %d chunk(s)" % (
        pass_definition.name,
        len(work_paragraphs),
        len(work_groups),
    )


def resolve_work_groups(
    mode: str,
    work_paragraphs: tuple[str, ...],
    plan: tuple[tuple[str, ...], ...],
    pass_name: str,
    budget: int,
) -> tuple[tuple[tuple[str, ...], ...], str | None]:
    groups = regroup_by_plan(work_paragraphs, plan)
    if groups is not None:
        return groups, None

    warning = (
        "zh2en: [%s] paragraph count changed by a previous pass; "
        "grouping working text independently" % pass_name
    )
    if mode == "paragraph":
        return tuple((paragraph,) for paragraph in work_paragraphs), warning

    return make_chunks(work_paragraphs, budget), warning


def build_ascii_fix_user(source_paragraph: str, output_paragraph: str) -> str:
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Rewrite the translated paragraph as pure ASCII English."
        % (source_paragraph, output_paragraph)
    )


def build_ascii_retry_user(
    source_paragraph: str, output_paragraph: str, result: str
) -> str:
    return (
        "Source paragraph (original language):\n%s\n\n"
        "Translated paragraph (must become pure ASCII English):\n%s\n\n"
        "Your previous reply still contained these non-ASCII "
        "characters: %s. Rewrite the translated paragraph again, "
        "inferring English for every one of them from the source and "
        "context. Reply with ASCII characters only."
        % (source_paragraph, output_paragraph, non_ascii_sample(result))
    )


def build_unit_retry_user(
    source_chunk: str,
    context: tuple[str, ...],
    bad_output: str,
    problem: str,
) -> str:
    return "\n\n".join(
        part
        for part in (
            "Your previous reply below does not satisfy the output rules: %s."
            % problem,
            "Previous reply:\n%s" % bad_output,
            *context_parts(context),
            source_chunk,
            "Translate the source text again, fixing the problem; output only "
            "the translation.",
        )
        if part
    )


def run_analysis(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    usage: Usage,
) -> IO[Result[ChatOutcome, str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    return io_map(
        chat(ctx, pass_definition.instruction, text, model, params, usage),
        lambda result: result_map(result, lambda pair: (pair[0].strip(), pair[1])),
    )


def analyze_document(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[ChatOutcome, str]]:
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
            Err(
                "document requires analysis in %d parts, over the "
                "%d-token budget; raise api.max_tokens (--max-tokens / "
                "TRANSLATE_MAX_TOKENS) or shorten the input"
                % (len(chunks), ctx.config.max_tokens)
            )
        )

    return run_analysis(ctx, pass_definition, full_text, usage)


def run_analysis_once(
    ctx: Context, pass_definition: PassDefinition, full_text: str, usage: Usage
) -> IO[Result[ChatOutcome, str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    key = cache_key(full_text, model, pass_salt(pass_definition), overrides=params)

    def compute(current_usage: Usage) -> IO[Result[ChatOutcome, str]]:
        def store(
            result: Result[ChatOutcome, str],
        ) -> IO[Result[ChatOutcome, str]]:
            if isinstance(result, Err):
                return io_result(result)

            analysis, new_usage = result.value
            return io_map(
                (
                    cache_write(ctx.cache_directory, key, analysis)
                    if ctx.use_cache
                    else io_pure(None)
                ),
                lambda _: Ok((analysis, new_usage)),
            )

        return io_bind(
            analyze_document(ctx, pass_definition, full_text, current_usage), store
        )

    if not ctx.use_cache:
        return compute(usage)

    def use_cached(cached: str | None) -> IO[Result[ChatOutcome, str]]:
        if cached is None:
            return compute(usage)

        return io_map(
            verbose_log(ctx, "zh2en: [%s] cache hit" % pass_definition.name),
            lambda _: Ok((cached, usage)),
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
) -> IO[Result[ChatOutcome, str]]:
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
) -> IO[Result[ChatOutcome, str]]:
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
    ) -> IO[Result[ChatOutcome, str]]:
        if index > ctx.settings.ascii_fix_attempts:
            return io_result(Ok((last_result, current_usage)))

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
                result, lambda pair: ascii_outcome(index, pair[0], pair[1])
            ),
        )

    def ascii_outcome(
        index: int, result: str, current_usage: Usage
    ) -> IO[Result[ChatOutcome, str]]:
        if result.isascii():
            return io_map(
                (
                    cache_write(ctx.cache_directory, key, result)
                    if ctx.use_cache
                    else io_pure(None)
                ),
                lambda _: Ok((result, current_usage)),
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

    def start(current_usage: Usage) -> IO[Result[ChatOutcome, str]]:
        return attempt(
            1,
            build_ascii_fix_user(source_paragraph, output_paragraph),
            "",
            current_usage,
        )

    if not ctx.use_cache:
        return start(usage)

    def use_cached(cached: str | None) -> IO[Result[ChatOutcome, str]]:
        if cached is None or not cached.isascii():
            return start(usage)

        return io_map(
            verbose_log(ctx, "zh2en: ascii: cache hit"),
            lambda _: Ok((cached, usage)),
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
) -> IO[Result[ChatOutcome, str]]:
    mechanical = to_ascii_mechanical(paragraph, ctx.settings.ascii_character_map)
    if mechanical.isascii():
        return io_map(
            verbose_log(
                ctx,
                "zh2en: ascii: paragraph %d/%d converted mechanically"
                % (index + 1, total),
            ),
            lambda _: Ok((mechanical + separator, usage)),
        )

    def repaired(
        result: Result[ChatOutcome, str],
    ) -> IO[Result[ChatOutcome, str]]:
        if isinstance(result, Err):
            return io_result(result)

        final, warning = drop_non_ascii(
            result.value[0],
            ctx.settings.ascii_character_map,
            ctx.settings.ascii_fix_attempts,
            index,
        )

        def emit(_: None) -> Result[ChatOutcome, str]:
            return Ok((final + separator, result.value[1]))

        return (
            io_map(ctx.console.log(warning), emit) if warning else io_result(emit(None))
        )

    return io_bind(
        verbose_log(
            ctx,
            "zh2en: ascii: paragraph %d/%d still non-ASCII; asking the "
            "LLM to repair it" % (index + 1, total),
        ),
        lambda _: io_bind(
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
) -> IO[Result[ChatOutcome, str]]:
    paragraphs, separators = split_paragraphs(text)
    accumulator_type = tuple[tuple[str, ...], Usage]

    def step(
        accumulator: accumulator_type, indexed: tuple[int, str]
    ) -> IO[Result[accumulator_type, str]]:
        outputs, current_usage = accumulator
        index, paragraph = indexed
        separator = separators[index] if index < len(separators) else ""
        if paragraph.isascii():
            return io_result(Ok((outputs + (paragraph + separator,), current_usage)))

        return io_map(
            repair_paragraph(
                ctx,
                pass_definition,
                paragraph,
                separator,
                index,
                len(paragraphs),
                source_paragraphs,
                current_usage,
            ),
            lambda result: result_map(
                result, lambda pair: (outputs + (pair[0],), pair[1])
            ),
        )

    initial: Result[accumulator_type, str] = Ok(((), usage))
    return io_map(
        fold_io(enumerate(paragraphs), step, initial),
        lambda result: result_map(result, lambda pair: ("".join(pair[0]), pair[1])),
    )


def enforce_pass_ascii(
    ctx: Context,
    pass_definition: PassDefinition,
    text: str,
    source_paragraphs: tuple[str, ...],
    usage: Usage,
    started_at: float,
) -> IO[Result[ChatOutcome, str]]:
    def conclude(
        result: Result[ChatOutcome, str],
    ) -> IO[Result[ChatOutcome, str]]:
        if isinstance(result, Err):
            return io_map(
                ctx.console.interrupt(),
                lambda _: Err(
                    "zh2en: pass [%s] ascii enforcement failed: %s"
                    % (pass_definition.name, result.error)
                ),
            )

        fixed, new_usage = result.value
        prompt, completion, cost = usage_delta(usage, new_usage)

        def report(ended_at: float) -> IO[Result[ChatOutcome, str]]:
            return io_map(
                ctx.console.finish(
                    usage_line("Done", ended_at - started_at, prompt, completion, cost)
                    if prompt or completion or cost
                    else "Done."
                ),
                lambda _: Ok((fixed, new_usage)),
            )

        return io_bind(now(ctx.clock), report)

    return io_bind(
        io_bind(
            ctx.console.write_partial("Enforcing ASCII... "),
            lambda _: ensure_ascii_output(
                ctx, pass_definition, text, source_paragraphs, usage
            ),
        ),
        conclude,
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
                result, lambda pair: State(pair[0], state.analysis, pair[1])
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
        analysis_result: Result[ChatOutcome, str],
    ) -> IO[StateResult]:
        if isinstance(analysis_result, Err):
            return io_result(
                Err(
                    "zh2en: pass [%s] failed: %s"
                    % (pass_definition.name, analysis_result.error)
                )
            )

        analysis, analysis_usage = analysis_result.value

        def after_stage(ended_at: float) -> IO[StateResult]:
            return io_bind(
                log_stage(
                    ctx.console,
                    "Done",
                    started_at,
                    ended_at,
                    state.usage,
                    analysis_usage,
                ),
                lambda _: conclude_state(
                    ctx,
                    pass_definition,
                    State(text=state.text, analysis=analysis, usage=analysis_usage),
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


@dataclass(frozen=True)
class UnitCall:
    index: int
    total: int
    source_chunk: str
    work_chunk: str
    context: tuple[str, ...]
    key: str
    trailing_separator: str


def plan_unit_calls(
    ctx: Context,
    pass_definition: PassDefinition,
    plan: tuple[tuple[str, ...], ...],
    work_groups: tuple[tuple[str, ...], ...],
    trailing_separators: tuple[str, ...],
) -> tuple[UnitCall, ...]:
    model, params = resolve_call_settings(ctx.config, pass_definition)
    flat_plan = tuple(chain.from_iterable(plan))
    plan_starts = tuple(accumulate(map(len, plan), initial=0))

    def neighbour_context(index: int) -> tuple[str, ...]:
        if pass_definition.mode != "paragraph" or index >= len(plan):
            return ()

        start = plan_starts[index]
        end = plan_starts[index + 1]
        return (flat_plan[start - 1 : start] if start > 0 else ()) + (
            flat_plan[end : end + 1] if end < len(flat_plan) else ()
        )

    def call(index: int, work_group: tuple[str, ...]) -> UnitCall:
        source_chunk = "\n\n".join(plan[index]) if index < len(plan) else ""
        work_chunk = "\n\n".join(work_group)
        context = neighbour_context(index)
        return UnitCall(
            index=index,
            total=len(work_groups),
            source_chunk=source_chunk,
            work_chunk=work_chunk,
            context=context,
            key=cache_key(
                source_chunk,
                model,
                pass_salt(pass_definition),
                work_chunk,
                overrides=params,
                context="\n\n".join(context),
            ),
            trailing_separator=trailing_separators[index]
            if index < len(trailing_separators)
            else "",
        )

    return tuple(call(index, group) for index, group in enumerate(work_groups))


def run_unit(
    ctx: Context,
    pass_definition: PassDefinition,
    call: UnitCall,
    analysis: str | None,
    usage: Usage,
) -> IO[Result[UnitOutcome, str]]:
    model, params = resolve_call_settings(ctx.config, pass_definition)

    def assess_initial(
        translated: str, unit_usage: Usage
    ) -> IO[Result[UnitOutcome, str]]:
        problem = unit_output_problem(call.source_chunk, translated, ctx.settings)
        if problem is None:
            return io_result(Ok((translated, True, unit_usage)))

        return repair(1, translated, unit_usage, problem)

    def repair(
        attempt_index: int,
        bad_output: str,
        unit_usage: Usage,
        problem: str,
    ) -> IO[Result[UnitOutcome, str]]:
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
                lambda _: Ok((bad_output, False, unit_usage)),
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
    ) -> Callable[[Result[ChatOutcome, str]], IO[Result[UnitOutcome, str]]]:
        def continue_after(
            reply_result: Result[ChatOutcome, str],
        ) -> IO[Result[UnitOutcome, str]]:
            if isinstance(reply_result, Err):
                return io_result(reply_result)

            translated, new_usage = reply_result.value
            problem = unit_output_problem(call.source_chunk, translated, ctx.settings)
            if problem is None:
                return io_result(Ok((translated, True, new_usage)))

            return repair(attempt_index + 1, translated, new_usage, problem)

        return continue_after

    def assessed(
        reply_result: Result[ChatOutcome, str],
    ) -> IO[Result[UnitOutcome, str]]:
        if isinstance(reply_result, Err):
            return io_result(reply_result)

        translated, new_usage = reply_result.value
        return assess_initial(translated, new_usage)

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
) -> IO[Result[UnitAccumulator, str]]:
    calls = plan_unit_calls(
        ctx, pass_definition, plan, work_groups, trailing_separators
    )

    def step(
        accumulator: UnitAccumulator, call: UnitCall
    ) -> IO[Result[UnitAccumulator, str]]:
        outputs, current_usage = accumulator

        if not call.source_chunk:
            return io_map(
                verbose_log(
                    ctx,
                    "zh2en: [%s] unit %d/%d has no matching source; "
                    "passing it through unchanged"
                    % (pass_definition.name, call.index + 1, call.total),
                ),
                lambda _: Ok(
                    (
                        outputs + (call.work_chunk + call.trailing_separator,),
                        current_usage,
                    )
                ),
            )

        def store(
            result: Result[UnitOutcome, str],
        ) -> IO[Result[UnitAccumulator, str]]:
            if isinstance(result, Err):
                return io_result(
                    Err(
                        "zh2en: [%s] failed on unit %d: %s"
                        % (pass_definition.name, call.index + 1, result.error)
                    )
                )

            translated, valid, new_usage = result.value
            return io_map(
                io_bind(
                    (
                        cache_write(ctx.cache_directory, call.key, translated)
                        if valid and ctx.use_cache
                        else io_pure(None)
                    ),
                    lambda _: verbose_log(
                        ctx,
                        "zh2en: [%s] unit %d/%d done"
                        % (pass_definition.name, call.index + 1, call.total),
                    ),
                ),
                lambda _: Ok(
                    (outputs + (translated + call.trailing_separator,), new_usage)
                ),
            )

        if not ctx.use_cache:
            return io_bind(
                run_unit(ctx, pass_definition, call, analysis, current_usage), store
            )

        def with_cached(cached: str | None) -> IO[Result[UnitAccumulator, str]]:
            if cached is not None:
                return io_map(
                    verbose_log(
                        ctx,
                        "zh2en: [%s] unit %d/%d cache hit"
                        % (pass_definition.name, call.index + 1, call.total),
                    ),
                    lambda _: Ok(
                        (outputs + (cached + call.trailing_separator,), current_usage)
                    ),
                )

            return io_bind(
                run_unit(ctx, pass_definition, call, analysis, current_usage), store
            )

        return io_bind(cache_read(ctx.cache_directory, call.key), with_cached)

    return fold_io(calls, step, Ok(((), usage)))


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
            units_result: Result[UnitAccumulator, str],
        ) -> IO[StateResult]:
            if isinstance(units_result, Err):
                return io_result(units_result)

            next_state = State(
                text=ensure_blank_line_separators("".join(units_result.value[0])),
                analysis=state.analysis,
                usage=units_result.value[1],
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

        return io_bind(
            verbose_log(
                ctx, plan_info_message(pass_definition, work_paragraphs, work_groups)
            ),
            lambda _: io_bind(
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

    return io_bind(ctx.console.log(warning) if warning else io_pure(None), proceed)


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

        return io_bind(
            ctx.console.log(
                mismatch_message(
                    count, "re-running the pass with one call per paragraph"
                )
            ),
            lambda _: io_bind(
                attempt(
                    replace(pass_definition, mode="paragraph"),
                    replace(state, usage=result.value.usage),
                    ctx.clock(),
                ),
                after_retry,
            ),
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
            return io_bind(
                ctx.console.log(
                    "Starting pass %d/%d [%s]..."
                    % (number, len(pass_definitions), pass_definition.name)
                ),
                lambda _: run_pass(
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
            return io_map(ctx.console.log(result.error), lambda _: 1)

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
            (
                ctx.console.log("zh2en: done in %.1fs" % total_elapsed)
                if ctx.verbose
                else io_pure(None)
            ),
            after_done,
        )

    return io_bind(write_stdout(stdout, state.text), after_write)

from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, chain

from zh2en.errors import HttpError, TranslationError
from zh2en.monads import NOTHING, Just, Maybe
from zh2en.settings import (
    Context,
    PassDefinition,
    Settings,
    pass_salt,
    resolve_call_settings,
)
from zh2en.text import (
    AsciiDrop,
    cache_key,
    count_paragraphs,
    estimate_tokens,
    make_chunks,
    non_ascii_sample,
    regroup_by_plan,
    split_paragraphs,
    unit_separators,
)


def transient(error: TranslationError) -> bool:
    if isinstance(error, HttpError):
        if error.kind == "unreachable":
            return True

        if error.kind == "status" and error.status is not None:
            return error.status == 429 or error.status >= 500

    return False


def plan_backoff(base_delay: float, cap: float, attempts: int) -> tuple[float, ...]:
    def delay(index: int) -> float:
        exponential: float = base_delay * 2.0**index
        return min(exponential, cap)

    return tuple(delay(index) for index in range(max(attempts, 0)))


def retry_delay(error: TranslationError, planned: float) -> float:
    if isinstance(error, HttpError) and error.retry_after is not None:
        return max(planned, error.retry_after)

    return planned


def unit_output_problem(
    source_text: str, output: str, settings: Settings
) -> Maybe[str]:
    if not output.strip():
        return Just("the reply was empty")

    expected = count_paragraphs(source_text)
    found = count_paragraphs(output)
    if found != expected:
        return Just(
            "the reply has %d paragraph(s) but the source has %d" % (found, expected)
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
) -> tuple[tuple[tuple[str, ...], ...], Maybe[str]]:
    grouped = regroup_by_plan(work_paragraphs, plan)
    if isinstance(grouped, Just):
        return grouped.value, NOTHING

    warning: Maybe[str] = Just(
        f"zh2en: [{pass_name}] paragraph count changed by a previous pass; "
        "grouping working text independently"
    )
    if mode == "paragraph":
        return tuple((paragraph,) for paragraph in work_paragraphs), warning

    return make_chunks(work_paragraphs, budget), warning


def build_ascii_fix_user(source_paragraph: str, output_paragraph: str) -> str:
    return (
        f"Source paragraph (original language):\n{source_paragraph}\n\n"
        "Translated paragraph (must become pure ASCII English):\n"
        f"{output_paragraph}\n\n"
        "Rewrite the translated paragraph as pure ASCII English."
    )


def build_ascii_retry_user(
    source_paragraph: str, output_paragraph: str, result: str
) -> str:
    return (
        f"Source paragraph (original language):\n{source_paragraph}\n\n"
        "Translated paragraph (must become pure ASCII English):\n"
        f"{output_paragraph}\n\n"
        "Your previous reply still contained these non-ASCII "
        f"characters: {non_ascii_sample(result)}. Rewrite the translated "
        "paragraph again, inferring English for every one of them from the "
        "source and context. Reply with ASCII characters only."
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
            f"Your previous reply below does not satisfy the output rules: {problem}.",
            f"Previous reply:\n{bad_output}",
            *context_parts(context),
            source_chunk,
            "Translate the source text again, fixing the problem; output only "
            "the translation.",
        )
        if part
    )


def ascii_drop_warning(drop: AsciiDrop) -> str:
    return (
        "zh2en: ascii: warning: paragraph %d still contained "
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
            trailing_separator=(
                trailing_separators[index] if index < len(trailing_separators) else ""
            ),
        )

    return tuple(call(index, group) for index, group in enumerate(work_groups))


def plan_report(
    ctx: Context,
    pass_definitions: tuple[PassDefinition, ...],
    text: str,
) -> str:
    source_paragraphs, separators = split_paragraphs(text)
    chunk_plan = make_chunks(source_paragraphs, ctx.settings.chunk_budget_tokens)
    paragraph_plan = tuple((paragraph,) for paragraph in source_paragraphs)

    def report_pass(indexed: tuple[int, PassDefinition]) -> tuple[str, ...]:
        number, pass_definition = indexed
        header = "pass %d/%d [%s]" % (
            number,
            len(pass_definitions),
            pass_definition.name,
        )
        if pass_definition.mode == "analysis":
            return (f"{header}: mode=analysis, 1 call with the whole document",)

        plan = paragraph_plan if pass_definition.mode == "paragraph" else chunk_plan
        work_groups, warning = resolve_work_groups(
            pass_definition.mode,
            source_paragraphs,
            plan,
            pass_definition.name,
            ctx.settings.chunk_budget_tokens,
        )
        calls = plan_unit_calls(
            ctx,
            pass_definition,
            plan,
            work_groups,
            unit_separators(plan, separators),
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
            *((f"  warning: {warning.value}",) if isinstance(warning, Just) else ()),
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

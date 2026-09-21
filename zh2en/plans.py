from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate, chain

from zh2en.config import (
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
)


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
            trailing_separator=trailing_separators[index]
            if index < len(trailing_separators)
            else "",
        )

    return tuple(call(index, group) for index, group in enumerate(work_groups))

from __future__ import annotations

from functools import reduce

from l2l.cache import cached_translation
from l2l.effects import now
from l2l.errors import TranslationError, fail_ascii
from l2l.http import repaired_call
from l2l.messages import (
    ascii_cache_hit_message,
    ascii_llm_message,
    ascii_mechanical_message,
    ascii_retry_message,
    stage_done_line,
)
from l2l.monads import (
    IO,
    NOTHING,
    Err,
    Just,
    Maybe,
    Ok,
    Result,
    io_and_then,
    io_bind,
    io_map,
    io_result,
    io_traverse,
    result_map,
)
from l2l.plans import (
    ascii_drop_warning,
    build_ascii_fix_user,
    build_ascii_retry_user,
    verbose_log,
)
from l2l.settings import Context, PassDefinition, resolve_call_settings, salt
from l2l.text import (
    CACHE_SALT_VERSION,
    Translated,
    Usage,
    cache_key,
    drop_non_ascii,
    split_paragraphs,
    to_ascii_mechanical,
    usage_add,
    usage_delta,
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
        salt(
            CACHE_SALT_VERSION,
            "ascii-fix",
            pass_definition.name,
            ctx.settings.ascii_fix_instruction,
        ),
        output_paragraph,
        overrides=params,
    )

    def validate(result: str) -> Maybe[str]:
        return NOTHING if result.isascii() else Just("reply was not pure ASCII")

    def on_repair(failed: int, _problem: str) -> IO[None]:
        return verbose_log(
            ctx, ascii_retry_message(failed, ctx.settings.ascii_fix_attempts)
        )

    def start() -> IO[Result[Translated, TranslationError]]:
        call = repaired_call(
            ctx,
            model,
            params,
            ctx.settings.ascii_fix_instruction,
            build_ascii_fix_user(source_paragraph, output_paragraph),
            validate,
            lambda bad_output, _problem: build_ascii_retry_user(
                source_paragraph, output_paragraph, bad_output
            ),
            ctx.settings.ascii_fix_attempts - 1,
            on_repair,
            None,
        )
        return io_map(
            call(usage),
            lambda outcome: result_map(
                outcome, lambda repaired: Translated(repaired[0], repaired[1])
            ),
        )

    return cached_translation(
        ctx,
        key,
        usage,
        compute=start,
        hit_log=verbose_log(ctx, ascii_cache_hit_message()),
        acceptable=str.isascii,
    )


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
            verbose_log(ctx, ascii_mechanical_message(index, total)),
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
            io_map(ctx.console.log(ascii_drop_warning(drop.value)), emit)
            if isinstance(drop, Just)
            else io_result(emit(None))
        )

    return io_and_then(
        verbose_log(ctx, ascii_llm_message(index, total)),
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

    def paragraph_part(
        indexed: tuple[int, str],
    ) -> IO[Result[tuple[str, Usage], TranslationError]]:
        index, paragraph = indexed
        separator = separators[index] if index < len(separators) else ""
        if paragraph.isascii():
            return io_result(Ok((paragraph + separator, Usage())))

        return io_map(
            repair_paragraph(
                ctx,
                pass_definition,
                paragraph,
                separator,
                index,
                len(paragraphs),
                source_paragraphs,
                usage,
            ),
            lambda result: result_map(
                result,
                lambda translated: (
                    translated.text,
                    Usage(*usage_delta(usage, translated.usage)),
                ),
            ),
        )

    def collect(parts: tuple[tuple[str, Usage], ...]) -> Translated:
        return Translated(
            "".join(text for text, _ in parts),
            reduce(usage_add, (delta for _, delta in parts), usage),
        )

    return io_map(
        io_traverse(enumerate(paragraphs), paragraph_part),
        lambda result: result_map(result, collect),
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
                    stage_done_line(ended_at - started_at, prompt, completion, cost)
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

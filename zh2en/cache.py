from __future__ import annotations

from collections.abc import Callable

from zh2en.effects import cache_read, cache_write
from zh2en.errors import TranslationError
from zh2en.monads import (
    IO,
    NOTHING,
    Just,
    Maybe,
    Ok,
    Result,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_when,
)
from zh2en.settings import Context
from zh2en.text import Translated, Usage


def const_acceptable(_: str) -> bool:
    return True


def cache_lookup(ctx: Context, key: str) -> IO[Maybe[str]]:
    if not ctx.use_cache:
        return io_pure(NOTHING)

    return cache_read(ctx.cache_directory, key)


def cache_store(ctx: Context, key: str, value: str, condition: bool) -> IO[None]:
    return io_when(
        ctx.use_cache and condition, cache_write(ctx.cache_directory, key, value)
    )


def store_translation(
    ctx: Context,
    key: str,
    result: Result[Translated, TranslationError],
    acceptable: Callable[[str], bool],
) -> IO[Result[Translated, TranslationError]]:
    if isinstance(result, Ok) and acceptable(result.value.text):
        return io_map(cache_store(ctx, key, result.value.text, True), lambda _: result)

    return io_result(result)


def cached_translation(
    ctx: Context,
    key: str,
    usage: Usage,
    compute: Callable[[], IO[Result[Translated, TranslationError]]],
    hit_log: IO[None],
    acceptable: Callable[[str], bool] = const_acceptable,
) -> IO[Result[Translated, TranslationError]]:
    def compute_and_store() -> IO[Result[Translated, TranslationError]]:
        return io_bind(
            compute(),
            lambda result: store_translation(ctx, key, result, acceptable),
        )

    def use_cached(cached: Maybe[str]) -> IO[Result[Translated, TranslationError]]:
        if isinstance(cached, Just) and acceptable(cached.value):
            return io_map(hit_log, lambda _: Ok(Translated(cached.value, usage)))

        return compute_and_store()

    return io_bind(cache_lookup(ctx, key), use_cached)

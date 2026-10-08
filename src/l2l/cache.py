from __future__ import annotations

from collections.abc import Callable

from l2l.effects import cache_read, cache_write
from l2l.errors import TranslationError
from l2l.monads import (
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
    io_when_unit,
)
from l2l.settings import Context
from l2l.text import Translated, Usage


def always_acceptable(_: str) -> bool:
    return True


def non_empty(text: str) -> bool:
    return bool(text.strip())


def cache_lookup(
    run_context: Context,
    key: str,
    acceptable: Callable[[str], bool] = always_acceptable,
) -> IO[Maybe[str]]:
    def checked(cached: Maybe[str]) -> Maybe[str]:
        return (
            cached if isinstance(cached, Just) and acceptable(cached.value) else NOTHING
        )

    if not run_context.use_cache:
        return io_pure(NOTHING)

    return io_map(cache_read(run_context.cache_directory, key), checked)


def discard_cache_write_outcome(_stored: bool) -> None:
    return None


def cache_store(
    run_context: Context, key: str, value: str, condition: bool
) -> IO[None]:
    return io_when_unit(
        run_context.use_cache and condition,
        io_map(
            cache_write(run_context.cache_directory, key, value),
            discard_cache_write_outcome,
        ),
    )


def store_translation(
    run_context: Context,
    key: str,
    result: Result[Translated, TranslationError],
    acceptable: Callable[[str], bool],
) -> IO[Result[Translated, TranslationError]]:
    if isinstance(result, Ok) and acceptable(result.value.text):
        return io_map(
            cache_store(run_context, key, result.value.text, True), lambda _: result
        )

    return io_result(result)


def cached_translation(
    run_context: Context,
    key: str,
    usage: Usage,
    compute: Callable[[], IO[Result[Translated, TranslationError]]],
    hit_log: IO[None],
    acceptable: Callable[[str], bool] = always_acceptable,
) -> IO[Result[Translated, TranslationError]]:
    def compute_and_store() -> IO[Result[Translated, TranslationError]]:
        return io_bind(
            compute(),
            lambda result: store_translation(run_context, key, result, acceptable),
        )

    def use_cached(cached: Maybe[str]) -> IO[Result[Translated, TranslationError]]:
        if isinstance(cached, Just):
            return io_map(hit_log, lambda _: Ok(Translated(cached.value, usage)))

        return compute_and_store()

    return io_bind(cache_lookup(run_context, key, acceptable), use_cached)

import io
from collections.abc import Callable
from pathlib import Path

from fakes import make_console, make_context

from zh2en.cache import (
    cache_lookup,
    cache_store,
    cached_translation,
    const_acceptable,
)
from zh2en.errors import TranslationError, fail_http
from zh2en.monads import IO, NOTHING, Err, Just, Ok, Result, io_pure
from zh2en.settings import Context
from zh2en.text import Translated, Usage, cache_key


def make_cached_context(
    tmp_path: Path, use_cache: bool = True, verbose: bool = True
) -> tuple[Context, io.StringIO]:
    console, stream = make_console()

    def never_open(request: object, timeout: float) -> Result[object, TranslationError]:
        raise AssertionError("endpoint must not be contacted")

    ctx = make_context(
        console,
        never_open,
        cache_directory=str(tmp_path),
        use_cache=use_cache,
        verbose=verbose,
    )
    return ctx, stream


def counting_compute(
    results: list[Translated],
) -> tuple[Callable[[], IO[Result[Translated, TranslationError]]], list[int]]:
    calls: list[int] = []

    def compute() -> IO[Result[Translated, TranslationError]]:
        def thunk() -> Result[Translated, TranslationError]:
            calls.append(len(calls))
            return Ok(results[min(len(calls), len(results)) - 1])

        return IO(thunk)

    return compute, calls


def test_cached_translation_computes_and_stores(tmp_path: Path) -> None:
    ctx, _ = make_cached_context(tmp_path)
    compute, calls = counting_compute([Translated("output", Usage(3, 4, 0.5))])
    key = cache_key("source", "model-x")

    result = cached_translation(ctx, key, Usage(), compute, io_pure(None)).run()

    assert result == Ok(Translated("output", Usage(3, 4, 0.5)))
    assert calls == [0]
    assert cache_lookup(ctx, key).run() == Just("output")


def test_cached_translation_hits_cache_without_computing(tmp_path: Path) -> None:
    ctx, stream = make_cached_context(tmp_path)
    compute, calls = counting_compute([Translated("fresh", Usage())])
    key = cache_key("source", "model-x")
    cache_store(ctx, key, "cached text", True).run()

    result = cached_translation(
        ctx, key, Usage(9, 9, 9.0), compute, ctx.console.log("cache hit")
    ).run()

    assert result == Ok(Translated("cached text", Usage(9, 9, 9.0)))
    assert calls == []
    assert "cache hit" in stream.getvalue()


def test_cached_translation_hit_log_only_when_verbose(tmp_path: Path) -> None:
    ctx, stream = make_cached_context(tmp_path, verbose=False)
    compute, calls = counting_compute([Translated("fresh", Usage())])
    key = cache_key("source", "model-x")
    cache_store(ctx, key, "cached text", True).run()

    result = cached_translation(ctx, key, Usage(), compute, io_pure(None)).run()

    assert result == Ok(Translated("cached text", Usage()))
    assert calls == []
    assert stream.getvalue() == ""


def test_cached_translation_skips_unacceptable_cached_value(tmp_path: Path) -> None:
    ctx, _ = make_cached_context(tmp_path)
    compute, calls = counting_compute([Translated("fresh ascii", Usage(1, 1, 0.0))])
    key = cache_key("source", "model-x")
    cache_store(ctx, key, "non-ascii 中", True).run()

    result = cached_translation(
        ctx, key, Usage(), compute, io_pure(None), acceptable=str.isascii
    ).run()

    assert result == Ok(Translated("fresh ascii", Usage(1, 1, 0.0)))
    assert calls == [0]
    assert cache_lookup(ctx, key).run() == Just("fresh ascii")


def test_cached_translation_does_not_store_unacceptable_result(tmp_path: Path) -> None:
    ctx, _ = make_cached_context(tmp_path)
    compute, calls = counting_compute([Translated("still 非 ascii", Usage(1, 1, 0.0))])
    key = cache_key("source", "model-x")

    result = cached_translation(
        ctx, key, Usage(), compute, io_pure(None), acceptable=str.isascii
    ).run()

    assert result == Ok(Translated("still 非 ascii", Usage(1, 1, 0.0)))
    assert calls == [0]
    assert cache_lookup(ctx, key).run() == NOTHING


def test_cached_translation_error_is_returned_and_not_stored(tmp_path: Path) -> None:
    ctx, _ = make_cached_context(tmp_path)
    failure: Result[Translated, TranslationError] = fail_http("unreachable", "down")

    def compute() -> IO[Result[Translated, TranslationError]]:
        return IO(lambda: failure)

    key = cache_key("source", "model-x")
    result = cached_translation(ctx, key, Usage(), compute, io_pure(None)).run()

    assert isinstance(result, Err)
    assert cache_lookup(ctx, key).run() == NOTHING


def test_cache_lookup_and_store_respect_use_cache(tmp_path: Path) -> None:
    ctx, _ = make_cached_context(tmp_path, use_cache=False)
    key = cache_key("source", "model-x")

    assert cache_lookup(ctx, key).run() == NOTHING
    cache_store(ctx, key, "value", True).run()
    assert cache_lookup(ctx, key).run() == NOTHING

    storing_ctx, _ = make_cached_context(tmp_path, use_cache=True)
    cache_store(storing_ctx, key, "value", True).run()
    assert cache_lookup(storing_ctx, key).run() == Just("value")


def test_cache_store_condition_gates_write(tmp_path: Path) -> None:
    ctx, _ = make_cached_context(tmp_path)
    key = cache_key("source", "model-x")

    cache_store(ctx, key, "value", False).run()
    assert cache_lookup(ctx, key).run() == NOTHING


def test_const_acceptable_accepts_everything() -> None:
    assert const_acceptable("")
    assert const_acceptable("anything")

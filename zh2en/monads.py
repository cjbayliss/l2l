from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")
E = TypeVar("E")
R = TypeVar("R")
A = TypeVar("A")
S = TypeVar("S")


@dataclass(frozen=True)
class Ok(Generic[T]):
    value: T


@dataclass(frozen=True)
class Err(Generic[E]):
    error: E


type Result[T, E] = Ok[T] | Err[E]


def result_map(result: Result[T, E], fn: Callable[[T], R]) -> Result[R, E]:
    return Ok(fn(result.value)) if isinstance(result, Ok) else result


def result_bind(result: Result[T, E], fn: Callable[[T], Result[R, E]]) -> Result[R, E]:
    return fn(result.value) if isinstance(result, Ok) else result


def result_bind_io(
    result: Result[T, E], fn: Callable[[T], IO[Result[R, E]]]
) -> IO[Result[R, E]]:
    if isinstance(result, Err):
        return io_result(result)

    return fn(result.value)


def result_map_error(result: Result[T, E], fn: Callable[[E], R]) -> Result[T, R]:
    return result if isinstance(result, Ok) else Err(fn(result.error))


def result_either(
    result: Result[T, E], on_ok: Callable[[T], R], on_err: Callable[[E], R]
) -> R:
    return on_ok(result.value) if isinstance(result, Ok) else on_err(result.error)


def result_or_else(result: Result[T, E], fallback: Callable[[], T]) -> T:
    return result.value if isinstance(result, Ok) else fallback()


def result_map2(
    first: Result[T, E], second: Result[R, E], fn: Callable[[T, R], A]
) -> Result[A, E]:
    if isinstance(first, Err):
        return first

    if isinstance(second, Err):
        return Err(second.error)

    return Ok(fn(first.value, second.value))


def result_zip(first: Result[T, E], second: Result[R, E]) -> Result[tuple[T, R], E]:
    return result_map2(first, second, lambda left, right: (left, right))


def fold_while(
    items: Iterable[S],
    step: Callable[[A, S], Result[A, E]],
    initial: Result[A, E],
) -> Result[A, E]:
    outcome = initial
    iterator = iter(items)
    while not isinstance(outcome, Err):
        try:
            item = next(iterator)
        except StopIteration:
            return outcome

        outcome = step(outcome.value, item)

    return outcome


def results_sequence(results: Iterable[Result[T, E]]) -> Result[tuple[T, ...], E]:
    def step(
        values: tuple[T, ...], result: Result[T, E]
    ) -> Result[tuple[T, ...], E]:
        return result_map(result, lambda value: values + (value,))

    return fold_while(results, step, Ok(()))


@dataclass(frozen=True)
class Just(Generic[T]):
    value: T


@dataclass(frozen=True)
class Nothing:
    pass


type Maybe[T] = Just[T] | Nothing


NOTHING: Nothing = Nothing()


def maybe_map(maybe: Maybe[T], fn: Callable[[T], R]) -> Maybe[R]:
    return Just(fn(maybe.value)) if isinstance(maybe, Just) else maybe


def maybe_bind(maybe: Maybe[T], fn: Callable[[T], Maybe[R]]) -> Maybe[R]:
    return fn(maybe.value) if isinstance(maybe, Just) else maybe


def maybe_either(
    maybe: Maybe[T], on_just: Callable[[T], R], on_nothing: Callable[[], R]
) -> R:
    return on_just(maybe.value) if isinstance(maybe, Just) else on_nothing()


def maybe_or_else(maybe: Maybe[T], fallback: T) -> T:
    return maybe.value if isinstance(maybe, Just) else fallback


def maybe_or_else_get(maybe: Maybe[T], fallback: Callable[[], T]) -> T:
    return maybe.value if isinstance(maybe, Just) else fallback()


def maybe_from_optional(value: T | None) -> Maybe[T]:
    return Just(value) if value is not None else NOTHING


def maybe_to_optional(maybe: Maybe[T]) -> T | None:
    return maybe.value if isinstance(maybe, Just) else None


def maybe_to_result(maybe: Maybe[T], if_nothing: Callable[[], E]) -> Result[T, E]:
    return Ok(maybe.value) if isinstance(maybe, Just) else Err(if_nothing())


def maybe_zip(first: Maybe[T], second: Maybe[R]) -> Maybe[tuple[T, R]]:
    return (
        Just((first.value, second.value))
        if isinstance(first, Just) and isinstance(second, Just)
        else NOTHING
    )


def fold_maybe(
    items: Iterable[S],
    step: Callable[[A, S], Maybe[A]],
    initial: Maybe[A],
) -> Maybe[A]:
    outcome = initial
    iterator = iter(items)
    while isinstance(outcome, Just):
        try:
            item = next(iterator)
        except StopIteration:
            return outcome

        outcome = step(outcome.value, item)

    return outcome


def maybes_sequence(maybes: Iterable[Maybe[T]]) -> Maybe[tuple[T, ...]]:
    def step(
        collected: tuple[T, ...], maybe: Maybe[T]
    ) -> Maybe[tuple[T, ...]]:
        return maybe_map(maybe, lambda value: collected + (value,))

    return fold_maybe(maybes, step, Just(()))


@dataclass(frozen=True)
class IO(Generic[T]):
    run: Callable[[], T]


type IOResult[T, E] = IO[Result[T, E]]


def io_pure(value: T) -> IO[T]:
    return IO(lambda: value)


def io_result(value: Result[T, E]) -> IO[Result[T, E]]:
    return IO(lambda: value)


def io_map(io_value: IO[T], fn: Callable[[T], R]) -> IO[R]:
    return IO(lambda: fn(io_value.run()))


def io_bind(io_value: IO[T], fn: Callable[[T], IO[R]]) -> IO[R]:
    return IO(lambda: fn(io_value.run()).run())


def io_and_then(io_value: IO[T], next_value: IO[R]) -> IO[R]:
    def thunk() -> R:
        io_value.run()
        return next_value.run()

    return IO(thunk)


def io_when(condition: bool, io_value: IO[T]) -> IO[T | None]:
    def thunk() -> T | None:
        return io_value.run() if condition else None

    return IO(thunk)


def io_unless(condition: bool, io_value: IO[T]) -> IO[T | None]:
    return io_when(not condition, io_value)


def io_catch(
    io_value: IO[T], handler: Callable[[Exception], Result[T, E]]
) -> IO[Result[T, E]]:
    def thunk() -> Result[T, E]:
        try:
            return Ok(io_value.run())
        except Exception as error:
            return handler(error)

    return IO(thunk)


def io_sequence(io_values: Iterable[IO[T]]) -> IO[tuple[T, ...]]:
    return IO(lambda: tuple(io_value.run() for io_value in io_values))


def io_pair(first: IO[T], second: IO[R]) -> IO[tuple[T, R]]:
    def thunk() -> tuple[T, R]:
        return (first.run(), second.run())

    return IO(thunk)


def fold_io(
    items: Iterable[S],
    step: Callable[[A, S], IO[Result[A, E]]],
    initial: Result[A, E],
) -> IO[Result[A, E]]:
    def thunk() -> Result[A, E]:
        outcome = initial
        for item in tuple(items):
            if isinstance(outcome, Err):
                return outcome

            outcome = step(outcome.value, item).run()

        return outcome

    return IO(thunk)


def io_traverse(
    items: Iterable[S], fn: Callable[[S], IO[Result[T, E]]]
) -> IO[Result[tuple[T, ...], E]]:
    def step(
        collected: tuple[T, ...], item: S
    ) -> IO[Result[tuple[T, ...], E]]:
        return io_map(
            fn(item),
            lambda outcome: result_map(outcome, lambda value: collected + (value,)),
        )

    return fold_io(items, step, Ok(()))


def io_result_map(io_value: IOResult[T, E], fn: Callable[[T], R]) -> IOResult[R, E]:
    return io_map(io_value, lambda outcome: result_map(outcome, fn))


def io_result_bind(
    io_value: IOResult[T, E], fn: Callable[[T], IOResult[R, E]]
) -> IOResult[R, E]:
    return io_bind(io_value, lambda outcome: result_bind_io(outcome, fn))


def io_memoize(io_value: IO[T]) -> IO[T]:
    lock = threading.Lock()
    cached: Maybe[T] = NOTHING

    def thunk() -> T:
        nonlocal cached
        with lock:
            if isinstance(cached, Nothing):
                cached = Just(io_value.run())

            return cached.value

    return IO(thunk)


@dataclass
class Ref(Generic[T]):
    value: T
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


def new_ref(value: T) -> IO[Ref[T]]:
    return IO(lambda: Ref(value))


def read_ref(reference: Ref[T]) -> IO[T]:
    return IO(lambda: reference.value)


def write_ref(reference: Ref[T], value: T) -> IO[T]:
    def thunk() -> T:
        with reference.lock:
            reference.value = value
            return value

    return IO(thunk)


def modify_ref(reference: Ref[T], fn: Callable[[T], T]) -> IO[T]:
    def thunk() -> T:
        with reference.lock:
            updated = fn(reference.value)
            reference.value = updated
            return updated

    return IO(thunk)

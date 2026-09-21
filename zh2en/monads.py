from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
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

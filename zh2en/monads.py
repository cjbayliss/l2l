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


Result = Ok[T] | Err[E]


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


def io_pure(value: T) -> IO[T]:
    return IO(lambda: value)


def io_result(value: Result[T, E]) -> IO[Result[T, E]]:
    return IO(lambda: value)


def io_map(io_value: IO[T], fn: Callable[[T], R]) -> IO[R]:
    return IO(lambda: fn(io_value.run()))


def io_bind(io_value: IO[T], fn: Callable[[T], IO[R]]) -> IO[R]:
    return IO(lambda: fn(io_value.run()).run())


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

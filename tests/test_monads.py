import threading
from collections.abc import Iterator

from zh2en.monads import (
    IO,
    NOTHING,
    Err,
    Just,
    Ok,
    Result,
    fold_io_lazy,
    io_and_then,
    io_atomic,
    io_bind,
    io_catch,
    io_map,
    io_pair,
    io_pure,
    io_result,
    io_result_bind,
    io_result_map,
    io_traverse,
    io_unless,
    io_using,
    io_when,
    io_when_unit,
    repeat_until,
    result_bind,
    result_either,
    result_map,
    result_map2,
    result_map_error,
    result_or_else,
    result_zip,
)


def test_result_map_transforms_ok_only() -> None:
    assert result_map(Ok(2), lambda value: value + 1) == Ok(3)
    assert result_map(Err("boom"), lambda value: value + 1) == Err("boom")


def test_result_bind_short_circuits() -> None:
    assert result_bind(Ok(2), lambda value: Ok(value * 3)) == Ok(6)
    assert result_bind(Err("boom"), lambda value: Ok(value * 3)) == Err("boom")


def test_result_map_error_transforms_err_only() -> None:
    assert result_map_error(Ok(2), str.upper) == Ok(2)
    assert result_map_error(Err("boom"), str.upper) == Err("BOOM")


def test_result_either_folds_both_channels() -> None:
    assert result_either(Ok(2), str, len) == "2"
    assert result_either(Err("boom"), str, len) == 4


def test_result_or_else_unwraps_or_falls_back() -> None:
    assert result_or_else(Ok(2), lambda: 0) == 2
    assert result_or_else(Err("boom"), lambda: 0) == 0


def test_result_map2_combines_two_oks() -> None:
    assert result_map2(Ok(2), Ok(3), lambda left, right: left + right) == Ok(5)


def test_result_map2_left_error_wins() -> None:
    assert result_map2(
        Err("left"), Err("right"), lambda left, right: left + right
    ) == Err("left")


def test_result_map2_right_error_propagates() -> None:
    assert result_map2(Ok(2), Err("boom"), lambda left, right: left + right) == Err(
        "boom"
    )


def test_result_zip_pairs_values() -> None:
    assert result_zip(Ok(1), Ok("a")) == Ok((1, "a"))
    assert result_zip(Err("boom"), Ok("a")) == Err("boom")
    assert result_zip(Ok(1), Err("boom")) == Err("boom")


def test_io_pure_map_bind_compose() -> None:
    program = io_map(io_bind(io_pure(2), lambda value: io_pure(value * 3)), str)
    assert program.run() == "6"


def test_io_and_then_runs_in_order_and_keeps_second() -> None:
    log: list[str] = []

    def record(label: str, value: str) -> IO[str]:
        def thunk() -> str:
            log.append(label)
            return value

        return IO(thunk)

    program = io_and_then(
        record("first", "a"),
        io_map(record("second", "b"), str.upper),
    )
    assert program.run() == "B"
    assert log == ["first", "second"]


def test_io_when_and_unless_gate_execution() -> None:
    ran: IO[str] = io_pure("ran")
    assert io_when(True, ran).run() == Just("ran")
    assert io_when(False, ran).run() is NOTHING
    assert io_unless(False, ran).run() == Just("ran")
    assert io_unless(True, ran).run() is NOTHING


def test_io_when_unit_gates_execution() -> None:
    log: list[str] = []

    def action() -> IO[None]:
        def thunk() -> None:
            log.append("ran")

        return IO(thunk)

    assert io_when_unit(True, action()).run() is None
    assert log == ["ran"]
    assert io_when_unit(False, action()).run() is None
    assert log == ["ran"]


def test_io_atomic_holds_lock_while_running() -> None:
    lock = threading.Lock()
    log: list[str] = []

    def action() -> IO[str]:
        def thunk() -> str:
            log.append("inside")
            return "value"

        return IO(thunk)

    assert io_atomic(lock, action()).run() == "value"
    assert log == ["inside"]


def test_io_using_enters_and_exits_resource() -> None:
    log: list[str] = []

    class Resource:
        def __enter__(self) -> str:
            log.append("enter")
            return "resource"

        def __exit__(self, *args: object) -> None:
            log.append("exit")

    program = io_using(Resource(), lambda entered: io_pure(len(entered)))
    assert program.run() == 8
    assert log == ["enter", "exit"]


def test_repeat_until_runs_until_event_set() -> None:
    until = threading.Event()
    log: list[str] = []

    def action() -> IO[None]:
        def thunk() -> None:
            log.append("tick")
            until.set()

        return IO(thunk)

    repeat_until(action(), until, 0.001)()
    assert log == ["tick"]


def test_io_catch_converts_exceptions_to_result() -> None:
    def boom() -> str:
        raise ValueError("bad")

    def handler(error: Exception) -> Result[str, str]:
        return Err(str(error))

    assert io_catch(IO(boom), handler).run() == Err("bad")
    assert io_catch(io_pure("fine"), handler).run() == Ok("fine")


def test_io_traverse_collects_successes() -> None:
    def unit(number: int) -> IO[Result[tuple[int, str], str]]:
        return io_result(Ok((number, str(number))))

    program: IO[Result[tuple[tuple[int, str], ...], str]] = io_traverse((1, 2), unit)
    assert program.run() == Ok(((1, "1"), (2, "2")))


def test_io_traverse_short_circuits_on_error() -> None:
    def step(number: int) -> IO[Result[int, str]]:
        outcome: Result[int, str] = (
            Err("bad %d" % number) if number == 2 else Ok(number)
        )
        return io_result(outcome)

    program: IO[Result[tuple[int, ...], str]] = io_traverse((1, 2, 3), step)
    assert program.run() == Err("bad 2")


def test_io_pair_runs_both_and_pairs_results() -> None:
    log: list[str] = []

    def record(label: str, value: int) -> IO[int]:
        def thunk() -> int:
            log.append(label)
            return value

        return IO(thunk)

    program = io_pair(record("first", 1), record("second", 2))
    assert program.run() == (1, 2)
    assert log == ["first", "second"]


def test_io_result_map_lifts_over_io() -> None:
    assert io_result_map(io_result(Ok(2)), lambda value: value + 1).run() == Ok(3)
    assert io_result_map(io_result(Err("e")), lambda value: value + 1).run() == Err("e")


def test_io_result_bind_chains_io_results() -> None:
    def step(value: int) -> IO[Result[int, str]]:
        return io_result(Ok(value + 1))

    assert io_result_bind(io_result(Ok(2)), step).run() == Ok(3)
    assert io_result_bind(io_result(Err("e")), step).run() == Err("e")


def test_fold_io_lazy_pulls_items_lazily_and_stops_on_error() -> None:
    consumed: list[int] = []

    def items() -> Iterator[int]:
        for number in range(5):
            consumed.append(number)
            yield number

    def step(total: int, number: int) -> IO[Result[int, str]]:
        if number == 2:
            return io_result(Err("stopped at 2"))

        return io_result(Ok(total + number))

    program = fold_io_lazy(items(), step, Ok(0))
    assert program.run() == Err("stopped at 2")
    assert consumed == [0, 1, 2]


def test_fold_io_lazy_consumes_everything_on_success() -> None:
    def step(total: int, number: int) -> IO[Result[int, str]]:
        return io_result(Ok(total + number))

    program = fold_io_lazy((1, 2, 3), step, Ok(0))
    assert program.run() == Ok(6)


def test_fold_io_lazy_empty_items_returns_initial() -> None:
    program: IO[Result[int, str]] = fold_io_lazy(
        (), lambda total, value: io_result(Ok(total)), Ok(7)
    )
    assert program.run() == Ok(7)

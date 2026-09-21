from zh2en.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_and_then,
    io_bind,
    io_catch,
    io_map,
    io_pure,
    io_result,
    io_result_bind,
    io_result_map,
    io_traverse,
    io_unless,
    io_when,
    result_bind,
    result_either,
    result_map,
    result_map_error,
    result_or_else,
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
    assert io_when(True, ran).run() == "ran"
    assert io_when(False, ran).run() is None
    assert io_unless(False, ran).run() == "ran"
    assert io_unless(True, ran).run() is None


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

    program: IO[Result[tuple[tuple[int, str], ...], str]] = io_traverse(
        (1, 2), unit
    )
    assert program.run() == Ok(((1, "1"), (2, "2")))


def test_io_traverse_short_circuits_on_error() -> None:
    def step(number: int) -> IO[Result[int, str]]:
        outcome: Result[int, str] = (
            Err("bad %d" % number) if number == 2 else Ok(number)
        )
        return io_result(outcome)

    program: IO[Result[tuple[int, ...], str]] = io_traverse((1, 2, 3), step)
    assert program.run() == Err("bad 2")


def test_io_result_map_lifts_over_io() -> None:
    assert io_result_map(io_result(Ok(2)), lambda value: value + 1).run() == Ok(3)
    assert io_result_map(io_result(Err("e")), lambda value: value + 1).run() == Err("e")


def test_io_result_bind_chains_io_results() -> None:
    def step(value: int) -> IO[Result[int, str]]:
        return io_result(Ok(value + 1))

    assert io_result_bind(io_result(Ok(2)), step).run() == Ok(3)
    assert io_result_bind(io_result(Err("e")), step).run() == Err("e")

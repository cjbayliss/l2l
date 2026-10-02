import threading

from l2l.monads import (
    IO,
    NOTHING,
    Cons,
    Err,
    Just,
    Ok,
    Ref,
    Result,
    cons,
    cons_all,
    cons_to_tuple,
    fold_io_push,
    io_and_then,
    io_atomic,
    io_bind,
    io_catch,
    io_catch_result,
    io_map,
    io_pair,
    io_pure,
    io_result,
    io_result_bind,
    io_result_map,
    io_traverse,
    io_unless,
    io_when,
    io_when_unit,
    line_push,
    read_ref,
    ref_gate,
    ref_write_when,
    repeat_until,
    result_bind,
    result_either,
    result_map,
    result_map2,
    result_map_error,
    result_or_else,
    result_zip,
    write_ref,
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


def test_fold_io_push_threads_state_and_stops_on_error() -> None:
    state: Ref[Result[int, str]] = Ref(Ok(0))

    def advance(total: int, number: int) -> IO[Result[int, str]]:
        if number == 0:
            return io_result(Err("stopped at 0"))

        return io_result(Ok(total + number))

    sink = fold_io_push(advance, state)
    sink(1)
    sink(2)
    assert read_ref(state).run() == Ok(3)
    sink(0)
    assert read_ref(state).run() == Err("stopped at 0")
    sink(5)
    assert read_ref(state).run() == Err("stopped at 0")


def test_line_push_splits_chunks_into_newline_terminated_lines() -> None:
    lines: list[bytes] = []
    remainder: Ref[bytes] = Ref(b"")
    feed = line_push(lines.append, remainder)

    feed(b"ab\nc")
    feed(b"d\n\ne")

    assert lines == [b"ab\n", b"cd\n", b"\n"]
    assert read_ref(remainder).run() == b"e"


def test_ref_write_when_sets_the_value_only_on_matching_events() -> None:
    def over_ten(item: int) -> bool:
        return item > 10

    flag: Ref[bool] = Ref(False)
    consume = ref_write_when(over_ten, flag, True)

    consume(5)
    assert read_ref(flag).run() is False
    consume(11)
    assert read_ref(flag).run() is True


def test_ref_gate_drops_items_once_the_flag_is_set() -> None:
    passed: list[int] = []
    flag: Ref[bool] = Ref(False)
    consume = ref_gate(flag, passed.append)

    consume(1)
    write_ref(flag, True).run()
    consume(2)

    assert passed == [1]


def test_cons_prepends_and_materialises_in_order() -> None:
    items = cons(1)
    items = cons(2, items)
    items = cons(3, items)
    assert cons_to_tuple(items) == (1, 2, 3)
    assert cons_to_tuple(None) == ()
    assert cons("x") == Cons("x", None)


def test_cons_all_accumulates_head_items_before_the_tail() -> None:
    items = cons_all(("b", "c"), cons("a"))
    assert cons_to_tuple(items) == ("a", "b", "c")
    assert cons_all((), cons("a")) == cons("a")
    assert cons_all(()) is None


def test_io_catch_result_catches_exceptions_in_result_channel() -> None:
    def boom() -> Result[int, str]:
        raise RuntimeError("disaster")

    def handler(error: Exception) -> Result[int, str]:
        return Err("caught: %s" % error)

    assert io_catch_result(IO(boom), handler).run() == Err("caught: disaster")
    assert io_catch_result(io_result(Ok(1)), handler).run() == Ok(1)

import threading

from l2l.monads import (
    IO,
    io_memoize,
    modify_reference,
    modify_reference_with,
    new_reference,
    read_reference,
    write_reference,
)


def test_read_ref_returns_initial_value() -> None:
    reference = new_reference(7).run()
    assert read_reference(reference).run() == 7


def test_modify_ref_applies_function_and_returns_new_value() -> None:
    reference = new_reference(7).run()
    assert modify_reference(reference, lambda value: value + 1).run() == 8
    assert read_reference(reference).run() == 8


def test_modify_ref_with_returns_transition_output() -> None:
    reference = new_reference(7).run()

    def double_and_label(value: int) -> tuple[int, str]:
        return value * 2, "was %d" % value

    assert modify_reference_with(reference, double_and_label).run() == "was 7"
    assert read_reference(reference).run() == 14


def test_write_ref_replaces_value() -> None:
    reference = new_reference(7).run()
    assert write_reference(reference, 9).run() == 9
    assert read_reference(reference).run() == 9


def test_ref_reads_reflect_writes() -> None:
    reference = new_reference(0).run()
    write_reference(reference, 1).run()
    modify_reference(reference, lambda value: value * 10).run()
    assert read_reference(reference).run() == 10


def test_io_memoize_runs_effect_once() -> None:
    executions: list[int] = []

    def effect() -> int:
        executions.append(len(executions))
        return len(executions)

    memoized = io_memoize(IO(effect))
    assert memoized.run() == 1
    assert memoized.run() == 1
    assert len(executions) == 1


def test_io_memoize_shares_value_across_calls() -> None:
    counter = new_reference(0).run()

    def effect() -> int:
        return modify_reference(counter, lambda value: value + 1).run()

    memoized = io_memoize(IO(effect))
    assert memoized.run() == memoized.run() == 1


def test_ref_survives_concurrent_modifications() -> None:
    reference = new_reference(0).run()

    def bump() -> None:
        for _ in range(100):
            modify_reference(reference, lambda value: value + 1).run()

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    assert read_reference(reference).run() == 400

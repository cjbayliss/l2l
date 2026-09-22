from hypothesis import given
from hypothesis import strategies as st

from l2l.monads import (
    NOTHING,
    Err,
    Just,
    Maybe,
    Nothing,
    Ok,
    maybe_bind,
    maybe_either,
    maybe_from_optional,
    maybe_map,
    maybe_or_else,
    maybe_or_else_get,
    maybe_to_optional,
    maybe_to_result,
    maybe_zip,
    maybes_sequence,
)

integers = st.integers()
maybe_integers: st.SearchStrategy[Maybe[int]] = st.integers().map(Just) | st.just(
    NOTHING
)


def halve(value: int) -> Maybe[int]:
    return Just(value // 2) if value % 2 == 0 else NOTHING


def test_maybe_map_transforms_just_only() -> None:
    assert maybe_map(Just(2), str) == Just("2")
    assert maybe_map(NOTHING, str) == NOTHING


def test_maybe_bind_short_circuits() -> None:
    assert maybe_bind(Just(2), halve) == Just(1)
    assert maybe_bind(NOTHING, halve) == NOTHING


def test_maybe_either_folds_both_branches() -> None:
    assert maybe_either(Just(1), str, lambda: "none") == "1"
    assert maybe_either(NOTHING, str, lambda: "none") == "none"


def test_maybe_or_else_unwraps_or_falls_back() -> None:
    assert maybe_or_else(Just(1), 0) == 1
    assert maybe_or_else(NOTHING, 0) == 0


def test_maybe_or_else_get_is_lazy() -> None:
    def fallback() -> int:
        raise AssertionError("fallback must not run")

    assert maybe_or_else_get(Just(1), fallback) == 1
    assert maybe_or_else_get(NOTHING, lambda: 0) == 0


def test_maybe_from_optional_wraps_none_as_nothing() -> None:
    assert maybe_from_optional(None) == NOTHING
    assert maybe_from_optional(0) == Just(0)


def test_maybe_to_optional_unwraps() -> None:
    assert maybe_to_optional(Just(1)) == 1
    assert maybe_to_optional(NOTHING) is None


def test_maybe_to_result_lifts_into_error_channel() -> None:
    assert maybe_to_result(Just(1), lambda: "missing") == Ok(1)
    assert maybe_to_result(NOTHING, lambda: "missing") == Err("missing")


def test_maybe_zip_pairs_justs_only() -> None:
    assert maybe_zip(Just(1), Just("a")) == Just((1, "a"))
    assert maybe_zip(Just(1), NOTHING) == NOTHING
    assert maybe_zip(NOTHING, Just("a")) == NOTHING


def test_nothing_instances_are_equal() -> None:
    assert Nothing() == NOTHING
    assert maybe_to_optional(NOTHING) is None


@given(maybe_integers)
def test_maybe_functor_identity(maybe: Maybe[int]) -> None:
    assert maybe_map(maybe, lambda value: value) == maybe


@given(maybe_integers)
def test_maybe_functor_composition(maybe: Maybe[int]) -> None:
    def f(value: int) -> int:
        return value * 2

    def g(value: int) -> int:
        return value + 1

    assert maybe_map(maybe_map(maybe, g), f) == maybe_map(
        maybe, lambda value: f(g(value))
    )


@given(integers)
def test_maybe_left_identity(value: int) -> None:
    assert maybe_bind(Just(value), halve) == halve(value)


@given(maybe_integers)
def test_maybe_right_identity(maybe: Maybe[int]) -> None:
    assert maybe_bind(maybe, Just) == maybe


@given(integers)
def test_maybe_bind_associativity(value: int) -> None:
    def stringify(parsed: int) -> Maybe[str]:
        return Just(str(parsed)) if parsed % 3 else NOTHING

    left = maybe_bind(maybe_bind(Just(value), halve), stringify)
    right = maybe_bind(Just(value), lambda parsed: maybe_bind(halve(parsed), stringify))
    assert left == right


@given(st.lists(maybe_integers))
def test_maybes_sequence_matches_manual_fold(values: list[Maybe[int]]) -> None:
    justs: list[Just[int]] = [maybe for maybe in values if isinstance(maybe, Just)]
    if len(justs) != len(values):
        assert maybes_sequence(values) == NOTHING
    else:
        assert maybes_sequence(values) == Just(tuple(maybe.value for maybe in justs))

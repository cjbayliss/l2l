import pytest

from zh2en.config import build_settings
from zh2en.http import build_chat_payload
from zh2en.messages import usage_line
from zh2en.monads import NOTHING, Just
from zh2en.plans import (
    build_pass_user,
    build_unit_retry_user,
    context_parts,
    unit_output_problem,
)
from zh2en.text import (
    AsciiDrop,
    Usage,
    add_usage,
    cache_key,
    drop_non_ascii,
    ensure_blank_line_separators,
    estimate_tokens,
    is_cjk_char,
    make_chunks,
    non_ascii_sample,
    parse_cost,
    regroup_by_plan,
    split_paragraphs,
    split_sentences,
    split_to_budget,
    split_units_to_budget,
    to_ascii_mechanical,
    unit_separators,
    usage_add,
    usage_delta,
)


def test_split_paragraphs_blank_lines() -> None:
    paragraphs, separators = split_paragraphs("a\n\nb\n\nc")
    assert paragraphs == ("a", "b", "c")
    assert separators == ("\n\n", "\n\n")


def test_split_paragraphs_lone_newlines() -> None:
    paragraphs, separators = split_paragraphs("a\nb\nc")
    assert paragraphs == ("a", "b", "c")
    assert separators == ("\n", "\n")


def test_split_paragraphs_blank_majority() -> None:
    paragraphs, separators = split_paragraphs("a\nb\n\nc")
    assert paragraphs == ("a\nb", "c")
    assert separators == ("\n\n",)


def test_ensure_blank_line_separators() -> None:
    assert ensure_blank_line_separators("a\nb") == "a\n\nb"
    assert ensure_blank_line_separators("a\nb\n\nc") == "a\nb\n\nc"


def test_make_chunks_respects_budget() -> None:
    assert make_chunks(("a", "b", "c"), 2) == (("a", "b"), ("c",))
    assert make_chunks(("a", "b"), 10) == (("a", "b"),)


def test_make_chunks_keeps_oversized_paragraph_alone() -> None:
    assert make_chunks(("a", "longer"), 1) == (("a",), ("longer",))


def test_split_sentences() -> None:
    assert split_sentences("一。二。三", "。") == ("一。", "二。", "三")
    assert split_sentences("a! b? c.", "!?.") == ("a!", " b?", " c.")


def test_split_to_budget() -> None:
    pieces = split_to_budget("一。二。三。", 2, "。")
    assert pieces == ("一。", "二。", "三。")


def test_split_units_to_budget() -> None:
    units = split_units_to_budget(("一。二。", "三。"), 2, "。")
    assert units == ("一。", "二。", "三。")


def test_unit_separators() -> None:
    plan = (("a", "b"), ("c",))
    assert unit_separators(plan, ("s1", "s2", "s3")) == ("s2", "")


def test_regroup_by_plan() -> None:
    plan = (("a", "b"), ("c",))
    groups = regroup_by_plan(("a", "b", "c"), plan)
    assert groups == Just((("a", "b"), ("c",)))
    assert regroup_by_plan(("a", "b"), plan) == NOTHING


def test_estimate_tokens() -> None:
    assert estimate_tokens("中文") == 2
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("中文abc") == 3


def test_is_cjk_char() -> None:
    assert is_cjk_char("中")
    assert is_cjk_char("。")
    assert not is_cjk_char("a")
    assert not is_cjk_char(" ")


def test_non_ascii_sample() -> None:
    assert non_ascii_sample("abé×中", 3) == "é×中"
    assert non_ascii_sample("abc") == ""


def test_to_ascii_mechanical() -> None:
    character_map = build_settings().ascii_character_map
    assert to_ascii_mechanical("café — «x»…", character_map) == 'cafe - "x"...'


def test_drop_non_ascii_paths() -> None:
    character_map = build_settings().ascii_character_map
    assert drop_non_ascii("fine", character_map, 3, 0) == ("fine", NOTHING)
    fixed, drop = drop_non_ascii("café", character_map, 3, 0)
    assert fixed == "cafe"
    assert drop == NOTHING
    kept, drop = drop_non_ascii("a 中 b", character_map, 3, 1)
    assert kept == "a b"
    assert drop == Just(AsciiDrop(1, "中", 3))


def test_usage_add_accumulates_usages() -> None:
    total = usage_add(Usage(10, 5, 0.5), Usage(4, 9, 0.25))
    assert total == Usage(14, 14, 0.75)


def test_usage_line_and_delta() -> None:
    start = Usage(10, 5, 0.5)
    end = Usage(14, 9, 0.9)
    assert usage_delta(start, end) == pytest.approx((4, 4, 0.4))
    line = usage_line("Done", 30.0, 4, 4, 0.4)
    assert line.startswith("Done: 30.0s")


def test_parse_cost_falls_back_to_zero() -> None:
    assert parse_cost(0.25) == 0.25
    assert parse_cost(None) == 0.0
    assert parse_cost("nope") == 0.0
    assert parse_cost([]) == 0.0


def test_add_usage_accumulates() -> None:
    total = add_usage(Usage(10, 5, 0.5), {"prompt_tokens": 4, "cost": 0.25})
    assert total == Usage(14, 5, 0.75)
    tolerated = add_usage(Usage(), {"cost": "bad"})
    assert tolerated == Usage(0, 0, 0.0)


def test_build_chat_payload() -> None:
    payload = build_chat_payload("m", "s", "u", {"temperature": 1, "x": None})
    assert payload["model"] == "m"
    assert payload["temperature"] == 1
    assert "x" not in payload
    assert payload["messages"][0] == {"role": "system", "content": "s"}


def test_build_pass_user() -> None:
    user = build_pass_user("src", "draft", " brief ")
    assert user == "brief\n\nsrc\n\ndraft"
    same = build_pass_user("src", "src", None)
    assert same == "src"
    blank = build_pass_user("src", None, "   ")
    assert blank == "src"


def test_unit_output_problem_accepts_matching_reply() -> None:
    settings = build_settings()
    problem = unit_output_problem("你好。\n\n世界。", "Hello.\n\nWorld.", settings)
    assert problem == NOTHING


def test_unit_output_problem_flags_empty_reply() -> None:
    settings = build_settings()
    problem = unit_output_problem("你好。", "   ", settings)
    assert isinstance(problem, Just)
    assert "empty" in problem.value


def test_unit_output_problem_flags_paragraph_mismatch() -> None:
    settings = build_settings()
    extra = unit_output_problem("你好。", "Hello.\n\nWorld.", settings)
    assert isinstance(extra, Just)
    assert "has 2 paragraph(s) but the source has 1" in extra.value
    missing = unit_output_problem("你好。\n\n世界。", "Hello.", settings)
    assert isinstance(missing, Just)
    assert "has 1 paragraph(s) but the source has 2" in missing.value


def test_unit_output_problem_flags_implausible_length() -> None:
    settings = build_settings()
    problem = unit_output_problem("嗯。", "word " * 40, settings)
    assert isinstance(problem, Just)
    assert "tokens" in problem.value


def test_context_parts_and_build_pass_user_with_context() -> None:
    assert context_parts(()) == ()
    user = build_pass_user("src", None, None, ("before", "after"))
    assert user.count("reference only") == 1
    assert "before\n\nafter" in user
    assert user.endswith("src")


def test_build_unit_retry_user_includes_problem_and_reply() -> None:
    user = build_unit_retry_user("src", (), "bad reply", "the reply was empty")
    assert "the reply was empty" in user
    assert "Previous reply:\nbad reply" in user
    assert "\n\nsrc\n\n" in user
    assert user.endswith("output only the translation.")


def test_cache_key_distinguishes_context() -> None:
    base = cache_key("你好。", "m", "salt", "你好。")
    repeated = cache_key("你好。", "m", "salt", "你好。", context="世界。")
    assert base != repeated
    assert repeated == cache_key("你好。", "m", "salt", "你好。", context="世界。")

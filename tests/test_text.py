import pytest

import zh2en as z


def test_split_paragraphs_blank_lines() -> None:
    paragraphs, separators = z.split_paragraphs("a\n\nb\n\nc")
    assert paragraphs == ("a", "b", "c")
    assert separators == ("\n\n", "\n\n")


def test_split_paragraphs_lone_newlines() -> None:
    paragraphs, separators = z.split_paragraphs("a\nb\nc")
    assert paragraphs == ("a", "b", "c")
    assert separators == ("\n", "\n")


def test_split_paragraphs_blank_majority() -> None:
    paragraphs, separators = z.split_paragraphs("a\nb\n\nc")
    assert paragraphs == ("a\nb", "c")
    assert separators == ("\n\n",)


def test_ensure_blank_line_separators() -> None:
    assert z.ensure_blank_line_separators("a\nb") == "a\n\nb"
    assert z.ensure_blank_line_separators("a\nb\n\nc") == "a\nb\n\nc"


def test_make_chunks_respects_budget() -> None:
    assert z.make_chunks(("a", "b", "c"), 2) == (("a", "b"), ("c",))
    assert z.make_chunks(("a", "b"), 10) == (("a", "b"),)


def test_make_chunks_keeps_oversized_paragraph_alone() -> None:
    assert z.make_chunks(("a", "longer"), 1) == (("a",), ("longer",))


def test_split_sentences() -> None:
    assert z.split_sentences("一。二。三", "。") == ("一。", "二。", "三")
    assert z.split_sentences("a! b? c.", "!?.") == ("a!", " b?", " c.")


def test_split_to_budget() -> None:
    pieces = z.split_to_budget("一。二。三。", 2, "。")
    assert pieces == ("一。", "二。", "三。")


def test_split_units_to_budget() -> None:
    units = z.split_units_to_budget(("一。二。", "三。"), 2, "。")
    assert units == ("一。", "二。", "三。")


def test_unit_separators() -> None:
    plan = (("a", "b"), ("c",))
    assert z.unit_separators(plan, ("s1", "s2", "s3")) == ("s2", "")


def test_regroup_by_plan() -> None:
    plan = (("a", "b"), ("c",))
    groups = z.regroup_by_plan(("a", "b", "c"), plan)
    assert groups == (("a", "b"), ("c",))
    assert z.regroup_by_plan(("a", "b"), plan) is None


def test_estimate_tokens() -> None:
    assert z.estimate_tokens("中文") == 2
    assert z.estimate_tokens("abcd") == 1
    assert z.estimate_tokens("中文abc") == 3


def test_is_cjk_char() -> None:
    assert z.is_cjk_char("中")
    assert z.is_cjk_char("。")
    assert not z.is_cjk_char("a")
    assert not z.is_cjk_char(" ")


def test_non_ascii_sample() -> None:
    assert z.non_ascii_sample("abé×中", 3) == "é×中"
    assert z.non_ascii_sample("abc") == ""


def test_to_ascii_mechanical() -> None:
    character_map = z.build_settings().ascii_character_map
    assert z.to_ascii_mechanical("café — «x»…", character_map) == 'cafe - "x"...'


def test_drop_non_ascii_paths() -> None:
    character_map = z.build_settings().ascii_character_map
    assert z.drop_non_ascii("fine", character_map, 3, 0) == ("fine", None)
    fixed, warning = z.drop_non_ascii("café", character_map, 3, 0)
    assert fixed == "cafe"
    assert warning is None
    kept, warning = z.drop_non_ascii("a 中 b", character_map, 3, 1)
    assert kept == "a b"
    assert warning is not None
    assert "paragraph 2" in warning


def test_usage_line_and_delta() -> None:
    start = z.Usage(10, 5, 0.5)
    end = z.Usage(14, 9, 0.9)
    assert z.usage_delta(start, end) == pytest.approx((4, 4, 0.4))
    line = z.usage_line("Done", 30.0, 4, 4, 0.4)
    assert line.startswith("Done: 30.0s")


def test_build_chat_payload() -> None:
    payload = z.build_chat_payload("m", "s", "u", {"temperature": 1, "x": None})
    assert payload["model"] == "m"
    assert payload["temperature"] == 1
    assert "x" not in payload
    assert payload["messages"][0] == {"role": "system", "content": "s"}


def test_build_pass_user() -> None:
    user = z.build_pass_user("src", "draft", " brief ")
    assert user == "brief\n\nsrc\n\ndraft"
    same = z.build_pass_user("src", "src", None)
    assert same == "src"
    blank = z.build_pass_user("src", None, "   ")
    assert blank == "src"


def test_unit_output_problem_accepts_matching_reply() -> None:
    settings = z.build_settings()
    problem = z.unit_output_problem("你好。\n\n世界。", "Hello.\n\nWorld.", settings)
    assert problem is None


def test_unit_output_problem_flags_empty_reply() -> None:
    settings = z.build_settings()
    problem = z.unit_output_problem("你好。", "   ", settings)
    assert problem is not None
    assert "empty" in problem


def test_unit_output_problem_flags_paragraph_mismatch() -> None:
    settings = z.build_settings()
    extra = z.unit_output_problem("你好。", "Hello.\n\nWorld.", settings)
    assert extra is not None
    assert "has 2 paragraph(s) but the source has 1" in extra
    missing = z.unit_output_problem("你好。\n\n世界。", "Hello.", settings)
    assert missing is not None
    assert "has 1 paragraph(s) but the source has 2" in missing


def test_unit_output_problem_flags_implausible_length() -> None:
    settings = z.build_settings()
    problem = z.unit_output_problem("嗯。", "word " * 40, settings)
    assert problem is not None
    assert "tokens" in problem


def test_context_parts_and_build_pass_user_with_context() -> None:
    assert z.context_parts(()) == ()
    user = z.build_pass_user("src", None, None, ("before", "after"))
    assert user.count("reference only") == 1
    assert "before\n\nafter" in user
    assert user.endswith("src")


def test_build_unit_retry_user_includes_problem_and_reply() -> None:
    user = z.build_unit_retry_user("src", (), "bad reply", "the reply was empty")
    assert "the reply was empty" in user
    assert "Previous reply:\nbad reply" in user
    assert "\n\nsrc\n\n" in user
    assert user.endswith("output only the translation.")


def test_cache_key_distinguishes_context() -> None:
    base = z.cache_key("你好。", "m", "salt", "你好。")
    repeated = z.cache_key("你好。", "m", "salt", "你好。", context="世界。")
    assert base != repeated
    assert repeated == z.cache_key("你好。", "m", "salt", "你好。", context="世界。")

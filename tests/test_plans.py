from __future__ import annotations

from fakes import (
    FakeHttp,
    chunk_pass,
    make_console,
    make_context,
    paragraph_pass,
    update_pass,
)

from l2l.monads import Just, Nothing
from l2l.plans import plan_retranslation_calls, plan_unit_calls, resolve_work_groups
from l2l.report import plan_report
from l2l.settings import PassDefinition
from l2l.text import unit_separators


def test_plan_unit_calls_builds_keys_and_context() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open)
    plan = (("一。",), ("二。",), ("三。",))
    separators = unit_separators(plan, ("s1", "s2", "s3"))
    calls = plan_unit_calls(
        run_context,
        paragraph_pass(),
        plan,
        (("一。",), ("二。",), ("三。",)),
        separators,
        None,
    )
    assert [call.source_chunk for call in calls] == ["一。", "二。", "三。"]
    assert [call.analysis for call in calls] == [None, None, None]
    assert [call.context for call in calls] == [("二。",), ("一。", "三。"), ("二。",)]
    assert [call.trailing_separator for call in calls] == ["s1", "s2", ""]
    assert len({call.key for call in calls}) == 3
    briefed = plan_unit_calls(
        run_context,
        paragraph_pass(),
        plan,
        (("一。",), ("二。",), ("三。",)),
        separators,
        "Brief.",
    )
    assert [call.analysis for call in briefed] == ["Brief.", "Brief.", "Brief."]
    assert [call.key for call in briefed] != [call.key for call in calls]


def test_plan_retranslation_calls_targets_flagged_paragraphs() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open)
    sources = ("一。", "二。", "三。")
    calls = plan_retranslation_calls(
        run_context,
        paragraph_pass(),
        sources,
        sources,
        (0, 2),
        ("s1", "s2", "s3"),
        None,
    )
    assert [call.source_chunk for call in calls] == ["一。", "三。"]
    assert [call.work_chunk for call in calls] == ["一。", "三。"]
    assert [call.index for call in calls] == [0, 1]
    assert [call.total for call in calls] == [2, 2]
    assert [call.trailing_separator for call in calls] == ["s1", "s3"]
    assert [call.context for call in calls] == [("二。",), ("二。",)]
    assert len({call.key for call in calls}) == 2
    plan = (("一。",), ("二。",), ("三。",))
    unit_calls = plan_unit_calls(
        run_context,
        paragraph_pass(),
        plan,
        plan,
        unit_separators(plan, ("s1", "s2", "s3")),
        None,
    )
    assert calls[0].key != unit_calls[0].key
    assert calls[1].key != unit_calls[2].key


def test_plan_report_lists_units_and_keys() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open)
    text = "你好。\n\n世界。"
    report = plan_report(run_context, (chunk_pass(), paragraph_pass()), text)
    lines = report.splitlines()
    assert lines[0] == "source: 8 character(s), 2 paragraph(s), ~7 tokens"
    assert (
        lines[1]
        == "pass 1/2 [translate]: mode=chunk, 1 unit(s), source budget ~3500 tokens"
    )
    assert "unit 1/1: ~7 source tokens, cache key " in lines[2]
    assert lines[3] == "pass 2/2 [translate]: mode=paragraph, 2 unit(s)"
    assert "unit 1/2:" in lines[4]
    assert "unit 2/2:" in lines[5]


def test_plan_report_reports_analysis_pass() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open)
    analysis = PassDefinition("prep", "Brief.", "analysis", {}, None, False)
    report = plan_report(run_context, (analysis,), "你好。")
    assert "pass 1/1 [prep]: mode=analysis, 1 call with the whole document" in report


def test_plan_report_exposes_budget_violations() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open, maximum_tokens=200)
    report = plan_report(run_context, (chunk_pass(),), "a" * 1200)
    assert "source budget ~183 tokens" in report
    assert (
        "warning: [translate] paragraph 1 is ~300 source tokens, "
        "over the ~183-token source budget" in report
    )


def test_plan_report_reserves_draft_space_in_later_passes() -> None:
    console, _ = make_console()
    run_context = make_context(console, FakeHttp([]).open, maximum_tokens=400)
    paragraph = "a" * 400
    text = paragraph + "\n\n" + paragraph
    report = plan_report(run_context, (chunk_pass(), chunk_pass("polish")), text)
    assert (
        "pass 1/2 [translate]: mode=chunk, 1 unit(s), source budget ~383 tokens"
        in report
    )
    assert (
        "pass 2/2 [polish]: mode=chunk, 2 unit(s), source budget ~191 tokens" in report
    )


def test_resolve_work_groups_regroups_paragraph_mode_independently() -> None:
    paragraphs = ("一。", "二。", "三。")
    stale_plan = (("一。", "二。"),)
    groups, warning = resolve_work_groups("paragraph", paragraphs, stale_plan, "p", 100)
    assert groups == (("一。",), ("二。",), ("三。",))
    assert isinstance(warning, Just)


def test_resolve_work_groups_rechunks_chunk_mode_independently() -> None:
    paragraphs = tuple("一。" for _ in range(5))
    stale_plan = (("一。",),)
    groups, warning = resolve_work_groups("chunk", paragraphs, stale_plan, "p", 1)
    assert all(len(group) == 1 for group in groups)
    assert len(groups) == 5
    assert isinstance(warning, Just)


def test_resolve_work_groups_keeps_matching_plan() -> None:
    paragraphs = ("一。", "二。")
    plan = (("一。",), ("二。",))
    groups, warning = resolve_work_groups("chunk", paragraphs, plan, "p", 100)
    assert groups == (("一。",), ("二。",))
    assert isinstance(warning, Nothing)


def test_plan_report_shows_update_glossary_pass() -> None:
    console, _stderr = make_console()
    run_context = make_context(console, FakeHttp([]).open)
    report = plan_report(
        run_context,
        (update_pass(target_language="Chinese"), chunk_pass()),
        "你好。",
    )
    assert (
        "pass 1/2 [glossary]: mode=update_glossary, 1 call with the whole "
        "document, target language Chinese" in report
    )
    assert "pass 2/2 [translate]: mode=chunk, 1 unit(s)" in report


def test_plan_report_update_glossary_falls_back_to_context_language() -> None:
    console, _stderr = make_console()
    run_context = make_context(console, FakeHttp([]).open, target_language="French")
    report = plan_report(run_context, (update_pass(),), "你好。")
    assert "target language French" in report

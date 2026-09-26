import io
import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeHttp,
    FakePlainResponse,
    FakeStreamResponse,
    make_console,
    make_context,
    stream_chunks,
    with_usage,
)

from l2l import cli
from l2l.errors import TranslationError, describe, fail_http
from l2l.http import chat
from l2l.monads import (
    IO,
    Err,
    Just,
    Nothing,
    Ok,
    Result,
    fold_io,
    fold_while,
    io_pure,
    io_result,
)
from l2l.pipeline import analyze_document, run_pipeline
from l2l.plans import (
    ascii_drop_warning,
    plan_report,
    plan_retranslation_calls,
    plan_unit_calls,
    resolve_work_groups,
)
from l2l.settings import PassDefinition
from l2l.text import AsciiDrop, Usage, unit_separators

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def chunk_pass(
    name: str = "translate",
    ascii_output: bool = False,
    ensure_paragraphs: bool | int = False,
    retranslate_untranslated: bool = False,
) -> PassDefinition:
    return PassDefinition(
        name,
        "T.",
        "chunk",
        {},
        None,
        ascii_output,
        ensure_paragraphs,
        retranslate_untranslated,
    )


def paragraph_pass(
    name: str = "translate", retranslate_untranslated: bool = False
) -> PassDefinition:
    return PassDefinition(
        name,
        "T.",
        "paragraph",
        {},
        None,
        False,
        False,
        retranslate_untranslated,
    )


def test_fold_io_handles_thousands_of_items() -> None:
    def step(pair: tuple[int, None], _: int) -> IO[Result[tuple[int, None], str]]:
        return io_pure(Ok((pair[0] + 1, None)))

    outcome = fold_io(range(5000), step, Ok((0, None))).run()
    assert isinstance(outcome, Ok)
    assert outcome.value[0] == 5000


def test_fold_while_stops_before_consuming_next_item() -> None:
    pulled: list[int] = []

    def items() -> Iterator[int]:
        for value in (1, 2, 3):
            pulled.append(value)
            yield value

    def step(count: int, value: int) -> Result[int, str]:
        return Err("stop") if value == 2 else Ok(count + value)

    outcome = fold_while(items(), step, Ok(0))
    assert isinstance(outcome, Err)
    assert outcome.error == "stop"
    assert pulled == [1, 2]


def test_plan_unit_calls_builds_keys_and_context() -> None:
    console, _ = make_console()
    ctx = make_context(console, FakeHttp([]).open)
    plan = (("一。",), ("二。",), ("三。",))
    separators = unit_separators(plan, ("s1", "s2", "s3"))
    calls = plan_unit_calls(
        ctx,
        paragraph_pass(),
        plan,
        (("一。",), ("二。",), ("三。",)),
        separators,
    )
    assert [call.source_chunk for call in calls] == ["一。", "二。", "三。"]
    assert [call.context for call in calls] == [("二。",), ("一。", "三。"), ("二。",)]
    assert [call.trailing_separator for call in calls] == ["s1", "s2", ""]
    assert len({call.key for call in calls}) == 3


def test_plan_retranslation_calls_targets_flagged_paragraphs() -> None:
    console, _ = make_console()
    ctx = make_context(console, FakeHttp([]).open)
    sources = ("一。", "二。", "三。")
    calls = plan_retranslation_calls(
        ctx, paragraph_pass(), sources, sources, (0, 2), ("s1", "s2", "s3")
    )
    assert [call.source_chunk for call in calls] == ["一。", "三。"]
    assert [call.work_chunk for call in calls] == ["一。", "三。"]
    assert [call.index for call in calls] == [0, 1]
    assert [call.total for call in calls] == [2, 2]
    assert [call.trailing_separator for call in calls] == ["s1", "s3"]
    assert [call.context for call in calls] == [("二。",), ("二。",)]
    assert len({call.key for call in calls}) == 2


def test_fold_io_short_circuits_on_error() -> None:
    calls: list[int] = []

    def step(pair: tuple[int, None], item: int) -> IO[Result[tuple[int, None], str]]:
        calls.append(item)
        if item == 1:
            return io_result(Err("boom"))

        return io_pure(Ok((pair[0] + 1, None)))

    outcome = fold_io(range(4), step, Ok((0, None))).run()
    assert isinstance(outcome, Err)
    assert outcome.error == "boom"
    assert calls == [0, 1]


def test_chat_streamed_reports_usage() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hi"), USAGE))])
    ctx = make_context(console, http.open)
    result = chat(ctx, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Hi"
    assert result.value.usage == Usage(5, 6, 0.2)


def test_chat_streamed_filters_think_prefix() -> None:
    console, stderr = make_console()
    chunks = stream_chunks("<think>h</think>", "Body")
    http = FakeHttp([FakeStreamResponse(with_usage(chunks, USAGE))])
    ctx = make_context(console, http.open)
    result = chat(ctx, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Body"


def test_chat_plain_when_stream_disabled() -> None:
    console, stderr = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    ctx = make_context(console, http.open)
    result = chat(ctx, "s", "u", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value.text == "Plain"
    payload = json.loads(http.requests[0].data)
    assert payload["stream"] is False


def test_chat_rejects_oversized_request() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    ctx = make_context(console, http.open, max_tokens=1)
    result = chat(ctx, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "over the 1-token budget" in describe(result.error)
    assert http.requests == []


def test_run_pipeline_runs_analysis_before_translation() -> None:
    console, stderr = make_console()
    brief = with_usage(stream_chunks("Names: Qin Yu."), USAGE)
    translated = with_usage(stream_chunks("Hello."), USAGE)
    http = FakeHttp([FakeStreamResponse(brief), FakeStreamResponse(translated)])
    ctx = make_context(console, http.open)
    analysis = PassDefinition("prep", "Summarise.", "analysis", {}, None, False)
    stdout = io.StringIO()
    code = run_pipeline(ctx, (analysis, chunk_pass()), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    second_user = json.loads(http.requests[1].data)["messages"][1]["content"]
    assert "Names: Qin Yu." in second_user


def test_run_pipeline_wraps_analysis_failure() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, TranslationError]:
        return fail_http("unreachable", "down")

    ctx = make_context(console, open_fail)
    analysis = PassDefinition("prep", "Summarise.", "analysis", {}, None, False)
    code = run_pipeline(ctx, (analysis,), "你好。", 0.0, io.StringIO()).run()
    assert code == 1
    logged = stderr.getvalue()
    assert "pass [prep] failed" in logged
    assert "could not reach endpoint: down" in logged


def test_run_pipeline_repairs_non_ascii_via_llm() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hi 中"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hi there"), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hi there\n"
    assert len(http.requests) == 2


def test_run_pipeline_drops_non_ascii_after_failed_repairs() -> None:
    console, stderr = make_console()
    stubborn = "中 x"
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE))]
        + [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE))
            for _ in range(3)
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == " x\n"
    logged = stderr.getvalue()
    assert "still contained non-ASCII characters (中) after 3 LLM attempts" in logged
    assert len(http.requests) == 4


def test_retranslation_retranslates_an_echoed_paragraph() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("Hello.\n\n世界。\n\nThree."), USAGE)
            ),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。\n\n三。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n\nThree.\n"
    assert len(http.requests) == 2
    logged = stderr.getvalue()
    assert "1 of 3 paragraph(s) look untranslated" in logged
    assert "(more than half)" not in logged
    retry = json.loads(http.requests[1].data)
    assert retry["messages"][0]["content"] == "T."
    retry_user = retry["messages"][1]["content"]
    assert retry_user.startswith("Context paragraphs (reference only")
    assert "你好。" in retry_user
    assert "三。" in retry_user
    assert retry_user.endswith("世界。")


def test_retranslation_accepts_non_latin_targets() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks("Привет.\n\nМир."), USAGE))]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Привет.\n\nМир.\n"
    assert len(http.requests) == 1
    assert "look untranslated" not in stderr.getvalue()


def test_retranslation_retranslates_only_flagged_paragraphs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\n世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2


def test_retranslation_fails_when_paragraph_still_untranslated() -> None:
    console, stderr = make_console()
    echo = "你好。\n\n世界。"
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        echo,
        0.0,
        io.StringIO(),
    ).run()
    assert code == 1
    assert len(http.requests) == 4
    logged = stderr.getvalue()
    assert "2 of 2 paragraph(s) look untranslated (more than half)" in logged
    assert (
        "source script cjk, target script unknown; using the ASCII-letter rule"
        in logged
    )
    assert "still untranslated after retranslation" in logged
    assert "你好。" in logged


def test_retranslation_disabled_by_default() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [FakeStreamResponse(with_usage(stream_chunks("你好。\n\n世界。"), USAGE))]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n\n世界。\n"
    assert len(http.requests) == 1


def test_retranslation_skipped_when_paragraph_count_differs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("Hello.\n\n世界。\n\n世界。"), USAGE)
            )
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert len(http.requests) == 1


def test_retranslation_ignores_letterless_paragraphs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。\n\n123"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n123",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\n123\n"
    assert len(http.requests) == 2


def test_retranslation_runs_before_ascii_enforcement() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好，世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks('Hello, "world".'), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(ascii_output=True, retranslate_untranslated=True),),
        "你好，世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == 'Hello, "world".\n'
    assert len(http.requests) == 2


def test_retranslation_results_are_cached(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    def run() -> tuple[int, str, int]:
        console, _ = make_console()
        echo = "你好。\n\n世界。"
        http = FakeHttp(
            [
                FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
                FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
            ]
        )
        ctx = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            ctx,
            (chunk_pass(retranslate_untranslated=True),),
            "你好。\n\n世界。",
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run()
    second_code, second_output, second_calls = run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (4, 0)
    assert first_output == second_output == "Hello.\n\nWorld.\n"


def test_retranslation_majority_retries_pass_then_succeeds() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。\n\n世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2
    logged = stderr.getvalue()
    assert "2 of 2 paragraph(s) look untranslated (more than half)" in logged
    assert "retranslating them" not in logged


def test_retranslation_majority_retries_then_falls_back_per_paragraph() -> None:
    console, stderr = make_console()
    echo = "你好。\n\n世界。"
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        echo,
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 4
    logged = stderr.getvalue()
    assert logged.count("(more than half)") == 1
    assert "2 of 2 paragraph(s) look untranslated; retranslating them" in logged


def test_retranslation_majority_retry_uses_fresh_cache_keys(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    echo = "你好。\n\n世界。"

    def first_run() -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp(
            [
                FakeStreamResponse(with_usage(stream_chunks(echo), USAGE)),
                FakeStreamResponse(
                    with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)
                ),
            ]
        )
        ctx = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            ctx,
            (chunk_pass(retranslate_untranslated=True),),
            echo,
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    def second_run() -> tuple[int, str, int]:
        console, _ = make_console()
        http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("BAD"), USAGE))])
        ctx = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
        )
        stdout = io.StringIO()
        code = run_pipeline(
            ctx,
            (chunk_pass(retranslate_untranslated=True),),
            echo,
            0.0,
            stdout,
        ).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = first_run()
    second_code, second_output, second_calls = second_run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (2, 0)
    assert first_output == second_output == "Hello.\n\nWorld.\n"


def test_retranslation_paragraph_mode_skips_majority_retry() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("你好。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("世界。"), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (paragraph_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 4
    logged = stderr.getvalue()
    assert "(more than half)" not in logged
    assert "2 of 2 paragraph(s) look untranslated; retranslating them" in logged


def test_retranslation_at_exact_half_keeps_per_paragraph_path() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("Hello.\n\n世界。\n\nWarm.\n\n冬天。"), USAGE)
            ),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Winter."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (chunk_pass(retranslate_untranslated=True),),
        "你好。\n\n世界。\n\n春天。\n\n冬天。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n\nWarm.\n\nWinter.\n"
    assert len(http.requests) == 3
    logged = stderr.getvalue()
    assert "2 of 4 paragraph(s) look untranslated; retranslating them" in logged
    assert "(more than half)" not in logged


def test_retranslation_skips_analysis_passes() -> None:
    console, _ = make_console()
    brief = with_usage(stream_chunks("Brief."), USAGE)
    http = FakeHttp([FakeStreamResponse(brief)])
    ctx = make_context(console, http.open)
    analysis = PassDefinition(
        "prep", "Summarise.", "analysis", {}, None, False, None, True
    )
    stdout = io.StringIO()
    code = run_pipeline(ctx, (analysis,), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n"
    assert len(http.requests) == 1


def test_ascii_enforcement_skips_analysis_passes() -> None:
    console, _ = make_console()
    brief = with_usage(stream_chunks("Brief."), USAGE)
    http = FakeHttp([FakeStreamResponse(brief)])
    ctx = make_context(console, http.open)
    analysis = PassDefinition("prep", "Summarise.", "analysis", {}, None, True)
    stdout = io.StringIO()
    code = run_pipeline(ctx, (analysis,), "你好。\n\n世界。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "你好。\n\n世界。\n"
    assert len(http.requests) == 1


def test_run_pipeline_translates() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert "TOTAL" in stderr.getvalue()


def test_run_pipeline_total_elapsed_measures_since_started() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    readings: list[float] = []
    current = 100.0

    def clock() -> float:
        nonlocal current
        current += 5.0
        readings.append(current)
        return current

    ctx = make_context(console, http.open, clock=clock)
    stdout = io.StringIO()
    started = clock()
    code = run_pipeline(ctx, (chunk_pass(),), "你好。", started, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    elapsed = readings[-1] - readings[0]
    total = [line for line in stderr.getvalue().splitlines() if "TOTAL" in line][0]
    assert "TOTAL: %.1fs" % elapsed in total
    rate = USAGE["completion_tokens"] / elapsed if elapsed > 0 else 0.0
    assert "%.1f tok/s" % rate in total


def test_ascii_drop_warning_mentions_paragraph_sample_and_attempts() -> None:
    warning = ascii_drop_warning(AsciiDrop(index=1, sample="中", attempts=3))
    assert warning == (
        "l2l: ascii: warning: paragraph 2 still contained non-ASCII "
        "characters (中) after 3 LLM attempts; dropping them"
    )


def test_run_pipeline_reports_unit_failure() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, TranslationError]:
        return fail_http("unreachable", "down")

    ctx = make_context(console, open_fail)
    code = run_pipeline(ctx, (chunk_pass(),), "你好。", 0.0, io.StringIO()).run()
    assert code == 1
    logged = stderr.getvalue()
    assert "failed on unit 1" in logged
    assert "could not reach endpoint: down" in logged


def test_run_pipeline_cache_hit(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    chunks = with_usage(stream_chunks("Hello."), USAGE)

    def run() -> tuple[int, str, int]:
        console, stderr = make_console()
        http = FakeHttp([FakeStreamResponse(chunks)])
        ctx = make_context(
            console, http.open, cache_directory=str(cache_directory), use_cache=True
        )
        stdout = io.StringIO()
        code = run_pipeline(ctx, (chunk_pass(),), "你好。", 0.0, stdout).run()
        return code, stdout.getvalue(), len(http.requests)

    first_code, first_output, first_calls = run()
    second_code, second_output, second_calls = run()
    assert (first_code, second_code) == (0, 0)
    assert (first_calls, second_calls) == (1, 0)
    assert first_output == second_output == "Hello.\n"


def test_analysis_fails_fast_when_document_needs_multiple_parts() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    ctx = make_context(console, http.open, max_tokens=40)
    analysis_pass = PassDefinition("prep", "Brief.", "analysis", {}, None, False)
    result = analyze_document(ctx, analysis_pass, "一。二。三。四。", Usage()).run()
    assert isinstance(result, Err)
    assert "requires analysis in" in describe(result.error)
    assert http.requests == []


def test_run_pipeline_enforces_ascii_mechanically() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Café — déjà."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Cafe - deja.\n"
    assert len(http.requests) == 1


def test_ensure_paragraphs_retries_pass_in_paragraph_mode() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo.\n\nThree."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ensure_paragraphs=True),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    logged = stderr.getvalue()
    assert "output has 3 paragraph(s), source has 2" in logged
    assert "re-running the pass with one call per paragraph" in logged
    assert "failed validation 2 time(s)" in logged
    assert len(http.requests) == 5


def test_ensure_paragraphs_warns_when_retry_still_differs() -> None:
    console, stderr = make_console()
    stubborn = "One.\n\nTwo.\n\nThree."
    hello = "Hello.\n\nSurprise."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(stubborn), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(hello), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(hello), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(hello), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ensure_paragraphs=True),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nSurprise.\n\nWorld.\n"
    logged = stderr.getvalue()
    assert "still differs (3 vs 2)" in logged
    assert "continuing" in logged
    assert len(http.requests) == 7


def test_unit_validation_repairs_hallucinated_paragraph(tmp_path: Path) -> None:
    console, stderr = make_console()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(
                    stream_chunks(
                        "# T.\n\n Shen Yue never expected Qin Yu to say"
                        " that.\n\n She froze for a moment."
                    ),
                    USAGE,
                )
            ),
            FakeStreamResponse(with_usage(stream_chunks("Chapter 4: Ruined."), USAGE)),
        ]
    )
    ctx = make_context(
        console, http.open, cache_directory=str(cache_directory), use_cache=True
    )
    stdout = io.StringIO()
    code = run_pipeline(
        ctx,
        (paragraph_pass(),),
        "第413章 被亲妈祸害的女孩 2",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Chapter 4: Ruined.\n"
    assert len(http.requests) == 2
    first_user = json.loads(http.requests[0].data)["messages"][1]["content"]
    assert "reference only" not in first_user
    retry_user = json.loads(http.requests[1].data)["messages"][1]["content"]
    assert "does not satisfy the output rules" in retry_user
    assert "Shen Yue" in retry_user
    cached = list(cache_directory.glob("*.txt"))
    assert len(cached) == 1
    assert cached[0].read_text(encoding="utf-8") == "Chapter 4: Ruined."


def test_unit_validation_gives_up_warns_and_skips_cache(tmp_path: Path) -> None:
    console, stderr = make_console()
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    bad = "One.\n\nTwo."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(bad), USAGE)),
        ]
    )
    ctx = make_context(
        console, http.open, cache_directory=str(cache_directory), use_cache=True
    )
    stdout = io.StringIO()
    code = run_pipeline(ctx, (paragraph_pass(),), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n\nTwo.\n"
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "uncached" in logged
    assert len(http.requests) == 3
    assert list(cache_directory.glob("*.txt")) == []


def test_unit_validation_repairs_dropped_chunk_paragraph() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ensure_paragraphs=True),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].data)["messages"][1]["content"]
    assert "has 1 paragraph(s) but the source has 2" in retry_user


def test_paragraph_check_disabled_without_ensure_paragraphs() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    assert len(http.requests) == 1
    assert "paragraph(s), source has" not in stderr.getvalue()


def test_ensure_paragraphs_tolerance_accepts_small_drift() -> None:
    console, stderr = make_console()
    drifting = "One.\n\nTwo.\n\nThree."
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks(drifting), USAGE))])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ensure_paragraphs=1),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == drifting + "\n"
    assert len(http.requests) == 1
    assert "output has" not in stderr.getvalue()


def test_ensure_paragraphs_tolerance_exceeded_retries_in_paragraph_mode() -> None:
    console, stderr = make_console()
    wild = "One.\n\nTwo.\n\nThree.\n\nFour."
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks(wild), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(wild), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks(wild), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("World."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(ensure_paragraphs=1),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    logged = stderr.getvalue()
    assert "output has 4 paragraph(s), source has 2 (allowed ±1)" in logged
    assert "re-running the pass with one call per paragraph" in logged
    assert len(http.requests) == 5


def test_paragraph_mode_stays_strict_despite_tolerance() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("One.\n\nTwo."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("One."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    merged = PassDefinition("translate", "T.", "paragraph", {}, None, False, 2)
    code = run_pipeline(ctx, (merged,), "你好。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].data)["messages"][1]["content"]
    assert "has 2 paragraph(s) but the source has 1" in retry_user
    assert "(allowed" not in retry_user


def test_unit_validation_rejects_implausible_length() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("word " * 40), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Okay."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(ctx, (paragraph_pass(),), "嗯。", 0.0, stdout).run()
    assert code == 0
    assert stdout.getvalue() == "Okay.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].data)["messages"][1]["content"]
    assert "tokens against a source" in retry_user


def test_paragraph_mode_supplies_neighbour_context() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(with_usage(stream_chunks("One."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Two."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Three."), USAGE)),
        ]
    )
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (paragraph_pass(),), "一。\n\n二。\n\n三。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "One.\n\nTwo.\n\nThree.\n"
    users = [
        json.loads(request.data)["messages"][1]["content"] for request in http.requests
    ]
    assert users[0].count("reference only") == 1
    assert "二。" in users[0]
    assert "三。" not in users[0]
    assert users[1].count("reference only") == 1
    assert "一。" in users[1]
    assert "三。" in users[1]
    assert users[2].count("reference only") == 1
    assert "二。" in users[2]
    assert "一。" not in users[2]


def test_main_empty_stdin_succeeds_without_config() -> None:
    code = cli.main(
        [], {}, io.StringIO("   "), io.StringIO(), io.StringIO(), time.time
    ).run()
    assert code == 0


def test_main_end_to_end_with_config(tmp_path: Path) -> None:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        "[api]\n"
        'base_url = "http://endpoint.test/v1"\n'
        'api_key = "key"\n'
        'model = "m"\n'
        "max_tokens = 1000\n"
        "\n[[pass]]\n"
        'name = "translate"\n'
        'mode = "chunk"\n'
        'instruction = "Translate."'
    )
    chunks = with_usage(stream_chunks("Hello."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(config_path), "--cache-dir", str(tmp_path / "cache"), "--no-cache"],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    assert len(http.requests) == 1


def test_version_flag_exits() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--version"])


def test_parse_arguments_reports_invalid_flags() -> None:
    result = cli.parse_arguments(["--nope"]).run()
    assert isinstance(result, Err)
    assert "invalid arguments" in describe(result.error)


def test_parse_arguments_reraises_version_exit() -> None:
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--version"]).run()


def test_main_check_config_prints_report_without_http(tmp_path: Path) -> None:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        "[api]\n"
        'base_url = "http://endpoint.test/v1"\n'
        'api_key = "secret-key"\n'
        'model = "m"\n'
        "\n[[pass]]\n"
        'name = "translate"\n'
        'mode = "chunk"\n'
        'instruction = "Translate."'
    )
    http = FakeHttp([])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(config_path), "--check-config"],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    report = stdout.getvalue()
    assert "api.base_url: http://endpoint.test/v1" in report
    assert "api.model: m" in report
    assert "api.api_key: secr...ey" in report
    assert "pass 1/1 [translate]: mode=chunk" in report
    assert http.requests == []


def test_main_check_config_reports_config_errors(tmp_path: Path) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(tmp_path / "missing.toml"), "--check-config"],
        {},
        io.StringIO(""),
        stdout,
        stderr,
        time.time,
        FakeHttp([]).open,
    ).run()
    assert code == 2
    assert "l2l: config file not found" in stderr.getvalue()


def test_main_dry_run_prints_plan_without_http(tmp_path: Path) -> None:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        "[api]\n"
        'base_url = "http://endpoint.test/v1"\n'
        'api_key = "key"\n'
        'model = "m"\n'
        "\n[[pass]]\n"
        'name = "translate"\n'
        'mode = "chunk"\n'
        'instruction = "Translate."'
    )
    http = FakeHttp([])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(config_path), "--dry-run"],
        {},
        io.StringIO("你好。\n\n世界。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    plan = stdout.getvalue()
    assert "source: 8 character(s), 2 paragraph(s), ~7 tokens" in plan
    assert "pass 1/1 [translate]: mode=chunk, 1 unit(s)" in plan
    assert "cache key " in plan
    assert http.requests == []


def test_main_no_stream_flag_forces_plain_responses(tmp_path: Path) -> None:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        "[api]\n"
        'base_url = "http://endpoint.test/v1"\n'
        'api_key = "key"\n'
        'model = "m"\n'
        "max_tokens = 1000\n"
        "\n[[pass]]\n"
        'name = "translate"\n'
        'mode = "chunk"\n'
        'instruction = "Translate."'
    )
    body = {
        "choices": [{"message": {"content": "Plain."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(config_path), "--no-stream", "--no-cache"],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Plain.\n"
    assert json.loads(http.requests[0].data)["stream"] is False


def test_main_cache_prune_removes_old_entries(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    old = cache_directory / "old.txt"
    old.write_text("stale", encoding="utf-8")
    fresh = cache_directory / "fresh.txt"
    fresh.write_text("keep", encoding="utf-8")
    month_ago = time.time() - 40 * 86400
    os.utime(old, (month_ago, month_ago))

    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(tmp_path / "unused.toml"),
            "--cache-prune",
            "30",
            "--cache-dir",
            str(cache_directory),
        ],
        {},
        io.StringIO(""),
        stdout,
        stderr,
        time.time,
        FakeHttp([]).open,
    ).run()
    assert code == 0
    assert not old.exists()
    assert fresh.exists()
    assert "pruned 1 cache entry" in stderr.getvalue()


def test_main_cache_prune_rejects_non_positive_days(tmp_path: Path) -> None:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        ["--cache-prune", "0", "--cache-dir", str(tmp_path)],
        {},
        io.StringIO(""),
        stdout,
        stderr,
        time.time,
        FakeHttp([]).open,
    ).run()
    assert code == 2
    assert "positive number of days" in stderr.getvalue()


def test_cli_exits_with_program_code(monkeypatch: pytest.MonkeyPatch) -> None:
    exits: list[int] = []
    monkeypatch.setattr(cli, "main", lambda *args: io_pure(3))
    monkeypatch.setattr(sys, "exit", exits.append)
    cli.cli()
    assert exits == [3]


def test_cli_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    exits: list[int] = []
    monkeypatch.setattr(cli, "main", interrupted)
    monkeypatch.setattr(sys, "exit", exits.append)
    cli.cli()
    assert exits == [130]


def test_cli_handles_broken_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise BrokenPipeError

    exits: list[int] = []
    monkeypatch.setattr(cli, "main", broken)
    monkeypatch.setattr(os, "open", lambda path, flags: -1)
    monkeypatch.setattr(os, "dup2", lambda fd, target: None)
    monkeypatch.setattr(sys, "exit", exits.append)
    cli.cli()
    assert exits == [141]


def test_parse_args_defaults() -> None:
    arguments = cli.parse_args(["cfg.toml", "--no-cache", "-v"])
    assert arguments.config == "cfg.toml"
    assert arguments.no_cache
    assert not arguments.ensure_paragraphs
    assert arguments.verbose
    assert arguments.cache_dir is None
    assert arguments.log_keep == 30

    arguments = cli.parse_args(["cfg.toml", "--ensure-paragraphs"])
    assert arguments.ensure_paragraphs

    arguments = cli.parse_args(["cfg.toml", "--log-keep", "0"])
    assert arguments.log_keep == 0


def test_plan_report_lists_units_and_keys() -> None:
    console, _ = make_console()
    ctx = make_context(console, FakeHttp([]).open)
    text = "你好。\n\n世界。"
    report = plan_report(ctx, (chunk_pass(), paragraph_pass()), text)
    lines = report.splitlines()
    assert lines[0] == "source: 8 character(s), 2 paragraph(s), ~7 tokens"
    assert lines[1] == "pass 1/2 [translate]: mode=chunk, 1 unit(s)"
    assert "unit 1/1: ~7 source tokens, cache key " in lines[2]
    assert lines[3] == "pass 2/2 [translate]: mode=paragraph, 2 unit(s)"
    assert "unit 1/2:" in lines[4] and "unit 2/2:" in lines[5]


def test_plan_report_reports_analysis_pass() -> None:
    console, _ = make_console()
    ctx = make_context(console, FakeHttp([]).open)
    analysis = PassDefinition("prep", "Brief.", "analysis", {}, None, False)
    report = plan_report(ctx, (analysis,), "你好。")
    assert "pass 1/1 [prep]: mode=analysis, 1 call with the whole document" in report


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


def test_main_log_keep_prunes_old_logs(tmp_path: Path) -> None:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        "[api]\n"
        'base_url = "http://endpoint.test/v1"\n'
        'api_key = "key"\n'
        'model = "m"\n'
        "\n[[pass]]\n"
        'name = "translate"\n'
        'mode = "chunk"\n'
        'instruction = "Translate."'
    )
    cache_directory = tmp_path / "cache"
    logs = cache_directory / "logs"
    logs.mkdir(parents=True)
    ancient = logs / "ancient.log"
    ancient.write_text("old", encoding="utf-8")
    month_ago = time.time() - 40 * 86400
    os.utime(ancient, (month_ago, month_ago))

    chunks = with_usage(stream_chunks("Hello."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(config_path),
            "--cache-dir",
            str(cache_directory),
            "--no-cache",
            "--log-keep",
            "30",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert not ancient.exists()
    assert len(list(logs.glob("*.log"))) == 1

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from fakes import (
    FakeHttp,
    FakeStreamResponse,
    chunk_pass,
    make_console,
    make_context,
    stream_chunks,
    with_usage,
)

from l2l.ascii import ascii_collapse, ascii_fix_llm, stubborn_paragraphs
from l2l.errors import TranslationError, fail_http
from l2l.monads import Ok, Result
from l2l.pipeline import run_pipeline
from l2l.plans import ascii_drop_warning
from l2l.settings import build_settings
from l2l.text import AsciiDrop, Usage

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def test_ascii_drop_warning_mentions_paragraph_sample_and_attempts() -> None:
    warning = ascii_drop_warning(AsciiDrop(index=1, sample="中", attempts=3))
    assert warning == (
        "l2l: ascii: warning: paragraph 2 still contained non-ASCII "
        "characters (中) after 3 LLM attempts; dropping them"
    )


def test_ascii_collapse_requires_majority_and_floor() -> None:
    assert not ascii_collapse(1, 1, 4)
    assert not ascii_collapse(3, 5, 4)
    assert not ascii_collapse(2, 5, 4)
    assert ascii_collapse(4, 5, 4)
    assert ascii_collapse(5, 5, 4)


def test_stubborn_paragraphs_skip_mechanically_fixable_text() -> None:
    character_map = build_settings().ascii_character_map
    stubborn = stubborn_paragraphs(("Café.", "你好。", "Plain."), character_map)
    assert stubborn == ("你好。",)


def test_ascii_enforcement_reports_a_failing_repair_call() -> None:
    console, stderr = make_console()
    requests: list[Any] = []

    def open_once_then_fail(
        request: Any, timeout: float
    ) -> Result[Any, TranslationError]:
        requests.append(request)
        if len(requests) == 1:
            return Ok(FakeStreamResponse(with_usage(stream_chunks("Hi 中"), USAGE)))

        return fail_http("unreachable", "down")

    run_context = make_context(console, open_once_then_fail)
    code = run_pipeline(
        run_context, (chunk_pass(ascii_output=True),), "文本。", 0.0, io.StringIO()
    ).run()
    assert code == 1
    logged = stderr.getvalue()
    assert "pass [translate] ascii enforcement failed" in logged
    assert "could not reach endpoint: down" in logged
    assert len(requests) == 4


def test_ascii_enforcement_fails_fast_when_most_paragraphs_need_llm_repair() -> None:
    console, stderr = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(stream_chunks("一。\n\n二。\n\n三。\n\n四。\n\n五。"), USAGE)
            )
        ]
    )
    run_context = make_context(console, http.open, target_language="Zorblaxian")
    code = run_pipeline(
        run_context,
        (chunk_pass(ascii_output=True),),
        "1。\n\n2。\n\n3。\n\n4。\n\n5。",
        0.0,
        io.StringIO(),
    ).run()
    assert code == 1
    assert len(http.requests) == 1
    logged = stderr.getvalue()
    assert "5 of 5 paragraph(s) would need LLM ASCII repair" in logged
    assert "untranslated text instead of translating" in logged


def test_ascii_enforcement_repairs_a_minority_of_stubborn_paragraphs() -> None:
    console, _ = make_console()
    http = FakeHttp(
        [
            FakeStreamResponse(
                with_usage(
                    stream_chunks("Hi 中。\n\nYo 中。\n\nHey 中。\n\nCafé.\n\nPlain."),
                    USAGE,
                )
            ),
            FakeStreamResponse(with_usage(stream_chunks("Hi there."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Yo there."), USAGE)),
            FakeStreamResponse(with_usage(stream_chunks("Hey there."), USAGE)),
        ]
    )
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context,
        (chunk_pass(ascii_output=True),),
        "一。\n\n二。\n\n三。\n\n四。\n\n五。",
        0.0,
        stdout,
    ).run()
    assert code == 0
    assert stdout.getvalue() == (
        "Hi there.\n\nYo there.\n\nHey there.\n\nCafe.\n\nPlain.\n"
    )
    assert len(http.requests) == 4


def test_run_pipeline_enforces_ascii_mechanically() -> None:
    console, _stderr = make_console()
    chunks = with_usage(stream_chunks("Café — déjà."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    run_context = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        run_context, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Cafe - deja.\n"
    assert len(http.requests) == 1


def test_ascii_repair_cache_keys_differ_by_target_language(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hi."), USAGE))])

    def repair(language: str) -> None:
        console, _ = make_console()
        run_context = make_context(
            console,
            http.open,
            cache_directory=str(cache_directory),
            use_cache=True,
            target_language=language,
        )
        result = ascii_fix_llm(
            run_context, chunk_pass(ascii_output=True), "你好。", "Hi 中", Usage()
        ).run()
        assert isinstance(result, Ok)

    repair("English")
    repair("French")
    assert len(http.requests) == 2

    repair("French")
    assert len(http.requests) == 2

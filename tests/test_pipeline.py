import io
import json
import time
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

from zh2en import cli
from zh2en.config import PassDefinition
from zh2en.http import chat
from zh2en.monads import IO, Err, Ok, Result, fold_io, io_pure, io_result
from zh2en.pipeline import analyze_document, run_pipeline
from zh2en.text import Usage

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def chunk_pass(name: str = "translate", ascii_output: bool = False) -> PassDefinition:
    return PassDefinition(name, "T.", "chunk", {}, None, ascii_output)


def paragraph_pass(name: str = "translate") -> PassDefinition:
    return PassDefinition(name, "T.", "paragraph", {}, None, False)


def test_fold_io_handles_thousands_of_items() -> None:
    def step(pair: tuple[int, None], _: int) -> IO[Result[tuple[int, None], str]]:
        return io_pure(Ok((pair[0] + 1, None)))

    outcome = fold_io(range(5000), step, Ok((0, None))).run()
    assert isinstance(outcome, Ok)
    assert outcome.value[0] == 5000


def test_fold_io_short_circuits_on_error() -> None:
    calls: list[int] = []

    def step(
        pair: tuple[int, None], item: int
    ) -> IO[Result[tuple[int, None], str]]:
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
    content, usage = result.value
    assert content == "Hi"
    assert usage == Usage(5, 6, 0.2)


def test_chat_streamed_filters_think_prefix() -> None:
    console, stderr = make_console()
    chunks = stream_chunks("<think>h</think>", "Body")
    http = FakeHttp([FakeStreamResponse(with_usage(chunks, USAGE))])
    ctx = make_context(console, http.open)
    result = chat(ctx, "sys", "user text", "m", {}, Usage()).run()
    assert isinstance(result, Ok)
    assert result.value[0] == "Body"


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
    assert result.value[0] == "Plain"
    payload = json.loads(http.requests[0].data)
    assert payload["stream"] is False


def test_chat_rejects_oversized_request() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    ctx = make_context(console, http.open, max_tokens=1)
    result = chat(ctx, "s", "u", "m", {}, Usage()).run()
    assert isinstance(result, Err)
    assert "over the 1-token budget" in result.error
    assert http.requests == []


def test_run_pipeline_translates() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert "TOTAL" in stderr.getvalue()


def test_run_pipeline_reports_unit_failure() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> Result[Any, str]:
        return Err("could not reach endpoint: down")

    ctx = make_context(console, open_fail)
    code = run_pipeline(
        ctx, (chunk_pass(),), "你好。", 0.0, io.StringIO()
    ).run()
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
        code = run_pipeline(
            ctx, (chunk_pass(),), "你好。", 0.0, stdout
        ).run()
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
    assert "requires analysis in" in result.error
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
    ctx = make_context(console, http.open, ensure_paragraphs=True)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
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
    ctx = make_context(console, http.open, ensure_paragraphs=True)
    stdout = io.StringIO()
    code = run_pipeline(
        ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
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
    code = run_pipeline(
        ctx, (paragraph_pass(),), "你好。", 0.0, stdout
    ).run()
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
        ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert len(http.requests) == 2
    retry_user = json.loads(http.requests[1].data)["messages"][1]["content"]
    assert "has 1 paragraph(s) but the source has 2" in retry_user


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
    code = run_pipeline(
        ctx, (paragraph_pass(),), "嗯。", 0.0, stdout
    ).run()
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


def test_main_end_to_end_with_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "zh2en.toml"
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
    monkeypatch.setattr(cli, "urllib_open", http.open)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(config_path), "--cache-dir", str(tmp_path / "cache"), "--no-cache"],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    assert len(http.requests) == 1


def test_version_flag_exits() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args(["--version"])


def test_parse_args_defaults() -> None:
    arguments = cli.parse_args(["cfg.toml", "--no-cache", "-v"])
    assert arguments.config == "cfg.toml"
    assert arguments.no_cache
    assert not arguments.ensure_paragraphs
    assert arguments.verbose
    assert arguments.cache_dir is None

    arguments = cli.parse_args(["cfg.toml", "--ensure-paragraphs"])
    assert arguments.ensure_paragraphs

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

import zh2en as z

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def chunk_pass(name: str = "translate", ascii_output: bool = False) -> z.PassDefinition:
    return z.PassDefinition(name, "T.", "chunk", {}, None, ascii_output)


def test_fold_io_handles_thousands_of_items() -> None:
    def step(pair: tuple[int, None], _: int) -> z.IO[z.Result[tuple[int, None], str]]:
        return z.io_pure(z.Ok((pair[0] + 1, None)))

    outcome = z.fold_io(range(5000), step, z.Ok((0, None))).run()
    assert isinstance(outcome, z.Ok)
    assert outcome.value[0] == 5000


def test_fold_io_short_circuits_on_error() -> None:
    calls: list[int] = []

    def step(
        pair: tuple[int, None], item: int
    ) -> z.IO[z.Result[tuple[int, None], str]]:
        calls.append(item)
        if item == 1:
            return z.io_result(z.Err("boom"))

        return z.io_pure(z.Ok((pair[0] + 1, None)))

    outcome = z.fold_io(range(4), step, z.Ok((0, None))).run()
    assert isinstance(outcome, z.Err)
    assert outcome.error == "boom"
    assert calls == [0, 1]


def test_chat_streamed_reports_usage() -> None:
    console, stderr = make_console()
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hi"), USAGE))])
    ctx = make_context(console, http.open)
    result = z.chat(ctx, "sys", "user text", "m", {}, z.Usage()).run()
    assert isinstance(result, z.Ok)
    content, usage = result.value
    assert content == "Hi"
    assert usage == z.Usage(5, 6, 0.2)


def test_chat_streamed_filters_think_prefix() -> None:
    console, stderr = make_console()
    chunks = stream_chunks("<think>h</think>", "Body")
    http = FakeHttp([FakeStreamResponse(with_usage(chunks, USAGE))])
    ctx = make_context(console, http.open)
    result = z.chat(ctx, "sys", "user text", "m", {}, z.Usage()).run()
    assert isinstance(result, z.Ok)
    assert result.value[0] == "Body"


def test_chat_plain_when_stream_disabled() -> None:
    console, stderr = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    ctx = make_context(console, http.open)
    result = z.chat(ctx, "s", "u", "m", {"stream": False}, z.Usage()).run()
    assert isinstance(result, z.Ok)
    assert result.value[0] == "Plain"
    payload = json.loads(http.requests[0].data)
    assert payload["stream"] is False


def test_chat_rejects_oversized_request() -> None:
    console, stderr = make_console()
    http = FakeHttp([])
    ctx = make_context(console, http.open, max_tokens=1)
    result = z.chat(ctx, "s", "u", "m", {}, z.Usage()).run()
    assert isinstance(result, z.Err)
    assert "over the 1-token budget" in result.error
    assert http.requests == []


def test_run_pipeline_translates() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Hello.\n\nWorld."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = z.run_pipeline(
        ctx, (chunk_pass(),), "你好。\n\n世界。", 0.0, stdout, time.time
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Hello.\n\nWorld.\n"
    assert "TOTAL" in stderr.getvalue()


def test_run_pipeline_reports_unit_failure() -> None:
    console, stderr = make_console()

    def open_fail(request: Any, timeout: float) -> z.Result[Any, str]:
        return z.Err("could not reach endpoint: down")

    ctx = make_context(console, open_fail)
    code = z.run_pipeline(
        ctx, (chunk_pass(),), "你好。", 0.0, io.StringIO(), time.time
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
        code = z.run_pipeline(
            ctx, (chunk_pass(),), "你好。", 0.0, stdout, time.time
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
    analysis_pass = z.PassDefinition("prep", "Brief.", "analysis", {}, None, False)
    result = z.analyze_document(ctx, analysis_pass, "一。二。三。四。", z.Usage()).run()
    assert isinstance(result, z.Err)
    assert "requires analysis in" in result.error
    assert http.requests == []


def test_run_pipeline_enforces_ascii_mechanically() -> None:
    console, stderr = make_console()
    chunks = with_usage(stream_chunks("Café — déjà."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    ctx = make_context(console, http.open)
    stdout = io.StringIO()
    code = z.run_pipeline(
        ctx, (chunk_pass(ascii_output=True),), "文本。", 0.0, stdout, time.time
    ).run()
    assert code == 0
    assert stdout.getvalue() == "Cafe - deja.\n"
    assert len(http.requests) == 1


def test_main_empty_stdin_succeeds_without_config() -> None:
    code = z.main(
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
    monkeypatch.setattr(z, "urllib_open", http.open)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = z.main(
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
        z.parse_args(["--version"])


def test_parse_args_defaults() -> None:
    arguments = z.parse_args(["cfg.toml", "--no-cache", "-v"])
    assert arguments.config == "cfg.toml"
    assert arguments.no_cache
    assert arguments.verbose
    assert arguments.cache_dir is None

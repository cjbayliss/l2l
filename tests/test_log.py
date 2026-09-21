import io
import re
import time
from collections.abc import Callable
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
from zh2en.effects import RunLog, log_error, run_log_write
from zh2en.http import chat
from zh2en.monads import Err, Ok, Result
from zh2en.text import Usage

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}

CONFIG_TEXT = (
    "[api]\n"
    'base_url = "http://endpoint.test/v1"\n'
    'api_key = "secret-key"\n'
    'model = "m"\n'
    "max_tokens = 1000\n"
    "\n[[pass]]\n"
    'name = "translate"\n'
    'mode = "chunk"\n'
    'instruction = "Translate."'
)


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "zh2en.toml"
    config_path.write_text(CONFIG_TEXT)
    return config_path


def run_zh2en(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    open_http: Callable[[Any, float], Result[Any, str]],
    flags: list[str],
) -> tuple[int, str, str]:
    monkeypatch.setattr(cli, "urllib_open", open_http)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [str(config_path), "--cache-dir", str(tmp_path / "cache"), *flags],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def test_run_log_captures_stream_request_and_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = with_usage(stream_chunks("Hello."), USAGE)
    http = FakeHttp([FakeStreamResponse(chunks)])
    code, _, _ = run_zh2en(
        write_config(tmp_path), tmp_path, monkeypatch, http.open, ["--no-cache"]
    )
    assert code == 0

    logs = list((tmp_path / "cache" / "logs").glob("*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "REQUEST (stream)" in content
    assert '"model": "m"' in content
    assert '"stream": true' in content
    assert "secret-key" not in content
    assert 'data: {"choices": [{"delta": {"content": "Hello."}}]}' in content


def test_run_log_captures_plain_request_and_response(tmp_path: Path) -> None:
    log_path = tmp_path / "run.log"
    console, _ = make_console()
    body = {
        "choices": [{"message": {"content": "Plain"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    ctx = make_context(console, http.open, log=RunLog(str(log_path)))
    result = chat(ctx, "sys", "user", "m", {"stream": False}, Usage()).run()
    assert isinstance(result, Ok)
    content = log_path.read_text(encoding="utf-8")
    assert "REQUEST" in content
    assert '"stream": false' in content
    assert "RESPONSE" in content
    assert '"Plain"' in content


def test_run_log_captures_http_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def open_fail(request: Any, timeout: float) -> Result[Any, str]:
        return Err("could not reach endpoint: down")

    code, _, _ = run_zh2en(
        write_config(tmp_path), tmp_path, monkeypatch, open_fail, ["--no-cache"]
    )
    assert code == 1

    logs = list((tmp_path / "cache" / "logs").glob("*.log"))
    assert len(logs) == 1
    content = logs[0].read_text(encoding="utf-8")
    assert "REQUEST (stream)" in content
    assert "ERROR" in content
    assert "could not reach endpoint: down" in content


def test_show_log_path_prints_log_path_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    code, _, stderr = run_zh2en(
        write_config(tmp_path), tmp_path, monkeypatch, http.open, ["--no-cache", "-l"]
    )
    assert code == 0

    match = re.search(r"zh2en: log: (\S+)", stderr)
    assert match is not None
    assert Path(match.group(1)).exists()


def test_show_log_path_short_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    code, _, stderr = run_zh2en(
        write_config(tmp_path), tmp_path, monkeypatch, http.open, ["-l"]
    )
    assert code == 0
    assert "zh2en: log: " in stderr


def test_verbose_alone_does_not_print_log_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    code, _, stderr = run_zh2en(
        write_config(tmp_path), tmp_path, monkeypatch, http.open, ["--no-cache", "-v"]
    )
    assert code == 0
    assert "zh2en: log:" not in stderr


def test_without_verbose_log_path_not_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    http = FakeHttp([FakeStreamResponse(with_usage(stream_chunks("Hello."), USAGE))])
    code, _, stderr = run_zh2en(
        write_config(tmp_path), tmp_path, monkeypatch, http.open, ["--no-cache"]
    )
    assert code == 0
    assert "zh2en: log:" not in stderr


def test_run_log_write_ignores_empty_path() -> None:
    run_log_write(RunLog(""), "x").run()
    log_error(RunLog(""), "boom").run()


def test_parse_args_show_log_path() -> None:
    assert cli.parse_args(["cfg.toml", "-l"]).show_log_path
    assert cli.parse_args(["cfg.toml", "--show-log-path"]).show_log_path
    assert not cli.parse_args(["cfg.toml"]).show_log_path

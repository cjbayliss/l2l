import io
import time
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeHttp, FakePlainResponse, FakeStreamResponse, stream_chunks

from l2l import cli
from l2l.errors import TranslationError
from l2l.monads import Result

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
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(CONFIG_TEXT)
    return config_path


def run_l2l(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    stdin: str = "你好。",
) -> tuple[int, str, str]:
    def open_http(request: Any, timeout: float) -> Result[Any, TranslationError]:
        raise AssertionError("endpoint must not be contacted")

    monkeypatch.setattr(cli, "urllib_open", open_http)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        arguments,
        {},
        io.StringIO(stdin),
        stdout,
        stderr,
        time.time,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def test_empty_stdin_exits_zero_without_config(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, pytest.MonkeyPatch(), [str(tmp_path)], "")
    assert code == 0
    assert stdout == ""
    assert stderr == ""


def test_invalid_arguments_exit_two(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, pytest.MonkeyPatch(), ["--no-such-flag"])
    assert code == 2
    assert stdout == ""
    assert "invalid arguments" in stderr


def test_missing_config_fails_with_error(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, pytest.MonkeyPatch(), [str(tmp_path / "absent.toml")]
    )
    assert code == 2
    assert stdout == ""
    assert "not found" in stderr


def test_check_config_prints_report_and_exits(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, pytest.MonkeyPatch(), [str(write_config(tmp_path)), "--check-config"]
    )
    assert code == 0
    assert stderr == ""
    assert "api.base_url: http://endpoint.test/v1" in stdout
    assert "api.api_key: secr...ey" in stdout
    assert "pass 1/1 [translate]: mode=chunk" in stdout


def test_check_config_reports_setup_errors(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[api]\nunknown_key = 1\n")
    code, stdout, stderr = run_l2l(
        tmp_path, pytest.MonkeyPatch(), [str(broken), "--check-config"]
    )
    assert code == 2
    assert stdout == ""
    assert "[api]: unknown key(s): unknown_key" in stderr


def test_dry_run_prints_plan_without_calling_endpoint(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, pytest.MonkeyPatch(), [str(write_config(tmp_path)), "--dry-run"]
    )
    assert code == 0
    assert stderr == ""
    assert "source: 3 character(s), 1 paragraph(s)" in stdout
    assert "pass 1/1 [translate]: mode=chunk, 1 unit(s)" in stdout
    assert "unit 1/1" in stdout
    assert "cache key" in stdout


def test_cache_prune_removes_old_entries(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    entry = cache_dir / "stale.txt"
    entry.write_text("old")
    import os

    stale = time.time() - 30 * 86400
    os.utime(entry, (stale, stale))
    fresh = cache_dir / "fresh.txt"
    fresh.write_text("new")

    code, _, stderr = run_l2l(
        tmp_path,
        pytest.MonkeyPatch(),
        [str(tmp_path), "--cache-dir", str(cache_dir), "--cache-prune", "7"],
    )
    assert code == 0
    assert "pruned 1 cache entry" in stderr
    assert not entry.exists()
    assert fresh.exists()


def test_cache_prune_requires_positive_days(tmp_path: Path) -> None:
    code, _, stderr = run_l2l(
        tmp_path,
        pytest.MonkeyPatch(),
        [str(tmp_path), "--cache-dir", str(tmp_path), "--cache-prune", "0"],
    )
    assert code == 2
    assert "positive number of days" in stderr


def test_end_to_end_translation_writes_stdout(tmp_path: Path) -> None:
    body = {
        "choices": [{"message": {"content": "Hello."}}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 2, "cost": 0.01},
    }
    chunks = [
        *stream_chunks("Hello."),
        {"choices": [{"delta": {}}], "usage": body["usage"]},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "urllib_open", http.open)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
    ).run()
    assert code == 0
    assert stdout.getvalue().strip() == "Hello."
    assert "Starting pass 1/1 [translate]..." in stderr.getvalue()
    assert "Done: " in stderr.getvalue()
    assert "TOTAL" in stderr.getvalue()


def test_plain_call_end_to_end(tmp_path: Path) -> None:
    body = {
        "choices": [{"message": {"content": "Plain."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "urllib_open", http.open)
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--no-stream",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
    ).run()
    assert code == 0
    assert stdout.getvalue().strip() == "Plain."


def test_cache_roundtrip_across_runs(tmp_path: Path) -> None:
    body = {
        "choices": [{"message": {"content": "Hello."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp(
        [
            FakeStreamResponse(
                [
                    *stream_chunks("Hello."),
                    {"choices": [{"delta": {}}], "usage": body["usage"]},
                ]
            )
        ]
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cli, "urllib_open", http.open)
    flags = [str(write_config(tmp_path)), "--cache-dir", str(tmp_path / "cache")]

    first_stdout, first_stderr = io.StringIO(), io.StringIO()
    first = cli.main(
        flags, {}, io.StringIO("你好。"), first_stdout, first_stderr, time.time
    ).run()
    assert first == 0
    assert first_stdout.getvalue().strip() == "Hello."

    # Second run with a refusing endpoint: success proves the cache served it.
    def refusing_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
        raise AssertionError("cache should have served the second run")

    monkeypatch.setattr(cli, "urllib_open", refusing_open)
    second_stdout, second_stderr = io.StringIO(), io.StringIO()
    second = cli.main(
        flags, {}, io.StringIO("你好。"), second_stdout, second_stderr, time.time
    ).run()
    assert second == 0
    assert second_stdout.getvalue().strip() == "Hello."
    # Zero token usage on the second run proves the cache served it.
    assert "prompt=0, completion=0" in second_stderr.getvalue()

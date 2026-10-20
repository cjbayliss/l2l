import io
import json
import os
import pty
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fakes import (
    FakeHttp,
    FakePlainResponse,
    FakeStreamResponse,
    stream_chunks,
    with_usage,
)

from l2l import cli, keys
from l2l.console import toggle_verbose
from l2l.errors import FileError, TranslationError
from l2l.monads import IO, Err, Just, Ok, Result, io_result

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

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(CONFIG_TEXT)
    return config_path


def run_l2l(
    tmp_path: Path,
    arguments: list[str],
    stdin: str = "你好。",
) -> tuple[int, str, str]:
    def open_http(request: Any, timeout: float) -> Result[Any, TranslationError]:
        raise AssertionError("endpoint must not be contacted")

    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        arguments,
        {},
        io.StringIO(stdin),
        stdout,
        stderr,
        time.time,
        open_http,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def test_empty_stdin_exits_zero_without_config(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, [str(tmp_path)], "")
    assert code == 0
    assert stdout == ""
    assert stderr == ""


def test_invalid_arguments_exit_two(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, ["--no-such-flag"])
    assert code == 2
    assert stdout == ""
    assert "invalid arguments" in stderr


def test_missing_config_fails_with_error(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, [str(tmp_path / "absent.toml")])
    assert code == 2
    assert stdout == ""
    assert "not found" in stderr


def test_check_config_prints_report_and_exits(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, [str(write_config(tmp_path)), "--check-config"]
    )
    assert code == 0
    assert stderr == ""
    assert "api.base_url: http://endpoint.test/v1" in stdout
    assert "api.api_key: secr...ey" in stdout
    assert "pass 1/1 [translate]: mode=chunk" in stdout


def test_check_config_reports_setup_errors(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[api]\nunknown_key = 1\n")
    code, stdout, stderr = run_l2l(tmp_path, [str(broken), "--check-config"])
    assert code == 2
    assert stdout == ""
    assert "[api]: unknown key(s): unknown_key" in stderr


def test_dry_run_prints_plan_without_calling_endpoint(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, [str(write_config(tmp_path)), "--dry-run"])
    assert code == 0
    assert stderr == ""
    assert "source: 3 character(s), 1 paragraph(s)" in stdout
    assert "pass 1/1 [translate]: mode=chunk, 1 unit(s)" in stdout
    assert "unit 1/1" in stdout
    assert "cache key" in stdout


def test_cache_prune_removes_old_entries(tmp_path: Path) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    entry = cache_directory / "stale.txt"
    entry.write_text("old")
    import os

    stale = time.time() - 30 * 86400
    os.utime(entry, (stale, stale))
    fresh = cache_directory / "fresh.txt"
    fresh.write_text("new")

    code, _, stderr = run_l2l(
        tmp_path,
        [str(tmp_path), "--cache-dir", str(cache_directory), "--cache-prune", "7"],
    )
    assert code == 0
    assert "pruned 1 cache entry" in stderr
    assert not entry.exists()
    assert fresh.exists()


def test_cache_prune_requires_positive_days(tmp_path: Path) -> None:
    code, _, stderr = run_l2l(
        tmp_path,
        [str(tmp_path), "--cache-dir", str(tmp_path), "--cache-prune", "0"],
    )
    assert code == 2
    assert "positive number of days" in stderr


def test_cache_prune_reports_scan_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()

    monkeypatch.setattr(
        cli,
        "cache_entry_paths",
        lambda directory: io_result(
            Err(FileError(directory, "list entries in", "[Errno 13] Permission denied"))
        ),
    )

    code, _, stderr = run_l2l(
        tmp_path,
        [str(tmp_path), "--cache-dir", str(cache_directory), "--cache-prune", "7"],
    )
    assert code == 2
    assert "could not list entries in" in stderr
    assert "pruned" not in stderr


def test_cache_prune_reports_age_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    (cache_directory / "stale.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        cli,
        "file_age",
        lambda path, now_value: io_result(
            Err(FileError(path, "read the age of", "[Errno 13] Permission denied"))
        ),
    )

    code, _, stderr = run_l2l(
        tmp_path,
        [str(tmp_path), "--cache-dir", str(cache_directory), "--cache-prune", "7"],
    )
    assert code == 2
    assert "could not read the age of" in stderr
    assert "pruned" not in stderr


def test_cache_prune_reports_removal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    stale = cache_directory / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    stale_time = time.time() - 30 * 86400
    os.utime(stale, (stale_time, stale_time))

    monkeypatch.setattr(
        cli,
        "remove_file",
        lambda path: io_result(
            Err(FileError(path, "remove", "[Errno 13] Permission denied"))
        ),
    )

    code, _, stderr = run_l2l(
        tmp_path,
        [str(tmp_path), "--cache-dir", str(cache_directory), "--cache-prune", "7"],
    )
    assert code == 2
    assert "could not remove" in stderr
    assert "pruned" not in stderr


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
        http.open,
    ).run()
    assert code == 0
    assert stdout.getvalue().strip() == "Hello."
    assert "Starting pass 1/1 [translate]..." in stderr.getvalue()
    assert "Done: " in stderr.getvalue()
    assert "TOTAL" in stderr.getvalue()


def test_strict_validation_failure_ends_the_cli_run(tmp_path: Path) -> None:
    bad = "One.\n\nTwo."
    chunks = [
        *stream_chunks(bad),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--ensure-paragraphs",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 1
    assert stdout.getvalue() == ""
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "pass [translate] failed" in logged
    assert "still failed validation after 2 repair attempt(s)" in logged
    assert len(http.requests) == 6


def test_best_effort_flag_keeps_the_unvalidated_reply(tmp_path: Path) -> None:
    bad = "One.\n\nTwo."
    chunks = [
        *stream_chunks(bad),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--ensure-paragraphs",
            "--best-effort",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert stdout.getvalue() == bad + "\n"
    logged = stderr.getvalue()
    assert "failed validation 2 time(s)" in logged
    assert "Keeping the last reply, uncached" in logged
    assert len(http.requests) == 6


def test_plain_call_end_to_end(tmp_path: Path) -> None:
    body = {
        "choices": [{"message": {"content": "Plain."}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0},
    }
    http = FakeHttp([FakePlainResponse(body)])
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
        http.open,
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
    flags = [str(write_config(tmp_path)), "--cache-dir", str(tmp_path / "cache")]

    first_stdout, first_stderr = io.StringIO(), io.StringIO()
    first = cli.main(
        flags,
        {},
        io.StringIO("你好。"),
        first_stdout,
        first_stderr,
        time.time,
        http.open,
    ).run()
    assert first == 0
    assert first_stdout.getvalue().strip() == "Hello."

    def refusing_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
        raise AssertionError("cache should have served the second run")

    second_stdout, second_stderr = io.StringIO(), io.StringIO()
    second = cli.main(
        flags,
        {},
        io.StringIO("你好。"),
        second_stdout,
        second_stderr,
        time.time,
        refusing_open,
    ).run()
    assert second == 0
    assert second_stdout.getvalue().strip() == "Hello."
    assert "prompt=0, completion=0" in second_stderr.getvalue()


@pytest.mark.skipif(os.name == "nt", reason="requires a POSIX TTY")
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_tab_key_toggles_verbose_during_a_live_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_master, output_slave = pty.openpty()
    toggle_master, toggle_slave = pty.openpty()
    in_request = threading.Event()
    toggled = threading.Event()

    def watching_toggle(console: Any) -> Any:
        toggled.set()
        return toggle_verbose(console)

    def gated_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
        in_request.set()
        assert toggled.wait(5.0), "the tab key never toggled verbose mode"
        chunks = with_usage(stream_chunks("Hello."), USAGE)
        return Ok(FakeStreamResponse(chunks))

    def feed_tab() -> None:
        if in_request.wait(5.0):
            os.write(toggle_master, b"\t")

    monkeypatch.setattr(cli, "toggle_verbose", watching_toggle)
    monkeypatch.setattr(keys, "open_tty", lambda: IO(lambda: Just(toggle_slave)))
    writer = threading.Thread(target=feed_tab, daemon=True)
    writer.start()

    stderr = os.fdopen(output_slave, "w")
    stdout = io.StringIO()
    try:
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
            gated_open,
        ).run()
    finally:
        stderr.close()
        os.close(output_master)
        os.close(toggle_master)
        os.close(toggle_slave)

    assert code == 0
    assert stdout.getvalue() == "Hello.\n"
    assert toggled.is_set()


def test_translation_request_carries_glossary_block(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        '[{"original": "南京", "translated": "Nanjing"}]', encoding="utf-8"
    )
    chunks = [
        *stream_chunks("Hello."),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--glossary",
            str(glossary),
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    payload = json.loads(http.requests[0].body.decode("utf-8"))
    system = payload["messages"][0]["content"]
    assert "Glossary (mandatory)" in system
    assert '- 南京 = "Nanjing"' in system


def run_write_glossary(
    tmp_path: Path,
    arguments: list[str],
    http: FakeHttp,
    stdin: str = "book text",
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            *arguments,
        ],
        {},
        io.StringIO(stdin),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def glossary_reply(entries: list[dict[str, Any]]) -> FakeStreamResponse:
    chunks = [
        *stream_chunks("```json\n" + json.dumps(entries, ensure_ascii=False) + "\n```"),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    return FakeStreamResponse(chunks)


def test_write_glossary_happy_path(tmp_path: Path) -> None:
    carried = tmp_path / "generic.json"
    carried.write_text(
        '[{"original": "南京", "translated": "Nanjing"}]', encoding="utf-8"
    )
    output = tmp_path / "book-glossary.json"
    http = FakeHttp(
        [
            glossary_reply(
                [
                    {
                        "original": "林黛玉",
                        "translated": "Lin Daiyu",
                        "pronouns": "she/her",
                    }
                ]
            )
        ]
    )
    code, _, stderr = run_write_glossary(
        tmp_path,
        ["--write-glossary", str(output), "--glossary", str(carried)],
        http,
    )
    assert code == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [item["original"] for item in document] == ["南京", "林黛玉"]
    assert document[1]["pronouns"] == "she/her"
    assert "wrote 2 glossary entries to %s" % output in stderr
    payload = json.loads(http.requests[0].body.decode("utf-8"))
    user_message = payload["messages"][1]["content"]
    assert payload["model"] == "m"
    assert "Target language: English" in user_message
    assert "Already decided" in user_message
    assert "book text" in user_message


def test_write_glossary_target_language_reaches_the_payload(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    http = FakeHttp([glossary_reply([{"original": "南京", "translated": "Nanjing"}])])
    code, _, _ = run_write_glossary(
        tmp_path,
        ["--write-glossary", str(output), "--target-language", "French"],
        http,
    )
    assert code == 0
    payload = json.loads(http.requests[0].body.decode("utf-8"))
    assert "Target language: French" in payload["messages"][1]["content"]


def test_write_glossary_invalid_reply_exits_one(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    chunks = [
        *stream_chunks("not json"),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    code, _, stderr = run_write_glossary(
        tmp_path, ["--write-glossary", str(output)], http
    )
    assert code == 1
    assert "not valid JSON" in stderr
    assert not output.exists()


def test_write_glossary_empty_stdin_exits_two(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    code, _, stderr = run_l2l(
        tmp_path,
        [str(write_config(tmp_path)), "--write-glossary", str(output)],
        "",
    )
    assert code == 2
    assert "requires text on stdin" in stderr
    assert not output.exists()


UPDATE_GLOSSARY_CONFIG_TEXT = CONFIG_TEXT.replace(
    '\n[[pass]]\nname = "translate"\nmode = "chunk"\ninstruction = "Translate."',
    "\n[[pass]]\n"
    'name = "glossary"\n'
    'mode = "update_glossary"\n'
    "\n[[pass]]\n"
    'name = "translate"\n'
    'mode = "chunk"\n'
    'instruction = "Translate."',
)


def write_update_glossary_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(UPDATE_GLOSSARY_CONFIG_TEXT)
    return config_path


def test_update_glossary_pass_updates_file_and_flows_into_later_passes(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        '[{"original": "南京", "translated": "Nanjing"}]', encoding="utf-8"
    )
    extraction = glossary_reply(
        [
            {"original": "南京", "translated": "Nanking"},
            {"original": "大观园", "translated": "Grand View Garden"},
        ]
    )
    translation_chunks = [
        *stream_chunks("Hello."),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([extraction, FakeStreamResponse(translation_chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_update_glossary_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--glossary",
            str(glossary),
            "--target-language",
            "French",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert stdout.getvalue().strip() == "Hello."

    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert [item["original"] for item in document] == ["南京", "大观园"]
    assert document[0]["translated"] == "Nanjing"
    assert document[1]["translated"] == "Grand View Garden"

    extraction_payload = json.loads(http.requests[0].body)
    extraction_user = extraction_payload["messages"][1]["content"]
    assert "Target language: French" in extraction_user
    assert "Already decided" in extraction_user

    translation_payload = json.loads(http.requests[1].body)
    translation_system = translation_payload["messages"][0]["content"]
    assert '- 南京 = "Nanjing"' in translation_system
    assert '- 大观园 = "Grand View Garden"' in translation_system

    logged = stderr.getvalue()
    assert "Starting pass 1/2 [glossary]..." in logged
    assert "Starting pass 2/2 [translate]..." in logged
    assert "TOTAL" in logged


def test_update_glossary_pass_writes_only_the_first_glossary_file(
    tmp_path: Path,
) -> None:
    first_glossary = tmp_path / "first.json"
    first_glossary.write_text(
        '[{"original": "南京", "translated": "Nanjing"}]', encoding="utf-8"
    )
    second_glossary = tmp_path / "second.json"
    second_glossary.write_text(
        '[{"original": "长城", "translated": "Great Wall"}]', encoding="utf-8"
    )
    extraction = glossary_reply(
        [{"original": "大观园", "translated": "Grand View Garden"}]
    )
    translation_chunks = [
        *stream_chunks("Hello."),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([extraction, FakeStreamResponse(translation_chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(write_update_glossary_config(tmp_path)),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--glossary",
            str(first_glossary),
            "--glossary",
            str(second_glossary),
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert stdout.getvalue().strip() == "Hello."

    first_document = json.loads(first_glossary.read_text(encoding="utf-8"))
    assert [item["original"] for item in first_document] == [
        "南京",
        "长城",
        "大观园",
    ]
    assert first_document[0]["translated"] == "Nanjing"
    assert first_document[2]["translated"] == "Grand View Garden"
    assert json.loads(second_glossary.read_text(encoding="utf-8")) == [
        {"original": "长城", "translated": "Great Wall"}
    ]


def test_update_glossary_pass_target_language_overrides_cli_flag(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        UPDATE_GLOSSARY_CONFIG_TEXT.replace(
            'mode = "update_glossary"',
            'mode = "update_glossary"\ntarget_language = "Chinese"',
        )
    )
    http = FakeHttp([glossary_reply([{"original": "大观园", "translated": "G"}])])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            str(config_path),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--no-cache",
            "--glossary",
            str(glossary),
            "--target-language",
            "French",
        ],
        {},
        io.StringIO("你好。"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    payload = json.loads(http.requests[0].body)
    assert "Target language: Chinese" in payload["messages"][1]["content"]


def test_update_glossary_pass_without_glossary_flag_exits_two(
    tmp_path: Path,
) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, [str(write_update_glossary_config(tmp_path)), "--dry-run"]
    )
    assert code == 2
    assert stdout == ""
    assert (
        "an update_glossary pass requires at least one --glossary file on the "
        "command line" in stderr
    )


def test_dry_run_shows_update_glossary_pass(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    code, stdout, stderr = run_l2l(
        tmp_path,
        [
            str(write_update_glossary_config(tmp_path)),
            "--dry-run",
            "--glossary",
            str(glossary),
        ],
    )
    assert code == 0
    assert (
        "pass 1/2 [glossary]: mode=update_glossary, 1 call with the whole "
        "document, target language English" in stdout
    )
    assert "pass 2/2 [translate]: mode=chunk, 1 unit(s)" in stdout


def test_update_glossary_cached_run_still_writes_the_file(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    cache_directory = tmp_path / "cache"
    extraction = glossary_reply([{"original": "大观园", "translated": "G"}])
    translation_chunks = [
        *stream_chunks("Hello."),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]

    def run() -> tuple[int, str, int, str]:
        http = FakeHttp([extraction, FakeStreamResponse(translation_chunks)])
        stdout, stderr = io.StringIO(), io.StringIO()
        code = cli.main(
            [
                str(write_update_glossary_config(tmp_path)),
                "--cache-dir",
                str(cache_directory),
                "--glossary",
                str(glossary),
            ],
            {},
            io.StringIO("你好。"),
            stdout,
            stderr,
            time.time,
            http.open,
        ).run()
        return code, stdout.getvalue(), len(http.requests), stderr.getvalue()

    first_code, first_output, first_calls, _first_stderr = run()
    second_code, second_output, second_calls, second_stderr = run()
    assert (first_code, second_code) == (0, 0)
    assert first_output == second_output == "Hello.\n"
    assert first_calls == 2
    assert second_calls == 0
    assert "Done: 0.0s, prompt=0, completion=0" in second_stderr
    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert [item["original"] for item in document] == ["大观园"]

import io
import json
import os
import sys
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

from l2l import __version__, cli, config, effects, pipeline
from l2l.errors import FileError, TranslationError, describe, fail_http
from l2l.monads import Err, Ok, Result, io_pure, io_result
from l2l.settings import Arguments

CONFIG_TEXT = (
    'target_language = "English"\n'
    "\n[api]\n"
    'base_url = "http://endpoint.test/v1"\n'
    'api_key = "secret-key"\n'
    'model = "m"\n'
    "max_tokens = 1000\n"
    "\n[[pass]]\n"
    'name = "translate"\n'
    'mode = "chunk"\n'
    'instruction = "Translate to {target_language}."'
)

USAGE = {"prompt_tokens": 5, "completion_tokens": 6, "cost": 0.2}


@pytest.fixture(autouse=True)
def isolate_user_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config,
        "user_config_path",
        lambda _environment: io_pure(str(tmp_path / "absent-config.toml")),
    )


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
    environment = {
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
    }
    code = cli.main(
        arguments,
        environment,
        io.StringIO(stdin),
        stdout,
        stderr,
        time.time,
        open_http,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def test_empty_stdin_exits_zero_without_config(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, [], "")
    assert code == 0
    assert stdout == ""
    assert stderr == ""


def test_invalid_arguments_exit_two(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, ["--no-such-flag"])
    assert code == 2
    assert stdout == ""
    assert "invalid arguments" in stderr


def test_missing_config_fails_with_error(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(tmp_path, ["-c", str(tmp_path / "absent.toml")])
    assert code == 2
    assert stdout == ""
    assert "could not read" in stderr


def test_missing_target_language_exits_two(tmp_path: Path) -> None:
    config_path = tmp_path / "no-language.toml"
    config_path.write_text(CONFIG_TEXT.replace('target_language = "English"\n', ""))
    code, stdout, stderr = run_l2l(tmp_path, ["-c", str(config_path), "--check-config"])
    assert code == 2
    assert stdout == ""
    assert (
        "target language is required; pass --target-language or set "
        "target_language in a config file" in stderr
    )


def test_cli_target_language_overrides_config(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path,
        ["-c", str(write_config(tmp_path)), "--check-config", "-l", "French"],
    )
    assert code == 0
    assert stderr == ""
    assert "target_language: French" in stdout


def test_check_config_prints_report_and_exits(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, ["-c", str(write_config(tmp_path)), "--check-config"]
    )
    assert code == 0
    assert stderr == ""
    assert "api.base_url: http://endpoint.test/v1" in stdout
    assert "api.model: m" in stdout
    assert "api.api_key: secr...ey" in stdout
    assert "target_language: English (scripts: Latin)" in stdout
    assert "pass 1/1 [translate]: mode=chunk" in stdout
    assert "target_language=English scripts=Latin" in stdout
    assert "uses_target_language_placeholder=True" in stdout


def test_check_config_reports_setup_errors(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    broken.write_text("[api]\nunknown_key = 1\n")
    code, stdout, stderr = run_l2l(tmp_path, ["-c", str(broken), "--check-config"])
    assert code == 2
    assert stdout == ""
    assert "[api]: unknown key(s): unknown_key" in stderr


def test_dry_run_prints_plan_without_calling_endpoint(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, ["-c", str(write_config(tmp_path)), "--dry-run"]
    )
    assert code == 0
    assert stderr == ""
    assert "source: 3 character(s), 1 paragraph(s)" in stdout
    assert (
        "pass 1/1 [translate]: target language English (scripts: Latin), "
        "mode=chunk, 1 unit(s)" in stdout
    )
    assert "unit 1/1" in stdout
    assert "cache key" in stdout


def test_prune_cache_removes_old_entries(tmp_path: Path) -> None:
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
        [
            "-c",
            str(tmp_path),
            "--cache-directory",
            str(cache_directory),
            "--prune-cache",
            "7",
        ],
    )
    assert code == 0
    assert "pruned 1 cache entry" in stderr
    assert not entry.exists()
    assert fresh.exists()


def test_prune_cache_requires_positive_days(tmp_path: Path) -> None:
    code, _, stderr = run_l2l(
        tmp_path,
        ["-c", str(tmp_path), "--cache-directory", str(tmp_path), "--prune-cache", "0"],
    )
    assert code == 2
    assert "positive number of days" in stderr


def test_prune_cache_reports_scan_failure(
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
        [
            "-c",
            str(tmp_path),
            "--cache-directory",
            str(cache_directory),
            "--prune-cache",
            "7",
        ],
    )
    assert code == 2
    assert "could not list entries in" in stderr
    assert "pruned" not in stderr


def test_prune_cache_reports_age_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    (cache_directory / "stale.txt").write_text("old", encoding="utf-8")

    monkeypatch.setattr(
        effects,
        "file_age",
        lambda path, now_value: io_result(
            Err(FileError(path, "read the age of", "[Errno 13] Permission denied"))
        ),
    )

    code, _, stderr = run_l2l(
        tmp_path,
        [
            "-c",
            str(tmp_path),
            "--cache-directory",
            str(cache_directory),
            "--prune-cache",
            "7",
        ],
    )
    assert code == 2
    assert "could not read the age of" in stderr
    assert "pruned" not in stderr


def test_prune_cache_reports_removal_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_directory = tmp_path / "cache"
    cache_directory.mkdir()
    stale = cache_directory / "stale.txt"
    stale.write_text("old", encoding="utf-8")
    stale_time = time.time() - 30 * 86400
    os.utime(stale, (stale_time, stale_time))

    monkeypatch.setattr(
        effects,
        "remove_file",
        lambda path: io_result(
            Err(FileError(path, "remove", "[Errno 13] Permission denied"))
        ),
    )

    code, _, stderr = run_l2l(
        tmp_path,
        [
            "-c",
            str(tmp_path),
            "--cache-directory",
            str(cache_directory),
            "--prune-cache",
            "7",
        ],
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
            "-c",
            str(write_config(tmp_path)),
            "--cache-directory",
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
    assert stdout.getvalue() == "Hello.\n"
    assert len(http.requests) == 1
    assert "Starting pass 1/1 [translate]..." in stderr.getvalue()
    assert "Done: " in stderr.getvalue()
    assert "TOTAL" in stderr.getvalue()


def test_validation_mode_controls_the_cli_outcome(tmp_path: Path) -> None:
    bad = "One.\n\nTwo."
    chunks = [
        *stream_chunks(bad),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    for best_effort, expected_code, expected_output, expected_log in (
        (False, 1, "", "still failed validation after 2 repair attempt(s)"),
        (True, 0, bad + "\n", "Keeping the last reply, uncached"),
    ):
        http = FakeHttp([FakeStreamResponse(chunks)])
        stdout, stderr = io.StringIO(), io.StringIO()
        arguments = [
            "-c",
            str(write_config(tmp_path)),
            "--cache-directory",
            str(tmp_path / "cache"),
            "--no-cache",
            "--ensure-paragraphs",
        ] + (["--best-effort"] if best_effort else [])
        code = cli.main(
            arguments,
            {},
            io.StringIO("你好。"),
            stdout,
            stderr,
            time.time,
            http.open,
        ).run()
        assert code == expected_code
        assert stdout.getvalue() == expected_output
        logged = stderr.getvalue()
        assert "failed validation 2 time(s)" in logged
        assert expected_log in logged
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
            "-c",
            str(write_config(tmp_path)),
            "--cache-directory",
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
    assert stdout.getvalue() == "Plain.\n"
    assert json.loads(http.requests[0].body)["stream"] is False


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
    flags = [
        "-c",
        str(write_config(tmp_path)),
        "--cache-directory",
        str(tmp_path / "cache"),
    ]

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
            "-c",
            str(write_config(tmp_path)),
            "--cache-directory",
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


def run_extract_glossary(
    tmp_path: Path,
    arguments: list[str],
    http: FakeHttp,
    stdin: str = "book text",
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            "-c",
            str(write_config(tmp_path)),
            "--cache-directory",
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


def test_extract_glossary_happy_path(tmp_path: Path) -> None:
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
    code, _, stderr = run_extract_glossary(
        tmp_path,
        ["--extract-glossary", str(output), "--glossary", str(carried)],
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
    assert "Already known names" in user_message
    assert "南京" in user_message
    assert "book text" in user_message


def test_extract_glossary_target_language_reaches_the_payload(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    http = FakeHttp([glossary_reply([{"original": "南京", "translated": "Nanjing"}])])
    code, _, _ = run_extract_glossary(
        tmp_path,
        ["--extract-glossary", str(output), "--target-language", "French"],
        http,
    )
    assert code == 0
    payload = json.loads(http.requests[0].body.decode("utf-8"))
    assert "Target language: French" in payload["messages"][1]["content"]


def test_extract_glossary_invalid_reply_exits_one(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    chunks = [
        *stream_chunks("not json"),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    code, _, stderr = run_extract_glossary(
        tmp_path, ["--extract-glossary", str(output)], http
    )
    assert code == 1
    assert "not valid JSON" in stderr
    assert not output.exists()


def test_extract_glossary_empty_stdin_exits_two(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    code, _, stderr = run_l2l(
        tmp_path,
        ["-c", str(write_config(tmp_path)), "--extract-glossary", str(output)],
        "",
    )
    assert code == 2
    assert "requires text on stdin" in stderr
    assert not output.exists()


def test_extract_glossary_ignores_pipeline_passes(tmp_path: Path) -> None:
    output = tmp_path / "book-glossary.json"
    http = FakeHttp([glossary_reply([{"original": "南京", "translated": "Nanjing"}])])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            "-c",
            str(write_update_glossary_config(tmp_path)),
            "--cache-directory",
            str(tmp_path / "cache"),
            "--extract-glossary",
            str(output),
        ],
        {},
        io.StringIO("book text"),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    assert code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {
            "original": "南京",
            "translated": "Nanjing",
            "pronouns": None,
            "type": "other",
        }
    ]


UPDATE_GLOSSARY_CONFIG_TEXT = CONFIG_TEXT.replace(
    '\n[[pass]]\nname = "translate"\nmode = "chunk"\n'
    'instruction = "Translate to {target_language}."',
    "\n[[pass]]\n"
    'name = "glossary"\n'
    'mode = "update_glossary"\n'
    "\n[[pass]]\n"
    'name = "translate"\n'
    'mode = "chunk"\n'
    'instruction = "Translate to {target_language}."',
)


def write_update_glossary_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(UPDATE_GLOSSARY_CONFIG_TEXT)
    return config_path


def run_update_glossary_pass(
    tmp_path: Path,
    config_path: Path,
    glossary: Path,
    http: FakeHttp,
    *extra_arguments: str,
    cache_directory: Path | None = None,
    use_cache: bool = False,
    stdin: str = "你好。",
) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            "-c",
            str(config_path),
            "--cache-directory",
            str(cache_directory or tmp_path / "cache"),
            *(() if use_cache else ("--no-cache",)),
            "--glossary",
            str(glossary),
            *extra_arguments,
        ],
        {},
        io.StringIO(stdin),
        stdout,
        stderr,
        time.time,
        http.open,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def translation_response(text: str) -> FakeStreamResponse:
    return FakeStreamResponse(with_usage(stream_chunks(text), USAGE))


def test_update_glossary_pass_updates_file_and_flows_into_later_passes(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text(
        '[{"original": "南京", "translated": "Nanjing"}]', encoding="utf-8"
    )
    http = FakeHttp(
        [
            glossary_reply(
                [
                    {"original": "南京", "translated": "Nanking"},
                    {"original": "大观园", "translated": "Grand View Garden"},
                ]
            ),
            translation_response("Hello."),
        ]
    )
    code, stdout, stderr = run_update_glossary_pass(
        tmp_path,
        write_update_glossary_config(tmp_path),
        glossary,
        http,
        "--target-language",
        "French",
    )
    assert code == 0
    assert stdout.strip() == "Hello."

    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert [item["original"] for item in document] == ["南京", "大观园"]
    assert document[0]["translated"] == "Nanjing"
    assert document[1]["translated"] == "Grand View Garden"

    extraction_payload = json.loads(http.requests[0].body)
    extraction_user = extraction_payload["messages"][1]["content"]
    assert "Target language: French" in extraction_user
    assert "Already known names" in extraction_user

    translation_payload = json.loads(http.requests[1].body)
    translation_system = translation_payload["messages"][0]["content"]
    assert '- 南京 = "Nanjing"' in translation_system
    assert '- 大观园 = "Grand View Garden"' in translation_system

    assert "Starting pass 1/2 [glossary]..." in stderr
    assert "Starting pass 2/2 [translate]..." in stderr
    assert "TOTAL" in stderr


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
            "-c",
            str(write_update_glossary_config(tmp_path)),
            "--cache-directory",
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


def test_update_glossary_pass_applies_the_include_types_filter(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    config_path = tmp_path / "l2l.toml"
    config_path.write_text(
        UPDATE_GLOSSARY_CONFIG_TEXT.replace(
            'mode = "update_glossary"',
            'mode = "update_glossary"\ninclude_types = ["character"]',
        )
    )
    http = FakeHttp(
        [
            glossary_reply(
                [
                    {
                        "original": "林黛玉",
                        "translated": "Lin Daiyu",
                        "type": "character",
                    },
                    {
                        "original": "大观园",
                        "translated": "Grand View Garden",
                        "type": "place",
                    },
                ]
            ),
            translation_response("Hello."),
        ]
    )
    code, stdout, _ = run_update_glossary_pass(
        tmp_path, config_path, glossary, http, "--target-language", "French"
    )
    assert code == 0
    assert stdout.strip() == "Hello."

    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert [item["original"] for item in document] == ["林黛玉"]

    translation_system = json.loads(http.requests[1].body)["messages"][0]["content"]
    assert '- 林黛玉 = "Lin Daiyu"' in translation_system
    assert "大观园" not in translation_system


@pytest.mark.parametrize(
    ("extra_arguments", "requests_structured_output"),
    [((), True), (("--no-structured-output",), False)],
)
def test_update_glossary_pass_response_format_follows_the_flag(
    tmp_path: Path,
    extra_arguments: tuple[str, ...],
    requests_structured_output: bool,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    http = FakeHttp(
        [
            glossary_reply([{"original": "大观园", "translated": "G"}]),
            translation_response("Hello."),
        ]
    )
    code, _stdout, _stderr = run_update_glossary_pass(
        tmp_path,
        write_update_glossary_config(tmp_path),
        glossary,
        http,
        *extra_arguments,
    )
    assert code == 0
    payload = json.loads(http.requests[0].body)
    assert ("response_format" in payload) is requests_structured_output


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
    http = FakeHttp(
        [
            glossary_reply([{"original": "大观园", "translated": "G"}]),
            translation_response("Hello."),
        ]
    )
    code, _stdout, _stderr = run_update_glossary_pass(
        tmp_path, config_path, glossary, http, "--target-language", "French"
    )
    assert code == 0
    payload = json.loads(http.requests[0].body)
    assert "Target language: Chinese" in payload["messages"][1]["content"]


def test_update_glossary_pass_without_glossary_flag_exits_two(
    tmp_path: Path,
) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path, ["-c", str(write_update_glossary_config(tmp_path)), "--dry-run"]
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
    code, stdout, _stderr = run_l2l(
        tmp_path,
        [
            "-c",
            str(write_update_glossary_config(tmp_path)),
            "--dry-run",
            "--glossary",
            str(glossary),
        ],
    )
    assert code == 0
    assert (
        "pass 1/2 [glossary]: target language English (scripts: Latin), "
        "mode=update_glossary, 1 call with the whole document" in stdout
    )
    assert (
        "pass 2/2 [translate]: target language English (scripts: Latin), "
        "mode=chunk, 1 unit(s)" in stdout
    )


def test_update_glossary_extraction_reply_is_cached_across_runs(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    cache_directory = tmp_path / "cache"

    def run() -> tuple[int, str, int]:
        http = FakeHttp(
            [
                glossary_reply([{"original": "大观园", "translated": "G"}]),
                translation_response("Hello."),
            ]
        )
        code, stdout, _stderr = run_update_glossary_pass(
            tmp_path,
            write_update_glossary_config(tmp_path),
            glossary,
            http,
            cache_directory=cache_directory,
            use_cache=True,
        )
        return code, stdout, len(http.requests)

    first_code, first_output, first_calls = run()
    second_code, second_output, second_calls = run()
    assert (first_code, second_code) == (0, 0)
    assert first_output == second_output == "Hello.\n"
    assert (first_calls, second_calls) == (2, 0)
    document = json.loads(glossary.read_text(encoding="utf-8"))
    assert [item["original"] for item in document] == ["大观园"]


def test_parse_arguments_rejects_invalid_or_conflicting_modes() -> None:
    invalid_argument_groups = (
        (["--nope"], "invalid arguments"),
        (["-c", "cfg.toml", "--max"], "invalid arguments"),
        (
            ["-c", "cfg.toml", "--dry-run", "--extract-glossary", "glossary.json"],
            "--extract-glossary cannot be combined with --dry-run",
        ),
        (
            ["-c", "cfg.toml", "--check-config", "--prune-cache", "7"],
            "cannot be combined with",
        ),
        (
            ["-c", "cfg.toml", "--check-config", "--extract-glossary", "g.json"],
            "cannot be combined with",
        ),
        (["-c", "cfg.toml", "--check-config", "--dry-run"], "cannot be combined with"),
        (
            ["-c", "cfg.toml", "--prune-cache", "7", "--extract-glossary", "g.json"],
            "cannot be combined with",
        ),
        (["-c", "cfg.toml", "--prune-cache", "7", "--dry-run"], "cannot be combined"),
    )
    for arguments, fragment in invalid_argument_groups:
        result = cli.parse_arguments(arguments).run()
        assert isinstance(result, Err)
        assert fragment in describe(result.error)


def test_parse_arguments_accepts_single_modes() -> None:
    for arguments in (
        ["-c", "cfg.toml", "--check-config"],
        ["-c", "cfg.toml", "--prune-cache", "7"],
        ["-c", "cfg.toml", "--extract-glossary", "glossary.json"],
        ["-c", "cfg.toml", "--dry-run"],
    ):
        result = cli.parse_arguments(arguments).run()
        assert isinstance(result, Ok)
        assert isinstance(result.value, Arguments)


def test_main_prints_version_and_help_without_http(tmp_path: Path) -> None:
    for arguments, expected in (
        (["--version"], "l2l " + __version__ + "\n"),
        (["--help"], "usage:"),
    ):
        code, stdout, stderr = run_l2l(tmp_path, arguments, "")
        assert code == 0
        assert expected in stdout
        assert stderr == ""


def test_main_rejects_dry_run_with_extract_glossary_without_http(
    tmp_path: Path,
) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path,
        [
            "-c",
            str(tmp_path / "l2l.toml"),
            "--dry-run",
            "--extract-glossary",
            str(tmp_path / "glossary.json"),
        ],
    )
    assert code == 2
    assert stdout == ""
    assert "l2l: --extract-glossary cannot be combined with --dry-run" in stderr


def test_main_log_retention_days_prunes_old_logs(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
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
            "-c",
            str(config_path),
            "--cache-directory",
            str(cache_directory),
            "--no-cache",
            "--log-retention-days",
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


def test_cli_exits_with_program_code(monkeypatch: pytest.MonkeyPatch) -> None:
    exits: list[int] = []
    monkeypatch.setattr(cli, "main", lambda *arguments: io_pure(3))
    monkeypatch.setattr(sys, "exit", exits.append)
    cli.cli()
    assert exits == [3]


def test_cli_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(*arguments: Any, **keyword_arguments: Any) -> Any:
        raise KeyboardInterrupt

    exits: list[int] = []
    monkeypatch.setattr(cli, "main", interrupted)
    monkeypatch.setattr(sys, "exit", exits.append)
    cli.cli()
    assert exits == [130]


def test_cli_handles_broken_pipe(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*arguments: Any, **keyword_arguments: Any) -> Any:
        raise BrokenPipeError

    exits: list[int] = []
    monkeypatch.setattr(cli, "main", broken)
    monkeypatch.setattr(os, "open", lambda path, flags: -1)
    monkeypatch.setattr(os, "dup2", lambda fd, target: None)
    monkeypatch.setattr(sys, "exit", exits.append)
    cli.cli()
    assert exits == [141]


def test_parse_args_defaults_and_short_flags() -> None:
    defaults = cli.parse_args(["-c", "cfg.toml", "--no-cache", "-V"])
    assert defaults.config == "cfg.toml"
    assert defaults.no_cache
    assert not defaults.ensure_paragraphs
    assert defaults.verbose
    assert defaults.cache_directory is None
    assert defaults.log_retention_days == 30
    assert defaults.target_language is None
    assert cli.parse_args(["-c", "cfg.toml", "--ensure-paragraphs"]).ensure_paragraphs
    assert (
        cli.parse_args(
            ["-c", "cfg.toml", "--log-retention-days", "0"]
        ).log_retention_days
        == 0
    )

    arguments = cli.parse_args(
        [
            "-c",
            "cfg.toml",
            "-u",
            "http://endpoint.test/v1",
            "-k",
            "secret",
            "-m",
            "m",
            "-t",
            "5",
            "-T",
            "1000",
            "-s",
            "-g",
            "first.json",
            "-g",
            "second.json",
            "-l",
            "French",
            "-P",
            "-b",
            "-p",
            "-n",
            "-V",
        ]
    )
    assert arguments.config == "cfg.toml"
    assert arguments.base_url == "http://endpoint.test/v1"
    assert arguments.api_key == "secret"
    assert arguments.model == "m"
    assert arguments.timeout == 5.0
    assert arguments.maximum_tokens == 1000
    assert arguments.stream is True
    assert arguments.glossary == ("first.json", "second.json")
    assert arguments.target_language == "French"
    assert arguments.ensure_paragraphs
    assert arguments.best_effort
    assert arguments.print_log_path
    assert arguments.dry_run
    assert arguments.verbose


def test_update_glossary_pass_with_malformed_reply_exits_one(tmp_path: Path) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    chunks = [
        *stream_chunks("not json at all"),
        {"choices": [{"delta": {}}], "usage": USAGE},
    ]
    http = FakeHttp([FakeStreamResponse(chunks)])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            "-c",
            str(write_update_glossary_config(tmp_path)),
            "--cache-directory",
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
    assert code == 1
    assert "not valid JSON" in stderr.getvalue()
    assert json.loads(glossary.read_text(encoding="utf-8")) == []
    assert len(http.requests) == 1


def test_update_glossary_pass_reports_a_failed_extraction_call(
    tmp_path: Path,
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    requests: list[Any] = []

    def unreachable_open(request: Any, timeout: float) -> Result[Any, TranslationError]:
        requests.append(request)
        return fail_http("unreachable", "down")

    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            "-c",
            str(write_update_glossary_config(tmp_path)),
            "--cache-directory",
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
        unreachable_open,
    ).run()
    assert code == 1
    assert "could not reach endpoint: down" in stderr.getvalue()
    assert json.loads(glossary.read_text(encoding="utf-8")) == []
    assert len(requests) == 3


def test_update_glossary_pass_reports_a_glossary_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    glossary = tmp_path / "glossary.json"
    glossary.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "write_text_file",
        lambda path, content, description: io_result(
            Err(
                FileError(
                    path,
                    "write " + description,
                    "[Errno 13] Permission denied",
                )
            )
        ),
    )
    http = FakeHttp([glossary_reply([{"original": "大观园", "translated": "G"}])])
    stdout, stderr = io.StringIO(), io.StringIO()
    code = cli.main(
        [
            "-c",
            str(write_update_glossary_config(tmp_path)),
            "--cache-directory",
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
    assert code == 1
    assert "could not write glossary file" in stderr.getvalue()
    assert json.loads(glossary.read_text(encoding="utf-8")) == []
    assert len(http.requests) == 1


def test_unknown_target_language_warns_on_stderr(tmp_path: Path) -> None:
    code, stdout, stderr = run_l2l(
        tmp_path,
        ["-c", str(write_config(tmp_path)), "--dry-run", "-l", "Zorblaxian"],
    )
    assert code == 0
    assert "target language Zorblaxian (scripts: unknown)" in stdout
    assert "warning: target language Zorblaxian is not recognized" in stderr


def test_target_language_separates_the_cache_and_fills_the_prompt(
    tmp_path: Path,
) -> None:
    http = FakeHttp(
        [
            FakeStreamResponse(
                [
                    *stream_chunks("Hello."),
                    {
                        "choices": [{"delta": {}}],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "cost": 0.0,
                        },
                    },
                ]
            )
        ]
    )
    base_flags = [
        "-c",
        str(write_config(tmp_path)),
        "--cache-directory",
        str(tmp_path / "cache"),
    ]

    def run(language: str) -> int:
        return cli.main(
            [*base_flags, "-l", language],
            {},
            io.StringIO("你好。"),
            io.StringIO(),
            io.StringIO(),
            time.time,
            http.open,
        ).run()

    assert run("English") == 0
    assert run("French") == 0
    assert len(http.requests) == 2
    systems = [
        json.loads(request.body)["messages"][0]["content"] for request in http.requests
    ]
    assert systems == ["Translate to English.", "Translate to French."]

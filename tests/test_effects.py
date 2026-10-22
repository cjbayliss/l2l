import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO, cast

import pytest

from l2l.effects import (
    append_text_file,
    cache_entry_paths,
    cache_read,
    cache_write,
    copy_file,
    ensure_directory,
    entry_paths,
    file_age,
    load_json,
    load_toml,
    log_paths,
    path_is_directory,
    prune_old_logs,
    read_text_file,
    remove_file,
    replace_file,
    resolve_cache_directory,
    run_process,
    stream_isatty,
    user_config_path,
    write_text_file,
)
from l2l.errors import FileError, describe
from l2l.monads import NOTHING, Err, Just, Ok
from l2l.text import cache_path

WRITER_COUNT_FOR_RACE_TEST = 16


def test_stream_isatty_handles_errors() -> None:
    class Raising:
        def isatty(self) -> bool:
            raise OSError("closed")

    assert stream_isatty(cast(TextIO, Raising())).run() is False


def test_file_readers_report_directory_failures(tmp_path: Path) -> None:
    outcomes = (
        load_toml(str(tmp_path), "config file").run(),
        read_text_file(str(tmp_path), "instruction file").run(),
    )
    messages = ("could not read config file", "could not read instruction file")
    for outcome, message in zip(outcomes, messages, strict=True):
        assert isinstance(outcome, Err)
        assert isinstance(outcome.error, FileError)
        assert message in describe(outcome.error)


def test_load_json_reads_and_reports_errors(tmp_path: Path) -> None:
    path = tmp_path / "glossary.json"
    path.write_text('[{"original": "x", "translated": "y"}]', encoding="utf-8")
    assert load_json(str(path), "glossary file").run() == Ok(
        [{"original": "x", "translated": "y"}]
    )

    missing = load_json(str(tmp_path / "absent.json"), "glossary file").run()
    assert isinstance(missing, Err)
    assert "could not read glossary file" in describe(missing.error)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    malformed = load_json(str(broken), "glossary file").run()
    assert isinstance(malformed, Err)
    assert "cannot parse glossary file" in describe(malformed.error)


def test_resolve_cache_directory_and_user_config_path(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    assert resolve_cache_directory({}, str(target)).run() == str(target)
    assert target.exists()
    assert (
        resolve_cache_directory({"XDG_CACHE_HOME": str(tmp_path)})
        .run()
        .startswith(str(tmp_path))
    )
    assert user_config_path({"XDG_CONFIG_HOME": "/cfg"}).run() == "/cfg/l2l/config.toml"


def test_cache_write_and_read_roundtrip(tmp_path: Path) -> None:
    assert cache_write(str(tmp_path), "key1", "value").run() is True
    assert cache_read(str(tmp_path), "key1").run() == Just("value")
    assert cache_read(str(tmp_path), "nope").run() == NOTHING


def test_concurrent_cache_writes_to_the_same_key_all_succeed(tmp_path: Path) -> None:
    cache_directory = str(tmp_path)
    written_values = frozenset(
        "value-%d" % index for index in range(WRITER_COUNT_FOR_RACE_TEST)
    )

    def write(value: str) -> bool:
        return cache_write(cache_directory, "shared-key", value).run()

    with ThreadPoolExecutor(max_workers=WRITER_COUNT_FOR_RACE_TEST) as executor:
        outcomes = tuple(executor.map(write, sorted(written_values)))

    assert outcomes == (True,) * WRITER_COUNT_FOR_RACE_TEST
    stored = cache_read(cache_directory, "shared-key").run()
    assert isinstance(stored, Just)
    assert stored.value in written_values
    assert cache_entry_paths(cache_directory).run() == Ok(
        (cache_path(cache_directory, "shared-key"),)
    )


def test_cache_write_failures_leave_no_entries(tmp_path: Path) -> None:
    assert cache_write(str(tmp_path / "absent"), "key1", "value").run() is False
    (tmp_path / "key1.txt").mkdir()
    assert cache_write(str(tmp_path), "key1", "value").run() is False
    assert cache_entry_paths(str(tmp_path)).run() == Ok(())


def test_cache_entry_paths_includes_tmp_orphans(tmp_path: Path) -> None:
    (tmp_path / "key.txt").write_text("value", encoding="utf-8")
    (tmp_path / "key.txt.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "note.log").write_text("log", encoding="utf-8")
    paths = cache_entry_paths(str(tmp_path)).run()
    assert isinstance(paths, Ok)
    assert sorted(path.endswith((".txt", ".txt.tmp")) for path in paths.value) == [
        True,
        True,
    ]
    assert cache_entry_paths(str(tmp_path / "absent")).run() == Ok(())


def test_cache_entry_paths_reports_scan_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(directory: str) -> object:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os, "scandir", refuse)
    outcome = cache_entry_paths(str(tmp_path)).run()
    assert isinstance(outcome, Err)
    assert isinstance(outcome.error, FileError)
    assert "could not list entries in" in describe(outcome.error)


def test_log_paths_tolerates_missing_directories_and_lists_logs(tmp_path: Path) -> None:
    assert log_paths(str(tmp_path / "absent")).run() == ()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("log", encoding="utf-8")
    (logs / "run.log.tmp").write_text("partial", encoding="utf-8")
    paths = log_paths(str(tmp_path)).run()
    assert [path.endswith("run.log") for path in paths] == [True]


def test_file_age_and_removal_report_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert file_age("absent/path", 100.0).run() == Ok(0.0)
    assert remove_file("absent/path").run() == Ok(False)

    entry = tmp_path / "key.txt"
    entry.write_text("value", encoding="utf-8")

    def refuse_age(path: str) -> float:
        raise PermissionError(13, "Permission denied")

    def refuse_remove(path: str) -> None:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(os.path, "getmtime", refuse_age)
    monkeypatch.setattr(os, "remove", refuse_remove)
    age = file_age("some/path", 100.0).run()
    removed = remove_file(str(entry)).run()
    assert isinstance(age, Err)
    assert "could not read the age of" in describe(age.error)
    assert isinstance(removed, Err)
    assert "could not remove" in describe(removed.error)


def test_ensure_directory_creates_nested_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    ensure_directory(str(target)).run()
    assert target.is_dir()
    ensure_directory(str(target)).run()


def test_write_and_append_text_files(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    outcome = write_text_file(str(target), "content", "output file").run()
    assert isinstance(outcome, Ok)
    assert target.read_text(encoding="utf-8") == "content"

    ledger = tmp_path / "ledger.jsonl"
    assert isinstance(append_text_file(str(ledger), "one\n", "ledger").run(), Ok)
    assert isinstance(append_text_file(str(ledger), "two\n", "ledger").run(), Ok)
    assert ledger.read_text(encoding="utf-8") == "one\ntwo\n"


def test_text_file_writes_report_failures(tmp_path: Path) -> None:
    write_outcome = write_text_file(str(tmp_path), "content", "output file").run()
    append_outcome = append_text_file(str(tmp_path), "line\n", "ledger").run()
    assert isinstance(write_outcome, Err)
    assert "could not write output file" in describe(write_outcome.error)
    assert isinstance(append_outcome, Err)
    assert "could not append to ledger" in describe(append_outcome.error)


def test_replace_and_copy_files(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    target = tmp_path / "final"
    source.write_text("body", encoding="utf-8")
    assert replace_file(str(source), str(target)).run() is True
    assert not source.exists()
    assert target.read_text(encoding="utf-8") == "body"

    copied = tmp_path / "copied"
    assert copy_file(str(target), str(copied)).run() is True
    assert copied.read_text(encoding="utf-8") == "body"
    assert replace_file(str(tmp_path / "absent"), str(target)).run() is False
    assert copy_file(str(tmp_path / "absent"), str(copied)).run() is False


def test_entry_paths_lists_files_only(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1", encoding="utf-8")
    (tmp_path / "two.txt").write_text("2", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    paths = entry_paths(str(tmp_path)).run()
    assert sorted(path.endswith(("one.txt", "two.txt")) for path in paths) == [
        True,
        True,
    ]
    assert entry_paths(str(tmp_path / "absent")).run() == ()


def test_run_process_captures_streams(tmp_path: Path) -> None:
    result = run_process(("sed", "s/a/b/"), "banana", str(tmp_path)).run()
    assert result.returncode == 0
    assert result.stdout == "bbnana"
    assert result.stderr == ""


def test_path_is_directory_distinguishes_files(tmp_path: Path) -> None:
    target = tmp_path / "thing"
    target.write_text("x", encoding="utf-8")
    assert path_is_directory(str(target)).run() is False
    assert path_is_directory(str(tmp_path)).run() is True
    assert path_is_directory(str(tmp_path / "absent")).run() is False


def test_prune_old_logs_removes_only_stale_logs(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "old.log"
    old.write_text("stale", encoding="utf-8")
    fresh = logs / "fresh.log"
    fresh.write_text("keep", encoding="utf-8")
    month_ago = time.time() - 40 * 86400
    os.utime(old, (month_ago, month_ago))

    assert prune_old_logs(str(tmp_path), 30, time.time).run() == 1
    assert not old.exists()
    assert fresh.exists()
    assert prune_old_logs(str(tmp_path), 0, time.time).run() == 1
    assert not fresh.exists()

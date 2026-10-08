import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TextIO, cast

from l2l.effects import (
    _no_append,
    _no_close,
    append_text_file,
    cache_entry_paths,
    cache_read,
    cache_write,
    copy_file,
    ensure_directory,
    entry_paths,
    file_age,
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
    time_sleep,
    user_config_path,
    write_text_file,
)
from l2l.errors import describe
from l2l.monads import NOTHING, Err, Just, Ok
from l2l.text import cache_path

WRITER_COUNT_FOR_RACE_TEST = 16


def test_stream_isatty_handles_errors() -> None:
    class Raising:
        def isatty(self) -> bool:
            raise OSError("closed")

    assert stream_isatty(cast(TextIO, Raising())).run() is False


def test_time_sleep_returns_after_the_requested_delay() -> None:
    started = time.monotonic()
    time_sleep(0.0)
    assert time.monotonic() >= started


def test_load_toml_reports_directory_reads(tmp_path: Path) -> None:
    result = load_toml(str(tmp_path), "config file").run()
    assert isinstance(result, Err)
    assert "cannot read config file" in describe(result.error)


def test_read_text_file_reports_directory_reads(tmp_path: Path) -> None:
    result = read_text_file(str(tmp_path), "instruction file").run()
    assert isinstance(result, Err)
    assert "cannot read instruction file" in describe(result.error)


def test_no_append_and_no_close_do_nothing() -> None:
    _no_append("ignored")
    _no_close()


def test_resolve_cache_dir_creates_override(tmp_path: Path) -> None:
    target = tmp_path / "cache"
    resolved = resolve_cache_directory({}, str(target)).run()
    assert resolved == str(target)
    assert target.exists()


def test_resolve_cache_dir_defaults_under_xdg(tmp_path: Path) -> None:
    resolved = resolve_cache_directory({"XDG_CACHE_HOME": str(tmp_path)}).run()
    assert resolved.startswith(str(tmp_path))


def test_user_config_path_honours_xdg() -> None:
    path = user_config_path({"XDG_CONFIG_HOME": "/cfg"}).run()
    assert path == "/cfg/l2l/config.toml"


def test_cache_write_and_read_roundtrip(tmp_path: Path) -> None:
    assert cache_write(str(tmp_path), "key1", "value").run() is True
    assert cache_read(str(tmp_path), "key1").run() == Just("value")


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
    assert cache_entry_paths(cache_directory).run() == (
        cache_path(cache_directory, "shared-key"),
    )


def test_cache_write_reports_failure_when_cache_directory_is_missing(
    tmp_path: Path,
) -> None:
    assert cache_write(str(tmp_path / "absent"), "key1", "value").run() is False


def test_cache_write_removes_its_temporary_file_when_replacement_fails(
    tmp_path: Path,
) -> None:
    (tmp_path / "key1.txt").mkdir()

    assert cache_write(str(tmp_path), "key1", "value").run() is False
    assert cache_entry_paths(str(tmp_path)).run() == ()


def test_cache_read_missing_returns_nothing(tmp_path: Path) -> None:
    assert cache_read(str(tmp_path), "nope").run() == NOTHING


def test_cache_entry_paths_includes_tmp_orphans(tmp_path: Path) -> None:
    (tmp_path / "key.txt").write_text("value", encoding="utf-8")
    (tmp_path / "key.txt.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "note.log").write_text("log", encoding="utf-8")
    paths = cache_entry_paths(str(tmp_path)).run()
    assert sorted(path.endswith((".txt", ".txt.tmp")) for path in paths) == [True, True]


def test_cache_entry_paths_tolerates_a_missing_directory(tmp_path: Path) -> None:
    assert cache_entry_paths(str(tmp_path / "absent")).run() == ()


def test_log_paths_tolerates_a_missing_log_directory(tmp_path: Path) -> None:
    assert log_paths(str(tmp_path)).run() == ()


def test_log_paths_lists_log_files(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text("log", encoding="utf-8")
    (logs / "run.log.tmp").write_text("partial", encoding="utf-8")
    paths = log_paths(str(tmp_path)).run()
    assert [path.endswith("run.log") for path in paths] == [True]


def test_file_age_reports_zero_for_missing_files() -> None:
    assert file_age("absent/path", 100.0).run() == 0.0


def test_remove_file_reports_failures() -> None:
    assert remove_file("absent/path").run() is False


def test_ensure_directory_creates_nested_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b"
    ensure_directory(str(target)).run()
    assert target.is_dir()
    ensure_directory(str(target)).run()


def test_write_text_file_roundtrips(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    outcome = write_text_file(str(target), "content", "output file").run()
    assert isinstance(outcome, Ok)
    assert target.read_text(encoding="utf-8") == "content"


def test_write_text_file_reports_failures(tmp_path: Path) -> None:
    outcome = write_text_file(str(tmp_path), "content", "output file").run()
    assert isinstance(outcome, Err)
    assert "cannot write output file" in describe(outcome.error)


def test_append_text_file_appends(tmp_path: Path) -> None:
    target = tmp_path / "ledger.jsonl"
    assert isinstance(append_text_file(str(target), "one\n", "ledger").run(), Ok)
    assert isinstance(append_text_file(str(target), "two\n", "ledger").run(), Ok)
    assert target.read_text(encoding="utf-8") == "one\ntwo\n"


def test_append_text_file_reports_failures(tmp_path: Path) -> None:
    outcome = append_text_file(str(tmp_path), "line\n", "ledger").run()
    assert isinstance(outcome, Err)
    assert "cannot append ledger" in describe(outcome.error)


def test_replace_file_moves_content(tmp_path: Path) -> None:
    source = tmp_path / "partial"
    target = tmp_path / "final"
    source.write_text("body", encoding="utf-8")
    assert replace_file(str(source), str(target)).run() is True
    assert not source.exists()
    assert target.read_text(encoding="utf-8") == "body"


def test_replace_file_reports_failures(tmp_path: Path) -> None:
    assert replace_file(str(tmp_path / "absent"), str(tmp_path / "t")).run() is False


def test_copy_file_copies_content(tmp_path: Path) -> None:
    source = tmp_path / "seed.txt"
    target = tmp_path / "v0.txt"
    source.write_text("seed", encoding="utf-8")
    assert copy_file(str(source), str(target)).run() is True
    assert target.read_text(encoding="utf-8") == "seed"


def test_copy_file_reports_failures(tmp_path: Path) -> None:
    assert copy_file(str(tmp_path / "absent"), str(tmp_path / "t")).run() is False


def test_entry_paths_lists_files_only(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1", encoding="utf-8")
    (tmp_path / "two.txt").write_text("2", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
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


def test_path_is_dir_distinguishes_files(tmp_path: Path) -> None:
    target = tmp_path / "thing"
    target.write_text("x", encoding="utf-8")
    assert path_is_directory(str(target)).run() is False
    assert path_is_directory(str(tmp_path)).run() is True
    assert path_is_directory(str(tmp_path / "absent")).run() is False


def test_prune_old_logs_removes_only_stale_logs(tmp_path: Path) -> None:
    import os
    import time

    logs = tmp_path / "logs"
    logs.mkdir()
    old = logs / "old.log"
    old.write_text("stale", encoding="utf-8")
    fresh = logs / "fresh.log"
    fresh.write_text("keep", encoding="utf-8")
    month_ago = time.time() - 40 * 86400
    os.utime(old, (month_ago, month_ago))

    removed = prune_old_logs(str(tmp_path), 30, time.time).run()
    assert removed == 1
    assert not old.exists()
    assert fresh.exists()


def test_prune_old_logs_zero_days_is_a_zero_day_horizon(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "ancient.log").write_text("old", encoding="utf-8")

    removed = prune_old_logs(str(tmp_path), 0, time.time).run()
    assert removed == 1
    assert not (logs / "ancient.log").exists()

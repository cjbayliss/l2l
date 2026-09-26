import time
from pathlib import Path
from typing import TextIO, cast

from l2l.effects import (
    _no_append,
    _no_close,
    cache_entry_paths,
    cache_read,
    cache_write,
    file_age,
    load_toml,
    log_paths,
    prune_old_logs,
    read_text_file,
    remove_file,
    resolve_cache_dir,
    stream_isatty,
    time_sleep,
    user_config_path,
)
from l2l.errors import describe
from l2l.monads import NOTHING, Err, Just


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
    resolved = resolve_cache_dir({}, str(target)).run()
    assert resolved == str(target)
    assert target.exists()


def test_resolve_cache_dir_defaults_under_xdg(tmp_path: Path) -> None:
    resolved = resolve_cache_dir({"XDG_CACHE_HOME": str(tmp_path)}).run()
    assert resolved.startswith(str(tmp_path))


def test_user_config_path_honours_xdg() -> None:
    path = user_config_path({"XDG_CONFIG_HOME": "/cfg"}).run()
    assert path == "/cfg/l2l/config.toml"


def test_cache_write_and_read_roundtrip(tmp_path: Path) -> None:
    cache_write(str(tmp_path), "key1", "value").run()
    assert cache_read(str(tmp_path), "key1").run() == Just("value")


def test_cache_read_missing_returns_nothing(tmp_path: Path) -> None:
    assert cache_read(str(tmp_path), "nope").run() == NOTHING


def test_cache_entry_paths_includes_tmp_orphans(tmp_path: Path) -> None:
    (tmp_path / "key.txt").write_text("value", encoding="utf-8")
    (tmp_path / "key.txt.tmp").write_text("partial", encoding="utf-8")
    (tmp_path / "note.log").write_text("log", encoding="utf-8")
    paths = cache_entry_paths(str(tmp_path)).run()
    assert sorted(p.endswith((".txt", ".txt.tmp")) for p in paths) == [True, True]


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
    assert [p.endswith("run.log") for p in paths] == [True]


def test_file_age_reports_zero_for_missing_files() -> None:
    assert file_age("absent/path", 100.0).run() == 0.0


def test_remove_file_reports_failures() -> None:
    assert remove_file("absent/path").run() is False


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

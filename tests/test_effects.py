import time
from pathlib import Path
from typing import TextIO, cast

from zh2en.effects import (
    cache_entry_paths,
    cache_read,
    cache_write,
    io_isatty,
    prune_old_logs,
    resolve_cache_dir,
    user_config_path,
)
from zh2en.monads import NOTHING, Just


def test_io_isatty_handles_errors() -> None:
    class Raising:
        def isatty(self) -> bool:
            raise OSError("closed")

    assert io_isatty(cast(TextIO, Raising())).run() is False


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
    assert path == "/cfg/zh2en/config.toml"


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
    """A zero-day horizon prunes everything; the CLI gates on `keep > 0`."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "ancient.log").write_text("old", encoding="utf-8")

    removed = prune_old_logs(str(tmp_path), 0, time.time).run()
    assert removed == 1
    assert not (logs / "ancient.log").exists()

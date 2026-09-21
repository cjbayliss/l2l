from pathlib import Path
from typing import TextIO, cast

from zh2en.effects import (
    cache_read,
    cache_write,
    io_isatty,
    resolve_cache_dir,
    user_config_path,
)


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
    assert cache_read(str(tmp_path), "key1").run() == "value"


def test_cache_read_missing_returns_none(tmp_path: Path) -> None:
    assert cache_read(str(tmp_path), "nope").run() is None

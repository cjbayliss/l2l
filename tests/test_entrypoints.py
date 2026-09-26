import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
        timeout=120,
    )


def test_python_dash_m_l2l_prints_help() -> None:
    completed = run(["-m", "l2l", "--help"])
    assert completed.returncode == 0
    assert "usage:" in completed.stdout


def test_cli_module_guard_runs_the_program(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["l2l/cli.py", "--version"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ROOT / "l2l" / "cli.py"), run_name="__main__")

    assert raised.value.code == 0


def test_package_main_guard_runs_the_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["l2l", "--version"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ROOT / "l2l" / "__main__.py"), run_name="__main__")

    assert raised.value.code == 0


def test_importing_the_entry_module_does_not_run_the_cli() -> None:
    import l2l.__main__ as entry

    assert entry.__name__ == "l2l.__main__"

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_python_dash_m_l2l_prints_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "l2l", "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout


def test_main_module_entrypoint_runs_the_program(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["l2l.__main__", "--version"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("l2l.__main__", run_name="__main__")
    assert raised.value.code == 0


def test_cli_module_guard_runs_the_program(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["src/l2l/cli.py", "--version"])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(ROOT / "src" / "l2l" / "cli.py"), run_name="__main__")
    assert raised.value.code == 0

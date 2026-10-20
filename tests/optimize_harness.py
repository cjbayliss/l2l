from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

from l2l.effects import ProcessResult
from l2l.monads import IO, io_pure
from tools.optimize import EndpointReply, OpenEndpoint, RunCommand, main

JUDGE_TEMPLATE = (
    "SOURCE: <<<SOURCE>>> ||| A: <<<TRANSLATION_A>>> ||| B: <<<TRANSLATION_B>>> judge"
)
REWRITE_TEMPLATE = (
    "CURRENT: <<<CURRENT_PROMPT>>> HISTORY: <<<HISTORY>>> "
    "CRITIQUES: <<<CRITIQUES>>> propose better"
)
BASE_TOML = (
    "[api]\n"
    'base_url = "http://endpoint.test/v1"\n'
    'api_key = "key"\n'
    'model = "m"\n\n'
    "[[pass]]\n"
    'name = "translate"\n'
    'model = "m"\n'
    'instruction = "translate this"\n'
)
DIRECT_ENVIRONMENT = {"no_proxy": "127.0.0.1,localhost"}
PROXY_VARIABLES = ("http_proxy", "https_proxy", "all_proxy")


def workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    working_directory = tmp_path / "lab"
    chapters_directory = working_directory / "chapters"
    chapters_directory.mkdir(parents=True)
    for name in ("ch1", "ch2"):
        (chapters_directory / (name + ".txt")).write_text(
            "source of " + name, encoding="utf-8"
        )
    (chapters_directory / "holdout.txt").write_text("holdout source", encoding="utf-8")
    base = tmp_path / "base.toml"
    base.write_text(BASE_TOML, encoding="utf-8")
    seed = tmp_path / "seed.txt"
    seed.write_text("seed instruction", encoding="utf-8")
    judge_template = tmp_path / "judge.txt"
    judge_template.write_text(JUDGE_TEMPLATE, encoding="utf-8")
    rewrite_template = tmp_path / "rewrite.txt"
    rewrite_template.write_text(REWRITE_TEMPLATE, encoding="utf-8")
    return working_directory, base, seed, judge_template, rewrite_template


def environment_for(tmp_path: Path) -> dict[str, str]:
    return {
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
        "no_proxy": "127.0.0.1,localhost",
    }


def make_launcher(
    launched: list[tuple[str, ...]],
    returncode: int = 0,
    side_effect: Callable[[tuple[str, ...], str], None] | None = None,
) -> RunCommand:
    def run(command: tuple[str, ...], stdin_text: str) -> IO[ProcessResult]:
        launched.append(command)
        if side_effect is not None:
            side_effect(command, stdin_text)
        if returncode != 0:
            return io_pure(ProcessResult(returncode, "", "boom"))
        version = os.path.basename(command[3]).removesuffix(".toml")
        return io_pure(ProcessResult(0, "T[%s] %s" % (version, stdin_text), ""))

    return run


def verdict_body(content: str) -> EndpointReply:
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    return EndpointReply(json.dumps(body), body)


def run_main(
    argv: list[str],
    tmp_path: Path,
    launcher: RunCommand,
    opener: OpenEndpoint | None = None,
) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        argv,
        environment_for(tmp_path),
        stdout,
        stderr,
        lambda seconds: None,
        "py",
        launcher,
        opener,
    ).run()
    return code, stdout.getvalue(), stderr.getvalue()


def evolve_argv(
    base: Path,
    working_directory: Path,
    seed: Path,
    judge_template: Path,
    rewrite_template: Path,
    extra: list[str],
) -> list[str]:
    return [
        "--base-config",
        str(base),
        "--workdir",
        str(working_directory),
        "--seed",
        str(seed),
        "--rounds",
        "1",
        "--stall",
        "2",
        "--judge-model",
        "judge-x",
        "--rewriter-model",
        "rewrite-x",
        "--judge-template",
        str(judge_template),
        "--rewrite-template",
        str(rewrite_template),
        *extra,
    ]


def compare_argv(
    base: Path,
    working_directory: Path,
    prompt_a: Path,
    prompt_b: Path,
    judge_template: Path,
    extra: list[str],
) -> list[str]:
    return [
        "--base-config",
        str(base),
        "--workdir",
        str(working_directory),
        "--compare",
        str(prompt_a),
        str(prompt_b),
        "--judge-model",
        "judge-x",
        "--judge-template",
        str(judge_template),
        *extra,
    ]


def marker_version(text: str) -> int:
    found = re.findall(r"T\[v(\d+)\]", text)
    return int(found[0]) if found else -1

from __future__ import annotations

import io
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from socket import socket
from typing import Any

from http_server import JsonReply, json_reply, local_endpoint, raw_reply

from l2l.effects import ProcessResult
from l2l.errors import HttpError, TranslationError, describe, fail_http
from l2l.monads import IO, Err, Ok, Result, io_pure, io_result
from tools.optimize import (
    Api,
    EndpointReply,
    OpenEndpoint,
    RunCommand,
    curl_endpoint,
    main,
    optimize,
    repo_run,
)

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


def judge_only_failure_opener(
    calls: list[Mapping[str, Any]],
) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        if payload["model"] != "rewrite-x":
            return io_result(fail_http("unreachable", "connection refused", 7))
        content = str(payload["messages"][0]["content"])
        current = content.split("CURRENT: ", 1)[1].split(" HISTORY:", 1)[0].strip()
        return io_result(Ok(verdict_body("better: " + current)))

    return opener


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


def closed_local_port() -> int:
    probe = socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def marker_version(text: str) -> int:
    found = re.findall(r"T\[v(\d+)\]", text)
    return int(found[0]) if found else -1


def judge_verdict_responder(body: bytes, headers: dict[str, str]) -> JsonReply:
    payload = json.loads(body)
    content = str(payload["messages"][0]["content"])
    if payload["model"] == "rewrite-x":
        current = content.split("CURRENT: ", 1)[1].split(" HISTORY:", 1)[0].strip()
        return json_reply(verdict_reply_payload("better: " + current))
    candidate_a_text = content.split(" A: ", 1)[1].split(" ||| B: ", 1)[0]
    candidate_b_text = content.split(" ||| B: ", 1)[1].split(" judge", 1)[0]
    winner = (
        "A"
        if marker_version(candidate_a_text) >= marker_version(candidate_b_text)
        else "B"
    )
    return json_reply(
        verdict_reply_payload(
            json.dumps({"winner": winner, "critique": "picked", "evidence": ["e1"]})
        )
    )


def verdict_reply_payload(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
    }


def tie_responder(body: bytes, headers: dict[str, str]) -> JsonReply:
    return json_reply(
        verdict_reply_payload(
            json.dumps({"winner": "tie", "critique": "even", "evidence": []})
        )
    )


def test_summarize_compare_declares_a_tie(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    with local_endpoint(tie_responder) as endpoint:
        argv = compare_argv(
            base,
            working_directory,
            prompt_a,
            prompt_b,
            judge_template,
            ["--base-url", endpoint.url],
        )
        code, stdout, _stderr = run_main(argv, tmp_path, make_launcher([]))
    assert code == 0
    assert "compare verdict: A wins 0, B wins 0, ties 2 -> tie wins" in stdout
    summaries = list((working_directory / "judge" / "compare").glob("*/summary.json"))
    assert json.loads(summaries[0].read_text(encoding="utf-8"))["winner"] == "tie"


def test_repo_run_executes_a_process_in_the_repository() -> None:
    result = repo_run((sys.executable, "-c", "print('repo-ok')"), "").run()
    assert result.returncode == 0
    assert result.stdout == "repo-ok\n"


def test_evolution_runs_over_a_local_endpoint(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    with local_endpoint(judge_verdict_responder) as endpoint:
        argv = evolve_argv(
            base,
            working_directory,
            seed,
            judge_template,
            rewrite_template,
            ["--base-url", endpoint.url],
        )
        code, stdout, _stderr = run_main(argv, tmp_path, make_launcher([]))
    assert code == 0
    assert (
        "round 1: candidate v1 wins 2, incumbent v0 wins 0, ties 0 -> promoted"
        in stdout
    )
    assert "final incumbent: v1" in stdout
    assert "holdout verdict: v1 wins" in stdout
    assert (working_directory / "prompts" / "v1.txt").read_text(encoding="utf-8") == (
        "better: seed instruction"
    )
    assert (working_directory / "out" / "v1" / "ch1.en.txt").read_text(
        encoding="utf-8"
    ) == "T[v1] source of ch1"
    assert endpoint.requests[0].headers["Authorization"] == "Bearer key"
    assert endpoint.requests[0].path == "/v1/chat/completions"


def test_evolution_fails_when_the_endpoint_is_unreachable(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    argv = evolve_argv(
        base,
        working_directory,
        seed,
        judge_template,
        rewrite_template,
        ["--base-url", "http://127.0.0.1:%d/v1" % closed_local_port()],
    )
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]))
    assert code == 1
    assert "could not reach endpoint" in stderr


def test_evolution_fails_when_judge_calls_are_unreachable(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    argv = evolve_argv(
        base, working_directory, seed, judge_template, rewrite_template, []
    )
    code, _stdout, stderr = run_main(
        argv, tmp_path, make_launcher([]), judge_only_failure_opener([])
    )
    assert code == 1
    assert "could not reach endpoint" in stderr


def test_optimize_entrypoint_exits_nonzero_when_the_endpoint_is_unreachable(
    monkeypatch: Any, tmp_path: Path
) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    argv = [
        "optimize",
        *evolve_argv(
            base,
            working_directory,
            seed,
            judge_template,
            rewrite_template,
            ["--base-url", "http://127.0.0.1:%d/v1" % closed_local_port()],
        ),
    ]
    exits: list[object] = []
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "exit", exits.append)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    optimize()
    assert exits == [1]


def test_optimize_entrypoint_reports_keyboard_interrupt_as_130(
    monkeypatch: Any, tmp_path: Path
) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    argv = [
        "optimize",
        *evolve_argv(
            base,
            working_directory,
            seed,
            judge_template,
            rewrite_template,
            ["--base-url", "http://127.0.0.1:%d/v1" % closed_local_port()],
        ),
    ]
    exits: list[object] = []

    def interrupted(seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(sys, "exit", exits.append)
    monkeypatch.setattr("time.sleep", interrupted)
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    optimize()
    assert exits == [130]


def test_curl_endpoint_round_trips_a_local_server() -> None:
    with local_endpoint(
        lambda body, headers: json_reply(verdict_reply_payload("hello"))
    ) as endpoint:
        opener = curl_endpoint(Api(endpoint.url, "key"), DIRECT_ENVIRONMENT, 5.0)
        outcome = opener({"model": "m"}).run()
    assert isinstance(outcome, Ok)
    assert outcome.value.body == verdict_reply_payload("hello")
    assert json.loads(outcome.value.text)["choices"][0]["message"]["content"] == "hello"


def test_curl_endpoint_reports_a_status_failure_from_a_local_server() -> None:
    with local_endpoint(lambda body, headers: raw_reply(500, b"boom")) as endpoint:
        opener = curl_endpoint(Api(endpoint.url, "key"), DIRECT_ENVIRONMENT, 5.0)
        outcome = opener({"model": "m"}).run()
    assert isinstance(outcome, Err)
    failure = outcome.error
    assert isinstance(failure, HttpError)
    assert failure.status == 500
    assert "boom" in describe(failure)


def test_curl_endpoint_reports_invalid_json_from_a_local_server() -> None:
    with local_endpoint(lambda body, headers: raw_reply(200, b"garbage{")) as endpoint:
        opener = curl_endpoint(Api(endpoint.url, "key"), DIRECT_ENVIRONMENT, 5.0)
        outcome = opener({"model": "m"}).run()
    assert isinstance(outcome, Err)
    assert "invalid JSON" in describe(outcome.error)


def test_curl_endpoint_reports_a_non_object_reply_from_a_local_server() -> None:
    with local_endpoint(lambda body, headers: raw_reply(200, b"[1, 2]")) as endpoint:
        opener = curl_endpoint(Api(endpoint.url, "key"), DIRECT_ENVIRONMENT, 5.0)
        outcome = opener({"model": "m"}).run()
    assert isinstance(outcome, Err)
    assert "non-object endpoint reply" in describe(outcome.error)


def test_curl_endpoint_reports_an_unreachable_local_port() -> None:
    opener = curl_endpoint(
        Api("http://127.0.0.1:%d/v1" % closed_local_port(), "key"),
        DIRECT_ENVIRONMENT,
        5.0,
    )
    outcome = opener({"model": "m"}).run()
    assert isinstance(outcome, Err)
    failure = outcome.error
    assert isinstance(failure, HttpError)
    assert failure.kind == "unreachable"

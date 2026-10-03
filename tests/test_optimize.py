from __future__ import annotations

import io
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from l2l.effects import ProcessResult
from l2l.errors import TranslationError, describe, fail_http
from l2l.monads import IO, Err, Ok, Result, io_pure, io_result
from tools.optimize import (
    ChapterJudging,
    CompareTally,
    EndpointReply,
    OpenEndpoint,
    Options,
    RunCommand,
    Tally,
    chat_payload,
    compact,
    emit_body,
    error_verdict,
    extract_object,
    feedback_lines,
    hash_text,
    history_window,
    load_base_config,
    main,
    order_outcome,
    output_path,
    parse_arguments,
    parse_params,
    record_compare,
    record_judging,
    render_config,
    reply_content,
    resume_state,
    select_chapters,
    strip_reply,
    text_stem,
    toml_value,
    usable_verdict,
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


def workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    workdir = tmp_path / "lab"
    chapters_dir = workdir / "chapters"
    chapters_dir.mkdir(parents=True)
    for name in ("ch1", "ch2"):
        (chapters_dir / (name + ".txt")).write_text(
            "source of " + name, encoding="utf-8"
        )
    (chapters_dir / "holdout.txt").write_text("holdout source", encoding="utf-8")
    base = tmp_path / "base.toml"
    base.write_text(BASE_TOML, encoding="utf-8")
    seed = tmp_path / "seed.txt"
    seed.write_text("seed instruction", encoding="utf-8")
    judge_template = tmp_path / "judge.txt"
    judge_template.write_text(JUDGE_TEMPLATE, encoding="utf-8")
    rewrite_template = tmp_path / "rewrite.txt"
    rewrite_template.write_text(REWRITE_TEMPLATE, encoding="utf-8")
    return workdir, base, seed, judge_template, rewrite_template


def evolve_argv(
    tmp_path: Path,
    workdir: Path,
    base: Path,
    seed: Path,
    judge_template: Path,
    rewrite_template: Path,
) -> list[str]:
    return [
        "--base-config",
        str(base),
        "--workdir",
        str(workdir),
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
    ]


def environment_for(tmp_path: Path) -> dict[str, str]:
    return {"HOME": str(tmp_path), "XDG_CONFIG_HOME": str(tmp_path / "xdg")}


def version_marker(text: str) -> int:
    found = re.findall(r"T\[v(\d+)\]", text)
    return int(found[0]) if found else -1


def pick_winner(content: str) -> str:
    a_text = content.split(" A: ", 1)[1].split(" ||| B: ", 1)[0]
    b_text = content.split(" ||| B: ", 1)[1].split(" judge", 1)[0]
    return "A" if version_marker(a_text) >= version_marker(b_text) else "B"


def verdict_body(reply: str) -> EndpointReply:
    body = {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}
    return EndpointReply(json.dumps(body), body)


def make_opener(
    calls: list[Mapping[str, Any]],
    judge_reply: str = "verdict",
    rewrite_reply: str = "better",
) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        model = str(payload["model"])
        content = str(payload["messages"][0]["content"])
        if model == "rewrite-x":
            current = content.split("CURRENT: ", 1)[1].split(" HISTORY:", 1)[0].strip()
            reply = current if rewrite_reply == "same" else "better: " + current
        elif judge_reply == "verdict":
            reply = json.dumps(
                {
                    "winner": pick_winner(content),
                    "critique": "picked",
                    "evidence": ["e1"],
                }
            )
        else:
            reply = judge_reply
        return io_result(Ok(verdict_body(reply)))

    return opener


def make_launcher(launched: list[tuple[str, ...]], returncode: int = 0) -> RunCommand:
    def run(command: tuple[str, ...], stdin_text: str) -> IO[ProcessResult]:
        launched.append(command)
        version = os.path.basename(command[3]).removesuffix(".toml")
        if returncode != 0:
            return io_pure(ProcessResult(returncode, "", "boom"))
        return io_pure(ProcessResult(0, "T[%s] %s" % (version, stdin_text), ""))

    return run


def run_main(
    argv: list[str],
    tmp_path: Path,
    launcher: RunCommand,
    opener: OpenEndpoint,
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


def test_compact_collapses_whitespace() -> None:
    assert compact("a\n\n b\t c ", 10) == "a b c"


def test_hash_text_is_stable() -> None:
    assert hash_text("abc") == hash_text("abc")
    assert len(hash_text("abc")) == 12


def test_text_stem_and_output_path() -> None:
    assert text_stem("/x/ch1.txt") == "ch1"
    assert text_stem("/x/holdout") == "holdout"
    assert output_path("/lab", "v1", "/lab/chapters/ch1.txt") == os.path.join(
        "/lab", "out", "v1", "ch1.en.txt"
    )


def test_strip_reply_removes_fences() -> None:
    assert strip_reply("```\nbody\n```") == "body"
    assert strip_reply("```python\nbody") == "body"
    assert strip_reply("plain") == "plain"


def test_extract_object_variants() -> None:
    verdict = extract_object('prefix {"winner": "A"} suffix')
    assert isinstance(verdict, Ok)
    assert verdict.value == {"winner": "A"}
    assert isinstance(extract_object("no braces"), Err)
    assert isinstance(extract_object("{not json}"), Err)
    assert isinstance(extract_object("[1, 2]"), Err)


def test_usable_verdict_and_error_verdict() -> None:
    verdict = usable_verdict(Ok({"winner": "B"}))
    assert isinstance(verdict, Ok)
    rejected = usable_verdict(Ok({"winner": "x"}))
    assert isinstance(rejected, Err)
    assert "winner must be A, B, or tie" in describe(rejected.error)
    error = error_verdict("junk reply")
    assert error["winner"] == "error"
    assert "junk reply" in error["critique"]


def test_reply_content_extracts_choice() -> None:
    outcome = reply_content(verdict_body(" hello "))
    assert isinstance(outcome, Ok)
    assert outcome.value == " hello "


def test_reply_content_reports_endpoint_error() -> None:
    body = {"error": {"message": "quota"}}
    outcome = reply_content(EndpointReply(json.dumps(body), body))
    assert isinstance(outcome, Err)
    assert "endpoint error" in describe(outcome.error)


def test_reply_content_hints_on_length_finish() -> None:
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
    outcome = reply_content(EndpointReply(json.dumps(body), body))
    assert isinstance(outcome, Err)
    assert "finish_reason=length" in describe(outcome.error)
    assert "--judge-params" in describe(outcome.error)


def test_order_outcome_and_feedback_lines() -> None:
    assert order_outcome("B", "B") == "candidate"
    assert order_outcome("A", "B") == "incumbent"
    assert order_outcome("tie", "B") == "tie"
    first = {"critique": "c1", "evidence": ["a", "b"]}
    second = {"critique": "c2"}
    lines = feedback_lines("ch1", first, second)
    assert lines[0] == "ch1 [incumbent shown as A] c1 | evidence: a; b"
    assert lines[1] == "ch1 [candidate shown as A] c2 | evidence: none quoted"


def test_toml_value_variants() -> None:
    assert toml_value(True) == Ok("true")
    assert toml_value("x") == Ok('"x"')
    assert toml_value(3) == Ok("3")
    assert toml_value(1.5) == Ok("1.5")
    assert toml_value([1, "a"]) == Ok('[1, "a"]')
    assert isinstance(toml_value(object()), Err)


def test_emit_body_and_render_config() -> None:
    body = emit_body("api", {"base_url": "u", "extra": {"a": 1}, "skip": {}})
    assert isinstance(body, Ok)
    assert body.value[0] == 'base_url = "u"'
    assert "[api.extra]" in body.value
    assert "a = 1" in body.value
    assert not any("skip" in line for line in body.value)
    rendered = render_config(
        {"base_url": "u"},
        {"cache": True},
        ({"name": "translate", "model": "m"},),
    )
    assert isinstance(rendered, Ok)
    assert rendered.value.startswith("[api]\n")
    assert "[options]" in rendered.value
    assert "[[pass]]" in rendered.value
    assert isinstance(render_config({"base_url": object()}, {}, ()), Err)


def test_load_base_config_variants() -> None:
    assert isinstance(load_base_config({}, None, "base.toml"), Err)
    multi: dict[str, Any] = {"pass": [{"name": "a"}, {"name": "b"}]}
    assert isinstance(load_base_config(multi, None, "base.toml"), Err)
    named = load_base_config(multi, "b", "base.toml")
    assert isinstance(named, Ok)
    assert named.value.target_name == "b"
    assert isinstance(load_base_config(multi, "zzz", "base.toml"), Err)
    single = load_base_config({"pass": [{"name": "only"}]}, None, "base.toml")
    assert isinstance(single, Ok)
    assert single.value.target_index == 0


def test_resume_state_from_ledger() -> None:
    ledger = (
        {"decision": "promoted", "candidate": "v1"},
        {"decision": "kept"},
        {"decision": "kept"},
    )
    state = resume_state((), ledger)
    assert state.incumbent == "v1"
    assert state.stall == 2
    assert state.round_no == 4
    fresh = resume_state((), ())
    assert fresh.incumbent == "v0"
    assert fresh.round_no == 1


def test_history_window_truncates() -> None:
    history = tuple({"round": index} for index in range(5))
    window = history_window(history, 2)
    assert json.loads(window.split("\n\n")[0]) == {"round": 3}
    assert history_window((), 2) == "(empty)"


def test_select_chapters_variants() -> None:
    paths = ("/lab/ch1.txt", "/lab/ch2.txt", "/lab/ch3.txt")
    picked = select_chapters(paths, "ch3, ch1", "/lab")
    assert isinstance(picked, Ok)
    assert picked.value == ("/lab/ch3.txt", "/lab/ch1.txt")
    unknown = select_chapters(paths, "ch9", "/lab")
    assert isinstance(unknown, Err)
    assert "unknown chapter" in describe(unknown.error)
    assert isinstance(select_chapters(paths, "  ", "/lab"), Err)


def test_parse_params_variants() -> None:
    assert parse_params(None, "judge-params") == Ok({})
    ok = parse_params('{"a": 1}', "judge-params")
    assert isinstance(ok, Ok)
    assert ok.value == {"a": 1}
    assert isinstance(parse_params("{", "judge-params"), Err)
    not_object = parse_params("[1]", "judge-params")
    assert isinstance(not_object, Err)
    assert "--judge-params must be a JSON object" in describe(not_object.error)


def test_chat_payload_merges_extra() -> None:
    payload = chat_payload("m", "hi", 0.5, 100, {"reasoning": {"effort": "low"}})
    assert payload["model"] == "m"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_parse_arguments_valid_and_invalid() -> None:
    argv = [
        "--base-config",
        "b.toml",
        "--workdir",
        "lab",
        "--judge-model",
        "j",
        "--rewriter-model",
        "r",
        "--judge-params",
        '{"effort": "low"}',
    ]
    parsed = parse_arguments(argv)
    assert isinstance(parsed, Ok)
    assert parsed.value.judge_model == "j"
    assert parsed.value.judge_extra == {"effort": "low"}
    missing = parse_arguments(
        ["--base-config", "b", "--workdir", "l", "--judge-model", "j"]
    )
    assert isinstance(missing, Err)
    assert "--rewriter-model is required" in describe(missing.error)
    zero_rounds = parse_arguments(
        [
            "--base-config",
            "b",
            "--workdir",
            "l",
            "--judge-model",
            "j",
            "--rounds",
            "0",
        ]
    )
    assert isinstance(zero_rounds, Err)


def test_record_judging_and_compare_tallies() -> None:
    judged = ChapterJudging("candidate", {}, {}, ("l1",))
    tally = record_judging(
        record_judging(Tally(), judged),
        judging := ChapterJudging("incumbent", {}, {}, ("l2",)),
    )
    tally = record_judging(tally, ChapterJudging("tie", {}, {}, ("l3",)))
    assert (tally.candidate, tally.incumbent, tally.tie) == (1, 1, 1)
    assert tally.lines == ("l1", "l2", "l3")
    compared = record_compare(
        record_compare(CompareTally(), {"chapter": "ch1"}, "a"),
        {"chapter": "ch2"},
        "b",
    )
    compared = record_compare(compared, {"chapter": "ch3"}, "tie")
    assert (compared.a, compared.b, compared.tie) == (1, 1, 1)
    _ = judging


def test_evolution_flow_promotes_and_checks_holdout(tmp_path: Path) -> None:
    workdir, base, seed, judge_template, rewrite_template = workspace(tmp_path)
    calls: list[Mapping[str, Any]] = []
    launched: list[tuple[str, ...]] = []
    code, stdout, _stderr = run_main(
        evolve_argv(tmp_path, workdir, base, seed, judge_template, rewrite_template),
        tmp_path,
        make_launcher(launched),
        make_opener(calls),
    )
    assert code == 0
    assert (
        "round 1: candidate v1 wins 2, incumbent v0 wins 0, ties 0 -> promoted"
        in stdout
    )
    assert "final incumbent: v1" in stdout
    assert "holdout verdict: v1 wins" in stdout
    ledger = (workdir / "ledger.jsonl").read_text(encoding="utf-8")
    assert json.loads(ledger.splitlines()[0])["decision"] == "promoted"
    history = (workdir / "history.jsonl").read_text(encoding="utf-8")
    assert json.loads(history.splitlines()[0])["version"] == "v1"
    assert (workdir / "prompts" / "v1.txt").read_text(encoding="utf-8") == (
        "better: seed instruction"
    )
    assert (workdir / "out" / "v1" / "ch1.en.txt").read_text(encoding="utf-8") == (
        "T[v1] source of ch1"
    )
    verdict = json.loads(
        (workdir / "judge" / "r1-ch1.json").read_text(encoding="utf-8")
    )
    assert verdict["outcome"] == "candidate"
    feedback = (workdir / "judge" / "r1-feedback.txt").read_text(encoding="utf-8")
    assert "ch1.txt [incumbent shown as A]" in feedback
    judge_calls = [call for call in calls if call["model"] == "judge-x"]
    assert len(judge_calls) == 6
    assert len(launched) == 6


def test_compare_flow_writes_summary(tmp_path: Path) -> None:
    workdir, base, _seed, judge_template, _rewrite_template = workspace(tmp_path)
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    calls: list[Mapping[str, Any]] = []
    first_version = "cmp-" + hash_text("alpha instruction")

    def compare_winner(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        content = str(payload["messages"][0]["content"])
        a_text = content.split(" A: ", 1)[1].split(" ||| B: ", 1)[0]
        winner = "A" if first_version in a_text else "B"
        return io_result(
            Ok(
                verdict_body(
                    json.dumps({"winner": winner, "critique": "ok", "evidence": []})
                )
            )
        )

    code, stdout, _stderr = run_main(
        [
            "--base-config",
            str(base),
            "--workdir",
            str(workdir),
            "--compare",
            str(prompt_a),
            str(prompt_b),
            "--chapters",
            "ch1",
            "--judge-model",
            "judge-x",
            "--judge-template",
            str(judge_template),
        ],
        tmp_path,
        make_launcher([]),
        compare_winner,
    )
    assert code == 0
    assert "chapter ch1.txt: A wins" in stdout
    assert "compare verdict: A wins 1, B wins 0, ties 0 -> A wins" in stdout
    assert "comparing over 1 chapter(s)" in stdout
    summaries = list((workdir / "judge" / "compare").glob("*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["winner"] == "A"
    assert summary["a_wins"] == 1
    chapter_verdict = json.loads(
        (summaries[0].parent / "ch1.json").read_text(encoding="utf-8")
    )
    assert chapter_verdict["outcome"] == "a"


def test_duplicate_candidate_skips_judging(tmp_path: Path) -> None:
    workdir, base, seed, judge_template, rewrite_template = workspace(tmp_path)
    calls: list[Mapping[str, Any]] = []
    code, stdout, _stderr = run_main(
        evolve_argv(tmp_path, workdir, base, seed, judge_template, rewrite_template),
        tmp_path,
        make_launcher([]),
        make_opener(calls, rewrite_reply="same"),
    )
    assert code == 0
    assert "candidate duplicates v0; skipping judging" in stdout
    assert "no challenger was ever promoted; holdout check skipped" in stdout
    ledger = (workdir / "ledger.jsonl").read_text(encoding="utf-8")
    entry = json.loads(ledger.splitlines()[0])
    assert entry["decision"] == "duplicate"
    assert entry["duplicate_of"] == "v0"
    judge_calls = [call for call in calls if call["model"] == "judge-x"]
    assert judge_calls == []


def test_unusable_judge_replies_record_error_verdict(tmp_path: Path) -> None:
    workdir, base, seed, judge_template, rewrite_template = workspace(tmp_path)
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = main(
        evolve_argv(tmp_path, workdir, base, seed, judge_template, rewrite_template),
        environment_for(tmp_path),
        stdout,
        stderr,
        lambda seconds: None,
        "py",
        make_launcher([]),
        make_opener([], judge_reply="definitely not json"),
    ).run()
    assert code == 0
    assert "unusable verdict" in stderr.getvalue()
    assert "giving up; recording an error verdict" in stderr.getvalue()
    assert "ties 2 -> kept" in stdout.getvalue()
    ledger_lines = (workdir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(ledger_lines[0])["decision"] == "kept"


def test_transport_failures_retry_then_fail(tmp_path: Path) -> None:
    workdir, base, seed, judge_template, rewrite_template = workspace(tmp_path)
    attempts: list[Mapping[str, Any]] = []

    def failing_opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        attempts.append(payload)
        return io_result(fail_http("status", "overloaded", 500))

    sleeps: list[float] = []
    code = main(
        evolve_argv(tmp_path, workdir, base, seed, judge_template, rewrite_template),
        environment_for(tmp_path),
        io.StringIO(),
        io.StringIO(),
        sleeps.append,
        "py",
        make_launcher([]),
        failing_opener,
    ).run()
    assert code == 1
    assert len(attempts) == 4
    assert sleeps == [2.0, 4.0, 6.0]


def test_l2l_failure_writes_error_log(tmp_path: Path) -> None:
    workdir, base, seed, judge_template, rewrite_template = workspace(tmp_path)
    code, _stdout, stderr = run_main(
        evolve_argv(tmp_path, workdir, base, seed, judge_template, rewrite_template),
        tmp_path,
        make_launcher([], returncode=2),
        make_opener([]),
    )
    assert code == 1
    assert "l2l exited with 2" in stderr
    error_log = workdir / "out" / "v0" / "ch1.en.txt.err"
    assert error_log.read_text(encoding="utf-8") == "boom"


def test_missing_seed_fails_with_hint(tmp_path: Path) -> None:
    workdir, base, _seed, judge_template, rewrite_template = workspace(tmp_path)
    argv = [
        "--base-config",
        str(base),
        "--workdir",
        str(workdir),
        "--rounds",
        "1",
        "--judge-model",
        "judge-x",
        "--rewriter-model",
        "rewrite-x",
        "--judge-template",
        str(judge_template),
        "--rewrite-template",
        str(rewrite_template),
    ]
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "pass --seed PATH once to create it" in stderr


def test_template_token_validation(tmp_path: Path) -> None:
    workdir, base, seed, _judge_template, rewrite_template = workspace(tmp_path)
    broken = tmp_path / "judge-broken.txt"
    broken.write_text("no tokens here", encoding="utf-8")
    argv = evolve_argv(tmp_path, workdir, base, seed, broken, rewrite_template)
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "judge template" in stderr
    assert "lacks <<<SOURCE>>>" in stderr


def test_workdir_file_rejected(tmp_path: Path) -> None:
    _workdir, base, _seed, judge_template, rewrite_template = workspace(tmp_path)
    not_a_dir = tmp_path / "file-lab"
    not_a_dir.write_text("x", encoding="utf-8")
    argv = [
        "--base-config",
        str(base),
        "--workdir",
        str(not_a_dir),
        "--seed",
        str(tmp_path / "seed.txt"),
        "--judge-model",
        "judge-x",
        "--rewriter-model",
        "rewrite-x",
        "--judge-template",
        str(judge_template),
        "--rewrite-template",
        str(rewrite_template),
    ]
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "workdir is not a directory" in stderr


def test_missing_base_config(tmp_path: Path) -> None:
    _workdir, _base, _seed, judge_template, rewrite_template = workspace(tmp_path)
    argv = [
        "--base-config",
        str(tmp_path / "absent.toml"),
        "--workdir",
        str(tmp_path / "lab2"),
        "--judge-model",
        "judge-x",
        "--rewriter-model",
        "rewrite-x",
        "--judge-template",
        str(judge_template),
        "--rewrite-template",
        str(rewrite_template),
    ]
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "base config not found" in stderr


def test_resolve_api_reports_missing_credentials(tmp_path: Path) -> None:
    from tools.optimize import resolve_api

    options = Options(
        base_config="b",
        workdir=str(tmp_path),
        seed=None,
        rounds=1,
        stall=1,
        judge_model="j",
        rewrite_model=None,
        translator_model=None,
        compare=None,
        holdout=False,
        chapters=None,
        pass_name=None,
        base_url=None,
        api_key=None,
        l2l=None,
        judge_template="t",
        rewrite_template="t",
        judge_temperature=0.0,
        rewrite_temperature=0.0,
        translator_temperature=0.0,
        call_timeout=1.0,
        call_max_tokens=10,
        history_depth=1,
        judge_extra={},
        rewrite_extra={},
    )
    outcome = resolve_api(options, {}, environment_for(tmp_path)).run()
    assert isinstance(outcome, Err)
    assert "no endpoint credentials" in describe(outcome.error)

    resolved = resolve_api(
        options,
        {"api": {"base_url": "http://x/v1/", "api_key": "k"}},
        environment_for(tmp_path),
    ).run()
    assert isinstance(resolved, Ok)
    assert resolved.value.base_url == "http://x/v1"
    assert resolved.value.api_key == "k"

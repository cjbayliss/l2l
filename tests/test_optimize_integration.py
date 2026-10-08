from __future__ import annotations

import io
import json
import os
import re
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import replace
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
    OptimizationEnvironment,
    Options,
    Reporter,
    RoundInputs,
    RunCommand,
    RunState,
    Tally,
    candidate_prompt,
    compare_chapters,
    conclude_round,
    curl_endpoint,
    discover_chapters,
    hash_text,
    holdout_check,
    judge_call,
    judge_step,
    load_base_config,
    main,
    optimize,
    prepare_version,
    previous_critiques,
    read_jsonl,
    record_duplicate,
    repo_run,
    resolve_api,
    rewrite_prompt,
    round_texts,
    strip_reply,
    translate_chapter,
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


def options_for(
    working_directory: Path,
    base: Path,
    seed: Path,
    judge_template: Path,
    rewrite_template: Path,
) -> Options:
    return Options(
        base_config=str(base),
        working_directory=str(working_directory),
        seed=str(seed),
        rounds=1,
        stall=2,
        judge_model="judge-x",
        rewrite_model="rewrite-x",
        translator_model=None,
        compare=None,
        holdout=False,
        chapters=None,
        pass_name=None,
        base_url=None,
        api_key=None,
        l2l=None,
        judge_template=str(judge_template),
        rewrite_template=str(rewrite_template),
        judge_temperature=0.0,
        rewrite_temperature=0.0,
        translator_temperature=0.0,
        call_timeout=1.0,
        call_maximum_tokens=10,
        history_depth=1,
        judge_extra={},
        rewrite_extra={},
    )


def record_into(lines: list[str]) -> Reporter:
    def report(line: str) -> IO[None]:
        return IO(lambda: lines.append(line))

    return report


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


def make_opener(
    calls: list[Mapping[str, Any]],
    judge_reply: str = "verdict",
) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        content = str(payload["messages"][0]["content"])
        if payload["model"] == "rewrite-x":
            current = content.split("CURRENT: ", 1)[1].split(" HISTORY:", 1)[0].strip()
            return io_result(Ok(verdict_body("better: " + current)))
        return io_result(Ok(verdict_body(judge_reply)))

    return opener


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


def candidate_side_opener(
    calls: list[Mapping[str, Any]], candidate_version: str
) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        content = str(payload["messages"][0]["content"])
        candidate_a_text = content.split(" A: ", 1)[1].split(" ||| B: ", 1)[0]
        winner = "A" if candidate_version in candidate_a_text else "B"
        return io_result(
            Ok(
                verdict_body(
                    json.dumps(
                        {"winner": winner, "critique": "picked", "evidence": ["e1"]}
                    )
                )
            )
        )

    return opener


def failing_opener(calls: list[Mapping[str, Any]]) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        return io_result(fail_http("unreachable", "connection refused", 7))

    return opener


def marker_judge_opener(calls: list[Mapping[str, Any]]) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        content = str(payload["messages"][0]["content"])
        candidate_a_text = content.split(" A: ", 1)[1].split(" ||| B: ", 1)[0]
        candidate_b_text = content.split(" ||| B: ", 1)[1].split(" judge", 1)[0]
        winner = (
            "A"
            if marker_version(candidate_a_text) >= marker_version(candidate_b_text)
            else "B"
        )
        return io_result(
            Ok(
                verdict_body(
                    json.dumps(
                        {"winner": winner, "critique": "picked", "evidence": ["e1"]}
                    )
                )
            )
        )

    return opener


def build_environment(
    working_directory: Path,
    options: Options,
    opener: OpenEndpoint | None = None,
    launcher: RunCommand | None = None,
    chapters: tuple[str, ...] = (),
) -> OptimizationEnvironment:
    built = load_base_config(tomllib.loads(BASE_TOML), None, "base.toml")
    assert isinstance(built, Ok)
    return OptimizationEnvironment(
        options=options,
        base=built.value,
        api=Api("http://endpoint.test/v1", "key"),
        working_directory=str(working_directory),
        environment=DIRECT_ENVIRONMENT,
        chapters=chapters,
        judge_template=JUDGE_TEMPLATE,
        rewrite_template=REWRITE_TEMPLATE,
        l2l_command=("py", "-m", "l2l"),
        opener=opener if opener is not None else make_opener([]),
        sleep=lambda seconds: None,
        say=record_into([]),
        warn=record_into([]),
        run_command=launcher if launcher is not None else make_launcher([]),
    )


def seeded_environment(tmp_path: Path) -> tuple[Path, OptimizationEnvironment]:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    options = options_for(
        working_directory, base, seed, judge_template, rewrite_template
    )
    return working_directory, build_environment(working_directory, options)


def write_generation_configs(
    working_directory: Path, versions: tuple[str, ...]
) -> None:
    generation_directory = working_directory / "gen"
    generation_directory.mkdir(parents=True, exist_ok=True)
    for version in versions:
        (generation_directory / (version + ".toml")).write_text(
            BASE_TOML, encoding="utf-8"
        )


def prepared_holdout(tmp_path: Path) -> tuple[Path, OptimizationEnvironment]:
    working_directory, environment = seeded_environment(tmp_path)
    (working_directory / "judge").mkdir(parents=True, exist_ok=True)
    prompts = working_directory / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "v0.txt").write_text("seed instruction", encoding="utf-8")
    (prompts / "v1.txt").write_text("final instruction", encoding="utf-8")
    write_generation_configs(working_directory, ("v0", "v1"))
    for version in ("v0", "v1"):
        output_directory = working_directory / "out" / version
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "holdout.en.txt").write_text(
            "T[%s] holdout source" % version, encoding="utf-8"
        )
    return working_directory, environment


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


def test_discover_chapters_reports_directory_and_listing_failures(
    tmp_path: Path,
) -> None:
    absent = discover_chapters(str(tmp_path / "chapters")).run()
    assert isinstance(absent, Err)
    assert "expected chapter .txt files plus holdout.txt" in describe(absent.error)

    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / "ch1.txt").write_text("solo", encoding="utf-8")
    one_chapter = discover_chapters(str(lonely)).run()
    assert isinstance(one_chapter, Err)
    assert "need at least two chapters" in describe(one_chapter.error)

    unheld = tmp_path / "unheld"
    unheld.mkdir()
    for name in ("ch1", "ch2"):
        (unheld / (name + ".txt")).write_text("source", encoding="utf-8")
    missing_holdout = discover_chapters(str(unheld)).run()
    assert isinstance(missing_holdout, Err)
    assert describe(missing_holdout.error).endswith("holdout.txt")


def test_compare_chapters_selects_holdout_or_filtered_chapters(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    options = options_for(
        working_directory, base, seed, judge_template, rewrite_template
    )

    holdout_only = compare_chapters(
        str(working_directory), replace(options, holdout=True)
    ).run()
    assert isinstance(holdout_only, Ok)
    assert holdout_only.value == (str(working_directory / "chapters" / "holdout.txt"),)

    (working_directory / "chapters" / "holdout.txt").unlink()
    absent_holdout = compare_chapters(
        str(working_directory), replace(options, holdout=True)
    ).run()
    assert isinstance(absent_holdout, Err)
    assert "holdout.txt" in describe(absent_holdout.error)
    (working_directory / "chapters" / "holdout.txt").write_text(
        "holdout source", encoding="utf-8"
    )

    unheld = compare_chapters(
        str(working_directory), replace(options, holdout=False, chapters=None)
    ).run()
    assert isinstance(unheld, Ok)
    assert unheld.value == (
        str(working_directory / "chapters" / "ch1.txt"),
        str(working_directory / "chapters" / "ch2.txt"),
    )

    filtered = compare_chapters(
        str(working_directory), replace(options, chapters="ch2, ch1")
    ).run()
    assert isinstance(filtered, Ok)
    assert filtered.value == (
        str(working_directory / "chapters" / "ch2.txt"),
        str(working_directory / "chapters" / "ch1.txt"),
    )

    unknown = compare_chapters(
        str(working_directory), replace(options, chapters="ch9")
    ).run()
    assert isinstance(unknown, Err)
    assert "unknown chapter" in describe(unknown.error)

    bare = tmp_path / "bare"
    bare.mkdir()
    missing_chapters = compare_chapters(str(bare), options).run()
    assert isinstance(missing_chapters, Err)
    assert "expected chapter .txt files plus holdout.txt" in describe(
        missing_chapters.error
    )


def test_read_jsonl_reads_a_ledger_and_tolerates_a_missing_file(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"decision": "promoted"})
        + "\n"
        + json.dumps({"decision": "kept"})
        + "\n",
        encoding="utf-8",
    )
    entries = read_jsonl(str(ledger)).run()
    assert entries == ({"decision": "promoted"}, {"decision": "kept"})
    assert read_jsonl(str(tmp_path / "absent.jsonl")).run() == ()


def test_strip_reply_handles_unterminated_fences() -> None:
    assert strip_reply("```python\nbetter body") == "better body"
    assert strip_reply("```") == ""


def blank_rewrite_opener(
    calls: list[Mapping[str, Any]],
) -> OpenEndpoint:
    def opener(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        return io_result(Ok(verdict_body("```\n```")))

    return opener


def test_rewrite_prompt_rejects_an_empty_instruction(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    outcome = rewrite_prompt(
        replace(environment, opener=blank_rewrite_opener([])),
        "current",
        "history",
        "critiques",
        1,
    ).run()
    assert isinstance(outcome, Err)
    assert "rewriter returned an empty instruction" in describe(outcome.error)


def test_prepare_version_reports_an_instruction_write_failure(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    prompts = working_directory / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "v1.txt").mkdir()
    outcome = prepare_version(environment, "v1", "evolved instruction").run()
    assert isinstance(outcome, Err)
    assert "cannot write instruction file" in describe(outcome.error)


def test_complete_warns_sleeps_and_retries_until_success(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    calls: list[Mapping[str, Any]] = []

    def flaky(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        calls.append(payload)
        if len(calls) == 1:
            return io_result(fail_http("unreachable", "connection reset", 7))
        return io_result(
            Ok(
                verdict_body(
                    json.dumps({"winner": "tie", "critique": "ok", "evidence": []})
                )
            )
        )

    retrying = replace(environment, opener=flaky)
    outcome = judge_call(retrying, "judge-x", "prompt", "label").run()
    assert isinstance(outcome, Ok)
    assert outcome.value == {"winner": "tie", "critique": "ok", "evidence": []}
    assert len(calls) == 2


def test_judge_call_gives_up_after_repeated_transport_failures(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    calls: list[Mapping[str, Any]] = []
    outcome = judge_call(
        replace(environment, opener=failing_opener(calls)),
        "judge-x",
        "prompt",
        "label",
    ).run()
    assert isinstance(outcome, Err)
    assert "connection refused" in describe(outcome.error)
    assert len(calls) == 4


def test_previous_critiques_reads_the_previous_round_feedback(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    feedback = working_directory / "judge" / "r0-feedback.txt"
    feedback.parent.mkdir(parents=True, exist_ok=True)
    feedback.write_text("earlier critique", encoding="utf-8")
    assert previous_critiques(environment, 1).run() == "earlier critique"

    feedback.unlink()
    assert previous_critiques(environment, 1).run() == "(none; this is the first round)"

    feedback.mkdir()
    assert previous_critiques(environment, 1).run() == "(none; this is the first round)"


def test_candidate_prompt_reuses_an_existing_candidate_file(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    calls: list[Mapping[str, Any]] = []
    said: list[str] = []
    reusing = replace(environment, opener=make_opener(calls), say=record_into(said))
    prompts = working_directory / "prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / "v1.txt").write_text("already written", encoding="utf-8")
    state = RunState(round_no=1, incumbent="v0", stall=0, history=())
    outcome = candidate_prompt(reusing, state, "current", "history", "critiques").run()
    assert isinstance(outcome, Ok)
    assert outcome.value == "already written"
    assert any("reusing" in line for line in said)
    assert calls == []


def test_candidate_prompt_reports_a_write_failure(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    state = RunState(round_no=1, incumbent="v0", stall=0, history=())
    outcome = candidate_prompt(
        environment, state, "current", "history", "critiques"
    ).run()
    assert isinstance(outcome, Err)
    assert "cannot write candidate instruction" in describe(outcome.error)


def test_prepare_version_swaps_the_translator_model(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    options = options_for(
        working_directory, base, seed, judge_template, rewrite_template
    )
    environment = build_environment(
        working_directory, replace(options, translator_model="trans-x")
    )
    outcome = prepare_version(environment, "v1", "evolved instruction").run()
    assert isinstance(outcome, Ok)
    digest = outcome.value
    generated = tomllib.loads(
        (working_directory / "gen" / "v1.toml").read_text(encoding="utf-8")
    )
    entry = generated["pass"][0]
    assert entry["model"] == "trans-x"
    assert entry["name"] == "translate-" + digest[:10]
    assert entry["instruction_file"] == "../prompts/v1.txt"
    assert (working_directory / "prompts" / "v1.txt").read_text(encoding="utf-8") == (
        "evolved instruction"
    )


def test_prepare_version_reports_a_generated_config_write_failure(
    tmp_path: Path,
) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    generation = working_directory / "gen"
    generation.mkdir(parents=True, exist_ok=True)
    (generation / "v1.toml").mkdir()
    outcome = prepare_version(environment, "v1", "evolved instruction").run()
    assert isinstance(outcome, Err)
    assert "cannot write generated config" in describe(outcome.error)


def test_translate_chapter_reports_a_partial_write_failure(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    prepared = prepare_version(environment, "v1", "instruction").run()
    assert isinstance(prepared, Ok)
    target_directory = working_directory / "out" / "v1"
    target_directory.mkdir(parents=True, exist_ok=True)
    (target_directory / "ch1.en.txt.partial").mkdir()
    chapter = str(working_directory / "chapters" / "ch1.txt")
    outcome = translate_chapter(environment, "v1", chapter).run()
    assert isinstance(outcome, Err)
    assert "cannot write partial output" in describe(outcome.error)


def test_translate_chapter_reports_a_failed_finalization(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    prepared = prepare_version(environment, "v1", "instruction").run()
    assert isinstance(prepared, Ok)
    target = working_directory / "out" / "v1" / "ch1.en.txt"

    def hijack(command: tuple[str, ...], stdin_text: str) -> None:
        target.mkdir(parents=True, exist_ok=True)

    obstructed = replace(environment, run_command=make_launcher([], side_effect=hijack))
    chapter = str(working_directory / "chapters" / "ch1.txt")
    outcome = translate_chapter(obstructed, "v1", chapter).run()
    assert isinstance(outcome, Err)
    assert "could not finalize" in describe(outcome.error)


def test_translate_chapter_reports_a_missing_generated_config(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    chapter = str(working_directory / "chapters" / "ch1.txt")
    outcome = translate_chapter(environment, "v1", chapter).run()
    assert isinstance(outcome, Err)
    assert "missing generated config" in describe(outcome.error)


def test_round_texts_reports_missing_translations(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    absent_chapter = str(working_directory / "chapters" / "absent.txt")
    missing_source = round_texts(environment, "v0", "v1", absent_chapter).run()
    assert isinstance(missing_source, Err)
    assert "chapter not found" in describe(missing_source.error)

    chapter = str(working_directory / "chapters" / "ch1.txt")
    missing_translations = round_texts(environment, "v0", "v1", chapter).run()
    assert isinstance(missing_translations, Err)
    assert "translation not found" in describe(missing_translations.error)


def test_judge_step_reports_judging_and_verdict_failures(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    chapter = str(working_directory / "chapters" / "ch1.txt")
    for version in ("v0", "v1"):
        output_directory = working_directory / "out" / version
        output_directory.mkdir(parents=True, exist_ok=True)
        (output_directory / "ch1.en.txt").write_text(
            "T[%s] source of ch1" % version, encoding="utf-8"
        )
    calls: list[Mapping[str, Any]] = []
    judged = judge_step(replace(environment, opener=failing_opener(calls)), 1, "v0")(
        Tally(), chapter
    ).run()
    assert isinstance(judged, Err)
    assert "connection refused" in describe(judged.error)

    verdict_directory = working_directory / "judge"
    verdict_directory.mkdir(parents=True, exist_ok=True)
    (verdict_directory / "r1-ch1.json").mkdir()
    written = judge_step(environment, 1, "v0")(Tally(), chapter).run()
    assert isinstance(written, Err)
    assert "cannot write verdict file" in describe(written.error)


def test_judge_step_reports_missing_texts(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    chapter = str(working_directory / "chapters" / "ch1.txt")
    outcome = judge_step(environment, 1, "v0")(Tally(), chapter).run()
    assert isinstance(outcome, Err)
    assert "translation not found" in describe(outcome.error)


def test_record_duplicate_reports_ledger_and_history_failures(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    state = RunState(round_no=1, incumbent="v0", stall=0, history=())
    inputs = RoundInputs("current", "candidate", "digest")
    (working_directory / "ledger.jsonl").mkdir()
    blocked = record_duplicate(environment, state, inputs, "v0").run()
    assert isinstance(blocked, Err)
    assert "cannot append ledger" in describe(blocked.error)

    (working_directory / "ledger.jsonl").rmdir()
    (working_directory / "history.jsonl").mkdir()
    blocked_history = record_duplicate(environment, state, inputs, "v0").run()
    assert isinstance(blocked_history, Err)
    assert "cannot append history" in describe(blocked_history.error)


def test_conclude_round_reports_write_failures(tmp_path: Path) -> None:
    working_directory, environment = seeded_environment(tmp_path)
    state = RunState(round_no=1, incumbent="v0", stall=0, history=())
    inputs = RoundInputs("current", "candidate", "digest")
    tally = Tally(candidate=1, incumbent=0, tie=0, lines=("line",))
    judge_directory = working_directory / "judge"
    judge_directory.mkdir(parents=True, exist_ok=True)
    (judge_directory / "r1-feedback.txt").mkdir()
    blocked_feedback = conclude_round(environment, state, inputs, tally).run()
    assert isinstance(blocked_feedback, Err)
    assert "cannot write round feedback" in describe(blocked_feedback.error)

    (judge_directory / "r1-feedback.txt").rmdir()
    (working_directory / "ledger.jsonl").mkdir()
    blocked_ledger = conclude_round(environment, state, inputs, tally).run()
    assert isinstance(blocked_ledger, Err)
    assert "cannot append ledger" in describe(blocked_ledger.error)

    (working_directory / "ledger.jsonl").rmdir()
    (working_directory / "history.jsonl").mkdir()
    blocked_history = conclude_round(environment, state, inputs, tally).run()
    assert isinstance(blocked_history, Err)
    assert "cannot append history" in describe(blocked_history.error)


def test_holdout_check_reports_judging_and_report_failures(tmp_path: Path) -> None:
    working_directory, environment = prepared_holdout(tmp_path)
    calls: list[Mapping[str, Any]] = []
    outcome = holdout_check(
        replace(environment, opener=failing_opener(calls)), "v1"
    ).run()
    assert isinstance(outcome, Err)
    assert "connection refused" in describe(outcome.error)

    judge_directory = working_directory / "judge"
    judge_directory.mkdir(parents=True, exist_ok=True)
    (judge_directory / "holdout.json").mkdir()
    blocked = holdout_check(replace(environment, opener=make_opener(calls)), "v1").run()
    assert isinstance(blocked, Err)
    assert "cannot write holdout verdict" in describe(blocked.error)


def test_holdout_check_reports_missing_translations(tmp_path: Path) -> None:
    working_directory, environment = prepared_holdout(tmp_path)
    unreadable = working_directory / "out" / "v1" / "holdout.en.txt"
    unreadable.unlink()
    unreadable.mkdir()
    outcome = holdout_check(environment, "v1").run()
    assert isinstance(outcome, Err)
    assert "cannot read translation" in describe(outcome.error)


def test_holdout_check_writes_the_verdict(tmp_path: Path) -> None:
    working_directory, environment = prepared_holdout(tmp_path)
    outcome = holdout_check(
        replace(environment, opener=marker_judge_opener([])), "v1"
    ).run()
    assert isinstance(outcome, Ok)
    report = json.loads(
        (working_directory / "judge" / "holdout.json").read_text(encoding="utf-8")
    )
    assert report["outcome"] == "candidate"
    assert report["seed_as_a"]["winner"] == "B"
    assert report["final_as_a"]["winner"] == "A"


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


def test_compare_step_reports_judging_and_verdict_failures(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    argv = compare_argv(base, working_directory, prompt_a, prompt_b, judge_template, [])
    code, _stdout, stderr = run_main(
        argv, tmp_path, make_launcher([]), failing_opener([])
    )
    assert code == 1
    assert "could not reach endpoint" in stderr

    versions = (
        "cmp-" + hash_text("alpha instruction"),
        "cmp-" + hash_text("beta instruction"),
    )
    verdict_directory = (
        working_directory / "judge" / "compare" / (versions[0] + "-vs-" + versions[1])
    )
    verdict_directory.mkdir(parents=True, exist_ok=True)
    (verdict_directory / "ch1.json").mkdir()
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "cannot write chapter verdict" in stderr


def test_compare_step_reports_missing_translations(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    version_a = "cmp-" + hash_text("alpha instruction")
    obstructed = working_directory / "out" / version_a
    obstructed.mkdir(parents=True)
    (obstructed / "ch1.en.txt").mkdir()
    argv = compare_argv(base, working_directory, prompt_a, prompt_b, judge_template, [])
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "cannot read translation" in stderr


def test_compare_finds_b_wins(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    argv = compare_argv(base, working_directory, prompt_a, prompt_b, judge_template, [])
    code, stdout, _stderr = run_main(
        argv,
        tmp_path,
        make_launcher([]),
        candidate_side_opener([], "cmp-" + hash_text("beta instruction")),
    )
    assert code == 0
    assert "compare verdict: A wins 0, B wins 2, ties 0 -> B wins" in stdout


def test_evolution_reports_missing_chapters(tmp_path: Path) -> None:
    _workdir, base, seed, judge_template, rewrite_template = workspace(tmp_path)
    empty_workdir = tmp_path / "chapterless"
    empty_workdir.mkdir()
    argv = evolve_argv(base, empty_workdir, seed, judge_template, rewrite_template, [])
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "expected chapter .txt files plus holdout.txt" in stderr


def test_evolution_reports_a_missing_judge_template(tmp_path: Path) -> None:
    working_directory, base, seed, _judge_template, rewrite_template = workspace(
        tmp_path
    )
    absent = tmp_path / "absent-judge.txt"
    argv = evolve_argv(base, working_directory, seed, absent, rewrite_template, [])
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "judge template not found" in stderr


def test_evolution_reports_missing_endpoint_credentials(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    credentialless = tmp_path / "credentialless.toml"
    credentialless.write_text(
        '[[pass]]\nname = "translate"\nmodel = "m"\ninstruction = "go"\n',
        encoding="utf-8",
    )
    argv = evolve_argv(
        credentialless,
        working_directory,
        seed,
        judge_template,
        rewrite_template,
        [],
    )
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "no endpoint credentials" in stderr


def test_summarize_compare_reports_a_summary_write_failure(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    versions = (
        "cmp-" + hash_text("alpha instruction"),
        "cmp-" + hash_text("beta instruction"),
    )
    verdict_directory = (
        working_directory / "judge" / "compare" / (versions[0] + "-vs-" + versions[1])
    )
    verdict_directory.mkdir(parents=True)
    (verdict_directory / "summary.json").mkdir()
    argv = compare_argv(base, working_directory, prompt_a, prompt_b, judge_template, [])
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "cannot write compare summary" in stderr


def test_compare_rejects_identical_prompts(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("same instruction", encoding="utf-8")
    argv = compare_argv(base, working_directory, prompt_a, prompt_a, judge_template, [])
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "both prompts are identical" in stderr


def test_compare_reports_a_missing_prompt_file(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, _rewrite_template = workspace(
        tmp_path
    )
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    argv = compare_argv(
        base,
        working_directory,
        prompt_a,
        tmp_path / "absent.txt",
        judge_template,
        [],
    )
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "prompt file not found" in stderr


def test_resolve_api_reads_the_user_config_table(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    options = options_for(
        working_directory, base, seed, judge_template, rewrite_template
    )
    user_config = tmp_path / "xdg" / "l2l" / "config.toml"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(
        'base_url = "http://user.test/v1"\napi_key = "user-key"\n',
        encoding="utf-8",
    )
    resolved = resolve_api(options, {}, environment_for(tmp_path)).run()
    assert isinstance(resolved, Ok)
    assert resolved.value == Api("http://user.test/v1", "user-key")

    user_config.write_text("not toml {{{", encoding="utf-8")
    broken = resolve_api(options, {}, environment_for(tmp_path)).run()
    assert isinstance(broken, Err)
    assert "cannot parse user config" in describe(broken.error)


def test_repo_run_executes_a_process_in_the_repository() -> None:
    result = repo_run((sys.executable, "-c", "print('repo-ok')"), "").run()
    assert result.returncode == 0
    assert result.stdout == "repo-ok\n"


def test_start_rounds_reports_a_seed_copy_failure(tmp_path: Path) -> None:
    working_directory, base, _seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    seed_directory = tmp_path / "seed-directory"
    seed_directory.mkdir()
    argv = evolve_argv(
        base, working_directory, seed_directory, judge_template, rewrite_template, []
    )
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "could not copy seed instruction file" in stderr


def test_compare_mode_reports_missing_chapters(tmp_path: Path) -> None:
    _workdir, base, _seed, judge_template, _rewrite_template = workspace(tmp_path)
    empty_workdir = tmp_path / "empty-lab"
    empty_workdir.mkdir()
    prompt_a = tmp_path / "a.txt"
    prompt_a.write_text("alpha instruction", encoding="utf-8")
    prompt_b = tmp_path / "b.txt"
    prompt_b.write_text("beta instruction", encoding="utf-8")
    argv = compare_argv(base, empty_workdir, prompt_a, prompt_b, judge_template, [])
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "expected chapter .txt files plus holdout.txt" in stderr


def test_evolution_reports_a_broken_rewrite_template(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    absent = tmp_path / "absent-rewrite.txt"
    code, _stdout, stderr = run_main(
        evolve_argv(base, working_directory, seed, judge_template, absent, []),
        tmp_path,
        make_launcher([]),
        make_opener([]),
    )
    assert code == 1
    assert "rewrite template not found" in stderr

    broken = tmp_path / "broken-rewrite.txt"
    broken.write_text("no tokens here", encoding="utf-8")
    code, _stdout, stderr = run_main(
        evolve_argv(base, working_directory, seed, judge_template, broken, []),
        tmp_path,
        make_launcher([]),
        make_opener([]),
    )
    assert code == 1
    assert "rewrite template" in stderr
    assert "lacks <<<CURRENT_PROMPT>>>" in stderr


def test_evolution_reports_base_config_failures(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    broken = tmp_path / "broken.toml"
    broken.write_text("not toml {{{", encoding="utf-8")
    code, _stdout, stderr = run_main(
        evolve_argv(
            broken, working_directory, seed, judge_template, rewrite_template, []
        ),
        tmp_path,
        make_launcher([]),
        make_opener([]),
    )
    assert code == 1
    assert "cannot parse base config" in stderr

    multi = tmp_path / "multi.toml"
    multi.write_text(
        BASE_TOML + '\n[[pass]]\nname = "second"\nmodel = "m"\n', encoding="utf-8"
    )
    code, _stdout, stderr = run_main(
        evolve_argv(
            multi, working_directory, seed, judge_template, rewrite_template, []
        ),
        tmp_path,
        make_launcher([]),
        make_opener([]),
    )
    assert code == 1
    assert "has 2 passes; pass --pass-name to pick one" in stderr


def test_main_reports_argument_check_failures(tmp_path: Path) -> None:
    working_directory, base, seed, judge_template, rewrite_template = workspace(
        tmp_path
    )
    argv = evolve_argv(
        base,
        working_directory,
        seed,
        judge_template,
        rewrite_template,
        ["--rounds", "0"],
    )
    code, _stdout, stderr = run_main(argv, tmp_path, make_launcher([]), make_opener([]))
    assert code == 1
    assert "--rounds must be at least 1" in stderr


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

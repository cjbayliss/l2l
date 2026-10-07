from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, TextIO

from l2l.effects import (
    ProcessResult,
    Sleep,
    append_text_file,
    copy_file,
    ensure_directory,
    entry_paths,
    load_toml,
    path_exists,
    path_is_directory,
    read_text_file,
    replace_file,
    run_process,
    user_config_path,
    write_stdout,
    write_text_file,
)
from l2l.errors import (
    ConfigError,
    TranslationError,
    describe,
    fail_config,
    fail_http,
)
from l2l.http import HttpRequest, curl_open, http_request, parse_json_body
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_sequence,
    io_traverse,
    io_when_unit,
    result_bind,
    result_bind_io,
    result_either,
    result_map,
    result_or_else,
    result_sequence,
)
from l2l.settings import Config

SOURCE_TOKEN = "<<<SOURCE>>>"
A_TOKEN = "<<<TRANSLATION_A>>>"
B_TOKEN = "<<<TRANSLATION_B>>>"
PROMPT_TOKEN = "<<<CURRENT_PROMPT>>>"
HISTORY_TOKEN = "<<<HISTORY>>>"
CRITIQUES_TOKEN = "<<<CRITIQUES>>>"

TOOLS_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TOOLS_DIRECTORY)
SEED_VERSION = "v0"
CHAT_ATTEMPTS = 4
VERDICT_ATTEMPTS = 3

type OpenEndpoint = Callable[
    [Mapping[str, Any]], IO[Result[EndpointReply, TranslationError]]
]
type Reporter = Callable[[str], IO[None]]
type RunCommand = Callable[[tuple[str, ...], str], IO[ProcessResult]]


@dataclass(frozen=True)
class Api:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class EndpointReply:
    text: str
    body: Mapping[str, Any]


@dataclass(frozen=True)
class BaseConfig:
    api: Mapping[str, Any]
    options: Mapping[str, Any]
    passes: tuple[Mapping[str, Any], ...]
    target_index: int
    target_name: str


@dataclass(frozen=True)
class Options:
    base_config: str
    working_directory: str
    seed: str | None
    rounds: int
    stall: int
    judge_model: str
    rewrite_model: str | None
    translator_model: str | None
    compare: tuple[str, str] | None
    holdout: bool
    chapters: str | None
    pass_name: str | None
    base_url: str | None
    api_key: str | None
    l2l: str | None
    judge_template: str
    rewrite_template: str
    judge_temperature: float
    rewrite_temperature: float
    translator_temperature: float
    call_timeout: float
    call_maximum_tokens: int
    history_depth: int
    judge_extra: Mapping[str, Any]
    rewrite_extra: Mapping[str, Any]


@dataclass(frozen=True)
class OptimizationEnvironment:
    options: Options
    base: BaseConfig
    api: Api
    working_directory: str
    environment: Mapping[str, str]
    chapters: tuple[str, ...]
    judge_template: str
    rewrite_template: str
    l2l_command: tuple[str, ...]
    opener: OpenEndpoint
    sleep: Sleep
    say: Reporter
    warn: Reporter
    run_command: RunCommand


@dataclass(frozen=True)
class RoundInputs:
    current_prompt: str
    candidate_prompt: str
    digest: str


@dataclass(frozen=True)
class RunState:
    round_no: int
    incumbent: str
    stall: int
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Tally:
    candidate: int = 0
    incumbent: int = 0
    tie: int = 0
    lines: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChapterJudging:
    outcome: str
    first: dict[str, Any]
    second: dict[str, Any]
    lines: tuple[str, ...]


@dataclass(frozen=True)
class CompareTally:
    a_wins: int = 0
    b_wins: int = 0
    tie: int = 0
    rounds: tuple[dict[str, Any], ...] = ()


def compact(text: str, limit: int = 400) -> str:
    return " ".join(text.split())[:limit]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def join_directory(working_directory: str, name: str) -> str:
    return os.path.join(working_directory, name)


def text_stem(path: str) -> str:
    return os.path.basename(path).removesuffix(".txt")


def output_path(working_directory: str, version: str, chapter: str) -> str:
    return os.path.join(
        join_directory(join_directory(working_directory, "out"), version),
        text_stem(chapter) + ".en.txt",
    )


def chat_payload(
    model: str,
    content: str,
    temperature: float,
    maximum_tokens: int,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": maximum_tokens,
        **dict(extra),
    }


def endpoint_error(reply: EndpointReply) -> str | None:
    reported = reply.body.get("error")
    return None if reported is None else json.dumps(reported)[:400]


def reply_content(reply: EndpointReply) -> Result[str, TranslationError]:
    failure = endpoint_error(reply)
    if failure is not None:
        return fail_http("protocol", "endpoint error: %s" % failure)

    choices = reply.body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    choice = message.get("content") if isinstance(message, dict) else None
    if isinstance(choice, str) and choice.strip():
        return Ok(choice)

    finish = first.get("finish_reason") if isinstance(first, dict) else None
    reasoning = isinstance(message, dict) and bool(message.get("reasoning"))
    hint = (
        "; raise --call-max-tokens or cap reasoning, e.g. --judge-params "
        '\'{"reasoning": {"effort": "low"}}\''
        if finish == "length"
        else ""
    )
    return fail_http(
        "protocol",
        "empty content (finish_reason=%s, reasoning=%s)%s: %s"
        % (finish, reasoning, hint, compact(reply.text)),
    )


def extract_object(text: str) -> Result[dict[str, Any], TranslationError]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return fail_config("no JSON object in reply: %s" % compact(text))

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        return fail_config("reply is not valid JSON: %s" % error)

    if not isinstance(parsed, dict):
        return fail_config("reply is not a JSON object")
    return Ok(parsed)


def usable_verdict(
    parsed: Result[dict[str, Any], TranslationError],
) -> Result[dict[str, Any], TranslationError]:
    def decide(verdict: dict[str, Any]) -> Result[dict[str, Any], TranslationError]:
        if verdict.get("winner") in ("A", "B", "tie"):
            return Ok(verdict)
        return fail_config("winner must be A, B, or tie")

    return result_bind(parsed, decide)


def error_verdict(text: str) -> dict[str, Any]:
    return {
        "winner": "error",
        "critique": "judge reply was unusable: %s" % compact(text),
        "evidence": [],
    }


def strip_reply(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[-1].strip().startswith("```"):
            lines = lines[1:-1]
        elif len(lines) >= 1:
            lines = lines[1:]
        stripped = "\n".join(lines).strip()
    return stripped


def fill_judge(template: str, source: str, left: str, right: str) -> str:
    return (
        template.replace(SOURCE_TOKEN, source)
        .replace(A_TOKEN, left)
        .replace(B_TOKEN, right)
    )


def fill_rewrite(
    template: str, current_prompt: str, history_text: str, critiques: str
) -> str:
    return (
        template.replace(PROMPT_TOKEN, current_prompt)
        .replace(HISTORY_TOKEN, history_text)
        .replace(CRITIQUES_TOKEN, critiques)
    )


def order_outcome(winner: str, candidate_side: str) -> str:
    if winner not in ("A", "B"):
        return "tie"
    if winner == candidate_side:
        return "candidate"
    return "incumbent"


def feedback_lines(
    chapter: str, first: Mapping[str, Any], second: Mapping[str, Any]
) -> tuple[str, ...]:
    def describe_verdict(label: str, verdict: Mapping[str, Any]) -> str:
        critique = str(verdict.get("critique") or "").strip()
        raw = verdict.get("evidence")
        evidence = "; ".join(str(item) for item in raw) if isinstance(raw, list) else ""
        return "%s [%s] %s | evidence: %s" % (
            chapter,
            label,
            critique,
            evidence or "none quoted",
        )

    return (
        describe_verdict("incumbent shown as A", first),
        describe_verdict("candidate shown as A", second),
    )


def toml_value(value: Any) -> Result[str, TranslationError]:
    if isinstance(value, bool):
        return Ok("true" if value else "false")
    if isinstance(value, str):
        return Ok(json.dumps(value))
    if isinstance(value, (int, float)):
        return Ok(repr(value))
    if isinstance(value, list):
        items = result_sequence(tuple(toml_value(item) for item in value))
        return result_map(items, lambda parts: "[" + ", ".join(parts) + "]")
    return fail_config("unsupported config value: %r" % (value,))


def emit_body(
    prefix: str, table: Mapping[str, Any]
) -> Result[tuple[str, ...], TranslationError]:
    def flat_line(item: tuple[str, Any]) -> Result[tuple[str, ...], TranslationError]:
        key, value = item
        if isinstance(value, dict):
            return Ok(())
        return result_map(
            toml_value(value), lambda rendered: ("%s = %s" % (key, rendered),)
        )

    def child_block(item: tuple[str, Any]) -> Result[tuple[str, ...], TranslationError]:
        key, value = item
        if not isinstance(value, dict) or not value:
            return Ok(())
        child = "%s.%s" % (prefix, key)
        return result_map(
            emit_body(child, value),
            lambda lines: ("", "[%s]" % child) + lines,
        )

    def joined(
        flat: tuple[tuple[str, ...], ...], nested: tuple[tuple[str, ...], ...]
    ) -> tuple[str, ...]:
        return tuple(line for part in flat + nested for line in part)

    def with_nested(
        flat: tuple[tuple[str, ...], ...],
    ) -> Result[tuple[str, ...], TranslationError]:
        nested = result_sequence(tuple(child_block(item) for item in table.items()))
        return result_map(nested, lambda parts: joined(flat, parts))

    return result_bind(
        result_sequence(tuple(flat_line(item) for item in table.items())),
        with_nested,
    )


def render_config(
    api: Mapping[str, Any],
    options: Mapping[str, Any],
    passes: tuple[Mapping[str, Any], ...],
) -> Result[str, TranslationError]:
    def flatten(groups: tuple[tuple[str, ...], ...]) -> str:
        lines = tuple(line for group in groups for line in group)
        return "\n".join(lines) + "\n"

    def pass_groups() -> Result[tuple[tuple[str, ...], ...], TranslationError]:
        def one(entry: Mapping[str, Any]) -> Result[tuple[str, ...], TranslationError]:
            return result_map(
                emit_body("pass", entry), lambda lines: ("", "[[pass]]") + lines
            )

        return result_sequence(tuple(one(entry) for entry in passes))

    def finish(
        api_lines: tuple[str, ...],
        options_lines: tuple[str, ...] | None,
        pass_parts: tuple[tuple[str, ...], ...],
    ) -> str:
        groups: tuple[tuple[str, ...], ...] = (("[api]",), api_lines)
        if options_lines is not None:
            groups = groups + (("", "[options]"), options_lines)
        return flatten(groups + pass_parts)

    def with_pass_parts(
        api_lines: tuple[str, ...], options_lines: tuple[str, ...] | None
    ) -> Callable[[tuple[tuple[str, ...], ...]], Result[str, TranslationError]]:
        def taken(parts: tuple[tuple[str, ...], ...]) -> Result[str, TranslationError]:
            return Ok(finish(api_lines, options_lines, parts))

        return taken

    def with_api_lines(api_lines: tuple[str, ...]) -> Result[str, TranslationError]:
        if not options:
            return result_bind(pass_groups(), with_pass_parts(api_lines, None))

        def with_option_lines(
            lines: tuple[str, ...],
        ) -> Result[str, TranslationError]:
            return result_bind(
                pass_groups(),
                with_pass_parts(api_lines, ("", "[options]") + lines),
            )

        return result_bind(emit_body("options", options), with_option_lines)

    return result_bind(emit_body("api", api), with_api_lines)


def as_table(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def load_base_config(
    data: Mapping[str, Any], pass_name: str | None, path: str
) -> Result[BaseConfig, TranslationError]:
    passes = tuple(MappingProxyType(dict(entry)) for entry in data.get("pass", []))
    if not passes:
        return fail_config("base config %s has no [[pass]] entries" % path)

    def target(index: int) -> Result[BaseConfig, TranslationError]:
        entry = passes[index]
        return Ok(
            BaseConfig(
                api=MappingProxyType(as_table(data.get("api"))),
                options=MappingProxyType(as_table(data.get("options"))),
                passes=passes,
                target_index=index,
                target_name=str(entry.get("name") or "translate"),
            )
        )

    def single() -> Result[BaseConfig, TranslationError]:
        if len(passes) == 1:
            return target(0)
        return fail_config(
            "base config %s has %d passes; pass --pass-name to pick one"
            % (path, len(passes))
        )

    if pass_name is None:
        return single()

    matches = tuple(
        index for index, entry in enumerate(passes) if entry.get("name") == pass_name
    )
    if not matches:
        return fail_config("base config %s has no pass named %r" % (path, pass_name))
    return target(matches[0])


def resume_state(
    history: tuple[dict[str, Any], ...], ledger: tuple[dict[str, Any], ...]
) -> RunState:
    promoted = tuple(
        index
        for index, entry in enumerate(ledger)
        if entry.get("decision") == "promoted"
    )
    incumbent = str(ledger[promoted[-1]]["candidate"]) if promoted else SEED_VERSION
    stall = len(ledger) - promoted[-1] - 1 if promoted else len(ledger)
    return RunState(
        round_no=len(ledger) + 1, incumbent=incumbent, stall=stall, history=history
    )


def history_window(history: tuple[dict[str, Any], ...], depth_limit: int) -> str:
    depth = max(0, len(history) - depth_limit)
    return (
        "\n\n".join(json.dumps(entry, ensure_ascii=False) for entry in history[depth:])
        or "(empty)"
    )


def select_chapters(
    paths: tuple[str, ...], selector: str, chapters_directory: str
) -> Result[tuple[str, ...], TranslationError]:
    by_stem = {text_stem(path): path for path in paths}

    def chosen() -> Result[tuple[str, ...], TranslationError]:
        selected: dict[str, str] = {}
        for raw in selector.split(","):
            name = raw.strip()
            if not name or name in selected:
                continue
            if name not in by_stem:
                return fail_config(
                    "unknown chapter %r in %s" % (name, chapters_directory)
                )
            selected = {**selected, name: by_stem[name]}
        if not selected:
            return fail_config(
                "--chapters named none of: %s" % ", ".join(sorted(by_stem))
            )
        return Ok(tuple(selected.values()))

    return chosen()


def discover_chapters(
    chapters_directory: str,
) -> IO[Result[tuple[str, ...], TranslationError]]:
    holdout = os.path.join(chapters_directory, "holdout.txt")

    def with_holdout(
        chapters: tuple[str, ...],
    ) -> Callable[[bool], Result[tuple[str, ...], TranslationError]]:
        def taken(present: bool) -> Result[tuple[str, ...], TranslationError]:
            if not present:
                return fail_config("missing %s" % holdout)
            return Ok(chapters)

        return taken

    def listed(paths: tuple[str, ...]) -> Result[tuple[str, ...], TranslationError]:
        chapters = tuple(
            sorted(
                path
                for path in paths
                if os.path.basename(path).endswith(".txt")
                and os.path.basename(path) != "holdout.txt"
            )
        )
        if len(chapters) < 2:
            return fail_config("need at least two chapters in %s" % chapters_directory)
        return Ok(chapters)

    def with_directory(
        is_directory: bool,
    ) -> IO[Result[tuple[str, ...], TranslationError]]:
        if not is_directory:
            return io_result(
                fail_config(
                    "missing %s (expected chapter .txt files plus holdout.txt)"
                    % chapters_directory
                )
            )

        def with_entries(
            paths: tuple[str, ...],
        ) -> IO[Result[tuple[str, ...], TranslationError]]:
            outcome = listed(paths)
            if isinstance(outcome, Err):
                return io_result(outcome)

            def with_presence(
                present: bool,
            ) -> Result[tuple[str, ...], TranslationError]:
                return with_holdout(outcome.value)(present)

            return io_map(path_exists(holdout), with_presence)

        return io_bind(entry_paths(chapters_directory), with_entries)

    return io_bind(path_is_directory(chapters_directory), with_directory)


def compare_chapters(
    working_directory: str, options: Options
) -> IO[Result[tuple[str, ...], TranslationError]]:
    chapters_directory = join_directory(working_directory, "chapters")
    holdout = os.path.join(chapters_directory, "holdout.txt")

    def with_holdout(present: bool) -> Result[tuple[str, ...], TranslationError]:
        if not present:
            return fail_config("missing %s" % holdout)
        return Ok((holdout,))

    def discovered(
        paths: Result[tuple[str, ...], TranslationError],
    ) -> IO[Result[tuple[str, ...], TranslationError]]:
        if isinstance(paths, Err) or options.chapters is None:
            return io_result(paths)
        return io_result(
            select_chapters(paths.value, options.chapters, chapters_directory)
        )

    if options.holdout:
        return io_map(path_exists(holdout), with_holdout)
    return io_bind(discover_chapters(chapters_directory), discovered)


def read_jsonl(path: str) -> IO[tuple[dict[str, Any], ...]]:
    def lines(
        text_result: Result[str, TranslationError],
    ) -> tuple[dict[str, Any], ...]:
        if isinstance(text_result, Err):
            return ()
        return tuple(
            json.loads(line) for line in text_result.value.splitlines() if line.strip()
        )

    return io_map(read_text_file(path, "ledger"), lines)


def parse_parameters(
    raw: str | None, flag: str
) -> Result[dict[str, Any], TranslationError]:
    if not raw:
        return Ok({})

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        return fail_config("--%s is not valid JSON: %s" % (flag, error))

    if not isinstance(parsed, dict):
        return fail_config("--%s must be a JSON object" % flag)
    return Ok(dict(parsed))


def optional(value: object) -> str | None:
    return None if value is None else str(value)


def build_options(
    raw: argparse.Namespace,
    compare: tuple[str, str] | None,
    judge_extra: Mapping[str, Any],
    rewrite_extra: Mapping[str, Any],
) -> Options:
    return Options(
        base_config=str(raw.base_config),
        working_directory=str(raw.working_directory),
        seed=optional(raw.seed),
        rounds=int(raw.rounds),
        stall=int(raw.stall),
        judge_model=str(raw.judge_model),
        rewrite_model=optional(raw.rewriter_model),
        translator_model=optional(raw.translator_model),
        compare=compare,
        holdout=bool(raw.holdout),
        chapters=optional(raw.chapters),
        pass_name=optional(raw.pass_name),
        base_url=optional(raw.base_url),
        api_key=optional(raw.api_key),
        l2l=optional(raw.l2l),
        judge_template=str(raw.judge_template),
        rewrite_template=str(raw.rewrite_template),
        judge_temperature=float(raw.judge_temperature),
        rewrite_temperature=float(raw.rewrite_temperature),
        translator_temperature=float(raw.translator_temperature),
        call_timeout=float(raw.call_timeout),
        call_maximum_tokens=int(raw.call_maximum_tokens),
        history_depth=int(raw.history_depth),
        judge_extra=judge_extra,
        rewrite_extra=rewrite_extra,
    )


def parse_arguments(argv: Sequence[str]) -> Result[Options, TranslationError]:
    parser = argparse.ArgumentParser(
        prog="optimize",
        description=(
            "Evolve an l2l translation instruction through rounds of "
            "pairwise judged challenges, or compare two instruction "
            "files directly with --compare."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="l2l config supplying the [api] table and the pass pipeline",
    )
    parser.add_argument(
        "--workdir",
        type=str,
        required=True,
        dest="working_directory",
        help="experiment directory holding chapters/, prompts/, out/, judge/",
    )
    parser.add_argument(
        "--seed",
        type=str,
        help="instruction file to copy to prompts/v0.txt when absent",
    )
    parser.add_argument("--rounds", type=int, default=6, help="maximum rounds")
    parser.add_argument(
        "--stall",
        type=int,
        default=2,
        help="stop after this many consecutive rounds without promotion",
    )
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--rewriter-model",
        help="model that proposes new instructions (evolution mode only)",
    )
    parser.add_argument(
        "--translator-model",
        help="override the swapped pass's model",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("PATH_A", "PATH_B"),
        help=(
            "instruction files to pit against each other instead of "
            "evolving (skips the rewrite loop and the holdout check)"
        ),
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="with --compare, judge only on chapters/holdout.txt",
    )
    parser.add_argument(
        "--chapters",
        help=("with --compare, comma-separated chapter stems to judge, e.g. ch1,ch3"),
    )
    parser.add_argument(
        "--pass-name",
        help="which pass's instruction to evolve (needed for multi-pass configs)",
    )
    parser.add_argument("--base-url", help="chat completions endpoint")
    parser.add_argument("--api-key", help="bearer token for the endpoint")
    parser.add_argument(
        "--l2l",
        help="command used to run l2l (default: '<python> -m l2l')",
    )
    parser.add_argument(
        "--judge-template",
        type=str,
        default=os.path.join(TOOLS_DIRECTORY, "judge.txt"),
    )
    parser.add_argument(
        "--rewrite-template",
        type=str,
        default=os.path.join(TOOLS_DIRECTORY, "rewrite.txt"),
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--rewrite-temperature", type=float, default=0.8)
    parser.add_argument("--translator-temperature", type=float, default=0.0)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument(
        "--call-max-tokens",
        type=int,
        default=32768,
        dest="call_maximum_tokens",
    )
    parser.add_argument(
        "--judge-params",
        dest="judge_parameters",
        help=(
            "JSON object merged into judge request bodies, e.g. "
            '\'{"reasoning": {"effort": "low"}}\' for thinking models'
        ),
    )
    parser.add_argument(
        "--rewrite-params",
        dest="rewrite_parameters",
        help="JSON object merged into rewrite request bodies",
    )
    parser.add_argument("--history-depth", type=int, default=6)
    raw = parser.parse_args(argv)
    compare = (
        (str(raw.compare[0]), str(raw.compare[1])) if raw.compare is not None else None
    )

    def check(valid: bool, message: str) -> Result[None, TranslationError]:
        return Ok(None) if valid else fail_config(message)

    checks = result_sequence(
        (
            check(raw.rounds >= 1, "--rounds must be at least 1"),
            check(raw.stall >= 1, "--stall must be at least 1"),
            check(
                not raw.holdout or compare is not None,
                "--holdout requires --compare",
            ),
            check(
                not raw.chapters or compare is not None,
                "--chapters requires --compare",
            ),
            check(
                compare is not None or raw.rewriter_model is not None,
                "--rewriter-model is required unless --compare is given",
            ),
        )
    )
    judge_extra = parse_parameters(raw.judge_parameters, "judge-params")
    rewrite_extra = parse_parameters(raw.rewrite_parameters, "rewrite-params")

    def with_first(first: dict[str, Any]) -> Result[Options, TranslationError]:
        def with_second(second: dict[str, Any]) -> Options:
            return build_options(
                raw,
                compare,
                MappingProxyType(first),
                MappingProxyType(second),
            )

        return result_map(rewrite_extra, with_second)

    return result_bind(result_bind(checks, lambda _: judge_extra), with_first)


def endpoint_config(api: Api, timeout: float) -> Config:
    return Config(
        base_url=api.base_url,
        api_key=api.api_key,
        model="",
        timeout=timeout,
        maximum_tokens=0,
        parameters=MappingProxyType({}),
    )


def curl_endpoint(
    api: Api, environment: Mapping[str, str], timeout: float
) -> OpenEndpoint:
    def open_endpoint(
        payload: Mapping[str, Any],
    ) -> IO[Result[EndpointReply, TranslationError]]:
        request: HttpRequest = http_request(endpoint_config(api, timeout), payload)

        def loaded(
            raw: Result[bytes, TranslationError],
        ) -> Result[EndpointReply, TranslationError]:
            parsed = parse_json_body(raw)
            if isinstance(parsed, Err):
                return parsed
            text, body = parsed.value
            if not isinstance(body, dict):
                return fail_http(
                    "protocol", "non-object endpoint reply: %s" % compact(text)
                )
            return Ok(EndpointReply(text, body))

        def respond(
            opened: Result[Any, TranslationError],
        ) -> IO[Result[EndpointReply, TranslationError]]:
            if isinstance(opened, Err):
                return io_result(opened)
            return io_map(opened.value.body(), loaded)

        return io_bind(io_result(curl_open(request, timeout, environment)), respond)

    return open_endpoint


def chain_result[T, R](
    step: Callable[[T], IO[Result[R, TranslationError]]],
) -> Callable[[Result[T, TranslationError]], IO[Result[R, TranslationError]]]:
    def continue_(
        outcome: Result[T, TranslationError],
    ) -> IO[Result[R, TranslationError]]:
        if isinstance(outcome, Err):
            return io_result(outcome)
        return step(outcome.value)

    return continue_


def complete(
    optimization_environment: OptimizationEnvironment,
    model: str,
    content: str,
    temperature: float,
    extra: Mapping[str, Any],
    label: str,
) -> IO[Result[str, TranslationError]]:
    payload = chat_payload(
        model,
        content,
        temperature,
        optimization_environment.options.call_maximum_tokens,
        extra,
    )

    def at(index: int) -> IO[Result[str, TranslationError]]:
        def decided(
            reply: Result[EndpointReply, TranslationError],
        ) -> IO[Result[str, TranslationError]]:
            outcome = result_bind(reply, reply_content)
            if isinstance(outcome, Ok) or index >= CHAT_ATTEMPTS:
                return io_result(outcome)

            def waited(_: None) -> IO[Result[str, TranslationError]]:
                return at(index + 1)

            return io_bind(
                optimization_environment.warn(
                    "%s: attempt %d/%d failed (%s)"
                    % (label, index, CHAT_ATTEMPTS, describe(outcome.error))
                ),
                lambda _: io_and_then(
                    IO(lambda: optimization_environment.sleep(2.0 * index)),
                    waited(None),
                ),
            )

        return io_bind(optimization_environment.opener(payload), decided)

    return at(1)


def judge_call(
    optimization_environment: OptimizationEnvironment,
    model: str,
    prompt: str,
    label: str,
) -> IO[Result[dict[str, Any], TranslationError]]:
    def at(index: int) -> IO[Result[dict[str, Any], TranslationError]]:
        def decided(
            text_result: Result[str, TranslationError],
        ) -> IO[Result[dict[str, Any], TranslationError]]:
            if isinstance(text_result, Err):
                return io_result(text_result)

            verdict_outcome = usable_verdict(extract_object(text_result.value))
            if isinstance(verdict_outcome, Ok):
                return io_result(verdict_outcome)

            def waited(_: None) -> IO[Result[dict[str, Any], TranslationError]]:
                return at(index + 1)

            def bail(_: None) -> IO[Result[dict[str, Any], TranslationError]]:
                return io_result(Ok(error_verdict(text_result.value)))

            if index >= VERDICT_ATTEMPTS:
                return io_bind(
                    optimization_environment.warn(
                        "%s: giving up; recording an error verdict" % label
                    ),
                    bail,
                )

            return io_bind(
                optimization_environment.warn(
                    "%s: attempt %d/%d unusable verdict (%s)"
                    % (label, index, VERDICT_ATTEMPTS, describe(verdict_outcome.error))
                ),
                lambda _: io_and_then(
                    IO(lambda: optimization_environment.sleep(2.0 * index)),
                    waited(None),
                ),
            )

        return io_bind(
            complete(
                optimization_environment,
                model,
                prompt,
                optimization_environment.options.judge_temperature,
                optimization_environment.options.judge_extra,
                label,
            ),
            decided,
        )

    return at(1)


def judge_chapter(
    optimization_environment: OptimizationEnvironment,
    source_text: str,
    incumbent_text: str,
    candidate_text: str,
    chapter: str,
    round_no: int,
) -> IO[Result[ChapterJudging, TranslationError]]:
    def side(
        left: str, right: str, label: str
    ) -> IO[Result[dict[str, Any], TranslationError]]:
        return judge_call(
            optimization_environment,
            optimization_environment.options.judge_model,
            fill_judge(
                optimization_environment.judge_template, source_text, left, right
            ),
            "judge r%d/%s (%s)" % (round_no, chapter, label),
        )

    def assemble(first: dict[str, Any], second: dict[str, Any]) -> ChapterJudging:
        one = order_outcome(str(first.get("winner")), "B")
        two = order_outcome(str(second.get("winner")), "A")
        if one == "candidate" and two == "candidate":
            outcome = "candidate"
        elif one == "incumbent" and two == "incumbent":
            outcome = "incumbent"
        else:
            outcome = "tie"
        return ChapterJudging(
            outcome, first, second, feedback_lines(chapter, first, second)
        )

    def with_first(
        first: dict[str, Any],
    ) -> IO[Result[ChapterJudging, TranslationError]]:
        def with_second(
            second: dict[str, Any],
        ) -> IO[Result[ChapterJudging, TranslationError]]:
            return io_result(Ok(assemble(first, second)))

        return io_bind(
            side(candidate_text, incumbent_text, "candidate as A"),
            chain_result(with_second),
        )

    return io_bind(
        side(incumbent_text, candidate_text, "incumbent as A"),
        chain_result(with_first),
    )


def rewrite_prompt(
    optimization_environment: OptimizationEnvironment,
    current_prompt: str,
    history_text: str,
    critiques: str,
    round_no: int,
) -> IO[Result[str, TranslationError]]:
    prompt_text = fill_rewrite(
        optimization_environment.rewrite_template,
        current_prompt,
        history_text,
        critiques,
    )

    def decided(
        text_result: Result[str, TranslationError],
    ) -> Result[str, TranslationError]:
        if isinstance(text_result, Err):
            return text_result
        improved = strip_reply(text_result.value)
        if not improved:
            return fail_config("rewriter returned an empty instruction")
        return Ok(improved)

    return io_map(
        complete(
            optimization_environment,
            optimization_environment.options.rewrite_model or "",
            prompt_text,
            optimization_environment.options.rewrite_temperature,
            optimization_environment.options.rewrite_extra,
            "rewrite r%d" % round_no,
        ),
        decided,
    )


def previous_critiques(
    optimization_environment: OptimizationEnvironment, round_no: int
) -> IO[str]:
    feedback_previous = os.path.join(
        join_directory(optimization_environment.working_directory, "judge"),
        "r%d-feedback.txt" % (round_no - 1),
    )

    def with_exists(exists: bool) -> IO[str]:
        if not exists:
            return io_pure("(none; this is the first round)")
        return io_map(
            read_text_file(feedback_previous, "previous feedback"),
            lambda outcome: result_or_else(
                outcome, lambda: "(none; this is the first round)"
            ),
        )

    return io_bind(path_exists(feedback_previous), with_exists)


def candidate_prompt(
    optimization_environment: OptimizationEnvironment,
    state: RunState,
    current: str,
    history_text: str,
    critiques: str,
) -> IO[Result[str, TranslationError]]:
    candidate_path = os.path.join(
        join_directory(optimization_environment.working_directory, "prompts"),
        "v%d.txt" % state.round_no,
    )

    def persisted(prompt: str) -> IO[Result[str, TranslationError]]:
        def with_write(
            written: Result[None, TranslationError],
        ) -> IO[Result[str, TranslationError]]:
            if isinstance(written, Err):
                return io_result(written)
            return io_result(Ok(prompt))

        return io_bind(
            write_text_file(candidate_path, prompt, "candidate instruction"),
            with_write,
        )

    def asked(_: None) -> IO[Result[str, TranslationError]]:
        return io_bind(
            rewrite_prompt(
                optimization_environment,
                current,
                history_text,
                critiques,
                state.round_no,
            ),
            chain_result(persisted),
        )

    def announced(_: None) -> IO[Result[str, TranslationError]]:
        return io_bind(
            optimization_environment.say(
                "round %d: asking %s for a new instruction"
                % (state.round_no, optimization_environment.options.rewrite_model)
            ),
            asked,
        )

    def reused(_: None) -> IO[Result[str, TranslationError]]:
        def announced_reuse(_: None) -> IO[Result[str, TranslationError]]:
            return read_text_file(candidate_path, "candidate instruction")

        return io_bind(
            optimization_environment.say(
                "round %d: reusing %s" % (state.round_no, candidate_path)
            ),
            announced_reuse,
        )

    def decide(exists: bool) -> IO[Result[str, TranslationError]]:
        if exists:
            return reused(None)
        return announced(None)

    def with_critiques(critiques: str) -> IO[Result[str, TranslationError]]:
        return io_bind(path_exists(candidate_path), decide)

    return io_bind(
        previous_critiques(optimization_environment, state.round_no), with_critiques
    )


def prepare_version(
    optimization_environment: OptimizationEnvironment, version: str, prompt_text: str
) -> IO[Result[str, TranslationError]]:
    base = optimization_environment.base
    prompts_directory = join_directory(
        optimization_environment.working_directory, "prompts"
    )
    generation_directory = join_directory(
        optimization_environment.working_directory, "gen"
    )
    digest = hash_text(prompt_text)

    def target_entry() -> Mapping[str, Any]:
        original = base.passes[base.target_index]
        stripped = {
            key: value
            for key, value in original.items()
            if key not in ("instruction", "instruction_file")
        }
        parameters = dict(original.get("params") or {})
        entry = {
            **stripped,
            "name": "%s-%s" % (base.target_name, digest[:10]),
            "instruction_file": "../prompts/%s.txt" % version,
            "params": {
                **parameters,
                "temperature": optimization_environment.options.translator_temperature,
            },
        }
        if optimization_environment.options.translator_model:
            return MappingProxyType(
                {**entry, "model": optimization_environment.options.translator_model}
            )
        return MappingProxyType(entry)

    passes = tuple(
        target_entry() if index == base.target_index else entry
        for index, entry in enumerate(base.passes)
    )

    def with_prompt(
        config_text: str,
    ) -> Callable[[Result[None, TranslationError]], IO[Result[str, TranslationError]]]:
        def taken(
            written: Result[None, TranslationError],
        ) -> IO[Result[str, TranslationError]]:
            if isinstance(written, Err):
                return io_result(written)

            def with_config(
                config_written: Result[None, TranslationError],
            ) -> Result[str, TranslationError]:
                return result_map(config_written, lambda _: digest)

            return io_bind(
                write_text_file(
                    os.path.join(generation_directory, version + ".toml"),
                    config_text,
                    "generated config",
                ),
                lambda outcome: io_result(with_config(outcome)),
            )

        return taken

    def with_config_text(config_text: str) -> IO[Result[str, TranslationError]]:
        return io_bind(
            write_text_file(
                os.path.join(prompts_directory, version + ".txt"),
                prompt_text,
                "instruction file",
            ),
            with_prompt(config_text),
        )

    def with_directories(_: None) -> IO[Result[str, TranslationError]]:
        return result_bind_io(
            render_config(base.api, base.options, passes), with_config_text
        )

    return io_bind(
        io_and_then(
            ensure_directory(prompts_directory), ensure_directory(generation_directory)
        ),
        with_directories,
    )


def seen_digests(
    optimization_environment: OptimizationEnvironment,
) -> IO[Result[dict[str, str], TranslationError]]:
    prompts_directory = join_directory(
        optimization_environment.working_directory, "prompts"
    )

    def keep(
        seen: dict[str, str], path: str
    ) -> IO[Result[dict[str, str], TranslationError]]:
        stem = os.path.basename(path)[: -len(".txt")]

        def merged(text: str) -> dict[str, str]:
            digest = hash_text(text)
            return seen if digest in seen else {**seen, digest: stem}

        def applied(
            text_result: Result[str, TranslationError],
        ) -> Result[dict[str, str], TranslationError]:
            return Ok(result_either(text_result, merged, lambda _error: seen))

        return io_map(read_text_file(path, "instruction file"), applied)

    def versioned(path: str) -> bool:
        name = os.path.basename(path)
        return name.startswith("v") and name.endswith(".txt")

    def listed(
        paths: tuple[str, ...],
    ) -> IO[Result[dict[str, str], TranslationError]]:
        return fold_io(
            tuple(sorted(path for path in paths if versioned(path))), keep, Ok({})
        )

    return io_bind(entry_paths(prompts_directory), listed)


def translate_chapter(
    optimization_environment: OptimizationEnvironment, version: str, chapter: str
) -> IO[Result[None, TranslationError]]:
    target = output_path(optimization_environment.working_directory, version, chapter)
    config_path = os.path.join(
        join_directory(optimization_environment.working_directory, "gen"),
        version + ".toml",
    )
    partial = target + ".partial"
    stderr_path = target + ".err"
    cache_directory = join_directory(
        optimization_environment.working_directory, "cache"
    )
    command = optimization_environment.l2l_command + (
        config_path,
        "--cache-dir",
        cache_directory,
    )

    def ran(result: ProcessResult) -> IO[Result[None, TranslationError]]:
        def with_error_log(
            _: Result[None, TranslationError],
        ) -> IO[Result[None, TranslationError]]:
            return io_result(
                fail_config(
                    "l2l exited with %d for %s; see %s"
                    % (result.returncode, chapter, stderr_path)
                )
            )

        if result.returncode != 0:
            return io_bind(
                write_text_file(stderr_path, result.stderr, "l2l error log"),
                with_error_log,
            )

        def with_partial(
            written: Result[None, TranslationError],
        ) -> IO[Result[None, TranslationError]]:
            if isinstance(written, Err):
                return io_result(written)

            def placed(moved: bool) -> Result[None, TranslationError]:
                return (
                    Ok(None) if moved else fail_config("could not finalize %s" % target)
                )

            return io_map(replace_file(partial, target), placed)

        return io_bind(
            write_text_file(partial, result.stdout, "partial output"), with_partial
        )

    def with_chapter(
        text_result: Result[str, TranslationError],
    ) -> IO[Result[None, TranslationError]]:
        return result_bind_io(
            text_result,
            lambda text: io_bind(
                optimization_environment.run_command(command, text), ran
            ),
        )

    def with_config(config_exists: bool) -> IO[Result[None, TranslationError]]:
        if not config_exists:
            return io_result(fail_config("missing generated config %s" % config_path))

        def with_directory(_: None) -> IO[Result[None, TranslationError]]:
            return io_bind(read_text_file(chapter, "chapter"), with_chapter)

        return io_bind(ensure_directory(os.path.dirname(target)), with_directory)

    def with_target(exists: bool) -> IO[Result[None, TranslationError]]:
        if exists:
            return io_result(Ok(None))
        return io_bind(path_exists(config_path), with_config)

    return io_bind(path_exists(target), with_target)


def translate_all(
    optimization_environment: OptimizationEnvironment, version: str
) -> IO[Result[None, TranslationError]]:
    def step(_: None, chapter: str) -> IO[Result[None, TranslationError]]:
        return translate_chapter(optimization_environment, version, chapter)

    return fold_io(optimization_environment.chapters, step, Ok(None))


def translate_versions(
    optimization_environment: OptimizationEnvironment, first: str, second: str
) -> IO[Result[None, TranslationError]]:
    def after_first(
        initial: Result[None, TranslationError],
    ) -> IO[Result[None, TranslationError]]:
        return result_bind_io(
            initial, lambda _: translate_all(optimization_environment, second)
        )

    return io_bind(translate_all(optimization_environment, first), after_first)


def round_texts(
    optimization_environment: OptimizationEnvironment,
    incumbent_version: str,
    candidate_version: str,
    chapter: str,
) -> IO[Result[tuple[str, str, str], TranslationError]]:
    def with_source(
        source_result: Result[str, TranslationError],
    ) -> IO[Result[tuple[str, str, str], TranslationError]]:
        if isinstance(source_result, Err):
            return io_result(source_result)

        def with_left(
            left_result: Result[str, TranslationError],
        ) -> IO[Result[tuple[str, str, str], TranslationError]]:
            if isinstance(left_result, Err):
                return io_result(left_result)

            def with_right(
                right_result: Result[str, TranslationError],
            ) -> Result[tuple[str, str, str], TranslationError]:
                return result_map(
                    right_result,
                    lambda right: (source_result.value, left_result.value, right),
                )

            return io_map(
                read_text_file(
                    output_path(
                        optimization_environment.working_directory,
                        candidate_version,
                        chapter,
                    ),
                    "translation",
                ),
                with_right,
            )

        return io_bind(
            read_text_file(
                output_path(
                    optimization_environment.working_directory,
                    incumbent_version,
                    chapter,
                ),
                "translation",
            ),
            with_left,
        )

    return io_bind(read_text_file(chapter, "chapter"), with_source)


def record_judging(tally: Tally, judging: ChapterJudging) -> Tally:
    if judging.outcome == "candidate":
        return replace(
            tally, candidate=tally.candidate + 1, lines=tally.lines + judging.lines
        )
    if judging.outcome == "incumbent":
        return replace(
            tally, incumbent=tally.incumbent + 1, lines=tally.lines + judging.lines
        )
    return replace(tally, tie=tally.tie + 1, lines=tally.lines + judging.lines)


def judging_json(judging: ChapterJudging) -> str:
    return (
        json.dumps(
            {
                "outcome": judging.outcome,
                "incumbent_as_a": judging.first,
                "candidate_as_a": judging.second,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )


def judge_step(
    optimization_environment: OptimizationEnvironment,
    round_no: int,
    incumbent_version: str,
) -> Callable[[Tally, str], IO[Result[Tally, TranslationError]]]:
    judge_directory = join_directory(
        optimization_environment.working_directory, "judge"
    )

    def step(tally: Tally, chapter: str) -> IO[Result[Tally, TranslationError]]:
        def with_judging(
            judging_result: Result[ChapterJudging, TranslationError],
        ) -> IO[Result[Tally, TranslationError]]:
            if isinstance(judging_result, Err):
                return io_result(judging_result)
            judging = judging_result.value
            verdict_path = os.path.join(
                judge_directory, "r%d-%s.json" % (round_no, text_stem(chapter))
            )

            def with_verdict(
                written: Result[None, TranslationError],
            ) -> IO[Result[Tally, TranslationError]]:
                if isinstance(written, Err):
                    return io_result(written)
                return io_result(Ok(record_judging(tally, judging)))

            return io_bind(
                write_text_file(verdict_path, judging_json(judging), "verdict file"),
                with_verdict,
            )

        def with_texts(
            texts: Result[tuple[str, str, str], TranslationError],
        ) -> IO[Result[Tally, TranslationError]]:
            if isinstance(texts, Err):
                return io_result(texts)
            source_text, incumbent_text, candidate_text = texts.value
            return io_bind(
                judge_chapter(
                    optimization_environment,
                    source_text,
                    incumbent_text,
                    candidate_text,
                    os.path.basename(chapter),
                    round_no,
                ),
                with_judging,
            )

        return io_bind(
            round_texts(
                optimization_environment, incumbent_version, "v%d" % round_no, chapter
            ),
            with_texts,
        )

    return step


def append_jsonl(
    path: str, entry: dict[str, Any], description: str
) -> IO[Result[None, TranslationError]]:
    return append_text_file(
        path, json.dumps(entry, ensure_ascii=False) + "\n", description
    )


def announce(
    optimization_environment: OptimizationEnvironment, lines: tuple[str, ...]
) -> IO[None]:
    def run_all(index: int) -> IO[None]:
        if index >= len(lines):
            return io_pure(None)
        return io_bind(
            optimization_environment.say(lines[index]), lambda _: run_all(index + 1)
        )

    return run_all(0)


def record_duplicate(
    optimization_environment: OptimizationEnvironment,
    state: RunState,
    inputs: RoundInputs,
    duplicate: str,
) -> IO[Result[RunState, TranslationError]]:
    candidate_version = "v%d" % state.round_no
    ledger_entry = {
        "round": state.round_no,
        "incumbent": state.incumbent,
        "candidate": candidate_version,
        "duplicate_of": duplicate,
        "decision": "duplicate",
    }
    history_entry = {
        "round": state.round_no,
        "version": candidate_version,
        "decision": "duplicate",
        "prompt": inputs.candidate_prompt,
    }
    next_state = RunState(
        round_no=state.round_no + 1,
        incumbent=state.incumbent,
        stall=state.stall + 1,
        history=state.history + (history_entry,),
    )

    def with_ledger(
        written: Result[None, TranslationError],
    ) -> IO[Result[RunState, TranslationError]]:
        if isinstance(written, Err):
            return io_result(written)

        def with_history(
            history_written: Result[None, TranslationError],
        ) -> IO[Result[RunState, TranslationError]]:
            if isinstance(history_written, Err):
                return io_result(history_written)

            def announced(_: None) -> Result[RunState, TranslationError]:
                return Ok(next_state)

            return io_map(
                optimization_environment.say(
                    "round %d: candidate duplicates %s; skipping judging"
                    % (state.round_no, duplicate)
                ),
                announced,
            )

        return io_bind(
            append_jsonl(
                os.path.join(
                    optimization_environment.working_directory, "history.jsonl"
                ),
                history_entry,
                "history",
            ),
            with_history,
        )

    return io_bind(
        append_jsonl(
            os.path.join(optimization_environment.working_directory, "ledger.jsonl"),
            ledger_entry,
            "ledger",
        ),
        with_ledger,
    )


def conclude_round(
    optimization_environment: OptimizationEnvironment,
    state: RunState,
    inputs: RoundInputs,
    tally: Tally,
) -> IO[Result[RunState, TranslationError]]:
    majority = len(optimization_environment.chapters) // 2 + 1
    promoted = tally.candidate >= majority
    decision = "promoted" if promoted else "kept"
    candidate_version = "v%d" % state.round_no
    feedback_path = os.path.join(
        join_directory(optimization_environment.working_directory, "judge"),
        "r%d-feedback.txt" % state.round_no,
    )
    ledger_entry = {
        "round": state.round_no,
        "incumbent": state.incumbent,
        "candidate": candidate_version,
        "candidate_wins": tally.candidate,
        "incumbent_wins": tally.incumbent,
        "ties": tally.tie,
        "decision": decision,
        "prompt_digest": inputs.digest,
    }
    history_entry = {
        "round": state.round_no,
        "version": candidate_version,
        "decision": decision,
        "candidate_wins": tally.candidate,
        "incumbent_wins": tally.incumbent,
        "prompt": inputs.candidate_prompt,
    }
    next_state = RunState(
        round_no=state.round_no + 1,
        incumbent=candidate_version if promoted else state.incumbent,
        stall=0 if promoted else state.stall + 1,
        history=state.history + (history_entry,),
    )

    def with_feedback(
        written: Result[None, TranslationError],
    ) -> IO[Result[RunState, TranslationError]]:
        if isinstance(written, Err):
            return io_result(written)

        def with_ledger(
            ledger_written: Result[None, TranslationError],
        ) -> IO[Result[RunState, TranslationError]]:
            if isinstance(ledger_written, Err):
                return io_result(ledger_written)

            def with_history(
                history_written: Result[None, TranslationError],
            ) -> IO[Result[RunState, TranslationError]]:
                if isinstance(history_written, Err):
                    return io_result(history_written)

                def announced(_: None) -> Result[RunState, TranslationError]:
                    return Ok(next_state)

                return io_map(
                    optimization_environment.say(
                        "round %d: candidate %s wins %d, incumbent %s wins %d, "
                        "ties %d -> %s"
                        % (
                            state.round_no,
                            candidate_version,
                            tally.candidate,
                            state.incumbent,
                            tally.incumbent,
                            tally.tie,
                            decision,
                        )
                    ),
                    announced,
                )

            return io_bind(
                append_jsonl(
                    os.path.join(
                        optimization_environment.working_directory, "history.jsonl"
                    ),
                    history_entry,
                    "history",
                ),
                with_history,
            )

        return io_bind(
            append_jsonl(
                os.path.join(
                    optimization_environment.working_directory, "ledger.jsonl"
                ),
                ledger_entry,
                "ledger",
            ),
            with_ledger,
        )

    return io_bind(
        write_text_file(feedback_path, "\n".join(tally.lines) + "\n", "round feedback"),
        with_feedback,
    )


def contest(
    optimization_environment: OptimizationEnvironment,
    state: RunState,
    inputs: RoundInputs,
) -> IO[Result[RunState, TranslationError]]:
    candidate_version = "v%d" % state.round_no

    def with_tally(tally: Tally) -> IO[Result[RunState, TranslationError]]:
        return conclude_round(optimization_environment, state, inputs, tally)

    def judged(_: None) -> IO[Result[RunState, TranslationError]]:
        return io_bind(
            fold_io(
                optimization_environment.chapters,
                judge_step(optimization_environment, state.round_no, state.incumbent),
                Ok(Tally()),
            ),
            chain_result(with_tally),
        )

    def seeded(_: str) -> IO[Result[RunState, TranslationError]]:
        return io_bind(
            translate_versions(
                optimization_environment, state.incumbent, candidate_version
            ),
            chain_result(judged),
        )

    return io_bind(
        prepare_version(
            optimization_environment, state.incumbent, inputs.current_prompt
        ),
        chain_result(seeded),
    )


def evaluate_round(
    optimization_environment: OptimizationEnvironment, state: RunState
) -> IO[Result[RunState, TranslationError]]:
    prompts_directory = join_directory(
        optimization_environment.working_directory, "prompts"
    )
    candidate_version = "v%d" % state.round_no
    history_text = history_window(
        state.history, optimization_environment.options.history_depth
    )

    def decided(
        inputs: RoundInputs, seen: dict[str, str]
    ) -> IO[Result[RunState, TranslationError]]:
        duplicate = seen.get(inputs.digest)
        if duplicate is not None and duplicate != candidate_version:
            return record_duplicate(optimization_environment, state, inputs, duplicate)
        return contest(optimization_environment, state, inputs)

    def with_seen(
        inputs: RoundInputs,
    ) -> Callable[[dict[str, str]], IO[Result[RunState, TranslationError]]]:
        def taken(seen: dict[str, str]) -> IO[Result[RunState, TranslationError]]:
            return decided(inputs, seen)

        return taken

    def with_digest(
        current: str, candidate: str
    ) -> Callable[[str], IO[Result[RunState, TranslationError]]]:
        def taken(digest: str) -> IO[Result[RunState, TranslationError]]:
            inputs = RoundInputs(current, candidate, digest)
            return io_bind(
                seen_digests(optimization_environment), chain_result(with_seen(inputs))
            )

        return taken

    def with_candidate(
        current: str,
    ) -> Callable[[str], IO[Result[RunState, TranslationError]]]:
        def taken(candidate: str) -> IO[Result[RunState, TranslationError]]:
            return io_bind(
                prepare_version(optimization_environment, candidate_version, candidate),
                chain_result(with_digest(current, candidate)),
            )

        return taken

    def with_critiques(
        current: str,
    ) -> Callable[[str], IO[Result[RunState, TranslationError]]]:
        def taken(critiques: str) -> IO[Result[RunState, TranslationError]]:
            return io_bind(
                candidate_prompt(
                    optimization_environment, state, current, history_text, critiques
                ),
                chain_result(with_candidate(current)),
            )

        return taken

    def with_current(current: str) -> IO[Result[RunState, TranslationError]]:
        return io_bind(
            previous_critiques(optimization_environment, state.round_no),
            with_critiques(current),
        )

    return io_bind(
        read_text_file(
            os.path.join(prompts_directory, state.incumbent + ".txt"),
            "current instruction",
        ),
        chain_result(with_current),
    )


def next_round(
    optimization_environment: OptimizationEnvironment,
) -> Callable[
    [Result[RunState, TranslationError]], IO[Result[RunState, TranslationError]]
]:
    def continue_(
        outcome: Result[RunState, TranslationError],
    ) -> IO[Result[RunState, TranslationError]]:
        if isinstance(outcome, Err):
            return io_result(outcome)
        return run_rounds(optimization_environment, outcome.value)

    return continue_


def run_rounds(
    optimization_environment: OptimizationEnvironment, state: RunState
) -> IO[Result[RunState, TranslationError]]:
    stopping = state.stall >= optimization_environment.options.stall

    if state.round_no > optimization_environment.options.rounds or stopping:

        def stopped(_: None) -> Result[RunState, TranslationError]:
            return Ok(state)

        return io_bind(
            io_when_unit(
                stopping,
                optimization_environment.say(
                    "stopping: %d consecutive rounds without promotion" % state.stall
                ),
            ),
            lambda _: io_result(stopped(None)),
        )

    return io_bind(
        evaluate_round(optimization_environment, state),
        next_round(optimization_environment),
    )


def read_prompt(
    optimization_environment: OptimizationEnvironment, version: str
) -> IO[Result[str, TranslationError]]:
    return read_text_file(
        os.path.join(
            join_directory(optimization_environment.working_directory, "prompts"),
            version + ".txt",
        ),
        "instruction file",
    )


def holdout_check(
    optimization_environment: OptimizationEnvironment, final_version: str
) -> IO[Result[None, TranslationError]]:
    holdout = os.path.join(
        join_directory(optimization_environment.working_directory, "chapters"),
        "holdout.txt",
    )
    report_path = os.path.join(
        join_directory(optimization_environment.working_directory, "judge"),
        "holdout.json",
    )

    def with_judging(
        judging_result: Result[ChapterJudging, TranslationError],
    ) -> IO[Result[None, TranslationError]]:
        if isinstance(judging_result, Err):
            return io_result(judging_result)
        judging = judging_result.value
        winner = {"candidate": final_version, "incumbent": SEED_VERSION, "tie": "tie"}[
            judging.outcome
        ]
        report = {
            "outcome": judging.outcome,
            "seed_as_a": judging.first,
            "final_as_a": judging.second,
        }

        def with_report(
            written: Result[None, TranslationError],
        ) -> IO[Result[None, TranslationError]]:
            if isinstance(written, Err):
                return io_result(written)

            def announced(_: None) -> Result[None, TranslationError]:
                return Ok(None)

            return io_map(
                announce(
                    optimization_environment,
                    (
                        "holdout verdict: %s wins" % winner,
                        "seed shown as A: %s" % judging.first.get("critique"),
                        "final shown as A: %s" % judging.second.get("critique"),
                    ),
                ),
                announced,
            )

        return io_bind(
            write_text_file(
                report_path,
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                "holdout verdict",
            ),
            with_report,
        )

    def with_texts(
        texts: Result[tuple[str, str, str], TranslationError],
    ) -> IO[Result[None, TranslationError]]:
        if isinstance(texts, Err):
            return io_result(texts)
        source_text, seed_output, final_output = texts.value
        return io_bind(
            judge_chapter(
                optimization_environment,
                source_text,
                seed_output,
                final_output,
                "holdout",
                0,
            ),
            with_judging,
        )

    def translated(_: None) -> IO[Result[None, TranslationError]]:
        def after_seed(
            seed_outcome: Result[None, TranslationError],
        ) -> IO[Result[None, TranslationError]]:
            return result_bind_io(
                seed_outcome,
                lambda _: translate_chapter(
                    optimization_environment, final_version, holdout
                ),
            )

        return io_bind(
            translate_chapter(optimization_environment, SEED_VERSION, holdout),
            after_seed,
        )

    def with_final_text(final_text: str) -> IO[Result[str, TranslationError]]:
        def prepare_final(_: str) -> IO[Result[str, TranslationError]]:
            return prepare_version(optimization_environment, final_version, final_text)

        def with_seed_text(seed_text: str) -> IO[Result[str, TranslationError]]:
            return io_bind(
                prepare_version(optimization_environment, SEED_VERSION, seed_text),
                chain_result(prepare_final),
            )

        return io_bind(
            read_prompt(optimization_environment, SEED_VERSION),
            chain_result(with_seed_text),
        )

    def seeded(_: None) -> IO[Result[str, TranslationError]]:
        return io_bind(
            read_prompt(optimization_environment, final_version),
            chain_result(with_final_text),
        )

    def judging(_: None) -> IO[Result[None, TranslationError]]:
        return io_bind(
            round_texts(optimization_environment, SEED_VERSION, final_version, holdout),
            with_texts,
        )

    def prepared(_: str) -> IO[Result[None, TranslationError]]:
        return io_bind(translated(None), chain_result(judging))

    return io_bind(seeded(None), chain_result(prepared))


def compare_step(
    optimization_environment: OptimizationEnvironment,
    versions: tuple[str, str],
    verdict_directory: str,
) -> Callable[[CompareTally, str], IO[Result[CompareTally, TranslationError]]]:
    def step(
        tally: CompareTally, chapter: str
    ) -> IO[Result[CompareTally, TranslationError]]:
        def with_judging(
            judging_result: Result[ChapterJudging, TranslationError],
        ) -> IO[Result[CompareTally, TranslationError]]:
            if isinstance(judging_result, Err):
                return io_result(judging_result)
            judging = judging_result.value
            side = {"incumbent": "a", "candidate": "b", "tie": "tie"}[judging.outcome]
            label = {"a": "A", "b": "B", "tie": "tie"}[side]
            entry = {
                "chapter": os.path.basename(chapter),
                "outcome": side,
                "a_as_a": judging.first,
                "b_as_a": judging.second,
            }
            verdict_path = os.path.join(verdict_directory, text_stem(chapter) + ".json")

            def with_verdict(
                written: Result[None, TranslationError],
            ) -> IO[Result[CompareTally, TranslationError]]:
                if isinstance(written, Err):
                    return io_result(written)

                def announced(_: None) -> Result[CompareTally, TranslationError]:
                    return Ok(record_compare(tally, entry, side))

                return io_map(
                    announce(
                        optimization_environment,
                        (
                            "chapter %s: %s wins" % (os.path.basename(chapter), label),
                            "  A shown as A: %s" % judging.first.get("critique"),
                            "  B shown as A: %s" % judging.second.get("critique"),
                        ),
                    ),
                    announced,
                )

            return io_bind(
                write_text_file(
                    verdict_path,
                    json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
                    "chapter verdict",
                ),
                with_verdict,
            )

        def with_texts(
            texts: Result[tuple[str, str, str], TranslationError],
        ) -> IO[Result[CompareTally, TranslationError]]:
            if isinstance(texts, Err):
                return io_result(texts)
            source_text, left, right = texts.value
            return io_bind(
                judge_chapter(
                    optimization_environment,
                    source_text,
                    left,
                    right,
                    os.path.basename(chapter),
                    0,
                ),
                with_judging,
            )

        def judging(_: None) -> IO[Result[CompareTally, TranslationError]]:
            return io_bind(
                round_texts(
                    optimization_environment, versions[0], versions[1], chapter
                ),
                with_texts,
            )

        def after_first(
            first: Result[None, TranslationError],
        ) -> IO[Result[None, TranslationError]]:
            return result_bind_io(
                first,
                lambda _: translate_chapter(
                    optimization_environment, versions[1], chapter
                ),
            )

        def translated(_: None) -> IO[Result[None, TranslationError]]:
            return io_bind(
                translate_chapter(optimization_environment, versions[0], chapter),
                after_first,
            )

        return io_bind(translated(None), chain_result(judging))

    return step


def record_compare(
    tally: CompareTally, entry: dict[str, Any], side: str
) -> CompareTally:
    if side == "a":
        return replace(tally, a_wins=tally.a_wins + 1, rounds=tally.rounds + (entry,))
    if side == "b":
        return replace(tally, b_wins=tally.b_wins + 1, rounds=tally.rounds + (entry,))
    return replace(tally, tie=tally.tie + 1, rounds=tally.rounds + (entry,))


def summarize_compare(
    optimization_environment: OptimizationEnvironment,
    versions: tuple[str, str],
    verdict_directory: str,
    paths: tuple[str, str],
) -> Callable[[CompareTally], IO[Result[None, TranslationError]]]:
    def finish(tally: CompareTally) -> IO[Result[None, TranslationError]]:
        majority = len(optimization_environment.chapters) // 2 + 1
        if tally.a_wins >= majority:
            winner = "A"
        elif tally.b_wins >= majority:
            winner = "B"
        else:
            winner = "tie"
        summary = {
            "prompt_a": {"path": paths[0], "version": versions[0]},
            "prompt_b": {"path": paths[1], "version": versions[1]},
            "chapters": tuple(
                os.path.basename(chapter)
                for chapter in optimization_environment.chapters
            ),
            "a_wins": tally.a_wins,
            "b_wins": tally.b_wins,
            "ties": tally.tie,
            "winner": winner,
            "rounds": tally.rounds,
        }
        summary_path = os.path.join(verdict_directory, "summary.json")

        def with_summary(
            written: Result[None, TranslationError],
        ) -> IO[Result[None, TranslationError]]:
            if isinstance(written, Err):
                return io_result(written)

            def announced(_: None) -> Result[None, TranslationError]:
                return Ok(None)

            return io_map(
                announce(
                    optimization_environment,
                    (
                        "compare verdict: A wins %d, B wins %d, ties %d -> %s wins"
                        % (tally.a_wins, tally.b_wins, tally.tie, winner),
                        "summary: %s" % summary_path,
                    ),
                ),
                announced,
            )

        return io_bind(
            write_text_file(
                summary_path,
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                "compare summary",
            ),
            with_summary,
        )

    return finish


def begin_compare(
    optimization_environment: OptimizationEnvironment,
    paths: tuple[str, str],
    prompts: tuple[str, str],
) -> IO[Result[None, TranslationError]]:
    versions = ("cmp-" + hash_text(prompts[0]), "cmp-" + hash_text(prompts[1]))
    if versions[0] == versions[1]:
        return io_result(fail_config("both prompts are identical (%s)" % paths[0]))
    verdict_directory = os.path.join(
        optimization_environment.working_directory,
        "judge",
        "compare",
        versions[0] + "-vs-" + versions[1],
    )

    def summarize(tally: CompareTally) -> IO[Result[None, TranslationError]]:
        return summarize_compare(
            optimization_environment, versions, verdict_directory, paths
        )(tally)

    def judged(_: str) -> IO[Result[None, TranslationError]]:
        return io_bind(
            fold_io(
                optimization_environment.chapters,
                compare_step(optimization_environment, versions, verdict_directory),
                Ok(CompareTally()),
            ),
            chain_result(summarize),
        )

    def with_second(_: str) -> IO[Result[None, TranslationError]]:
        return io_bind(
            prepare_version(optimization_environment, versions[1], prompts[1]),
            chain_result(judged),
        )

    def with_first(_: None) -> IO[Result[None, TranslationError]]:
        return io_bind(
            prepare_version(optimization_environment, versions[0], prompts[0]),
            chain_result(with_second),
        )

    def with_directory(_: None) -> IO[Result[None, TranslationError]]:
        return with_first(None)

    return io_bind(ensure_directory(verdict_directory), with_directory)


def run_compare(
    optimization_environment: OptimizationEnvironment,
) -> IO[Result[None, TranslationError]]:
    options = optimization_environment.options
    selected = options.compare or ("", "")
    paths = (os.path.abspath(selected[0]), os.path.abspath(selected[1]))

    def existing(path: str) -> IO[Result[bool, TranslationError]]:
        return io_map(path_exists(path), Ok)

    def with_right_text(
        left: str,
    ) -> Callable[[str], IO[Result[None, TranslationError]]]:
        def taken(right: str) -> IO[Result[None, TranslationError]]:
            return begin_compare(optimization_environment, paths, (left, right))

        return taken

    def with_left(left: str) -> IO[Result[None, TranslationError]]:
        return io_bind(
            read_text_file(paths[1], "instruction"),
            chain_result(with_right_text(left)),
        )

    def with_existence(
        outcomes: Result[tuple[bool, ...], TranslationError],
    ) -> IO[Result[None, TranslationError]]:
        if isinstance(outcomes, Err):
            return io_result(outcomes)
        missing = next(
            (
                path
                for path, present in zip(paths, outcomes.value, strict=True)
                if not present
            ),
            None,
        )
        if missing is not None:
            return io_result(fail_config("prompt file not found: %s" % missing))
        return io_bind(read_text_file(paths[0], "instruction"), chain_result(with_left))

    return io_bind(io_traverse(paths, existing), with_existence)


def resolve_api(
    options: Options, base_data: Mapping[str, Any], environment: Mapping[str, str]
) -> IO[Result[Api, TranslationError]]:
    base_api = as_table(base_data.get("api"))

    def with_user(user_api: Mapping[str, Any]) -> Result[Api, TranslationError]:
        base_url = (
            options.base_url
            or environment.get("TRANSLATE_BASE_URL")
            or base_api.get("base_url")
            or user_api.get("base_url")
        )
        api_key = (
            options.api_key
            or environment.get("TRANSLATE_API_KEY")
            or base_api.get("api_key")
            or user_api.get("api_key")
        )
        if not base_url or not api_key:
            return fail_config(
                "no endpoint credentials: pass --base-url/--api-key, set "
                "TRANSLATE_BASE_URL/TRANSLATE_API_KEY, or set api.base_url and "
                "api.api_key in a config file"
            )
        return Ok(Api(base_url=str(base_url).rstrip("/"), api_key=str(api_key)))

    def with_user_table(
        loaded: Result[dict[str, Any], TranslationError],
    ) -> Result[Api, TranslationError]:
        if isinstance(loaded, Err):
            return loaded
        return with_user(as_table(loaded.value))

    def with_user_path(maybe_path: str) -> IO[Result[Api, TranslationError]]:
        def with_present(present: bool) -> IO[Result[Api, TranslationError]]:
            if not present:
                return io_result(with_user({}))
            return io_map(load_toml(maybe_path, "user config"), with_user_table)

        return io_bind(path_exists(maybe_path), with_present)

    return io_bind(user_config_path(environment), with_user_path)


def report_failure(warn: Reporter, error: TranslationError) -> IO[int]:
    def done(_: None) -> IO[int]:
        return io_pure(1)

    return io_bind(warn(describe(error)), done)


def done_zero(_: None) -> IO[int]:
    return io_pure(0)


def stream_reporter(stream: TextIO) -> Reporter:
    return lambda line: write_stdout(stream, line)


def template_error(
    text: str, tokens: tuple[str, ...], path: str, label: str
) -> TranslationError | None:
    for token in tokens:
        if token not in text:
            return ConfigError("%s template %s lacks %s" % (label, path, token))
    return None


def repo_run(command: tuple[str, ...], stdin_text: str) -> IO[ProcessResult]:
    return run_process(command, stdin_text, REPO_ROOT)


def base_environment(
    api: Api,
    base: BaseConfig,
    working_directory: str,
    judge_template: str,
    l2l_command: tuple[str, ...],
    options: Options,
    environment: Mapping[str, str],
    sleep: Sleep,
    say: Reporter,
    warn: Reporter,
    run_command: RunCommand,
    opener: OpenEndpoint | None,
) -> OptimizationEnvironment:
    return OptimizationEnvironment(
        options=options,
        base=base,
        api=api,
        working_directory=working_directory,
        environment=environment,
        chapters=(),
        judge_template=judge_template,
        rewrite_template="",
        l2l_command=l2l_command,
        opener=(
            curl_endpoint(api, environment, options.call_timeout)
            if opener is None
            else opener
        ),
        sleep=sleep,
        say=say,
        warn=warn,
        run_command=run_command,
    )


def finish_evolution(
    optimization_environment: OptimizationEnvironment,
    rounds: IO[Result[RunState, TranslationError]],
) -> IO[int]:
    def failed(error: TranslationError) -> IO[int]:
        return report_failure(optimization_environment.warn, error)

    def with_holdout(final: str) -> Callable[[None], IO[int]]:
        def done(_: None) -> IO[int]:
            return io_bind(
                holdout_check(optimization_environment, final),
                lambda outcome: result_either(outcome, lambda _: io_pure(0), failed),
            )

        return done

    def conclude(final: str) -> Callable[[None], IO[int]]:
        def done(_: None) -> IO[int]:
            def holdout_or_stop(_: None) -> IO[int]:
                if final == SEED_VERSION:
                    return io_bind(
                        optimization_environment.say(
                            "no challenger was ever promoted; holdout check skipped"
                        ),
                        done_zero,
                    )
                return with_holdout(final)(None)

            return io_bind(
                optimization_environment.say(
                    "final incumbent: %s (%s)"
                    % (
                        final,
                        os.path.join(
                            join_directory(
                                optimization_environment.working_directory, "prompts"
                            ),
                            final + ".txt",
                        ),
                    )
                ),
                holdout_or_stop,
            )

        return done

    def with_final(outcome: Result[RunState, TranslationError]) -> IO[int]:
        if isinstance(outcome, Err):
            return failed(outcome.error)
        return conclude(outcome.value.incumbent)(None)

    return io_bind(rounds, with_final)


def run_and_finish(optimization_environment: OptimizationEnvironment) -> IO[int]:
    def resumed(
        loaded: tuple[tuple[dict[str, Any], ...], ...],
    ) -> IO[int]:
        ledger, history = loaded[0], loaded[1]
        return finish_evolution(
            optimization_environment,
            run_rounds(optimization_environment, resume_state(history, ledger)),
        )

    return io_bind(
        io_sequence(
            (
                read_jsonl(
                    os.path.join(
                        optimization_environment.working_directory, "ledger.jsonl"
                    )
                ),
                read_jsonl(
                    os.path.join(
                        optimization_environment.working_directory, "history.jsonl"
                    )
                ),
            )
        ),
        resumed,
    )


def announce_plan(optimization_environment: OptimizationEnvironment) -> IO[int]:
    per_round = (
        1
        + len(optimization_environment.chapters)
        + 2 * len(optimization_environment.chapters)
    )

    def done(_: None) -> IO[int]:
        return run_and_finish(optimization_environment)

    return io_bind(
        optimization_environment.say(
            "workdir: %s; chapters: %d; up to %d endpoint calls per round "
            "(1 rewrite + up to %d translations + %d judgments)"
            % (
                optimization_environment.working_directory,
                len(optimization_environment.chapters),
                per_round,
                len(optimization_environment.chapters),
                2 * len(optimization_environment.chapters),
            )
        ),
        done,
    )


def start_rounds(optimization_environment: OptimizationEnvironment) -> IO[int]:
    seed_path = os.path.join(
        join_directory(optimization_environment.working_directory, "prompts"),
        SEED_VERSION + ".txt",
    )
    seed = optimization_environment.options.seed or ""

    def with_copied(copied: bool) -> Result[None, TranslationError]:
        return (
            Ok(None) if copied else fail_config("could not copy seed instruction file")
        )

    def copy_seed(exists: bool) -> IO[Result[None, TranslationError]]:
        if exists or seed == "":
            return io_result(Ok(None))
        return io_map(
            copy_file(os.path.abspath(seed), seed_path),
            with_copied,
        )

    def with_seed_present(present: bool) -> IO[int]:
        if not present:
            return report_failure(
                optimization_environment.warn,
                ConfigError(
                    "missing %s (pass --seed PATH once to create it)" % seed_path
                ),
            )
        return announce_plan(optimization_environment)

    def seeded(outcome: Result[None, TranslationError]) -> IO[int]:
        if isinstance(outcome, Err):
            return report_failure(optimization_environment.warn, outcome.error)
        return io_bind(path_exists(seed_path), with_seed_present)

    return io_bind(io_bind(path_exists(seed_path), copy_seed), seeded)


def run_compare_mode(optimization_environment: OptimizationEnvironment) -> IO[int]:
    def failed(error: TranslationError) -> IO[int]:
        return report_failure(optimization_environment.warn, error)

    def with_chapters(
        chapters_result: Result[tuple[str, ...], TranslationError],
    ) -> IO[int]:
        if isinstance(chapters_result, Err):
            return failed(chapters_result.error)
        located = replace(optimization_environment, chapters=chapters_result.value)

        def announced(_: None) -> IO[int]:
            return io_bind(
                located.say(
                    "workdir: %s; comparing over %d chapter(s); up to %d endpoint "
                    "calls (2 translations + 2 judgments per chapter)"
                    % (
                        located.working_directory,
                        len(located.chapters),
                        4 * len(located.chapters),
                    )
                ),
                lambda _: io_bind(
                    run_compare(located),
                    lambda outcome: result_either(
                        outcome, lambda _: io_pure(0), failed
                    ),
                ),
            )

        return announced(None)

    return io_bind(
        compare_chapters(
            optimization_environment.working_directory, optimization_environment.options
        ),
        with_chapters,
    )


def run_evolution_mode(optimization_environment: OptimizationEnvironment) -> IO[int]:
    def with_chapters(
        chapters_result: Result[tuple[str, ...], TranslationError],
    ) -> IO[int]:
        if isinstance(chapters_result, Err):
            return report_failure(optimization_environment.warn, chapters_result.error)
        located = replace(optimization_environment, chapters=chapters_result.value)

        def with_rewrite(template_result: Result[str, TranslationError]) -> IO[int]:
            if isinstance(template_result, Err):
                return report_failure(located.warn, template_result.error)
            failure = template_error(
                template_result.value,
                (PROMPT_TOKEN, HISTORY_TOKEN, CRITIQUES_TOKEN),
                located.options.rewrite_template,
                "rewrite",
            )
            if failure is not None:
                return report_failure(located.warn, failure)
            return start_rounds(
                replace(located, rewrite_template=template_result.value)
            )

        return io_bind(
            read_text_file(located.options.rewrite_template, "rewrite template"),
            with_rewrite,
        )

    return io_bind(
        discover_chapters(
            join_directory(optimization_environment.working_directory, "chapters")
        ),
        with_chapters,
    )


def valid_working_directory(options: Options) -> IO[Result[str, TranslationError]]:
    working_directory = os.path.abspath(options.working_directory)

    def with_is_directory(is_directory: bool) -> Result[str, TranslationError]:
        if is_directory:
            return Ok(working_directory)
        return fail_config("workdir is not a directory: %s" % working_directory)

    def with_exists(exists: bool) -> IO[Result[str, TranslationError]]:
        if not exists:
            return io_result(Ok(working_directory))
        return io_map(path_is_directory(working_directory), with_is_directory)

    return io_bind(path_exists(working_directory), with_exists)


def enter_workspace(
    options: Options,
    environment: Mapping[str, str],
    sleep: Sleep,
    python_executable: str,
    run_command: RunCommand,
    opener: OpenEndpoint | None,
    say: Reporter,
    warn: Reporter,
) -> IO[int]:
    def entered(working_directory_result: Result[str, TranslationError]) -> IO[int]:
        if isinstance(working_directory_result, Err):
            return report_failure(warn, working_directory_result.error)
        working_directory = working_directory_result.value

        def with_directories(_: None) -> IO[int]:
            return load_workspace(
                options,
                environment,
                sleep,
                python_executable,
                run_command,
                opener,
                say,
                warn,
                working_directory,
            )

        def ensure_rest(_: None) -> IO[None]:
            return io_and_then(
                ensure_directory(join_directory(working_directory, "prompts")),
                io_and_then(
                    ensure_directory(join_directory(working_directory, "judge")),
                    ensure_directory(join_directory(working_directory, "cache")),
                ),
            )

        def with_root(_: None) -> IO[None]:
            return ensure_rest(None)

        return io_bind(
            io_and_then(ensure_directory(working_directory), with_root(None)),
            with_directories,
        )

    return io_bind(valid_working_directory(options), entered)


def load_workspace(
    options: Options,
    environment: Mapping[str, str],
    sleep: Sleep,
    python_executable: str,
    run_command: RunCommand,
    opener: OpenEndpoint | None,
    say: Reporter,
    warn: Reporter,
    working_directory: str,
) -> IO[int]:
    base_path = os.path.abspath(options.base_config)
    l2l_command = (
        (python_executable, "-m", "l2l")
        if options.l2l is None
        else tuple(shlex.split(options.l2l))
    )

    def with_judge(
        base: BaseConfig, api: Api
    ) -> Callable[[Result[str, TranslationError]], IO[int]]:
        def taken(judge_result: Result[str, TranslationError]) -> IO[int]:
            if isinstance(judge_result, Err):
                return report_failure(warn, judge_result.error)
            failure = template_error(
                judge_result.value,
                (SOURCE_TOKEN, A_TOKEN, B_TOKEN),
                options.judge_template,
                "judge",
            )
            if failure is not None:
                return report_failure(warn, failure)
            optimization_environment = base_environment(
                api,
                base,
                working_directory,
                judge_result.value,
                l2l_command,
                options,
                environment,
                sleep,
                say,
                warn,
                run_command,
                opener,
            )
            return (
                run_compare_mode(optimization_environment)
                if options.compare is not None
                else run_evolution_mode(optimization_environment)
            )

        return taken

    def with_api(
        base: BaseConfig,
    ) -> Callable[[Result[Api, TranslationError]], IO[int]]:
        def taken(api_result: Result[Api, TranslationError]) -> IO[int]:
            if isinstance(api_result, Err):
                return report_failure(warn, api_result.error)
            return io_bind(
                read_text_file(options.judge_template, "judge template"),
                with_judge(base, api_result.value),
            )

        return taken

    def with_base_data(loaded: Result[dict[str, Any], TranslationError]) -> IO[int]:
        if isinstance(loaded, Err):
            return report_failure(warn, loaded.error)
        built = load_base_config(loaded.value, options.pass_name, base_path)
        if isinstance(built, Err):
            return report_failure(warn, built.error)
        return io_bind(
            resolve_api(options, loaded.value, environment), with_api(built.value)
        )

    def with_base_path(exists: bool) -> IO[int]:
        if not exists:
            return report_failure(
                warn, ConfigError("base config not found: %s" % base_path)
            )
        return io_bind(load_toml(base_path, "base config"), with_base_data)

    return io_bind(path_exists(base_path), with_base_path)


def main(
    argv: Sequence[str],
    environment: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    sleep: Sleep,
    python_executable: str,
    run_command: RunCommand = repo_run,
    opener: OpenEndpoint | None = None,
) -> IO[int]:
    say = stream_reporter(stdout)
    warn = stream_reporter(stderr)

    def with_options(options_result: Result[Options, TranslationError]) -> IO[int]:
        if isinstance(options_result, Err):
            return report_failure(warn, options_result.error)
        return enter_workspace(
            options_result.value,
            environment,
            sleep,
            python_executable,
            run_command,
            opener,
            say,
            warn,
        )

    return io_bind(io_result(parse_arguments(argv)), with_options)


def optimize() -> None:
    try:
        code = main(
            sys.argv[1:],
            dict(os.environ),
            sys.stdout,
            sys.stderr,
            time.sleep,
            sys.executable,
        ).run()
    except KeyboardInterrupt:
        print("optimize: interrupted", file=sys.stderr)
        code = 130

    sys.exit(code)


if __name__ == "__main__":
    optimize()

import argparse
import hashlib
import http.client
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_TOKEN = "<<<SOURCE>>>"
A_TOKEN = "<<<TRANSLATION_A>>>"
B_TOKEN = "<<<TRANSLATION_B>>>"
PROMPT_TOKEN = "<<<CURRENT_PROMPT>>>"
HISTORY_TOKEN = "<<<HISTORY>>>"
CRITIQUES_TOKEN = "<<<CRITIQUES>>>"

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
SEED_VERSION = "v0"


@dataclass(frozen=True)
class Api:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class BaseConfig:
    api: dict[str, Any]
    options: dict[str, Any]
    passes: list[dict[str, Any]]
    target_index: int
    target_name: str


def parse_args(argv: list[str] | None) -> argparse.Namespace:
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
        type=Path,
        required=True,
        help="l2l config supplying the [api] table and the pass pipeline",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        required=True,
        help="experiment directory holding chapters/, prompts/, out/, judge/",
    )
    parser.add_argument(
        "--seed",
        type=Path,
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
        type=Path,
        default=TOOLS_DIR / "judge.txt",
    )
    parser.add_argument(
        "--rewrite-template",
        type=Path,
        default=TOOLS_DIR / "rewrite.txt",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--rewrite-temperature", type=float, default=0.8)
    parser.add_argument("--translator-temperature", type=float, default=0.0)
    parser.add_argument("--call-timeout", type=float, default=600.0)
    parser.add_argument("--call-max-tokens", type=int, default=32768)
    parser.add_argument(
        "--judge-params",
        help=(
            "JSON object merged into judge request bodies, e.g. "
            '\'{"reasoning": {"effort": "low"}}\' for thinking models'
        ),
    )
    parser.add_argument(
        "--rewrite-params",
        help="JSON object merged into rewrite request bodies",
    )
    parser.add_argument("--history-depth", type=int, default=6)
    return parser.parse_args(argv)


def parse_params(raw: str | None, flag: str) -> dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SystemExit("--%s must be a JSON object" % flag)
    return parsed


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def user_config_path() -> Path | None:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    candidate = base / "l2l" / "config.toml"
    return candidate if candidate.is_file() else None


def resolve_api(args: argparse.Namespace, base_data: dict[str, Any]) -> Api:
    base_api = base_data.get("api", {})
    user_api: dict[str, Any] = {}
    user_path = user_config_path()
    if user_path is not None:
        user_api = load_toml(user_path).get("api", {})
    base_url = (
        args.base_url
        or os.environ.get("TRANSLATE_BASE_URL")
        or base_api.get("base_url")
        or user_api.get("base_url")
    )
    api_key = (
        args.api_key
        or os.environ.get("TRANSLATE_API_KEY")
        or base_api.get("api_key")
        or user_api.get("api_key")
    )
    if not base_url or not api_key:
        raise SystemExit(
            "no endpoint credentials: pass --base-url/--api-key, set "
            "TRANSLATE_BASE_URL/TRANSLATE_API_KEY, or set api.base_url and "
            "api.api_key in a config file"
        )
    return Api(base_url=str(base_url), api_key=str(api_key))


def load_base_config(
    data: dict[str, Any],
    pass_name: str | None,
    path: Path,
) -> BaseConfig:
    passes: list[dict[str, Any]] = [dict(entry) for entry in data.get("pass", [])]
    if not passes:
        raise SystemExit("base config %s has no [[pass]] entries" % path)
    if pass_name is not None:
        matches = [
            index
            for index, entry in enumerate(passes)
            if entry.get("name") == pass_name
        ]
        if not matches:
            raise SystemExit("base config %s has no pass named %r" % (path, pass_name))
        index = matches[0]
    elif len(passes) == 1:
        index = 0
    else:
        raise SystemExit(
            "base config %s has %d passes; pass --pass-name to pick one"
            % (path, len(passes))
        )
    target = passes[index]
    name = str(target.get("name") or "translate")
    api: dict[str, Any] = dict(data.get("api", {}))
    options: dict[str, Any] = dict(data.get("options", {}))
    return BaseConfig(api, options, passes, index, name)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    raise SystemExit("unsupported config value: %r" % (value,))


def emit_body(prefix: str, table: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    nested = [
        (key, value)
        for key, value in table.items()
        if isinstance(value, dict) and value
    ]
    for key, value in table.items():
        if isinstance(value, dict):
            continue
        lines.append("%s = %s" % (key, toml_value(value)))
    for key, value in nested:
        child = "%s.%s" % (prefix, key)
        lines.append("")
        lines.append("[%s]" % child)
        lines.extend(emit_body(child, value))
    return lines


def render_config(
    api: dict[str, Any],
    options: dict[str, Any],
    passes: list[dict[str, Any]],
) -> str:
    lines = ["[api]"]
    lines.extend(emit_body("api", api))
    if options:
        lines.append("")
        lines.append("[options]")
        lines.extend(emit_body("options", options))
    for entry in passes:
        lines.append("")
        lines.append("[[pass]]")
        lines.extend(emit_body("pass", entry))
    return "\n".join(lines) + "\n"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prepare_version(
    workdir: Path,
    base: BaseConfig,
    version: str,
    prompt_text: str,
    translator_model: str | None,
    temperature: float,
) -> str:
    digest = hash_text(prompt_text)
    prompts_dir = workdir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / (version + ".txt")).write_text(prompt_text, encoding="utf-8")
    passes = [dict(entry) for entry in base.passes]
    original = passes[base.target_index]
    entry = {
        key: value
        for key, value in original.items()
        if key not in ("instruction", "instruction_file")
    }
    entry["name"] = "%s-%s" % (base.target_name, digest[:10])
    entry["instruction_file"] = "../prompts/%s.txt" % version
    params: dict[str, Any] = dict(entry.get("params", {}))
    params["temperature"] = temperature
    entry["params"] = params
    if translator_model:
        entry["model"] = translator_model
    passes[base.target_index] = entry
    gen_dir = workdir / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    config_path = gen_dir / (version + ".toml")
    config_path.write_text(
        render_config(base.api, base.options, passes), encoding="utf-8"
    )
    return digest


def discover_chapters(chapters_dir: Path) -> list[Path]:
    if not chapters_dir.is_dir():
        raise SystemExit(
            "missing %s (expected chapter .txt files plus holdout.txt)" % chapters_dir
        )
    chapters = sorted(
        path for path in chapters_dir.glob("*.txt") if path.name != "holdout.txt"
    )
    if len(chapters) < 2:
        raise SystemExit("need at least two chapters in %s" % chapters_dir)
    holdout = chapters_dir / "holdout.txt"
    if not holdout.is_file():
        raise SystemExit("missing %s" % holdout)
    return chapters


def output_path(out_dir: Path, version: str, chapter: Path) -> Path:
    return out_dir / version / (chapter.stem + ".en.txt")


def translate_chapter(
    l2l_cmd: list[str],
    workdir: Path,
    version: str,
    chapter: Path,
    cache_dir: Path,
) -> None:
    target = output_path(workdir / "out", version, chapter)
    if target.is_file():
        return
    config = workdir / "gen" / (version + ".toml")
    if not config.is_file():
        raise SystemExit("missing generated config %s" % config)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.parent / (target.name + ".partial")
    stderr_path = target.parent / (target.name + ".err")
    with (
        chapter.open("rb") as stdin,
        partial.open("wb") as stdout,
        stderr_path.open("wb") as stderr,
    ):
        result = subprocess.run(
            [*l2l_cmd, str(config), "--cache-dir", str(cache_dir)],
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            cwd=REPO_ROOT,
        )
    if result.returncode != 0:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            "l2l exited with %d for %s; see %s"
            % (result.returncode, chapter, stderr_path)
        )
    partial.replace(target)


def read_output(out_dir: Path, version: str, chapter: Path) -> str:
    path = output_path(out_dir, version, chapter)
    if not path.is_file():
        raise SystemExit("missing translation %s" % path)
    return path.read_text(encoding="utf-8")


def compact(text: str, limit: int = 400) -> str:
    return " ".join(text.split())[:limit]


def chat(
    api: Api,
    model: str,
    content: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    extra: dict[str, Any] | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(extra or {})
    request = urllib.request.Request(
        api.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api.api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise ValueError("HTTP %d: %s" % (error.code, compact(detail))) from None
    try:
        body = json.loads(raw)
    except json.JSONDecodeError:
        raise ValueError("non-JSON reply: %s" % compact(raw)) from None
    if not isinstance(body, dict):
        raise ValueError("unexpected reply: %s" % compact(raw))
    if body.get("error") is not None:
        raise ValueError("endpoint error: %s" % json.dumps(body["error"])[:400])
    choices = body.get("choices")
    first = choices[0] if isinstance(choices, list) and choices else None
    message = first.get("message") if isinstance(first, dict) else None
    choice = message.get("content") if isinstance(message, dict) else None
    if not isinstance(choice, str) or not choice.strip():
        finish = first.get("finish_reason") if isinstance(first, dict) else None
        reasoning = isinstance(message, dict) and bool(message.get("reasoning"))
        hint = (
            "; raise --call-max-tokens or cap reasoning, e.g. --judge-params "
            '\'{"reasoning": {"effort": "low"}}\''
            if finish == "length"
            else ""
        )
        raise ValueError(
            "empty content (finish_reason=%s, reasoning=%s)%s: %s"
            % (finish, reasoning, hint, compact(raw))
        )
    return choice


def complete(
    api: Api,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    label: str,
    extra: dict[str, Any] | None = None,
    attempts: int = 4,
) -> str:
    message = "no attempts made"
    for attempt in range(1, attempts + 1):
        try:
            return chat(api, model, prompt, temperature, timeout, max_tokens, extra)
        except (
            OSError,
            http.client.HTTPException,
            KeyError,
            IndexError,
            ValueError,
        ) as error:
            message = "%s: %s" % (type(error).__name__, error)
            print(
                "%s: attempt %d/%d failed (%s)" % (label, attempt, attempts, message),
                file=sys.stderr,
            )
            if attempt < attempts:
                time.sleep(2.0 * attempt)
    raise SystemExit("%s failed after %d attempts: %s" % (label, attempts, message))


def extract_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in reply")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("reply is not a JSON object")
    return parsed


def judge_call(
    api: Api,
    model: str,
    prompt: str,
    temperature: float,
    timeout: float,
    max_tokens: int,
    label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = ""
    for attempt in range(1, 4):
        text = complete(
            api, model, prompt, temperature, timeout, max_tokens, label, extra
        )
        try:
            verdict = extract_object(text)
            winner = verdict.get("winner")
            if winner in ("A", "B", "tie"):
                return verdict
            raise ValueError("winner must be A, B, or tie")
        except ValueError as error:
            print(
                "%s: attempt %d/3 unusable verdict (%s)" % (label, attempt, error),
                file=sys.stderr,
            )
            if attempt < 3:
                time.sleep(2.0 * attempt)
    print("%s: giving up; recording an error verdict" % label, file=sys.stderr)
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


def order_outcome(winner: str, candidate_side: str) -> str:
    if winner not in ("A", "B"):
        return "tie"
    if winner == candidate_side:
        return "candidate"
    return "incumbent"


def judge_chapter(
    api: Api,
    args: argparse.Namespace,
    template: str,
    source_text: str,
    incumbent_text: str,
    candidate_text: str,
    chapter: str,
    round_no: int,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    first = judge_call(
        api,
        args.judge_model,
        fill_judge(template, source_text, incumbent_text, candidate_text),
        args.judge_temperature,
        args.call_timeout,
        args.call_max_tokens,
        "judge r%d/%s (incumbent as A)" % (round_no, chapter),
        args.judge_extra,
    )
    second = judge_call(
        api,
        args.judge_model,
        fill_judge(template, source_text, candidate_text, incumbent_text),
        args.judge_temperature,
        args.call_timeout,
        args.call_max_tokens,
        "judge r%d/%s (candidate as A)" % (round_no, chapter),
        args.judge_extra,
    )
    one = order_outcome(str(first.get("winner")), "B")
    two = order_outcome(str(second.get("winner")), "A")
    if one == "candidate" and two == "candidate":
        outcome = "candidate"
    elif one == "incumbent" and two == "incumbent":
        outcome = "incumbent"
    else:
        outcome = "tie"
    return outcome, first, second


def feedback_lines(
    chapter: str,
    first: dict[str, Any],
    second: dict[str, Any],
) -> list[str]:
    def describe(label: str, verdict: dict[str, Any]) -> str:
        critique = str(verdict.get("critique") or "").strip()
        raw = verdict.get("evidence")
        evidence = "; ".join(str(item) for item in raw) if isinstance(raw, list) else ""
        return "%s [%s] %s | evidence: %s" % (
            chapter,
            label,
            critique,
            evidence or "none quoted",
        )

    return [
        describe("incumbent shown as A", first),
        describe("candidate shown as A", second),
    ]


def rewrite_prompt(
    api: Api,
    args: argparse.Namespace,
    template: str,
    current_prompt: str,
    history_text: str,
    critiques: str,
    round_no: int,
) -> str:
    prompt = (
        template.replace(PROMPT_TOKEN, current_prompt)
        .replace(HISTORY_TOKEN, history_text)
        .replace(CRITIQUES_TOKEN, critiques)
    )
    text = complete(
        api,
        args.rewriter_model,
        prompt,
        args.rewrite_temperature,
        args.call_timeout,
        args.call_max_tokens,
        "rewrite r%d" % round_no,
        args.rewrite_extra,
    )
    improved = strip_reply(text)
    if not improved:
        raise SystemExit("rewriter returned an empty instruction")
    return improved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def seen_digests(prompts_dir: Path) -> dict[str, str]:
    seen: dict[str, str] = {}
    for path in sorted(prompts_dir.glob("v*.txt")):
        digest = hash_text(path.read_text(encoding="utf-8"))
        if digest not in seen:
            seen[digest] = path.stem
    return seen


def run_rounds(
    api: Api,
    args: argparse.Namespace,
    workdir: Path,
    base: BaseConfig,
    chapters: list[Path],
    judge_template: str,
    rewrite_template: str,
    l2l_cmd: list[str],
) -> str:
    prompts_dir = workdir / "prompts"
    judge_dir = workdir / "judge"
    out_dir = workdir / "out"
    cache_dir = workdir / "cache"
    ledger_path = workdir / "ledger.jsonl"
    history_path = workdir / "history.jsonl"

    ledger = read_jsonl(ledger_path)
    history = read_jsonl(history_path)
    incumbent = SEED_VERSION
    for entry in ledger:
        if entry.get("decision") == "promoted":
            incumbent = str(entry["candidate"])
    stall_count = 0
    for entry in reversed(ledger):
        if entry.get("decision") == "promoted":
            break
        stall_count += 1

    round_no = len(ledger) + 1
    while round_no <= args.rounds and stall_count < args.stall:
        candidate_version = "v%d" % round_no
        current_prompt = (prompts_dir / (incumbent + ".txt")).read_text(
            encoding="utf-8"
        )
        candidate_path = prompts_dir / (candidate_version + ".txt")
        feedback_previous = judge_dir / ("r%d-feedback.txt" % (round_no - 1))
        critiques = (
            feedback_previous.read_text(encoding="utf-8")
            if feedback_previous.is_file()
            else "(none; this is the first round)"
        )
        depth = max(0, len(history) - args.history_depth)
        history_text = (
            "\n\n".join(
                json.dumps(entry, ensure_ascii=False) for entry in history[depth:]
            )
            or "(empty)"
        )

        if candidate_path.is_file():
            candidate_prompt = candidate_path.read_text(encoding="utf-8")
            print("round %d: reusing %s" % (round_no, candidate_path))
        else:
            print(
                "round %d: asking %s for a new instruction"
                % (round_no, args.rewriter_model)
            )
            candidate_prompt = rewrite_prompt(
                api,
                args,
                rewrite_template,
                current_prompt,
                history_text,
                critiques,
                round_no,
            )
            candidate_path.write_text(candidate_prompt, encoding="utf-8")

        digest = prepare_version(
            workdir,
            base,
            candidate_version,
            candidate_prompt,
            args.translator_model,
            args.translator_temperature,
        )
        seen = seen_digests(prompts_dir)
        duplicate = seen.get(digest)
        if duplicate is not None and duplicate != candidate_version:
            print(
                "round %d: candidate duplicates %s; skipping judging"
                % (round_no, duplicate)
            )
            append_jsonl(
                ledger_path,
                {
                    "round": round_no,
                    "incumbent": incumbent,
                    "candidate": candidate_version,
                    "duplicate_of": duplicate,
                    "decision": "duplicate",
                },
            )
            history.append(
                {
                    "round": round_no,
                    "version": candidate_version,
                    "decision": "duplicate",
                    "prompt": candidate_prompt,
                }
            )
            append_jsonl(history_path, history[-1])
            stall_count += 1
            round_no += 1
            continue

        prepare_version(
            workdir,
            base,
            incumbent,
            current_prompt,
            args.translator_model,
            args.translator_temperature,
        )
        for chapter in chapters:
            translate_chapter(l2l_cmd, workdir, incumbent, chapter, cache_dir)
            translate_chapter(l2l_cmd, workdir, candidate_version, chapter, cache_dir)

        wins = {"candidate": 0, "incumbent": 0, "tie": 0}
        lines: list[str] = []
        for chapter in chapters:
            source_text = chapter.read_text(encoding="utf-8")
            incumbent_text = read_output(out_dir, incumbent, chapter)
            candidate_text = read_output(out_dir, candidate_version, chapter)
            outcome, first, second = judge_chapter(
                api,
                args,
                judge_template,
                source_text,
                incumbent_text,
                candidate_text,
                chapter.name,
                round_no,
            )
            verdict_path = judge_dir / ("r%d-%s.json" % (round_no, chapter.stem))
            verdict_path.write_text(
                json.dumps(
                    {
                        "outcome": outcome,
                        "incumbent_as_a": first,
                        "candidate_as_a": second,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            wins[outcome] += 1
            lines.extend(feedback_lines(chapter.name, first, second))

        feedback_path = judge_dir / ("r%d-feedback.txt" % round_no)
        feedback_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        majority = len(chapters) // 2 + 1
        promoted = wins["candidate"] >= majority
        decision = "promoted" if promoted else "kept"
        append_jsonl(
            ledger_path,
            {
                "round": round_no,
                "incumbent": incumbent,
                "candidate": candidate_version,
                "candidate_wins": wins["candidate"],
                "incumbent_wins": wins["incumbent"],
                "ties": wins["tie"],
                "decision": decision,
                "prompt_digest": digest,
            },
        )
        history.append(
            {
                "round": round_no,
                "version": candidate_version,
                "decision": decision,
                "candidate_wins": wins["candidate"],
                "incumbent_wins": wins["incumbent"],
                "prompt": candidate_prompt,
            }
        )
        append_jsonl(history_path, history[-1])
        print(
            "round %d: candidate %s wins %d, incumbent %s wins %d, ties %d -> %s"
            % (
                round_no,
                candidate_version,
                wins["candidate"],
                incumbent,
                wins["incumbent"],
                wins["tie"],
                decision,
            )
        )
        if promoted:
            incumbent = candidate_version
            stall_count = 0
        else:
            stall_count += 1
        round_no += 1

    if stall_count >= args.stall:
        print("stopping: %d consecutive rounds without promotion" % stall_count)
    return incumbent


def holdout_check(
    api: Api,
    args: argparse.Namespace,
    workdir: Path,
    base: BaseConfig,
    final_version: str,
    judge_template: str,
    l2l_cmd: list[str],
) -> None:
    chapters_dir = workdir / "chapters"
    holdout = chapters_dir / "holdout.txt"
    out_dir = workdir / "out"
    cache_dir = workdir / "cache"
    prompts_dir = workdir / "prompts"
    judge_dir = workdir / "judge"

    prepare_version(
        workdir,
        base,
        SEED_VERSION,
        (prompts_dir / (SEED_VERSION + ".txt")).read_text(encoding="utf-8"),
        args.translator_model,
        args.translator_temperature,
    )
    prepare_version(
        workdir,
        base,
        final_version,
        (prompts_dir / (final_version + ".txt")).read_text(encoding="utf-8"),
        args.translator_model,
        args.translator_temperature,
    )
    translate_chapter(l2l_cmd, workdir, SEED_VERSION, holdout, cache_dir)
    translate_chapter(l2l_cmd, workdir, final_version, holdout, cache_dir)

    outcome, first, second = judge_chapter(
        api,
        args,
        judge_template,
        holdout.read_text(encoding="utf-8"),
        read_output(out_dir, SEED_VERSION, holdout),
        read_output(out_dir, final_version, holdout),
        "holdout",
        0,
    )
    (judge_dir / "holdout.json").write_text(
        json.dumps(
            {"outcome": outcome, "seed_as_a": first, "final_as_a": second},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    winner = {
        "candidate": final_version,
        "incumbent": SEED_VERSION,
        "tie": "tie",
    }[outcome]
    print("holdout verdict: %s wins" % winner)
    print("seed shown as A: %s" % first.get("critique"))
    print("final shown as A: %s" % second.get("critique"))


def compare_chapters(workdir: Path, args: argparse.Namespace) -> list[Path]:
    chapters_dir = workdir / "chapters"
    if args.holdout:
        holdout = chapters_dir / "holdout.txt"
        if not holdout.is_file():
            raise SystemExit("missing %s" % holdout)
        return [holdout]
    chapters = discover_chapters(chapters_dir)
    if not args.chapters:
        return chapters
    by_stem = {chapter.stem: chapter for chapter in chapters}
    selected: dict[str, Path] = {}
    for raw in args.chapters.split(","):
        name = raw.strip()
        if not name or name in selected:
            continue
        chapter = by_stem.get(name)
        if chapter is None:
            raise SystemExit("unknown chapter %r in %s" % (name, chapters_dir))
        selected[name] = chapter
    if not selected:
        raise SystemExit("--chapters named none of: %s" % ", ".join(sorted(by_stem)))
    return list(selected.values())


def compare_prompts(
    api: Api,
    args: argparse.Namespace,
    workdir: Path,
    base: BaseConfig,
    chapters: list[Path],
    judge_template: str,
    l2l_cmd: list[str],
) -> None:
    out_dir = workdir / "out"
    cache_dir = workdir / "cache"
    paths = [Path(raw).resolve() for raw in args.compare]
    for path in paths:
        if not path.is_file():
            raise SystemExit("prompt file not found: %s" % path)
    prompts = [path.read_text(encoding="utf-8") for path in paths]
    versions = ["cmp-" + hash_text(prompt) for prompt in prompts]
    if versions[0] == versions[1]:
        raise SystemExit("both prompts are identical (%s)" % paths[0])
    for version, prompt in zip(versions, prompts, strict=True):
        prepare_version(
            workdir,
            base,
            version,
            prompt,
            args.translator_model,
            args.translator_temperature,
        )
    verdict_dir = workdir / "judge" / "compare" / (versions[0] + "-vs-" + versions[1])
    verdict_dir.mkdir(parents=True, exist_ok=True)

    wins = {"a": 0, "b": 0, "tie": 0}
    rounds: list[dict[str, Any]] = []
    for chapter in chapters:
        translate_chapter(l2l_cmd, workdir, versions[0], chapter, cache_dir)
        translate_chapter(l2l_cmd, workdir, versions[1], chapter, cache_dir)
        outcome, first, second = judge_chapter(
            api,
            args,
            judge_template,
            chapter.read_text(encoding="utf-8"),
            read_output(out_dir, versions[0], chapter),
            read_output(out_dir, versions[1], chapter),
            chapter.name,
            0,
        )
        side = {"incumbent": "a", "candidate": "b", "tie": "tie"}[outcome]
        wins[side] += 1
        rounds.append(
            {
                "chapter": chapter.name,
                "outcome": side,
                "a_as_a": first,
                "b_as_a": second,
            }
        )
        (verdict_dir / (chapter.stem + ".json")).write_text(
            json.dumps(rounds[-1], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        label = {"a": "A", "b": "B", "tie": "tie"}[side]
        print("chapter %s: %s wins" % (chapter.name, label))
        print("  A shown as A: %s" % first.get("critique"))
        print("  B shown as A: %s" % second.get("critique"))

    majority = len(chapters) // 2 + 1
    if wins["a"] >= majority:
        winner = "A"
    elif wins["b"] >= majority:
        winner = "B"
    else:
        winner = "tie"
    summary = {
        "prompt_a": {"path": str(paths[0]), "version": versions[0]},
        "prompt_b": {"path": str(paths[1]), "version": versions[1]},
        "chapters": [chapter.name for chapter in chapters],
        "a_wins": wins["a"],
        "b_wins": wins["b"],
        "ties": wins["tie"],
        "winner": winner,
        "rounds": rounds,
    }
    summary_path = verdict_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "compare verdict: A wins %d, B wins %d, ties %d -> %s wins"
        % (wins["a"], wins["b"], wins["tie"], winner)
    )
    print("summary: %s" % summary_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be at least 1")
    if args.stall < 1:
        raise SystemExit("--stall must be at least 1")
    if args.holdout and args.compare is None:
        raise SystemExit("--holdout requires --compare")
    if args.chapters and args.compare is None:
        raise SystemExit("--chapters requires --compare")
    if args.compare is None and args.rewriter_model is None:
        raise SystemExit("--rewriter-model is required unless --compare is given")

    workdir = args.workdir.resolve()
    if workdir.exists() and not workdir.is_dir():
        raise SystemExit("workdir is not a directory: %s" % workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for directory in (workdir / "prompts", workdir / "judge", workdir / "cache"):
        directory.mkdir(parents=True, exist_ok=True)

    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise SystemExit("base config not found: %s" % base_path)
    base_data = load_toml(base_path)
    base = load_base_config(base_data, args.pass_name, base_path)
    api = resolve_api(args, base_data)

    judge_template = args.judge_template.read_text(encoding="utf-8")
    for token in (SOURCE_TOKEN, A_TOKEN, B_TOKEN):
        if token not in judge_template:
            raise SystemExit(
                "judge template %s lacks %s" % (args.judge_template, token)
            )

    l2l_cmd = (
        [sys.executable, "-m", "l2l"] if args.l2l is None else shlex.split(args.l2l)
    )
    args.judge_extra = parse_params(args.judge_params, "judge-params")

    if args.compare is not None:
        chapters = compare_chapters(workdir, args)
        print(
            "workdir: %s; comparing over %d chapter(s); up to %d endpoint "
            "calls (2 translations + 2 judgments per chapter)"
            % (workdir, len(chapters), 4 * len(chapters))
        )
        compare_prompts(api, args, workdir, base, chapters, judge_template, l2l_cmd)
        return 0

    args.rewrite_extra = parse_params(args.rewrite_params, "rewrite-params")
    rewrite_template = args.rewrite_template.read_text(encoding="utf-8")
    for token in (PROMPT_TOKEN, HISTORY_TOKEN, CRITIQUES_TOKEN):
        if token not in rewrite_template:
            raise SystemExit(
                "rewrite template %s lacks %s" % (args.rewrite_template, token)
            )

    chapters = discover_chapters(workdir / "chapters")
    prompts_dir = workdir / "prompts"

    seed_path = prompts_dir / (SEED_VERSION + ".txt")
    if args.seed is not None and not seed_path.is_file():
        shutil.copyfile(args.seed.resolve(), seed_path)
    if not seed_path.is_file():
        raise SystemExit("missing %s (pass --seed PATH once to create it)" % seed_path)

    per_round = 1 + len(chapters) + 2 * len(chapters)
    print(
        "workdir: %s; chapters: %d; up to %d endpoint calls per round "
        "(1 rewrite + up to %d translations + %d judgments)"
        % (workdir, len(chapters), per_round, len(chapters), 2 * len(chapters))
    )

    final_version = run_rounds(
        api,
        args,
        workdir,
        base,
        chapters,
        judge_template,
        rewrite_template,
        l2l_cmd,
    )
    print(
        "final incumbent: %s (%s)"
        % (final_version, prompts_dir / (final_version + ".txt"))
    )
    if final_version == SEED_VERSION:
        print("no challenger was ever promoted; holdout check skipped")
        return 0
    holdout_check(
        api,
        args,
        workdir,
        base,
        final_version,
        judge_template,
        l2l_cmd,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

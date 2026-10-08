from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from l2l.errors import TranslationError, fail_config
from l2l.monads import Ok, Result, result_bind, result_map, result_sequence

from .locations import JUDGE_TEMPLATE_PATH, REWRITE_TEMPLATE_PATH


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
        default=JUDGE_TEMPLATE_PATH,
    )
    parser.add_argument(
        "--rewrite-template",
        type=str,
        default=REWRITE_TEMPLATE_PATH,
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

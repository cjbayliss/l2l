from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from l2l.effects import (
    ensure_directory,
    path_exists,
    read_text_file,
    write_text_file,
)
from l2l.errors import TranslationError, fail_config
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_result_bind,
    io_traverse,
    result_bind_io,
)

from .environment import OptimizationEnvironment
from .instructions import prepare_version, read_prompt
from .judging import ChapterJudging, judge_chapter, round_texts
from .locations import join_directory, text_stem
from .replies import hash_text
from .state import SEED_VERSION
from .translations import translate_chapter


@dataclass(frozen=True)
class CompareTally:
    a_wins: int = 0
    b_wins: int = 0
    tie: int = 0
    rounds: tuple[dict[str, Any], ...] = ()


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
            return io_result_bind(
                prepare_version(optimization_environment, SEED_VERSION, seed_text),
                prepare_final,
            )

        return io_result_bind(
            read_prompt(optimization_environment, SEED_VERSION), with_seed_text
        )

    def seeded(_: None) -> IO[Result[str, TranslationError]]:
        return io_result_bind(
            read_prompt(optimization_environment, final_version),
            with_final_text,
        )

    def judging(_: None) -> IO[Result[None, TranslationError]]:
        return io_bind(
            round_texts(optimization_environment, SEED_VERSION, final_version, holdout),
            with_texts,
        )

    def prepared(_: str) -> IO[Result[None, TranslationError]]:
        return io_result_bind(translated(None), judging)

    return io_result_bind(seeded(None), prepared)


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

        return io_result_bind(translated(None), judging)

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
        return io_result_bind(
            fold_io(
                optimization_environment.chapters,
                compare_step(optimization_environment, versions, verdict_directory),
                Ok(CompareTally()),
            ),
            summarize,
        )

    def with_second(_: str) -> IO[Result[None, TranslationError]]:
        return io_result_bind(
            prepare_version(optimization_environment, versions[1], prompts[1]),
            judged,
        )

    def with_first(_: None) -> IO[Result[None, TranslationError]]:
        return io_result_bind(
            prepare_version(optimization_environment, versions[0], prompts[0]),
            with_second,
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
        return io_result_bind(
            read_text_file(paths[1], "instruction"), with_right_text(left)
        )

    def with_existence(
        outcomes: tuple[bool, ...],
    ) -> IO[Result[None, TranslationError]]:
        missing = next(
            (
                path
                for path, present in zip(paths, outcomes, strict=True)
                if not present
            ),
            None,
        )
        if missing is not None:
            return io_result(fail_config("prompt file not found: %s" % missing))
        return io_result_bind(read_text_file(paths[0], "instruction"), with_left)

    return io_result_bind(io_traverse(paths, existing), with_existence)

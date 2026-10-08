from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from l2l.effects import read_text_file, write_text_file
from l2l.errors import TranslationError, describe
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_and_then,
    io_bind,
    io_map,
    io_result,
    io_result_bind,
    result_map,
)

from .endpoint import complete
from .environment import OptimizationEnvironment
from .locations import join_directory, output_path, text_stem
from .replies import (
    error_verdict,
    extract_object,
    feedback_lines,
    fill_judge,
    order_outcome,
    usable_verdict,
)

VERDICT_ATTEMPTS = 3


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

        return io_result_bind(
            side(candidate_text, incumbent_text, "candidate as A"),
            with_second,
        )

    return io_result_bind(
        side(incumbent_text, candidate_text, "incumbent as A"),
        with_first,
    )


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

from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any

from l2l.effects import append_text_file, read_text_file, write_text_file
from l2l.errors import TranslationError
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    fold_io,
    io_bind,
    io_map,
    io_result,
    io_result_bind,
    io_when_unit,
)

from .environment import OptimizationEnvironment
from .instructions import (
    candidate_prompt,
    prepare_version,
    previous_critiques,
    seen_digests,
)
from .judging import Tally, judge_step
from .locations import join_directory
from .state import SEED_VERSION, RoundInputs, RunState
from .translations import translate_versions


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


def append_jsonl(
    path: str, entry: dict[str, Any], description: str
) -> IO[Result[None, TranslationError]]:
    return append_text_file(
        path, json.dumps(entry, ensure_ascii=False) + "\n", description
    )


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
        return io_result_bind(
            fold_io(
                optimization_environment.chapters,
                judge_step(optimization_environment, state.round_no, state.incumbent),
                Ok(Tally()),
            ),
            with_tally,
        )

    def seeded(_: str) -> IO[Result[RunState, TranslationError]]:
        return io_result_bind(
            translate_versions(
                optimization_environment, state.incumbent, candidate_version
            ),
            judged,
        )

    return io_result_bind(
        prepare_version(
            optimization_environment, state.incumbent, inputs.current_prompt
        ),
        seeded,
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
            return io_result_bind(
                seen_digests(optimization_environment), with_seen(inputs)
            )

        return taken

    def with_candidate(
        current: str,
    ) -> Callable[[str], IO[Result[RunState, TranslationError]]]:
        def taken(candidate: str) -> IO[Result[RunState, TranslationError]]:
            return io_result_bind(
                prepare_version(optimization_environment, candidate_version, candidate),
                with_digest(current, candidate),
            )

        return taken

    def with_critiques(
        current: str,
    ) -> Callable[[str], IO[Result[RunState, TranslationError]]]:
        def taken(critiques: str) -> IO[Result[RunState, TranslationError]]:
            return io_result_bind(
                candidate_prompt(
                    optimization_environment, state, current, history_text, critiques
                ),
                with_candidate(current),
            )

        return taken

    def with_current(current: str) -> IO[Result[RunState, TranslationError]]:
        return io_bind(
            previous_critiques(optimization_environment, state.round_no),
            with_critiques(current),
        )

    return io_result_bind(
        read_text_file(
            os.path.join(prompts_directory, state.incumbent + ".txt"),
            "current instruction",
        ),
        with_current,
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

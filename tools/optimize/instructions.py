from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

from l2l.effects import (
    ensure_directory,
    entry_paths,
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
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_result_bind,
    result_bind_io,
    result_either,
    result_or_else,
)

from .configuration import render_config
from .endpoint import complete
from .environment import OptimizationEnvironment
from .locations import join_directory
from .replies import fill_rewrite, hash_text, strip_reply
from .state import RunState


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
        return io_result_bind(
            write_text_file(candidate_path, prompt, "candidate instruction"),
            lambda _: io_result(Ok(prompt)),
        )

    def asked(_: None) -> IO[Result[str, TranslationError]]:
        return io_result_bind(
            rewrite_prompt(
                optimization_environment,
                current,
                history_text,
                critiques,
                state.round_no,
            ),
            persisted,
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

            return io_result_bind(
                write_text_file(
                    os.path.join(generation_directory, version + ".toml"),
                    config_text,
                    "generated config",
                ),
                lambda _: io_result(Ok(digest)),
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

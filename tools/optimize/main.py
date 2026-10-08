from __future__ import annotations

import os
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, TextIO

from l2l.effects import (
    ProcessResult,
    Sleep,
    copy_file,
    ensure_directory,
    load_toml,
    path_exists,
    path_is_directory,
    read_text_file,
    run_process,
    user_config_path,
    write_stdout,
)
from l2l.errors import ConfigError, TranslationError, describe, fail_config
from l2l.monads import (
    IO,
    Err,
    Ok,
    Result,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_sequence,
    result_either,
)

from .chapters import compare_chapters, discover_chapters
from .compare import holdout_check, run_compare
from .configuration import BaseConfig, as_table, load_base_config
from .endpoint import curl_endpoint
from .environment import (
    Api,
    OpenEndpoint,
    OptimizationEnvironment,
    Reporter,
    RunCommand,
)
from .locations import REPO_ROOT, join_directory
from .options import Options, parse_arguments
from .replies import (
    A_TOKEN,
    B_TOKEN,
    CRITIQUES_TOKEN,
    HISTORY_TOKEN,
    PROMPT_TOKEN,
    SOURCE_TOKEN,
)
from .rounds import read_jsonl, resume_state, run_rounds
from .state import SEED_VERSION, RunState


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

from __future__ import annotations

import os

from l2l.effects import (
    ProcessResult,
    ensure_directory,
    path_exists,
    read_text_file,
    replace_file,
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
    io_result,
    result_bind_io,
)

from .environment import OptimizationEnvironment
from .locations import join_directory, output_path


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

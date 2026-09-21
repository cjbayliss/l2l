from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TextIO

from zh2en import __version__
from zh2en.config import load_setup, setup_report
from zh2en.console import Console, StatusLine
from zh2en.effects import (
    RunLog,
    cache_entry_paths,
    file_age,
    io_isatty,
    open_run_log,
    read_stdin,
    remove_file,
    resolve_cache_dir,
    time_sleep,
    write_stdout,
)
from zh2en.errors import TranslationError, describe, fail_config
from zh2en.http import urllib_open
from zh2en.monads import (
    IO,
    NOTHING,
    Err,
    Ok,
    Result,
    fold_io,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_when,
    maybe_either,
    result_or_else,
)
from zh2en.pipeline import run_pipeline
from zh2en.plans import plan_report
from zh2en.settings import Arguments, Context, Setup, build_settings


def parse_args(arguments: Sequence[str]) -> Arguments:
    parser = argparse.ArgumentParser(
        prog="zh2en",
        description="Translate Chinese text from stdin to English on stdout.",
    )
    parser.add_argument(
        "config",
        metavar="CONFIG",
        nargs="?",
        help="TOML config file defining [api] settings and [[pass]] passes "
        "(default: $TRANSLATE_CONFIG, then ./zh2en.toml, then "
        "~/.config/zh2en/config.toml)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="zh2en " + __version__,
    )
    parser.add_argument(
        "--base-url",
        help="API endpoint; overrides [api] base_url and TRANSLATE_BASE_URL",
    )
    parser.add_argument(
        "--api-key",
        help="API key; overrides [api] api_key and TRANSLATE_API_KEY",
    )
    parser.add_argument(
        "--model",
        help="default model; overrides [api] model and TRANSLATE_MODEL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="request timeout in seconds; overrides [api] timeout and "
        "TRANSLATE_TIMEOUT",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="request token budget; overrides [api] max_tokens and "
        "TRANSLATE_MAX_TOKENS",
    )
    parser.add_argument(
        "--cache-dir",
        help="translation cache directory (default: $XDG_CACHE_HOME/zh2en)",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    parser.add_argument(
        "--ensure-paragraphs",
        action="store_true",
        help="after each pass, check the output paragraph count against the "
        "source; re-run a mismatching pass with one call per paragraph",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) and LLM reasoning "
        "traces to stderr",
    )
    parser.add_argument(
        "--show-log-path",
        "-l",
        action="store_true",
        help="print the run log's path to stderr at startup",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="print the resolved configuration and exit without translating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the per-pass call plan and exit without calling the endpoint",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force streamed (--stream) or plain (--no-stream) responses; "
        "default follows api.params.stream",
    )
    parser.add_argument(
        "--cache-prune",
        metavar="DAYS",
        type=int,
        help="delete cache entries older than DAYS days and exit",
    )
    parsed = parser.parse_args(arguments)
    return Arguments(
        config=parsed.config,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        model=parsed.model,
        timeout=parsed.timeout,
        max_tokens=parsed.max_tokens,
        no_cache=parsed.no_cache,
        ensure_paragraphs=parsed.ensure_paragraphs,
        verbose=parsed.verbose,
        show_log_path=parsed.show_log_path,
        cache_dir=parsed.cache_dir,
        check_config=parsed.check_config,
        dry_run=parsed.dry_run,
        stream=parsed.stream,
        cache_prune=parsed.cache_prune,
    )


def parse_arguments(
    arguments: Sequence[str],
) -> IO[Result[Arguments, TranslationError]]:
    def thunk() -> Result[Arguments, TranslationError]:
        try:
            return Ok(parse_args(arguments))
        except SystemExit as exit_error:
            if exit_error.code == 0:
                raise
            return fail_config("invalid arguments; run zh2en --help")

    return IO(thunk)


def main(
    arguments: Sequence[str],
    environment: Mapping[str, str],
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    def after_parse(
        parsed_result: Result[Arguments, TranslationError],
    ) -> IO[int]:
        if isinstance(parsed_result, Err):
            def report_parse_failure(live: bool) -> IO[int]:
                console = Console(stderr, StatusLine(stderr, live))
                return io_map(
                    console.log(describe(parsed_result.error)), lambda _: 2
                )

            return io_bind(io_isatty(stderr), report_parse_failure)

        parsed = parsed_result.value
        if parsed.check_config:
            return check_config_program(parsed, environment, stdout, stderr, clock)

        if parsed.cache_prune is not None:
            return prune_program(parsed, environment, stderr, clock)

        return io_bind(
            read_stdin(stdin),
            lambda text: run_main_program(
                parsed, environment, text, stdout, stderr, clock
            ),
        )

    return io_bind(parse_arguments(arguments), after_parse)


def check_config_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    def with_console(live: bool) -> IO[int]:
        console = Console(stderr, StatusLine(stderr, live))

        def use_setup(setup_result: Result[Setup, TranslationError]) -> IO[int]:
            if isinstance(setup_result, Err):
                return io_map(console.log(describe(setup_result.error)), lambda _: 2)

            setup = setup_result.value
            report = setup_report(
                setup, parsed.ensure_paragraphs or setup.ensure_paragraphs
            )
            return io_map(write_stdout(stdout, report + "\n"), lambda _: 0)

        return io_bind(load_setup(parsed, environment), use_setup)

    return io_bind(io_isatty(stderr), with_console)


def prune_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    def with_console(live: bool) -> IO[int]:
        console = Console(stderr, StatusLine(stderr, live))
        days = parsed.cache_prune or 0
        if days <= 0:
            return io_map(
                console.log(
                    "zh2en: --cache-prune requires a positive number of days"
                ),
                lambda _: 2,
            )

        def with_cache_dir(cache_directory: str) -> IO[int]:
            def removed(count: int, path: str) -> IO[Result[int, TranslationError]]:
                def maybe_remove(age: float) -> IO[Result[int, TranslationError]]:
                    if age <= days * 86400.0:
                        return io_result(Ok(count))

                    return io_map(
                        remove_file(path),
                        lambda was_removed: Ok(count + (1 if was_removed else 0)),
                    )

                return io_bind(file_age(path, clock()), maybe_remove)

            def report(pruned: Result[int, TranslationError]) -> IO[int]:
                count = result_or_else(pruned, lambda: 0)
                noun = "entry" if count == 1 else "entries"
                return io_map(
                    console.log("zh2en: pruned %d cache %s" % (count, noun)),
                    lambda _: 0,
                )

            return io_bind(
                io_bind(
                    cache_entry_paths(cache_directory),
                    lambda paths: fold_io(paths, removed, Ok(0)),
                ),
                report,
            )

        return io_bind(resolve_cache_dir(environment, parsed.cache_dir), with_cache_dir)

    return io_bind(io_isatty(stderr), with_console)


def run_main_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    text: str,
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    if not text.strip():
        return io_pure(0)

    def with_console(live: bool) -> IO[int]:
        console = Console(stderr, StatusLine(stderr, live))
        started = clock()

        def use_setup(setup_result: Result[Setup, TranslationError]) -> IO[int]:
            if isinstance(setup_result, Err):
                return io_map(
                    console.log(describe(setup_result.error)), lambda _: 2
                )

            setup = setup_result.value

            if parsed.dry_run:
                planning_ctx = Context(
                    config=setup.config,
                    settings=build_settings(),
                    use_cache=False,
                    cache_directory="",
                    verbose=parsed.verbose,
                    ensure_paragraphs=parsed.ensure_paragraphs
                    or setup.ensure_paragraphs,
                    console=console,
                    open_http=urllib_open,
                    log=RunLog(NOTHING, clock),
                    clock=clock,
                    sleep=time_sleep,
                    stream=parsed.stream,
                )
                return io_map(
                    write_stdout(
                        stdout,
                        plan_report(planning_ctx, setup.passes, text) + "\n",
                    ),
                    lambda _: 0,
                )

            def with_cache_dir(cache_directory: str) -> IO[int]:
                def with_log(log: RunLog) -> IO[int]:
                    announced: IO[None] = maybe_either(
                        log.path,
                        lambda path: io_when(
                            parsed.show_log_path,
                            console.log("zh2en: log: %s" % path),
                        ),
                        lambda: io_pure(None),
                    )

                    def run_with_log(_: None) -> IO[int]:
                        return run_pipeline(
                            Context(
                                config=setup.config,
                                settings=build_settings(),
                                use_cache=not parsed.no_cache,
                                cache_directory=cache_directory,
                                verbose=parsed.verbose,
                                ensure_paragraphs=parsed.ensure_paragraphs
                                or setup.ensure_paragraphs,
                                console=console,
                                open_http=urllib_open,
                                log=log,
                                clock=clock,
                                sleep=time_sleep,
                                stream=parsed.stream,
                            ),
                            setup.passes,
                            text,
                            started,
                            stdout,
                        )

                    return io_bind(announced, run_with_log)

                return io_bind(open_run_log(cache_directory, clock), with_log)

            return io_bind(
                resolve_cache_dir(environment, parsed.cache_dir), with_cache_dir
            )

        return io_bind(load_setup(parsed, environment), use_setup)

    return io_bind(io_isatty(stderr), with_console)


def cli() -> None:
    try:
        code = main(
            sys.argv[1:],
            dict(os.environ),
            sys.stdin,
            sys.stdout,
            sys.stderr,
            time.time,
        ).run()
    except KeyboardInterrupt:
        print("zh2en: interrupted", file=sys.stderr)
        code = 130
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 141

    sys.exit(code)


if __name__ == "__main__":
    cli()

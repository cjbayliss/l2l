from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, TextIO

from l2l import __version__
from l2l.config import load_setup
from l2l.console import Console, StatusLine, toggle_verbose
from l2l.context import Context, OpenHTTP
from l2l.effects import (
    RunLog,
    Sleep,
    cache_entry_paths,
    close_run_log,
    file_age,
    now,
    open_run_log,
    prune_old_logs,
    read_stdin,
    remove_file,
    resolve_cache_directory,
    stream_isatty,
    time_sleep,
    write_stdout,
    write_text_file,
)
from l2l.errors import TranslationError, describe, fail_config
from l2l.glossary import (
    entries_to_json_text,
    extraction_user_message,
    merge_glossary_entries,
    parse_model_glossary_reply,
)
from l2l.http import chat, curl_open
from l2l.keys import start_tab_listener
from l2l.monads import (
    IO,
    NOTHING,
    Err,
    Ok,
    Reference,
    Result,
    fold_io,
    io_and_then,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_when_unit,
    maybe_either,
)
from l2l.pipeline import run_pipeline
from l2l.report import plan_report, setup_report, setup_warnings
from l2l.settings import Arguments, Setup, build_settings
from l2l.text import Translated, Usage


@dataclass(frozen=True)
class EarlyExitText(BaseException):
    text: str


class HelpExitAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        raise EarlyExitText(parser.format_help())


class VersionExitAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        raise EarlyExitText("l2l " + __version__)


def parse_args(arguments: Sequence[str]) -> Arguments:
    parser = argparse.ArgumentParser(
        prog="l2l",
        description="Translate text between any pair of languages on "
        "stdin/stdout using any OpenAI-compatible chat completions endpoint.",
        add_help=False,
        allow_abbrev=False,
    )
    meta_group = parser.add_argument_group("meta")
    configuration_group = parser.add_argument_group("configuration and API")
    translation_group = parser.add_argument_group("translation behaviour")
    cache_group = parser.add_argument_group("cache and logs")
    modes_group = parser.add_argument_group("modes")

    meta_group.add_argument(
        "-h",
        "--help",
        action=HelpExitAction,
        nargs=0,
        help="show this help message and exit",
    )
    meta_group.add_argument(
        "-v",
        "--version",
        action=VersionExitAction,
        nargs=0,
        help="show the version and exit",
    )
    meta_group.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) and LLM reasoning "
        "traces to stderr; press Tab while running to toggle them "
        "interactively",
    )

    configuration_group.add_argument(
        "-c",
        "--config",
        metavar="FILE",
        help="TOML config file defining [api] settings and [[pass]] passes "
        "(default: $TRANSLATE_CONFIG, then ./l2l.toml, then "
        "~/.config/l2l/config.toml)",
    )
    configuration_group.add_argument(
        "-u",
        "--base-url",
        metavar="URL",
        help="API endpoint; overrides [api] base_url and TRANSLATE_BASE_URL",
    )
    configuration_group.add_argument(
        "-k",
        "--api-key",
        metavar="KEY",
        help="API key; overrides [api] api_key and TRANSLATE_API_KEY",
    )
    configuration_group.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        help="default model; overrides [api] model and TRANSLATE_MODEL",
    )
    configuration_group.add_argument(
        "-t",
        "--timeout",
        metavar="SECONDS",
        type=float,
        help="request timeout in seconds; overrides [api] timeout and "
        "TRANSLATE_TIMEOUT",
    )
    configuration_group.add_argument(
        "-T",
        "--maximum-tokens",
        metavar="N",
        type=int,
        dest="maximum_tokens",
        help="request token budget; overrides [api] max_tokens and "
        "TRANSLATE_MAX_TOKENS",
    )
    configuration_group.add_argument(
        "-s",
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force streamed (--stream) or plain (--no-stream) responses; "
        "default follows api.params.stream",
    )

    translation_group.add_argument(
        "-g",
        "--glossary",
        action="append",
        metavar="FILE",
        default=[],
        help="JSON glossary file with mandatory renderings; repeatable; "
        "later files override earlier entries; generated and updated "
        "entries are written to the first file",
    )
    translation_group.add_argument(
        "-l",
        "--target-language",
        metavar="LANG",
        default=None,
        help="target language; required unless target_language is set in a config file",
    )
    translation_group.add_argument(
        "-P",
        "--ensure-paragraphs",
        action="store_true",
        help="after each pass, check the output paragraph count against the "
        "source; re-run a mismatching pass with one call per paragraph",
    )
    translation_group.add_argument(
        "-b",
        "--best-effort",
        action="store_true",
        help="keep the last reply and continue when a unit still fails "
        "validation after all repair attempts (default: fail the run "
        "with exit code 1)",
    )

    cache_group.add_argument(
        "--cache-directory",
        metavar="DIR",
        dest="cache_directory",
        help="translation cache directory (default: $XDG_CACHE_HOME/l2l)",
    )
    cache_group.add_argument(
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    cache_group.add_argument(
        "--log-retention-days",
        metavar="DAYS",
        type=int,
        dest="log_retention_days",
        default=30,
        help="delete run logs older than DAYS days at startup (0 keeps "
        "every log; default: %(default)s)",
    )
    cache_group.add_argument(
        "-p",
        "--print-log-path",
        action="store_true",
        dest="print_log_path",
        help="print the run log's path to stderr at startup",
    )

    modes_group.add_argument(
        "--check-config",
        action="store_true",
        help="print the resolved configuration and exit without translating",
    )
    modes_group.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="print the per-pass call plan and exit without calling the endpoint",
    )
    modes_group.add_argument(
        "--extract-glossary",
        metavar="FILE",
        dest="extract_glossary",
        help="extract a starting glossary from the stdin text with one "
        "model call, write it to FILE, and exit",
    )
    modes_group.add_argument(
        "--prune-cache",
        metavar="DAYS",
        type=int,
        dest="prune_cache",
        help="delete cache entries older than DAYS days and exit",
    )

    parsed = parser.parse_args(arguments)
    return Arguments(
        config=parsed.config,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        model=parsed.model,
        timeout=parsed.timeout,
        maximum_tokens=parsed.maximum_tokens,
        no_cache=parsed.no_cache,
        ensure_paragraphs=parsed.ensure_paragraphs,
        verbose=parsed.verbose,
        print_log_path=parsed.print_log_path,
        cache_directory=parsed.cache_directory,
        check_config=parsed.check_config,
        dry_run=parsed.dry_run,
        stream=parsed.stream,
        prune_cache=parsed.prune_cache,
        log_retention_days=parsed.log_retention_days,
        best_effort=parsed.best_effort,
        glossary=tuple(parsed.glossary),
        extract_glossary=parsed.extract_glossary,
        target_language=parsed.target_language,
    )


type ParsedArguments = Result[Arguments | EarlyExitText, TranslationError]


def active_mode_flags(parsed: Arguments) -> tuple[str, ...]:
    return tuple(
        flag_name
        for flag_name, is_active in (
            ("--check-config", parsed.check_config),
            ("--prune-cache", parsed.prune_cache is not None),
            ("--extract-glossary", parsed.extract_glossary is not None),
            ("--dry-run", parsed.dry_run),
        )
        if is_active
    )


def reject_exclusive_modes(parsed: Arguments) -> ParsedArguments:
    active = active_mode_flags(parsed)
    if len(active) > 1:
        return fail_config(
            "%s cannot be combined with %s" % (" ".join(active[:-1]), active[-1])
        )
    return Ok(parsed)


def parse_arguments(
    arguments: Sequence[str],
) -> IO[ParsedArguments]:
    def thunk() -> ParsedArguments:
        try:
            return reject_exclusive_modes(parse_args(arguments))
        except EarlyExitText as early_exit:
            return Ok(early_exit)
        except SystemExit:
            return fail_config("invalid arguments; run l2l --help")

    return IO(thunk)


def curl_open_http(environment: Mapping[str, str]) -> OpenHTTP:
    return lambda request, timeout: curl_open(request, timeout, environment)


def main(
    arguments: Sequence[str],
    environment: Mapping[str, str],
    stdin: TextIO,
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
    open_http: OpenHTTP | None = None,
    sleep: Sleep = time_sleep,
) -> IO[int]:
    opener = open_http if open_http is not None else curl_open_http(environment)

    def after_parse(
        parsed_result: ParsedArguments,
    ) -> IO[int]:
        if isinstance(parsed_result, Err):

            def report_parse_failure(live: bool) -> IO[int]:
                console = Console(stderr, StatusLine(stderr, live))
                return io_map(console.log(describe(parsed_result.error)), lambda _: 2)

            return io_bind(stream_isatty(stderr), report_parse_failure)

        parsed = parsed_result.value
        if isinstance(parsed, EarlyExitText):
            return io_map(write_stdout(stdout, parsed.text), lambda _: 0)

        if parsed.check_config:
            return check_config_program(parsed, environment, stdout, stderr, clock)

        if parsed.prune_cache is not None:
            return prune_cache_program(parsed, environment, stderr, clock)

        if parsed.extract_glossary is not None:
            glossary_output_path = parsed.extract_glossary
            return io_bind(
                read_stdin(stdin),
                lambda text: extract_glossary_program(
                    parsed,
                    glossary_output_path,
                    environment,
                    text,
                    stderr,
                    clock,
                    opener,
                    sleep,
                ),
            )

        return io_bind(
            read_stdin(stdin),
            lambda text: run_main_program(
                parsed, environment, text, stdout, stderr, clock, opener, sleep
            ),
        )

    return io_bind(parse_arguments(arguments), after_parse)


def with_console_io(
    stderr: TextIO,
    bound: Callable[[Console], IO[int]],
    verbose: bool = False,
) -> IO[int]:
    def with_live(live: bool) -> IO[int]:
        return bound(
            Console(stderr, StatusLine(stderr, live), verbose=Reference(verbose))
        )

    return io_bind(stream_isatty(stderr), with_live)


def build_context(
    parsed: Arguments,
    setup: Setup,
    console: Console,
    clock: Callable[[], float],
    open_http: OpenHTTP,
    sleep: Sleep,
    cache_directory: str = "",
    use_cache: bool = False,
    log: RunLog | None = None,
) -> Context:
    return Context(
        config=setup.config,
        settings=build_settings(),
        use_cache=use_cache,
        cache_directory=cache_directory,
        console=console,
        open_http=open_http,
        log=log if log is not None else RunLog(NOTHING, clock),
        clock=clock,
        sleep=sleep,
        stream=parsed.stream,
        best_effort=parsed.best_effort,
        glossary_entries=setup.glossary_entries,
        glossary_update_path=parsed.glossary[0] if parsed.glossary else None,
        target_language=setup.target_language,
    )


def dry_run_program(
    parsed: Arguments,
    setup: Setup,
    console: Console,
    clock: Callable[[], float],
    open_http: OpenHTTP,
    sleep: Sleep,
    text: str,
    stdout: TextIO,
) -> IO[int]:
    planning_run_context = build_context(
        parsed, setup, console, clock, open_http, sleep
    )
    return io_map(
        write_stdout(
            stdout, plan_report(planning_run_context, setup.passes, text) + "\n"
        ),
        lambda _: 0,
    )


def log_setup_warnings(console: Console, setup: Setup) -> IO[None]:
    return reduce(
        io_and_then,
        (console.log("warning: %s" % warning) for warning in setup_warnings(setup)),
        io_pure(None),
    )


def check_config_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    def with_console(console: Console) -> IO[int]:
        def use_setup(setup_result: Result[Setup, TranslationError]) -> IO[int]:
            if isinstance(setup_result, Err):
                return io_map(console.log(describe(setup_result.error)), lambda _: 2)

            setup = setup_result.value
            report = setup_report(setup, setup.ensure_paragraphs)
            warnings = "".join(
                "\nwarning: %s" % warning for warning in setup_warnings(setup)
            )
            return io_map(write_stdout(stdout, report + warnings + "\n"), lambda _: 0)

        return io_bind(load_setup(parsed, environment), use_setup)

    return with_console_io(stderr, with_console)


def prune_cache_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    def with_console(console: Console) -> IO[int]:
        days = parsed.prune_cache
        if days is None or days <= 0:
            return io_map(
                console.log("l2l: --prune-cache requires a positive number of days"),
                lambda _: 2,
            )

        def with_cache_directory(cache_directory: str) -> IO[int]:
            def removed(count: int, path: str) -> IO[Result[int, TranslationError]]:
                def maybe_remove(
                    aged: Result[float, TranslationError],
                ) -> IO[Result[int, TranslationError]]:
                    match aged:
                        case Err(error):
                            return io_result(Err(error))
                        case Ok(age):
                            if age <= days * 86400.0:
                                return io_result(Ok(count))

                            return io_map(remove_file(path), removal_outcome)

                def removal_outcome(
                    removal: Result[bool, TranslationError],
                ) -> Result[int, TranslationError]:
                    match removal:
                        case Err(error):
                            return Err(error)
                        case Ok(was_removed):
                            return Ok(count + (1 if was_removed else 0))

                return io_bind(
                    now(clock),
                    lambda now_value: io_bind(file_age(path, now_value), maybe_remove),
                )

            def report(pruned: Result[int, TranslationError]) -> IO[int]:
                match pruned:
                    case Err(error):
                        return io_map(console.log(describe(error)), lambda _: 2)
                    case Ok(count):
                        noun = "entry" if count == 1 else "entries"
                        return io_map(
                            console.log("l2l: pruned %d cache %s" % (count, noun)),
                            lambda _: 0,
                        )

            def with_entry_paths(
                scanned: Result[tuple[str, ...], TranslationError],
            ) -> IO[Result[int, TranslationError]]:
                match scanned:
                    case Err(error):
                        return io_result(Err(error))
                    case Ok(paths):
                        return fold_io(paths, removed, Ok(0))

            return io_bind(
                io_bind(cache_entry_paths(cache_directory), with_entry_paths),
                report,
            )

        return io_bind(
            resolve_cache_directory(environment, parsed.cache_directory),
            with_cache_directory,
        )

    return with_console_io(stderr, with_console)


def run_main_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    text: str,
    stdout: TextIO,
    stderr: TextIO,
    clock: Callable[[], float],
    open_http: OpenHTTP,
    sleep: Sleep,
) -> IO[int]:
    if not text.strip():
        return io_pure(0)

    def with_console(console: Console) -> IO[int]:
        def use_setup(setup_result: Result[Setup, TranslationError]) -> IO[int]:
            if isinstance(setup_result, Err):
                return io_map(console.log(describe(setup_result.error)), lambda _: 2)

            setup = setup_result.value
            warnings = log_setup_warnings(console, setup)
            if parsed.dry_run:
                return io_and_then(
                    warnings,
                    dry_run_program(
                        parsed, setup, console, clock, open_http, sleep, text, stdout
                    ),
                )

            def with_started(started: float) -> IO[int]:
                def with_cache_directory(cache_directory: str) -> IO[int]:
                    def with_log(log: RunLog) -> IO[int]:
                        announced: IO[None] = maybe_either(
                            log.path,
                            lambda path: io_when_unit(
                                parsed.print_log_path,
                                console.log("l2l: log: %s" % path),
                            ),
                            lambda: io_pure(None),
                        )

                        def with_retention(_: None) -> IO[int]:
                            retention = io_when_unit(
                                parsed.log_retention_days > 0,
                                io_map(
                                    prune_old_logs(
                                        cache_directory,
                                        parsed.log_retention_days,
                                        clock,
                                    ),
                                    lambda _: None,
                                ),
                            )
                            return io_and_then(retention, run_with_log())

                        def run_with_log() -> IO[int]:
                            pipeline = run_pipeline(
                                build_context(
                                    parsed,
                                    setup,
                                    console,
                                    clock,
                                    open_http,
                                    sleep,
                                    cache_directory=cache_directory,
                                    use_cache=not parsed.no_cache,
                                    log=log,
                                ),
                                setup.passes,
                                text,
                                started,
                                stdout,
                            )

                            def execute() -> int:
                                def toggle() -> None:
                                    toggle_verbose(console).run()

                                stop_listener = start_tab_listener(
                                    console, toggle
                                ).run()
                                try:
                                    return pipeline.run()
                                finally:
                                    stop_listener.run()
                                    close_run_log(log).run()

                            return IO(execute)

                        return io_bind(announced, with_retention)

                    return io_bind(open_run_log(cache_directory, clock), with_log)

                return io_bind(
                    resolve_cache_directory(environment, parsed.cache_directory),
                    with_cache_directory,
                )

            return io_and_then(warnings, io_bind(now(clock), with_started))

        return io_bind(load_setup(parsed, environment), use_setup)

    return with_console_io(stderr, with_console, verbose=parsed.verbose)


def extract_glossary_program(
    parsed: Arguments,
    output_path: str,
    environment: Mapping[str, str],
    text: str,
    stderr: TextIO,
    clock: Callable[[], float],
    open_http: OpenHTTP,
    sleep: Sleep,
) -> IO[int]:
    def with_console(console: Console) -> IO[int]:
        def use_setup(setup_result: Result[Setup, TranslationError]) -> IO[int]:
            if isinstance(setup_result, Err):
                return io_map(console.log(describe(setup_result.error)), lambda _: 2)

            if not text.strip():
                return io_map(
                    console.log("l2l: --extract-glossary requires text on stdin"),
                    lambda _: 2,
                )

            setup = setup_result.value
            run_context = build_context(parsed, setup, console, clock, open_http, sleep)

            def handle_reply(
                reply_result: Result[Translated, TranslationError],
            ) -> IO[int]:
                if isinstance(reply_result, Err):
                    return io_map(
                        console.log(describe(reply_result.error)), lambda _: 1
                    )

                reply_entries = parse_model_glossary_reply(
                    reply_result.value.text, output_path
                )
                if isinstance(reply_entries, Err):
                    return io_map(
                        console.log(describe(reply_entries.error)), lambda _: 1
                    )

                merged = merge_glossary_entries(
                    (setup.glossary_entries, reply_entries.value)
                )

                def report(written: Result[None, TranslationError]) -> IO[int]:
                    if isinstance(written, Err):
                        return io_map(console.log(describe(written.error)), lambda _: 1)

                    return io_map(
                        console.log(
                            "l2l: wrote %d glossary entries to %s"
                            % (len(merged), output_path)
                        ),
                        lambda _: 0,
                    )

                return io_bind(
                    write_text_file(
                        output_path, entries_to_json_text(merged), "glossary file"
                    ),
                    report,
                )

            def extract() -> IO[Result[Translated, TranslationError]]:
                def toggle() -> None:
                    toggle_verbose(console).run()

                def execute() -> Result[Translated, TranslationError]:
                    stop_listener = start_tab_listener(console, toggle).run()
                    try:
                        return chat(
                            run_context,
                            run_context.settings.glossary_extraction_instruction,
                            extraction_user_message(
                                setup.target_language,
                                setup.glossary_entries,
                                text,
                            ),
                            run_context.config.model,
                            {},
                            Usage(),
                        ).run()
                    finally:
                        stop_listener.run()

                return IO(execute)

            return io_and_then(
                log_setup_warnings(console, setup), io_bind(extract(), handle_reply)
            )

        return io_bind(load_setup(parsed, environment), use_setup)

    return with_console_io(stderr, with_console, verbose=parsed.verbose)


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
        print("l2l: interrupted", file=sys.stderr)
        code = 130
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        code = 141

    sys.exit(code)


if __name__ == "__main__":
    cli()

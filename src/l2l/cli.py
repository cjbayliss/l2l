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
    LOG_SECONDS_PER_DAY,
    RunLog,
    Sleep,
    cache_entry_paths,
    close_run_log,
    count_removed_file,
    now,
    open_run_log,
    prune_old_logs,
    read_stdin,
    remove_file_if_older,
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
    glossary_response_format,
    integrate_glossary_reply,
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


def add_meta_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("meta")
    group.add_argument(
        "-h",
        "--help",
        action=HelpExitAction,
        nargs=0,
        help="show this help message and exit",
    )
    group.add_argument(
        "-v",
        "--version",
        action=VersionExitAction,
        nargs=0,
        help="show the version and exit",
    )
    group.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="diagnostics (chunks, cache hits, timings) and LLM reasoning "
        "traces to stderr; press Tab while running to toggle them "
        "interactively",
    )


def add_configuration_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("configuration and API")
    group.add_argument(
        "-c",
        "--config",
        metavar="FILE",
        help="TOML config file defining [api] settings and [[pass]] passes "
        "(default: $TRANSLATE_CONFIG, then ./l2l.toml, then "
        "~/.config/l2l/config.toml)",
    )
    group.add_argument(
        "-u",
        "--base-url",
        metavar="URL",
        help="API endpoint; overrides [api] base_url and TRANSLATE_BASE_URL",
    )
    group.add_argument(
        "-k",
        "--api-key",
        metavar="KEY",
        help="API key; overrides [api] api_key and TRANSLATE_API_KEY",
    )
    group.add_argument(
        "-m",
        "--model",
        metavar="MODEL",
        help="default model; overrides [api] model and TRANSLATE_MODEL",
    )
    group.add_argument(
        "-t",
        "--timeout",
        metavar="SECONDS",
        type=float,
        help="request timeout in seconds; overrides [api] timeout and "
        "TRANSLATE_TIMEOUT",
    )
    group.add_argument(
        "-T",
        "--maximum-tokens",
        metavar="N",
        type=int,
        dest="maximum_tokens",
        help="request token budget; overrides [api] max_tokens and "
        "TRANSLATE_MAX_TOKENS",
    )
    group.add_argument(
        "-s",
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force streamed (--stream) or plain (--no-stream) responses; "
        "default follows api.params.stream",
    )
    group.add_argument(
        "--no-structured-output",
        action="store_true",
        dest="no_structured_output",
        help="request glossary extraction without the JSON-schema response "
        "format, for endpoints that do not support structured output",
    )


def add_translation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("translation behaviour")
    group.add_argument(
        "-g",
        "--glossary",
        action="append",
        metavar="FILE",
        default=[],
        help="JSON glossary file with mandatory renderings; repeatable; "
        "later files override earlier entries; generated and updated "
        "entries are written to the first file",
    )
    group.add_argument(
        "-l",
        "--target-language",
        metavar="LANG",
        default=None,
        help="target language; required unless target_language is set in a config file",
    )
    group.add_argument(
        "-P",
        "--ensure-paragraphs",
        action="store_true",
        help="after each pass, check the output paragraph count against the "
        "source; re-run a mismatching pass with one call per paragraph",
    )
    group.add_argument(
        "-b",
        "--best-effort",
        action="store_true",
        help="keep the last reply and continue when a unit still fails "
        "validation after all repair attempts (default: fail the run "
        "with exit code 1)",
    )


def add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("cache and logs")
    group.add_argument(
        "--cache-directory",
        metavar="DIR",
        dest="cache_directory",
        help="translation cache directory (default: $XDG_CACHE_HOME/l2l)",
    )
    group.add_argument(
        "--no-cache", action="store_true", help="bypass the translation cache"
    )
    group.add_argument(
        "--log-retention-days",
        metavar="DAYS",
        type=int,
        dest="log_retention_days",
        default=30,
        help="delete run logs older than DAYS days at startup (0 keeps "
        "every log; default: %(default)s)",
    )
    group.add_argument(
        "-p",
        "--print-log-path",
        action="store_true",
        dest="print_log_path",
        help="print the run log's path to stderr at startup",
    )


def add_mode_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("modes")
    group.add_argument(
        "--check-config",
        action="store_true",
        help="print the resolved configuration and exit without translating",
    )
    group.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="print the per-pass call plan and exit without calling the endpoint",
    )
    group.add_argument(
        "--extract-glossary",
        metavar="FILE",
        dest="extract_glossary",
        help="extract a starting glossary from the stdin text with one "
        "model call, write it to FILE, and exit",
    )
    group.add_argument(
        "--prune-cache",
        metavar="DAYS",
        type=int,
        dest="prune_cache",
        help="delete cache entries older than DAYS days and exit",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="l2l",
        description="Translate text between any pair of languages on "
        "stdin/stdout using any OpenAI-compatible chat completions endpoint.",
        add_help=False,
        allow_abbrev=False,
    )
    add_meta_arguments(parser)
    add_configuration_arguments(parser)
    add_translation_arguments(parser)
    add_cache_arguments(parser)
    add_mode_arguments(parser)
    return parser


def arguments_from_namespace(parsed: argparse.Namespace) -> Arguments:
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
        no_structured_output=parsed.no_structured_output,
    )


def parse_args(arguments: Sequence[str]) -> Arguments:
    return arguments_from_namespace(build_parser().parse_args(arguments))


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


def report_message_and_exit(console: Console, message: str, exit_code: int) -> IO[int]:
    return io_map(console.log(message), lambda _: exit_code)


def report_error_and_exit(
    console: Console, error: TranslationError, exit_code: int
) -> IO[int]:
    return report_message_and_exit(console, describe(error), exit_code)


def with_loaded_setup(
    parsed: Arguments,
    environment: Mapping[str, str],
    console: Console,
    program: Callable[[Setup], IO[int]],
    resolve_pipeline_passes: bool = True,
) -> IO[int]:
    def use_setup(setup_result: Result[Setup, TranslationError]) -> IO[int]:
        match setup_result:
            case Err(error):
                return report_error_and_exit(console, error, 2)

            case Ok(setup):
                return program(setup)

    return io_bind(load_setup(parsed, environment, resolve_pipeline_passes), use_setup)


def with_cache_directory(
    environment: Mapping[str, str],
    requested: str | None,
    program: Callable[[str], IO[int]],
) -> IO[int]:
    return io_bind(resolve_cache_directory(environment, requested), program)


def with_tab_listener[T](
    console: Console, action: IO[T], finalize: IO[None] | None = None
) -> IO[T]:
    def execute() -> T:
        def toggle() -> None:
            toggle_verbose(console).run()

        stop_listener = start_tab_listener(console, toggle).run()
        try:
            return action.run()
        finally:
            stop_listener.run()
            if finalize is not None:
                finalize.run()

    return IO(execute)


def announce_run_log(parsed: Arguments, console: Console, log: RunLog) -> IO[None]:
    return maybe_either(
        log.path,
        lambda path: io_when_unit(
            parsed.print_log_path,
            console.log("l2l: log: %s" % path),
        ),
        lambda: io_pure(None),
    )


def prune_old_run_logs(
    parsed: Arguments, cache_directory: str, clock: Callable[[], float]
) -> IO[None]:
    return io_when_unit(
        parsed.log_retention_days > 0,
        io_map(
            prune_old_logs(cache_directory, parsed.log_retention_days, clock),
            lambda _: None,
        ),
    )


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
            return with_console_io(
                stderr,
                lambda console: report_error_and_exit(console, parsed_result.error, 2),
            )

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
        no_structured_output=parsed.no_structured_output,
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
    def print_setup(setup: Setup) -> IO[int]:
        report = setup_report(setup, setup.ensure_paragraphs)
        warnings = "".join(
            "\nwarning: %s" % warning for warning in setup_warnings(setup)
        )
        return io_map(write_stdout(stdout, report + warnings + "\n"), lambda _: 0)

    def with_console(console: Console) -> IO[int]:
        return with_loaded_setup(parsed, environment, console, print_setup)

    return with_console_io(stderr, with_console)


def prune_cache_entry(
    clock: Callable[[], float],
    days: int,
    count: int,
    path: str,
) -> IO[Result[int, TranslationError]]:
    return io_bind(
        now(clock),
        lambda now_value: io_map(
            remove_file_if_older(path, now_value, days * LOG_SECONDS_PER_DAY),
            lambda removal: count_removed_file(count, removal),
        ),
    )


def report_cache_prune(
    console: Console, pruned: Result[int, TranslationError]
) -> IO[int]:
    match pruned:
        case Err(error):
            return report_error_and_exit(console, error, 2)

        case Ok(count):
            noun = "entry" if count == 1 else "entries"
            return report_message_and_exit(
                console, "l2l: pruned %d cache %s" % (count, noun), 0
            )


def prune_cache_program(
    parsed: Arguments,
    environment: Mapping[str, str],
    stderr: TextIO,
    clock: Callable[[], float],
) -> IO[int]:
    days = parsed.prune_cache

    def with_console(console: Console) -> IO[int]:
        if days is None or days <= 0:
            return report_message_and_exit(
                console,
                "l2l: --prune-cache requires a positive number of days",
                2,
            )

        def prune(directory: str) -> IO[int]:
            def with_entry_paths(
                scanned: Result[tuple[str, ...], TranslationError],
            ) -> IO[Result[int, TranslationError]]:
                match scanned:
                    case Err(error):
                        return io_result(Err(error))

                    case Ok(paths):
                        return fold_io(
                            paths,
                            lambda count, path: prune_cache_entry(
                                clock, days, count, path
                            ),
                            Ok(0),
                        )

            return io_bind(
                io_bind(cache_entry_paths(directory), with_entry_paths),
                lambda pruned: report_cache_prune(console, pruned),
            )

        return with_cache_directory(environment, parsed.cache_directory, prune)

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

    def run_with_setup(setup: Setup, console: Console) -> IO[int]:
        warnings = log_setup_warnings(console, setup)
        if parsed.dry_run:
            return io_and_then(
                warnings,
                dry_run_program(
                    parsed, setup, console, clock, open_http, sleep, text, stdout
                ),
            )

        def with_started(started: float) -> IO[int]:
            def run_in_cache_directory(cache_directory: str) -> IO[int]:
                def run_logged_pipeline(log: RunLog) -> IO[int]:
                    return io_and_then(
                        announce_run_log(parsed, console, log),
                        io_and_then(
                            prune_old_run_logs(parsed, cache_directory, clock),
                            with_tab_listener(
                                console,
                                run_pipeline(
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
                                ),
                                close_run_log(log),
                            ),
                        ),
                    )

                return io_bind(
                    open_run_log(cache_directory, clock), run_logged_pipeline
                )

            return with_cache_directory(
                environment, parsed.cache_directory, run_in_cache_directory
            )

        return io_and_then(warnings, io_bind(now(clock), with_started))

    def with_console(console: Console) -> IO[int]:
        return with_loaded_setup(
            parsed, environment, console, lambda setup: run_with_setup(setup, console)
        )

    return with_console_io(stderr, with_console, verbose=parsed.verbose)


def write_extracted_glossary(
    console: Console,
    output_path: str,
    setup: Setup,
    reply_result: Result[Translated, TranslationError],
) -> IO[int]:
    match reply_result:
        case Err(error):
            return report_error_and_exit(console, error, 1)

        case Ok(reply):
            reply_entries = parse_model_glossary_reply(reply.text, output_path)
            match reply_entries:
                case Err(error):
                    return report_error_and_exit(console, error, 1)

                case Ok(entries):
                    merged = integrate_glossary_reply(setup.glossary_entries, entries)

                    def report(written: Result[None, TranslationError]) -> IO[int]:
                        match written:
                            case Err(error):
                                return report_error_and_exit(console, error, 1)

                            case Ok(_):
                                return report_message_and_exit(
                                    console,
                                    "l2l: wrote %d glossary entries to %s"
                                    % (len(merged), output_path),
                                    0,
                                )

                    return io_bind(
                        write_text_file(
                            output_path,
                            entries_to_json_text(merged),
                            "glossary file",
                        ),
                        report,
                    )


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
    def extract_reply(
        run_context: Context, setup: Setup
    ) -> IO[Result[Translated, TranslationError]]:
        return with_tab_listener(
            run_context.console,
            chat(
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
                response_format=glossary_response_format(
                    run_context.no_structured_output
                ),
            ),
        )

    def run_with_setup(console: Console, setup: Setup) -> IO[int]:
        if not text.strip():
            return report_message_and_exit(
                console,
                "l2l: --extract-glossary requires text on stdin",
                2,
            )

        run_context = build_context(parsed, setup, console, clock, open_http, sleep)
        return io_and_then(
            log_setup_warnings(console, setup),
            io_bind(
                extract_reply(run_context, setup),
                lambda reply: write_extracted_glossary(
                    console, output_path, setup, reply
                ),
            ),
        )

    def with_console(console: Console) -> IO[int]:
        return with_loaded_setup(
            parsed,
            environment,
            console,
            lambda setup: run_with_setup(console, setup),
            resolve_pipeline_passes=False,
        )

    return with_console_io(stderr, with_console)


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

from __future__ import annotations

import os
from collections.abc import Callable

from l2l.effects import entry_paths, path_exists, path_is_directory
from l2l.errors import TranslationError, fail_config
from l2l.monads import IO, Err, Ok, Result, io_bind, io_map, io_result

from .locations import join_directory, text_stem
from .options import Options


def select_chapters(
    paths: tuple[str, ...], selector: str, chapters_directory: str
) -> Result[tuple[str, ...], TranslationError]:
    by_stem = {text_stem(path): path for path in paths}

    def chosen() -> Result[tuple[str, ...], TranslationError]:
        selected: dict[str, str] = {}
        for raw in selector.split(","):
            name = raw.strip()
            if not name or name in selected:
                continue
            if name not in by_stem:
                return fail_config(
                    "unknown chapter %r in %s" % (name, chapters_directory)
                )
            selected = {**selected, name: by_stem[name]}
        if not selected:
            return fail_config(
                "--chapters named none of: %s" % ", ".join(sorted(by_stem))
            )
        return Ok(tuple(selected.values()))

    return chosen()


def discover_chapters(
    chapters_directory: str,
) -> IO[Result[tuple[str, ...], TranslationError]]:
    holdout = os.path.join(chapters_directory, "holdout.txt")

    def with_holdout(
        chapters: tuple[str, ...],
    ) -> Callable[[bool], Result[tuple[str, ...], TranslationError]]:
        def taken(present: bool) -> Result[tuple[str, ...], TranslationError]:
            if not present:
                return fail_config("missing %s" % holdout)
            return Ok(chapters)

        return taken

    def listed(paths: tuple[str, ...]) -> Result[tuple[str, ...], TranslationError]:
        chapters = tuple(
            sorted(
                path
                for path in paths
                if os.path.basename(path).endswith(".txt")
                and os.path.basename(path) != "holdout.txt"
            )
        )
        if len(chapters) < 2:
            return fail_config("need at least two chapters in %s" % chapters_directory)
        return Ok(chapters)

    def with_directory(
        is_directory: bool,
    ) -> IO[Result[tuple[str, ...], TranslationError]]:
        if not is_directory:
            return io_result(
                fail_config(
                    "missing %s (expected chapter .txt files plus holdout.txt)"
                    % chapters_directory
                )
            )

        def with_entries(
            paths: tuple[str, ...],
        ) -> IO[Result[tuple[str, ...], TranslationError]]:
            outcome = listed(paths)
            if isinstance(outcome, Err):
                return io_result(outcome)

            def with_presence(
                present: bool,
            ) -> Result[tuple[str, ...], TranslationError]:
                return with_holdout(outcome.value)(present)

            return io_map(path_exists(holdout), with_presence)

        return io_bind(entry_paths(chapters_directory), with_entries)

    return io_bind(path_is_directory(chapters_directory), with_directory)


def compare_chapters(
    working_directory: str, options: Options
) -> IO[Result[tuple[str, ...], TranslationError]]:
    chapters_directory = join_directory(working_directory, "chapters")
    holdout = os.path.join(chapters_directory, "holdout.txt")

    def with_holdout(present: bool) -> Result[tuple[str, ...], TranslationError]:
        if not present:
            return fail_config("missing %s" % holdout)
        return Ok((holdout,))

    def discovered(
        paths: Result[tuple[str, ...], TranslationError],
    ) -> IO[Result[tuple[str, ...], TranslationError]]:
        if isinstance(paths, Err) or options.chapters is None:
            return io_result(paths)
        return io_result(
            select_chapters(paths.value, options.chapters, chapters_directory)
        )

    if options.holdout:
        return io_map(path_exists(holdout), with_holdout)
    return io_bind(discover_chapters(chapters_directory), discovered)

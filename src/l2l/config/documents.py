from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

from l2l.config.models import DocumentResult
from l2l.effects import (
    current_working_directory,
    load_toml,
    path_exists,
    user_config_path,
)
from l2l.errors import TranslationError, fail_config
from l2l.monads import (
    IO,
    Ok,
    Result,
    io_bind,
    io_pair,
    io_pure,
    io_result,
    io_result_bind,
    result_map,
)


def validate_document(
    path: str, document: dict[str, Any]
) -> Result[None, TranslationError]:
    if not document:
        return Ok(None)

    unknown = sorted(
        set(document)
        - {
            "api",
            "options",
            "pass",
            "glossary",
            "target_language",
            "source_language",
            "prompts",
            "style",
            "output",
        }
    )
    if unknown:
        return fail_config(
            "%s: unknown top-level key(s): %s (expected [api], [options], "
            "glossary, target_language, source_language, [prompts], [style], "
            "[output], "
            "[[pass]])" % (path, ", ".join(unknown))
        )

    glossary = document.get("glossary")
    if glossary is not None and (
        not isinstance(glossary, list)
        or not all(isinstance(item, str) and item.strip() for item in glossary)
    ):
        return fail_config("%s: glossary must be a list of file paths" % path)

    return Ok(None)


def read_document(
    path: str | None, description: str, exists: bool
) -> IO[Result[dict[str, Any], TranslationError]]:
    if not exists:
        return io_result(Ok({}))

    return io_result_bind(
        load_toml(path or "", description),
        lambda document: io_result(
            result_map(validate_document(path or "", document), lambda _: document)
        ),
    )


def resolve_config_path(
    requested: str | None,
    environment: Mapping[str, str],
    path_exists: Callable[[str], bool] = os.path.exists,
    current_directory: IO[str] | None = None,
) -> IO[str | None]:
    if requested:
        return io_pure(requested)

    configured = environment.get("TRANSLATE_CONFIG", "").strip()
    if configured:
        return io_pure(configured)

    def pick(current_working_directory_value: str, user_path: str) -> str | None:
        local = os.path.join(current_working_directory_value, "l2l.toml")
        if path_exists(local):
            return local

        return user_path if path_exists(user_path) else None

    return io_bind(
        current_directory
        if current_directory is not None
        else current_working_directory(),
        lambda current_working_directory_value: io_bind(
            user_config_path(environment),
            lambda user_path: IO(
                lambda: pick(current_working_directory_value, user_path)
            ),
        ),
    )


def read_setup_documents(
    user_path: str, selected_path: str | None
) -> IO[tuple[DocumentResult, DocumentResult]]:
    def with_user_exists(
        user_exists: bool,
    ) -> IO[tuple[DocumentResult, DocumentResult]]:
        return io_pair(
            read_document(user_path, "user config", user_exists),
            read_document(selected_path, "config file", bool(selected_path)),
        )

    return io_bind(path_exists(user_path), with_user_exists)

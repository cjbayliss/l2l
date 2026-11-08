from __future__ import annotations

from typing import TYPE_CHECKING

from l2l.effects import (
    load_toml,
    path_exists,
    user_config_path,
)
from l2l.errors import TranslationError, fail_config
from l2l.json_values import JsonObject, is_non_empty_string_array
from l2l.monads import (
    IO,
    Ok,
    Result,
    io_bind,
    io_map,
    io_pure,
    io_result,
    io_result_bind,
    result_map,
)
lazy from l2l.config.models import DocumentResult
lazy from l2l.settings import Unset

if TYPE_CHECKING:
    from collections.abc import Mapping


def validate_document(
    path: str, document: JsonObject
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
            "filter",
            "target_language",
            "source_language",
            "prompts",
            "style",
            "output",
        }
    )
    if unknown:
        return fail_config(
            "{}: unknown top-level key(s): {} (expected [api], [options], "
            "glossary, target_language, source_language, [filter], [prompts], "
            "[style], "
            "[output], "
            "[pass.<name>])".format(path, ", ".join(unknown))
        )

    glossary = document.get("glossary")
    if glossary is not None and not is_non_empty_string_array(glossary):
        return fail_config(f"{path}: glossary must be a list of file paths")

    return Ok(None)


def read_document(
    path: str | None, description: str, exists: bool
) -> IO[Result[JsonObject, TranslationError]]:
    if not exists:
        return io_result(Ok({}))

    return io_result_bind(
        load_toml(path or "", description),
        lambda document: io_result(
            result_map(validate_document(path or "", document), lambda _: document)
        ),
    )


def resolve_config_path(
    requested: str | Unset,
    environment: Mapping[str, str],
) -> IO[str]:
    match requested:
        case str() as path if path.strip():
            return io_pure(path)

        case _:
            return io_map(user_config_path(environment), lambda path: path)


def read_setup_document(path: str, requested: bool) -> IO[DocumentResult]:
    # An absent default config reads as an empty document (the run then fails
    # on missing passes); a file named with -c must be readable.
    def with_exists(exists: bool) -> IO[DocumentResult]:
        return read_document(path, "config file", exists or requested)

    return io_bind(path_exists(path), with_exists)

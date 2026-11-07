from __future__ import annotations

import os
from dataclasses import replace
from typing import TYPE_CHECKING

from l2l.effects import load_json
from l2l.errors import TranslationError, fail_config
from l2l.glossary import (
    merge_glossary_entries,
    parse_glossary_document,
)
from l2l.json_values import JsonObject, JsonValue, is_non_empty_string_array
from l2l.monads import (
    IO,
    Ok,
    Result,
    io_map,
    io_traverse,
    result_bind,
    result_bind_io,
    result_map,
    result_sequence,
)
lazy from l2l.settings import Setup

if TYPE_CHECKING:
    from l2l.config.models import DocumentResult, Glossaries


def document_glossary_paths(
    path: str | None, document: JsonObject
) -> Result[tuple[str, ...], TranslationError]:
    value = document.get("glossary")
    if value is None:
        return Ok(())

    if not path:
        return fail_config(
            "glossary configured in a config document without a file path"
        )

    if not is_non_empty_string_array(value):
        return fail_config("%s: glossary must be a list of file paths" % path)

    base_directory = os.path.dirname(os.path.abspath(path))
    return Ok(tuple(os.path.join(base_directory, item) for item in value))


def glossary_paths(
    user_path: str,
    user_document: JsonObject,
    selected_path: str | None,
    selected_document: JsonObject,
    command_line_paths: tuple[str, ...],
) -> Result[tuple[str, ...], TranslationError]:
    return result_map(
        result_sequence(
            document_glossary_paths(path, document)
            for path, document in (
                (user_path, user_document),
                (selected_path, selected_document),
            )
            if document
        ),
        lambda groups: (
            tuple(path for group in groups for path in group)
            + tuple(command_line_paths)
        ),
    )


def load_glossaries(
    paths: tuple[str, ...],
) -> IO[Result[Glossaries, TranslationError]]:
    def parsed(
        loaded_result: Result[tuple[JsonValue, ...], TranslationError],
    ) -> Result[Glossaries, TranslationError]:
        return result_bind(
            loaded_result,
            lambda loaded: result_map(
                result_sequence(
                    parse_glossary_document(item_path, document)
                    for item_path, document in zip(paths, loaded, strict=True)
                ),
                lambda groups: (paths, merge_glossary_entries(groups)),
            ),
        )

    return io_map(
        io_traverse(paths, lambda item_path: load_json(item_path, "glossary file")),
        parsed,
    )


def resolve_glossaries(
    user_path: str,
    user_document_result: DocumentResult,
    selected_path: str | None,
    selected_document_result: DocumentResult,
    command_line_paths: tuple[str, ...],
) -> IO[Result[Glossaries, TranslationError]]:
    def with_documents(
        documents: tuple[JsonObject, JsonObject],
    ) -> IO[Result[Glossaries, TranslationError]]:
        return result_bind_io(
            glossary_paths(
                user_path,
                documents[0],
                selected_path,
                documents[1],
                command_line_paths,
            ),
            load_glossaries,
        )

    return result_bind_io(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        with_documents,
    )


def setup_with_glossary(setup: Setup, glossaries: Glossaries) -> Setup:
    paths, entries = glossaries
    return replace(
        setup,
        glossary_paths=paths,
        glossary_entries=entries,
    )


def require_glossary_for_update_passes(
    command_line_paths: tuple[str, ...], setup: Setup
) -> Result[Setup, TranslationError]:
    if command_line_paths or not any(
        pass_definition.mode == "update_glossary" for pass_definition in setup.passes
    ):
        return Ok(setup)

    return fail_config(
        "an update_glossary pass requires at least one --glossary file "
        "on the command line"
    )

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING

from l2l.effects import load_json, path_exists
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
    io_bind,
    io_map,
    io_result,
    io_traverse,
    result_bind,
    result_bind_io,
    result_map,
    result_sequence,
)
lazy from l2l.settings import Setup

if TYPE_CHECKING:
    from l2l.config.models import DocumentResult, Glossaries
    from l2l.glossary import GlossaryEntry


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
        return fail_config(f"{path}: glossary must be a list of file paths")

    base_directory = os.path.dirname(os.path.abspath(path))
    return Ok(tuple(os.path.join(base_directory, item) for item in value))


def glossary_paths(
    path: str,
    document: JsonObject,
    command_line_paths: tuple[str, ...],
) -> Result[tuple[str, ...], TranslationError]:
    def with_command_line_paths(document_paths: tuple[str, ...]) -> tuple[str, ...]:
        return document_paths + tuple(command_line_paths)

    return result_map(
        document_glossary_paths(path, document),
        with_command_line_paths,
    )


def load_glossary_document(
    path: str, default_path: str | None
) -> IO[Result[JsonValue, TranslationError]]:
    if path != default_path:
        return load_json(path, "glossary file")

    def after_existence(exists: bool) -> IO[Result[JsonValue, TranslationError]]:
        # The default state file is managed by l2l itself and may not exist
        # yet; treat it as an empty glossary instead of failing the run.
        return load_json(path, "glossary file") if exists else io_result(Ok([]))

    return io_bind(path_exists(path), after_existence)


def parsed_glossaries(
    paths: tuple[str, ...], loaded: tuple[JsonValue, ...]
) -> Result[Glossaries, TranslationError]:
    def glossaries_with_merged_entries(
        groups: tuple[tuple[GlossaryEntry, ...], ...],
    ) -> Glossaries:
        return (paths, merge_glossary_entries(groups))

    return result_map(
        result_sequence(
            parse_glossary_document(item_path, document)
            for item_path, document in zip(paths, loaded, strict=True)
        ),
        glossaries_with_merged_entries,
    )


def load_glossaries(
    paths: tuple[str, ...], default_path: str | None = None
) -> IO[Result[Glossaries, TranslationError]]:
    def parsed(
        loaded_result: Result[tuple[JsonValue, ...], TranslationError],
    ) -> Result[Glossaries, TranslationError]:
        return result_bind(loaded_result, partial(parsed_glossaries, paths))

    return io_map(
        io_traverse(paths, partial(load_glossary_document, default_path=default_path)),
        parsed,
    )


def resolve_glossaries(
    path: str,
    document_result: DocumentResult,
    command_line_paths: tuple[str, ...],
    default_path: str | None = None,
) -> IO[Result[Glossaries, TranslationError]]:
    def document_glossary(
        document: JsonObject,
    ) -> Result[tuple[str, ...], TranslationError]:
        return glossary_paths(path, document, command_line_paths)

    def glossaries_for_paths(
        paths: tuple[str, ...],
    ) -> IO[Result[Glossaries, TranslationError]]:
        return load_glossaries(paths, default_path)

    return result_bind_io(
        result_bind(document_result, document_glossary),
        glossaries_for_paths,
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

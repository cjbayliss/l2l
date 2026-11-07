from __future__ import annotations

from collections.abc import Callable
from functools import reduce

from l2l.errors import TranslationError, fail_config
from l2l.json_values import JsonObject, is_non_empty_string
from l2l.languages import normalized_source_language
from l2l.monads import Ok, Result, result_bind, result_map
from l2l.settings import Unset
lazy from l2l.config.models import DocumentResult

LanguageResult = Result[str | None, TranslationError]
DocumentLanguageParser = Callable[[str | None, JsonObject], LanguageResult]


def document_language(
    path: str | None, document: JsonObject, key: str
) -> LanguageResult:
    language = document.get(key)
    if language is None:
        return Ok(None)

    if not is_non_empty_string(language):
        return fail_config(
            "{}: {} must be a non-empty string".format(path or "config", key)
        )

    return Ok(language.strip())


def document_target_language(path: str | None, document: JsonObject) -> LanguageResult:
    return document_language(path, document, "target_language")


def command_line_language(value: str | Unset, option: str) -> LanguageResult:
    match value:
        case Unset():
            return Ok(None)

        case str() as provided if not provided.strip():
            return fail_config(f"{option} must be a non-empty string")

        case str() as provided:
            return Ok(provided.strip())


def command_line_target_language(target_language: str | Unset) -> LanguageResult:
    return command_line_language(target_language, "--target-language")


def document_source_language(path: str | None, document: JsonObject) -> LanguageResult:
    return document_language(path, document, "source_language")


def command_line_source_language(source_language: str | Unset) -> LanguageResult:
    return command_line_language(source_language, "--source-language")


def require_target_language(
    target_language: str | None,
) -> Result[str, TranslationError]:
    if target_language is None:
        return fail_config(
            "target language is required; pass --target-language or set "
            "target_language in a config file"
        )

    return Ok(target_language)


def layered_language(
    command_line_result: LanguageResult,
    user_path: str,
    user_document_result: DocumentResult,
    selected_path: str | None,
    selected_document_result: DocumentResult,
    parse_document_language: DocumentLanguageParser,
) -> LanguageResult:
    def configured(
        documents: tuple[JsonObject, JsonObject],
    ) -> LanguageResult:
        layers = tuple(
            (path, document)
            for path, document in (
                (selected_path, documents[1]),
                (user_path, documents[0]),
            )
            if document
        )
        return reduce(
            lambda found, layer: result_bind(
                found,
                lambda value: (
                    Ok(value)
                    if value is not None
                    else parse_document_language(layer[0], layer[1])
                ),
            ),
            layers,
            command_line_result,
        )

    return result_bind(
        result_bind(
            user_document_result,
            lambda user_document: result_map(
                selected_document_result,
                lambda selected_document: (user_document, selected_document),
            ),
        ),
        configured,
    )


def resolve_target_language(
    command_line_value: str | Unset,
    user_path: str,
    user_document_result: DocumentResult,
    selected_path: str | None,
    selected_document_result: DocumentResult,
) -> Result[str, TranslationError]:
    return result_bind(
        layered_language(
            command_line_target_language(command_line_value),
            user_path,
            user_document_result,
            selected_path,
            selected_document_result,
            document_target_language,
        ),
        require_target_language,
    )


def resolve_source_language(
    command_line_value: str | Unset,
    user_path: str,
    user_document_result: DocumentResult,
    selected_path: str | None,
    selected_document_result: DocumentResult,
) -> Result[str, TranslationError]:
    return result_map(
        layered_language(
            command_line_source_language(command_line_value),
            user_path,
            user_document_result,
            selected_path,
            selected_document_result,
            document_source_language,
        ),
        lambda value: normalized_source_language(value or "auto"),
    )

from __future__ import annotations

from collections.abc import Callable

from l2l.errors import TranslationError, fail_config
from l2l.json_values import JsonObject, is_non_empty_string
from l2l.languages import normalized_source_language
from l2l.monads import Ok, Result, result_bind, result_map
lazy from l2l.config.models import DocumentResult

LanguageResult = Result[str | None, TranslationError]
DocumentLanguageParser = Callable[[str, JsonObject], LanguageResult]


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


def document_source_language(path: str | None, document: JsonObject) -> LanguageResult:
    return document_language(path, document, "source_language")


def require_target_language(
    target_language: str | None,
) -> Result[str, TranslationError]:
    if target_language is None:
        return fail_config(
            "target language is required; set target_language in a config file"
        )

    return Ok(target_language)


def document_language_result(
    path: str,
    document_result: DocumentResult,
    parse_document_language: DocumentLanguageParser,
) -> LanguageResult:
    def parse_bound_document(document: JsonObject) -> LanguageResult:
        return parse_document_language(path, document)

    return result_bind(document_result, parse_bound_document)


def resolve_target_language(
    path: str,
    document_result: DocumentResult,
) -> Result[str, TranslationError]:
    return result_bind(
        document_language_result(path, document_result, document_target_language),
        require_target_language,
    )


def normalized_configured_source_language(value: str | None) -> str:
    return normalized_source_language(value or "auto")


def resolve_source_language(
    path: str,
    document_result: DocumentResult,
) -> Result[str, TranslationError]:
    return result_map(
        document_language_result(path, document_result, document_source_language),
        normalized_configured_source_language,
    )

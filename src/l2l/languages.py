from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import unicodedataplus
from langcodes import Language
from langcodes.tag_parser import LanguageTagError

SCRIPT_VARIANT_COMPONENTS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "Hanb": ("Hani", "Bopo"),
        "Hrkt": ("Hira", "Kana"),
        "Jamo": ("Hang",),
        "Jpan": ("Hani", "Hira", "Kana"),
        "Kore": ("Hang", "Hani"),
        "Hans": ("Hani",),
        "Hant": ("Hani",),
    }
)

UNICODE_SCRIPT_NAMES: Mapping[str, str] = MappingProxyType(
    {
        str(aliases[0]): str(script_name).replace("_", " ")
        for script_name, aliases in unicodedataplus.property_value_aliases[
            "script"
        ].items()
        if aliases
    }
)


def normalized_language_name(language: str) -> str:
    return " ".join(
        "".join(
            character if character.isalnum() else " "
            for character in language.casefold()
        ).split()
    )


NON_LANGUAGE_CODES = frozenset({"mul", "und", "zxx"})

AUTO_SOURCE_LANGUAGE = "auto"
MIXED_SOURCE_LANGUAGE = "mixed"
SOURCE_LANGUAGE_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "auto": AUTO_SOURCE_LANGUAGE,
        "guess": AUTO_SOURCE_LANGUAGE,
        "detect": AUTO_SOURCE_LANGUAGE,
        "und": AUTO_SOURCE_LANGUAGE,
        "mixed": MIXED_SOURCE_LANGUAGE,
        "mul": MIXED_SOURCE_LANGUAGE,
    }
)


def normalized_source_language(value: str) -> str:
    stripped = value.strip()
    alias = SOURCE_LANGUAGE_ALIASES.get(stripped.casefold())
    return alias if alias is not None else stripped


def language_from_name_or_tag(language_name_or_tag: str) -> Language | None:
    try:
        parsed_language = Language.get(language_name_or_tag)
    except LanguageTagError, LookupError:
        parsed_language = None

    if parsed_language is not None and parsed_language.is_valid():
        return parsed_language

    try:
        return Language.find(language_name_or_tag)
    except LanguageTagError, LookupError, ValueError:
        return None


def expected_language_scripts(language_name_or_tag: str) -> tuple[str, ...]:
    language = language_from_name_or_tag(language_name_or_tag)
    if language is None or language.to_tag().split("-")[0] in NON_LANGUAGE_CODES:
        return ()

    script_code = language.maximize().script
    if not script_code or script_code == "Zzzz":
        return ()

    return SCRIPT_VARIANT_COMPONENTS.get(script_code, (script_code,))


def expected_target_scripts(target_language: str) -> tuple[str, ...]:
    return expected_language_scripts(target_language)


def language_base_and_region(
    language_name_or_tag: str,
) -> tuple[str, str] | None:
    language = language_from_name_or_tag(language_name_or_tag)
    if language is None or language.language is None:
        return None

    territory = language.territory
    return (str(language.language), str(territory) if territory else "")


def expected_source_scripts(source_language: str) -> tuple[str, ...]:
    normalized = normalized_source_language(source_language)
    if normalized in (AUTO_SOURCE_LANGUAGE, MIXED_SOURCE_LANGUAGE):
        return ()

    return expected_language_scripts(normalized)


def source_language_description(source_language: str) -> str:
    normalized = normalized_source_language(source_language)
    match normalized:
        case "auto":
            return "the source language (identify it from the text)"

        case "mixed":
            return "the source languages (identify the language of each passage)"

        case explicit:
            return explicit


def script_description(script_code: str) -> str:
    return UNICODE_SCRIPT_NAMES.get(script_code, script_code)


def scripts_description(target_scripts: tuple[str, ...]) -> str:
    if not target_scripts:
        return "unknown"

    return ", ".join(map(script_description, target_scripts))


def instruction_uses_target_language(instruction: str, target_language: str) -> bool:
    if "{target_language}" in instruction:
        return True

    normalized_instruction = normalized_language_name(instruction)
    words = frozenset(normalized_instruction.split())
    return any(
        word in words for word in normalized_language_name(target_language).split()
    )

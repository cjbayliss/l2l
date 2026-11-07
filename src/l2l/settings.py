from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal
lazy from collections.abc import Mapping

from l2l.glossary import (
    GlossaryEntry,
    entries_to_json_text,
)
from l2l.languages import normalized_language_name, normalized_source_language
from l2l.prompts import (
    ASCII_FIX_INSTRUCTION_TEXT,
    GLOSSARY_EXTRACTION_INSTRUCTION_TEXT,
    OutputPolicy,
    OutputPolicyPatch,
    PromptRole,
    PromptSectionName,
    PromptSectionOrigin,
    PromptSectionOverride,
    StylePolicy,
    StylePolicyPatch,
)
from l2l.text import CACHE_SALT_VERSION
lazy from l2l.json_values import JsonValue

type ParagraphTriState = bool | int


@dataclass(frozen=True, slots=True)
class Unset:
    pass


UNSET: Final = Unset()


def value_if_set[T](value: T | Unset) -> T | None:
    return None if isinstance(value, Unset) else value


@dataclass(frozen=True, slots=True)
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    maximum_tokens: int
    parameters: Mapping[str, JsonValue]


type PassMode = Literal["analysis", "chunk", "paragraph", "update_glossary"]
PASS_MODES: Final[tuple[PassMode, ...]] = (
    "analysis",
    "chunk",
    "paragraph",
    "update_glossary",
)


def parse_pass_mode(value: object) -> PassMode | None:
    if isinstance(value, str) and value in PASS_MODES:
        return value

    return None


@dataclass(frozen=True, slots=True)
class PassDefinition:
    name: str
    instruction: str
    mode: PassMode
    parameters: Mapping[str, JsonValue]
    model: str | None
    ascii: bool | None
    ensure_paragraphs: ParagraphTriState | None = None
    retranslate_untranslated: bool | None = None
    target_language: str | None = None
    include_types: tuple[str, ...] = ()
    style: StylePolicy = field(default_factory=StylePolicy)
    output: OutputPolicy = field(default_factory=OutputPolicy)
    unit_paragraph_tolerance: int | None = None
    pass_paragraph_tolerance: int | None = None
    prompt_role: PromptRole | None = None
    prompt_sections: tuple[PromptSectionName, ...] = ()
    prompt_omitted_sections: tuple[PromptSectionName, ...] = ()
    prompt_section_origins: Mapping[PromptSectionName, PromptSectionOrigin] = (
        MappingProxyType({})
    )


@dataclass(frozen=True, slots=True)
class PassDraft:
    name: str
    mode: PassMode
    instruction: str | None
    parameters: Mapping[str, JsonValue]
    model: str | Unset = UNSET
    ascii: bool | Unset = UNSET
    ensure_paragraphs: ParagraphTriState | Unset = UNSET
    retranslate_untranslated: bool | Unset = UNSET
    target_language: str | Unset = UNSET
    include_types: tuple[str, ...] = ()
    prompts: Mapping[PromptSectionName, PromptSectionOverride] = MappingProxyType({})
    style_patch: StylePolicyPatch = field(default_factory=StylePolicyPatch)
    output_patch: OutputPolicyPatch = field(default_factory=OutputPolicyPatch)
    ensure_paragraphs_explicit: ParagraphTriState | Unset = UNSET


@dataclass(frozen=True, slots=True)
class Settings:
    chunk_budget_tokens: int
    analysis_reserve_tokens: int
    ascii_fix_attempts: int
    ascii_collapse_floor: int
    sentence_boundary_characters: str
    ascii_fix_instruction: str
    ascii_character_map: Mapping[str, str]
    unit_fix_attempts: int
    unit_output_max_ratio: float
    retry_attempts: int
    retry_base_delay: float
    retry_cap: float
    provider_block_attempts: int
    glossary_extraction_instruction: str


@dataclass(frozen=True, kw_only=True, slots=True)
class OptionDefaults:
    ascii: bool = False
    ensure_paragraphs: ParagraphTriState = False
    retranslate_untranslated: bool = False
    ensure_paragraphs_explicit: bool = False


@dataclass(frozen=True, kw_only=True, slots=True)
class Arguments:
    config: str | Unset = UNSET
    base_url: str | Unset = UNSET
    api_key: str | Unset = UNSET
    model: str | Unset = UNSET
    timeout: float | Unset = UNSET
    maximum_tokens: int | Unset = UNSET
    no_cache: bool = False
    ensure_paragraphs: bool = False
    verbose: bool = False
    print_log_path: bool = False
    cache_directory: str | Unset = UNSET
    check_config: bool = False
    dry_run: bool = False
    stream: bool | Unset = UNSET
    prune_cache: int | Unset = UNSET
    log_retention_days: int = 30
    best_effort: bool = False
    glossary: tuple[str, ...] = ()
    extract_glossary: str | Unset = UNSET
    target_language: str | Unset = UNSET
    source_language: str | Unset = UNSET
    no_structured_output: bool = False


@dataclass(frozen=True, slots=True)
class Setup:
    config: Config
    passes: tuple[PassDefinition, ...]
    ensure_paragraphs: ParagraphTriState
    target_language: str
    source_language: str
    config_warnings: tuple[str, ...] = ()
    glossary_paths: tuple[str, ...] = ()
    glossary_entries: tuple[GlossaryEntry, ...] = ()


def build_settings() -> Settings:
    return Settings(
        chunk_budget_tokens=3500,
        analysis_reserve_tokens=128,
        ascii_fix_attempts=3,
        ascii_collapse_floor=4,
        unit_fix_attempts=2,
        unit_output_max_ratio=6.0,
        retry_attempts=2,
        retry_base_delay=1.0,
        retry_cap=30.0,
        provider_block_attempts=8,
        glossary_extraction_instruction=GLOSSARY_EXTRACTION_INSTRUCTION_TEXT,
        sentence_boundary_characters="。！？…⋯!?；;\n؟؛।॥։።",
        ascii_fix_instruction=ASCII_FIX_INSTRUCTION_TEXT,
        ascii_character_map=MappingProxyType(
            {
                "\u00a0": " ",
                "\u2007": " ",
                "\u2009": " ",
                "\u202f": " ",
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": ",",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
                "\u2032": "'",
                "\u2033": '"',
                "\u2039": "'",
                "\u203a": "'",
                "\u00ab": '"',
                "\u00bb": '"',
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
                "\u2026": "...",
                "\u2025": "..",
                "\u2022": "*",
                "\u2023": "*",
                "\u2024": "*",
                "\u2219": "*",
                "\u00b7": " ",
                "\u2190": "<-",
                "\u2192": "->",
                "\u2194": "<->",
                "\u2264": "<=",
                "\u2265": ">=",
                "\u2260": "!=",
                "\u3002": ".",
                "\u3001": ",",
                "\u300c": '"',
                "\u300d": '"',
                "\u300e": '"',
                "\u300f": '"',
                "\u3008": "<",
                "\u3009": ">",
                "\u300a": '"',
                "\u300b": '"',
                "\u3010": "[",
                "\u3011": "]",
            }
        ),
    )


API_SETTING_KEYS: Final = (
    "base_url",
    "api_key",
    "model",
    "timeout",
    "max_tokens",
    "params",
)
PASS_KEYS: Final = (
    "name",
    "instruction",
    "instruction_file",
    "mode",
    "ascii",
    "ensure_paragraphs",
    "retranslate_untranslated",
    "model",
    "params",
    "target_language",
    "include_types",
    "prompts",
    "style",
)


DEFAULT_TIMEOUT: Final = 120.0
DEFAULT_MAXIMUM_TOKENS: Final = 100000


@dataclass(frozen=True, slots=True)
class PartialApiSettings:
    base_url: str | Unset = UNSET
    api_key: str | Unset = UNSET
    model: str | Unset = UNSET
    timeout: float | Unset = UNSET
    maximum_tokens: int | Unset = UNSET
    parameters: Mapping[str, JsonValue] | Unset = UNSET

    def merge(self, extra: PartialApiSettings) -> PartialApiSettings:
        def higher_layer_wins[L](lower: L | Unset, higher: L | Unset) -> L | Unset:
            match higher:
                case Unset():
                    return lower

                case _:
                    return higher

        match (self.parameters, extra.parameters):
            case (Unset(), Unset()):
                parameters: Mapping[str, JsonValue] | Unset = UNSET

            case _:
                parameters = MappingProxyType(
                    {
                        **(value_if_set(self.parameters) or {}),
                        **(value_if_set(extra.parameters) or {}),
                    }
                )

        return PartialApiSettings(
            base_url=higher_layer_wins(self.base_url, extra.base_url),
            api_key=higher_layer_wins(self.api_key, extra.api_key),
            model=higher_layer_wins(self.model, extra.model),
            timeout=higher_layer_wins(self.timeout, extra.timeout),
            maximum_tokens=higher_layer_wins(self.maximum_tokens, extra.maximum_tokens),
            parameters=parameters,
        )


DEFAULT_API_SETTINGS: Final = PartialApiSettings(
    base_url="",
    api_key="",
    model="",
    timeout=DEFAULT_TIMEOUT,
    maximum_tokens=DEFAULT_MAXIMUM_TOKENS,
    parameters=MappingProxyType({}),
)


def resolve_call_settings(
    config: Config, pass_definition: PassDefinition
) -> tuple[str, Mapping[str, JsonValue]]:
    return (pass_definition.model or config.model), MappingProxyType(
        {**config.parameters, **pass_definition.parameters}
    )


def paragraph_tolerance(value: bool | int | None) -> int | None:
    if value is True:
        return 0

    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value

    return None


def salt(*parts: str) -> str:
    return "\x00".join(parts)


def effective_target_language(
    global_target_language: str, pass_target_language: str | None
) -> str:
    return (
        pass_target_language
        if pass_target_language is not None
        else global_target_language
    )


def glossary_salt_component(
    pass_definition: PassDefinition,
    glossary: tuple[GlossaryEntry, ...],
) -> str:
    match pass_definition.mode:
        case "update_glossary":
            return ""

        case _:
            return entries_to_json_text(glossary)


def output_policy_salt_component(pass_definition: PassDefinition) -> str:
    if pass_definition.prompt_role is None:
        return ""

    policy_values = (
        ("unit_tolerance", pass_definition.unit_paragraph_tolerance),
        ("pass_tolerance", pass_definition.pass_paragraph_tolerance),
        ("maximum_characters", pass_definition.output.maximum_characters),
    )
    return salt(
        *(
            "%s=%s" % (label, "none" if value is None else value)
            for label, value in policy_values
        )
    )


def pass_salt(
    pass_definition: PassDefinition,
    glossary: tuple[GlossaryEntry, ...] = (),
    retry_attempt: int = 0,
    effective_language: str = "",
    source_language: str = "",
) -> str:
    base = salt(
        CACHE_SALT_VERSION,
        pass_definition.name,
        pass_definition.instruction,
        normalized_language_name(effective_language),
        normalized_source_language(source_language),
        glossary_salt_component(pass_definition, glossary),
    )
    output_component = output_policy_salt_component(pass_definition)
    if output_component:
        base = salt(base, output_component)

    return salt(base, "retry%d" % retry_attempt) if retry_attempt else base


def retranslation_salt(
    pass_definition: PassDefinition,
    glossary: tuple[GlossaryEntry, ...] = (),
    effective_language: str = "",
    source_language: str = "",
) -> str:
    return salt(
        pass_salt(
            pass_definition,
            glossary,
            effective_language=effective_language,
            source_language=source_language,
        ),
        "retranslate",
    )

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
lazy from collections.abc import Mapping

from l2l.errors import TranslationError
from l2l.glossary import GlossaryEntry
from l2l.json_values import JsonObject
from l2l.monads import Result
from l2l.prompts import (
    OutputPolicyPatch,
    PromptSectionName,
    PromptSectionOverride,
    StylePolicyPatch,
)
from l2l.settings import OptionDefaults, PassDraft, Setup

type DocumentResult = Result[JsonObject, TranslationError]


@dataclass(frozen=True, kw_only=True, slots=True)
class ResolvedPasses:
    option_defaults: OptionDefaults
    global_prompts: Mapping[PromptSectionName, PromptSectionOverride]
    global_style_patch: StylePolicyPatch
    global_output_patch: OutputPolicyPatch
    pass_drafts: tuple[PassDraft, ...]

    @classmethod
    def empty(cls) -> ResolvedPasses:
        return cls(
            option_defaults=OptionDefaults(),
            global_prompts=MappingProxyType({}),
            global_style_patch=StylePolicyPatch(),
            global_output_patch=OutputPolicyPatch(),
            pass_drafts=(),
        )


type Glossaries = tuple[tuple[str, ...], tuple[GlossaryEntry, ...]]
type SetupResult = Result[Setup, TranslationError]

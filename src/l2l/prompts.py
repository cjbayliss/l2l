from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, final, get_args
lazy from collections.abc import Mapping

from l2l.languages import (
    AUTO_SOURCE_LANGUAGE,
    MIXED_SOURCE_LANGUAGE,
    NON_LANGUAGE_CODES,
    expected_target_scripts,
    language_base_and_region,
    normalized_source_language,
    scripts_description,
    source_language_description,
)

PromptSectionName = Literal[
    "role",
    "task",
    "languages",
    "source_guidance",
    "target_guidance",
    "accuracy",
    "fluency",
    "style",
    "context_handling",
    "output_rules",
]

PROMPT_SECTION_NAMES: Final[tuple[PromptSectionName, ...]] = (
    "role",
    "task",
    "languages",
    "source_guidance",
    "target_guidance",
    "accuracy",
    "fluency",
    "style",
    "context_handling",
    "output_rules",
)

PromptRole = Literal["translator", "reviewer", "analyser"]

PROMPT_ROLES: Final[tuple[PromptRole, ...]] = ("translator", "reviewer", "analyser")

PromptSectionOrigin = Literal[
    "built-in", "global-inline", "global-file", "pass-inline", "pass-file"
]

TRANSLATOR_ROLE_TEXT: Final = "You are a professional literary translator of fiction."

TRANSLATOR_TASK_TEXT: Final = "Translate the supplied source text."

LANGUAGES_SECTION_TEXT: Final = (
    "Translate from {source_language} into {target_language}."
)

ACCURACY_SECTION_TEXT: Final = (
    "- Translate every sentence and keep every detail, however minor: "
    "actions, temporal markers, and asides. Never summarise, expand, "
    "censor, or drop content.\n"
    "- Before finalising, check every sentence for who does what to whom: "
    "attach vocatives to the correct line, keep who accuses or rebukes "
    "whom exactly as in the source, never shift subjects between clauses, "
    "and never invent agents or contrasts.\n"
    "- Resolve possessives and time references explicitly; never leave "
    "them vague.\n"
    "- Use one consistent pronoun per character, including non-human "
    "speakers; never flip who does what in contrastive sentences.\n"
    "- Render idioms by meaning with matching force, and use the same "
    "rendering for a recurring term on every occurrence, including "
    "callbacks."
)

FLUENCY_SECTION_TEXT: Final = (
    "- No doubled words, stiff comma-spliced constructions, opaque "
    "literalisms, or unnatural lexical choices: choose the natural "
    "{target_language} word, not the closest-looking one.\n"
    "- No source punctuation artefacts: render tildes, separating dashes, "
    "and similar marks naturally in {target_language} instead of copying "
    "them."
)

CONTEXT_HANDLING_SECTION_TEXT: Final = (
    "Treat briefs, drafts, already-translated context, and source context "
    "as reference only: do not translate them, do not continue the story "
    "from them, and do not include them in the output."
)

OUTPUT_RULES_SECTION_TEXT: Final = (
    "Reply with the translated text only: no notes, comments, or source "
    "text. If a title is present, render it fully at the top; never "
    "truncate it or add words it does not carry. If the source is "
    "ambiguous, pick the most contextually plausible reading and "
    "continue; never mark uncertainty."
)

EXACT_PARAGRAPH_RULE_TEXT: Final = (
    "Keep the source's paragraph structure exactly: one output paragraph "
    "per source paragraph, in the same order, separated by blank lines; "
    "do not merge, split, or reorder paragraphs."
)

APPROXIMATE_PARAGRAPH_RULE_TEXT: Final = (
    "Keep the paragraph structure close to the source."
)

MAXIMUM_CHARACTERS_RULE_TEXT: Final = (
    "Keep each output paragraph to at most %d characters."
)

OutputParagraphs = Literal["exact", "approximate"]


@final
@dataclass(frozen=True, slots=True)
class OutputPolicy:
    paragraphs: OutputParagraphs = "exact"
    maximum_characters: int | None = None


@final
@dataclass(frozen=True, slots=True)
class OutputPolicyPatch:
    paragraphs: OutputParagraphs | None = None
    maximum_characters: int | None = None

    def merge(self, extra: OutputPolicyPatch) -> OutputPolicyPatch:
        return OutputPolicyPatch(
            paragraphs=(
                extra.paragraphs if extra.paragraphs is not None else self.paragraphs
            ),
            maximum_characters=(
                extra.maximum_characters
                if extra.maximum_characters is not None
                else self.maximum_characters
            ),
        )

    def resolved(self) -> OutputPolicy:
        return OutputPolicy(
            paragraphs=self.paragraphs or "exact",
            maximum_characters=self.maximum_characters,
        )


REVIEWER_ROLE_TEXT: Final = (
    "You are the review stage of a fiction translation pipeline, "
    "repairing a draft produced by an earlier pass."
)

REVIEWER_TASK_TEXT: Final = (
    "Produce the final translation: keep what the draft already gets "
    "right and repair the rest. When no draft is present, translate the "
    "source yourself under the same rules."
)

REVIEWER_OUTPUT_RULES_TEXT: Final = (
    "Reply with the corrected translation only: no notes, no change "
    "list, no diff or markup, and none of the source, brief, or draft "
    "echoed back. If a draft paragraph still reads as the source "
    "instead of translating it, translate that paragraph in full. If "
    "the source is ambiguous, commit to the most contextually plausible "
    "reading and continue; never mark uncertainty."
)

ANALYSER_ROLE_TEXT: Final = (
    "You are the preparation stage of a fiction translation pipeline."
)

ANALYSER_TASK_TEXT: Final = (
    "You receive one whole document of source text. Do not translate it: "
    "read it and reply with a compact brief. Later translation passes "
    "see the document only in small pieces and cannot look ahead; your "
    "brief is prepended to every piece they translate, so it must carry "
    "everything they would otherwise miss."
)

ANALYSER_LANGUAGES_SECTION_TEXT: Final = (
    "The source is {source_language}; write the brief in {target_language}."
)

ANALYSER_ACCURACY_SECTION_TEXT: Final = (
    "Never invent what the document does not support; ground every "
    "statement in the source. Where the text is ambiguous, commit to "
    "the most contextually plausible reading and state it as fact. "
    "Record every place the source omits the subject: say exactly who "
    "acts or speaks. Never drift between pronouns for one character, "
    "including non-human speakers."
)

ANALYSER_OPENING_LINE: Final = "Preparation brief (reference only):"

ANALYSER_HEADINGS: Final[tuple[str, ...]] = (
    "Story",
    "Cast",
    "Terms",
    "Idioms",
    "Traps",
)

ANALYSER_WORD_LIMIT: Final = 400

ANALYSER_QUOTATION_RULE_TEXT: Final = (
    'Quote the source only in glossary and idiom entries, as original = "rendering".'
)

ANALYSER_OUTPUT_RULES_TEXT: Final = "\n".join(
    (
        "Reply with the brief only: no preamble, no closing remarks, no "
        "markdown fences, and none of the document's own prose beyond "
        "short quoted phrases.",
        f'Open with the exact line "{ANALYSER_OPENING_LINE}" so the translator '
        "can tell the brief apart from the source text.",
        ANALYSER_QUOTATION_RULE_TEXT,
        "Use these headings, dropping any that would add nothing:",
        "- Story: what the document does overall, then the scenes in "
        "order with their paragraph ranges, so a translator holding one "
        "piece knows where it sits and what precedes it.",
        "- Cast: one line per named or speaking character: the source "
        "name, its fixed rendering, who they are, and the pronoun to "
        "use.",
        "- Terms: recurring nouns, titles, and domain terms with one "
        "fixed rendering each, to be reused on every occurrence.",
        "- Idioms: set phrases where force matters more than literals, "
        "noting the register that a literal reading would flatten.",
        "- Traps: anything easy to get wrong, one line each with its resolution.",
        f"Keep the whole brief under {ANALYSER_WORD_LIMIT} words: "
        "it is repeated for every call.",
    )
)

StyleFormality = Literal["preserve", "formal", "informal"]
StyleHonorifics = Literal["keep", "drop", "titles"]
StyleRegisterFidelity = Literal["preserve", "soften"]


@final
@dataclass(frozen=True, slots=True)
class StylePolicy:
    formality: StyleFormality = "preserve"
    honorifics: StyleHonorifics = "keep"
    register_fidelity: StyleRegisterFidelity = "preserve"


@final
@dataclass(frozen=True, slots=True)
class StylePolicyPatch:
    formality: StyleFormality | None = None
    honorifics: StyleHonorifics | None = None
    register_fidelity: StyleRegisterFidelity | None = None

    def merge(self, extra: StylePolicyPatch) -> StylePolicyPatch:
        return StylePolicyPatch(
            formality=(
                extra.formality if extra.formality is not None else self.formality
            ),
            honorifics=(
                extra.honorifics if extra.honorifics is not None else self.honorifics
            ),
            register_fidelity=(
                extra.register_fidelity
                if extra.register_fidelity is not None
                else self.register_fidelity
            ),
        )

    def resolved(self) -> StylePolicy:
        return StylePolicy(
            formality=self.formality or "preserve",
            honorifics=self.honorifics or "keep",
            register_fidelity=self.register_fidelity or "preserve",
        )


TARGET_REGION_LABELS: Final[Mapping[tuple[str, str], str]] = MappingProxyType(
    {
        ("en", "AU"): "Australian English",
        ("en", "GB"): "British English",
        ("en", "US"): "American English",
    }
)

FORMALITY_PRONOUNS: Final[Mapping[tuple[str, str], str]] = MappingProxyType(
    {
        ("fr", "formal"): "Address characters with the formal pronoun (vous).",
        ("fr", "informal"): "Address characters with the informal pronoun (tu).",
        ("de", "formal"): "Address characters with the formal pronoun (Sie).",
        ("de", "informal"): "Address characters with the informal pronoun (du).",
        ("es", "formal"): "Address characters with the formal pronoun (usted).",
        ("es", "informal"): "Address characters with the informal pronoun (tú).",
        ("ru", "formal"): "Address characters with the formal pronoun (вы).",
        ("ru", "informal"): "Address characters with the informal pronoun (ты).",
    }
)

JAPANESE_KOREAN_SOURCE_BASES: Final = frozenset({"ja", "ko"})

CHINESE_SOURCE_GUIDANCE_TEXT: Final = "\n".join(
    (
        "- Chinese often omits the subject: say exactly who acts or speaks.",
        "- Attach vocatives to the correct line; keep who accuses or "
        "rebukes whom exactly as in the source.",
        "- Render idioms by meaning with matching force.",
        "- Verify romanized names syllable by syllable (pinyin) and keep "
        "them consistent.",
        "- Render tildes and repeated characters naturally instead of copying them.",
    )
)

JAPANESE_SOURCE_GUIDANCE_TEXT: Final = "\n".join(
    (
        "- Romanize personal names in consistent Hepburn.",
        "- Keep Japanese name order (surname first) unless the surrounding "
        "translation already uses Western order.",
        "- Render onomatopoeia with a natural equivalent of matching tone "
        "and intensity, not a romanization.",
    )
)

GENERIC_SOURCE_GUIDANCE_TEXT: Final = "\n".join(
    (
        "- Render idioms by meaning; avoid literal calques.",
        "- Use consistent renderings for untranslatable terms.",
    )
)

MIXED_SOURCE_GUIDANCE_TEXT: Final = (
    "- Identify the language of each passage rather than assuming one "
    "dominant language, and translate every passage."
)

HONORIFICS_GUIDANCE_TEXTS: Final[Mapping[StyleHonorifics, str]] = MappingProxyType(
    {
        "keep": "- Render honorific suffixes consistently; do not silently "
        "drop them or inflate them into titles.",
        "drop": "- Drop honorific suffixes naturally; do not romanize them.",
        "titles": "- Render honorifics as natural titles in the target "
        "language instead of romanized suffixes.",
    }
)


def target_region_guidance_text(target_language: str) -> str:
    components = language_base_and_region(target_language)
    if components is None:
        return ""

    label = TARGET_REGION_LABELS.get(components)
    return f"Use {label} spelling and conventions." if label else ""


def target_scripts_guidance_text(target_language: str) -> str:
    scripts = expected_target_scripts(target_language)
    if not scripts:
        return ""

    return "Write the translation in the {} script{}.".format(
        scripts_description(scripts),
        "" if len(scripts) == 1 else "s",
    )


def source_preset_guidance_text(source_language: str) -> str:
    normalized = normalized_source_language(source_language)
    if normalized in (AUTO_SOURCE_LANGUAGE, MIXED_SOURCE_LANGUAGE):
        return MIXED_SOURCE_GUIDANCE_TEXT if normalized == MIXED_SOURCE_LANGUAGE else ""

    components = language_base_and_region(normalized)
    if components is None:
        return ""

    base = components[0]
    if base == "zh":
        return CHINESE_SOURCE_GUIDANCE_TEXT

    if base == "ja":
        return JAPANESE_SOURCE_GUIDANCE_TEXT

    if base in NON_LANGUAGE_CODES:
        return ""

    return GENERIC_SOURCE_GUIDANCE_TEXT


def honorifics_guidance_text(source_language: str, policy: StylePolicy) -> str:
    components = language_base_and_region(normalized_source_language(source_language))
    if components is None or components[0] not in JAPANESE_KOREAN_SOURCE_BASES:
        return ""

    return HONORIFICS_GUIDANCE_TEXTS[policy.honorifics]


def formality_guidance_text(target_language: str, policy: StylePolicy) -> str:
    if policy.formality == "preserve":
        return ""

    components = language_base_and_region(target_language)
    base = components[0] if components else ""
    pronoun_rule = FORMALITY_PRONOUNS.get((base, policy.formality))
    if pronoun_rule is not None:
        return pronoun_rule

    if base == "ja":
        return (
            "Write dialogue and narration in polite 敬体 (desu/masu) style."
            if policy.formality == "formal"
            else "Write dialogue and narration in plain 常体 (da/dearu) style."
        )

    return (
        "Write the translation in a formal register."
        if policy.formality == "formal"
        else "Write the translation in an informal register."
    )


REGISTER_FIDELITY_GUIDANCE_TEXTS: Final[Mapping[StyleRegisterFidelity, str]] = (
    MappingProxyType(
        {
            "preserve": "Keep vulgarity, scorn, and sarcasm at their full "
            "source force.",
            "soften": "Vulgarity, scorn, or sarcasm may be softened, but "
            "preserve their intent.",
        }
    )
)


def register_fidelity_guidance_text(policy: StylePolicy) -> str:
    return REGISTER_FIDELITY_GUIDANCE_TEXTS[policy.register_fidelity]


GLOSSARY_EXTRACTION_INSTRUCTION_TEXT: Final = """
You extract glossary entries from source text. Find named and
speaking characters (assign pronouns only where they are meaningful and
the character can be referred to pronominally) and place names. Reply
with a JSON object {"entries": [...]}: each entry has "original"
holding the exact source name, "translated" with the rendering in the
target language (or null when unknown), "pronouns" with the pronouns
(or null), "type" which is one of "character", "place", "other",
and "notes" with a short note translators need to render the entry
correctly (who or what it is, how it behaves in the story, or how the
rendering was chosen), or null when nothing needs explaining.
Output JSON only: no markdown fences, no commentary. The target
language is named in the user message.
"""

ASCII_FIX_INSTRUCTION_TEXT: Final = """
This paragraph failed to be fully translated or contains non-ASCII
characters. Please analyse it and only output a clean translation that
uses ASCII characters only.
"""


def paragraph_rule_text(exact_paragraphs: bool) -> str:
    return (
        EXACT_PARAGRAPH_RULE_TEXT
        if exact_paragraphs
        else APPROXIMATE_PARAGRAPH_RULE_TEXT
    )


@final
@dataclass(frozen=True, slots=True)
class PromptSectionOverride:
    text: str
    origin: PromptSectionOrigin


@final
@dataclass(frozen=True, slots=True)
class PromptSections:
    texts: Mapping[PromptSectionName, str]

    def __post_init__(self) -> None:
        missing = frozenset(get_args(PromptSectionName)) - frozenset(self.texts)
        extra = frozenset(self.texts) - frozenset(get_args(PromptSectionName))
        if missing or extra:
            section_names = ", ".join(PROMPT_SECTION_NAMES)
            msg = f"prompt sections must cover exactly {section_names}"
            raise ValueError(msg)

    def assembled(self) -> str:
        stripped = {
            name: text.strip()
            for name, text in (
                (name, self.texts[name]) for name in PROMPT_SECTION_NAMES
            )
        }
        return "\n\n".join(
            stripped[name] for name in PROMPT_SECTION_NAMES if stripped[name]
        )

    def included(self) -> tuple[PromptSectionName, ...]:
        return tuple(name for name in PROMPT_SECTION_NAMES if self.texts[name].strip())

    def omitted(self) -> tuple[PromptSectionName, ...]:
        return tuple(
            name for name in PROMPT_SECTION_NAMES if not self.texts[name].strip()
        )


def instruction_with_languages(
    instruction: str, source_language: str, target_language: str
) -> str:
    return instruction.replace(
        "{source_language}", source_language_description(source_language)
    ).replace("{target_language}", target_language)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class PassPromptRequest:
    role: PromptRole
    source_language: str
    target_language: str
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    style_policy: StylePolicy = field(default_factory=StylePolicy)
    overrides: Mapping[PromptSectionName, str] = MappingProxyType({})
    override_origins: Mapping[PromptSectionName, PromptSectionOrigin] = (
        MappingProxyType({})
    )


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class RenderedPassPrompt:
    instruction: str
    role: PromptRole
    included_sections: tuple[PromptSectionName, ...]
    omitted_sections: tuple[PromptSectionName, ...]
    section_origins: Mapping[PromptSectionName, PromptSectionOrigin]


TRANSLATOR_SECTIONS: Final[Mapping[PromptSectionName, str]] = MappingProxyType(
    {
        "role": TRANSLATOR_ROLE_TEXT,
        "task": TRANSLATOR_TASK_TEXT,
        "languages": LANGUAGES_SECTION_TEXT,
        "source_guidance": "",
        "target_guidance": "",
        "accuracy": ACCURACY_SECTION_TEXT,
        "fluency": FLUENCY_SECTION_TEXT,
        "style": "",
        "context_handling": CONTEXT_HANDLING_SECTION_TEXT,
        "output_rules": OUTPUT_RULES_SECTION_TEXT,
    }
)

REVIEWER_SECTIONS: Final[Mapping[PromptSectionName, str]] = MappingProxyType(
    {
        "role": REVIEWER_ROLE_TEXT,
        "task": REVIEWER_TASK_TEXT,
        "languages": LANGUAGES_SECTION_TEXT,
        "source_guidance": "",
        "target_guidance": "",
        "accuracy": ACCURACY_SECTION_TEXT,
        "fluency": FLUENCY_SECTION_TEXT,
        "style": "",
        "context_handling": CONTEXT_HANDLING_SECTION_TEXT,
        "output_rules": REVIEWER_OUTPUT_RULES_TEXT,
    }
)

ANALYSER_SECTIONS: Final[Mapping[PromptSectionName, str]] = MappingProxyType(
    {
        "role": ANALYSER_ROLE_TEXT,
        "task": ANALYSER_TASK_TEXT,
        "languages": ANALYSER_LANGUAGES_SECTION_TEXT,
        "source_guidance": "",
        "target_guidance": "",
        "accuracy": ANALYSER_ACCURACY_SECTION_TEXT,
        "fluency": "",
        "style": "",
        "context_handling": "",
        "output_rules": ANALYSER_OUTPUT_RULES_TEXT,
    }
)


ROLE_SECTIONS_BY_ROLE: Final[Mapping[PromptRole, Mapping[PromptSectionName, str]]] = (
    MappingProxyType(
        {
            "translator": TRANSLATOR_SECTIONS,
            "reviewer": REVIEWER_SECTIONS,
            "analyser": ANALYSER_SECTIONS,
        }
    )
)


def role_sections(role: PromptRole) -> Mapping[PromptSectionName, str]:
    return ROLE_SECTIONS_BY_ROLE[role]


def merged_output_rules(static_text: str, generated_rule: str) -> str:
    if not generated_rule:
        return static_text

    if not static_text.strip():
        return generated_rule

    return f"{static_text}\n\n{generated_rule}"


def joined_fragments(*fragments: str) -> str:
    return "\n".join(fragment for fragment in fragments if fragment)


def generated_guidance_fragments(
    request: PassPromptRequest,
) -> Mapping[PromptSectionName, str]:
    if request.role not in ("translator", "reviewer"):
        return MappingProxyType({})

    return MappingProxyType(
        {
            "source_guidance": joined_fragments(
                source_preset_guidance_text(request.source_language),
                honorifics_guidance_text(request.source_language, request.style_policy),
            ),
            "target_guidance": joined_fragments(
                target_region_guidance_text(request.target_language),
                target_scripts_guidance_text(request.target_language),
            ),
            "style": joined_fragments(
                formality_guidance_text(request.target_language, request.style_policy),
                register_fidelity_guidance_text(request.style_policy),
            ),
        }
    )


def generated_output_rule_text(request: PassPromptRequest) -> str:
    if request.role not in ("translator", "reviewer"):
        return ""

    return joined_fragments(
        paragraph_rule_text(request.output_policy.paragraphs == "exact"),
        (
            MAXIMUM_CHARACTERS_RULE_TEXT % request.output_policy.maximum_characters
            if request.output_policy.maximum_characters is not None
            else ""
        ),
    )


def render_pass_prompt(request: PassPromptRequest) -> RenderedPassPrompt:
    sections: dict[PromptSectionName, str] = {
        **role_sections(request.role),
        **request.overrides,
    }
    generated_rule = generated_output_rule_text(request)
    sections["output_rules"] = merged_output_rules(
        sections["output_rules"], generated_rule
    )
    for name, fragment in generated_guidance_fragments(request).items():
        sections[name] = merged_output_rules(sections[name], fragment)
    prompt_sections = PromptSections(MappingProxyType(sections))
    instruction = instruction_with_languages(
        prompt_sections.assembled(),
        request.source_language,
        request.target_language,
    )
    return RenderedPassPrompt(
        instruction=instruction,
        role=request.role,
        included_sections=prompt_sections.included(),
        omitted_sections=prompt_sections.omitted(),
        section_origins=MappingProxyType(
            {
                name: request.override_origins.get(name, "built-in")
                for name in PROMPT_SECTION_NAMES
            }
        ),
    )

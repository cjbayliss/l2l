from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from l2l.config.resolution import (
    finalize_pass_draft,
    finalize_passes,
    selected_pass_roles,
)
from l2l.monads import Ok
from l2l.prompts import (
    ANALYSER_HEADINGS,
    ANALYSER_OPENING_LINE,
    ANALYSER_OUTPUT_RULES_TEXT,
    APPROXIMATE_PARAGRAPH_RULE_TEXT,
    ASCII_FIX_INSTRUCTION_TEXT,
    CHINESE_SOURCE_GUIDANCE_TEXT,
    EXACT_PARAGRAPH_RULE_TEXT,
    FLUENCY_SECTION_TEXT,
    GENERIC_SOURCE_GUIDANCE_TEXT,
    GLOSSARY_EXTRACTION_INSTRUCTION_TEXT,
    HONORIFICS_GUIDANCE_TEXTS,
    JAPANESE_SOURCE_GUIDANCE_TEXT,
    MIXED_SOURCE_GUIDANCE_TEXT,
    PROMPT_SECTION_NAMES,
    REGISTER_FIDELITY_GUIDANCE_TEXTS,
    REVIEWER_SECTIONS,
    OutputPolicy,
    OutputPolicyPatch,
    PassPromptRequest,
    PromptSections,
    StylePolicy,
    StylePolicyPatch,
    formality_guidance_text,
    honorifics_guidance_text,
    instruction_with_languages,
    paragraph_rule_text,
    render_pass_prompt,
    source_preset_guidance_text,
    target_region_guidance_text,
    target_scripts_guidance_text,
)
from l2l.settings import UNSET, PassDraft, PassMode, build_settings


def translator_request(
    source_language: str = "Chinese", target_language: str = "English"
) -> PassPromptRequest:
    return PassPromptRequest(
        role="translator",
        source_language=source_language,
        target_language=target_language,
    )


def reviewer_request(
    source_language: str = "Chinese", target_language: str = "English"
) -> PassPromptRequest:
    return PassPromptRequest(
        role="reviewer",
        source_language=source_language,
        target_language=target_language,
    )


def analyser_request(
    source_language: str = "Chinese", target_language: str = "English"
) -> PassPromptRequest:
    return PassPromptRequest(
        role="analyser",
        source_language=source_language,
        target_language=target_language,
        output_policy=OutputPolicy(),
    )


TRANSLATOR_GOLDEN = "\n".join(
    (
        "You are a professional literary translator of fiction.",
        "",
        "Translate the supplied source text.",
        "",
        "Translate from Chinese into English.",
        "",
        "- Chinese often omits the subject: say exactly who acts or speaks.",
        "- Attach vocatives to the correct line; keep who accuses or rebukes "
        "whom exactly as in the source.",
        "- Render idioms by meaning with matching force.",
        "- Verify romanized names syllable by syllable (pinyin) and keep them "
        "consistent.",
        "- Render tildes and repeated characters naturally instead of copying them.",
        "",
        "Write the translation in the Latin script.",
        "",
        "- Translate every sentence and keep every detail, however minor: actions, "
        "temporal markers, and asides. Never summarise, expand, censor, or drop "
        "content.",
        "- Before finalising, check every sentence for who does what to whom: attach "
        "vocatives to the correct line, keep who accuses or rebukes whom exactly "
        "as in the source, never shift subjects between clauses, and never invent "
        "agents or "
        "contrasts.",
        "- Resolve possessives and time references explicitly; never leave them vague.",
        "- Use one consistent pronoun per character, including non-human speakers; "
        "never flip who does what in contrastive sentences.",
        "- Render idioms by meaning with matching force, and use the same rendering "
        "for a recurring term on every occurrence, including callbacks.",
        "",
        "- No doubled words, stiff comma-spliced constructions, opaque literalisms, or "
        "unnatural lexical choices: choose the natural English word, not the "
        "closest-looking one.",
        "- No source punctuation artefacts: render tildes, separating dashes, and "
        "similar marks naturally in English instead of copying them.",
        "",
        "Keep vulgarity, scorn, and sarcasm at their full source force.",
        "",
        "Treat briefs, drafts, already-translated context, and source context as "
        "reference only: do not translate them, do not continue the story from them, "
        "and do not "
        "include them in the output.",
        "",
        "Reply with the translated text only: no notes, comments, or source text. If a "
        "title is present, render it fully at the top; never truncate it or add words "
        "it does not carry. If the source is ambiguous, pick the most contextually "
        "plausible "
        "reading and continue; never mark uncertainty.",
        "",
        "Keep the source's paragraph structure exactly: one output paragraph per "
        "source paragraph, in the same order, separated by blank lines; do not merge, "
        "split, or "
        "reorder paragraphs.",
    )
)


def test_translator_golden_prompt() -> None:
    rendered = render_pass_prompt(translator_request())
    assert rendered.instruction == TRANSLATOR_GOLDEN
    assert rendered.role == "translator"
    assert rendered.included_sections == PROMPT_SECTION_NAMES
    assert rendered.omitted_sections == ()
    assert all(
        rendered.section_origins[name] == "built-in" for name in PROMPT_SECTION_NAMES
    )


def test_reviewer_prompt_uses_reviewer_sections_and_generated_guidance() -> None:
    rendered = render_pass_prompt(reviewer_request("Japanese"))
    assert rendered.role == "reviewer"
    assert rendered.instruction.startswith(REVIEWER_SECTIONS["role"])
    assert REVIEWER_SECTIONS["task"] in rendered.instruction
    assert REVIEWER_SECTIONS["output_rules"] in rendered.instruction
    assert JAPANESE_SOURCE_GUIDANCE_TEXT in rendered.instruction
    assert CHINESE_SOURCE_GUIDANCE_TEXT not in rendered.instruction


def test_analyser_prompt_defines_the_brief_contract() -> None:
    rendered = render_pass_prompt(analyser_request())
    assert rendered.role == "analyser"
    assert 'Open with the exact line "Preparation brief (reference only):"' in (
        rendered.instruction
    )
    for heading in ANALYSER_HEADINGS:
        assert f"- {heading}:" in rendered.instruction
    assert "under 400 words" in rendered.instruction


def test_dedicated_template_bytes_are_unchanged() -> None:
    settings = build_settings()
    assert settings.glossary_extraction_instruction == (
        GLOSSARY_EXTRACTION_INSTRUCTION_TEXT
    )
    assert settings.ascii_fix_instruction == ASCII_FIX_INSTRUCTION_TEXT


def test_rendered_instruction_has_no_padding_or_blank_gaps() -> None:
    rendered = render_pass_prompt(translator_request())
    assert not rendered.instruction.startswith("\n")
    assert not rendered.instruction.endswith("\n")
    assert "\n\n\n" not in rendered.instruction


def test_unknown_braces_are_left_untouched() -> None:
    substituted = instruction_with_languages(
        "Keep {like: this} and {target_language}.", "auto", "English"
    )
    assert substituted == "Keep {like: this} and English."


def test_effective_per_pass_target_language_appears_in_the_built_in() -> None:
    rendered = render_pass_prompt(translator_request(target_language="British English"))
    assert "Translate from Chinese into British English." in rendered.instruction


def test_paragraph_rule_is_generated_not_static() -> None:
    exact = render_pass_prompt(translator_request())
    assert paragraph_rule_text(True) in exact.instruction
    assert paragraph_rule_text(False) not in exact.instruction
    approximate = render_pass_prompt(
        PassPromptRequest(
            role="translator",
            source_language="Chinese",
            target_language="English",
            output_policy=OutputPolicy(paragraphs="approximate"),
        )
    )
    assert paragraph_rule_text(False) in approximate.instruction
    assert paragraph_rule_text(True) not in approximate.instruction
    analyser = render_pass_prompt(
        PassPromptRequest(
            role="analyser",
            source_language="Chinese",
            target_language="English",
            output_policy=OutputPolicy(),
        )
    )
    assert paragraph_rule_text(True) not in analyser.instruction
    assert paragraph_rule_text(False) not in analyser.instruction


def legacy_draft() -> PassDraft:
    return PassDraft(
        "translate", "translate", "Translate to {target_language}.", {}, UNSET, False
    )


def omitted_draft(mode: PassMode) -> PassDraft:
    return PassDraft(mode, mode, None, {}, UNSET, False)


def test_selected_pass_roles_across_modes() -> None:
    drafts = (
        PassDraft("analyse", "analysis", "Analyse.", {}, UNSET, False),
        PassDraft("glossary", "update_glossary", None, {}, UNSET, False),
        PassDraft("first", "translate", None, {}, UNSET, False),
        PassDraft("second", "review", None, {}, UNSET, False),
        PassDraft("third", "translate", None, {}, UNSET, False),
    )
    assert selected_pass_roles(drafts) == (
        "analyser",
        None,
        "translator",
        "reviewer",
        "translator",
    )


def test_explicit_instruction_bytes_pass_through_unchanged() -> None:
    for draft, expected in (
        (legacy_draft(), "Translate to {target_language}."),
        (
            PassDraft("translate", "translate", "From file.", {}, UNSET, False),
            "From file.",
        ),
    ):
        result = finalize_pass_draft(
            draft,
            "translator",
            "English",
            "auto",
            {},
            StylePolicyPatch(),
            OutputPolicyPatch(),
            None,
        )
        assert isinstance(result, Ok)
        assert result.value[0].instruction == expected
        assert result.value[0].prompt_role is None


def test_update_glossary_omitted_instruction_stays_dedicated() -> None:
    result = finalize_pass_draft(
        omitted_draft("update_glossary"),
        None,
        "English",
        "auto",
        {},
        StylePolicyPatch(),
        OutputPolicyPatch(),
        None,
    )
    assert isinstance(result, Ok)
    assert result.value[0].instruction == ""
    assert result.value[0].prompt_role is None


def test_omitted_text_pass_renders_built_in() -> None:
    result = finalize_pass_draft(
        omitted_draft("translate"),
        "translator",
        "English",
        "Chinese",
        {},
        StylePolicyPatch(),
        OutputPolicyPatch(),
        None,
    )
    assert isinstance(result, Ok)
    finalized = result.value[0]
    assert finalized.prompt_role == "translator"
    assert finalized.instruction == TRANSLATOR_GOLDEN
    assert finalized.prompt_omitted_sections == ()
    assert finalized.prompt_section_origins["accuracy"] == "built-in"


def test_review_pass_renders_reviewer_and_later_translate_passes_stay_translator() -> (
    None
):
    finalized = finalize_passes(
        (
            omitted_draft("translate"),
            omitted_draft("review"),
            omitted_draft("translate"),
        ),
        "English",
        "auto",
        {},
        StylePolicyPatch(),
        OutputPolicyPatch(),
        None,
    )
    assert isinstance(finalized, Ok)
    roles = tuple(definition.prompt_role for definition in finalized.value.passes)
    assert roles == ("translator", "reviewer", "translator")


def test_analysis_without_instruction_renders_the_analyser() -> None:
    result = finalize_passes(
        (omitted_draft("analysis"),),
        "English",
        "auto",
        {},
        StylePolicyPatch(),
        OutputPolicyPatch(),
        None,
    )
    assert isinstance(result, Ok)
    finalized = result.value.passes[0]
    assert finalized.prompt_role == "analyser"
    assert finalized.instruction.startswith(
        "You are the preparation stage of a fiction translation pipeline."
    )
    assert ANALYSER_OPENING_LINE in finalized.instruction


def test_finalize_passes_renders_every_omitted_text_pass() -> None:
    finalized = finalize_passes(
        (legacy_draft(), omitted_draft("translate")),
        "English",
        "auto",
        {},
        StylePolicyPatch(),
        OutputPolicyPatch(),
        None,
    )
    assert isinstance(finalized, Ok)
    assert finalized.value.passes[0].instruction == "Translate to {target_language}."
    assert finalized.value.passes[1].prompt_role == "translator"


def test_incomplete_section_mappings_are_rejected() -> None:
    incomplete = dict.fromkeys(PROMPT_SECTION_NAMES[:-1], "")
    with pytest.raises(ValueError, match="prompt sections must cover exactly"):
        PromptSections(MappingProxyType(incomplete))


def test_target_region_labels_and_bare_or_unknown_regions() -> None:
    assert (
        target_region_guidance_text("en-AU")
        == "Use Australian English spelling and conventions."
    )
    assert (
        target_region_guidance_text("en-GB")
        == "Use British English spelling and conventions."
    )
    assert (
        target_region_guidance_text("en-US")
        == "Use American English spelling and conventions."
    )
    assert target_region_guidance_text("en") == ""
    assert target_region_guidance_text("en-XX") == ""
    assert target_region_guidance_text("Zorblaxian") == ""
    assert target_region_guidance_text("en-AU") == target_region_guidance_text("en_AU")
    assert target_region_guidance_text("en-AU") == target_region_guidance_text("EN-AU")


def test_target_scripts_guidance_only_for_known_scripts() -> None:
    assert (
        target_scripts_guidance_text("English")
        == "Write the translation in the Latin script."
    )
    assert (
        target_scripts_guidance_text("Japanese")
        == "Write the translation in the Han, Hiragana, Katakana scripts."
    )
    assert target_scripts_guidance_text("Zorblaxian") == ""


def test_source_presets_by_base_language() -> None:
    assert source_preset_guidance_text("Chinese") == CHINESE_SOURCE_GUIDANCE_TEXT
    assert source_preset_guidance_text("zh-Hant") == CHINESE_SOURCE_GUIDANCE_TEXT
    assert source_preset_guidance_text("Japanese") == JAPANESE_SOURCE_GUIDANCE_TEXT
    assert source_preset_guidance_text("ja") == JAPANESE_SOURCE_GUIDANCE_TEXT
    assert source_preset_guidance_text("French") == GENERIC_SOURCE_GUIDANCE_TEXT
    assert source_preset_guidance_text("Zorblaxian") == ""
    assert source_preset_guidance_text("auto") == ""
    assert source_preset_guidance_text("mixed") == MIXED_SOURCE_GUIDANCE_TEXT
    assert "dominant language" in MIXED_SOURCE_GUIDANCE_TEXT
    assert "untranslated" not in MIXED_SOURCE_GUIDANCE_TEXT


def test_honorifics_guidance_only_for_japanese_and_korean() -> None:
    keep = StylePolicy()
    assert (
        honorifics_guidance_text("Japanese", keep) == HONORIFICS_GUIDANCE_TEXTS["keep"]
    )
    assert honorifics_guidance_text("Korean", keep) != ""
    assert honorifics_guidance_text("Chinese", keep) == ""
    assert honorifics_guidance_text("auto", keep) == ""
    drop = replace(keep, honorifics="drop")
    titles = replace(keep, honorifics="titles")
    assert honorifics_guidance_text("ja", drop) == HONORIFICS_GUIDANCE_TEXTS["drop"]
    assert honorifics_guidance_text("ja", titles) == HONORIFICS_GUIDANCE_TEXTS["titles"]


def test_formality_guidance_table_and_fallback() -> None:
    preserve = StylePolicy()
    assert formality_guidance_text("Japanese", preserve) == ""
    assert (
        formality_guidance_text("ja", replace(preserve, formality="formal"))
        == "Write dialogue and narration in polite 敬体 (desu/masu) style."
    )
    assert (
        formality_guidance_text("ja", replace(preserve, formality="informal"))
        == "Write dialogue and narration in plain 常体 (da/dearu) style."
    )
    assert (
        formality_guidance_text("fr", replace(preserve, formality="formal"))
        == "Address characters with the formal pronoun (vous)."
    )
    assert (
        formality_guidance_text("de", replace(preserve, formality="informal"))
        == "Address characters with the informal pronoun (du)."
    )
    assert (
        formality_guidance_text("es", replace(preserve, formality="formal"))
        == "Address characters with the formal pronoun (usted)."
    )
    assert (
        formality_guidance_text("ru", replace(preserve, formality="informal"))
        == "Address characters with the informal pronoun (ты)."
    )
    assert (
        formality_guidance_text("en", replace(preserve, formality="formal"))
        == "Write the translation in a formal register."
    )
    assert (
        formality_guidance_text("Zorblaxian", replace(preserve, formality="informal"))
        == "Write the translation in an informal register."
    )


def test_register_fidelity_fragments_never_contradict_fluency() -> None:
    assert (
        REGISTER_FIDELITY_GUIDANCE_TEXTS["preserve"]
        == "Keep vulgarity, scorn, and sarcasm at their full source force."
    )
    assert (
        REGISTER_FIDELITY_GUIDANCE_TEXTS["soften"]
        == "Vulgarity, scorn, or sarcasm may be softened, but preserve "
        "their intent."
    )
    assert "sarcasm" not in FLUENCY_SECTION_TEXT


def test_generated_guidance_applies_to_translator_and_not_analyser() -> None:
    translator = render_pass_prompt(translator_request())
    assert CHINESE_SOURCE_GUIDANCE_TEXT in translator.instruction
    assert REGISTER_FIDELITY_GUIDANCE_TEXTS["preserve"] in translator.instruction
    assert EXACT_PARAGRAPH_RULE_TEXT in translator.instruction
    analyser = render_pass_prompt(analyser_request())
    assert CHINESE_SOURCE_GUIDANCE_TEXT not in analyser.instruction
    assert REGISTER_FIDELITY_GUIDANCE_TEXTS["preserve"] not in analyser.instruction
    assert EXACT_PARAGRAPH_RULE_TEXT not in analyser.instruction
    assert ANALYSER_OUTPUT_RULES_TEXT in analyser.instruction


def test_generated_guidance_survives_section_overrides() -> None:
    rendered = render_pass_prompt(
        PassPromptRequest(
            role="reviewer",
            source_language="Japanese",
            target_language="en-AU",
            style_policy=StylePolicy(formality="formal"),
            overrides=MappingProxyType(
                {
                    "source_guidance": "Custom source note.",
                    "target_guidance": "",
                    "style": "",
                }
            ),
            override_origins=MappingProxyType(
                {
                    "source_guidance": "pass-inline",
                    "target_guidance": "pass-inline",
                    "style": "pass-inline",
                }
            ),
        )
    )
    assert "Custom source note." in rendered.instruction
    assert JAPANESE_SOURCE_GUIDANCE_TEXT in rendered.instruction
    assert "- Render honorific suffixes consistently" in rendered.instruction
    assert "Use Australian English spelling and conventions." in rendered.instruction
    assert "Write the translation in the Latin script." in rendered.instruction
    assert "Address characters with the formal pronoun" not in rendered.instruction
    assert REGISTER_FIDELITY_GUIDANCE_TEXTS["preserve"] in rendered.instruction
    assert rendered.section_origins["source_guidance"] == "pass-inline"
    assert rendered.section_origins["target_guidance"] == "pass-inline"
    assert rendered.section_origins["style"] == "pass-inline"
    assert rendered.section_origins["accuracy"] == "built-in"


def test_generated_output_rules_survive_section_overrides() -> None:
    request = PassPromptRequest(
        role="translator",
        source_language="Chinese",
        target_language="English",
        output_policy=OutputPolicy(paragraphs="approximate", maximum_characters=50),
        overrides={"output_rules": "Reply with the translation only."},
    )
    rendered = render_pass_prompt(request)
    assert "Reply with the translation only." in rendered.instruction
    assert APPROXIMATE_PARAGRAPH_RULE_TEXT in rendered.instruction
    assert "Keep each output paragraph to at most 50 characters." in (
        rendered.instruction
    )


def test_output_policy_golden_prompt() -> None:
    rendered = render_pass_prompt(
        PassPromptRequest(
            role="translator",
            source_language="Chinese",
            target_language="English",
            output_policy=OutputPolicy(paragraphs="approximate", maximum_characters=30),
        )
    )
    assert APPROXIMATE_PARAGRAPH_RULE_TEXT in rendered.instruction
    assert "Keep each output paragraph to at most 30 characters." in (
        rendered.instruction
    )
    assert rendered.included_sections == PROMPT_SECTION_NAMES

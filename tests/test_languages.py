from l2l.languages import (
    expected_target_scripts,
    instruction_uses_target_language,
    normalized_language_name,
    script_description,
    scripts_description,
)


def test_normalized_language_name_strips_decorations_and_case() -> None:
    assert normalized_language_name("  English  ") == "english"
    assert normalized_language_name("ENGLISH") == "english"
    assert normalized_language_name("Chinese (Simplified)") == "chinese simplified"
    assert normalized_language_name("zh-CN") == "zh cn"
    assert normalized_language_name("") == ""


def test_expected_target_scripts_resolves_names_codes_and_variants() -> None:
    expected_scripts = (
        ("English", ("Latn",)),
        ("en", ("Latn",)),
        ("eng", ("Latn",)),
        ("French", ("Latn",)),
        ("fr", ("Latn",)),
        ("Chinese", ("Hani",)),
        ("zh", ("Hani",)),
        ("Mandarin", ("Hani",)),
        ("Cantonese", ("Hani",)),
        ("Russian", ("Cyrl",)),
        ("uk", ("Cyrl",)),
        ("Arabic", ("Arab",)),
        ("Farsi", ("Arab",)),
        ("Greek", ("Grek",)),
        ("Hebrew", ("Hebr",)),
        ("Thai", ("Thai",)),
        ("Hindi", ("Deva",)),
        ("Bengali", ("Beng",)),
        ("Tamil", ("Taml",)),
        ("Telugu", ("Telu",)),
        ("Kannada", ("Knda",)),
        ("Malayalam", ("Mlym",)),
        ("Gujarati", ("Gujr",)),
        ("Punjabi", ("Guru",)),
        ("Sinhala", ("Sinh",)),
        ("Lao", ("Laoo",)),
        ("Burmese", ("Mymr",)),
        ("Khmer", ("Khmr",)),
        ("Georgian", ("Geor",)),
        ("Armenian", ("Armn",)),
        ("Amharic", ("Ethi",)),
        ("Tibetan", ("Tibt",)),
        ("Japanese", ("Hani", "Hira", "Kana")),
        ("ja", ("Hani", "Hira", "Kana")),
        ("jpn", ("Hani", "Hira", "Kana")),
        ("Korean", ("Hang", "Hani")),
        ("ko", ("Hang", "Hani")),
        ("kor", ("Hang", "Hani")),
        ("ja-Jpan", ("Hani", "Hira", "Kana")),
        ("ja-Hrkt", ("Hira", "Kana")),
        ("ko-Kore", ("Hang", "Hani")),
        ("ko-Jamo", ("Hang",)),
        ("zh-Hans", ("Hani",)),
        ("zh-Hant", ("Hani",)),
        ("zh-Hanb", ("Hani", "Bopo")),
        ("SIMPLIFIED CHINESE", ("Hani",)),
        ("Traditional Chinese", ("Hani",)),
        ("chinese-simplified", ("Hani",)),
        ("English (US)", ("Latn",)),
        ("sr-Latn", ("Latn",)),
        ("pa-Arab", ("Arab",)),
        ("Klingon", ("Latn",)),
    )
    assert all(
        expected_target_scripts(language) == scripts
        for language, scripts in expected_scripts
    )


def test_expected_target_scripts_returns_nothing_for_unknown_names() -> None:
    for language in (
        "Zorblaxian",
        "not a language",
        "Devanagari",
        "Cyrillic",
        "",
        "und",
        "mul",
        "zxx",
    ):
        assert expected_target_scripts(language) == ()


def test_script_descriptions() -> None:
    assert script_description("Latn") == "Latin"
    assert script_description("Hira") == "Hiragana"
    assert script_description("Kana") == "Katakana"
    assert script_description("Hani") == "Han"
    assert script_description("Zzzz") == "Unknown"
    assert script_description("NotACode") == "NotACode"
    assert scripts_description(("Hani", "Hira", "Kana")) == "Han, Hiragana, Katakana"
    assert scripts_description(("Latn",)) == "Latin"
    assert scripts_description(()) == "unknown"


def test_instruction_uses_target_language() -> None:
    assert instruction_uses_target_language(
        "Translate into {target_language}.", "Chinese"
    )
    assert instruction_uses_target_language("Translate into English.", "English")
    assert instruction_uses_target_language("Translate into FRENCH.", "French")
    assert instruction_uses_target_language(
        "Traduire en chinese simplified.", "Chinese (Simplified)"
    )
    assert not instruction_uses_target_language("Translate.", "English")
    assert not instruction_uses_target_language("Translate into German.", "English")

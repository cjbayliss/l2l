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


def test_expected_target_scripts_resolves_names_and_codes() -> None:
    assert expected_target_scripts("English") == ("Latn",)
    assert expected_target_scripts("en") == ("Latn",)
    assert expected_target_scripts("eng") == ("Latn",)
    assert expected_target_scripts("French") == ("Latn",)
    assert expected_target_scripts("fr") == ("Latn",)
    assert expected_target_scripts("Spanish") == ("Latn",)
    assert expected_target_scripts("Cyrillic") == ()


def test_expected_target_scripts_resolves_non_latin_scripts() -> None:
    assert expected_target_scripts("Chinese") == ("Hani",)
    assert expected_target_scripts("zh") == ("Hani",)
    assert expected_target_scripts("Mandarin") == ("Hani",)
    assert expected_target_scripts("Cantonese") == ("Hani",)
    assert expected_target_scripts("Russian") == ("Cyrl",)
    assert expected_target_scripts("uk") == ("Cyrl",)
    assert expected_target_scripts("Arabic") == ("Arab",)
    assert expected_target_scripts("Farsi") == ("Arab",)
    assert expected_target_scripts("Greek") == ("Grek",)
    assert expected_target_scripts("Hebrew") == ("Hebr",)
    assert expected_target_scripts("Thai") == ("Thai",)
    assert expected_target_scripts("th") == ("Thai",)
    assert expected_target_scripts("Hindi") == ("Deva",)
    assert expected_target_scripts("Bengali") == ("Beng",)
    assert expected_target_scripts("Tamil") == ("Taml",)
    assert expected_target_scripts("Telugu") == ("Telu",)
    assert expected_target_scripts("Kannada") == ("Knda",)
    assert expected_target_scripts("Malayalam") == ("Mlym",)
    assert expected_target_scripts("Gujarati") == ("Gujr",)
    assert expected_target_scripts("Punjabi") == ("Guru",)
    assert expected_target_scripts("Sinhala") == ("Sinh",)
    assert expected_target_scripts("Lao") == ("Laoo",)
    assert expected_target_scripts("Burmese") == ("Mymr",)
    assert expected_target_scripts("Khmer") == ("Khmr",)
    assert expected_target_scripts("Georgian") == ("Geor",)
    assert expected_target_scripts("Armenian") == ("Armn",)
    assert expected_target_scripts("Amharic") == ("Ethi",)
    assert expected_target_scripts("Tibetan") == ("Tibt",)


def test_expected_target_scripts_preserves_multiscript_order() -> None:
    assert expected_target_scripts("Japanese") == ("Hani", "Hira", "Kana")
    assert expected_target_scripts("ja") == ("Hani", "Hira", "Kana")
    assert expected_target_scripts("jpn") == ("Hani", "Hira", "Kana")
    assert expected_target_scripts("Korean") == ("Hang", "Hani")
    assert expected_target_scripts("ko") == ("Hang", "Hani")
    assert expected_target_scripts("kor") == ("Hang", "Hani")


def test_expected_target_scripts_resolves_cldr_script_variants() -> None:
    assert expected_target_scripts("ja-Jpan") == ("Hani", "Hira", "Kana")
    assert expected_target_scripts("ja-Hrkt") == ("Hira", "Kana")
    assert expected_target_scripts("ko-Kore") == ("Hang", "Hani")
    assert expected_target_scripts("ko-Jamo") == ("Hang",)
    assert expected_target_scripts("zh-Hans") == ("Hani",)
    assert expected_target_scripts("zh-Hant") == ("Hani",)
    assert expected_target_scripts("zh-Hanb") == ("Hani", "Bopo")


def test_expected_target_scripts_is_case_and_punctuation_insensitive() -> None:
    assert expected_target_scripts("SIMPLIFIED CHINESE") == ("Hani",)
    assert expected_target_scripts("Traditional Chinese") == ("Hani",)
    assert expected_target_scripts("chinese-simplified") == ("Hani",)


def test_expected_target_scripts_resolves_decorated_names_and_tags() -> None:
    assert expected_target_scripts("English (US)") == ("Latn",)
    assert expected_target_scripts("sr-Latn") == ("Latn",)
    assert expected_target_scripts("pa-Arab") == ("Arab",)
    assert expected_target_scripts("Klingon") == ("Latn",)


def test_expected_target_scripts_returns_nothing_for_unknown_names() -> None:
    assert expected_target_scripts("Zorblaxian") == ()
    assert expected_target_scripts("not a language") == ()
    assert expected_target_scripts("Devanagari") == ()
    assert expected_target_scripts("") == ()
    assert expected_target_scripts("und") == ()
    assert expected_target_scripts("mul") == ()
    assert expected_target_scripts("zxx") == ()


def test_script_description_uses_unicode_script_names() -> None:
    assert script_description("Latn") == "Latin"
    assert script_description("Hira") == "Hiragana"
    assert script_description("Kana") == "Katakana"
    assert script_description("Hani") == "Han"
    assert script_description("Zzzz") == "Unknown"
    assert script_description("NotACode") == "NotACode"


def test_scripts_description_formats_known_and_unknown_scripts() -> None:
    assert scripts_description(("Hani", "Hira", "Kana")) == "Han, Hiragana, Katakana"
    assert scripts_description(("Latn",)) == "Latin"
    assert scripts_description(()) == "unknown"


def test_instruction_uses_target_language_accepts_placeholders_and_names() -> None:
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

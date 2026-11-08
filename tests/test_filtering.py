import re

from l2l.config.parsing import parse_filter_table
from l2l.config.resolution import resolve_filter_settings
from l2l.errors import ConfigError, describe
from l2l.filtering import (
    FilterSettings,
    FilterSpec,
    duplicate_paragraph_removals,
    filter_paragraphs,
    removal_counts,
    short_paragraph_removals,
    spec_removals,
)
from l2l.monads import Err, Ok
lazy from l2l.config.models import DocumentResult


def test_removal_helpers_normalize_whitespace() -> None:
    paragraphs = ("First\tparagraph", "Noise", "First  paragraph")
    assert duplicate_paragraph_removals(paragraphs) == (2,)
    assert short_paragraph_removals((" \t ", "ab", " a b ", "abcd"), 3) == (0, 1)


def test_spec_removals_handles_each_kind_and_pattern_fallback() -> None:
    paragraphs = ("keep", "remove 12", "keep")
    duplicate = FilterSpec("duplicate_paragraphs")
    pattern = FilterSpec("pattern", pattern=re.compile(r"\d+"))
    short = FilterSpec("short_paragraphs", minimum_characters=7)
    assert spec_removals(duplicate, paragraphs) == (2,)
    assert spec_removals(pattern, paragraphs) == (1,)
    assert spec_removals(short, paragraphs) == (0, 2)
    assert spec_removals(FilterSpec("pattern"), paragraphs) == ()


def test_filter_paragraphs_composes_filters_and_canonicalizes_separators() -> None:
    text = "First\n\nDrop 12\n\n\nFirst\n\nTiny\n\nLast paragraph"
    outcome = filter_paragraphs(
        text,
        (
            FilterSpec("duplicate_paragraphs"),
            FilterSpec("pattern", pattern=re.compile(r"Drop|Tiny")),
            FilterSpec("short_paragraphs", minimum_characters=5),
        ),
    )
    assert outcome.text == "First\n\nLast paragraph"
    assert outcome.paragraph_count == 5
    assert tuple((removal.index, removal.kind) for removal in outcome.removals) == (
        (2, "duplicate_paragraphs"),
        (1, "pattern"),
        (3, "pattern"),
    )
    assert removal_counts(outcome.removals) == (
        ("duplicate_paragraphs", 1),
        ("pattern", 2),
    )
    assert filter_paragraphs("One\n\n\nTwo", ()).text == "One\n\nTwo"


def test_parse_filter_table_accepts_script_and_all_filter_forms() -> None:
    result = parse_filter_table(
        "f",
        {
            "script": "python3 'clean chapter.py' --fast",
            "filters": [
                "duplicate_paragraphs",
                {"kind": "duplicate_paragraphs"},
                {"kind": "pattern", "pattern": "^advert"},
                {"kind": "short_paragraphs", "minimum_characters": 4},
            ],
        },
    )
    assert isinstance(result, Ok)
    assert result.value.script == ("python3", "clean chapter.py", "--fast")
    assert result.value.filters == (
        FilterSpec("duplicate_paragraphs"),
        FilterSpec("duplicate_paragraphs"),
        FilterSpec("pattern", pattern=re.compile("^advert")),
        FilterSpec("short_paragraphs", minimum_characters=4),
    )

    array_command = parse_filter_table("f", {"script": ["python3", "clean.py"]})
    assert array_command == Ok(FilterSettings(script=("python3", "clean.py")))


def test_parse_filter_table_rejects_invalid_tables_and_empty_settings() -> None:
    invalid: tuple[tuple[object, str], ...] = (
        (None, "must be a table"),
        ({"unknown": True}, "unknown key"),
        ({"filters": "duplicate_paragraphs"}, "filters must be an array"),
        ({"filters": []}, "filters must not be empty"),
        ({}, "requires at least one"),
        ({"script": []}, "script must be"),
        ({"script": [""]}, "script must be"),
        ({"script": ""}, "script must be"),
        ({"script": "'unterminated"}, "not valid syntax"),
    )
    for table, message in invalid:
        result = parse_filter_table("f", table)
        assert isinstance(result, Err)
        assert message in describe(result.error)


def test_parse_filter_table_rejects_invalid_filter_specs() -> None:
    invalid = (
        ("pattern", "must be"),
        ({"pattern": "x"}, "filter kind must be one of"),
        ({"kind": "unknown"}, "filter kind must be one of"),
        (
            {"kind": "duplicate_paragraphs", "minimum_characters": 3},
            "unknown or inapplicable",
        ),
        ({"kind": "pattern"}, "requires pattern"),
        ({"kind": "pattern", "pattern": ""}, "non-empty string"),
        ({"kind": "pattern", "pattern": "["}, "pattern is invalid"),
        (
            {"kind": "short_paragraphs"},
            "requires minimum_characters",
        ),
        (
            {"kind": "short_paragraphs", "minimum_characters": True},
            "positive integer",
        ),
        (
            {"kind": "short_paragraphs", "minimum_characters": 0},
            "positive integer",
        ),
    )
    for spec, message in invalid:
        result = parse_filter_table("f", {"filters": [spec]})
        assert isinstance(result, Err)
        assert message in describe(result.error)


def test_resolve_filter_settings_uses_selected_layer_then_user_layer() -> None:
    user: DocumentResult = Ok(
        {"filter": {"filters": [{"kind": "short_paragraphs", "minimum_characters": 3}]}}
    )
    selected: DocumentResult = Ok({"filter": {"script": ["python3", "selected.py"]}})
    layered = resolve_filter_settings("user", user, "selected", selected)
    assert layered == Ok(FilterSettings(script=("python3", "selected.py")))
    assert resolve_filter_settings("user", user, None, Ok({})) == Ok(
        FilterSettings(filters=(FilterSpec("short_paragraphs", minimum_characters=3),))
    )
    assert resolve_filter_settings("user", Ok({}), None, Ok({})) == Ok(None)


def test_resolve_filter_settings_propagates_document_and_parse_errors() -> None:
    failure: DocumentResult = Err(ConfigError("bad document"))
    assert resolve_filter_settings("user", failure, None, Ok({})) == failure
    assert resolve_filter_settings("user", Ok({}), None, failure) == failure
    invalid = resolve_filter_settings(
        "user", Ok({"filter": {"filters": ["unknown"]}}), None, Ok({})
    )
    assert isinstance(invalid, Err)
    assert "duplicate_paragraphs" in describe(invalid.error)

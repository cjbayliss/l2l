from l2l.errors import (
    AsciiCollapseError,
    AsciiError,
    BudgetError,
    ConfigError,
    HttpError,
    InvalidOutputError,
    MissingSettings,
    PassError,
    TranslationError,
    UnitError,
    UntranslatedError,
    describe,
    fail_ascii,
    fail_ascii_collapse,
    fail_budget,
    fail_config,
    fail_http,
    fail_invalid_output,
    fail_missing_settings,
    fail_pass,
    fail_unit,
    fail_untranslated,
)
from l2l.monads import Err


def test_fail_config_and_missing_settings() -> None:
    failure = fail_config("config file not found: x.toml")
    assert isinstance(failure, Err)
    assert failure.error == ConfigError("config file not found: x.toml")
    assert describe(failure.error).startswith("l2l: config file not found")

    missing = fail_missing_settings(("api.model", "api.api_key"))
    assert isinstance(missing, Err)
    assert missing.error == MissingSettings(("api.model", "api.api_key"))
    assert "api.model, api.api_key" in describe(missing.error)


def test_describe_http_and_budget_errors() -> None:
    descriptions = {
        describe(HttpError("status", "bad request", 400)): "HTTP 400",
        describe(HttpError("unreachable", "connection reset")): "could not reach",
        describe(HttpError("stream", "rate limited")): "stream error",
        describe(HttpError("interrupted", "timed out")): "interrupted",
        describe(HttpError("protocol", "invalid JSON response: x")): "invalid JSON",
    }
    for text, fragment in descriptions.items():
        assert fragment in text

    single = describe(BudgetError(1200, 1000))
    assert "1200" in single
    assert "1000-token budget" in single
    assert "api.max_tokens" in single
    analysis = describe(BudgetError(1200, 1000, parts=3))
    assert analysis.startswith("document requires analysis in 3 parts")
    assert "1000-token budget" in analysis


def test_nested_errors_keep_their_context() -> None:
    inner: TranslationError = fail_http("unreachable", "down").error
    unit = UnitError("translate", 1, inner)
    assert "failed on unit 2" in describe(unit)
    assert "could not reach endpoint: down" in describe(unit)

    wrapped: TranslationError = fail_pass(
        "prep", fail_unit("prep", 0, inner).error
    ).error
    assert isinstance(wrapped, PassError)
    assert "pass [prep] failed" in describe(wrapped)
    assert "failed on unit 1" in describe(wrapped)


def test_describe_ascii_untranslated_and_invalid_output() -> None:
    inner: TranslationError = fail_budget(500, 100).error
    failure = fail_ascii("translate", inner)
    assert isinstance(failure, Err)
    assert isinstance(failure.error, AsciiError)
    assert "ascii enforcement failed" in describe(failure.error)
    assert "request is ~500" in describe(failure.error)

    collapse = fail_ascii_collapse("refine", 40, 40).error
    assert collapse == AsciiCollapseError("refine", 40, 40)
    wrapped = fail_ascii("refine", collapse)
    assert isinstance(wrapped.error, AsciiError)
    assert "40 of 40 paragraph(s)" in describe(wrapped.error)
    assert "untranslated text" in describe(wrapped.error)

    untranslated = fail_untranslated("translate", 2, "你好。")
    assert isinstance(untranslated, Err)
    assert isinstance(untranslated.error, UntranslatedError)
    assert "paragraph 3 is still untranslated" in describe(untranslated.error)
    assert "你好。" in describe(untranslated.error)

    invalid = fail_invalid_output("translate", 2, "the output failed validation")
    assert isinstance(invalid, Err)
    assert isinstance(invalid.error, InvalidOutputError)
    assert invalid.error.pass_name == "translate"
    assert "still failed validation after 2 repair attempt(s)" in describe(
        invalid.error
    )
    assert "--best-effort" in describe(invalid.error)


def test_constructors_build_expected_variants() -> None:
    assert fail_http("status", "nope", 500).error == HttpError("status", "nope", 500)
    assert fail_pass("p", fail_config("x").error).error == PassError(
        "p", ConfigError("x")
    )
    assert fail_unit("p", 3, fail_config("x").error).error == UnitError(
        "p", 3, ConfigError("x")
    )

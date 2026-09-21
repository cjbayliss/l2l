from zh2en.errors import (
    AsciiError,
    BudgetError,
    ConfigError,
    HttpError,
    MissingSettings,
    PassError,
    TranslationError,
    UnitError,
    describe,
    fail_ascii,
    fail_budget,
    fail_config,
    fail_http,
    fail_missing_settings,
    fail_pass,
    fail_unit,
)
from zh2en.monads import Err


def test_fail_config_wraps_message_with_prefix() -> None:
    failure = fail_config("config file not found: x.toml")
    assert isinstance(failure, Err)
    assert failure.error == ConfigError("config file not found: x.toml")
    assert describe(failure.error) == "zh2en: config file not found: x.toml"


def test_fail_missing_settings_renders_field_list() -> None:
    failure = fail_missing_settings(("api.model", "api.api_key"))
    assert isinstance(failure, Err)
    assert failure.error == MissingSettings(("api.model", "api.api_key"))
    assert describe(failure.error) == (
        "zh2en: missing required API settings: api.model, api.api_key"
    )


def test_describe_http_kinds() -> None:
    assert describe(HttpError("status", "bad request", 400)) == (
        "HTTP 400 from endpoint: bad request"
    )
    assert describe(HttpError("unreachable", "connection reset")) == (
        "could not reach endpoint: connection reset"
    )
    assert describe(HttpError("stream", "rate limited")) == (
        "endpoint stream error: rate limited"
    )
    assert describe(HttpError("interrupted", "timed out")) == (
        "stream interrupted: timed out"
    )
    assert describe(HttpError("protocol", "invalid JSON response: x")) == (
        "invalid JSON response: x"
    )


def test_describe_budget_single_request() -> None:
    text = describe(BudgetError(1200, 1000))
    assert text == (
        "request is ~1200 tokens, over the 1000-token budget; raise "
        "api.max_tokens (--max-tokens / TRANSLATE_MAX_TOKENS) or shorten "
        "the input"
    )


def test_describe_budget_analysis_parts() -> None:
    text = describe(BudgetError(1200, 1000, parts=3))
    assert text.startswith(
        "document requires analysis in 3 parts, over the 1000-token budget"
    )


def test_describe_wraps_nested_errors_without_double_prefix() -> None:
    inner: TranslationError = fail_http("unreachable", "down").error
    unit = UnitError("translate", 2, inner)
    assert describe(unit) == (
        "zh2en: [translate] failed on unit 2: could not reach endpoint: down"
    )

    wrapped: TranslationError = fail_pass(
        "prep", fail_unit("prep", 1, inner).error
    ).error
    assert isinstance(wrapped, PassError)
    assert describe(wrapped) == (
        "zh2en: pass [prep] failed: zh2en: [prep] failed on unit 1: "
        "could not reach endpoint: down"
    )


def test_describe_ascii_error() -> None:
    inner: TranslationError = fail_budget(500, 100).error
    failure = fail_ascii("translate", inner)
    assert isinstance(failure, Err)
    assert isinstance(failure.error, AsciiError)
    assert describe(failure.error).startswith(
        "zh2en: pass [translate] ascii enforcement failed: request is ~500"
    )


def test_constructors_build_expected_variants() -> None:
    assert fail_http("status", "nope", 500).error == HttpError("status", "nope", 500)
    assert fail_pass("p", fail_config("x").error).error == PassError(
        "p", ConfigError("x")
    )
    assert fail_unit("p", 3, fail_config("x").error).error == UnitError(
        "p", 3, ConfigError("x")
    )

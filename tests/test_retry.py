from zh2en.errors import BudgetError, ConfigError, HttpError, describe, fail_http
from zh2en.plans import plan_backoff, retry_delay, transient


def test_transient_kinds() -> None:
    assert transient(HttpError("unreachable", "down"))
    assert transient(HttpError("status", "slow down", 429))
    assert transient(HttpError("status", "boom", 503))
    assert not transient(HttpError("status", "nope", 400))
    assert not transient(HttpError("protocol", "bad json"))
    assert not transient(HttpError("stream", "mid-stream"))
    assert not transient(HttpError("interrupted", "cut"))
    assert not transient(ConfigError("bad config"))
    assert not transient(BudgetError(10, 5))


def test_plan_backoff_exponential_and_capped() -> None:
    assert plan_backoff(1.0, 30.0, 2) == (1.0, 2.0)
    assert plan_backoff(1.0, 1.5, 5) == (1.0, 1.5, 1.5, 1.5, 1.5)
    assert plan_backoff(1.0, 30.0, 0) == ()
    assert plan_backoff(0.5, 30.0, 3) == (0.5, 1.0, 2.0)


def test_retry_delay_honours_retry_after() -> None:
    assert retry_delay(HttpError("status", "x", 429, 7.0), 1.0) == 7.0
    assert retry_delay(HttpError("status", "x", 429, 0.5), 1.0) == 1.0
    assert retry_delay(HttpError("unreachable", "x"), 2.0) == 2.0


def test_describe_renders_retry_after_error() -> None:
    failure = fail_http("status", "too many", 429, 3.0).error
    assert isinstance(failure, HttpError)
    assert describe(failure) == "HTTP 429 from endpoint: too many"

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from zh2en.monads import Err


@dataclass(frozen=True)
class ConfigError:
    """A configuration, environment, or setup problem."""

    message: str


@dataclass(frozen=True)
class MissingSettings:
    """Required API settings absent from every configuration layer."""

    fields: tuple[str, ...]


HttpKind = Literal["status", "unreachable", "protocol", "stream", "interrupted"]


@dataclass(frozen=True)
class HttpError:
    """A transport or protocol failure against the chat endpoint."""

    kind: HttpKind
    detail: str
    status: int | None = None


@dataclass(frozen=True)
class BudgetError:
    """A request or document exceeding the configured token budget."""

    request_tokens: int
    budget: int
    parts: int = 1


@dataclass(frozen=True)
class PassError:
    """A pass failed; the inner error is the underlying cause."""

    pass_name: str
    inner: TranslationError


@dataclass(frozen=True)
class UnitError:
    """A unit call failed; the inner error is the underlying cause."""

    pass_name: str
    unit_index: int
    inner: TranslationError


@dataclass(frozen=True)
class AsciiError:
    """ASCII enforcement failed; the inner error is the underlying cause."""

    pass_name: str
    inner: TranslationError


type SetupError = ConfigError | MissingSettings
type TranslationError = (
    SetupError | HttpError | BudgetError | PassError | UnitError | AsciiError
)


def fail_config(message: str) -> Err[TranslationError]:
    return Err(ConfigError(message))


def fail_missing_settings(fields: tuple[str, ...]) -> Err[TranslationError]:
    return Err(MissingSettings(fields))


def fail_http(
    kind: HttpKind, detail: str, status: int | None = None
) -> Err[TranslationError]:
    return Err(HttpError(kind, detail, status))


def fail_budget(
    request_tokens: int, budget: int, parts: int = 1
) -> Err[TranslationError]:
    return Err(BudgetError(request_tokens, budget, parts))


def fail_pass(pass_name: str, inner: TranslationError) -> Err[TranslationError]:
    return Err(PassError(pass_name, inner))


def fail_unit(
    pass_name: str, unit_index: int, inner: TranslationError
) -> Err[TranslationError]:
    return Err(UnitError(pass_name, unit_index, inner))


def fail_ascii(pass_name: str, inner: TranslationError) -> Err[TranslationError]:
    return Err(AsciiError(pass_name, inner))


def describe_http(error: HttpError) -> str:
    if error.kind == "status":
        return "HTTP %s from endpoint: %s" % (error.status, error.detail)

    if error.kind == "unreachable":
        return "could not reach endpoint: %s" % error.detail

    if error.kind == "stream":
        return "endpoint stream error: %s" % error.detail

    if error.kind == "interrupted":
        return "stream interrupted: %s" % error.detail

    return error.detail


def describe_budget(error: BudgetError) -> str:
    subject = (
        "document requires analysis in %d parts" % error.parts
        if error.parts > 1
        else "request is ~%d tokens" % error.request_tokens
    )
    return (
        "%s, over the %d-token budget; raise api.max_tokens "
        "(--max-tokens / TRANSLATE_MAX_TOKENS) or shorten the input"
        % (subject, error.budget)
    )


def describe(error: TranslationError) -> str:
    if isinstance(error, ConfigError):
        return "zh2en: " + error.message

    if isinstance(error, MissingSettings):
        return "zh2en: missing required API settings: " + ", ".join(error.fields)

    if isinstance(error, HttpError):
        return describe_http(error)

    if isinstance(error, BudgetError):
        return describe_budget(error)

    if isinstance(error, PassError):
        return "zh2en: pass [%s] failed: %s" % (error.pass_name, describe(error.inner))

    if isinstance(error, UnitError):
        return "zh2en: [%s] failed on unit %d: %s" % (
            error.pass_name,
            error.unit_index,
            describe(error.inner),
        )

    if isinstance(error, AsciiError):
        return "zh2en: pass [%s] ascii enforcement failed: %s" % (
            error.pass_name,
            describe(error.inner),
        )

    assert_never(error)

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never

from l2l.monads import Err


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
    retry_after: float | None = None


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


@dataclass(frozen=True)
class UntranslatedError:
    """A paragraph is still untranslated after retranslation."""

    pass_name: str
    index: int
    sample: str


type SetupError = ConfigError | MissingSettings
type TranslationError = (
    SetupError
    | HttpError
    | BudgetError
    | PassError
    | UnitError
    | AsciiError
    | UntranslatedError
)


def fail_config(message: str) -> Err[TranslationError]:
    return Err(ConfigError(message))


def fail_missing_settings(fields: tuple[str, ...]) -> Err[TranslationError]:
    return Err(MissingSettings(fields))


def fail_http(
    kind: HttpKind,
    detail: str,
    status: int | None = None,
    retry_after: float | None = None,
) -> Err[TranslationError]:
    return Err(HttpError(kind, detail, status, retry_after))


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


def fail_untranslated(pass_name: str, index: int, sample: str) -> Err[TranslationError]:
    return Err(UntranslatedError(pass_name, index, sample))


def describe_http(error: HttpError) -> str:
    match error.kind:
        case "status":
            return "HTTP %s from endpoint: %s" % (error.status, error.detail)

        case "unreachable":
            return "could not reach endpoint: %s" % error.detail

        case "stream":
            return "endpoint stream error: %s" % error.detail

        case "interrupted":
            return "stream interrupted: %s" % error.detail

        case "protocol":
            return error.detail

        case other:
            assert_never(other)


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
    match error:
        case ConfigError():
            return "l2l: " + error.message

        case MissingSettings():
            return "l2l: missing required API settings: " + ", ".join(error.fields)

        case HttpError():
            return describe_http(error)

        case BudgetError():
            return describe_budget(error)

        case PassError():
            return "l2l: pass [%s] failed: %s" % (
                error.pass_name,
                describe(error.inner),
            )

        case UnitError():
            return "l2l: [%s] failed on unit %d: %s" % (
                error.pass_name,
                error.unit_index,
                describe(error.inner),
            )

        case AsciiError():
            return "l2l: pass [%s] ascii enforcement failed: %s" % (
                error.pass_name,
                describe(error.inner),
            )

        case UntranslatedError():
            return (
                "l2l: pass [%s] paragraph %d is still untranslated after "
                "retranslation: %r" % (error.pass_name, error.index + 1, error.sample)
            )

        case other:
            assert_never(other)

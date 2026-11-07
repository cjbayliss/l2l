from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, assert_never, final

from l2l.monads import Err


@final
@dataclass(frozen=True, slots=True)
class ConfigError:
    message: str


@final
@dataclass(frozen=True, slots=True)
class MissingSettings:
    fields: tuple[str, ...]


HttpKind = Literal["status", "unreachable", "protocol", "stream", "interrupted"]


@final
@dataclass(frozen=True, slots=True)
class HttpError:
    kind: HttpKind
    detail: str
    status: int | None = None
    retry_after: float | None = None


@final
@dataclass(frozen=True, slots=True)
class BudgetError:
    request_tokens: int
    budget: int
    parts: int = 1


@final
@dataclass(frozen=True, slots=True)
class PassError:
    pass_name: str
    inner: TranslationError


@final
@dataclass(frozen=True, slots=True)
class UnitError:
    pass_name: str
    unit_index: int
    inner: TranslationError


@final
@dataclass(frozen=True, slots=True)
class AsciiError:
    pass_name: str
    inner: TranslationError


@final
@dataclass(frozen=True, slots=True)
class AsciiCollapseError:
    pass_name: str
    stubborn: int
    total: int


@final
@dataclass(frozen=True, slots=True)
class UntranslatedError:
    pass_name: str
    index: int
    sample: str


@final
@dataclass(frozen=True, slots=True)
class InvalidOutputError:
    pass_name: str
    attempts: int
    problem: str


@final
@dataclass(frozen=True, slots=True)
class GlossaryReplyError:
    pass_name: str
    problem: str


@final
@dataclass(frozen=True, slots=True)
class FileError:
    path: str
    operation: str
    detail: str


type SetupError = ConfigError | MissingSettings
type TranslationError = (
    SetupError
    | HttpError
    | BudgetError
    | PassError
    | UnitError
    | AsciiError
    | AsciiCollapseError
    | UntranslatedError
    | InvalidOutputError
    | GlossaryReplyError
    | FileError
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


def fail_ascii_collapse(
    pass_name: str, stubborn: int, total: int
) -> Err[TranslationError]:
    return Err(AsciiCollapseError(pass_name, stubborn, total))


def fail_untranslated(pass_name: str, index: int, sample: str) -> Err[TranslationError]:
    return Err(UntranslatedError(pass_name, index, sample))


def fail_invalid_output(
    pass_name: str, attempts: int, problem: str
) -> Err[TranslationError]:
    return Err(InvalidOutputError(pass_name, attempts, problem))


def fail_file_error(path: str, operation: str, detail: str) -> Err[TranslationError]:
    return Err(FileError(path, operation, detail))


def describe_http(error: HttpError) -> str:
    match error.kind:
        case "status":
            return f"HTTP {error.status} from endpoint: {error.detail}"

        case "unreachable":
            return f"could not reach endpoint: {error.detail}"

        case "stream":
            return f"endpoint stream error: {error.detail}"

        case "interrupted":
            return f"stream interrupted: {error.detail}"

        case "protocol":
            return error.detail

        case _:
            assert_never(error.kind)


def describe_budget(error: BudgetError) -> str:
    subject = (
        f"document requires analysis in {error.parts} parts"
        if error.parts > 1
        else f"request is ~{error.request_tokens} tokens"
    )
    return (
        f"{subject}, over the {error.budget}-token budget; raise api.max_tokens "
        "(--maximum-tokens / TRANSLATE_MAX_TOKENS) or shorten the input"
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
            return f"l2l: pass [{error.pass_name}] failed: {describe(error.inner)}"

        case UnitError():
            return (
                f"l2l: [{error.pass_name}] failed on unit {error.unit_index + 1}: "
                f"{describe(error.inner)}"
            )

        case AsciiError():
            return (
                f"l2l: pass [{error.pass_name}] ascii enforcement failed: "
                f"{describe(error.inner)}"
            )

        case AsciiCollapseError():
            return (
                f"{error.stubborn} of {error.total} paragraph(s) would need LLM ASCII "
                "repair; an upstream pass probably returned untranslated text "
                "instead of translating"
            )

        case UntranslatedError():
            return (
                f"l2l: pass [{error.pass_name}] paragraph {error.index + 1} is still "
                f"untranslated after retranslation: {error.sample!r}"
            )

        case GlossaryReplyError():
            return (
                f"l2l: [{error.pass_name}] the model reply is not valid glossary "
                f"output: {error.problem}"
            )

        case InvalidOutputError():
            return (
                f"l2l: pass [{error.pass_name}] output still failed validation after "
                f"{error.attempts} repair attempt(s): {error.problem}; pass "
                "--best-effort to keep the last reply instead"
            )

        case FileError():
            return f"l2l: could not {error.operation} {error.path}: {error.detail}"

        case _:
            assert_never(error)

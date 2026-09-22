"""Frozen configuration data, defaults, and small pure accessors.

`settings` is the data vocabulary of the package; `config` builds it from
TOML, environment, and arguments. Dependencies point downward only.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, Protocol

from zh2en.console import Console
from zh2en.effects import Clock, RunLog, Sleep
from zh2en.errors import TranslationError
from zh2en.monads import Result
from zh2en.text import CACHE_SALT_VERSION


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    model: str
    timeout: float
    max_tokens: int
    params: Mapping[str, Any]


PassMode = Literal["analysis", "chunk", "paragraph"]


@dataclass(frozen=True)
class PassDefinition:
    name: str
    instruction: str
    mode: PassMode
    params: Mapping[str, Any]
    model: str | None
    ascii: bool | None
    ensure_paragraphs: bool | None = None


@dataclass(frozen=True)
class Settings:
    chunk_budget_tokens: int
    analysis_reserve_tokens: int
    ascii_fix_attempts: int
    sentence_boundary_characters: str
    ascii_fix_instruction: str
    ascii_character_map: Mapping[str, str]
    unit_fix_attempts: int
    unit_output_max_ratio: float
    retry_attempts: int
    retry_base_delay: float
    retry_cap: float


@dataclass(frozen=True)
class Arguments:
    config: str | None
    base_url: str | None
    api_key: str | None
    model: str | None
    timeout: float | None
    max_tokens: int | None
    no_cache: bool
    ensure_paragraphs: bool
    verbose: bool
    show_log_path: bool
    cache_dir: str | None
    check_config: bool = False
    dry_run: bool = False
    stream: bool | None = None
    cache_prune: int | None = None
    log_keep: int = 30


@dataclass(frozen=True)
class Setup:
    config: Config
    passes: tuple[PassDefinition, ...]
    ensure_paragraphs: bool


class HttpResponse(Protocol):
    """The slice of an endpoint response the package consumes: full-body
    reads for plain calls, line iteration for SSE streams."""

    def read(self, amount: int = -1) -> bytes: ...

    def __iter__(self) -> Iterator[bytes]: ...

    def __enter__(self) -> HttpResponse: ...

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> object: ...


OpenHTTP = Callable[[Any, float], Result[HttpResponse, TranslationError]]


@dataclass(frozen=True)
class Context:
    config: Config
    settings: Settings
    use_cache: bool
    cache_directory: str
    console: Console
    open_http: OpenHTTP
    log: RunLog
    clock: Clock
    sleep: Sleep
    stream: bool | None = None


def build_settings() -> Settings:
    return Settings(
        chunk_budget_tokens=3500,
        analysis_reserve_tokens=128,
        ascii_fix_attempts=3,
        unit_fix_attempts=2,
        unit_output_max_ratio=6.0,
        retry_attempts=2,
        retry_base_delay=1.0,
        retry_cap=30.0,
        sentence_boundary_characters="。！？…⋯!?；;\n؟؛।॥։።",
        ascii_fix_instruction="""
This paragraph failed to be fully translated or contains non-ASCII
characters. Please analyse it and only output a clean translation that
uses ASCII characters only.
""",
        ascii_character_map=MappingProxyType(
            {
                "\u00a0": " ",
                "\u2007": " ",
                "\u2009": " ",
                "\u202f": " ",
                "\u2018": "'",
                "\u2019": "'",
                "\u201a": ",",
                "\u201b": "'",
                "\u201c": '"',
                "\u201d": '"',
                "\u201e": '"',
                "\u201f": '"',
                "\u2032": "'",
                "\u2033": '"',
                "\u2039": "'",
                "\u203a": "'",
                "\u00ab": '"',
                "\u00bb": '"',
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2015": "-",
                "\u2212": "-",
                "\u2026": "...",
                "\u2025": "..",
                "\u2022": "*",
                "\u2023": "*",
                "\u2024": "*",
                "\u2219": "*",
                "\u00b7": " ",
                "\u2190": "<-",
                "\u2192": "->",
                "\u2194": "<->",
                "\u2264": "<=",
                "\u2265": ">=",
                "\u2260": "!=",
                "\u3002": ".",
                "\u3001": ",",
                "\u300c": '"',
                "\u300d": '"',
                "\u300e": '"',
                "\u300f": '"',
                "\u3008": "<",
                "\u3009": ">",
                "\u300a": '"',
                "\u300b": '"',
                "\u3010": "[",
                "\u3011": "]",
            }
        ),
    )


API_SETTING_KEYS = ("base_url", "api_key", "model", "timeout", "max_tokens", "params")
PASS_KEYS = (
    "name",
    "instruction",
    "instruction_file",
    "mode",
    "ascii",
    "ensure_paragraphs",
    "model",
    "params",
)


DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_TOKENS = 100000


@dataclass(frozen=True)
class PartialApiSettings:
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    timeout: float | None = None
    max_tokens: int | None = None
    params: Mapping[str, Any] | None = None

    def merge(self, extra: PartialApiSettings) -> PartialApiSettings:
        if self.params is not None and extra.params is not None:
            params: Mapping[str, Any] | None = {**self.params, **extra.params}
        elif extra.params is None:
            params = self.params
        else:
            params = extra.params

        return PartialApiSettings(
            base_url=extra.base_url if extra.base_url is not None else self.base_url,
            api_key=extra.api_key if extra.api_key is not None else self.api_key,
            model=extra.model if extra.model is not None else self.model,
            timeout=extra.timeout if extra.timeout is not None else self.timeout,
            max_tokens=(
                extra.max_tokens if extra.max_tokens is not None else self.max_tokens
            ),
            params=params,
        )


DEFAULT_API_SETTINGS = PartialApiSettings(
    base_url="",
    api_key="",
    model="",
    timeout=DEFAULT_TIMEOUT,
    max_tokens=DEFAULT_MAX_TOKENS,
    params=MappingProxyType({}),
)


def resolve_call_settings(
    config: Config, pass_definition: PassDefinition
) -> tuple[str, dict[str, Any]]:
    return (pass_definition.model or config.model), {
        **dict(config.params),
        **dict(pass_definition.params),
    }


def salt(*parts: str) -> str:
    """Join cache-key salt segments with the NUL separator."""
    return "\x00".join(parts)


def pass_salt(pass_definition: PassDefinition) -> str:
    return salt(CACHE_SALT_VERSION, pass_definition.name, pass_definition.instruction)

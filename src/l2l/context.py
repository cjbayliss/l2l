from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, final

from l2l.errors import TranslationError
from l2l.monads import IO, Reference, Result
from l2l.settings import UNSET, Unset
lazy from l2l.console import Console
lazy from l2l.glossary import GlossaryEntry

if TYPE_CHECKING:
    from l2l.effects import Clock, RunLog, Sleep
    from l2l.settings import Config, Settings


class HttpStreamResponse(Protocol):
    def consume[S](
        self,
        step: Callable[[S, bytes], IO[Result[S, TranslationError]]],
        initial: Result[S, TranslationError],
    ) -> IO[Result[S, TranslationError]]: ...


class HttpResponse(HttpStreamResponse, Protocol):
    def body(self) -> IO[Result[bytes, TranslationError]]: ...


@final
@dataclass(frozen=True, slots=True)
class HttpRequest:
    url: str
    body: bytes
    headers: Mapping[str, str]


OpenHTTP = Callable[[HttpRequest, float], Result[HttpResponse, TranslationError]]


@final
@dataclass(frozen=True, kw_only=True, slots=True)
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
    target_language: str
    source_language: str
    stream: bool | Unset = UNSET
    best_effort: bool = False
    glossary_entries: tuple[GlossaryEntry, ...] = ()
    glossary_update_path: str | None = None
    no_structured_output: bool = False
    provider_blocklist: Reference[frozenset[str]] = field(
        default_factory=lambda: Reference(frozenset[str]())
    )

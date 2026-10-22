from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from l2l.console import Console
from l2l.effects import Clock, RunLog, Sleep
from l2l.errors import TranslationError
from l2l.glossary import GlossaryEntry
from l2l.monads import IO, Result
from l2l.settings import Config, Settings


class HttpResponse(Protocol):
    def body(self) -> IO[Result[bytes, TranslationError]]: ...

    def consume(
        self,
        step: Callable[[Any, bytes], IO[Result[Any, TranslationError]]],
        initial: Result[Any, TranslationError],
    ) -> IO[Result[Any, TranslationError]]: ...


OpenHTTP = Callable[[Any, float], Result[HttpResponse, TranslationError]]


@dataclass(frozen=True, kw_only=True)
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
    stream: bool | None = None
    best_effort: bool = False
    glossary_entries: tuple[GlossaryEntry, ...] = ()
    glossary_update_path: str | None = None

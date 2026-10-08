from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from l2l.effects import ProcessResult, Sleep
from l2l.errors import TranslationError
from l2l.monads import IO, Result

from .configuration import BaseConfig
from .options import Options
from .replies import EndpointReply

type OpenEndpoint = Callable[
    [Mapping[str, Any]], IO[Result[EndpointReply, TranslationError]]
]
type Reporter = Callable[[str], IO[None]]
type RunCommand = Callable[[tuple[str, ...], str], IO[ProcessResult]]


@dataclass(frozen=True)
class Api:
    base_url: str
    api_key: str


@dataclass(frozen=True)
class OptimizationEnvironment:
    options: Options
    base: BaseConfig
    api: Api
    working_directory: str
    environment: Mapping[str, str]
    chapters: tuple[str, ...]
    judge_template: str
    rewrite_template: str
    l2l_command: tuple[str, ...]
    opener: OpenEndpoint
    sleep: Sleep
    say: Reporter
    warn: Reporter
    run_command: RunCommand

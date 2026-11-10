from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import final


@final
class ExitKind(Enum):
    SUCCESS = "success"
    RUNTIME_ERROR = "runtime_error"
    USAGE_ERROR = "usage_error"
    INTERRUPTED = "interrupted"
    BROKEN_PIPE = "broken_pipe"


@final
@dataclass(frozen=True, slots=True)
class ExitState:
    kind: ExitKind
    message: str | None = None


def exit_code(kind: ExitKind) -> int:
    match kind:
        case ExitKind.SUCCESS:
            return 0
        case ExitKind.RUNTIME_ERROR:
            return 1
        case ExitKind.USAGE_ERROR:
            return 2
        case ExitKind.INTERRUPTED:
            return 130
        case ExitKind.BROKEN_PIPE:
            return 141

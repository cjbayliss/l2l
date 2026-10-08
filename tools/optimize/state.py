from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SEED_VERSION = "v0"


@dataclass(frozen=True)
class RoundInputs:
    current_prompt: str
    candidate_prompt: str
    digest: str


@dataclass(frozen=True)
class RunState:
    round_no: int
    incumbent: str
    stall: int
    history: tuple[dict[str, Any], ...]

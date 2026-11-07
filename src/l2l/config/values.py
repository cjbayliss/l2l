from __future__ import annotations

from typing import TypeIs

lazy from l2l.settings import ParagraphTriState


def is_paragraph_tri_state(value: object) -> TypeIs[ParagraphTriState]:
    match value:
        case bool():
            return True

        case int():
            return value > 0

        case _:
            return False

from __future__ import annotations

from typing import Any


def is_json_value(value: Any) -> bool:
    match value:
        case bool() | int() | float() | str() | None:
            return True

        case list():
            return all(is_json_value(item) for item in value)

        case dict():
            return all(is_json_value(item) for item in value.values())

        case _:
            return False


def is_paragraph_tri_state(value: Any) -> bool:
    match value:
        case bool():
            return True

        case int():
            return value > 0

        case _:
            return False

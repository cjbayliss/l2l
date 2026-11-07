from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeIs

type JsonArray = Sequence[JsonValue]
type JsonObject = Mapping[str, JsonValue]
type JsonValue = bool | int | float | str | JsonArray | JsonObject | None


def is_object_list(value: object) -> TypeIs[list[object]]:
    return isinstance(value, list)


def is_object_table(value: object) -> TypeIs[Mapping[object, object]]:
    return isinstance(value, dict)


def is_string_keyed_table(value: object) -> TypeIs[Mapping[str, object]]:
    if not is_object_table(value):
        return False

    return all(isinstance(key, str) for key in value)


def is_json_value(value: object) -> TypeIs[JsonValue]:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True

    if is_object_list(value):
        return all(is_json_value(item) for item in value)

    if not is_object_table(value):
        return False

    return all(is_json_value(item) for item in value.values())


def is_json_object(value: object) -> TypeIs[JsonObject]:
    if not is_string_keyed_table(value):
        return False

    return all(is_json_value(item) for item in value.values())


def is_json_array(value: object) -> TypeIs[JsonArray]:
    if not is_object_list(value):
        return False

    return all(is_json_value(item) for item in value)


def is_non_empty_string(value: object) -> TypeIs[str]:
    return isinstance(value, str) and bool(value.strip())


def is_string_array(value: object) -> TypeIs[list[str]]:
    if not is_object_list(value):
        return False

    return all(isinstance(item, str) for item in value)


def is_non_empty_string_array(value: object) -> TypeIs[list[str]]:
    if not is_object_list(value):
        return False

    return all(is_non_empty_string(item) for item in value)


def is_non_empty_table_array(value: object) -> TypeIs[list[Mapping[str, object]]]:
    if not is_object_list(value):
        return False

    if len(value) == 0:
        return False

    return all(isinstance(item, dict) for item in value)

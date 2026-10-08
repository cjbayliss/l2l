from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from l2l.errors import TranslationError, fail_config
from l2l.monads import Ok, Result, result_bind, result_map, result_sequence


@dataclass(frozen=True)
class BaseConfig:
    api: Mapping[str, Any]
    options: Mapping[str, Any]
    passes: tuple[Mapping[str, Any], ...]
    target_index: int
    target_name: str


def toml_value(value: Any) -> Result[str, TranslationError]:
    if isinstance(value, bool):
        return Ok("true" if value else "false")
    if isinstance(value, str):
        return Ok(json.dumps(value))
    if isinstance(value, (int, float)):
        return Ok(repr(value))
    if isinstance(value, list):
        items = result_sequence(tuple(toml_value(item) for item in value))
        return result_map(items, lambda parts: "[" + ", ".join(parts) + "]")
    return fail_config("unsupported config value: %r" % (value,))


def emit_body(
    prefix: str, table: Mapping[str, Any]
) -> Result[tuple[str, ...], TranslationError]:
    def flat_line(item: tuple[str, Any]) -> Result[tuple[str, ...], TranslationError]:
        key, value = item
        if isinstance(value, dict):
            return Ok(())
        return result_map(
            toml_value(value), lambda rendered: ("%s = %s" % (key, rendered),)
        )

    def child_block(item: tuple[str, Any]) -> Result[tuple[str, ...], TranslationError]:
        key, value = item
        if not isinstance(value, dict) or not value:
            return Ok(())
        child = "%s.%s" % (prefix, key)
        return result_map(
            emit_body(child, value),
            lambda lines: ("", "[%s]" % child) + lines,
        )

    def joined(
        flat: tuple[tuple[str, ...], ...], nested: tuple[tuple[str, ...], ...]
    ) -> tuple[str, ...]:
        return tuple(line for part in flat + nested for line in part)

    def with_nested(
        flat: tuple[tuple[str, ...], ...],
    ) -> Result[tuple[str, ...], TranslationError]:
        nested = result_sequence(tuple(child_block(item) for item in table.items()))
        return result_map(nested, lambda parts: joined(flat, parts))

    return result_bind(
        result_sequence(tuple(flat_line(item) for item in table.items())),
        with_nested,
    )


def render_config(
    api: Mapping[str, Any],
    options: Mapping[str, Any],
    passes: tuple[Mapping[str, Any], ...],
) -> Result[str, TranslationError]:
    def flatten(groups: tuple[tuple[str, ...], ...]) -> str:
        lines = tuple(line for group in groups for line in group)
        return "\n".join(lines) + "\n"

    def pass_groups() -> Result[tuple[tuple[str, ...], ...], TranslationError]:
        def one(entry: Mapping[str, Any]) -> Result[tuple[str, ...], TranslationError]:
            return result_map(
                emit_body("pass", entry), lambda lines: ("", "[[pass]]") + lines
            )

        return result_sequence(tuple(one(entry) for entry in passes))

    def finish(
        api_lines: tuple[str, ...],
        options_lines: tuple[str, ...] | None,
        pass_parts: tuple[tuple[str, ...], ...],
    ) -> str:
        groups: tuple[tuple[str, ...], ...] = (("[api]",), api_lines)
        if options_lines is not None:
            groups = groups + (("", "[options]"), options_lines)
        return flatten(groups + pass_parts)

    def with_pass_parts(
        api_lines: tuple[str, ...], options_lines: tuple[str, ...] | None
    ) -> Callable[[tuple[tuple[str, ...], ...]], Result[str, TranslationError]]:
        def taken(parts: tuple[tuple[str, ...], ...]) -> Result[str, TranslationError]:
            return Ok(finish(api_lines, options_lines, parts))

        return taken

    def with_api_lines(api_lines: tuple[str, ...]) -> Result[str, TranslationError]:
        if not options:
            return result_bind(pass_groups(), with_pass_parts(api_lines, None))

        def with_option_lines(
            lines: tuple[str, ...],
        ) -> Result[str, TranslationError]:
            return result_bind(pass_groups(), with_pass_parts(api_lines, lines))

        return result_bind(emit_body("options", options), with_option_lines)

    return result_bind(emit_body("api", api), with_api_lines)


def as_table(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def load_base_config(
    data: Mapping[str, Any], pass_name: str | None, path: str
) -> Result[BaseConfig, TranslationError]:
    passes = tuple(MappingProxyType(dict(entry)) for entry in data.get("pass", []))
    if not passes:
        return fail_config("base config %s has no [[pass]] entries" % path)

    def target(index: int) -> Result[BaseConfig, TranslationError]:
        entry = passes[index]
        return Ok(
            BaseConfig(
                api=MappingProxyType(as_table(data.get("api"))),
                options=MappingProxyType(as_table(data.get("options"))),
                passes=passes,
                target_index=index,
                target_name=str(entry.get("name") or "translate"),
            )
        )

    def single() -> Result[BaseConfig, TranslationError]:
        if len(passes) == 1:
            return target(0)
        return fail_config(
            "base config %s has %d passes; pass --pass-name to pick one"
            % (path, len(passes))
        )

    if pass_name is None:
        return single()

    matches = tuple(
        index for index, entry in enumerate(passes) if entry.get("name") == pass_name
    )
    if not matches:
        return fail_config("base config %s has no pass named %r" % (path, pass_name))
    return target(matches[0])

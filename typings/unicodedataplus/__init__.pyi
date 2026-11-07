from collections.abc import Mapping, Sequence

property_value_aliases: Mapping[str, Mapping[str, Sequence[str]]]

def script_extensions(character: str) -> list[str]: ...

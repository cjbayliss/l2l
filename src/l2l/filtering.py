from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal, final

from l2l.text import split_paragraphs

type FilterKind = Literal["duplicate_paragraphs", "pattern", "short_paragraphs"]

FILTER_KINDS: Final[tuple[FilterKind, ...]] = (
    "duplicate_paragraphs",
    "pattern",
    "short_paragraphs",
)


def parse_filter_kind(value: object) -> FilterKind | None:
    if isinstance(value, str) and value in FILTER_KINDS:
        return value

    return None


@final
@dataclass(frozen=True, slots=True)
class FilterSpec:
    kind: FilterKind
    pattern: re.Pattern[str] | None = None
    minimum_characters: int = 0


@final
@dataclass(frozen=True, slots=True)
class FilterSettings:
    script: tuple[str, ...] | None = None
    filters: tuple[FilterSpec, ...] = ()


@final
@dataclass(frozen=True, slots=True)
class FilterRemoval:
    index: int
    kind: FilterKind
    text: str


@final
@dataclass(frozen=True, slots=True)
class FilterOutcome:
    text: str
    paragraph_count: int
    removals: tuple[FilterRemoval, ...]


WHITESPACE_PATTERN: Final = re.compile(r"\s+")


def normalized_paragraph(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def duplicate_paragraph_removals(paragraphs: tuple[str, ...]) -> tuple[int, ...]:
    seen: set[str] = set()
    duplicates: list[int] = []
    for index, paragraph in enumerate(paragraphs):
        body = normalized_paragraph(paragraph)
        if body in seen:
            duplicates.append(index)
        else:
            seen.add(body)

    return tuple(duplicates)


def pattern_paragraph_removals(
    paragraphs: tuple[str, ...], pattern: re.Pattern[str]
) -> tuple[int, ...]:
    return tuple(
        index for index, paragraph in enumerate(paragraphs) if pattern.search(paragraph)
    )


def short_paragraph_removals(
    paragraphs: tuple[str, ...], minimum_characters: int
) -> tuple[int, ...]:
    return tuple(
        index
        for index, paragraph in enumerate(paragraphs)
        if len(normalized_paragraph(paragraph)) < minimum_characters
    )


def spec_removals(spec: FilterSpec, paragraphs: tuple[str, ...]) -> tuple[int, ...]:
    match spec.kind:
        case "duplicate_paragraphs":
            return duplicate_paragraph_removals(paragraphs)

        case "pattern":
            match spec.pattern:
                case None:
                    return ()

                case compiled:
                    return pattern_paragraph_removals(paragraphs, compiled)

        case "short_paragraphs":
            return short_paragraph_removals(paragraphs, spec.minimum_characters)


def filter_paragraphs(text: str, specs: tuple[FilterSpec, ...]) -> FilterOutcome:
    paragraphs, _separators = split_paragraphs(text)
    removed: dict[int, FilterKind] = {}
    removals: list[FilterRemoval] = []
    for spec in specs:
        for index in spec_removals(spec, paragraphs):
            if index in removed:
                continue

            removed[index] = spec.kind
            removals.append(FilterRemoval(index, spec.kind, paragraphs[index]))

    kept = tuple(
        paragraph for index, paragraph in enumerate(paragraphs) if index not in removed
    )
    return FilterOutcome("\n\n".join(kept), len(paragraphs), tuple(removals))


def removal_counts(
    removals: tuple[FilterRemoval, ...],
) -> tuple[tuple[FilterKind, int], ...]:
    counts: dict[FilterKind, int] = {}
    for removal in removals:
        counts[removal.kind] = counts.get(removal.kind, 0) + 1

    return tuple(counts.items())

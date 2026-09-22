"""Architectural guardrails for the functional core.

The package's design rules are enforced as tests: `monads` stands alone,
dependencies point strictly downward through the layer map, and `IO.run`
is executed only at the sanctioned edges.
"""

import ast
from itertools import chain
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "l2l"

# Bottom-to-top layers; a module may import only from strictly lower
# layers (the bare `l2l` root, which carries only `__version__`, may be
# imported from anywhere).
LAYERS: tuple[tuple[str, ...], ...] = (
    ("monads", "messages"),
    ("errors", "text"),
    ("console", "effects"),
    ("keys", "settings"),
    ("config", "plans"),
    ("cache", "http"),
    ("ascii",),
    ("pipeline",),
    ("cli",),
)

LEVELS: dict[str, int] = {
    name: level for level, names in enumerate(LAYERS) for name in names
}

# Where `IO.run` may appear: `cli` is the single program entry point and
# `monads` hosts the combinator runners (`io_atomic`, `io_using`,
# `repeat_until`). Every other module only composes IO values.
IO_EDGES = frozenset({"cli.py", "monads.py"})


def module_names() -> frozenset[str]:
    return frozenset(
        path.stem
        for path in PACKAGE.glob("*.py")
        if path.stem not in ("__init__", "__main__")
    )


def parse_module(name: str) -> ast.Module:
    source = (PACKAGE / (name + ".py")).read_text(encoding="utf-8")
    return ast.parse(source, filename=name + ".py")


def l2l_imports(tree: ast.Module) -> set[str]:
    """Package names imported by a module; "" is the `l2l` root itself."""
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            parts = (node.module or "").split(".")
            if parts[0] == "l2l":
                imported.add(parts[1] if len(parts) > 1 else "")

        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "l2l":
                    imported.add(parts[1] if len(parts) > 1 else "")

    return imported


def test_layers_cover_every_module() -> None:
    assert module_names() == frozenset(chain.from_iterable(LAYERS))


def test_monads_imports_nothing_from_the_package() -> None:
    assert l2l_imports(parse_module("monads")) == set()


def test_messages_imports_nothing() -> None:
    assert l2l_imports(parse_module("messages")) == set()


def test_dependencies_point_strictly_downward() -> None:
    for name in sorted(module_names()):
        for imported in sorted(l2l_imports(parse_module(name))):
            if imported == "":
                continue

            assert LEVELS[imported] < LEVELS[name], (
                f"{name} (level {LEVELS[name]}) imports "
                f"{imported} (level {LEVELS[imported]})"
            )


def test_io_runs_only_at_sanctioned_edges() -> None:
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name in IO_EDGES:
            continue

        assert ".run(" not in path.read_text(encoding="utf-8"), (
            f"{path.name} must not execute IO; compose it instead"
        )

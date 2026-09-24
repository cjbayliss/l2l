"""Architectural guardrails for the functional core.

The package's design rules are enforced as tests: `monads` stands alone,
dependencies point strictly downward through the layer map, `IO.run` is
executed only at the sanctioned edges, data is immutable, errors are
values, mutation is confined to `Ref`, and effectful stdlib modules
stay in their sanctioned homes.
"""

import ast
from collections.abc import Callable
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
IO_EDGES = frozenset({"cli", "monads"})

# The package's single sanctioned mutable cell: `monads.Ref`.
MUTABLE_DATACLASS_EXEMPT = frozenset({("monads", "Ref")})

# Functions allowed to assign to attributes or subscripts: the `Ref`
# operations, plus `io_memoize`, which writes through a `Ref`.
MUTATION_ASSIGNMENT_ALLOWLIST = frozenset(
    {
        ("monads", "io_memoize"),
        ("monads", "modify_ref"),
        ("monads", "modify_ref_with"),
        ("monads", "write_ref"),
    }
)

# Functions allowed to call container mutator methods: `cons_to_tuple`
# drains into a local accumulator; `run_log_write` invokes the `RunLog`'s
# injected append closure; `launch`/`start` start daemon threads.
MUTATOR_CALL_ALLOWLIST = frozenset(
    {
        ("console", "launch"),
        ("effects", "run_log_write"),
        ("keys", "start"),
        ("monads", "cons_to_tuple"),
    }
)

# `raise` is confined to the error edge: `cli` re-raises after handling;
# `errors` may raise while validating its own constructors' inputs.
RAISE_MODULES = frozenset({"cli", "errors"})

# Stdlib roots that perform I/O, concurrency, or global process state.
# Everything else (`json`, `re`, `functools`, ...) counts as pure.
EFFECT_MODULES = frozenset(
    {
        "http",
        "os",
        "select",
        "shutil",
        "signal",
        "socket",
        "subprocess",
        "sys",
        "termios",
        "threading",
        "time",
        "tomllib",
        "tty",
        "urllib",
    }
)

# Which effectful stdlib roots each module may import; modules absent
# from this map get an empty allowlist (deny by default).
EFFECT_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "cli": frozenset({"os", "sys", "time"}),
    "config": frozenset({"os"}),
    "console": frozenset({"os", "shutil", "sys", "threading", "time"}),
    "effects": frozenset({"os", "time", "tomllib"}),
    "http": frozenset({"http", "urllib"}),
    "keys": frozenset({"os", "select", "sys", "termios", "threading", "tty"}),
    "monads": frozenset({"threading"}),
    "text": frozenset({"os"}),
}

# Builtin calls that need no import, so the import allowlist cannot see
# them: `open` lives only in `effects`, `print` only in `cli` (the
# program edge); the rest are banned everywhere.
BUILTIN_CALL_EDGES: dict[str, frozenset[str]] = {
    "__import__": frozenset(),
    "breakpoint": frozenset(),
    "eval": frozenset(),
    "exec": frozenset(),
    "input": frozenset(),
    "open": frozenset({"effects"}),
    "print": frozenset({"cli"}),
}

# Container methods that mutate their receiver.
MUTATING_METHODS = frozenset(
    {
        "add",
        "append",
        "clear",
        "discard",
        "extend",
        "insert",
        "pop",
        "remove",
        "reverse",
        "setdefault",
        "sort",
        "update",
    }
)


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


def import_roots(tree: ast.Module) -> set[str]:
    """First segments of every absolute stdlib/external import."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".")[0])

    return roots


def module_aliases(tree: ast.Module) -> frozenset[str]:
    """Names bound to imports, so `os.remove` is not a container mutator."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                (alias.asname or alias.name).split(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            aliases.update(alias.asname or alias.name for alias in node.names)

    return frozenset(aliases)


def visit_scoped(
    node: ast.AST,
    on_node: Callable[[ast.AST, tuple[str, ...]], None],
    functions: tuple[str, ...] = (),
) -> None:
    """Visit `node` and its children, threading enclosing function names."""
    names = functions
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        names = functions + (node.name,)

    on_node(node, names)
    for child in ast.iter_child_nodes(node):
        visit_scoped(child, on_node, names)


def scoped_violations(
    name: str,
    tree: ast.Module,
    matches: Callable[[ast.stmt | ast.expr, frozenset[str]], bool],
    allowlist: frozenset[tuple[str, str]],
) -> list[ast.stmt | ast.expr]:
    """Nodes matching `matches` outside functions named in `allowlist`."""
    aliases = module_aliases(tree)
    found: list[ast.stmt | ast.expr] = []

    def on_node(node: ast.AST, functions: tuple[str, ...]) -> None:
        if not isinstance(node, (ast.stmt, ast.expr)) or not matches(node, aliases):
            return

        if not any((name, function) in allowlist for function in functions):
            found.append(node)

    visit_scoped(tree, on_node)
    return found


def decorator_root(decorator: ast.expr) -> str:
    if isinstance(decorator, ast.Call):
        return decorator_root(decorator.func)

    if isinstance(decorator, ast.Name):
        return decorator.id

    if isinstance(decorator, ast.Attribute):
        return decorator.attr

    return ""


def is_mutation_assignment(node: ast.stmt | ast.expr, aliases: frozenset[str]) -> bool:
    """Assignment to an attribute or subscript (`x.y = ...`, `x[k] = ...`)."""
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        return False

    return any(isinstance(target, (ast.Attribute, ast.Subscript)) for target in targets)


def is_mutator_call(node: ast.stmt | ast.expr, aliases: frozenset[str]) -> bool:
    """A mutator call on a plain local/parameter (`values.append(...)`, ...).

    Receivers that are imported modules (`os.remove`) or attribute chains
    (`self.halt.clear`, `thread.start`) are not container mutations.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in MUTATING_METHODS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id not in aliases
    )


def dataclass_violation(name: str, node: ast.ClassDef) -> str | None:
    decorators = [
        decorator
        for decorator in node.decorator_list
        if decorator_root(decorator) == "dataclass"
    ]
    if not decorators or (name, node.name) in MUTABLE_DATACLASS_EXEMPT:
        return None

    frozen = any(
        isinstance(decorator, ast.Call)
        and any(
            keyword.arg == "frozen"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in decorator.keywords
        )
        for decorator in decorators
    )
    return (
        None
        if frozen
        else f"{name}.{node.name} must be @dataclass(frozen=True); "
        "use `replace` to derive new values"
    )


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
    for name in sorted(module_names()):
        if name in IO_EDGES:
            continue

        for node in ast.walk(parse_module(name)):
            assert not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run"
            ), f"{name}:{node.lineno} executes IO; compose IO values instead"


def test_dataclasses_are_frozen() -> None:
    for name in sorted(module_names()):
        for node in ast.walk(parse_module(name)):
            if isinstance(node, ast.ClassDef):
                violation = dataclass_violation(name, node)
                assert violation is None, violation


def test_no_global_or_nonlocal_statements() -> None:
    for name in sorted(module_names()):
        for node in ast.walk(parse_module(name)):
            assert not isinstance(node, (ast.Global, ast.Nonlocal)), (
                f"{name}:{node.lineno} uses {type(node).__name__.lower()}; "
                "thread state through parameters and return values instead"
            )


def test_mutation_assignments_only_where_sanctioned() -> None:
    problems = [
        f"{name}:{node.lineno}"
        for name in sorted(module_names())
        for node in scoped_violations(
            name,
            parse_module(name),
            is_mutation_assignment,
            MUTATION_ASSIGNMENT_ALLOWLIST,
        )
    ]
    assert not problems, (
        f"attribute/subscript assignments outside the allowlist: "
        f"{'; '.join(problems)}; derive new values instead (`replace`, "
        "tuples, `Ref` in `monads`)"
    )


def test_mutator_calls_only_where_sanctioned() -> None:
    problems = [
        f"{name}:{node.lineno}"
        for name in sorted(module_names())
        for node in scoped_violations(
            name,
            parse_module(name),
            is_mutator_call,
            MUTATOR_CALL_ALLOWLIST,
        )
    ]
    assert not problems, (
        f"container mutator calls outside the allowlist: {'; '.join(problems)}; "
        "build new collections instead (tuples, frozensets, comprehensions)"
    )


def test_raise_only_at_the_error_edge() -> None:
    for name in sorted(module_names()):
        if name in RAISE_MODULES:
            continue

        for node in ast.walk(parse_module(name)):
            assert not isinstance(node, ast.Raise), (
                f"{name}:{node.lineno} raises; return a `Result` from "
                "`errors` constructors instead"
            )


def test_effectful_imports_stay_in_sanctioned_modules() -> None:
    for name in sorted(module_names()):
        allowed = EFFECT_IMPORT_ALLOWLIST.get(name, frozenset())
        for root in sorted(import_roots(parse_module(name))):
            if root in EFFECT_MODULES:
                assert root in allowed, (
                    f"{name} imports effectful stdlib module {root!r}; keep "
                    "effects in the sanctioned modules or extend the "
                    "module's allowlist deliberately"
                )


def test_effect_import_allowlist_stays_current() -> None:
    assert frozenset(EFFECT_IMPORT_ALLOWLIST) <= module_names()
    assert all(
        effects <= EFFECT_MODULES for effects in EFFECT_IMPORT_ALLOWLIST.values()
    )


def banned_builtin_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """Bare calls to effectful builtins, with their locations."""
    return [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in BUILTIN_CALL_EDGES
    ]


def test_builtin_effect_calls_stay_at_the_edges() -> None:
    problems = [
        f"{name}:{lineno} calls {call!r}"
        for name in sorted(module_names())
        for lineno, call in banned_builtin_calls(parse_module(name))
        if name not in BUILTIN_CALL_EDGES[call]
    ]
    assert not problems, (
        f"builtin effect calls outside sanctioned edges: {'; '.join(problems)}"
    )

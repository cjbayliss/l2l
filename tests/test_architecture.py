import ast
import io
import sys
import tokenize
from collections.abc import Callable
from itertools import chain
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "l2l"
TESTS = Path(__file__).resolve().parent
TOOLS = Path(__file__).resolve().parents[1] / "tools"

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

IO_EDGES = frozenset({"cli", "monads"})

MUTABLE_DATACLASS_EXEMPT = frozenset({("monads", "Ref")})

MUTATION_ASSIGNMENT_ALLOWLIST = frozenset(
    {
        ("monads", "io_memoize"),
        ("monads", "modify_ref"),
        ("monads", "modify_ref_with"),
        ("monads", "write_ref"),
    }
)

MUTATOR_CALL_ALLOWLIST = frozenset(
    {
        ("effects", "run_log_write"),
    }
)

RAISE_MODULES = frozenset({"cli", "errors"})

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
        "pycurl",
    }
)

EFFECT_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "cli": frozenset({"os", "sys", "time"}),
    "config": frozenset({"os"}),
    "console": frozenset({"os", "shutil", "sys", "threading", "time"}),
    "effects": frozenset({"os", "shutil", "subprocess", "time", "tomllib"}),
    "http": frozenset({"pycurl"}),
    "keys": frozenset({"os", "select", "sys", "termios", "threading", "tty"}),
    "monads": frozenset({"threading"}),
    "text": frozenset({"os"}),
}

BUILTIN_CALL_EDGES: dict[str, frozenset[str]] = {
    "__import__": frozenset(),
    "breakpoint": frozenset(),
    "eval": frozenset(),
    "exec": frozenset(),
    "input": frozenset(),
    "open": frozenset({"effects"}),
    "print": frozenset({"cli"}),
}

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

REF_HELPERS = frozenset(
    {"new_ref", "read_ref", "write_ref", "modify_ref", "modify_ref_with"}
)

THIRD_PARTY_ROOTS = frozenset({"l2l", "pycurl"})


TOOL_IO_EDGES = frozenset({"optimize"})

TOOL_EFFECT_IMPORT_ALLOWLIST: dict[str, frozenset[str]] = {
    "optimize": frozenset({"os", "sys", "time"})
}

TOOL_BUILTIN_CALL_EDGES: dict[str, frozenset[str]] = {
    "__import__": frozenset(),
    "breakpoint": frozenset(),
    "eval": frozenset(),
    "exec": frozenset(),
    "input": frozenset(),
    "open": frozenset(),
    "print": frozenset({"optimize"}),
}


def module_names() -> frozenset[str]:
    return frozenset(
        path.stem
        for path in PACKAGE.glob("*.py")
        if path.stem not in ("__init__", "__main__")
    )


def tool_module_names() -> frozenset[str]:
    return frozenset(
        path.stem for path in TOOLS.glob("*.py") if path.stem != "__init__"
    )


def parse_module(name: str) -> ast.Module:
    source = (PACKAGE / (name + ".py")).read_text(encoding="utf-8")
    return ast.parse(source, filename=name + ".py")


def parse_tool_module(name: str) -> ast.Module:
    source = (TOOLS / (name + ".py")).read_text(encoding="utf-8")
    return ast.parse(source, filename="tools/" + name + ".py")


def scanned() -> list[tuple[str, ast.Module]]:
    package = [(name, parse_module(name)) for name in sorted(module_names())]
    tools = [(name, parse_tool_module(name)) for name in sorted(tool_module_names())]
    return package + tools


def effect_allowlist_for(name: str) -> frozenset[str]:
    if name in module_names():
        return EFFECT_IMPORT_ALLOWLIST.get(name, frozenset())
    return TOOL_EFFECT_IMPORT_ALLOWLIST.get(name, frozenset())


def builtin_call_allowed(name: str, call: str) -> bool:
    if name in module_names():
        return name in BUILTIN_CALL_EDGES.get(call, frozenset())
    return name in TOOL_BUILTIN_CALL_EDGES.get(call, frozenset())


def l2l_imports(tree: ast.Module) -> set[str]:
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
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            roots.add((node.module or "").split(".")[0])

    return roots


def module_aliases(tree: ast.Module) -> frozenset[str]:
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
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        return False

    return any(isinstance(target, (ast.Attribute, ast.Subscript)) for target in targets)


def is_mutator_call(node: ast.stmt | ast.expr, aliases: frozenset[str]) -> bool:
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
        else "%s.%s must be @dataclass(frozen=True); "
        "use `replace` to derive new values" % (name, node.name)
    )


def test_layers_cover_every_module() -> None:
    assert module_names() == frozenset(chain.from_iterable(LAYERS))


def test_tools_cover_expected_modules() -> None:
    assert tool_module_names() == frozenset({"optimize"})


def test_tool_imports_stay_sanctioned() -> None:
    sanctioned = frozenset(sys.stdlib_module_names) | {"l2l"}
    problems = [
        "tools/%s imports %r" % (name, root)
        for name in sorted(tool_module_names())
        for root in sorted(import_roots(parse_tool_module(name)))
        if root not in sanctioned
    ]
    assert not problems, (
        "tools imports outside the standard library and l2l: %s" % "; ".join(problems)
    )


def test_tool_imports_point_strictly_downward() -> None:
    forbidden = ("cli", "__main__")
    for name in sorted(tool_module_names()):
        for imported in sorted(l2l_imports(parse_tool_module(name))):
            if imported == "":
                continue
            assert imported not in forbidden, (
                "tools/%s imports l2l.%s; tools compose the package, not its "
                "entry points" % (name, imported)
            )
            assert LEVELS[imported] < LEVELS["cli"], (
                "tools/%s imports l2l.%s above the composition boundary"
                % (name, imported)
            )


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
                "%s (level %d) imports %s (level %d)"
                % (
                    name,
                    LEVELS[name],
                    imported,
                    LEVELS[imported],
                )
            )


def io_run_lines(name: str, tree: ast.Module) -> list[int]:
    aliases = module_aliases(tree)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run"
        and not (
            isinstance(node.func.value, ast.Name) and node.func.value.id in aliases
        )
    ]


def test_io_runs_only_at_sanctioned_edges() -> None:
    edges = IO_EDGES | TOOL_IO_EDGES
    for name, tree in scanned():
        if name in edges:
            continue

        lines = io_run_lines(name, tree)
        assert not lines, "%s:%d executes IO; compose IO values instead" % (
            name,
            lines[0],
        )


def test_dataclasses_are_frozen() -> None:
    for name, tree in scanned():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                violation = dataclass_violation(name, node)
                assert violation is None, violation


def test_no_global_or_nonlocal_statements() -> None:
    for name, tree in scanned():
        for node in ast.walk(tree):
            assert not isinstance(node, (ast.Global, ast.Nonlocal)), (
                "%s:%d uses %s; thread state through parameters and "
                "return values instead"
                % (name, node.lineno, type(node).__name__.lower())
            )


def test_mutation_assignments_only_where_sanctioned() -> None:
    problems = [
        "%s:%d" % (name, node.lineno)
        for name, tree in scanned()
        for node in scoped_violations(
            name,
            tree,
            is_mutation_assignment,
            MUTATION_ASSIGNMENT_ALLOWLIST,
        )
    ]
    assert not problems, (
        "attribute/subscript assignments outside the allowlist: %s; "
        "derive new values instead (`replace`, tuples, `Ref` in `monads`)"
        % "; ".join(problems)
    )


def test_mutator_calls_only_where_sanctioned() -> None:
    problems = [
        "%s:%d" % (name, node.lineno)
        for name, tree in scanned()
        for node in scoped_violations(
            name,
            tree,
            is_mutator_call,
            MUTATOR_CALL_ALLOWLIST,
        )
    ]
    assert not problems, (
        "container mutator calls outside the allowlist: %s; "
        "build new collections instead (tuples, frozensets, comprehensions)"
        % "; ".join(problems)
    )


def test_raise_only_at_the_error_edge() -> None:
    for name, tree in scanned():
        if name in RAISE_MODULES:
            continue

        for node in ast.walk(tree):
            assert not isinstance(node, ast.Raise), (
                "%s:%d raises; return a `Result` from `errors` constructors "
                "instead" % (name, node.lineno)
            )


def test_effectful_imports_stay_in_sanctioned_modules() -> None:
    for name, tree in scanned():
        allowed = effect_allowlist_for(name)
        for root in sorted(import_roots(tree)):
            if root in EFFECT_MODULES:
                assert root in allowed, (
                    "%s imports effectful stdlib module %r; keep effects in "
                    "the sanctioned modules or extend the module's allowlist "
                    "deliberately" % (name, root)
                )


def test_effect_import_allowlist_stays_current() -> None:
    assert frozenset(EFFECT_IMPORT_ALLOWLIST) <= module_names()
    assert all(
        effects <= EFFECT_MODULES for effects in EFFECT_IMPORT_ALLOWLIST.values()
    )
    assert frozenset(TOOL_EFFECT_IMPORT_ALLOWLIST) <= tool_module_names()
    assert all(
        effects <= EFFECT_MODULES for effects in TOOL_EFFECT_IMPORT_ALLOWLIST.values()
    )


def banned_builtin_calls(tree: ast.Module) -> list[tuple[int, str]]:
    return [
        (node.lineno, node.func.id)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in BUILTIN_CALL_EDGES
    ]


def test_builtin_effect_calls_stay_at_the_edges() -> None:
    problems = [
        "%s:%d calls %r" % (name, lineno, call)
        for name, tree in scanned()
        for lineno, call in banned_builtin_calls(tree)
        if not builtin_call_allowed(name, call)
    ]
    assert not problems, (
        "builtin effect calls outside sanctioned edges: %s" % "; ".join(problems)
    )


def mapping_annotation(annotation: ast.expr) -> bool:
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return mapping_annotation(annotation.left) or mapping_annotation(
            annotation.right
        )

    if isinstance(annotation, ast.Subscript):
        return mapping_annotation(annotation.value)

    return isinstance(annotation, ast.Name) and annotation.id == "Mapping"


def is_mapping_proxy_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MappingProxyType"
    )


def is_ref_helper_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in REF_HELPERS
    )


def test_package_imports_stay_sanctioned() -> None:
    sanctioned = frozenset(sys.stdlib_module_names) | THIRD_PARTY_ROOTS
    problems = [
        "%s imports %r" % (name, root)
        for name in sorted(module_names())
        for root in sorted(import_roots(parse_module(name)))
        if root not in sanctioned
    ]
    assert not problems, (
        "imports outside the standard library and pycurl: %s; keep the "
        "package standard-library only" % "; ".join(problems)
    )


def test_mapping_bindings_store_proxies() -> None:
    problems = [
        "%s:%d" % (name, node.lineno)
        for name, tree in scanned()
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and mapping_annotation(node.annotation)
        and not (
            node.value is None
            or is_mapping_proxy_call(node.value)
            or (isinstance(node.value, ast.Constant) and node.value.value is None)
        )
    ]
    assert not problems, (
        "Mapping bindings not wrapped in MappingProxyType: %s; wrap the "
        "mapping at the point of storage" % "; ".join(problems)
    )


def test_ref_operations_compose_into_io() -> None:
    edges = IO_EDGES | TOOL_IO_EDGES
    for name, tree in scanned():
        discarded = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and is_ref_helper_call(node.value)
        ]
        assert not discarded, (
            "%s:%d discards a Ref operation result; compose it with "
            "io_bind/io_and_then instead" % (name, discarded[0])
        )

        if name in edges:
            continue

        executed = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
            and is_ref_helper_call(node.func.value)
        ]
        assert not executed, (
            "%s:%d executes a Ref operation outside the IO edges; hand the "
            "IO value to a combinator instead" % (name, executed[0])
        )


def fstring_nodes(tree: ast.Module) -> list[ast.JoinedStr]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        and any(isinstance(value, ast.FormattedValue) for value in node.values)
    ]


def test_formatting_stays_printf_style() -> None:
    problems = [
        "%s:%d" % (name, node.lineno)
        for name, source in python_sources()
        for node in fstring_nodes(ast.parse(source))
    ]
    assert not problems, (
        "f-strings are banned; use printf-style %% formatting: %s" % "; ".join(problems)
    )


def python_sources() -> list[tuple[str, str]]:
    files = sorted(chain(PACKAGE.glob("*.py"), TESTS.glob("*.py"), TOOLS.glob("*.py")))
    return [(path.stem, path.read_text(encoding="utf-8")) for path in files]


def comment_lines(source: str) -> list[int]:
    return [
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


def docstring_nodes(tree: ast.Module) -> list[ast.Expr]:
    return [
        node.body[0]
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ]


def test_no_comments_anywhere() -> None:
    problems = [
        "%s:%d" % (name, line)
        for name, source in python_sources()
        for line in comment_lines(source)
    ]
    assert not problems, "comments are banned: %s" % "; ".join(problems)


def test_no_docstrings_anywhere() -> None:
    problems = [
        "%s:%d" % (name, node.lineno)
        for name, source in python_sources()
        for node in docstring_nodes(ast.parse(source))
    ]
    assert not problems, "docstrings are banned: %s" % "; ".join(problems)

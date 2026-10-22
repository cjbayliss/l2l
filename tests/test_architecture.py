import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = (ROOT / "src" / "l2l", ROOT / "tests", ROOT / "tools")
MUTABLE_DATACLASS_ALLOWLIST = frozenset({("monads", "Reference")})


def python_sources() -> list[tuple[str, str]]:
    return [
        (str(path.relative_to(ROOT)), path.read_text(encoding="utf-8"))
        for directory in SOURCE_DIRECTORIES
        for path in sorted(directory.rglob("*.py"))
    ]


def source_files() -> list[tuple[str, str]]:
    return [
        (name, source)
        for name, source in python_sources()
        if not name.startswith("tests/")
    ]


def decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def frozen_dataclass(decorator: ast.expr) -> bool:
    if decorator_name(decorator) != "dataclass" or not isinstance(decorator, ast.Call):
        return False
    return any(
        keyword.arg == "frozen"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in decorator.keywords
    )


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


def comment_lines(source: str) -> list[int]:
    return [
        token.start[0]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT
    ]


def fstring_nodes(tree: ast.Module) -> list[ast.JoinedStr]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        and any(isinstance(value, ast.FormattedValue) for value in node.values)
    ]


def test_dataclasses_are_frozen() -> None:
    problems = [
        "%s:%d %s" % (name, node.lineno, node.name)
        for name, source in source_files()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ClassDef)
        and (Path(name).stem, node.name) not in MUTABLE_DATACLASS_ALLOWLIST
        and any(decorator_name(item) == "dataclass" for item in node.decorator_list)
        and not any(frozen_dataclass(item) for item in node.decorator_list)
    ]
    assert not problems, "mutable dataclasses: %s" % "; ".join(problems)


def test_no_global_or_nonlocal_statements() -> None:
    problems = [
        "%s:%d %s" % (name, node.lineno, type(node).__name__)
        for name, source in source_files()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Global, ast.Nonlocal))
    ]
    assert not problems, "global or nonlocal statements: %s" % "; ".join(problems)


def test_formatting_stays_printf_style() -> None:
    problems = [
        "%s:%d" % (name, node.lineno)
        for name, source in python_sources()
        for node in fstring_nodes(ast.parse(source))
    ]
    assert not problems, "f-strings are banned: %s" % "; ".join(problems)


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

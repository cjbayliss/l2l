# Python Coding Agent Instructions

## Style of programming

- Write in **functional style with explicit effect tracking**:
  - Pure functions are the default: same inputs, same output, no side
    effects.
  - Effects are not hidden: any function that performs I/O, logging,
    random, or time is typed as returning `IO[...]` (see
    `src/l2l/monads.py`) and composed with `io_bind`/`io_map`/
    `result_bind_io` like any other value. `IO` values are descriptions
    of effects and are run only at the outermost boundary (`main`, or a
    thin adapter such as `effects.py`); everything else takes inputs and
    returns values.
  - Never mutate data. Treat all inputs as immutable. Return new values
    instead of modifying existing ones.
  - Prefer expressions over statements: comprehensions, ternaries, and
    `match` instead of accumulator loops and reassignment.
  - Structured data uses `@dataclass(frozen=True)` (or `NamedTuple`);
    plain functions are the default; classes only when a protocol or
    framework demands them. The single sanctioned exception is
    `Reference` in `monads.py` - a deliberately mutable, lock-protected
    cell for concurrency and stream state, which may only be created,
    read, and written through the `IO` operations in `monads.py`. Do not
    "fix" `Reference` to be frozen and do not create new mutable state
    outside it.
  - No global or module-level mutable state, no singletons. Module-level
    constants are `Final`.
  - Prefer `functools.reduce`, `itertools`, `map`/`filter`, or
    comprehensions over imperative loops when it stays readable.
  - Prefer raising no exceptions for expected control flow: return
    `None`, a sentinel, or an explicit result type (`Success | Failure`
    via union types) instead.

## Naming

- Never invent ad-hoc abbreviations or shortenings - spell domain words
  out completely (`maximum_allowed_connections`, not `max_conn`).
  Universally standard acronyms (HTTP, JSON, URL, CLI, API) count as
  words and are fine.
- Names must make the _what_ and _how_ of the code readable without
  comments: functions read as verbs describing behavior, variables as
  clear nouns of what they hold.
- Follow PEP 8 casing: `snake_case` for functions/variables/modules,
  `PascalCase` for types.

## Documentation

- This is an application, not a library: the README is the user
  documentation and type annotations are the contract. Do not write
  docstrings.
- Comments are allowed **only to explain _why_** - reasoning, intent,
  trade-offs, or constraints that the code itself cannot express (e.g.
  why an unusual approach was chosen, why an obvious-looking alternative
  was rejected, or why a workaround is needed).
- **Never write comments that describe _what_ the code does or _how_ it
  does it.** If something feels like it needs a _what_/_how_ comment,
  rename or restructure the code instead.
- If you find yourself writing a _what_ or _how_ comment, delete it; if
  the underlying _why_ matters, keep only that part.

## PEP compliance

- Strictly follow **PEP 8** (style, imports ordering). The exception is
  line length, that will be enforced by `ruff`.
- Imports: absolute, sorted per PEP 8; no wildcard imports.
- Exceptions: raise specific exception types - built-in ones or the
  project's own exceptions from `errors.py` (a typed error taxonomy
  replaces string-matching messages). Never bare `except:`.

### PEP 20 - the Zen of Python

Code must read as if the Zen of Python were a lint rule. In practice
this means: prefer the flat and the sparse over the nested and the
dense; explicit beats implicit; special cases are not special enough to
break the rules; errors are surfaced, never silently passed. If a line
needs a comment to explain the _what_ (see Documentation), the code is
not yet Pythonic - restructure it instead of annotating it.

### PEP 484 / PEP 526 - type annotations everywhere

Every function has annotations for **all** parameters and the return
type; code must pass `mypy --strict` and `pyright`, which enforce this.
Annotate local and module-level variables (PEP 526) where inference is
incomplete - empty containers, bindings assigned in multiple branches,
or `Final` constants - but do not add annotations the type checker can
infer exactly; they are noise. Never use legacy aliases such as
`typing.List` or quoted forward references where a direct name works -
always use the modern runtime syntax shown below.

```python
def parse_timestamp(raw: str) -> int:
    ...

rate_limit: Final = 10
pending: list[str] = []
```

Do not leave parameters or return types unannotated, do not use
`type: ignore` to silence real type errors (only with a _why_ comment
explaining the unavoidable workaround), and do not use `typing.Any`
where a precise type can be written.

### PEP 604 / PEP 585 - modern union and builtin generics

Use the modern syntax introduced by PEP 604 and PEP 585 for unions and
generic containers - `X | Y` instead of `typing.Union`/`Optional`, and
builtin generics (`list[int]`, `dict[str, int]`, `tuple[int, ...]`)
instead of `typing.List`/`Dict`/`Tuple`. `typing.Optional[X]` and
`typing.Union[X, Y]` are legacy - never write them.

```python
def find_user(identifier: int) -> User | None:
    ...

def lengths(texts: list[str]) -> dict[str, int]:
    return {text: len(text) for text in texts}
```

Do not "fix" `int | None` to `Optional[int]`, and do not rewrite
`list[int]` to `List[int]` - the modern forms are intentional and are
the only accepted spelling.

### PEP 557 - frozen dataclasses for structured data

Structured data is modeled with `@dataclass(frozen=True, slots=True)`
(PEP 557), which makes instances immutable and hashable, or with
`NamedTuple` when a tuple interface fits. Never define a dataclass
without `frozen=True` (the only exception is the `Reference` cell
described above), and never mutate a dataclass instance - use
`dataclasses.replace` to derive a modified copy.

```python
from dataclasses import dataclass, replace

@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    maximum_allowed_connections: int
    timeout_in_seconds: float

retry_settings = replace(connection_settings, timeout_in_seconds=5.0)
```

Do not use plain mutable classes, `__init__` boilerplate, or
`NamedTuple` mutation (`._replace` on a dataclass equivalent is
`replace`), and never add methods that mutate attributes.

### No class extension - every concrete class is `@final`

Class extension is banned by policy, in `src` and in `tests` alike:
never write `class Sub(Base)` and never design a class for others to
extend. Every concrete class therefore carries `@final` (PEP 591), which
makes both type checkers reject any future subclass at the inheritance
site (`mypy`: "Cannot inherit from final class"; `pyright`: "Base class
... is marked final"). Compose instead of inheriting: wrap, delegate, or
define a `Protocol` and satisfy it structurally - fakes in the test
suite implement protocols without inheritance.

```python
from typing import final


@final
@dataclass(frozen=True, slots=True)
class ConnectionSettings:
    maximum_allowed_connections: int
    timeout_in_seconds: float
```

`Protocol` classes are the one exemption - they are implemented
duck-typed, and mypy rejects `@final` on protocols with abstract
members. The narrow framework-mandated exceptions already present
(`BaseException` subclasses for control flow, `argparse.Action`
subclasses) do not generalize: do not add new ones without a _why_.

### PEP 634 - structural pattern matching

Use `match`/`case` statements (PEP 634) instead of chains of `if`/
`elif` when branching on the _shape_ of data: literal and value
patterns, sequence patterns, mapping patterns, class patterns, and `|`
(or-patterns) inside cases.

```python
match response:
    case {"status": "ok", "payload": payload}:
        return parse(payload)
    case {"status": "error", "message": str() as message}:
        return Failure(message)
    case _:
        return Failure("unknown response shape")
```

Do not rewrite existing `match` statements into `if`/`elif` chains, and
do not use `if`/`elif` ladders that test one subject for multiple shapes

- write a `match` instead. Always include a `case _:` arm when the
  subject is not exhaustively matched.

### PEP 618 - strict zip

Every call to `zip` over iterables of equal, known length must pass
`strict=True` (PEP 618) so unequal lengths raise `ValueError` instead of
silently truncating. Omit `strict` only when truncation is explicitly
the desired behavior - and if so, say so in a _why_ comment.

```python
pairs = list(zip(names, scores, strict=True))
```

Never "fix" `zip(..., strict=True)` by removing `strict=True`, and never
let a `zip` silently drop elements from the longer iterable.

### PEP 758 - parenthesized-free exception groups

**PEP 758 exception groups are intentional - do not "fix" them.**
Multiple exception types are written without parentheses when no `as`
clause is present:

```python
except OSError, ValueError:
    ...
```

This is valid Python 3.14+ syntax (PEP 758) and is used throughout the
codebase. It is NOT a tuple of exceptions, NOT a typo, and NOT legacy
Python 2 syntax. Never add parentheses around such exception lists,
never rewrite them to `(OSError, ValueError)`, and never flag them as
errors. Parentheses are only required when an `as` clause is used:

```python
except (json.JSONDecodeError, UnicodeDecodeError) as error:
    ...
```

The project targets Python >= 3.15 (`requires-python` in
`pyproject.toml`), so this syntax is fully supported by the runtime,
`ruff`, and `mypy`.

## Output

- Return complete, runnable code - no placeholders, no TODOs, no
  stubbed-out bodies. The one legitimate use of `...` is as the body of
  `Protocol` methods and stub definitions; `...` anywhere else in
  delivered code is a placeholder.

## Before finishing

```sh
uv format --preview-features format-command
uv run ruff check
uv run mypy --strict
uv run pyright
uv run pytest -q
```

CI (`.github/workflows/ci.yml`) runs exactly these checks - with
`uv format --check` instead of the format command, and `--locked` on the
`uv run` invocations - so all five must pass locally before you finish.

# Python Coding Agent Instructions

## Style of programming

- Write in **strictly functional style**:
  - All functions must be pure: same inputs, same output, no side
    effects.
  - Side effects (I/O, logging, random, time) are allowed only at the
    outermost boundary (`main` or a thin adapter layer); everything else
    takes inputs and returns values.
  - Never mutate data. Treat all inputs as immutable. Return new values
    instead of modifying existing ones.
  - Prefer expressions over statements: comprehensions, ternaries, and
    `match` instead of accumulator loops and reassignment.
  - No classes with mutable state or methods that mutate `self`. Use
    `@dataclass(frozen=True)` (or `NamedTuple`) for structured data.
    Plain functions are the default; classes only when a protocol or
    framework demands them.
  - No global or module-level mutable state, no singletons.
  - Prefer `functools.reduce`, `itertools`, `map`/`filter`, or
    comprehensions over imperative loops when it stays readable.
  - Prefer raising no exceptions for expected control flow: return
    `None`, a sentinel, or an explicit result type (`Success | Failure`
    via union types) instead.

## Naming

- Every name must be fully descriptive and self-explanatory. **Never
  abbreviate or shorten names** — spell words out completely
  (`maximum_allowed_connections`, not `max_conn`).
- Names must make the code readable without comments: functions read as
  verbs describing behavior, variables as clear nouns of what they hold.
- Follow PEP 8 casing: `snake_case` for functions/variables/modules,
  `PascalCase` for types.

## Documentation

- **Never write docstrings or comments.** Code and names must be fully
  self-documenting. If something feels like it needs a comment, rename
  or restructure it instead.

## PEP compliance

- Strictly follow **PEP 8** (style, imports ordering) and the **PEP 20**
  (Zen of Python). The exception is line length, that will be enforced
  by `ruff`.
- Fully type-annotate everything per **PEP 484/526**: use modern syntax
  (`list[int]`, `int | None`, **PEP 604/585**), `TypeAlias` where
  helpful. Code must pass `mypy --strict`.
- Use **PEP 557** dataclasses, **PEP 634** structural pattern matching,
  and **PEP 618** zip strictness where they improve clarity.
- Imports: absolute, sorted per PEP 8; no wildcard imports.
- Exceptions: raise specific built-in exception types, never bare
  `except:`.
- **PEP 758 exception groups are intentional — do not "fix" them.**
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

- Return complete, runnable code — no placeholders, no `...`, no TODOs.

## Before finishing

```sh
uv format --preview-features format-command
uv run ruff check
uv run mypy --strict
uv run pyright
uv run pytest -q
```

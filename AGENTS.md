# AGENTS.md

Guidance for humans and LLM agents editing l2l. The project uses a
**strict functional programming style**; every rule below is enforced by
`tests/test_architecture.py`, `ruff`, `mypy --strict`, and CI. Write
code that passes them the first time.

## The rules

1. **Pure functions by default.** A function that returns a non-IO type
   must not perform I/O, read global state, or mutate anything. Values
   in, values out.

2. **Effects are `IO` values.** Wrap every side effect in an `IO` thunk
   (`IO(lambda: ...)`) from `l2l.monads`, compose with `io_bind`,
   `io_map`, `io_and_then`, `fold_io`, `io_traverse`, ... and never call
   `.run()` outside `l2l/cli.py` (the program edge) and `l2l/monads.py`
   (the combinator runners). Raw effect *sources* (`Clock`, `Sleep`,
   terminal probes) are capability callables: inject them as parameters
   or frozen-field defaults, call them only inside IO thunks, and give
   any new source an `IO`-returning wrapper (see `terminal_size` in
   `l2l/console.py`).

3. **Errors are values.** Return `Result` (`Ok`/`Err`) or `Maybe`
   (`Just`/`Nothing`) instead of raising. Build errors with the
   constructors in `l2l/errors.py`; `raise` appears only as a bare
   re-raise in `cli.py`.

4. **Data is immutable.** Every dataclass is `@dataclass(frozen=True)`;
   derive new values with `dataclasses.replace`. Prefer tuples and
   frozensets over lists and sets, and `Cons` (in `monads`) when you
   need O(1) prepends. Do not call container mutators (`.append`,
   `.update`, `.sort`, ...) or assign to attributes/subscripts — build
   new collections instead. The single sanctioned mutable cell is
   `Ref` (in `monads`), read and written only inside `IO` via
   `new_ref`/`read_ref`/`write_ref`/`modify_ref*`.

5. **No `global`/`nonlocal`.** Thread state through parameters and
   return values.

6. **Respect the layers.** Modules import only from strictly lower
   layers; the map lives in `tests/test_architecture.py` (`LAYERS`) and
   a new module must be added there. Effectful stdlib modules (`os`,
   `sys`, `time`, `threading`, `urllib`, ...) may be imported only by
   the modules listed in that file's `EFFECT_IMPORT_ALLOWLIST`; pure
   helpers (`json`, `re`, `hashlib`, `functools`, ...) are unrestricted.

7. **Inject the world.** Take clocks, streams, environment mappings,
   and other effects as parameters (see `Clock`/`Sleep` in `effects`,
   `Mapping[str, str]` for the environment) so pure logic stays
   testable without patching.

8. **House style.** Python 3.14 only (PEP 695 generics, PEP 758
   unparenthesized excepts are fine); standard-library only; printf
   `%`-style formatting; no comments and no docstrings anywhere in
   `l2l/` or `tests/` — names must speak for themselves (enforced by
   `tests/test_architecture.py`).

## Testing

- Pure functions: plain pytest plus Hypothesis properties
  (`tests/test_properties.py`) where invariants exist.
- IO composition: fake effects (`tests/fakes.py`), run the composed
  `IO` once at the end, assert on captured outputs.
- Architecture: `tests/test_architecture.py` enforces the rules above
  via AST — layering, `IO.run` edges, frozen dataclasses, no
  `global`/`nonlocal`, no mutation outside `Ref`, no `raise`, no
  comments or docstrings, and effectful imports and builtin calls (`open`, `print`, `eval`, ...)
  confined to their sanctioned edges. If you add a sanctioned
  exception, extend the allowlist tables there deliberately — never
  weaken the checks.

## Before you finish

```sh
ruff format .
ruff check .
mypy
pytest
```

All four must pass. CI runs the same commands.

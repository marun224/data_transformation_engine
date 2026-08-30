"""Bending DuckDB to the reference engine's semantics -- the rules of PLAN.md 3.5.

P5 says the reference engine is the spec and silent behavioural drift is a bug. DuckDB is a
different engine with its own defaults, so a query that runs cleanly on both and
returns *different rows* is the failure mode this module exists to prevent.

**Why this is a rewrite pass and not a pile of special cases in `Column`.** The two
user surfaces build the same tree (P1), but they build it from different starting
points: `df.orderBy(col("x"))` constructs an `exp.Ordered` directly, while
`Session.sql("... ORDER BY x")` gets one from sqlglot's spark-dialect parser. A rule
implemented in `Column` would cover the first and miss the second, and the two
surfaces would quietly disagree -- exactly what P1 exists to rule out. One pass over
the finished tree covers both by construction.

It runs before the optimizer, so the optimizer sees the tree that is actually going
to execute, and pushdown reasons about the real predicate rather than a pre-rewrite
one.

**What is deliberately not here.** Rules DuckDB already gets right are tested, not
implemented -- see `tests/unit/test_conformance.py`. Adding a rewrite that changes
nothing costs readability in every generated query and buys no correctness.
"""

from __future__ import annotations

from sqlglot import exp

__all__ = ["ORDERING_RULE", "apply_compat_semantics"]

#: What the reference engine does with NULLs in `ORDER BY`, and what DuckDB 1.5 does unasked.
#:
#: The reference engine:  ASC -> NULLS FIRST, DESC -> NULLS LAST.
#: DuckDB: NULLS LAST for *both* directions.
#:
#: So ASC is a real divergence and DESC happens to agree today. Both are made
#: explicit anyway: "happens to agree" is a property of DuckDB 1.5.5, not a promise,
#: and an explicit clause costs nothing while a silent re-ordering costs a wrong
#: answer. (PLAN.md 3.5 records DuckDB as nulls-first on DESC; that was true of an
#: older DuckDB and is not true of 1.5.5 -- see divergence.md.)
ORDERING_RULE = "ASC nulls first, DESC nulls last -- always emitted explicitly"


def _fix_null_ordering(node: exp.Expression) -> exp.Expression:
    """Give every `ORDER BY` term an explicit NULLS FIRST/LAST."""
    if not isinstance(node, exp.Ordered):
        return node
    # An explicit `NULLS FIRST` from the user wins; only fill in what is unstated.
    if node.args.get("nulls_first") is not None:
        return node
    descending = bool(node.args.get("desc"))
    node.set("nulls_first", not descending)
    return node


def _is_explicit_try_cast(node: exp.Cast) -> bool:
    """True when the user wrote `try_cast(...)` rather than `cast(...)`.

    sqlglot's the reference engine dialect parses *both* spellings into `exp.TryCast`, because
    The reference engine's default cast is already lenient -- but it sets `safe=True` only on the
    explicit one. That flag is the difference between "be lenient because the reference engine is"
    and "be lenient because I asked", and only the first should follow `ansi_mode`.
    """
    return node.args.get("safe") is True


def _fix_casts(node: exp.Expression, *, ansi_mode: bool) -> exp.Expression:
    """Make every cast mean what it would mean in the reference engine under the current mode.

    The two surfaces arrive here spelled differently, which is the whole reason this
    is a tree pass:

        Session.sql("CAST(x AS INT)") ->  TryCast(safe=None)   (the spark dialect
                                          already knows the default is lenient)
        col("x").cast("int")          ->  Cast                 (we build it plainly)

    Non-ANSI, both must end up lenient; ANSI, both must end up strict. `TRY_CAST` is
    DuckDB's spelling of the lenient one and covers the same type pairs `CAST` does,
    lists and structs included, which is what makes a blanket rewrite safe rather
    than a table of special cases.
    """
    if not isinstance(node, exp.Cast):
        return node
    explicit = isinstance(node, exp.TryCast) and _is_explicit_try_cast(node)
    if explicit:
        # `try_cast(...)` means try_cast in both modes.
        return node
    target = exp.Cast if ansi_mode else exp.TryCast
    if type(node) is target:
        return node
    return target(this=node.this, to=node.args.get("to"))


def apply_compat_semantics(
    expression: exp.Expression, *, ansi_mode: bool = False
) -> exp.Expression:
    """Rewrite `expression` so DuckDB answers the question the reference engine would.

    `ansi_mode` mirrors `icetl.ansiMode`. It is off by default, matching
    the reference engine, and turning it on opts into strict casting -- an error where the default
    gives NULL. It does not make everything else strict: integer overflow still
    errors in both modes, because DuckDB cannot be asked to wrap and pretending
    otherwise would be the silent drift P5 forbids. See divergence.md.

    Returns a new tree; `expression` is not mutated.
    """
    rewritten = expression.transform(_fix_null_ordering, copy=True)
    return rewritten.transform(lambda node: _fix_casts(node, ansi_mode=ansi_mode), copy=False)

"""Which plan nodes make a scope return more rows than it read.

There is one such node here and it is `unnest`. Phase 6 puts it in the *select list*
rather than in the FROM clause, because DuckDB expands it there for free and
correlates repeated copies of it (`sql/generators.py` says why). That is a real
saving, and it costs this: a set-returning function sitting where an ordinary scalar
expression sits, which every rule that reasons about projections assumes is scalar.

Two of them assumed it, and both were wrong quietly:

  * `merge_subqueries` merged `SELECT count(*) FROM (SELECT unnest(x) FROM t)` into
    `SELECT count(*) FROM t`. The generator was unreferenced by the outer scope, so
    it was dropped -- and with it every row it would have produced. `count()` over an
    exploded frame returned the *table's* row count. **A silent wrong answer.**
  * The same rule inlined the generator into a `WHERE` over it, which DuckDB rejects
    outright: *UNNEST not supported here*. A loud failure, and the same cause.

So a plan containing a generator skips `merge_subqueries` (see `plan/optimizer.py`).
The cost is pushdown on exploded queries only, and pushdown is a speed concern where
this was an answer.

The list is deliberately short and deliberately ours: a name that is *not* here is
treated as scalar, so anything added to `sql/generators.py` must be added here too.
sqlglot's own `Explode`/`Posexplode`/`Unnest` nodes are included because the SQL
surface can parse them directly, even though the rule handles those correctly today.
"""

from __future__ import annotations

from sqlglot import exp

from icetl.plan.builder import as_expression

__all__ = ["GENERATOR_FUNCTIONS", "contains_generator", "is_generator"]

#: Function names that emit more than one row per input row. `sql/generators.py`
#: builds these as `exp.Anonymous`, which sqlglot has no reason to treat specially.
GENERATOR_FUNCTIONS = frozenset({"unnest"})

#: sqlglot's own set-returning nodes, reachable by parsing SQL that names them.
_GENERATOR_NODES = (exp.Explode, exp.Posexplode, exp.Unnest)


def is_generator(node: exp.Expression) -> bool:
    """True when `node` itself can turn one row into several."""
    if isinstance(node, _GENERATOR_NODES):
        return True
    return isinstance(node, exp.Anonymous) and str(node.this).lower() in GENERATOR_FUNCTIONS


def contains_generator(expression: exp.Expression) -> bool:
    """True when anything anywhere in `expression` can multiply rows.

    Whole-tree rather than per-scope: the rules this guards rewrite across scope
    boundaries, so a generator two levels down is exactly the one that gets moved.
    """
    return any(is_generator(as_expression(node)) for node in expression.walk())

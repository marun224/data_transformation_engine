"""Recognising the one query Iceberg's metadata can answer without reading a file.

`SELECT count(*) FROM t`, unfiltered, is the whole of it -- and it is worth its own
module because Iceberg already knows the answer. Every manifest entry carries a
`record_count`, so the row count is a sum over metadata that `plan_files()` has
fetched anyway. Handing the same question to DuckDB instead makes it open every data
file's parquet footer: roughly 2x on a 357-file table, and worse as the file count
grows (FINDINGS.md 3.4).

The recognition, not the counting, is the delicate part. A row count read from
metadata is exact only when **nothing** between the table and the count can change
the cardinality, so this is a whitelist of shapes rather than a search for
disqualifiers:

    SELECT count(*) FROM t                       the SQL surface
    SELECT count(*) FROM (SELECT a, b FROM t)    what df.count() builds

A `WHERE` is the case that makes the distinction matter rather than an academic one.
File pruning is an *over*-approximation -- `plan_files()` returns the files that may
hold a matching row, not the rows -- so summing their record counts under a filter
would confidently return a number that is too big. That is the same asymmetry the
row-level write path lives under (FINDINGS.md 6, rule 2), and here it is settled by
refusing the fast path rather than by loosening it.

Everything unrecognised falls through to DuckDB, which is merely the slower way to
get the identical answer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from icetl.plan.builder import source_key

if TYPE_CHECKING:
    from collections.abc import Mapping

    from icetl.plan.builder import ScanSource

__all__ = ["countable_scan"]

#: Args a `SELECT` may carry and still read exactly one row per row of its source.
#: Anything else -- a WHERE, a GROUP BY, a LIMIT, a join, a DISTINCT, a window -- is
#: disqualifying, and is caught by requiring every other arg to be empty. A whitelist
#: because sqlglot grows node kinds faster than we would notice: a new one arrives
#: here as "not countable", which costs speed and never an answer.
_CARDINALITY_NEUTRAL_ARGS = frozenset({"expressions", "from", "from_", "kind", "hint"})


def _is_plain_select(node: exp.Expression) -> bool:
    """True when `node` is a `SELECT` that neither filters, groups, nor reshapes."""
    if not isinstance(node, exp.Select):
        return False
    for key, value in node.args.items():
        if key not in _CARDINALITY_NEUTRAL_ARGS and value:
            return False
    return True


def _projects_only_columns(node: exp.Select) -> bool:
    """True when every projection is a column or a star.

    The gate against a generator. `select(F.explode(items))` emits one row per element
    and is spelled as an ordinary projection, so a rule reading only the FROM clause
    would count the table's rows and return them as the query's -- wrong, and wrong
    quietly. Columns and stars cannot do that; nothing else is admitted.
    """
    for projection in node.expressions:
        inner = projection.this if isinstance(projection, exp.Alias) else projection
        if not isinstance(inner, (exp.Column, exp.Star)):
            return False
    return True


def _counted_table(node: exp.Expression) -> exp.Table | None:
    """Unwrap `node` down to the table it reads, or None if it reads anything else."""
    while True:
        if isinstance(node, exp.Table):
            # A table function -- read_parquet, a metadata table already materialised
            # -- is not an Iceberg scan and has no manifest to ask.
            return None if isinstance(node.this, exp.Func) else node
        if isinstance(node, (exp.Subquery, exp.Paren)):
            node = node.this
            continue
        if isinstance(node, exp.Select):
            if not _is_plain_select(node) or not _projects_only_columns(node):
                return None
            source = node.args.get("from_") or node.args.get("from")
            if source is None:
                return None
            node = source.this
            continue
        return None


def _is_count_star(node: exp.Expression) -> bool:
    """True for `count(*)` exactly -- not `count(col)`, and not `count(DISTINCT ...)`.

    `count(col)` skips NULLs and `count(DISTINCT col)` needs the values themselves;
    neither is answerable from a row count.
    """
    if isinstance(node, exp.Alias):
        node = node.this
    if not isinstance(node, exp.Count):
        return False
    if node.args.get("distinct") or node.args.get("filter"):
        return False
    return isinstance(node.this, exp.Star)


def countable_scan(plan: exp.Expression, sources: Mapping[str, ScanSource]) -> ScanSource | None:
    """The source this plan is an unfiltered `count(*)` over, or None.

    None is the ordinary answer and means only "let DuckDB run it".
    """
    if not _is_plain_select(plan) or len(plan.expressions) != 1:
        return None
    if not _is_count_star(plan.expressions[0]):
        return None

    source = plan.args.get("from_") or plan.args.get("from")
    if source is None:
        return None
    table = _counted_table(source.this)
    if table is None:
        return None
    return sources.get(source_key(table, versioned=True))

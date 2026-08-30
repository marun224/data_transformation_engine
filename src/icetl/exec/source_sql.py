"""Turning a `ScanPlan` into the relation that replaces a table reference in the SQL.

Phase 1 substituted a bare `read_parquet(...)` call. Phase 2 substitutes a subquery,
because two things now need to be said at the point of the scan and cannot be said by
a table function alone:

    fast path      (SELECT "id", "amount" FROM read_parquet($p, union_by_name => true))

    renamed (3.4)  (SELECT "id", "old_name" AS "new_name" FROM read_parquet($p0, ...)
                    UNION ALL
                    SELECT "id", "new_name"              FROM read_parquet($p1, ...))

The union is therefore the *rename* path only: under copy-on-write (decision 11) a
scan is exactly its data files, so there is no second half to union in.

The explicit column list is projection pushdown reaching the reader (3.6): on the
200-column fixture it is the difference between handing DuckDB 200 columns and
handing it two. It is also always safe, because the list is exactly what the
optimizer proved the rest of the query references -- and when the optimizer could
not prove anything, the list is every column.

`union_by_name` stays on within a group. It is what makes an *added* column read
back as NULL from files written before it existed, which is correct; the case it
gets wrong is renames, and those never reach it now because the group above has
already aliased them by field-id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlglot import exp

from icetl.plan.builder import as_expression

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from icetl.exec.scan_planner import FileGroup, ScanPlan

__all__ = ["build_source"]


def _read_parquet(parameter: str) -> exp.Expression:
    """`read_parquet($parameter, union_by_name => true, hive_partitioning => false)`.

    The file list travels as a named parameter rather than inlined literals: a wide
    partitioned table can resolve to thousands of paths, and `explain()` stays
    readable when they are a `$name` instead of a page of quoted strings.

    **`hive_partitioning` must be off.** DuckDB turns it on by itself when it sees
    `key=value` directories, and an Iceberg warehouse is full of them --
    `.../data/as_at_date=2026-08-17/00000-...parquet`. It then synthesises the column
    from the *path* and type-casts it, so a `string` partition column comes back as
    `DATE`, shadowing the real values in the file. For Iceberg that is wrong twice
    over: the directory name holds the *transformed* value (a bucket number, a
    truncated string, a year), not the column's value, and Iceberg does not promise
    the value is in the path at all. Reading the data is the only correct answer.
    """
    return exp.Anonymous(
        this="read_parquet",
        expressions=[
            exp.Placeholder(this=parameter),
            exp.Kwarg(this=exp.var("union_by_name"), expression=exp.true()),
            exp.Kwarg(this=exp.var("hive_partitioning"), expression=exp.false()),
        ],
    )


def _projection(group: FileGroup, columns: tuple[str, ...]) -> list[exp.Expression]:
    """The SELECT list for one file group, aliased back to today's column names."""
    if group.projection is None:
        return [exp.column(name, quoted=True) for name in columns]

    out: list[exp.Expression] = []
    for alias in group.projection:
        if alias.stored is None:
            # These files were written before the column existed. Iceberg reads it
            # as NULL, and the cast keeps the branches of the UNION type-compatible.
            value = as_expression(exp.cast(exp.Null(), alias.duckdb_type, dialect="duckdb"))
        else:
            value = exp.column(alias.stored, quoted=True)
        if isinstance(value, exp.Column) and alias.stored == alias.output:
            out.append(value)
        else:
            out.append(as_expression(exp.alias_(value, alias.output, quoted=True)))
    return out


def build_source(
    plan: ScanPlan,
    *,
    parameters: dict[str, Any],
    register: Callable[[str, Any], None],
) -> exp.Expression:
    """The relation that stands in for `plan`'s table reference.

    `parameters` collects the file lists to bind at execution time; `register` is how
    an Arrow table -- the empty stand-in below is the only one left -- is made visible
    to DuckDB under a name.
    """
    view = plan.source.view
    columns = plan.columns

    if plan.is_empty:
        # `read_parquet([])` is an error in DuckDB, but a table with no files is an
        # ordinary state -- new, fully deleted, or pruned to nothing by a predicate.
        # An empty Arrow table of the right schema gives the right columns, no rows.
        name = f"{view}_empty"
        register(name, plan.source.resolved.table.schema().as_arrow().empty_table())
        return exp.select(*[exp.column(c, quoted=True) for c in columns]).from_(
            exp.to_identifier(name, quoted=True)
        )

    branches: list[exp.Expression] = []
    for index, group in enumerate(plan.groups):
        parameter = f"{view}_paths_{index}"
        parameters[parameter] = list(group.paths)
        branches.append(exp.select(*_projection(group, columns)).from_(_read_parquet(parameter)))

    relation = branches[0]
    for branch in branches[1:]:
        # Plain UNION ALL, not BY NAME: every branch above projects `columns` in the
        # same order by construction, so positional union is already correct and
        # spares DuckDB the name-matching pass.
        relation = exp.union(relation, branch, distinct=False)
    return relation

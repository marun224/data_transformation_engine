"""Building and rewriting the sqlglot plan tree.

An Iceberg table appears in the plan as an ordinary `exp.Table` node -- exactly what
sqlglot produces when it parses `FROM nyc.yellow_tripdata`. That is deliberate: the
DataFrame API builds the same node the SQL parser does, so one substitution pass
serves both surfaces.

The node stays a plain table reference right up until SQL is generated, at which
point `substitute_sources` swaps each one for whatever the caller needs:

    execution   ->  read_parquet($paths_0, union_by_name = true) AS yellow_tripdata
    analysis    ->  icetl_src_0 AS yellow_tripdata          (a zero-row Arrow view)

Substitution always preserves the alias the reference already carried, and invents
one from the table's own name when it had none, so `t.col` and `yellow_tripdata.col`
keep resolving after the swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from sqlglot import exp

from icetl.errors import UnsupportedFeatureError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from icetl.catalog import ResolvedTable

__all__ = [
    "ScanSource",
    "as_expression",
    "collect_source_keys",
    "source_key",
    "source_table",
    "substitute_sources",
    "wrap_as_subquery",
]


def as_expression(node: object) -> exp.Expression:
    """Narrow sqlglot's loose `Expr` trait base to the `Expression` it really is.

    sqlglot 30 annotates many of its builders and accessors -- `alias_`, `parse_one`,
    `Binary.left`, `Select.where` -- with the `Expr` mixin their return types share
    rather than with `Expression`, which every one of them actually returns. Narrowing
    in one named place keeps the rest of the codebase honestly typed against the real
    class, and leaves one obvious thing to delete when sqlglot tightens the hints.
    """
    return cast(exp.Expression, node)


@dataclass(frozen=True)
class ScanSource:
    """One Iceberg table referenced by a plan, at one snapshot.

    `key` is the reference as it is spelled in the tree (`fx.plain`), which is what
    substitution matches on -- and it carries the `VERSION AS OF` a reference was
    written with, so the same table at two snapshots is two sources and joins to
    itself. `view` is the name its zero-row stand-in is registered under during
    analysis; it is kept off the `key` namespace so a real table called `icetl_src_0`
    could not collide with it.

    `snapshot_id` is None for the ordinary "as it is now" read.
    """

    key: str
    resolved: ResolvedTable
    view: str
    snapshot_id: int | None = None


#: Separates a reference from the version it was asked for inside a source key. NUL
#: cannot appear in a SQL identifier, quoted or not, so a key carrying one is
#: unambiguously ours and never a table someone named awkwardly.
VERSION_MARK = chr(0) + "@"


def source_key(table: exp.Table, *, versioned: bool = False) -> str:
    """The dotted reference a table node spells, ignoring any alias.

    `nyc.yellow_tripdata AS t` -> `nyc.yellow_tripdata`.

    With `versioned`, a `VERSION AS OF` / `TIMESTAMP AS OF` is folded into the key, so
    two references to one table at different snapshots resolve to two sources. Callers
    that address a table to *write* to it leave it off: there is no writing to a
    snapshot, and a version on a write target is refused where it is read.
    """
    parts = [part for part in (table.catalog, table.db, table.name) if part]
    key = ".".join(parts)
    if not versioned:
        return key
    version = table.args.get("version")
    if version is None:
        return key
    kind = str(version.this or "").upper()
    value = version.args.get("expression")
    return f"{key}{VERSION_MARK}{kind}:{value.name if value is not None else ''}"


def assert_no_version(table: exp.Table, action: str) -> None:
    """Refuse `VERSION AS OF` where a statement is going to *write*.

    Time travel names a snapshot that has already been committed. There is nothing to
    write to it, and quietly writing to the table's current state instead would be a
    statement doing something other than what it says.
    """
    version = table.args.get("version")
    if version is not None:
        raise UnsupportedFeatureError(
            f"{action} a table at {str(version.this or '').upper()} AS OF",
            hint="A snapshot is history. Write to the table itself",
        )


def split_version(key: str) -> tuple[str, str | None, str | None]:
    """a key carrying the mark -> `("fx.plain", "VERSION", "5")`."""
    if VERSION_MARK not in key:
        return key, None, None
    reference, _, tail = key.partition(VERSION_MARK)
    kind, _, value = tail.partition(":")
    return reference, kind, value


def describe_key(key: str) -> str:
    """The key as a reader would write it, for an error message."""
    reference, kind, value = split_version(key)
    if kind is None:
        return reference
    return f"{reference} {kind} AS OF {value}"


def source_table(reference: str) -> exp.Table:
    """Build the plan node for a table reference like `fx.plain` or `cat.ns.plain`.

    Parts are quoted, so a namespace or table whose name collides with a SQL keyword
    still round-trips.
    """
    parts = reference.split(".")
    if len(parts) > 3:
        raise ValueError(f"Table reference {reference!r} has more than three parts.")
    identifiers = [exp.to_identifier(part, quoted=True) for part in parts]
    keys = ["catalog", "db", "this"][-len(identifiers) :]
    return exp.Table(**dict(zip(keys, identifiers, strict=True)))


def _cte_names(expression: exp.Expression) -> set[str]:
    """Names bound by `WITH` anywhere in the tree.

    A CTE reference parses as an `exp.Table`, so without this a query using one would
    send its CTE name to the catalog and fail as a missing table.
    """
    return {cte.alias_or_name for cte in expression.find_all(exp.CTE) if cte.alias_or_name}


def collect_source_keys(expression: exp.Expression) -> list[str]:
    """Every distinct table reference in `expression`, in first-seen order.

    Table *functions* (`read_parquet(...)`) and CTE references are skipped -- neither
    is something the catalog can resolve.
    """
    bound = _cte_names(expression)
    keys: list[str] = []
    for table in expression.find_all(exp.Table):
        if isinstance(table.this, exp.Func):
            continue
        key = source_key(table, versioned=True)
        if not key or key in bound or key in keys:
            continue
        keys.append(key)
    return keys


def substitute_sources(
    expression: exp.Expression,
    sources: Mapping[str, ScanSource],
    factory: Callable[[ScanSource], exp.Expression],
) -> exp.Expression:
    """Replace every known table reference with what `factory` returns for it.

    The original node's alias is carried over, falling back to the table's own name,
    so column qualifiers survive the swap. `expression` is not mutated.
    """

    def replace(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Table) or isinstance(node.this, exp.Func):
            return node
        source = sources.get(source_key(node, versioned=True))
        if source is None:
            return node
        replacement = factory(source)
        table_alias = exp.TableAlias(this=exp.to_identifier(node.alias or node.name, quoted=True))
        if isinstance(replacement, (exp.Select, exp.SetOperation, exp.Subquery)):
            # A whole relation, not a name: it has to be nested to sit in a FROM.
            # Phase 2 substitutes these -- the projection list, the field-id
            # aliasing, and the merge-on-read union all live inside one.
            inner = replacement.this if isinstance(replacement, exp.Subquery) else replacement
            return exp.Subquery(this=inner, alias=table_alias)
        return exp.Table(this=replacement, alias=table_alias)

    return expression.transform(replace, copy=True)


def wrap_as_subquery(expression: exp.Expression, alias: str) -> exp.Select:
    """`SELECT * FROM (expression) AS alias`.

    Every DataFrame operation that cannot safely extend the current SELECT nests it
    this way. Nesting is the correct default: a filter added to
    `SELECT a AS b FROM t` must see `b`, which only a subquery gives it. The verbose
    SQL that results is flattened by DuckDB's own optimizer today, and by ours from
    Phase 2, so it costs readability rather than speed.
    """
    return exp.select(exp.Star()).from_(
        exp.Subquery(this=expression, alias=exp.TableAlias(this=exp.to_identifier(alias)))
    )

"""Predicate and projection pushdown: what the optimized tree tells the scan planner.

Two independent jobs, both reading the *optimized* tree (PLAN.md 3.2, 3.6):

  **Projection.** After `qualify` expands `SELECT *` and `pushdown_projections`
  trims it, the columns still named against a table are exactly the columns that
  table must produce. On the 200-column fixture that is the difference between
  reading 200 columns and reading 2.

  **Predicate.** Conjuncts of the `WHERE` sitting directly above a scan are
  translated into PyIceberg `BooleanExpression`s, which prune manifests, partitions
  and -- via column stats -- whole data files, without opening them.

**The invariant that makes this safe:** a translated conjunct is *also* left in the
generated SQL, always. Nothing in this module removes a filter. PyIceberg's pruning
is stats-based and therefore approximate -- a file whose min/max straddles the
predicate is kept, and its non-matching rows come back -- so DuckDB re-applying the
predicate is what makes the answer right. Pushdown here is only ever a way to read
less. That one rule retires an entire family of wrong-results bugs, so every
translation below is free to be conservative: anything not understood is simply not
pushed, and the query stays correct by construction.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pyiceberg.expressions import (
    AlwaysTrue,
    And,
    BooleanExpression,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNull,
    LessThan,
    LessThanOrEqual,
    Not,
    NotEqualTo,
    Or,
    StartsWith,
)
from pyiceberg.expressions.visitors import bind, rewrite_not
from pyiceberg.types import IcebergType, TimestampType
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from icetl.plan.annotations import PlanAnnotations, ScanRequest
from icetl.plan.builder import as_expression, source_key

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pyiceberg.schema import Schema as IcebergSchema

    from icetl.plan.builder import ScanSource

__all__ = [
    "ColumnResolver",
    "binds_against",
    "extract_scan_requests",
    "is_exactly_translatable",
    "is_null_rejecting",
    "join_predicates",
    "null_padded_aliases",
    "scope_predicate",
    "translate_predicate",
]

# `2024-01-01` and nothing more -- the form SQL accepts for a timestamp and
# PyIceberg does not.
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")

# PyIceberg's predicates are pydantic models that define a positional
# `__init__(term, literal)` on their base class but declare extra fields on each
# subclass. mypy follows `dataclass_transform` and synthesises a keyword-only
# `__init__` per subclass, so it never sees the constructor Python actually calls.
# Naming them through `Any` keeps the call sites below readable rather than
# scattering `type: ignore` across every branch of the translator.
_eq: Any = EqualTo
_in: Any = In
_is_null: Any = IsNull
_gte: Any = GreaterThanOrEqual
_lte: Any = LessThanOrEqual
_starts_with: Any = StartsWith

# Comparison nodes, and the same comparison with its operands the other way round --
# needed because `100 > amount` means `amount < 100`.
_COMPARISONS: dict[type[exp.Expression], tuple[Any, Any]] = {
    exp.EQ: (EqualTo, EqualTo),
    exp.NEQ: (NotEqualTo, NotEqualTo),
    exp.GT: (GreaterThan, LessThan),
    exp.GTE: (GreaterThanOrEqual, LessThanOrEqual),
    exp.LT: (LessThan, GreaterThan),
    exp.LTE: (LessThanOrEqual, GreaterThanOrEqual),
}


class _Unset:
    """Distinguishes "not a literal" from the literal `None`."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


_UNSET = _Unset()


class ColumnResolver:
    """Maps qualified column nodes back to Iceberg columns for one table.

    Two things make this more than a `.name` lookup. The optimizer works in DuckDB's
    dialect, which is case-insensitive, so `VendorID` can come back as `vendorid`;
    and a node qualified with a *different* alias must be rejected outright, since a
    predicate that mentions another table cannot prune this one.

    The field *types* come along too, because a literal that is fine in SQL is not
    always a literal PyIceberg will accept -- see `normalise_literal`.
    """

    def __init__(self, alias: str, columns: Mapping[str, IcebergType]) -> None:
        self.alias = alias
        self._by_lower = {name.lower(): name for name in columns}
        self._types = dict(columns)

    def name(self, node: exp.Expression) -> str | None:
        """The Iceberg column `node` names, or None if it names something else."""
        if not isinstance(node, exp.Column):
            return None
        if node.table and node.table != self.alias:
            return None
        # A struct field access carries extra parts. Iceberg predicates address
        # top-level fields only, so those are left for DuckDB alone.
        if len(node.parts) > 2:
            return None
        return self._by_lower.get(node.name.lower())

    def field_type(self, name: str) -> IcebergType | None:
        return self._types.get(name)

    def normalise_literal(self, name: str, value: Any) -> Any:
        """Spell `value` the way PyIceberg's literal binding expects for that column.

        SQL is relaxed about timestamps -- the reference engine and DuckDB both read
        `tpep_pickup_datetime >= '2024-01-01'` as midnight on that day. PyIceberg is
        strict, and rejects anything that is not full ISO-8601, by *raising* rather
        than declining. Widening the date here is what lets the most ordinary filter
        anyone writes against a time-partitioned table actually prune.

        `timestamptz` is deliberately left alone: a bare date has no zone, DuckDB
        would resolve it against the session's, and guessing UTC could shift the
        boundary and prune away a file holding wanted rows. Pruning less is free;
        pruning wrongly is not.
        """
        field_type = self._types.get(name)
        if not isinstance(field_type, TimestampType) or not isinstance(value, str):
            return value
        text = value.strip().replace(" ", "T")
        if _DATE_ONLY.fullmatch(text):
            return f"{text}T00:00:00"
        return text

    def owns_every_column(self, node: exp.Expression) -> bool:
        """True when every column mentioned anywhere in `node` is one of this table's."""
        found = list(node.find_all(exp.Column))
        return bool(found) and all(self.name(column) is not None for column in found)


def binds_against(predicate: BooleanExpression, schema: IcebergSchema) -> bool:
    """True when PyIceberg can actually bind `predicate` to `schema`.

    The last line of the "never push what you do not understand" rule, and the one
    that makes it true rather than aspirational. Translation happens against sqlglot
    nodes, but PyIceberg only checks the *literal* when the scan is planned -- deep
    inside `plan_files()`, far from anywhere that could decline gracefully. A literal
    it dislikes raises there and takes the whole query with it.

    So every predicate is bound here first, where failing merely means the conjunct
    stays in the SQL and the scan prunes less.
    """
    try:
        bind(schema, rewrite_not(predicate), case_sensitive=True)
    except Exception:
        # Deliberately broad: pyiceberg raises ValueError, TypeError and its own
        # ValidationError from binding, and a new one appearing in a future release
        # must not become a failed query.
        return False
    return True


def _literal(node: exp.Expression) -> Any:
    """The Python value of a literal node, or `_UNSET` when it is not one.

    Casts are unwrapped rather than evaluated: PyIceberg binds the value against the
    field's own Iceberg type, so `CAST('2026-08-17' AS DATE)` and the bare string
    reach it identically -- and letting PyIceberg do the conversion means we never
    disagree with it about what a literal means.
    """
    if isinstance(node, exp.Paren):
        return _literal(node.this)
    if isinstance(node, (exp.Cast, exp.TryCast)):
        return _literal(node.this)
    if isinstance(node, exp.Neg):
        inner = _literal(node.this)
        if inner is _UNSET or isinstance(inner, (str, bool)):
            return _UNSET
        return -inner
    if isinstance(node, exp.Boolean):
        return node.this
    if isinstance(node, exp.Literal):
        if node.is_string:
            return node.this
        text = node.name
        return float(text) if ("." in text or "e" in text.lower()) else int(text)
    return _UNSET


def _operands(node: exp.Binary) -> tuple[exp.Expression, exp.Expression]:
    """A binary node's two sides.

    Read out of `args` rather than through `.left` / `.right`: sqlglot annotates
    those with its looser `Expr` base, and going via `args` keeps this module's
    types honest without a cast that would be wrong the day sqlglot tightens them.
    """
    return node.args["this"], node.args["expression"]


def _like_prefix(pattern: str) -> str | None:
    """The literal prefix of a `LIKE 'abc%'` pattern, if that is all the pattern is."""
    if not pattern.endswith("%"):
        return None
    body = pattern[:-1]
    if "%" in body or "_" in body or "\\" in body:
        return None
    return body or None


def translate_predicate(node: exp.Expression, resolver: ColumnResolver) -> BooleanExpression | None:
    """One sqlglot predicate as a PyIceberg expression, or `None` if it will not go.

    Returning `None` is an ordinary outcome, not a failure: the conjunct stays in the
    SQL and the scan simply prunes less.
    """
    if isinstance(node, exp.Paren):
        return translate_predicate(node.this, resolver)

    if isinstance(node, exp.And):
        # AND is the one operator that can be pushed *partially*: keeping only the
        # translatable half still prunes, because `A AND B` never needs a file that
        # `A` alone would reject.
        first, second = _operands(node)
        left = translate_predicate(first, resolver)
        right = translate_predicate(second, resolver)
        if left is None:
            return right
        if right is None:
            return left
        return And(left, right)

    if isinstance(node, exp.Or):
        # OR is all or nothing: dropping either side narrows the predicate to
        # something that could reject a file the query needs.
        first, second = _operands(node)
        left = translate_predicate(first, resolver)
        right = translate_predicate(second, resolver)
        return None if left is None or right is None else Or(left, right)

    if isinstance(node, exp.Not):
        inner = translate_predicate(node.this, resolver)
        return None if inner is None else Not(inner)

    if isinstance(node, exp.Is):
        name = resolver.name(node.this)
        if name is None or not isinstance(node.expression, exp.Null):
            return None
        return _is_null(name)

    if isinstance(node, exp.In):
        name = resolver.name(node.this)
        if name is None or node.args.get("query") is not None:
            return None
        values = [_literal(item) for item in node.expressions]
        if not values or any(value is _UNSET for value in values):
            return None
        return _in(name, [resolver.normalise_literal(name, value) for value in values])

    if isinstance(node, exp.Between):
        name = resolver.name(node.this)
        low, high = _literal(node.args["low"]), _literal(node.args["high"])
        if name is None or low is _UNSET or high is _UNSET:
            return None
        return And(
            _gte(name, resolver.normalise_literal(name, low)),
            _lte(name, resolver.normalise_literal(name, high)),
        )

    if isinstance(node, exp.Like):
        name = resolver.name(node.this)
        pattern = _literal(node.expression)
        if name is None or not isinstance(pattern, str):
            return None
        prefix = _like_prefix(pattern)
        return None if prefix is None else _starts_with(name, prefix)

    comparison = _COMPARISONS.get(type(node))
    if comparison is not None and isinstance(node, exp.Binary):
        direct, flipped = comparison
        first, second = _operands(node)
        name = resolver.name(first)
        if name is not None:
            value = _literal(second)
            if value is _UNSET:
                return None
            return direct(name, resolver.normalise_literal(name, value))
        name = resolver.name(second)
        if name is not None:
            value = _literal(first)
            if value is _UNSET:
                return None
            return flipped(name, resolver.normalise_literal(name, value))
        return None

    # A bare boolean column used as a predicate: `WHERE is_active`.
    name = resolver.name(node)
    return None if name is None else _eq(name, True)


def conjuncts(where: exp.Expression | None) -> list[exp.Expression]:
    """Split a WHERE clause into its top-level AND terms."""
    if where is None:
        return []
    condition = where.this if isinstance(where, exp.Where) else where
    pending, out = [condition], []
    while pending:
        node = pending.pop()
        if isinstance(node, exp.Paren):
            pending.append(node.this)
        elif isinstance(node, exp.And):
            pending.extend((node.left, node.right))
        else:
            out.append(node)
    return out


#: Node types whose translation above is *exact* -- the PyIceberg predicate matches the
#: same rows the SQL does, nulls included -- rather than merely a superset good enough
#: for file pruning. `exp.Like` is the one deliberate omission: `StartsWith` covers a
#: `'abc%'` pattern but the two spell escapes and collation differently, and the write
#: path cannot afford "close enough".
_EXACT_NODES: tuple[type[exp.Expression], ...] = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.Is,
    exp.In,
    exp.Between,
    exp.Column,
)


def is_exactly_translatable(node: exp.Expression) -> bool:
    """True when a translation of `node`, *if one is produced*, selects the same rows.

    A gate on the node's shape, not a promise that it translates: `upper(v) = 'A'` is an
    `EQ` and passes here, then `translate_predicate` declines it because neither side is
    a bare column. Both have to hold, which is why `scope_predicate` asks both.

    Read pushdown never needs this: there, a predicate that matches *more* rows than
    the SQL does costs a little wasted I/O and nothing else, because the SQL re-applies
    the filter afterwards. The **write** path has no such second chance -- a row-level
    operation deletes the rows a PyIceberg predicate matches and writes back the rows a
    SQL predicate kept, so the two must agree row for row or the difference is data loss
    (predicate too wide) or duplication (too narrow).

    So this is the gate that lets the same predicate be used in both languages at once.
    It is a whitelist rather than a check: a node type is admitted only after its
    translation has been read and found exact.
    """
    if isinstance(node, exp.Paren):
        return is_exactly_translatable(node.this)
    if isinstance(node, (exp.And, exp.Or)):
        first, second = _operands(node)
        return is_exactly_translatable(first) and is_exactly_translatable(second)
    if isinstance(node, exp.Not):
        return is_exactly_translatable(node.this)
    return isinstance(node, _EXACT_NODES)


def scope_predicate(
    terms: Sequence[exp.Expression],
    resolver: ColumnResolver,
    schema: IcebergSchema,
    *,
    exact_only: bool = False,
) -> tuple[BooleanExpression, list[exp.Expression], list[exp.Expression]]:
    """Split `terms` into the ones this table can be pruned by and the ones it cannot.

    Returns the combined PyIceberg predicate, the terms it was built from, and the
    terms left behind. The kept terms are returned as the caller's own nodes, so a
    caller that needs the predicate in *both* languages -- which the row-level write
    path does -- gets a SQL form guaranteed to be the same predicate rather than a
    second translation that might drift from the first.

    Every conjunct is bound one at a time, so a literal PyIceberg dislikes costs only
    its own term rather than the whole predicate.
    """
    predicate: BooleanExpression = AlwaysTrue()
    kept: list[exp.Expression] = []
    dropped: list[exp.Expression] = []
    for term in terms:
        translated = None
        if resolver.owns_every_column(term) and not (
            exact_only and not is_exactly_translatable(term)
        ):
            translated = translate_predicate(term, resolver)
            if translated is not None and not binds_against(translated, schema):
                translated = None
        if translated is None:
            dropped.append(term)
            continue
        kept.append(term)
        predicate = translated if isinstance(predicate, AlwaysTrue) else And(predicate, translated)
    return predicate, kept, dropped


def null_padded_aliases(expression: exp.Expression) -> set[str]:
    """Aliases in this scope that an outer join can fill with NULLs.

    The right of a `LEFT JOIN`, everything left of a `RIGHT JOIN`, and both sides of a
    `FULL JOIN`. A row of NULLs there is manufactured by the join, not read from the
    table -- which is what makes a `WHERE` conjunct over it something other than a
    filter on its rows.
    """
    if not isinstance(expression, exp.Select):
        return set()
    padded: set[str] = set()
    seen: list[str] = []
    source = _from_table(expression)
    if source is not None:
        seen.append(source.alias_or_name)
    for join in expression.args.get("joins") or []:
        side = str(join.args.get("side") or "").upper()
        right = join.this.alias_or_name
        if side in ("LEFT", "FULL"):
            padded.add(right)
        if side in ("RIGHT", "FULL"):
            padded.update(seen)
        seen.append(right)
    return padded


def _from_table(expression: exp.Select) -> exp.Expression | None:
    """The `FROM` relation, under whichever key this sqlglot spells it.

    sqlglot 30 renamed the argument to `from_` because `from` is a Python keyword
    (FINDINGS.md 2.3), and `args.get("from")` does not raise when it is wrong -- it
    returns None, which reads as "this query has no FROM clause". That silence cost a
    correct answer once already: `null_padded_aliases` below never saw the left-hand
    table, so the left of a `RIGHT`/`FULL JOIN` was never marked null-padded and the
    anti-join of 1.10 came back with every row. Both spellings are read so a future
    rename fails loudly by returning nothing at all rather than quietly here.
    """
    clause = expression.args.get("from_") or expression.args.get("from")
    return clause.this if clause is not None else None


def join_predicates(expression: exp.Expression) -> dict[str, list[exp.Expression]]:
    """`ON` conjuncts that are true filters on a table's own rows, keyed by alias.

    A `WHERE` clause is not the only place a filter ends up. sqlglot's
    `pushdown_predicates` folds a `WHERE` conjunct over an inner-joined table *into
    that join's `ON` clause*, so a query written with the filter in `WHERE` arrives
    here with an empty `WHERE` and the filter one level down -- and reading only the
    `WHERE` meant the same predicate pruned three files as a `LEFT JOIN` and none as
    an `INNER JOIN` (FINDINGS.md 3.5).

    Which side an `ON` clause filters is the whole subtlety, because a join preserves
    one side and filters the other:

        INNER / CROSS   both sides -- an unmatched row of either is simply gone
        LEFT            the right only; left rows survive unmatched, null-padded
        RIGHT           the left only, for the mirror-image reason
        FULL            neither

    On the filtered side the conjunct is applied to rows read from the table, before
    any null-padding, so -- unlike a `WHERE` conjunct over a null-padded alias -- it
    needs no `is_null_rejecting` gate. On the preserved side it is not a filter at
    all and must not prune. Anything whose shape is not on that list (a semi or anti
    join, a lateral) yields nothing, which costs pruning and never correctness.
    """
    if not isinstance(expression, exp.Select):
        return {}

    terms: dict[str, list[exp.Expression]] = {}
    left: list[str] = []
    source = _from_table(expression)
    if source is not None:
        left.append(source.alias_or_name)

    for join in expression.args.get("joins") or []:
        right = join.this.alias_or_name
        side = str(join.args.get("side") or "").upper()
        kind = str(join.args.get("kind") or "").upper()
        on = join.args.get("on")

        filtered: list[str] = []
        if kind in ("", "INNER", "OUTER"):
            if side == "":
                filtered = [*left, right]
            elif side == "LEFT":
                filtered = [right]
            elif side == "RIGHT":
                filtered = list(left)
            # FULL preserves both sides, so its ON clause filters neither.

        if on is not None:
            for alias in filtered:
                terms.setdefault(alias, []).extend(conjuncts(on))
        left.append(right)

    return terms


def is_null_rejecting(node: exp.Expression) -> bool:
    """True when `node` cannot be satisfied by a row of NULLs.

    The gate on pushing a `WHERE` conjunct into a table an outer join can null-pad.
    `WHERE b.id IS NULL` over the right of a `LEFT JOIN` is the anti-join idiom: it
    selects the rows where the join found **nothing**, so pushing it into `b`'s scan
    prunes away exactly the files that make the answer right, and the query returns
    every left row instead of the unmatched ones. Reading less is normally free; here
    it changes the answer, which is the one case pushdown must refuse.

    Conservative in the safe direction -- an unrecognised shape is treated as *not*
    null-rejecting, which costs pruning and never correctness.
    """
    if isinstance(node, exp.Paren):
        return is_null_rejecting(node.this)
    if isinstance(node, exp.And):
        first, second = _operands(node)
        return is_null_rejecting(first) or is_null_rejecting(second)
    if isinstance(node, exp.Or):
        first, second = _operands(node)
        return is_null_rejecting(first) and is_null_rejecting(second)
    if isinstance(node, exp.Not):
        # `IS NOT NULL` is the one negation that plainly rejects a NULL row.
        inner = node.this
        return isinstance(inner, exp.Is) and isinstance(inner.expression, exp.Null)
    if isinstance(node, exp.Is):
        return not isinstance(node.expression, exp.Null)
    return True


def _unattributed(scope: Any, resolver: ColumnResolver) -> list[str]:
    """Unqualified columns in `scope`, classified into what they actually refer to.

    `qualify` deliberately leaves one kind of column unqualified: a reference from
    `ORDER BY` or `HAVING` to a name in the *output* list, as in `... GROUP BY
    VendorID ORDER BY VendorID`. Those are references to the projection, not to the
    table, and treating them as unattributable would give up projection pushdown on
    the single most common analytic query shape there is.

    Returns the table columns such references resolve to; raises the caller's
    "read everything" case by returning None only when a name is neither.
    """
    output_names = {name.lower() for name in scope.expression.named_selects}
    resolved: list[str] = []
    for column in scope.unqualified_columns:
        name = resolver.name(column)
        if name is not None:
            # Also a real column of ours. Including it is a superset of what the
            # query needs, which is the safe direction.
            resolved.append(name)
        elif column.name.lower() not in output_names:
            raise _Unattributable(column.sql(dialect="duckdb"))
    return resolved


class _Unattributable(Exception):
    """An unqualified column that is neither ours nor an output reference."""


def _projected_columns(scope: Any, alias: str, resolver: ColumnResolver) -> tuple[str, ...] | None:
    """The columns `alias` must produce in `scope`, or None for "all of them"."""
    # A surviving star is a reference we cannot attribute to a table at all, so the
    # only safe answer is every column.
    if scope.stars:
        return None
    try:
        extra = _unattributed(scope, resolver)
    except _Unattributable:
        return None
    referenced = list(scope.source_columns(alias))
    names = [resolved for column in referenced if (resolved := resolver.name(column)) is not None]
    # A reference we could not resolve means our view of the scope is incomplete.
    if len(names) != len(referenced):
        return None
    return tuple(dict.fromkeys([*names, *extra]))


def extract_scan_requests(
    optimized: exp.Expression, sources: Mapping[str, ScanSource]
) -> PlanAnnotations:
    """Read the optimized tree and record what each table reference needs.

    Every scope is walked rather than just the outermost one, so a table inside a
    CTE or a derived table that survived optimisation is pruned like any other.
    """
    annotations = PlanAnnotations(optimized)

    for scope in traverse_scope(optimized):
        terms = conjuncts(scope.expression.args.get("where"))
        scope_expression = as_expression(scope.expression)
        padded = null_padded_aliases(scope_expression)
        on_terms = join_predicates(scope_expression)

        for alias, node in scope.sources.items():
            if not isinstance(node, exp.Table) or isinstance(node.this, exp.Func):
                continue
            source = sources.get(source_key(node, versioned=True))
            if source is None:
                continue

            schema = source.resolved.table.schema()
            resolver = ColumnResolver(
                alias, {field.name: field.field_type for field in schema.fields}
            )

            # A table an outer join can null-pad only accepts conjuncts that a row of
            # NULLs could not satisfy. See `is_null_rejecting`.
            usable = (
                [term for term in terms if is_null_rejecting(term)] if alias in padded else terms
            )
            # An `ON` conjunct on the side the join filters is a filter on this
            # table's own rows, whether or not the other side null-pads it.
            usable = [*usable, *on_terms.get(alias, [])]
            predicate, _, dropped = scope_predicate(usable, resolver, schema)
            # A term mentioning another table is not "unpushed" for this one, it is
            # simply none of its business, so it is not reported against it.
            unpushed = [
                term.sql(dialect="duckdb") for term in dropped if resolver.owns_every_column(term)
            ]

            annotations.annotate(
                node,
                ScanRequest(
                    source=source,
                    columns=_projected_columns(scope, alias, resolver),
                    predicate=predicate,
                    unpushed=tuple(unpushed),
                ),
            )

    return annotations

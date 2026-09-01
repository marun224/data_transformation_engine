"""`GroupedData` -- what `df.groupBy(...)` returns, and the aggregation it builds.

A `GroupedData` holds no plan of its own. It is a pending `GROUP BY`: the frame it came
from plus the grouping expressions. Only `agg()` (or one of the shortcuts that calls it)
produces a DataFrame, which keeps the lazy contract of P3 -- `groupBy` alone runs
nothing and resolves no schema.

**Output column order is grouping keys first, then aggregates**, which is what the
reference engine does and what a script indexing `row[0]` depends on.

The projection is built here rather than by `DataFrame.select` because a grouped
projection has a rule a plain one does not: every non-aggregate output must also be a
grouping key. SQL enforces that itself, so a mistake surfaces as a binder error from
`_analyze` rather than a wrong answer -- which is why this module does not re-check it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from sqlglot import exp

from icetl.errors import EngineTypeError, EngineValueError, UnsupportedFeatureError
from icetl.plan.builder import as_expression
from icetl.sql.column import Column

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame

__all__ = ["GroupedData"]

#: `agg({"amount": "sum"})` -- the dict form maps a column name to a function name.
#: Spelled here rather than resolved through `getattr(F, name)` so that an arbitrary
#: name cannot reach an arbitrary callable.
_DICT_FORM_FUNCTIONS = frozenset(
    {
        "avg",
        "collect_list",
        "collect_set",
        "count",
        "first",
        "last",
        "max",
        "mean",
        "min",
        "stddev",
        "sum",
        "variance",
    }
)


#: The reference caps a pivot at `spark.sql.pivotMaxValues` columns. The same guard is
#: worth having for the same reason: a pivot on a high-cardinality column does not fail,
#: it succeeds and returns thousands of columns.
_PIVOT_MAX_VALUES = 10_000


@dataclass(frozen=True)
class _Pivot:
    """A pending pivot: the expression to switch on, and the values to switch to."""

    expression: exp.Expression
    values: list[Any]


#: Grouping kind -> the `exp.Group` argument and node that spells it. `groupBy` is the
#: plain form; `rollup` and `cube` are grouping *sets*, which produce extra super-
#: aggregate rows with NULL standing in for each rolled-up key.
_GROUPING_SETS: dict[str, tuple[str, type[exp.Expression]]] = {
    "rollup": ("rollup", exp.Rollup),
    "cube": ("cube", exp.Cube),
}


class GroupedData:
    """The result of `df.groupBy(...)`, `df.rollup(...)` or `df.cube(...)`."""

    def __init__(
        self,
        df: DataFrame,
        grouping: list[exp.Expression],
        names: list[str],
        kind: str = "groupBy",
        pivot: _Pivot | None = None,
    ) -> None:
        self._df = df
        self._grouping = grouping
        self._names = names
        self._kind = kind
        self._pivot = pivot

    def __repr__(self) -> str:
        keys = ", ".join(self._names) if self._names else "<global>"
        label = "GroupedData" if self._kind == "groupBy" else f"GroupedData({self._kind})"
        return f"{label}[{keys}]"

    # -- the one method that builds a plan ---------------------------------

    def agg(self, *exprs: Any, **kwargs: Any) -> DataFrame:
        """Aggregate the group, returning grouping keys followed by the aggregates.

        Accepts columns (`agg(F.sum("amount"))`), the dict form
        (`agg({"amount": "sum"})`), or keyword aliases (`agg(total=F.sum("amount"))`).
        """
        if kwargs and exprs:
            raise EngineValueError(
                "agg() takes either positional expressions or keyword aliases, not both."
            )

        aggregates: list[exp.Expression] = []
        if kwargs:
            for alias, item in kwargs.items():
                if not isinstance(item, Column):
                    raise EngineTypeError(
                        f"agg({alias}=...) expects a Column, got {type(item).__name__}."
                    )
                aggregates.append(
                    as_expression(exp.alias_(item._expression.copy(), alias, quoted=True))
                )
        else:
            items = list(exprs)
            if len(items) == 1 and isinstance(items[0], dict):
                aggregates = self._from_dict(items[0])
            else:
                if len(items) == 1 and isinstance(items[0], (list, tuple)):
                    items = list(items[0])
                if not items:
                    raise EngineValueError("agg() needs at least one aggregate expression.")
                aggregates = [self._aggregate(item) for item in items]

        return self._build(aggregates)

    # -- pivot ---------------------------------------------------------------

    def pivot(self, pivot_col: str, values: list[Any] | None = None) -> GroupedData:
        """Turn the distinct values of `pivot_col` into columns.

        **This is the one transformation that runs a query.** With `values` omitted,
        the distinct values have to be known before the projection can be written, so
        they are fetched now -- exactly as the reference engine does, and for the same
        reason. Pass `values` explicitly to keep the call lazy, and to fix the column
        order regardless of what the data happens to hold.

        Compiled as conditional aggregation (`sum(CASE WHEN k = 'x' THEN v END)`) rather
        than through DuckDB's own `PIVOT`, which is a statement rather than an
        expression and would not compose with the rest of a plan.
        """
        if self._pivot is not None:
            raise EngineValueError("pivot() cannot be applied twice to the same grouping.")
        if self._kind != "groupBy":
            raise EngineValueError(f"pivot() cannot be combined with {self._kind}().")
        if not isinstance(pivot_col, str):
            raise EngineTypeError(f"pivot() expects a column name, got {type(pivot_col).__name__}.")

        expression = self._df._column_ref(pivot_col)._expression.copy()
        resolved = list(values) if values is not None else self._distinct_values(expression)
        if values is not None and not resolved:
            raise EngineValueError("pivot() needs at least one value.")
        if len(resolved) > _PIVOT_MAX_VALUES:
            raise EngineValueError(
                f"pivot() would produce {len(resolved)} columns, over the limit of "
                f"{_PIVOT_MAX_VALUES}. Pass `values` to choose the ones you want."
            )
        return GroupedData(
            self._df, self._grouping, self._names, self._kind, _Pivot(expression, resolved)
        )

    def _distinct_values(self, expression: exp.Expression) -> list[Any]:
        """The distinct values of the pivot column, ordered so the output is stable.

        NULL sorts last and becomes a column of its own, as it does in the reference --
        a pivot key of NULL is a group, not an absence.
        """
        plan = self._df._rebase_projection()
        plan.set(
            "expressions", [as_expression(exp.alias_(expression.copy(), "value", quoted=True))]
        )
        plan.set("distinct", exp.Distinct())
        rows = self._df._derive(cast(exp.Expression, plan)).collect()
        return sorted((row[0] for row in rows), key=lambda v: (v is None, str(v)))

    # -- shortcuts ----------------------------------------------------------

    def count(self) -> DataFrame:
        """Rows per group, in a column named `count`."""
        counted = exp.Count(this=exp.Star())
        return self._build([as_expression(exp.alias_(counted, "count", quoted=True))])

    def sum(self, *cols: str) -> DataFrame:
        return self._numeric("sum", cols)

    def avg(self, *cols: str) -> DataFrame:
        return self._numeric("avg", cols)

    #: The reference engine exposes both spellings.
    mean = avg

    def min(self, *cols: str) -> DataFrame:
        return self._numeric("min", cols)

    def max(self, *cols: str) -> DataFrame:
        return self._numeric("max", cols)

    # -- internals ----------------------------------------------------------

    def _numeric(self, function: str, cols: tuple[str, ...]) -> DataFrame:
        """`gd.sum("a", "b")` -- one aggregate per named column.

        With no columns the reference engine aggregates every numeric column; that
        needs the frame's types, so it is refused rather than guessed at.
        """
        if not cols:
            raise UnsupportedFeatureError(
                f"GroupedData.{function}() over every numeric column",
                phase="Phase 4",
                hint=f"Name the columns, e.g. .{function}('amount')",
            )
        from icetl.sql import functions as F

        builder = getattr(F, function)
        return self.agg(*[builder(name) for name in cols])

    def _from_dict(self, mapping: dict[Any, Any]) -> list[exp.Expression]:
        from icetl.sql import functions as F

        aggregates: list[exp.Expression] = []
        for column, function in mapping.items():
            if not isinstance(column, str) or not isinstance(function, str):
                raise EngineTypeError(
                    "agg({column: function}) takes strings on both sides, got "
                    f"{type(column).__name__}: {type(function).__name__}."
                )
            if function.lower() not in _DICT_FORM_FUNCTIONS:
                raise EngineValueError(
                    f"agg() does not support the function {function!r}. "
                    f"Known: {', '.join(sorted(_DICT_FORM_FUNCTIONS))}. "
                    f"Use the expression form for anything else, e.g. F.{function}('{column}')."
                )
            aggregates.append(self._aggregate(getattr(F, function.lower())(column)))
        return aggregates

    def _aggregate(self, item: Any) -> exp.Expression:
        """One aggregate in projection position, named as the reference engine names it."""
        if isinstance(item, str):
            raise EngineTypeError(
                f"agg() takes aggregate expressions, not the column name {item!r}. "
                f"Wrap it, e.g. F.sum({item!r})."
            )
        if not isinstance(item, Column):
            raise EngineTypeError(f"agg() expects a Column, got {type(item).__name__}.")
        expression = item._expression.copy()
        if isinstance(expression, exp.Alias):
            return expression
        return as_expression(exp.alias_(expression, item._output_name, quoted=True))

    def _build(self, aggregates: list[exp.Expression]) -> DataFrame:
        """`SELECT <keys>, <aggregates> FROM <plan> GROUP BY <keys>`."""
        plan = self._df._rebase_projection()
        projection = [
            as_expression(exp.alias_(key.copy(), name, quoted=True))
            for key, name in zip(self._grouping, self._names, strict=True)
        ]
        if self._pivot is not None:
            aggregates = self._pivoted(aggregates)
        plan.set("expressions", projection + aggregates)
        if self._grouping:
            plan.set("group", self._group_clause())
        return self._df._derive(cast(exp.Expression, plan))

    def _pivoted(self, aggregates: list[exp.Expression]) -> list[exp.Expression]:
        """One aggregate per (value, aggregate) pair, each seeing only its own rows.

        Naming follows the reference: with a single aggregate the column is the value
        alone (`a`), and with several it is `value_aggregate` (`a_total`) -- because with
        one aggregate the value is enough to identify the column, and with two it is not.
        """
        assert self._pivot is not None
        columns: list[exp.Expression] = []
        for value in self._pivot.values:
            condition = _matches(self._pivot.expression, value)
            for aggregate in aggregates:
                label = _pivot_label(value)
                name = label if len(aggregates) == 1 else f"{label}_{aggregate.alias_or_name}"
                bare = as_expression(aggregate.unalias())
                restricted = _restrict_aggregates(bare, condition)
                columns.append(as_expression(exp.alias_(restricted, name, quoted=True)))
        return columns

    def _group_clause(self) -> exp.Group:
        """The GROUP BY clause for this kind of grouping.

        `rollup` and `cube` go in their own `exp.Group` argument rather than in
        `expressions` -- a grouping set replaces the plain key list, it does not
        accompany one, and putting keys in both would group by them twice.
        """
        keys = [key.copy() for key in self._grouping]
        if self._kind == "groupBy":
            return exp.Group(expressions=keys)
        argument, node = _GROUPING_SETS[self._kind]
        return exp.Group(**{argument: [node(expressions=keys)]})


def _pivot_label(value: Any) -> str:
    """The column name a pivot value produces. NULL becomes the string `null`."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _matches(expression: exp.Expression, value: Any) -> exp.Expression:
    """`expression IS NOT DISTINCT FROM value`.

    Null-safe deliberately: `k = NULL` is never true, so a plain `=` would give every
    row of a NULL pivot group an empty column rather than its own.
    """
    from icetl.sql.column import to_literal

    return exp.NullSafeEQ(this=expression.copy(), expression=to_literal(value))


def _restrict_aggregates(expression: exp.Expression, condition: exp.Expression) -> exp.Expression:
    """Rewrite every aggregate in `expression` to see only the rows `condition` selects.

    `sum(v)` becomes `sum(CASE WHEN cond THEN v END)`, which is how the reference
    compiles a pivot too. Rewriting each aggregate rather than the whole expression is
    what makes a composite like `sum(a) / count(b)` come out right: the division has to
    happen after both sides are restricted, not before.

    `count(*)` has no argument to restrict, so it counts a literal instead --
    `count(CASE WHEN cond THEN 1 END)`, which still counts rows and still skips the
    ones the condition excludes.
    """

    def rewrite(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.AggFunc):
            return node
        inner = node.this
        if inner is None or isinstance(inner, exp.Star):
            inner = exp.Literal.number(1)
        node.set("this", exp.Case(ifs=[exp.If(this=condition.copy(), true=inner.copy())]))
        return node

    return expression.copy().transform(rewrite, copy=False)

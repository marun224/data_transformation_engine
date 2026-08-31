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


class GroupedData:
    """The result of `df.groupBy(...)`, awaiting an aggregation."""

    def __init__(
        self,
        df: DataFrame,
        grouping: list[exp.Expression],
        names: list[str],
    ) -> None:
        self._df = df
        self._grouping = grouping
        self._names = names

    def __repr__(self) -> str:
        keys = ", ".join(self._names) if self._names else "<global>"
        return f"GroupedData[{keys}]"

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
        plan.set("expressions", projection + aggregates)
        if self._grouping:
            plan.set("group", exp.Group(expressions=[key.copy() for key in self._grouping]))
        return self._df._derive(cast(exp.Expression, plan))

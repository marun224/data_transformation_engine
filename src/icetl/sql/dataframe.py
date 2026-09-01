"""`DataFrame` -- a lazy plan plus the actions that run it.

Every transformation returns a new DataFrame wrapping a new sqlglot tree. Nothing
touches DuckDB until an action is called, with one deliberate exception: the *schema*
is resolved eagerly, because the reference engine reports an unknown column at `df.select("typo")`
rather than at `df.collect()`, and scripts rely on that.

How a transformation decides whether to extend the current SELECT or nest it:

    extend the projection   when the plan is `SELECT * FROM ...` -- SQL evaluates
                            WHERE against the FROM, so an existing WHERE is fine
    extend the filter       when the projection is still `*` and nothing downstream
                            of WHERE (GROUP BY, ORDER BY, LIMIT, DISTINCT) exists
    extend the limit        when there is no LIMIT or OFFSET yet
    otherwise               nest: `SELECT ... FROM (plan) AS _qN`

Nesting is always correct; extending is the readability optimisation. Getting the
guards wrong would produce wrong answers, so each is narrow and separately tested.
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any, cast

import sqlglot
from sqlglot import exp

from icetl.compat import SQL_DIALECT
from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    EngineValueError,
    ParseException,
    UnsupportedFeatureError,
)
from icetl.exec.result import format_show, to_pandas, to_rows
from icetl.plan.builder import as_expression, wrap_as_subquery
from icetl.sql.column import Column
from icetl.sql.generators import SchemaAwareColumn
from icetl.sql.group import GroupedData
from icetl.sql.na import DataFrameNaFunctions
from icetl.sql.stat import DataFrameStatFunctions
from icetl.types import Row, StructType

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    import pyarrow as pa

    from icetl.plan.builder import ScanSource
    from icetl.sql.session import Session
    from icetl.sql.writer import DataFrameWriter

__all__ = ["DataFrame"]

#: The column `randomSplit` draws into. Named so it cannot collide with a real one.
_SPLIT_COLUMN = "_icetl_split"

_DEFAULT_SHOW_ROWS = 20
_DEFAULT_TRUNCATE = 20

# Clauses whose presence means a new projection or filter cannot simply be merged
# into the existing SELECT.
_POST_FILTER_CLAUSES = (
    "group",
    "order",
    "limit",
    "offset",
    "distinct",
    "having",
    "qualify",
    # `USING SAMPLE` draws from the rows the FROM and WHERE produced, so merging a new
    # filter into a sampled SELECT would filter *before* the draw instead of after it --
    # a different query with a plausible-looking answer.
    "sample",
)

#: The reference engine's `how` spellings -> (side, kind) on a sqlglot Join. Underscores
#: are stripped before lookup, so `left_outer` and `leftouter` both land here.
_JOIN_KINDS: dict[str, tuple[str | None, str | None]] = {
    "inner": (None, None),
    "cross": (None, "CROSS"),
    "left": ("LEFT", None),
    "leftouter": ("LEFT", "OUTER"),
    "right": ("RIGHT", None),
    "rightouter": ("RIGHT", "OUTER"),
    "outer": ("FULL", "OUTER"),
    "full": ("FULL", "OUTER"),
    "fullouter": ("FULL", "OUTER"),
    "semi": (None, "SEMI"),
    "leftsemi": (None, "SEMI"),
    "anti": (None, "ANTI"),
    "leftanti": (None, "ANTI"),
}

#: Set-operation method -> (sqlglot node, de-duplicates?, the name the reference engine
#: uses for it when a column count does not line up).
#:
#: `union` maps to `UNION ALL`, which is the one place the reference's naming misleads:
#: its `union` **keeps** duplicates, and `distinct()` is a separate step. `subtract` and
#: `intersect` do de-duplicate; the `*All` spellings are the multiset forms.
_SET_OPERATIONS: dict[str, tuple[type[exp.SetOperation], bool, str]] = {
    "union": (exp.Union, False, "Union"),
    "unionByName": (exp.Union, False, "Union"),
    "intersect": (exp.Intersect, True, "Intersect"),
    "intersectAll": (exp.Intersect, False, "Intersect"),
    "exceptAll": (exp.Except, False, "Except"),
    "subtract": (exp.Except, True, "Except"),
}

#: Clauses that bind to a whole set operation rather than to the branch that spells
#: them, so a branch carrying one has to be nested before it can be combined.
_BRANCH_NESTING_CLAUSES = ("limit", "offset", "order")


class DataFrame:
    """A lazily evaluated table of rows."""

    def __init__(
        self,
        session: Session,
        plan: exp.Expression,
        sources: dict[str, ScanSource],
    ) -> None:
        self._session = session
        self._plan = plan
        self._sources = sources
        self._schema: StructType | None = None
        #: Set by `cache()`; what `unpersist()` releases.
        self._cache_name: str | None = None

    # -- construction ------------------------------------------------------

    def _derive(self, plan: exp.Expression) -> DataFrame:
        """A new DataFrame over `plan`, sharing this one's session and sources."""
        return DataFrame(self._session, plan, self._sources)

    def _is_star_projection(self) -> bool:
        plan = self._plan
        return (
            isinstance(plan, exp.Select)
            and _has_from_clause(plan)
            and len(plan.expressions) == 1
            and isinstance(plan.expressions[0], exp.Star)
            and not plan.args.get("joins")
            and not plan.args.get("laterals")
        )

    def _projection_is_replaceable(self) -> bool:
        """True when the projection list can be swapped without nesting.

        Wider than `_is_star_projection` in one way: **joins are allowed**. Replacing
        `SELECT *` with `SELECT x.a, y.b` over `FROM x JOIN y` leaves the FROM and its
        aliases exactly where they were, so the new projection can still see them.
        Nesting here would hide them -- which is what made `df.alias("x").join(...)
        .select(col("x.a"))` fail with an unresolved `x`.
        """
        plan = self._plan
        return (
            isinstance(plan, exp.Select)
            and _has_from_clause(plan)
            and len(plan.expressions) == 1
            and isinstance(plan.expressions[0], exp.Star)
            and not plan.args.get("laterals")
        )

    def _rebase_projection(self) -> exp.Select:
        """A SELECT whose projection list is safe to replace wholesale."""
        if self._projection_is_replaceable() and not self._has_any(_POST_FILTER_CLAUSES):
            return cast(exp.Select, self._plan.copy())
        return wrap_as_subquery(self._plan, self._session._next_alias())

    def _has_any(self, keys: tuple[str, ...]) -> bool:
        plan = self._plan
        return isinstance(plan, exp.Select) and any(plan.args.get(key) for key in keys)

    # -- schema ------------------------------------------------------------

    @property
    def schema(self) -> StructType:
        """The output schema, resolved on first access and cached."""
        if self._schema is None:
            self._schema = self._session._analyze(self._plan, self._sources)
        return self._schema

    @property
    def columns(self) -> list[str]:
        return [field.name for field in self.schema.fields]

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        return [(f.name, f.dataType.simpleString()) for f in self.schema.fields]

    def printSchema(self, level: int | None = None) -> None:
        """Print the schema as a tree, optionally stopping after `level` levels."""
        print(self.schema.treeString(level), end="")

    def isLocal(self) -> bool:
        """Always True: everything runs in this process."""
        return True

    # -- column access -----------------------------------------------------

    def _resolve_name(self, name: str) -> str | None:
        """Match `name` against the schema case-insensitively, as the reference engine does.

        Returns the column's own spelling, or None when there is no match.
        """
        lowered = name.lower()
        for field in self.schema.fields:
            if field.name.lower() == lowered:
                return field.name
        return None

    def __getattr__(self, name: str) -> Column:
        # Reached only for names that are not real attributes, so the internals above
        # never route through here.
        if name.startswith("_"):
            raise AttributeError(name)
        resolved = self._resolve_name(name)
        if resolved is None:
            raise AttributeError(
                f"{name!r} is not a column of this DataFrame. Columns: {', '.join(self.columns)}"
            )
        return Column(exp.column(resolved, quoted=True))

    def __getitem__(self, item: Any) -> Any:
        """`df["a"]`, `df[0]`, `df[["a", "b"]]`, and `df[df.a > 1]`, as in the reference API."""
        if isinstance(item, str):
            resolved = self._resolve_name(item)
            if resolved is None:
                raise AnalysisException(
                    f"Column {item!r} does not exist. Columns: {', '.join(self.columns)}"
                )
            return Column(exp.column(resolved, quoted=True))
        if isinstance(item, Column):
            return self.filter(item)
        if isinstance(item, (list, tuple)):
            return self.select(*item)
        if isinstance(item, int):
            return Column(exp.column(self.columns[item], quoted=True))
        raise EngineTypeError(f"Cannot index a DataFrame with {type(item).__name__}.")

    # -- transformations ---------------------------------------------------

    def _projection(self, item: Any) -> exp.Expression:
        """One entry of a `select` list, in projection position.

        A bare string names a column here (unlike in operator position, where it is a
        literal), and an unaliased expression gets the reference engine's generated name attached so
        the output column is called what the reference engine would call it.
        """
        if isinstance(item, str):
            column = Column(exp.Star()) if item == "*" else self._column_ref(item)
        elif isinstance(item, Column):
            column = item
        else:
            raise EngineTypeError(
                f"select() takes column names or Column objects, got {type(item).__name__}."
            )

        expression = column._expression.copy()
        if isinstance(expression, (exp.Alias, exp.Star)) or (
            isinstance(expression, exp.Column) and isinstance(expression.this, exp.Star)
        ):
            return expression
        if isinstance(expression, exp.Column) and not expression.table:
            # An unqualified reference already carries the reference engine's name for it, so an
            # alias would only add `"a" AS "a"` noise to every generated query. A
            # *qualified* one still needs it: the reference engine names `t.a` just `a`.
            return expression
        return as_expression(exp.alias_(expression, column._output_name, quoted=True))

    def _column_ref(self, name: str) -> Column:
        """`col(name)`, but qualified names and `t.*` pass through untouched."""
        if "." in name or name.endswith("*"):
            from icetl.sql.functions import col

            return col(name)
        resolved = self._resolve_name(name)
        if resolved is None:
            raise AnalysisException(
                f"Column {name!r} does not exist. Columns: {', '.join(self.columns)}"
            )
        return Column(exp.column(resolved, quoted=True))

    def select(self, *cols: Any) -> DataFrame:
        """Project a set of columns or expressions.

        A generator (`explode`, `posexplode`, `inline`, `json_tuple`) expands here rather
        than in `F`, because how many columns it produces depends on the *type* of what
        it is exploding -- a list gives one, a map gives two, `inline` gives one per
        field -- and the type is only knowable with the frame in hand.
        """
        items = _flatten(cols)
        if not items:
            raise EngineValueError("select() needs at least one column.")

        projection: list[exp.Expression] = []
        generators = 0
        for item in items:
            if isinstance(item, SchemaAwareColumn):
                generators += 1 if item._is_generator else 0
                if generators > 1:
                    raise EngineValueError(
                        "select() takes at most one generator (explode, posexplode, "
                        "inline, json_tuple). Two would have to agree on how many rows "
                        "to produce, and there is no answer to that."
                    )
                projection.extend(item._expand(self))
            else:
                projection.append(self._projection(item))

        plan = self._rebase_projection()
        plan.set("expressions", projection)
        return self._derive(plan)

    def _type_of(self, expression: exp.Expression) -> Any:
        """The type this expression has against *this* frame.

        Resolved by analysing a one-column probe, which is the same zero-row binding the
        schema property uses -- so it costs a bind, not a query.
        """
        probe = self._rebase_projection()
        probe.set("expressions", [as_expression(exp.alias_(expression.copy(), "probe"))])
        return self._session._analyze(cast(exp.Expression, probe), self._sources).fields[0].dataType

    def selectExpr(self, *expr: str) -> DataFrame:
        """Project from the reference SQL dialect expression strings."""
        from icetl.sql.functions import expr as parse_expr

        return self.select(*[parse_expr(item) for item in _flatten(expr)])

    def filter(self, condition: Column | str) -> DataFrame:
        """Keep rows matching `condition`, given as a Column or a SQL string."""
        predicate = _to_predicate(condition)
        if self._is_star_projection() and not self._has_any(_POST_FILTER_CLAUSES):
            plan = cast(exp.Select, self._plan.copy())
        else:
            plan = wrap_as_subquery(self._plan, self._session._next_alias())
        # sqlglot's `where` ANDs with any existing predicate, which is what a second
        # `.filter()` on the same SELECT means.
        return self._derive(as_expression(plan.where(predicate, copy=False)))

    #: The reference engine exposes both spellings.
    where = filter

    def limit(self, num: int) -> DataFrame:
        """Keep at most `num` rows."""
        if not isinstance(num, int) or isinstance(num, bool):
            raise EngineTypeError(f"limit() expects an int, got {type(num).__name__}.")
        if num < 0:
            raise EngineValueError(f"limit() expects a non-negative count, got {num}.")
        if isinstance(self._plan, exp.Select) and not self._has_any(("limit", "offset")):
            plan = self._plan.copy()
        else:
            plan = wrap_as_subquery(self._plan, self._session._next_alias())
        return self._derive(plan.limit(num, copy=False))

    def withColumn(self, colName: str, col: Column) -> DataFrame:
        """Add a column, or replace one of the same name (case-insensitively)."""
        if not isinstance(colName, str):
            raise EngineTypeError(f"withColumn() expects a name, got {type(colName).__name__}.")
        if not isinstance(col, Column):
            raise EngineTypeError(
                f"withColumn() expects a Column, got {type(col).__name__}. "
                f"Wrap a plain value in F.lit()."
            )
        replacement = col.alias(colName)
        existing = self._resolve_name(colName)
        if existing is None:
            return self.select(*self.columns, replacement)
        return self.select(*[replacement if name == existing else name for name in self.columns])

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """Rename a column. Unknown names are ignored, as in the reference engine."""
        resolved = self._resolve_name(existing)
        if resolved is None:
            return self
        return self.select(
            *[
                Column(exp.column(name, quoted=True)).alias(new) if name == resolved else name
                for name in self.columns
            ]
        )

    def drop(self, *cols: str | Column) -> DataFrame:
        """Drop columns. Names that do not exist are ignored, as in the reference engine."""
        dropped: set[str] = set()
        for item in _flatten(cols):
            if isinstance(item, Column):
                expression = item._expression
                if not isinstance(expression, exp.Column):
                    raise EngineTypeError("drop() only accepts column references, not expressions.")
                name = expression.name
            elif isinstance(item, str):
                name = item
            else:
                raise EngineTypeError(
                    f"drop() expects names or Columns, got {type(item).__name__}."
                )
            resolved = self._resolve_name(name)
            if resolved is not None:
                dropped.add(resolved)

        remaining = [name for name in self.columns if name not in dropped]
        if not remaining:
            raise UnsupportedFeatureError(
                "Dropping every column, which leaves a DataFrame with no columns",
                phase="Phase 4",
            )
        if not dropped:
            return self
        return self.select(*remaining)

    # -- joins --------------------------------------------------------------

    def join(
        self,
        other: DataFrame,
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> DataFrame:
        """Join `other`, matching the reference engine's spellings of `how`.

        `on` decides how the key columns appear in the output, and the two forms differ:

            on="k"                  -> `USING (k)`, so **one** `k` column survives
            on=col("a.k") == col("b.k")  -> `ON ...`, so **both** are kept

        That is the reference behaviour, and the reason the string form is not just
        sugar for the Column form -- `divergence.md` records it. A semi or anti join
        emits the left side's columns only, whichever form is used.
        """
        if not isinstance(other, DataFrame):
            raise EngineTypeError(f"join() expects a DataFrame, got {type(other).__name__}.")
        if other._session is not self._session:
            raise EngineValueError("join() cannot combine DataFrames from different sessions.")
        if not isinstance(how, str):
            raise EngineTypeError(f"join() expects `how` as a string, got {type(how).__name__}.")

        normalised = how.strip().lower().replace("_", "")
        if normalised not in _JOIN_KINDS:
            raise EngineValueError(
                f"Unknown join type {how!r}. Known: {', '.join(sorted(set(_JOIN_KINDS)))}."
            )
        side, kind = _JOIN_KINDS[normalised]

        if on is None:
            # The reference engine turns a keyless join into a cartesian product.
            if normalised not in ("inner", "cross"):
                raise EngineValueError(
                    f"A {how!r} join needs an `on` condition; only inner and cross joins "
                    f"may be keyless."
                )
            side, kind = None, "CROSS"

        join_args: dict[str, Any] = {}
        if on is not None:
            keys = _join_keys(on)
            if keys is not None:
                join_args["using"] = [exp.column(key, quoted=True) for key in keys]
            elif isinstance(on, Column):
                join_args["on"] = on._expression.copy()
            else:
                raise EngineTypeError(
                    f"join() takes `on` as a column name, a list of names, or a Column, "
                    f"got {type(on).__name__}."
                )

        plan = self._as_join_base()
        relation = other._as_join_relation()
        plan.set(
            "joins",
            [
                *(plan.args.get("joins") or []),
                exp.Join(this=relation, side=side, kind=kind, **join_args),
            ],
        )
        return DataFrame(self._session, plan, {**self._sources, **other._sources})

    def crossJoin(self, other: DataFrame) -> DataFrame:
        """Every row of this frame paired with every row of `other`."""
        return self.join(other, on=None, how="cross")

    def _as_join_base(self) -> exp.Select:
        """This plan as a SELECT a join can be appended to, keeping any alias it has."""
        if self._is_joinable_select():
            return cast(exp.Select, self._plan.copy())
        return wrap_as_subquery(self._plan, self._session._next_alias())

    def _as_join_relation(self) -> exp.Expression:
        """This plan as the right-hand relation of a join.

        A frame that is exactly `SELECT * FROM (x) AS a` -- which is what `alias()`
        produces -- contributes `(x) AS a`, so `a.column` still resolves on the other
        side of the join. Anything else is nested under a generated alias.
        """
        if self._is_joinable_select():
            from_clause = _from_clause(cast(exp.Select, self._plan))
            if from_clause is not None:
                return as_expression(from_clause.this.copy())
        return exp.Subquery(
            this=self._plan.copy(),
            alias=exp.TableAlias(this=exp.to_identifier(self._session._next_alias())),
        )

    def _is_joinable_select(self) -> bool:
        """True when this plan is a bare `SELECT * FROM <one relation>`.

        Everything else -- a projection, a filter, an existing join -- has to be nested,
        because a join would otherwise change what those clauses see.
        """
        plan = self._plan
        return (
            isinstance(plan, exp.Select)
            and _has_from_clause(plan)
            and len(plan.expressions) == 1
            and isinstance(plan.expressions[0], exp.Star)
            and not plan.args.get("joins")
            and not plan.args.get("laterals")
            and not plan.args.get("where")
            and not self._has_any(_POST_FILTER_CLAUSES)
        )

    # -- grouping ----------------------------------------------------------

    def groupBy(self, *cols: Any) -> GroupedData:
        """Group by the given columns or expressions.

        Returns a `GroupedData`, which builds no plan until `agg()` is called -- so
        `groupBy` alone runs nothing and resolves no schema (P3).
        """
        return self._grouped(cols, "groupBy")

    #: The reference engine exposes both spellings.
    groupby = groupBy

    def rollup(self, *cols: Any) -> GroupedData:
        """Group by a rollup of the given columns: a hierarchy of subtotals.

        `rollup("a", "b")` produces the `(a, b)` groups, then the `(a)` subtotals, then
        one grand total -- `n + 1` grouping sets for `n` keys. A key that has been rolled
        up comes back as **NULL**, which is indistinguishable from a real NULL in the
        data; `F.grouping(col)` is what tells the two apart.
        """
        return self._grouped(cols, "rollup")

    def cube(self, *cols: Any) -> GroupedData:
        """Group by every combination of the given columns: `2**n` grouping sets.

        The same NULL caveat as `rollup`, and the same answer to it -- `F.grouping`.
        """
        return self._grouped(cols, "cube")

    def _grouped(self, cols: tuple[Any, ...], kind: str) -> GroupedData:
        items = _flatten(cols)
        grouping: list[exp.Expression] = []
        names: list[str] = []
        for item in items:
            if isinstance(item, str):
                column = self._column_ref(item)
            elif isinstance(item, Column):
                column = item
            else:
                raise EngineTypeError(
                    f"{kind}() takes column names or Column objects, got {type(item).__name__}."
                )
            expression = column._expression.copy()
            # An alias groups by the *expression* but is named by the alias, so the
            # GROUP BY clause must carry the underlying node, not the `AS` wrapper.
            if isinstance(expression, exp.Alias):
                names.append(expression.alias)
                expression = as_expression(expression.this)
            else:
                names.append(column._output_name)
            grouping.append(expression)
        return GroupedData(self, grouping, names, kind)

    def agg(self, *exprs: Any, **kwargs: Any) -> DataFrame:
        """Aggregate the whole frame, as `groupBy()` with no keys does."""
        return GroupedData(self, [], []).agg(*exprs, **kwargs)

    # -- set operations ----------------------------------------------------

    def union(self, other: DataFrame) -> DataFrame:
        """This frame's rows followed by `other`'s, matched **by position**.

        Two things surprise people here, and both are the reference engine's behaviour
        rather than ours:

        * **duplicates are kept.** `union` is SQL's `UNION ALL`. Chain `.distinct()`
          if you want SQL's `UNION`.
        * **columns line up by position, not by name.** Two frames with the same
          columns in a different order union into nonsense without complaint, because
          nothing about it is detectably wrong. `unionByName` is the safe spelling.

        Output column names come from *this* frame.
        """
        return self._set_operation(other, "union")

    #: The reference engine keeps both spellings, and both mean `UNION ALL`.
    unionAll = union

    def unionByName(self, other: DataFrame, allowMissingColumns: bool = False) -> DataFrame:
        """`union`, but lining columns up **by name** instead of by position.

        Names are matched case-insensitively, as elsewhere in the analyser. `other` is
        re-projected into this frame's column order, so the result's schema is this
        frame's.

        With `allowMissingColumns=False` the two frames must carry the same set of
        names, and anything else raises. With it True, a column present on only one
        side is kept and filled with NULL on the other; columns unique to `other` are
        appended after this frame's, in `other`'s order.

        The NULL filler is a bare `NULL`, not a cast one: a set operation takes each
        column's type from the branches that do have it, so DuckDB types the column
        from the other side and the two agree without us having to map the type.
        """
        self._check_set_operand(other, "unionByName")
        if not isinstance(allowMissingColumns, bool):
            raise EngineTypeError(
                "unionByName() expects allowMissingColumns as a bool, got "
                f"{type(allowMissingColumns).__name__}."
            )

        mine, theirs = self.columns, other.columns
        combined = list(mine) + [name for name in theirs if not _contains_name(mine, name)]

        if not allowMissingColumns:
            absent = [name for name in mine if not _contains_name(theirs, name)]
            added = [name for name in theirs if not _contains_name(mine, name)]
            if absent or added:
                raise AnalysisException(
                    "unionByName() needs both frames to carry the same column names. "
                    f"Missing on the right: {absent or 'none'}; "
                    f"missing on the left: {added or 'none'}. "
                    "Pass allowMissingColumns=True to fill the gaps with NULL."
                )

        left = self if mine == combined else self.select(*_aligned_to(mine, combined))
        right = other.select(*_aligned_to(theirs, combined))
        return left._combine(right, exp.Union, distinct=False)

    def intersect(self, other: DataFrame) -> DataFrame:
        """Distinct rows present in both frames. De-duplicates; NULL matches NULL."""
        return self._set_operation(other, "intersect")

    def intersectAll(self, other: DataFrame) -> DataFrame:
        """Rows present in both frames, keeping duplicates.

        A row appearing 3 times here and twice in `other` comes back twice: the
        multiset intersection, which is what SQL's `INTERSECT ALL` computes.
        """
        return self._set_operation(other, "intersectAll")

    def exceptAll(self, other: DataFrame) -> DataFrame:
        """Rows of this frame not in `other`, keeping duplicates.

        A row appearing 3 times here and once in `other` comes back twice -- the
        multiset difference. `subtract` is the de-duplicating form.
        """
        return self._set_operation(other, "exceptAll")

    def subtract(self, other: DataFrame) -> DataFrame:
        """Distinct rows of this frame not in `other`. De-duplicates; NULL matches NULL."""
        return self._set_operation(other, "subtract")

    def _set_operation(self, other: DataFrame, method: str) -> DataFrame:
        node_type, distinct, label = _SET_OPERATIONS[method]
        self._check_set_operand(other, method)

        # Resolved now rather than at the first action, because the reference engine
        # reports a width mismatch at the call that made it -- and because DuckDB's own
        # message for it names generated subquery aliases, not the user's frames.
        mine, theirs = len(self.columns), len(other.columns)
        if mine != theirs:
            raise AnalysisException(
                f"{label} can only be performed on tables with the same number of "
                f"columns, but the first table has {mine} column(s) and the second "
                f"table has {theirs} column(s)."
            )
        return self._combine(other, node_type, distinct=distinct)

    def _combine(
        self, other: DataFrame, node_type: type[exp.SetOperation], *, distinct: bool
    ) -> DataFrame:
        plan = node_type(
            this=self._as_set_branch(),
            expression=other._as_set_branch(),
            distinct=distinct,
        )
        return DataFrame(self._session, plan, {**self._sources, **other._sources})

    def _as_set_branch(self) -> exp.Expression:
        """This plan as one branch of a set operation, nested if it cannot stand bare.

        Two shapes have to be nested, and the first is the dangerous one:

        * **the branch is itself a set operation.** DuckDB binds `INTERSECT` tighter
          than `UNION ALL`, so inlining would turn `a.union(b).intersect(c)` into
          `a UNION ALL (b INTERSECT c)` -- a different query that runs perfectly well
          and answers wrongly. Nesting is what makes the Python call order the one
          that counts.
        * **the branch carries LIMIT, OFFSET or ORDER BY.** Those bind to the whole
          set operation, and DuckDB will not even parse them mid-branch.
        """
        plan = self._plan
        if isinstance(plan, exp.Select) and not self._has_any(_BRANCH_NESTING_CLAUSES):
            return plan.copy()
        return wrap_as_subquery(plan, self._session._next_alias())

    def _check_set_operand(self, other: DataFrame, method: str) -> None:
        if not isinstance(other, DataFrame):
            raise EngineTypeError(f"{method}() expects a DataFrame, got {type(other).__name__}.")
        if other._session is not self._session:
            raise EngineValueError(f"{method}() cannot combine DataFrames from different sessions.")

    # -- nulls ---------------------------------------------------------------

    @property
    def na(self) -> DataFrameNaFunctions:
        """The null-handling surface: `df.na.drop()`, `.fill()`, `.replace()`."""
        return DataFrameNaFunctions(self)

    def dropna(
        self,
        how: str = "any",
        thresh: int | None = None,
        subset: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """`df.na.drop(...)`, which the reference engine also spells this way."""
        return self.na.drop(how=how, thresh=thresh, subset=subset)

    def fillna(self, value: Any, subset: str | Sequence[str] | None = None) -> DataFrame:
        """`df.na.fill(...)`, which the reference engine also spells this way."""
        return self.na.fill(value, subset=subset)

    def replace(
        self,
        to_replace: Any,
        value: Any = None,
        subset: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """`df.na.replace(...)`, which the reference engine also spells this way."""
        return self.na.replace(to_replace, value=value, subset=subset)

    # -- statistics ----------------------------------------------------------

    @property
    def stat(self) -> DataFrameStatFunctions:
        """The statistics surface: `df.stat.corr()`, `.cov()`, `.crosstab()`, and so on."""
        return DataFrameStatFunctions(self)

    # -- de-duplication ---------------------------------------------------------

    def distinct(self) -> DataFrame:
        """Distinct rows, comparing every column. NULL matches NULL."""
        if isinstance(self._plan, exp.Select) and self._plan.args.get("distinct"):
            return self
        plan = self._rebase_distinct()
        plan.set("distinct", exp.Distinct())
        return self._derive(cast(exp.Expression, plan))

    def dropDuplicates(self, subset: Sequence[str] | None = None) -> DataFrame:
        """Distinct rows, or one arbitrary row per distinct combination of `subset`.

        **Which** row survives a `subset` de-duplication is not defined -- the reference
        does not promise one either, because on a partitioned engine it cannot. Rely on
        the *keys* being unique afterwards, not on which row carried them.

        Compiled as DuckDB's `DISTINCT ON`, which is the direct spelling of it.
        """
        if subset is None:
            return self.distinct()
        names = [subset] if isinstance(subset, str) else list(subset)
        if not names:
            return self.distinct()

        keys: list[exp.Expression] = []
        for name in names:
            if not isinstance(name, str):
                raise EngineTypeError(
                    f"dropDuplicates() takes column names, got {type(name).__name__}."
                )
            resolved = self._resolve_name(name)
            if resolved is None:
                raise AnalysisException(
                    f"Column {name!r} does not exist. Columns: {', '.join(self.columns)}"
                )
            keys.append(exp.column(resolved, quoted=True))

        plan = self._rebase_distinct()
        plan.set("distinct", exp.Distinct(on=exp.Tuple(expressions=keys)))
        return self._derive(cast(exp.Expression, plan))

    #: The reference engine exposes both spellings.
    drop_duplicates = dropDuplicates

    def _rebase_distinct(self) -> exp.Select:
        """A SELECT that DISTINCT can be added to without changing what it means.

        Nested whenever a row-limiting or row-selecting clause is already there: SQL
        applies DISTINCT *before* LIMIT, so folding it into `df.limit(3)` would
        de-duplicate the whole table and then take three, rather than taking three rows
        and de-duplicating those.
        """
        plan = self._plan
        if (
            isinstance(plan, exp.Select)
            and not plan.args.get("distinct")
            and not self._has_any(("limit", "offset", "sample"))
        ):
            return plan.copy()
        return wrap_as_subquery(self._plan, self._session._next_alias())

    # -- ordering ---------------------------------------------------------------

    def orderBy(self, *cols: Any, ascending: Any = None) -> DataFrame:
        """Sort by the given columns.

        Direction comes from either the column or the `ascending` argument:

            orderBy(col("a").desc())          per column, on the column
            orderBy("a", ascending=False)      one direction for all of them
            orderBy("a", "b", ascending=[True, False])   one per column

        `ascending` **wins** where both are given, as it does in the reference.

        Null ordering is not spelled here and does not need to be: `_fix_null_ordering`
        in `sql/conformance.py` is a tree pass over every `exp.Ordered`, so a sort built
        by this method and one parsed from `Session.sql()` come out identical -- nulls
        first ascending, last descending (P1).
        """
        items = _flatten(cols)
        if not items:
            raise EngineValueError("orderBy() needs at least one column.")
        directions = _sort_directions(ascending, len(items))

        ordered: list[exp.Expression] = []
        for index, item in enumerate(items):
            if isinstance(item, str):
                expression = self._column_ref(item)._expression.copy()
            elif isinstance(item, Column):
                expression = item._expression.copy()
            else:
                raise EngineTypeError(
                    f"orderBy() takes column names or Column objects, got {type(item).__name__}."
                )
            direction = directions[index] if directions is not None else None
            ordered.append(_as_ordered(expression, direction))

        plan = self._rebase_order()
        plan.set("order", exp.Order(expressions=ordered))
        return self._derive(cast(exp.Expression, plan))

    #: The reference engine exposes both spellings.
    sort = orderBy

    def sortWithinPartitions(self, *cols: Any, ascending: Any = None) -> DataFrame:
        """`sort`. There is one partition here, so sorting within it sorts the frame."""
        return self.orderBy(*cols, ascending=ascending)

    def _rebase_order(self) -> exp.Select:
        """A SELECT an ORDER BY can be attached to, replacing any it already carries.

        Replacing rather than stacking is right -- a later `orderBy` supersedes an
        earlier one -- but only while no LIMIT or OFFSET has been taken. Past one, the
        rows are already chosen, and re-ordering has to happen outside the clause that
        chose them: `df.limit(3).orderBy("a")` sorts those three rows, and folding it in
        would sort the whole table and then take three different ones.
        """
        plan = self._plan
        if isinstance(plan, exp.Select) and not self._has_any(("limit", "offset")):
            return plan.copy()
        return wrap_as_subquery(self._plan, self._session._next_alias())

    # -- sampling -------------------------------------------------------------

    def sample(
        self,
        withReplacement: bool | float | None = None,
        fraction: float | None = None,
        seed: int | None = None,
    ) -> DataFrame:
        """A random subset of the rows, each drawn independently with probability `fraction`.

        The reference's argument juggling is reproduced: `sample(0.5)` and
        `sample(0.5, 42)` both work, the first positional argument being read as the
        fraction when it is a float rather than a bool.

        **`fraction` is a probability, not a row count.** Sampling 0.5 of ten rows
        returns about five, not exactly five -- each row is decided on its own. Pass
        `seed` to make the draw repeatable.

        Sampling **with replacement is refused** rather than approximated: DuckDB draws
        each row at most once, and quietly handing a without-replacement sample to a
        caller who asked for the other is a wrong answer.
        """
        if isinstance(withReplacement, float):
            # `sample(0.5)` / `sample(0.5, 42)` -- the fraction came first.
            withReplacement, fraction, seed = False, withReplacement, fraction
        if withReplacement is None:
            withReplacement = False
        if not isinstance(withReplacement, bool):
            raise EngineTypeError(
                f"sample() expects withReplacement as a bool, got {type(withReplacement).__name__}."
            )
        if withReplacement:
            raise UnsupportedFeatureError(
                "sample(withReplacement=True)",
                hint=(
                    "DuckDB samples without replacement only; "
                    "sample(withReplacement=False, ...) is what it can do"
                ),
            )
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool):
            raise EngineTypeError(f"sample() expects a fraction, got {type(fraction).__name__}.")
        if not 0 <= fraction <= 1:
            raise EngineValueError(f"sample() expects a fraction in [0, 1], got {fraction}.")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise EngineTypeError(f"sample() expects seed as an int, got {type(seed).__name__}.")

        plan = self._rebase_sample()
        arguments: dict[str, Any] = {
            "method": exp.Var(this="BERNOULLI"),
            "percent": exp.Literal.number(repr(float(fraction) * 100)),
        }
        if seed is not None:
            arguments["seed"] = exp.Literal.number(str(int(seed)))
        plan.set("sample", exp.TableSample(**arguments))
        return self._derive(cast(exp.Expression, plan))

    def _rebase_sample(self) -> exp.Select:
        """A SELECT a sample clause can attach to, nesting if one is already there."""
        plan = self._plan
        if isinstance(plan, exp.Select) and not plan.args.get("sample"):
            return plan.copy()
        return wrap_as_subquery(self._plan, self._session._next_alias())

    def randomSplit(self, weights: Sequence[float], seed: int | None = None) -> list[DataFrame]:
        """Split into frames sized by `weights`, which are normalised to sum to 1.

        **The frame is materialised first**, and that is the whole design. Each split is
        a filter over one random number per row, so those numbers have to be the *same*
        numbers every time a split is collected -- otherwise a row could land in two
        splits or in none, and the parts would not add back up to the whole. Drawing
        them once into a temp table is what makes the splits disjoint and complete;
        re-running `random()` per action would not.

        The cost is a query now, as `cache()` costs one, rather than staying lazy. Given
        the alternative is a split that does not partition, that is the right trade.
        """
        values = self._split_weights(weights)
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise EngineTypeError(
                    f"randomSplit() expects seed as an int, got {type(seed).__name__}."
                )
            # DuckDB seeds its generator per connection, and setseed takes [-1, 1].
            self._session._engine.execute(f"SELECT setseed({(seed % 1000) / 1000.0})")

        columns = list(self.columns)
        draw = as_expression(exp.alias_(exp.Anonymous(this="random"), _SPLIT_COLUMN, quoted=True))
        keep = [Column(exp.column(name, quoted=True)) for name in columns]
        materialised = self.select(*keep, Column(draw)).cache()

        total = builtins.sum(values)
        splits: list[DataFrame] = []
        lower = 0.0
        for index, weight in enumerate(values):
            last = index == len(values) - 1
            upper = 1.0 if last else lower + weight / total
            reference = exp.column(_SPLIT_COLUMN, quoted=True)
            bounds: exp.Expression = exp.GTE(
                this=reference, expression=exp.Literal.number(repr(lower))
            )
            if not last:
                bounds = as_expression(
                    exp.and_(
                        bounds,
                        exp.LT(this=reference.copy(), expression=exp.Literal.number(repr(upper))),
                    )
                )
            splits.append(materialised.filter(Column(bounds)).select(*columns))
            lower = upper
        return splits

    def _split_weights(self, weights: Sequence[float]) -> list[float]:
        if isinstance(weights, (str, bytes)) or not hasattr(weights, "__iter__"):
            raise EngineTypeError(
                f"randomSplit() expects a sequence of weights, got {type(weights).__name__}."
            )
        values = list(weights)
        if not values:
            raise EngineValueError("randomSplit() needs at least one weight.")
        for weight in values:
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise EngineTypeError(
                    f"randomSplit() expects numbers, got {type(weight).__name__}."
                )
            if weight < 0:
                raise EngineValueError(f"randomSplit() expects weights >= 0, got {weight}.")
        if builtins.sum(values) <= 0:
            raise EngineValueError("randomSplit() needs at least one weight above zero.")
        return [float(value) for value in values]

    # -- materialisation --------------------------------------------------------

    def cache(self) -> DataFrame:
        """Run this frame now and keep the rows, returning a frame that reads them back.

        Two ways this differs from the reference, both deliberate and both recorded in
        `divergence.md`:

        * **It is eager.** The reference marks a frame and materialises it at the next
          action. Nothing here mutates a plan in place, so a lazy mark would have
          nowhere to live; running now and handing back a new frame says the same thing
          without the mutation.
        * **It returns a new frame** rather than `self`, so `cached = df.cache()` is the
          spelling that works. `df.cache()` alone caches nothing you can reach.

        The rows go to a DuckDB temp table, so they live in the session rather than in
        this process's Python heap, and `unpersist()` releases them.
        """
        name = self._session._materialize(self._execute())
        plan = exp.select(exp.Star()).from_(exp.to_identifier(name, quoted=True))
        cached = DataFrame(self._session, plan, {})
        cached._cache_name = name
        return cached

    def persist(self, storageLevel: Any = None) -> DataFrame:
        """`cache()`. There is one storage level here -- a DuckDB temp table."""
        if storageLevel is not None:
            raise UnsupportedFeatureError(
                "persist(storageLevel=...)",
                hint="There is one storage level here: the DuckDB temp table cache() uses",
            )
        return self.cache()

    def unpersist(self, blocking: bool = False) -> DataFrame:
        """Release the rows `cache()` materialised. A no-op on an uncached frame."""
        if self._cache_name is not None:
            self._session._release(self._cache_name)
        return self

    # -- partitioning -----------------------------------------------------------

    def repartition(self, numPartitions: Any = None, *cols: Any) -> DataFrame:
        """A no-op returning this frame.

        There is one partition here and there always will be: everything runs in one
        process against one DuckDB connection, which parallelises within a query by
        itself. The method exists so a script written against the reference runs
        unaltered -- repartitioning is a distribution concern, and there is no
        distribution to concern it.
        """
        return self

    def coalesce(self, numPartitions: int = 1) -> DataFrame:
        """A no-op, for the same reason as `repartition`."""
        return self

    # -- temporary views --------------------------------------------------------

    def createOrReplaceTempView(self, name: str) -> None:
        """Register this frame's *plan* under `name`, for `session.sql()` to reference.

        A view here is a plan, not rows -- nothing runs, and a query against the view
        reads the table the frame came from with pushdown intact. Reach for `cache()`
        when you want the rows held instead.
        """
        self._session._register_temp_view(name, self._plan, replace=True)

    def createTempView(self, name: str) -> None:
        """`createOrReplaceTempView`, but refusing to overwrite an existing view."""
        self._session._register_temp_view(name, self._plan, replace=False)

    def alias(self, alias: str) -> DataFrame:
        """Name this DataFrame, so its columns can be qualified as `alias.column`."""
        if not isinstance(alias, str):
            raise EngineTypeError(f"alias() expects a string, got {type(alias).__name__}.")
        return self._derive(wrap_as_subquery(self._plan, alias))

    # -- actions -----------------------------------------------------------

    def _execute(self) -> pa.Table:
        return self._session._execute(self._plan, self._sources, self.columns)

    def toArrow(self) -> pa.Table:
        """The whole result as an Arrow table."""
        return self._execute()

    def collect(self) -> list[Row]:
        """The whole result as a list of `Row`."""
        return to_rows(self._execute(), self.schema)

    def toPandas(self) -> pd.DataFrame:
        """The whole result as a pandas DataFrame."""
        return to_pandas(self._execute())

    def take(self, num: int) -> list[Row]:
        """The first `num` rows."""
        return self.limit(num).collect()

    def head(self, n: int | None = None) -> Row | list[Row] | None:
        """The first row, or the first `n` rows when `n` is given."""
        if n is None:
            rows = self.take(1)
            return rows[0] if rows else None
        return self.take(n)

    def first(self) -> Row | None:
        """The first row, or None when there are none."""
        rows = self.take(1)
        return rows[0] if rows else None

    def count(self) -> int:
        """The number of rows."""
        counting = exp.select(exp.Count(this=exp.Star())).from_(
            exp.Subquery(
                this=self._plan.copy(),
                alias=exp.TableAlias(this=exp.to_identifier(self._session._next_alias())),
            )
        )
        # `count(*)` names no column, which is what lets projection pushdown read
        # one column instead of all 200 on the wide table.
        result = self._session._execute(counting, self._sources, ["count(1)"])
        return int(result.column(0)[0].as_py())

    def show(
        self, n: int = _DEFAULT_SHOW_ROWS, truncate: bool | int = True, vertical: bool = False
    ) -> None:
        """Print the first `n` rows, in the reference engine's layout."""
        if not isinstance(n, int) or isinstance(n, bool):
            raise EngineTypeError(f"show() expects an int row count, got {type(n).__name__}.")
        if truncate is True:
            width = _DEFAULT_TRUNCATE
        elif truncate is False:
            width = 0
        elif isinstance(truncate, int):
            width = truncate
        else:
            raise EngineTypeError("show() expects truncate to be a bool or an int.")

        # One row past the limit, so the "only showing top n rows" footer is accurate.
        rows = self.take(n + 1)
        print(
            format_show(
                rows,
                self.schema,
                n=n,
                truncate=width,
                vertical=vertical,
                has_more=len(rows) > n,
            ),
            end="",
        )

    def explain(self, extended: bool | str = False, mode: str | None = None) -> None:
        """Print the DuckDB SQL this DataFrame will run, and what it will scan.

        `mode="extended"` adds the before-and-after optimizer trees, which is how you
        find out *why* a query did not prune: a plan the optimizer could not bind, a
        filter that could not be translated, or a column list nothing narrowed.
        """
        if isinstance(extended, str):
            mode, extended = extended, False
        if mode is not None and mode not in ("simple", "extended"):
            raise UnsupportedFeatureError(
                f"explain(mode={mode!r}); icetl supports 'simple' and 'extended'",
                phase="Phase 10",
            )
        verbose = bool(extended) or mode == "extended"
        print(self._explain_text(verbose=verbose), end="")

    def _explain_text(self, *, verbose: bool) -> str:
        compiled = self._session._compile(self._plan, self._sources, self.columns)
        optimized = compiled.optimized
        lines: list[str] = []

        if verbose:
            lines += [
                "== Logical Plan (icetl) ==",
                self._plan.sql(dialect="duckdb", pretty=True),
                "",
                "== Optimized Plan ==",
            ]
            if optimized is not None and optimized.applied:
                lines += [
                    f"  rules: {', '.join(optimized.stages)}",
                    optimized.optimized.sql(dialect="duckdb", pretty=True),
                ]
                if optimized.note:
                    lines.append(f"  note: {optimized.note}")
            else:
                lines.append(f"  (not applied: {optimized.note if optimized else 'not attempted'})")
            lines += [
                "",
                "== Analysed Schema ==",
                self.schema.treeString().rstrip("\n"),
                "",
            ]

        lines += ["== Physical Plan (DuckDB SQL) ==", compiled.sql, ""]
        lines.append("== Scans ==")
        if not compiled.scans:
            lines.append("  (no Iceberg tables -- this plan reads no files)")
        for scan in compiled.scans:
            lines.append(f"  {scan.source.resolved.qualified_name}: {scan.describe()}")
            lines.append(f"    columns: {scan.describe_columns()}")
            lines.append(f"    pushed filters: {scan.pushed_filter or 'none'}")
            if scan.unpushed_filters:
                # Named rather than hidden: these still run in SQL, so the answer is
                # right -- but they bought no pruning, which is exactly what someone
                # staring at a slow query needs to be told.
                lines.append(f"    kept in SQL only: {'; '.join(scan.unpushed_filters)}")
            if scan.renamed_columns:
                lines.append(f"    field-id reconciliation: {', '.join(scan.renamed_columns)}")
        return "\n".join(lines) + "\n"

    # -- deferred surface --------------------------------------------------

    @property
    def write(self) -> DataFrameWriter:
        """The write path: `df.write.mode("append").saveAsTable("ns.table")`."""
        from icetl.sql.writer import DataFrameWriter

        return DataFrameWriter(self)

    @property
    def rdd(self) -> Any:
        raise UnsupportedFeatureError(
            "df.rdd", hint="There is no RDD layer here, and there will not be one"
        )

    def __repr__(self) -> str:
        return f"DataFrame[{', '.join(f'{n}: {t}' for n, t in self.dtypes)}]"


def _from_clause(plan: exp.Select) -> exp.From | None:
    """The SELECT's own FROM node, or None.

    Scans direct arguments for the same reason `_has_from_clause` does: sqlglot v30
    spells the key `from_`, earlier releases `from`, and `plan.find()` would reach into
    a subquery.
    """
    for value in plan.args.values():
        if isinstance(value, exp.From):
            return value
    return None


def _has_from_clause(plan: exp.Select) -> bool:
    """True when the SELECT has its own FROM.

    Found by scanning the node's direct arguments rather than by key name: sqlglot
    spells the key `from_` in v30 and `from` in earlier releases, and `plan.find()`
    would wrongly match a FROM nested inside a subquery.
    """
    return any(isinstance(value, exp.From) for value in plan.args.values())


def _flatten(items: Any) -> list[Any]:
    """Accept both `select("a", "b")` and `select(["a", "b"])`, as the reference engine does."""
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        return list(items[0])
    return list(items)


def _sort_directions(ascending: Any, count: int) -> list[bool] | None:
    """Normalise `ascending` to one bool per column, or None when it was not given."""
    if ascending is None:
        return None
    if isinstance(ascending, (list, tuple)):
        directions = list(ascending)
        if len(directions) != count:
            raise EngineValueError(
                f"orderBy() got {count} column(s) but {len(directions)} ascending "
                f"flag(s); pass one flag, or one per column."
            )
    else:
        directions = [ascending] * count
    normalised: list[bool] = []
    for direction in directions:
        if isinstance(direction, bool):
            normalised.append(direction)
        elif isinstance(direction, int):
            # The reference accepts 0/1 here, and scripts written against it use them.
            normalised.append(bool(direction))
        else:
            raise EngineTypeError(
                f"orderBy() expects ascending as a bool or list of bools, got "
                f"{type(direction).__name__}."
            )
    return normalised


def _as_ordered(expression: exp.Expression, ascending: bool | None) -> exp.Expression:
    """One ORDER BY term, with `ascending` overriding a direction already on the column.

    An explicit `ascending` wins over `col("a").desc()`, which is the reference's
    precedence. Null placement is deliberately left unset: the conformance pass fills it
    in for both surfaces, and setting it here would mean this method and the SQL parser
    each had their own answer.
    """
    if ascending is None:
        if isinstance(expression, exp.Ordered):
            return expression
        return exp.Ordered(this=expression)
    inner = expression.this if isinstance(expression, exp.Ordered) else expression
    return exp.Ordered(this=inner, desc=not ascending)


def _contains_name(names: Sequence[str], name: str) -> bool:
    """True when `name` is in `names`, compared the way the analyser compares columns."""
    folded = name.casefold()
    return any(candidate.casefold() == folded for candidate in names)


def _aligned_to(available: Sequence[str], combined: Sequence[str]) -> list[Column]:
    """Projections putting `available` into `combined`'s order, NULL where it has none.

    A name `available` does not carry becomes `NULL AS name` rather than a cast NULL --
    see `unionByName` for why the type looks after itself.
    """
    by_key = {name.casefold(): name for name in available}
    projections: list[Column] = []
    for name in combined:
        source = by_key.get(name.casefold())
        if source is None:
            projections.append(Column(as_expression(exp.alias_(exp.Null(), name, quoted=True))))
        elif source == name:
            projections.append(Column(exp.column(source, quoted=True)))
        else:
            reference = exp.column(source, quoted=True)
            projections.append(Column(as_expression(exp.alias_(reference, name, quoted=True))))
    return projections


def _join_keys(on: Column | str | list[str]) -> list[str] | None:
    """The `USING` column names in `on`, or None when `on` is a Column condition."""
    if isinstance(on, str):
        return [on]
    if isinstance(on, (list, tuple)) and on and all(isinstance(item, str) for item in on):
        return list(on)
    return None


def _to_predicate(condition: Column | str) -> exp.Expression:
    if isinstance(condition, Column):
        return condition._expression.copy()
    if isinstance(condition, str):
        try:
            return as_expression(sqlglot.parse_one(condition, read=SQL_DIALECT))
        except Exception as exc:
            raise ParseException(f"Could not parse the filter {condition!r}: {exc}") from exc
    raise EngineTypeError(
        f"filter() expects a Column or a SQL string, got {type(condition).__name__}."
    )

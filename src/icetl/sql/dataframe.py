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
from icetl.types import Row, StructType

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    from icetl.plan.builder import ScanSource
    from icetl.sql.session import Session

__all__ = ["DataFrame"]

_DEFAULT_SHOW_ROWS = 20
_DEFAULT_TRUNCATE = 20

# Clauses whose presence means a new projection or filter cannot simply be merged
# into the existing SELECT.
_POST_FILTER_CLAUSES = ("group", "order", "limit", "offset", "distinct", "having", "qualify")


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

    def _rebase_projection(self) -> exp.Select:
        """A SELECT whose projection list is safe to replace wholesale."""
        if self._is_star_projection() and not self._has_any(_POST_FILTER_CLAUSES):
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
        if level is not None:
            raise UnsupportedFeatureError("printSchema(level=...)", phase="Phase 6")
        print(self.schema.treeString(), end="")

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
        """Project a set of columns or expressions."""
        items = _flatten(cols)
        if not items:
            raise EngineValueError("select() needs at least one column.")
        projection = [self._projection(item) for item in items]
        plan = self._rebase_projection()
        plan.set("expressions", projection)
        return self._derive(plan)

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
    def write(self) -> Any:
        raise UnsupportedFeatureError("df.write", phase="Phase 7")

    @property
    def rdd(self) -> Any:
        raise UnsupportedFeatureError(
            "df.rdd", hint="There is no RDD layer here, and there will not be one"
        )

    def __repr__(self) -> str:
        return f"DataFrame[{', '.join(f'{n}: {t}' for n, t in self.dtypes)}]"


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

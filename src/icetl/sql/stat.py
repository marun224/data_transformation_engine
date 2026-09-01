"""`df.stat` -- the statistics surface: quantiles, correlation, covariance, contingency.

Two of these return plain Python numbers (`approxQuantile`, `corr`, `cov`) and two
return DataFrames (`crosstab`, `freqItems`). The split is the reference's, and it is
also the honest one: a correlation is one number, and pretending it is a lazy frame
would buy nothing.

**`crosstab` and `freqItems` compute eagerly**, as they do in the reference. Both have
to know the data before they can know their own *columns* -- a contingency table's
columns are the distinct values of the second column -- so there is no lazy form
available. `freqItems` goes further and folds its answer into a literal plan with no
source at all, which is what makes the returned frame independent of the table it came
from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlglot import exp

from icetl.errors import AnalysisException, EngineTypeError, EngineValueError
from icetl.plan.builder import as_expression
from icetl.sql.column import Column, to_literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from icetl.sql.dataframe import DataFrame

__all__ = ["DataFrameStatFunctions"]

#: `simpleString()` names the numeric checks accept. Kept beside `na.py`'s copy rather
#: than shared: that one decides what a *fill value* may touch, this one decides what a
#: *statistic* may be computed over, and the two lists agreeing today is a coincidence.
_NUMERIC_TYPES = frozenset({"tinyint", "smallint", "int", "bigint", "float", "double"})


class DataFrameStatFunctions:
    """What `df.stat` returns. Holds the frame; each call is independent."""

    def __init__(self, df: DataFrame) -> None:
        self._df = df

    def __repr__(self) -> str:
        return f"DataFrameStatFunctions[{', '.join(self._df.columns)}]"

    # -- quantiles ----------------------------------------------------------

    def approxQuantile(
        self,
        col: str | Sequence[str],
        probabilities: Sequence[float],
        relativeError: float = 0.0,
    ) -> list[float] | list[list[float]]:
        """Quantiles of one column, or of several.

        Returns a flat list for a single column name and a list per column for a
        sequence of them -- the reference's return shape, which depends on the argument
        rather than on a flag.

        **`relativeError=0` is exact**, computed with `quantile_disc`, so the answer is
        a value that appears in the data rather than an interpolation between two that
        do. That matches the reference, whose quantiles are always observed values.
        Above zero, DuckDB's `approx_quantile` is used instead; its error bound is fixed
        rather than tunable, so the answer may be more accurate than the caller asked
        for and never less -- which is what `relativeError` actually promises.

        Nulls are ignored, as they are in the reference.
        """
        single = isinstance(col, str)
        names: list[str] = [col] if isinstance(col, str) else list(col)
        for name in names:
            if not isinstance(name, str):
                raise EngineTypeError(
                    f"approxQuantile() expects column names, got {type(name).__name__}."
                )
        probs = self._probabilities(probabilities)
        if not isinstance(relativeError, (int, float)) or isinstance(relativeError, bool):
            raise EngineTypeError(
                f"approxQuantile() expects relativeError as a number, got "
                f"{type(relativeError).__name__}."
            )
        if relativeError < 0:
            raise EngineValueError(
                f"approxQuantile() expects relativeError >= 0, got {relativeError}."
            )

        function = "quantile_disc" if relativeError == 0 else "approx_quantile"
        projections: list[Any] = []
        for index, name in enumerate(names):
            reference = self._numeric_column(name, "approxQuantile")
            for position, probability in enumerate(probs):
                call = exp.Anonymous(
                    this=function,
                    expressions=[reference.copy(), exp.Literal.number(repr(probability))],
                )
                projections.append(
                    Column(as_expression(exp.alias_(call, f"q_{index}_{position}", quoted=True)))
                )

        row = self._df.select(*projections).collect()[0]
        values = [None if v is None else float(v) for v in row]
        per_column = [values[i * len(probs) : (i + 1) * len(probs)] for i in range(len(names))]
        return per_column[0] if single else per_column  # type: ignore[return-value]

    # -- pairwise statistics ------------------------------------------------

    def corr(self, col1: str, col2: str, method: str = "pearson") -> float:
        """Pearson correlation of two numeric columns.

        `method` exists for signature compatibility and accepts only `"pearson"`, which
        is the only method the reference implements either.
        """
        if method != "pearson":
            raise EngineValueError(f"corr() supports method='pearson' only, got {method!r}.")
        return self._pairwise("corr", col1, col2, "corr")

    def cov(self, col1: str, col2: str) -> float:
        """**Sample** covariance of two numeric columns, as the reference's `cov`.

        `covar_samp`, not `covar_pop`: the reference divides by `n - 1`, and the two
        differ by enough on a small frame to matter.
        """
        return self._pairwise("covar_samp", col1, col2, "cov")

    def _pairwise(self, function: str, col1: str, col2: str, method: str) -> float:
        first = self._numeric_column(col1, method)
        second = self._numeric_column(col2, method)
        call = exp.Anonymous(this=function, expressions=[first, second])
        aggregated = Column(as_expression(exp.alias_(call, "value", quoted=True)))
        row = self._df.agg(aggregated).collect()[0]
        return float("nan") if row[0] is None else float(row[0])

    # -- contingency and frequency ------------------------------------------

    def crosstab(self, col1: str, col2: str) -> DataFrame:
        """A contingency table: `col1`'s values down the side, `col2`'s across the top.

        The first column is named `col1_col2` and holds `col1`'s values **as strings**,
        which is the reference's shape -- the row labels and the column labels have to
        be the same kind of thing, and the column labels are necessarily names. A pair
        that never occurs counts **0**, not NULL: an absent combination is an observed
        zero, not a missing measurement.
        """
        from icetl.sql import functions as F

        for name, label in ((col1, "col1"), (col2, "col2")):
            if not isinstance(name, str):
                raise EngineTypeError(
                    f"crosstab() expects {label} as a column name, got {type(name).__name__}."
                )
        first = self._resolve(col1, "crosstab")
        second = self._resolve(col2, "crosstab")

        counted = self._df.groupBy(first).pivot(second).agg(F.count("*"))
        label = f"{first}_{second}"
        # NULL becomes the string "null" on the row side too, so it reads the same as
        # the column the pivot named for it. A contingency table whose row label was
        # blank and whose matching column header said "null" would be a puzzle.
        row_label = exp.Coalesce(
            this=exp.Cast(this=exp.column(first, quoted=True), to=exp.DataType.build("VARCHAR")),
            expressions=[exp.Literal.string("null")],
        )
        projections: list[Any] = [Column(as_expression(exp.alias_(row_label, label, quoted=True)))]
        for name in counted.columns[1:]:
            filled = exp.Coalesce(
                this=exp.column(name, quoted=True), expressions=[exp.Literal.number(0)]
            )
            projections.append(Column(as_expression(exp.alias_(filled, name, quoted=True))))
        return counted.select(*projections)

    def freqItems(self, cols: str | Sequence[str], support: float = 0.01) -> DataFrame:
        """Values occurring in at least `support` of the rows, one array column each.

        Output columns are named `<col>_freqItems`, as the reference names them, and
        there is exactly one row.

        Computed **exactly** rather than approximately. The reference uses a sketch
        because it is counting across a cluster; counting here is one grouped scan per
        column, and an exact answer inside the same contract is strictly better than an
        approximate one. The result is folded into a literal plan with no source, so the
        returned frame is independent of the table it came from -- collect it twice and
        the table cannot have changed underneath it.
        """
        from icetl.sql import functions as F

        names = [cols] if isinstance(cols, str) else list(cols)
        if not names:
            raise EngineValueError("freqItems() needs at least one column.")
        if not isinstance(support, (int, float)) or isinstance(support, bool):
            raise EngineTypeError(
                f"freqItems() expects support as a number, got {type(support).__name__}."
            )
        if not 0 < support <= 1:
            raise EngineValueError(f"freqItems() expects 0 < support <= 1, got {support}.")

        projections: list[Any] = []
        for name in names:
            resolved = self._resolve(name, "freqItems")
            counts = self._df.groupBy(resolved).agg(F.count("*").alias("n")).collect()
            total = sum(int(row[1]) for row in counts)
            threshold = support * total
            items = [row[0] for row in counts if int(row[1]) >= threshold]
            array = exp.Array(expressions=[to_literal(item) for item in items])
            projections.append(
                as_expression(exp.alias_(array, f"{resolved}_freqItems", quoted=True))
            )

        from icetl.sql.dataframe import DataFrame as _DataFrame

        return _DataFrame(self._df._session, exp.Select(expressions=projections), {})

    # -- shared -------------------------------------------------------------

    def _probabilities(self, probabilities: Sequence[float]) -> list[float]:
        if isinstance(probabilities, (str, bytes)) or not hasattr(probabilities, "__iter__"):
            raise EngineTypeError(
                f"approxQuantile() expects a sequence of probabilities, got "
                f"{type(probabilities).__name__}."
            )
        values = list(probabilities)
        if not values:
            raise EngineValueError("approxQuantile() needs at least one probability.")
        for probability in values:
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise EngineTypeError(
                    f"approxQuantile() expects numbers, got {type(probability).__name__}."
                )
            if not 0 <= probability <= 1:
                raise EngineValueError(
                    f"approxQuantile() expects probabilities in [0, 1], got {probability}."
                )
        return [float(value) for value in values]

    def _resolve(self, name: str, method: str) -> str:
        resolved = self._df._resolve_name(name)
        if resolved is None:
            raise AnalysisException(
                f"{method}() was given the column {name!r}, which this frame does not "
                f"have. Columns: {', '.join(self._df.columns)}."
            )
        return resolved

    def _numeric_column(self, name: str, method: str) -> exp.Expression:
        """The column node, refused unless it is numeric.

        Refused rather than allowed through: DuckDB would happily correlate two dates
        after an implicit cast, and a number that means nothing is worse than an error.
        """
        resolved = self._resolve(name, method)
        for field in self._df.schema.fields:
            if field.name == resolved:
                type_name = field.dataType.simpleString()
                if type_name not in _NUMERIC_TYPES and not type_name.startswith("decimal"):
                    raise AnalysisException(
                        f"{method}() needs a numeric column; {resolved!r} is {type_name}."
                    )
                break
        return exp.column(resolved, quoted=True)

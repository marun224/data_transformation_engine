"""`pyspark.sql.functions` -- the `F.*` namespace.

**Built on sqlglot's typed nodes, not on raw SQL strings.** `sg.Concat` knows that
Spark's `concat` propagates NULL and DuckDB's does not, and generates `a || b`
accordingly; `sg.Split` knows Spark's separator is a regex and reaches for
`str_split_regex`. Every function written as a typed node gets that translation for
free and keeps it when DuckDB changes. `_fn` -- a bare `sg.Anonymous` -- is the
fallback for the cases sqlglot has no node for, and each use is a small bet that the
DuckDB spelling is also the Spark one, so each is spot-checked in the tests.

**Where Spark and DuckDB genuinely disagree**, the difference is fixed here and
recorded in `compat/divergence.md`. `dayofweek` is the one in this module: Spark
numbers Sunday 1, DuckDB numbers it 0, and nothing about that surfaces as an error.

**A string argument names a column**, which is the opposite of what a string means to
a `Column` operator. `F.upper("name")` upper-cases the column `name`; `col("a") ==
"name"` compares against the *string* "name". That asymmetry is PySpark's, and
`_col` versus `to_expression` is where it lives.

Parameter names match PySpark's exactly, because Spark scripts pass them by keyword.
That means shadowing the `str` builtin inside `expr`, which is why `_str` exists.
"""

from __future__ import annotations

from typing import Any

import sqlglot

# sqlglot's expression module is aliased `sg` here, not the usual `exp`, because
# Spark has a function *called* `exp` and defining it would shadow the module for
# every function below it -- silently, until one of them was called. The same hazard
# is why `_str` exists. Everywhere else in the codebase the alias is `exp`.
from sqlglot import exp as sg

from icetl.errors import ParseException, PySparkTypeError, PySparkValueError
from icetl.plan.builder import as_expression
from icetl.sql.column import Column, _column_from_name, to_expression, to_literal

__all__ = [
    "abs",
    "acos",
    "add_months",
    "any_value",
    "approx_count_distinct",
    "array",
    "array_contains",
    "array_distinct",
    "array_intersect",
    "array_join",
    "array_max",
    "array_min",
    "array_position",
    "array_remove",
    "array_sort",
    "array_union",
    "arrays_overlap",
    "asc",
    "asc_nulls_first",
    "asc_nulls_last",
    "ascii",
    "asin",
    "atan",
    "atan2",
    "avg",
    "bin",
    "bool_and",
    "bool_or",
    "btrim",
    "cbrt",
    "ceil",
    "ceiling",
    "coalesce",
    "col",
    "collect_list",
    "collect_set",
    "column",
    "concat",
    "concat_ws",
    "corr",
    "cos",
    "cosh",
    "count",
    "countDistinct",
    "count_if",
    "covar_pop",
    "covar_samp",
    "current_date",
    "current_timestamp",
    "date_add",
    "date_diff",
    "date_format",
    "date_sub",
    "date_trunc",
    "datediff",
    "dayofmonth",
    "dayofweek",
    "dayofyear",
    "degrees",
    "desc",
    "desc_nulls_first",
    "desc_nulls_last",
    "element_at",
    "exp",
    "expr",
    "factorial",
    "first",
    "flatten",
    "floor",
    "from_unixtime",
    "greatest",
    "hash",
    "hex",
    "hour",
    "hypot",
    "ifnull",
    "initcap",
    "instr",
    "isnan",
    "isnull",
    "kurtosis",
    "last",
    "last_day",
    "least",
    "left",
    "length",
    "levenshtein",
    "lit",
    "locate",
    "log",
    "log2",
    "log10",
    "lower",
    "lpad",
    "ltrim",
    "make_date",
    "map_keys",
    "map_values",
    "max",
    "max_by",
    "md5",
    "mean",
    "median",
    "min",
    "min_by",
    "minute",
    "mode",
    "month",
    "months_between",
    "nanvl",
    "negative",
    "nullif",
    "nvl",
    "nvl2",
    "percentile_approx",
    "pmod",
    "positive",
    "pow",
    "quarter",
    "radians",
    "regexp_extract",
    "regexp_replace",
    "repeat",
    "replace",
    "reverse",
    "right",
    "round",
    "rpad",
    "rtrim",
    "second",
    "sequence",
    "sha1",
    "sha2",
    "shiftleft",
    "shiftright",
    "signum",
    "sin",
    "sinh",
    "size",
    "skewness",
    "slice",
    "sort_array",
    "split",
    "sqrt",
    "stddev",
    "stddev_pop",
    "stddev_samp",
    "struct",
    "substring",
    "substring_index",
    "sum",
    "sum_distinct",
    "tan",
    "tanh",
    "timestamp_seconds",
    "to_date",
    "to_timestamp",
    "translate",
    "trim",
    "trunc",
    "unhex",
    "unix_timestamp",
    "upper",
    "var_pop",
    "var_samp",
    "variance",
    "weekofyear",
    "when",
    "year",
]

_str = str  # captured before `expr`'s parameter name shadows the builtin


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


def _col(value: Any) -> sg.Expression:
    """Coerce a function argument: a string names a *column*, as in PySpark."""
    if isinstance(value, Column):
        return value._copy()
    if isinstance(value, _str):
        return _column_from_name(value)._copy()
    raise PySparkTypeError(
        f"Expected a column name or Column, got {type(value).__name__}. "
        f"Wrap a literal value in F.lit()."
    )


def _value(value: Any) -> sg.Expression:
    """Coerce an argument that is a *value*, not a column reference."""
    return to_expression(value)


def _fn(name: _str, *args: sg.Expression) -> Column:
    """A function sqlglot has no typed node for.

    Every use is a bet that DuckDB spells it the way Spark does, so every use is
    spot-checked in `tests/fixture/test_functions.py` rather than assumed.
    """
    return Column(sg.Anonymous(this=name, expressions=list(args)))


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def col(col: _str) -> Column:
    """Reference a column by name.

    >>> col("amount")
    Column<'amount'>
    >>> col("t.amount")
    Column<'t.amount'>
    """
    if not isinstance(col, _str):
        raise PySparkTypeError(
            f"col() expects a column name as a string, got {type(col).__name__}."
        )
    return _column_from_name(col)


#: Spark's alias for `col`.
column = col


def lit(col: Any) -> Column:
    """A literal value as a Column. A Column passed in is returned unchanged."""
    if isinstance(col, Column):
        return col
    return Column(to_literal(col))


def expr(str: _str) -> Column:
    """Parse a Spark SQL expression into a Column.

    >>> expr("amount * 2")
    Column<'amount * 2'>
    """
    if not isinstance(str, _str):
        raise PySparkTypeError(f"expr() expects a string, got {type(str).__name__}.")
    try:
        parsed = sqlglot.parse_one(str, read="spark")
    except Exception as exc:
        raise ParseException(f"Could not parse the expression {str!r}: {exc}") from exc
    return Column(as_expression(parsed))


# ---------------------------------------------------------------------------
# Conditionals and null handling
# ---------------------------------------------------------------------------


def when(condition: Column, value: Any) -> Column:
    """Start a `CASE`. Chain `.when(...)` for more branches and `.otherwise(...)`.

    With no `otherwise`, an unmatched row is NULL -- which is Spark's behaviour and
    also SQL's, so no rule is needed to make it so.
    """
    if not isinstance(condition, Column):
        raise PySparkTypeError(
            f"when() expects a Column condition, got {type(condition).__name__}."
        )
    return Column(sg.Case(ifs=[sg.If(this=condition._copy(), true=_value(value))]))


def coalesce(*cols: Any) -> Column:
    """The first non-NULL of its arguments."""
    if not cols:
        raise PySparkValueError("coalesce() needs at least one column.")
    return Column(sg.Coalesce(this=_col(cols[0]), expressions=[_col(c) for c in cols[1:]]))


def nvl(col1: Any, col2: Any) -> Column:
    """`coalesce` of exactly two arguments."""
    return coalesce(col1, col2)


def nullif(col1: Any, col2: Any) -> Column:
    """NULL when the two are equal, else the first."""
    return Column(sg.Nullif(this=_col(col1), expression=_col(col2)))


def nanvl(col1: Any, col2: Any) -> Column:
    """`col1` unless it is NaN, in which case `col2`."""
    first = _col(col1)
    return Column(
        sg.Case(
            ifs=[sg.If(this=sg.Anonymous(this="isnan", expressions=[first]), true=_col(col2))],
            default=first,
        )
    )


def isnull(col: Any) -> Column:
    return Column(sg.Is(this=_col(col), expression=sg.null()))


def isnan(col: Any) -> Column:
    return _fn("isnan", _col(col))


def greatest(*cols: Any) -> Column:
    """The largest of its arguments, **ignoring NULLs** -- as Spark does.

    Deliberately `_fn` rather than `sg.Greatest`: sqlglot wraps the typed node in a
    `CASE WHEN any IS NULL THEN NULL` for DuckDB, which is right for dialects whose
    `GREATEST` propagates NULL and wrong for Spark. DuckDB's own `greatest` already
    skips NULLs, so the plain call is the conformant one.
    """
    if len(cols) < 2:
        raise PySparkValueError("greatest() needs at least two columns.")
    return _fn("greatest", *[_col(c) for c in cols])


def least(*cols: Any) -> Column:
    """The smallest of its arguments, **ignoring NULLs** -- as Spark does.

    `_fn` for the same reason as `greatest`.
    """
    if len(cols) < 2:
        raise PySparkValueError("least() needs at least two columns.")
    return _fn("least", *[_col(c) for c in cols])


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def asc(col: Any) -> Column:
    return _col_obj(col).asc()


def desc(col: Any) -> Column:
    return _col_obj(col).desc()


def asc_nulls_first(col: Any) -> Column:
    return _col_obj(col).asc_nulls_first()


def asc_nulls_last(col: Any) -> Column:
    return _col_obj(col).asc_nulls_last()


def desc_nulls_first(col: Any) -> Column:
    return _col_obj(col).desc_nulls_first()


def desc_nulls_last(col: Any) -> Column:
    return _col_obj(col).desc_nulls_last()


def _col_obj(value: Any) -> Column:
    return value if isinstance(value, Column) else Column(_col(value))


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def count(col: Any) -> Column:
    """`count(*)` when given the string `"*"`, else a non-NULL count of the column."""
    if col == "*" or (isinstance(col, Column) and isinstance(col._expression, sg.Star)):
        return Column(sg.Count(this=sg.Star()))
    return Column(sg.Count(this=_col(col)))


def countDistinct(*cols: Any) -> Column:
    if not cols:
        raise PySparkValueError("countDistinct() needs at least one column.")
    return Column(sg.Count(this=sg.Distinct(expressions=[_col(c) for c in cols])))


def approx_count_distinct(col: Any, rsd: float | None = None) -> Column:
    if rsd is not None:
        raise PySparkValueError(
            "approx_count_distinct(rsd=...) is not adjustable: DuckDB's HyperLogLog "
            "has a fixed error bound."
        )
    return _fn("approx_count_distinct", _col(col))


def sum(col: Any) -> Column:
    return Column(sg.Sum(this=_col(col)))


def sum_distinct(col: Any) -> Column:
    return Column(sg.Sum(this=sg.Distinct(expressions=[_col(col)])))


def avg(col: Any) -> Column:
    return Column(sg.Avg(this=_col(col)))


#: Spark exposes both spellings.
mean = avg


def min(col: Any) -> Column:
    return Column(sg.Min(this=_col(col)))


def max(col: Any) -> Column:
    return Column(sg.Max(this=_col(col)))


def first(col: Any, ignorenulls: bool = False) -> Column:
    """The first value in the group. DuckDB's `first` keeps NULLs, as Spark's does."""
    if ignorenulls:
        raise PySparkValueError(
            "first(ignorenulls=True) is not supported yet: DuckDB has no null-skipping "
            "first aggregate, and silently keeping them would be a wrong answer."
        )
    return _fn("first", _col(col))


def last(col: Any, ignorenulls: bool = False) -> Column:
    """The last value in the group. DuckDB's `last` keeps NULLs, as Spark's does."""
    if ignorenulls:
        raise PySparkValueError(
            "last(ignorenulls=True) is not supported yet: DuckDB has no null-skipping "
            "last aggregate, and silently keeping them would be a wrong answer."
        )
    return _fn("last", _col(col))


def stddev(col: Any) -> Column:
    """Sample standard deviation, as Spark's `stddev`."""
    return Column(sg.Stddev(this=_col(col)))


def variance(col: Any) -> Column:
    """Sample variance, as Spark's `variance`."""
    return Column(sg.Variance(this=_col(col)))


def collect_list(col: Any) -> Column:
    return Column(sg.ArrayAgg(this=_col(col)))


def collect_set(col: Any) -> Column:
    return Column(sg.ArrayUniqueAgg(this=_col(col)))


# ---------------------------------------------------------------------------
# Strings
# ---------------------------------------------------------------------------


def upper(col: Any) -> Column:
    return Column(sg.Upper(this=_col(col)))


def lower(col: Any) -> Column:
    return Column(sg.Lower(this=_col(col)))


def length(col: Any) -> Column:
    return Column(sg.Length(this=_col(col)))


def trim(col: Any) -> Column:
    return Column(sg.Trim(this=_col(col)))


def ltrim(col: Any) -> Column:
    return _fn("ltrim", _col(col))


def rtrim(col: Any) -> Column:
    return _fn("rtrim", _col(col))


def initcap(col: Any) -> Column:
    return Column(sg.Initcap(this=_col(col)))


def reverse(col: Any) -> Column:
    return _fn("reverse", _col(col))


def ascii(col: Any) -> Column:
    return _fn("ascii", _col(col))


def concat(*cols: Any) -> Column:
    """Concatenate. NULL in, NULL out -- which is Spark, and *not* DuckDB's `concat`.

    `sg.Concat` generates `a || b` for DuckDB, and `||` propagates NULL where
    `concat()` would silently skip it.
    """
    if not cols:
        raise PySparkValueError("concat() needs at least one column.")
    return Column(sg.Concat(expressions=[_col(c) for c in cols]))


def concat_ws(sep: _str, *cols: Any) -> Column:
    """Join with a separator, **skipping** NULLs -- which is Spark, and DuckDB agrees.

    Note the contrast with `concat`, which propagates NULL. Spark really does treat
    the two differently, and so this one is `_fn`: `sg.ConcatWs` would be wrapped in
    a NULL-propagating `CASE`, turning "skip" into "poison".
    """
    return _fn("concat_ws", to_literal(sep), *[_col(c) for c in cols])


def substring(str: Any, pos: int, len: int) -> Column:
    """1-indexed, as Spark and SQL both are."""
    return Column(sg.Substring(this=_col(str), start=_value(pos), length=_value(len)))


def split(str: Any, pattern: _str, limit: int = -1) -> Column:
    """Split on a **regex**, as Spark does -- not on a literal.

    `sg.RegexpSplit`, not `sg.Split`: the latter generates DuckDB's `str_split`, which
    treats the pattern literally and returns the whole string as a single element. No
    error, just one row where there should have been several -- which is why this is
    the node the Spark parser produces too, and why it has its own test.
    """
    if limit != -1:
        raise PySparkValueError("split(limit=...) is not supported yet.")
    return Column(sg.RegexpSplit(this=_col(str), expression=to_literal(pattern)))


def replace(src: Any, search: Any, replace: Any) -> Column:
    return _fn("replace", _col(src), _value(search), _value(replace))


def regexp_replace(str: Any, pattern: _str, replacement: _str) -> Column:
    """Replace **every** match, as Spark does.

    DuckDB's `regexp_replace` replaces only the first without the `g` flag, so the
    flag is not optional -- leaving it out changes the answer silently.
    """
    return _fn(
        "regexp_replace",
        _col(str),
        to_literal(pattern),
        to_literal(replacement),
        to_literal("g"),
    )


def regexp_extract(str: Any, pattern: _str, idx: int = 1) -> Column:
    """The `idx`-th capture group, or empty string when there is no match (Spark)."""
    return Column(
        sg.Coalesce(
            this=sg.Anonymous(
                this="regexp_extract",
                expressions=[_col(str), to_literal(pattern), to_literal(idx)],
            ),
            expressions=[to_literal("")],
        )
    )


def lpad(col: Any, len: int, pad: _str = " ") -> Column:
    return _fn("lpad", _col(col), _value(len), to_literal(pad))


def rpad(col: Any, len: int, pad: _str = " ") -> Column:
    return _fn("rpad", _col(col), _value(len), to_literal(pad))


def repeat(col: Any, n: int) -> Column:
    return _fn("repeat", _col(col), _value(n))


def locate(substr: _str, str: Any, pos: int = 1) -> Column:
    """1-indexed position of `substr`, or 0 when absent -- Spark's convention."""
    if pos != 1:
        raise PySparkValueError("locate(pos=...) beyond 1 is not supported yet.")
    return _fn("strpos", _col(str), to_literal(substr))


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


def abs(col: Any) -> Column:
    return Column(sg.Abs(this=_col(col)))


def ceil(col: Any) -> Column:
    return Column(sg.Ceil(this=_col(col)))


ceiling = ceil


def floor(col: Any) -> Column:
    return Column(sg.Floor(this=_col(col)))


def round(col: Any, scale: int = 0) -> Column:
    """Half away from zero, which is Spark's HALF_UP -- and DuckDB's default."""
    return Column(sg.Round(this=_col(col), decimals=_value(scale)))


def sqrt(col: Any) -> Column:
    return Column(sg.Sqrt(this=_col(col)))


def exp(col: Any) -> Column:
    return _fn("exp", _col(col))


def pow(col1: Any, col2: Any) -> Column:
    return Column(sg.Pow(this=_col(col1), expression=_value(col2)))


def log(arg1: Any, arg2: Any = None) -> Column:
    """`log(x)` is the natural log; `log(base, x)` is log to a base -- Spark's signature.

    The two-argument form is `_fn`, not `sg.Log`: DuckDB spells it `log(base, x)`,
    the same order Spark does, but the typed node renders its operands reversed, so
    `log(2, 8)` came out as `LOG(8, 2)` and quietly returned 1/3 instead of 3.
    """
    if arg2 is None:
        return Column(sg.Ln(this=_col(arg1)))
    return _fn("log", _value(arg1), _col(arg2))


def log2(col: Any) -> Column:
    return _fn("log2", _col(col))


def log10(col: Any) -> Column:
    return _fn("log10", _col(col))


def signum(col: Any) -> Column:
    return _fn("sign", _col(col))


# ---------------------------------------------------------------------------
# Date and time
# ---------------------------------------------------------------------------


def current_date() -> Column:
    return Column(sg.CurrentDate())


def current_timestamp() -> Column:
    return Column(sg.CurrentTimestamp())


def year(col: Any) -> Column:
    return Column(sg.Year(this=_col(col)))


def quarter(col: Any) -> Column:
    return Column(sg.Quarter(this=_col(col)))


def month(col: Any) -> Column:
    return Column(sg.Month(this=_col(col)))


def dayofmonth(col: Any) -> Column:
    return Column(sg.Day(this=_col(col)))


def dayofweek(col: Any) -> Column:
    """Spark numbers Sunday 1 through Saturday 7; DuckDB numbers Sunday 0.

    The `+ 1` is the whole difference, and without it every day is off by one with
    nothing raising -- which is why this is one of the few functions here that is not
    a straight pass-through. Recorded in `compat/divergence.md`.
    """
    return Column(
        sg.Paren(
            this=sg.Add(
                this=sg.Anonymous(this="dayofweek", expressions=[_col(col)]),
                expression=to_literal(1),
            )
        )
    )


def dayofyear(col: Any) -> Column:
    return _fn("dayofyear", _col(col))


def hour(col: Any) -> Column:
    return _fn("hour", _col(col))


def minute(col: Any) -> Column:
    return _fn("minute", _col(col))


def second(col: Any) -> Column:
    return _fn("second", _col(col))


def to_date(col: Any, format: _str | None = None) -> Column:
    if format is not None:
        raise PySparkValueError("to_date(format=...) is not supported yet.")
    return Column(sg.cast(_col(col), "DATE"))


def to_timestamp(col: Any, format: _str | None = None) -> Column:
    if format is not None:
        raise PySparkValueError("to_timestamp(format=...) is not supported yet.")
    return Column(sg.cast(_col(col), "TIMESTAMP"))


def _days(count: sg.Expression) -> sg.Expression:
    """`INTERVAL n DAY`, the argument DuckDB's `date_add` expects.

    Takes an already-coerced expression rather than a raw value, so `date_sub` can
    hand it a negation.
    """
    return sg.Anonymous(this="to_days", expressions=[count])


def _as_date(column: Column) -> Column:
    """Cast back to DATE.

    DuckDB's `date_add` widens a DATE to a TIMESTAMP; Spark's `date_add` returns a
    DATE. Without this the column type changes under the user and comparisons against
    a date stop matching.
    """
    return Column(sg.cast(column._copy(), "DATE"))


def date_add(start: Any, days: Any) -> Column:
    """`start` plus `days` whole days, as a DATE."""
    return _as_date(_fn("date_add", _col(start), _days(_value(days))))


def date_sub(start: Any, days: Any) -> Column:
    """`start` minus `days` whole days, as a DATE."""
    return _as_date(_fn("date_add", _col(start), _days(sg.Neg(this=_value(days)))))


def datediff(end: Any, start: Any) -> Column:
    """Whole days between two dates, `end - start` -- Spark's argument order."""
    return _fn("date_diff", to_literal("day"), _col(start), _col(end))


date_diff = datediff


def date_trunc(format: _str, timestamp: Any) -> Column:
    return _fn("date_trunc", to_literal(format), _col(timestamp))


def date_format(date: Any, format: _str) -> Column:
    """Format a date. **Spark's pattern syntax is Java's, DuckDB's is strftime.**

    They are not the same language -- `yyyy-MM-dd` versus `%Y-%m-%d` -- so this is
    deliberately not translated. See `compat/divergence.md`.
    """
    return _fn("strftime", _col(date), to_literal(format))


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def md5(col: Any) -> Column:
    return Column(sg.MD5(this=_col(col)))


def sha1(col: Any) -> Column:
    return _fn("sha1", _col(col))


def sha2(col: Any, numBits: int) -> Column:
    """Spark's `sha2(col, numBits)`. Only the widths DuckDB implements are accepted."""
    if numBits not in (256, 512):
        raise PySparkValueError(
            f"sha2(numBits={numBits}) is not available: DuckDB implements 256 and 512."
        )
    return _fn(f"sha{numBits}", _col(col))


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------


def array(*cols: Any) -> Column:
    return Column(sg.Array(expressions=[_col(c) for c in cols]))


def size(col: Any) -> Column:
    return Column(sg.ArraySize(this=_col(col)))


def struct(*cols: Any) -> Column:
    """A struct whose field names are the column names, as Spark does."""
    fields: list[sg.Expression] = []
    for item in cols:
        node = _col(item)
        name = item if isinstance(item, _str) else Column(node)._output_name
        fields.append(sg.PropertyEQ(this=sg.to_identifier(name, quoted=True), expression=node))
    return Column(sg.Struct(expressions=fields))


# ---------------------------------------------------------------------------
# Math -- trigonometry and bit twiddling
# ---------------------------------------------------------------------------


def sin(col: Any) -> Column:
    return _fn("sin", _col(col))


def cos(col: Any) -> Column:
    return _fn("cos", _col(col))


def tan(col: Any) -> Column:
    return _fn("tan", _col(col))


def asin(col: Any) -> Column:
    return _fn("asin", _col(col))


def acos(col: Any) -> Column:
    return _fn("acos", _col(col))


def atan(col: Any) -> Column:
    return _fn("atan", _col(col))


def atan2(col1: Any, col2: Any) -> Column:
    return _fn("atan2", _col(col1), _col(col2))


def sinh(col: Any) -> Column:
    return _fn("sinh", _col(col))


def cosh(col: Any) -> Column:
    return _fn("cosh", _col(col))


def tanh(col: Any) -> Column:
    return _fn("tanh", _col(col))


def degrees(col: Any) -> Column:
    return _fn("degrees", _col(col))


def radians(col: Any) -> Column:
    return _fn("radians", _col(col))


def cbrt(col: Any) -> Column:
    return _fn("cbrt", _col(col))


def factorial(col: Any) -> Column:
    return _fn("factorial", _col(col))


def hypot(col1: Any, col2: Any) -> Column:
    """`sqrt(a^2 + b^2)`. DuckDB has no `hypot`, so it is spelled out."""
    return sqrt(_binary_sum_of_squares(col1, col2))


def _binary_sum_of_squares(col1: Any, col2: Any) -> Column:
    left, right = _col(col1), _col(col2)
    return Column(
        sg.Add(
            this=sg.Pow(this=left, expression=to_literal(2)),
            expression=sg.Pow(this=right, expression=to_literal(2)),
        )
    )


def pmod(dividend: Any, divisor: Any) -> Column:
    """Spark's `pmod`: the remainder is **never negative**.

    SQL's `%` keeps the sign of the dividend -- DuckDB gives `-7 % 3 = -1` where Spark
    gives `2` -- so the result is shifted back into range. There is no DuckDB function
    for this; the formula is the implementation.
    """
    left, right = _col(dividend), _col(divisor)
    inner = sg.Paren(this=sg.Add(this=sg.Mod(this=left, expression=right), expression=right.copy()))
    return Column(sg.Mod(this=inner, expression=right.copy()))


def shiftleft(col: Any, numBits: int) -> Column:
    return Column(sg.BitwiseLeftShift(this=_col(col), expression=_value(numBits)))


def shiftright(col: Any, numBits: int) -> Column:
    return Column(sg.BitwiseRightShift(this=_col(col), expression=_value(numBits)))


def hex(col: Any) -> Column:
    return Column(sg.Hex(this=_col(col)))


def unhex(col: Any) -> Column:
    return Column(sg.Unhex(this=_col(col)))


def bin(col: Any) -> Column:
    """Binary representation as a string, as Spark's `bin`."""
    return _fn("bin", _col(col))


def negative(col: Any) -> Column:
    return Column(sg.Neg(this=_col(col)))


def positive(col: Any) -> Column:
    return Column(_col(col))


# ---------------------------------------------------------------------------
# Strings -- second tranche
# ---------------------------------------------------------------------------


def instr(str: Any, substr: _str) -> Column:
    """1-indexed position of `substr`, 0 when absent -- Spark's convention.

    Note the argument order is the reverse of `locate`, in Spark as here.
    """
    return _fn("strpos", _col(str), to_literal(substr))


def translate(srcCol: Any, matching: _str, replace: _str) -> Column:
    """Character-by-character substitution."""
    return _fn("translate", _col(srcCol), to_literal(matching), to_literal(replace))


def levenshtein(left: Any, right: Any) -> Column:
    return _fn("levenshtein", _col(left), _col(right))


def left(str: Any, len: Any) -> Column:
    return _fn("left", _col(str), _value(len))


def right(str: Any, len: Any) -> Column:
    return _fn("right", _col(str), _value(len))


def btrim(str: Any, trim: _str | None = None) -> Column:
    if trim is None:
        return _fn("trim", _col(str))
    return _fn("trim", _col(str), to_literal(trim))


def substring_index(str: Any, delim: _str, count: int) -> Column:
    """The substring before the `count`-th occurrence of `delim`.

    A negative `count` counts from the right, as in Spark. DuckDB has no equivalent,
    so this is built from `str_split` and `list_slice` -- and because the pieces are
    1-indexed while Spark's count is a *number of* delimiters, the slice bounds are
    where a mistake would hide. Covered by tests on both signs.
    """
    parts = sg.Anonymous(this="str_split", expressions=[_col(str), to_literal(delim)])
    if count >= 0:
        chosen = sg.Anonymous(
            this="list_slice", expressions=[parts, to_literal(1), to_literal(count)]
        )
    else:
        size_of = sg.Anonymous(this="len", expressions=[parts.copy()])
        start = sg.Paren(this=sg.Add(this=size_of, expression=to_literal(count + 1)))
        chosen = sg.Anonymous(
            this="list_slice",
            expressions=[parts, start, sg.Anonymous(this="len", expressions=[parts.copy()])],
        )
    return _fn("array_to_string", chosen, to_literal(delim))


# ---------------------------------------------------------------------------
# Dates -- second tranche
# ---------------------------------------------------------------------------


def add_months(start: Any, months: Any) -> Column:
    """`start` plus `months` calendar months, as a DATE.

    DuckDB widens the result to a TIMESTAMP, as it does for `date_add`.
    """
    interval = sg.Anonymous(this="to_months", expressions=[_value(months)])
    return _as_date(_fn("date_add", _col(start), interval))


def months_between(date1: Any, date2: Any, roundOff: bool = True) -> Column:
    """Whole months from `date2` to `date1` -- Spark's argument order.

    Spark returns a fractional month and rounds to 8 decimal places by default;
    DuckDB's `date_diff('month', ...)` truncates to whole months. The fractional part
    is **not** emulated, so this differs from Spark for dates that are not on the same
    day of the month. Recorded in `compat/divergence.md`.
    """
    if not roundOff:
        raise PySparkValueError("months_between(roundOff=False) is not supported yet.")
    return _fn("date_diff", to_literal("month"), _col(date2), _col(date1))


def last_day(date: Any) -> Column:
    """The last day of the month `date` falls in."""
    return _as_date(_fn("last_day", _col(date)))


def weekofyear(col: Any) -> Column:
    """ISO week number, as Spark's `weekofyear`."""
    return _fn("weekofyear", _col(col))


def trunc(date: Any, format: _str) -> Column:
    """Truncate a date to a unit. Spark's `trunc` returns a DATE."""
    return _as_date(_fn("date_trunc", to_literal(format), _col(date)))


def make_date(year: Any, month: Any, day: Any) -> Column:
    return _fn("make_date", _col(year), _col(month), _col(day))


def timestamp_seconds(col: Any) -> Column:
    """A UNIX epoch second count as a timestamp.

    `make_timestamp` takes microseconds and needs no timezone database; DuckDB's
    `to_timestamp` returns a TIMESTAMPTZ and fails without one.
    """
    micros = sg.Mul(this=_col(col), expression=to_literal(1_000_000))
    return _fn("make_timestamp", micros)


def from_unixtime(timestamp: Any, format: _str | None = None) -> Column:
    """An epoch second count formatted as a string.

    Spark's default format is `yyyy-MM-dd HH:mm:ss`; the strftime equivalent is used
    because the two pattern languages differ (see `date_format`).
    """
    if format is not None:
        raise PySparkValueError(
            "from_unixtime(format=...) is not supported: Spark's patterns are Java's "
            "and DuckDB's are strftime. Use date_format on the timestamp instead."
        )
    return _fn("strftime", timestamp_seconds(timestamp)._copy(), to_literal("%Y-%m-%d %H:%M:%S"))


def unix_timestamp(timestamp: Any = None, format: _str | None = None) -> Column:
    """Seconds since the epoch, as a whole number."""
    if format is not None:
        raise PySparkValueError("unix_timestamp(format=...) is not supported yet.")
    source = current_timestamp()._copy() if timestamp is None else _col(timestamp)
    return Column(sg.cast(sg.Anonymous(this="epoch", expressions=[source]), "BIGINT"))


# ---------------------------------------------------------------------------
# Aggregates -- second tranche
# ---------------------------------------------------------------------------


def stddev_samp(col: Any) -> Column:
    return _fn("stddev_samp", _col(col))


def stddev_pop(col: Any) -> Column:
    return _fn("stddev_pop", _col(col))


def var_samp(col: Any) -> Column:
    return _fn("var_samp", _col(col))


def var_pop(col: Any) -> Column:
    return _fn("var_pop", _col(col))


def skewness(col: Any) -> Column:
    return _fn("skewness", _col(col))


def kurtosis(col: Any) -> Column:
    return _fn("kurtosis", _col(col))


def corr(col1: Any, col2: Any) -> Column:
    return _fn("corr", _col(col1), _col(col2))


def covar_samp(col1: Any, col2: Any) -> Column:
    return _fn("covar_samp", _col(col1), _col(col2))


def covar_pop(col1: Any, col2: Any) -> Column:
    return _fn("covar_pop", _col(col1), _col(col2))


def median(col: Any) -> Column:
    return _fn("median", _col(col))


def mode(col: Any) -> Column:
    return _fn("mode", _col(col))


def any_value(col: Any, ignoreNulls: bool = False) -> Column:
    if ignoreNulls:
        raise PySparkValueError("any_value(ignoreNulls=True) is not supported yet.")
    return _fn("any_value", _col(col))


def count_if(col: Any) -> Column:
    return _fn("count_if", _col(col))


def max_by(col: Any, ord: Any) -> Column:
    return _fn("max_by", _col(col), _col(ord))


def min_by(col: Any, ord: Any) -> Column:
    return _fn("min_by", _col(col), _col(ord))


def bool_and(col: Any) -> Column:
    return _fn("bool_and", _col(col))


def bool_or(col: Any) -> Column:
    return _fn("bool_or", _col(col))


def percentile_approx(col: Any, percentage: float, accuracy: int | None = None) -> Column:
    """An approximate percentile. `accuracy` is not adjustable in DuckDB."""
    if accuracy is not None:
        raise PySparkValueError(
            "percentile_approx(accuracy=...) is not adjustable: DuckDB's t-digest has "
            "a fixed configuration."
        )
    return _fn("approx_quantile", _col(col), _value(percentage))


# ---------------------------------------------------------------------------
# Arrays and maps
# ---------------------------------------------------------------------------


def array_contains(col: Any, value: Any) -> Column:
    return _fn("list_contains", _col(col), _value(value))


def array_distinct(col: Any) -> Column:
    return _fn("list_distinct", _col(col))


def array_position(col: Any, value: Any) -> Column:
    """1-indexed position, **0 when absent** -- Spark's convention.

    DuckDB's `list_position` returns NULL for a missing element, which would read as
    "unknown" rather than "not there".
    """
    return Column(
        sg.Coalesce(
            this=sg.Anonymous(this="list_position", expressions=[_col(col), _value(value)]),
            expressions=[to_literal(0)],
        )
    )


def array_remove(col: Any, element: Any) -> Column:
    return _fn("list_filter", _col(col), _lambda_ne(element))


def _lambda_ne(element: Any) -> sg.Expression:
    """`x -> x <> element`, for `array_remove`."""
    return sg.Lambda(
        this=sg.NEQ(this=sg.column("x"), expression=_value(element)),
        expressions=[sg.to_identifier("x")],
    )


def array_sort(col: Any) -> Column:
    return _fn("list_sort", _col(col))


def sort_array(col: Any, asc: bool = True) -> Column:
    if not asc:
        return _fn("list_reverse_sort", _col(col))
    return _fn("list_sort", _col(col))


def array_max(col: Any) -> Column:
    return _fn("list_max", _col(col))


def array_min(col: Any) -> Column:
    return _fn("list_min", _col(col))


def array_join(col: Any, delimiter: _str, null_replacement: _str | None = None) -> Column:
    if null_replacement is not None:
        raise PySparkValueError("array_join(null_replacement=...) is not supported yet.")
    return _fn("array_to_string", _col(col), to_literal(delimiter))


def array_union(col1: Any, col2: Any) -> Column:
    """Spark's `array_union` de-duplicates; DuckDB's `list_concat` does not."""
    return _fn("list_distinct", _fn("list_concat", _col(col1), _col(col2))._copy())


def array_intersect(col1: Any, col2: Any) -> Column:
    return _fn("list_intersect", _col(col1), _col(col2))


def arrays_overlap(a1: Any, a2: Any) -> Column:
    return _fn("list_has_any", _col(a1), _col(a2))


def element_at(col: Any, extraction: Any) -> Column:
    """**1-indexed** for arrays, unlike `getItem` -- which is Spark's own inconsistency.

    For a map, `extraction` is the key.
    """
    return _fn("list_extract", _col(col), _value(extraction))


def slice(x: Any, start: int, length: int) -> Column:
    """1-indexed, as Spark's `slice`."""
    end = (
        sg.Add(this=_value(start), expression=to_literal(length - 1))
        if isinstance(start, int)
        else None
    )
    if end is None:
        raise PySparkValueError("slice() needs an integer start.")
    return _fn("list_slice", _col(x), _value(start), end)


def flatten(col: Any) -> Column:
    return _fn("flatten", _col(col))


def sequence(start: Any, stop: Any, step: Any = None) -> Column:
    if step is not None:
        return _fn("range", _col(start), _inclusive(stop), _value(step))
    return _fn("range", _col(start), _inclusive(stop))


def _inclusive(stop: Any) -> sg.Expression:
    """DuckDB's `range` excludes its endpoint; Spark's `sequence` includes it."""
    return sg.Paren(this=sg.Add(this=_col(stop), expression=to_literal(1)))


def map_keys(col: Any) -> Column:
    return _fn("map_keys", _col(col))


def map_values(col: Any) -> Column:
    return _fn("map_values", _col(col))


# ---------------------------------------------------------------------------
# Null handling -- second tranche
# ---------------------------------------------------------------------------


def ifnull(col1: Any, col2: Any) -> Column:
    return coalesce(col1, col2)


def nvl2(col1: Any, col2: Any, col3: Any) -> Column:
    """`col2` when `col1` is not NULL, else `col3`."""
    return Column(
        sg.Case(
            ifs=[
                sg.If(
                    this=sg.Not(this=sg.Is(this=_col(col1), expression=sg.null())),
                    true=_col(col2),
                )
            ],
            default=_col(col3),
        )
    )


def hash(*cols: Any) -> Column:
    """A hash of its arguments. **Not Spark's Murmur3** -- see divergence.md.

    Spark's `hash` is a specific Murmur3 variant whose values other Spark jobs may
    depend on; DuckDB's is its own. Values will not match Spark's, so this is safe for
    bucketing within one query and unsafe for anything persisted or compared across
    engines.

    Cast to `BIGINT` because DuckDB returns `UBIGINT`, and Spark has no unsigned
    64-bit type -- `icetl.plan.analysis` rightly refuses one rather than silently
    truncating, so an uncast hash column could not be described at all.
    """
    if not cols:
        raise PySparkValueError("hash() needs at least one column.")
    return Column(sg.cast(_fn("hash", *[_col(c) for c in cols])._copy(), "BIGINT"))

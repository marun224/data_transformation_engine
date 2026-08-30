"""`icetl.sql.functions` -- the `F.*` namespace.

**Built on sqlglot's typed nodes, not on raw SQL strings.** `sg.Concat` knows that
The reference engine's `concat` propagates NULL and DuckDB's does not, and generates `a || b`
accordingly; `sg.Split` knows the reference engine's separator is a regex and reaches for
`str_split_regex`. Every function written as a typed node gets that translation for
free and keeps it when DuckDB changes. `_fn` -- a bare `sg.Anonymous` -- is the
fallback for the cases sqlglot has no node for, and each use is a small bet that the
DuckDB spelling is also the reference engine one, so each is spot-checked in the tests.

**Where the reference engine and DuckDB genuinely disagree**, the difference is fixed here and
recorded in `compat/divergence.md`. `dayofweek` is the one in this module: the reference engine
numbers Sunday 1, DuckDB numbers it 0, and nothing about that surfaces as an error.

**A string argument names a column**, which is the opposite of what a string means to
a `Column` operator. `F.upper("name")` upper-cases the column `name`; `col("a") ==
"name"` compares against the *string* "name". That asymmetry is deliberate, and
`_col` versus `to_expression` is where it lives.

Parameter names are stable, because callers pass them by keyword.
That means shadowing the `str` builtin inside `expr`, which is why `_str` exists.
"""

from __future__ import annotations

from typing import Any

import sqlglot

# sqlglot's expression module is aliased `sg` here, not the usual `exp`, because
# The reference engine has a function *called* `exp` and defining it would shadow the module for
# every function below it -- silently, until one of them was called. The same hazard
# is why `_str` exists. Everywhere else in the codebase the alias is `exp`.
from sqlglot import exp as sg

from icetl.compat import SQL_DIALECT
from icetl.errors import EngineTypeError, EngineValueError, ParseException
from icetl.plan.builder import as_expression
from icetl.sql.column import Column, _column_from_name, to_expression, to_literal

__all__ = [
    "abs",
    "acos",
    "acosh",
    "add_months",
    "any",
    "any_value",
    "approx_count_distinct",
    "array",
    "array_agg",
    "array_append",
    "array_compact",
    "array_contains",
    "array_distinct",
    "array_except",
    "array_intersect",
    "array_join",
    "array_max",
    "array_min",
    "array_position",
    "array_prepend",
    "array_remove",
    "array_repeat",
    "array_size",
    "array_sort",
    "array_union",
    "arrays_overlap",
    "asc",
    "asc_nulls_first",
    "asc_nulls_last",
    "ascii",
    "asin",
    "asinh",
    "assert_true",
    "atan",
    "atan2",
    "atanh",
    "avg",
    "base64",
    "bin",
    "bit_and",
    "bit_count",
    "bit_length",
    "bit_or",
    "bit_xor",
    "bitwise_not",
    "bool_and",
    "bool_or",
    "btrim",
    "cardinality",
    "cbrt",
    "ceil",
    "ceiling",
    "char",
    "char_length",
    "character_length",
    "chr",
    "coalesce",
    "col",
    "collect_list",
    "collect_set",
    "column",
    "concat",
    "concat_ws",
    "contains",
    "corr",
    "cos",
    "cosh",
    "cot",
    "count",
    "countDistinct",
    "count_if",
    "covar_pop",
    "covar_samp",
    "csc",
    "curdate",
    "current_catalog",
    "current_database",
    "current_date",
    "current_schema",
    "current_timestamp",
    "current_user",
    "date_add",
    "date_diff",
    "date_format",
    "date_from_unix_date",
    "date_part",
    "date_sub",
    "date_trunc",
    "dateadd",
    "datediff",
    "datepart",
    "day",
    "dayofmonth",
    "dayofweek",
    "dayofyear",
    "degrees",
    "desc",
    "desc_nulls_first",
    "desc_nulls_last",
    "e",
    "element_at",
    "elt",
    "endswith",
    "equal_null",
    "every",
    "exp",
    "expm1",
    "expr",
    "extract",
    "factorial",
    "find_in_set",
    "first",
    "flatten",
    "floor",
    "format_number",
    "format_string",
    "from_unixtime",
    "get",
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
    "lcase",
    "least",
    "left",
    "length",
    "levenshtein",
    "lit",
    "ln",
    "localtimestamp",
    "locate",
    "log",
    "log1p",
    "log2",
    "log10",
    "lower",
    "lpad",
    "ltrim",
    "make_date",
    "make_timestamp",
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
    "next_day",
    "now",
    "nullif",
    "nvl",
    "nvl2",
    "octet_length",
    "overlay",
    "percentile",
    "percentile_approx",
    "pi",
    "pmod",
    "positive",
    "pow",
    "power",
    "printf",
    "quarter",
    "radians",
    "raise_error",
    "rand",
    "randn",
    "regexp",
    "regexp_count",
    "regexp_extract",
    "regexp_instr",
    "regexp_like",
    "regexp_replace",
    "regexp_substr",
    "regr_avgx",
    "regr_avgy",
    "regr_count",
    "regr_intercept",
    "regr_r2",
    "regr_slope",
    "regr_sxx",
    "regr_sxy",
    "regr_syy",
    "repeat",
    "replace",
    "reverse",
    "right",
    "rint",
    "round",
    "rpad",
    "rtrim",
    "sec",
    "second",
    "sequence",
    "session_user",
    "sha",
    "sha1",
    "sha2",
    "shiftleft",
    "shiftright",
    "shiftrightunsigned",
    "sign",
    "signum",
    "sin",
    "sinh",
    "size",
    "skewness",
    "slice",
    "some",
    "sort_array",
    "split",
    "split_part",
    "sqrt",
    "startswith",
    "std",
    "stddev",
    "stddev_pop",
    "stddev_samp",
    "struct",
    "substr",
    "substring",
    "substring_index",
    "sum",
    "sum_distinct",
    "tan",
    "tanh",
    "timestamp_micros",
    "timestamp_millis",
    "timestamp_seconds",
    "to_date",
    "to_timestamp",
    "to_unix_timestamp",
    "translate",
    "trim",
    "trunc",
    "try_divide",
    "try_to_timestamp",
    "ucase",
    "unbase64",
    "unhex",
    "unix_date",
    "unix_micros",
    "unix_millis",
    "unix_seconds",
    "unix_timestamp",
    "upper",
    "user",
    "var_pop",
    "var_samp",
    "variance",
    "weekday",
    "weekofyear",
    "when",
    "width_bucket",
    "xxhash64",
    "year",
]

_str = str  # captured before `expr`'s parameter name shadows the builtin


# ---------------------------------------------------------------------------
# Argument coercion
# ---------------------------------------------------------------------------


def _col(value: Any) -> sg.Expression:
    """Coerce a function argument: a string names a *column*, as in the reference API."""
    if isinstance(value, Column):
        return value._copy()
    if isinstance(value, _str):
        return _column_from_name(value)._copy()
    raise EngineTypeError(
        f"Expected a column name or Column, got {type(value).__name__}. "
        f"Wrap a literal value in F.lit()."
    )


def _value(value: Any) -> sg.Expression:
    """Coerce an argument that is a *value*, not a column reference."""
    return to_expression(value)


def _fn(name: _str, *args: sg.Expression) -> Column:
    """A function sqlglot has no typed node for.

    Every use is a bet that DuckDB spells it the way the reference engine does, so every use is
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
        raise EngineTypeError(f"col() expects a column name as a string, got {type(col).__name__}.")
    return _column_from_name(col)


#: The reference engine's alias for `col`.
column = col


def lit(col: Any) -> Column:
    """A literal value as a Column. A Column passed in is returned unchanged."""
    if isinstance(col, Column):
        return col
    return Column(to_literal(col))


def expr(str: _str) -> Column:
    """Parse a SQL expression into a Column.

    >>> expr("amount * 2")
    Column<'amount * 2'>
    """
    if not isinstance(str, _str):
        raise EngineTypeError(f"expr() expects a string, got {type(str).__name__}.")
    try:
        parsed = sqlglot.parse_one(str, read=SQL_DIALECT)
    except Exception as exc:
        raise ParseException(f"Could not parse the expression {str!r}: {exc}") from exc
    return Column(as_expression(parsed))


# ---------------------------------------------------------------------------
# Conditionals and null handling
# ---------------------------------------------------------------------------


def when(condition: Column, value: Any) -> Column:
    """Start a `CASE`. Chain `.when(...)` for more branches and `.otherwise(...)`.

    With no `otherwise`, an unmatched row is NULL -- which is the reference engine's behaviour and
    also SQL's, so no rule is needed to make it so.
    """
    if not isinstance(condition, Column):
        raise EngineTypeError(f"when() expects a Column condition, got {type(condition).__name__}.")
    return Column(sg.Case(ifs=[sg.If(this=condition._copy(), true=_value(value))]))


def coalesce(*cols: Any) -> Column:
    """The first non-NULL of its arguments."""
    if not cols:
        raise EngineValueError("coalesce() needs at least one column.")
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
    """The largest of its arguments, **ignoring NULLs** -- as the reference engine does.

    Deliberately `_fn` rather than `sg.Greatest`: sqlglot wraps the typed node in a
    `CASE WHEN any IS NULL THEN NULL` for DuckDB, which is right for dialects whose
    `GREATEST` propagates NULL and wrong for the reference engine. DuckDB's own `greatest` already
    skips NULLs, so the plain call is the conformant one.
    """
    if len(cols) < 2:
        raise EngineValueError("greatest() needs at least two columns.")
    return _fn("greatest", *[_col(c) for c in cols])


def least(*cols: Any) -> Column:
    """The smallest of its arguments, **ignoring NULLs** -- as the reference engine does.

    `_fn` for the same reason as `greatest`.
    """
    if len(cols) < 2:
        raise EngineValueError("least() needs at least two columns.")
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
        raise EngineValueError("countDistinct() needs at least one column.")
    return Column(sg.Count(this=sg.Distinct(expressions=[_col(c) for c in cols])))


def approx_count_distinct(col: Any, rsd: float | None = None) -> Column:
    if rsd is not None:
        raise EngineValueError(
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


#: The reference engine exposes both spellings.
mean = avg


def min(col: Any) -> Column:
    return Column(sg.Min(this=_col(col)))


def max(col: Any) -> Column:
    return Column(sg.Max(this=_col(col)))


def first(col: Any, ignorenulls: bool = False) -> Column:
    """The first value in the group. DuckDB's `first` keeps NULLs, as the reference does."""
    if ignorenulls:
        raise EngineValueError(
            "first(ignorenulls=True) is not supported yet: DuckDB has no null-skipping "
            "first aggregate, and silently keeping them would be a wrong answer."
        )
    return _fn("first", _col(col))


def last(col: Any, ignorenulls: bool = False) -> Column:
    """The last value in the group. DuckDB's `last` keeps NULLs, as the reference engine's does."""
    if ignorenulls:
        raise EngineValueError(
            "last(ignorenulls=True) is not supported yet: DuckDB has no null-skipping "
            "last aggregate, and silently keeping them would be a wrong answer."
        )
    return _fn("last", _col(col))


def stddev(col: Any) -> Column:
    """Sample standard deviation, as the reference engine's `stddev`."""
    return Column(sg.Stddev(this=_col(col)))


def variance(col: Any) -> Column:
    """Sample variance, as the reference engine's `variance`."""
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
    """Concatenate. NULL in, NULL out -- which is the reference engine, and *not* DuckDB's `concat`.

    `sg.Concat` generates `a || b` for DuckDB, and `||` propagates NULL where
    `concat()` would silently skip it.
    """
    if not cols:
        raise EngineValueError("concat() needs at least one column.")
    return Column(sg.Concat(expressions=[_col(c) for c in cols]))


def concat_ws(sep: _str, *cols: Any) -> Column:
    """Join with a separator, **skipping** NULLs -- the reference behaviour, which DuckDB shares.

    Note the contrast with `concat`, which propagates NULL. The reference engine really does treat
    the two differently, and so this one is `_fn`: `sg.ConcatWs` would be wrapped in
    a NULL-propagating `CASE`, turning "skip" into "poison".
    """
    return _fn("concat_ws", to_literal(sep), *[_col(c) for c in cols])


def substring(str: Any, pos: int, len: int) -> Column:
    """1-indexed, as the reference engine and SQL both are."""
    return Column(sg.Substring(this=_col(str), start=_value(pos), length=_value(len)))


def split(str: Any, pattern: _str, limit: int = -1) -> Column:
    """Split on a **regex**, as the reference engine does -- not on a literal.

    `sg.RegexpSplit`, not `sg.Split`: the latter generates DuckDB's `str_split`, which
    treats the pattern literally and returns the whole string as a single element. No
    error, just one row where there should have been several -- which is why this is
    the node the reference engine parser produces too, and why it has its own test.
    """
    if limit != -1:
        raise EngineValueError("split(limit=...) is not supported yet.")
    return Column(sg.RegexpSplit(this=_col(str), expression=to_literal(pattern)))


def replace(src: Any, search: Any, replace: Any) -> Column:
    return _fn("replace", _col(src), _value(search), _value(replace))


def regexp_replace(str: Any, pattern: _str, replacement: _str) -> Column:
    """Replace **every** match, as the reference engine does.

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
    """The `idx`-th capture group, or empty string when there is no match (the reference engine)."""
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
    """1-indexed position of `substr`, or 0 when absent -- the reference engine's convention."""
    if pos != 1:
        raise EngineValueError("locate(pos=...) beyond 1 is not supported yet.")
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
    """Half away from zero, which is the reference engine's HALF_UP -- and DuckDB's default."""
    return Column(sg.Round(this=_col(col), decimals=_value(scale)))


def sqrt(col: Any) -> Column:
    return Column(sg.Sqrt(this=_col(col)))


def exp(col: Any) -> Column:
    return _fn("exp", _col(col))


def pow(col1: Any, col2: Any) -> Column:
    return Column(sg.Pow(this=_col(col1), expression=_value(col2)))


def log(arg1: Any, arg2: Any = None) -> Column:
    """`log(x)` is the natural log; `log(base, x)` is log to a base -- the reference signature.

    The two-argument form is `_fn`, not `sg.Log`: DuckDB spells it `log(base, x)`,
    the same order the reference engine does, but the typed node renders its operands reversed, so
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
    """The reference engine numbers Sunday 1 through Saturday 7; DuckDB numbers Sunday 0.

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
        raise EngineValueError("to_date(format=...) is not supported yet.")
    return Column(sg.cast(_col(col), "DATE"))


def to_timestamp(col: Any, format: _str | None = None) -> Column:
    if format is not None:
        raise EngineValueError("to_timestamp(format=...) is not supported yet.")
    return Column(sg.cast(_col(col), "TIMESTAMP"))


def _days(count: sg.Expression) -> sg.Expression:
    """`INTERVAL n DAY`, the argument DuckDB's `date_add` expects.

    Takes an already-coerced expression rather than a raw value, so `date_sub` can
    hand it a negation.
    """
    return sg.Anonymous(this="to_days", expressions=[count])


def _as_date(column: Column) -> Column:
    """Cast back to DATE.

    DuckDB's `date_add` widens a DATE to a TIMESTAMP; the reference engine's `date_add` returns a
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
    """Whole days between two dates, `end - start` -- the reference engine's argument order."""
    return _fn("date_diff", to_literal("day"), _col(start), _col(end))


date_diff = datediff


def date_trunc(format: _str, timestamp: Any) -> Column:
    return _fn("date_trunc", to_literal(format), _col(timestamp))


def date_format(date: Any, format: _str) -> Column:
    """Format a date. **the reference engine's pattern syntax is Java's, DuckDB's is strftime.**

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
    """`sha2(col, numBits)`. Only the widths DuckDB implements are accepted."""
    if numBits not in (256, 512):
        raise EngineValueError(
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
    """A struct whose field names are the column names, as the reference engine does."""
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
    """The reference engine's `pmod`: the remainder is **never negative**.

    SQL's `%` keeps the sign of the dividend -- DuckDB gives `-7 % 3 = -1` where the reference
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
    """Binary representation as a string, as the reference engine's `bin`."""
    return _fn("bin", _col(col))


def negative(col: Any) -> Column:
    return Column(sg.Neg(this=_col(col)))


def positive(col: Any) -> Column:
    return Column(_col(col))


# ---------------------------------------------------------------------------
# Strings -- second tranche
# ---------------------------------------------------------------------------


def instr(str: Any, substr: _str) -> Column:
    """1-indexed position of `substr`, 0 when absent -- the reference engine's convention.

    Note the argument order is the reverse of `locate`, in the reference engine as here.
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

    A negative `count` counts from the right, as in the reference engine. DuckDB has no equivalent,
    so this is built from `str_split` and `list_slice` -- and because the pieces are
    1-indexed while the reference engine's count is a *number of* delimiters, the slice bounds are
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
    """Whole months from `date2` to `date1` -- the reference engine's argument order.

    The reference engine returns a fractional month and rounds to 8 decimal places by default;
    DuckDB's `date_diff('month', ...)` truncates to whole months. The fractional part
    is **not** emulated, so this differs from the reference engine for dates that are not on the
    same
    day of the month. Recorded in `compat/divergence.md`.
    """
    if not roundOff:
        raise EngineValueError("months_between(roundOff=False) is not supported yet.")
    return _fn("date_diff", to_literal("month"), _col(date2), _col(date1))


def last_day(date: Any) -> Column:
    """The last day of the month `date` falls in."""
    return _as_date(_fn("last_day", _col(date)))


def weekofyear(col: Any) -> Column:
    """ISO week number, as the reference engine's `weekofyear`."""
    return _fn("weekofyear", _col(col))


def trunc(date: Any, format: _str) -> Column:
    """Truncate a date to a unit. The reference engine's `trunc` returns a DATE."""
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

    The reference engine's default format is `yyyy-MM-dd HH:mm:ss`; the strftime equivalent is used
    because the two pattern languages differ (see `date_format`).
    """
    if format is not None:
        raise EngineValueError(
            "from_unixtime(format=...) is not supported: the reference patterns are Java's "
            "and DuckDB's are strftime. Use date_format on the timestamp instead."
        )
    return _fn("strftime", timestamp_seconds(timestamp)._copy(), to_literal("%Y-%m-%d %H:%M:%S"))


def unix_timestamp(timestamp: Any = None, format: _str | None = None) -> Column:
    """Seconds since the epoch, as a whole number."""
    if format is not None:
        raise EngineValueError("unix_timestamp(format=...) is not supported yet.")
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
        raise EngineValueError("any_value(ignoreNulls=True) is not supported yet.")
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
        raise EngineValueError(
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
    """1-indexed position, **0 when absent** -- the reference engine's convention.

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
        raise EngineValueError("array_join(null_replacement=...) is not supported yet.")
    return _fn("array_to_string", _col(col), to_literal(delimiter))


def array_union(col1: Any, col2: Any) -> Column:
    """The reference engine's `array_union` de-duplicates; DuckDB's `list_concat` does not."""
    return _fn("list_distinct", _fn("list_concat", _col(col1), _col(col2))._copy())


def array_intersect(col1: Any, col2: Any) -> Column:
    return _fn("list_intersect", _col(col1), _col(col2))


def arrays_overlap(a1: Any, a2: Any) -> Column:
    return _fn("list_has_any", _col(a1), _col(a2))


def element_at(col: Any, extraction: Any) -> Column:
    """**1-indexed** for arrays, unlike `getItem` -- the reference API's own inconsistency.

    For a map, `extraction` is the key.
    """
    return _fn("list_extract", _col(col), _value(extraction))


def slice(x: Any, start: int, length: int) -> Column:
    """1-indexed, as the reference engine's `slice`."""
    end = (
        sg.Add(this=_value(start), expression=to_literal(length - 1))
        if isinstance(start, int)
        else None
    )
    if end is None:
        raise EngineValueError("slice() needs an integer start.")
    return _fn("list_slice", _col(x), _value(start), end)


def flatten(col: Any) -> Column:
    return _fn("flatten", _col(col))


def sequence(start: Any, stop: Any, step: Any = None) -> Column:
    if step is not None:
        return _fn("range", _col(start), _inclusive(stop), _value(step))
    return _fn("range", _col(start), _inclusive(stop))


def _inclusive(stop: Any) -> sg.Expression:
    """DuckDB's `range` excludes its endpoint; the reference engine's `sequence` includes it."""
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
    """A hash of its arguments. **Not the reference engine's Murmur3** -- see divergence.md.

    The reference engine's `hash` is a specific Murmur3 variant whose values other the reference
    engine jobs may
    depend on; DuckDB's is its own. Values will not match the reference engine's, so this is safe
    for
    bucketing within one query and unsafe for anything persisted or compared across
    engines.

    Cast to `BIGINT` because DuckDB returns `UBIGINT`, and the reference engine has no unsigned
    64-bit type -- `icetl.plan.analysis` rightly refuses one rather than silently
    truncating, so an uncast hash column could not be described at all.
    """
    if not cols:
        raise EngineValueError("hash() needs at least one column.")
    return Column(sg.cast(_fn("hash", *[_col(c) for c in cols])._copy(), "BIGINT"))


# ---------------------------------------------------------------------------
# Strings -- third tranche
# ---------------------------------------------------------------------------


def char(col: Any) -> Column:
    """The character for a code point. The reference engine's `char` and `chr` are one function."""
    return _fn("chr", _col(col))


#: The reference engine spells `char` both ways.
chr = char


def char_length(col: Any) -> Column:
    """Length in *characters*, as `length` -- contrast `octet_length`."""
    return Column(sg.Length(this=_col(col)))


#: The reference engine's other spelling of `char_length`.
character_length = char_length


def bit_length(col: Any) -> Column:
    """Length in bits. `bit_length('abc')` is 24, and 48 for a two-byte character."""
    return _fn("bit_length", _col(col))


def octet_length(col: Any) -> Column:
    """Length in *bytes*, which differs from `length` for any non-ASCII string.

    DuckDB spells the byte count `strlen`; its `length` counts characters.
    """
    return _fn("strlen", _col(col))


def ucase(col: Any) -> Column:
    return upper(col)


def lcase(col: Any) -> Column:
    return lower(col)


def startswith(str: Any, prefix: Any) -> Column:
    return _fn("starts_with", _col(str), _col(prefix))


def endswith(str: Any, suffix: Any) -> Column:
    return _fn("ends_with", _col(str), _col(suffix))


def contains(left: Any, right: Any) -> Column:
    return _fn("contains", _col(left), _col(right))


def substr(str: Any, pos: Any, len: Any = None) -> Column:
    """`substring` under the name the reference engine also gives it."""
    if len is None:
        return _fn("substring", _col(str), _col(pos))
    return _fn("substring", _col(str), _col(pos), _col(len))


def overlay(src: Any, replace: Any, pos: Any, len: int = -1) -> Column:
    """Replace `len` characters of `src` from `pos` with `replace`.

    A negative `len` -- the default -- means "as many characters as `replace` is
    long", so `overlay('Basic SQL', '_', 6)` is `Basic_SQL`. DuckDB has no `overlay`,
    so it is composed from `substring`, which is 1-indexed as the reference engine's is.
    """
    source = _col(src)
    replacement = _col(replace)
    position = _col(pos)
    width: sg.Expression = (
        to_literal(len)
        if isinstance(len, int) and len >= 0
        else sg.Anonymous(this="length", expressions=[replacement.copy()])
    )
    head = sg.Anonymous(
        this="substring",
        expressions=[source, to_literal(1), sg.Sub(this=position, expression=to_literal(1))],
    )
    tail = sg.Anonymous(
        this="substring",
        expressions=[source.copy(), sg.Add(this=position.copy(), expression=width)],
    )
    return Column(sg.Concat(expressions=[head, replacement, tail]))


def format_string(format: _str, *cols: Any) -> Column:
    """`printf`-style formatting.

    The reference engine takes a Java format string, whose `%s`/`%d`/`%f` conversions are the ones
    DuckDB's `printf` also understands. Java-only conversions -- `%n`, argument
    indexes such as `%1$s` -- are not translated.
    """
    if not isinstance(format, _str):
        raise EngineTypeError(
            f"format_string() expects a format string, got {type(format).__name__}."
        )
    return _fn("printf", to_literal(format), *[_col(c) for c in cols])


#: The reference engine's other spelling of `format_string`.
printf = format_string


def format_number(col: Any, d: int) -> Column:
    """A number as a string with thousands separators and `d` decimal places.

    `format_number(12345.678, 2)` is `'12,345.68'`.
    """
    if not isinstance(d, int) or d < 0:
        raise EngineValueError("format_number() needs a non-negative integer d.")
    return _fn("format", to_literal("{:,." + _str(d) + "f}"), _col(col))


def split_part(src: Any, delimiter: Any, partNum: Any) -> Column:
    """The `partNum`-th field of `src`, splitting on a *literal* `delimiter`.

    1-indexed; a negative index counts from the end; out of range is the **empty
    string**, which is the reference engine's choice where DuckDB would give NULL.
    """
    if isinstance(partNum, int) and partNum == 0:
        raise EngineValueError("split_part() indexes from 1; 0 is not a valid partNum.")
    parts = sg.Anonymous(this="str_split", expressions=[_col(src), _col(delimiter)])
    return Column(
        sg.Coalesce(
            this=sg.Anonymous(this="list_extract", expressions=[parts, _col(partNum)]),
            expressions=[to_literal("")],
        )
    )


def find_in_set(str: Any, str_array: Any) -> Column:
    """The 1-based position of `str` among the comma-separated fields of `str_array`.

    Zero when it is absent, and zero when `str` itself contains a comma -- the reference engine
    refuses to match across two fields, which is what the `CASE` is for.
    """
    needle = _col(str)
    parts = sg.Anonymous(this="str_split", expressions=[_col(str_array), to_literal(",")])
    found = sg.Coalesce(
        this=sg.Anonymous(this="list_position", expressions=[parts, needle]),
        expressions=[to_literal(0)],
    )
    return Column(
        sg.Case(
            ifs=[
                sg.If(
                    this=sg.Anonymous(
                        this="contains", expressions=[needle.copy(), to_literal(",")]
                    ),
                    true=to_literal(0),
                )
            ],
            default=found,
        )
    )


def regexp_count(str: Any, regexp: Any) -> Column:
    """How many times `regexp` matches. No match is 0, not NULL."""
    return Column(
        sg.Length(
            this=sg.Anonymous(this="regexp_extract_all", expressions=[_col(str), _col(regexp)])
        )
    )


def regexp_substr(str: Any, regexp: Any) -> Column:
    """The first match, or **NULL** when there is none.

    DuckDB's `regexp_extract` returns the empty string for no match, which the reference engine
    reserves for a match that genuinely matched nothing -- so the `regexp_matches`
    guard is what keeps the two apart.
    """
    subject = _col(str)
    pattern = _col(regexp)
    return Column(
        sg.Case(
            ifs=[
                sg.If(
                    this=sg.Anonymous(this="regexp_matches", expressions=[subject, pattern]),
                    true=sg.Anonymous(
                        this="regexp_extract", expressions=[subject.copy(), pattern.copy()]
                    ),
                )
            ]
        )
    )


def regexp_instr(str: Any, regexp: Any) -> Column:
    """The 1-based position of the first match, or 0 when there is none."""
    subject = _col(str)
    pattern = _col(regexp)
    return Column(
        sg.Case(
            ifs=[
                sg.If(
                    this=sg.Anonymous(this="regexp_matches", expressions=[subject, pattern]),
                    true=sg.Anonymous(
                        this="instr",
                        expressions=[
                            subject.copy(),
                            sg.Anonymous(
                                this="regexp_extract",
                                expressions=[subject.copy(), pattern.copy()],
                            ),
                        ],
                    ),
                )
            ],
            default=to_literal(0),
        )
    )


def regexp_like(str: Any, regexp: Any) -> Column:
    """Whether `regexp` matches anywhere in `str` -- the function form of `rlike`."""
    return _fn("regexp_matches", _col(str), _col(regexp))


#: The reference engine's other spelling of `regexp_like`.
regexp = regexp_like


def elt(*inputs: Any) -> Column:
    """`elt(n, a, b, ...)` -- the `n`-th of the remaining arguments, 1-indexed.

    Out of range is NULL, as the reference engine's is with ANSI mode off.
    """
    if len(inputs) < 2:
        raise EngineValueError("elt() needs an index and at least one value.")
    index, values = inputs[0], inputs[1:]
    return _fn(
        "list_extract",
        sg.Anonymous(this="list_value", expressions=[_col(v) for v in values]),
        _col(index),
    )


def base64(col: Any) -> Column:
    """Binary as a base64 string. A string argument is read as its bytes."""
    return _fn("to_base64", sg.cast(_col(col), "BLOB"))


def unbase64(col: Any) -> Column:
    """A base64 string back to **binary** -- the reference engine's `unbase64` returns bytes."""
    return _fn("from_base64", _col(col))


# ---------------------------------------------------------------------------
# Math -- third tranche
# ---------------------------------------------------------------------------


def e() -> Column:
    """Euler's number. The reference engine's `e()` takes no arguments."""
    return _fn("exp", to_literal(1))


def pi() -> Column:
    return _fn("pi")


def ln(col: Any) -> Column:
    """The natural logarithm -- the reference engine's other name for `log` of one argument."""
    return _fn("ln", _col(col))


def log1p(col: Any) -> Column:
    """`ln(1 + x)`, computed as written. DuckDB has no `log1p`.

    The reference engine's own implementation is Java's `Math.log1p`, which keeps precision for a
    tiny `x` in a way `ln(1 + x)` does not, so the two drift in the last few digits
    near zero. Recorded in `divergence.md`.
    """
    return _fn("ln", sg.Paren(this=sg.Add(this=to_literal(1), expression=_col(col))))


def expm1(col: Any) -> Column:
    """`exp(x) - 1`, computed as written -- the same precision caveat as `log1p`."""
    return Column(
        sg.Paren(
            this=sg.Sub(
                this=sg.Anonymous(this="exp", expressions=[_col(col)]),
                expression=to_literal(1),
            )
        )
    )


def rint(col: Any) -> Column:
    """Round to the nearest integer, **halves to even** -- Java's `Math.rint`.

    `rint(2.5)` is 2.0 and `rint(3.5)` is 4.0. DuckDB's `round` goes half *away from
    zero* in both directions, so it gives 3 and 4, and -3 where the reference engine gives -2. The
    `CASE` picks the even neighbour on an exact half and defers to `round` otherwise.
    """
    value = _col(col)
    below = sg.Anonymous(this="floor", expressions=[value])
    is_half = sg.EQ(
        this=sg.Paren(this=sg.Sub(this=value.copy(), expression=below.copy())),
        expression=to_literal(0.5),
    )
    below_is_even = sg.EQ(
        this=sg.Mod(
            this=sg.cast(below.copy(), "BIGINT"),
            expression=to_literal(2),
        ),
        expression=to_literal(0),
    )
    even_neighbour = sg.Case(
        ifs=[sg.If(this=below_is_even, true=below.copy())],
        default=sg.Anonymous(this="ceil", expressions=[value.copy()]),
    )
    return Column(
        sg.cast(
            sg.Case(
                ifs=[sg.If(this=is_half, true=even_neighbour)],
                default=sg.Anonymous(this="round", expressions=[value.copy()]),
            ),
            "DOUBLE",
        )
    )


def cot(col: Any) -> Column:
    return _fn("cot", _col(col))


def _reciprocal_of(name: _str, col: Any) -> Column:
    """`1 / f(x)`. DuckDB has neither `csc` nor `sec`."""
    return Column(
        sg.Paren(
            this=sg.Div(
                this=to_literal(1.0),
                expression=sg.Anonymous(this=name, expressions=[_col(col)]),
            )
        )
    )


def csc(col: Any) -> Column:
    """`1 / sin(x)`."""
    return _reciprocal_of("sin", col)


def sec(col: Any) -> Column:
    """`1 / cos(x)`."""
    return _reciprocal_of("cos", col)


def acosh(col: Any) -> Column:
    return _fn("acosh", _col(col))


def asinh(col: Any) -> Column:
    return _fn("asinh", _col(col))


def atanh(col: Any) -> Column:
    return _fn("atanh", _col(col))


def sign(col: Any) -> Column:
    """The reference engine's other name for `signum`."""
    return signum(col)


def power(col1: Any, col2: Any) -> Column:
    """The reference engine's other name for `pow`."""
    return pow(col1, col2)


def bit_count(col: Any) -> Column:
    """How many bits are set. `bit_count(7)` is 3."""
    return _fn("bit_count", _col(col))


def bitwise_not(col: Any) -> Column:
    """The bitwise complement. `bitwise_not(5)` is -6 in two's complement."""
    return Column(sg.BitwiseNot(this=_col(col)))


def shiftrightunsigned(col: Any, numBits: int) -> Column:
    """A right shift that fills with zeroes -- Java's `>>>` on a 64-bit value.

    `shiftrightunsigned(-8, 1)` is 9223372036854775804, not -4: the sign bit is data,
    not a sign. DuckDB has no unsigned shift and refuses to cast a negative to
    `UBIGINT`, so the value is widened to `HUGEINT` and 2^64 added, which reproduces
    the unsigned bit pattern exactly before the shift.
    """
    if not isinstance(numBits, int) or numBits < 1 or numBits > 63:
        raise EngineValueError("shiftrightunsigned() supports a shift of 1 to 63 bits.")
    widened = sg.Paren(
        this=sg.Add(
            this=sg.cast(_col(col), "HUGEINT"),
            expression=sg.Paren(
                this=sg.BitwiseLeftShift(
                    this=sg.cast(to_literal(1), "HUGEINT"), expression=to_literal(64)
                )
            ),
        )
    )
    unsigned = sg.Paren(
        this=sg.Mod(
            this=widened,
            expression=sg.Paren(
                this=sg.BitwiseLeftShift(
                    this=sg.cast(to_literal(1), "HUGEINT"), expression=to_literal(64)
                )
            ),
        )
    )
    return Column(
        sg.cast(
            sg.Paren(this=sg.BitwiseRightShift(this=unsigned, expression=to_literal(numBits))),
            "BIGINT",
        )
    )


def try_divide(left: Any, right: Any) -> Column:
    """Division that is NULL rather than an error when the divisor is zero.

    With ANSI mode off this is what `/` already does, so the two agree; with ANSI mode
    on `/` raises and this still returns NULL, which is the point of the function.
    `sql/conformance.py` owns that rule, and `try_divide` opts out of it by marking the
    division as already safe.
    """
    return Column(
        sg.Case(
            ifs=[
                sg.If(
                    this=sg.EQ(this=_col(right), expression=to_literal(0)),
                    true=sg.null(),
                )
            ],
            default=sg.Div(this=_col(left), expression=_col(right)),
        )
    )


def width_bucket(v: Any, min: Any, max: Any, numBucket: Any) -> Column:
    """Which of `numBucket` equal-width buckets between `min` and `max` holds `v`.

    Buckets are 1-indexed; a value below the range is 0 and one at or above it is
    `numBucket + 1`, as the reference engine's are. Descending bounds -- `min` greater than `max` --
    work too, because the normalised fraction flips sign in both numerator and
    denominator.
    """
    low = _col(min)
    fraction = sg.Paren(
        this=sg.Div(
            this=sg.Paren(this=sg.Sub(this=_col(v), expression=low)),
            expression=sg.Paren(this=sg.Sub(this=_col(max), expression=low.copy())),
        )
    )
    buckets = _col(numBucket)
    return Column(
        sg.Case(
            ifs=[
                sg.If(
                    this=sg.LT(this=fraction, expression=to_literal(0)),
                    true=to_literal(0),
                ),
                sg.If(
                    this=sg.GTE(this=fraction.copy(), expression=to_literal(1)),
                    true=sg.Paren(this=sg.Add(this=buckets, expression=to_literal(1))),
                ),
            ],
            default=sg.Paren(
                this=sg.Add(
                    this=sg.cast(
                        sg.Anonymous(
                            this="floor",
                            expressions=[sg.Mul(this=fraction.copy(), expression=buckets.copy())],
                        ),
                        "BIGINT",
                    ),
                    expression=to_literal(1),
                )
            ),
        )
    )


def rand(seed: int | None = None) -> Column:
    """A uniform random double in [0, 1).

    **The `seed` argument is refused.** the reference engine's seed makes a query reproducible;
    DuckDB seeds its generator per *connection* with a `SETSEED` statement, not per
    expression, so accepting a seed would return unseeded values from a function whose
    whole purpose is reproducibility. Refusing is the honest answer.
    """
    if seed is not None:
        raise EngineValueError(
            "rand(seed=...) is not supported: DuckDB seeds its generator per "
            "connection, not per expression, so the result would not be reproducible. "
            "Call rand() without a seed."
        )
    return _fn("random")


def randn(seed: int | None = None) -> Column:
    """A standard normal random double, by Box-Muller from two uniforms.

    DuckDB has no normal generator. The seed is refused for the same reason as `rand`.
    """
    if seed is not None:
        raise EngineValueError(
            "randn(seed=...) is not supported: DuckDB seeds its generator per "
            "connection, not per expression. Call randn() without a seed."
        )
    uniform = sg.Anonymous(this="random")
    radius = sg.Anonymous(
        this="sqrt",
        expressions=[
            sg.Mul(
                this=to_literal(-2.0),
                expression=sg.Anonymous(this="ln", expressions=[uniform]),
            )
        ],
    )
    angle = sg.Anonymous(
        this="cos",
        expressions=[
            sg.Mul(
                this=sg.Mul(
                    this=to_literal(2.0), expression=sg.Anonymous(this="pi", expressions=[])
                ),
                expression=sg.Anonymous(this="random"),
            )
        ],
    )
    return Column(sg.Paren(this=sg.Mul(this=radius, expression=angle)))


# ---------------------------------------------------------------------------
# Dates -- third tranche
# ---------------------------------------------------------------------------


def now() -> Column:
    """The reference engine's other name for `current_timestamp`."""
    return current_timestamp()


def curdate() -> Column:
    """The reference engine's other name for `current_date`."""
    return current_date()


def localtimestamp() -> Column:
    """The current timestamp **without** a time zone -- the reference engine's TIMESTAMP_NTZ.

    `current_timestamp` carries a zone; this one deliberately does not, which is the
    only difference between them.
    """
    return _fn("current_localtimestamp")


def day(col: Any) -> Column:
    """The reference engine's other name for `dayofmonth`."""
    return dayofmonth(col)


def weekday(col: Any) -> Column:
    """Day of week with **Monday as 0** and Sunday as 6.

    Not to be confused with `dayofweek`, which is the reference engine's *other* numbering -- Sunday
    as 1. DuckDB's `dayofweek` is Sunday as 0, so both the reference engine spellings need
    arithmetic
    and neither can be passed through.
    """
    return Column(
        sg.Paren(
            this=sg.Mod(
                this=sg.Paren(
                    this=sg.Add(
                        this=sg.Anonymous(this="dayofweek", expressions=[_col(col)]),
                        expression=to_literal(6),
                    )
                ),
                expression=to_literal(7),
            )
        )
    )


#: The reference engine's other spelling of `date_add`.
dateadd = date_add


#: The day names the reference engine's `next_day` accepts, mapped to DuckDB's `dayofweek`.
_DAY_OF_WEEK: dict[_str, int] = {
    "SU": 0,
    "SUN": 0,
    "SUNDAY": 0,
    "MO": 1,
    "MON": 1,
    "MONDAY": 1,
    "TU": 2,
    "TUE": 2,
    "TUESDAY": 2,
    "WE": 3,
    "WED": 3,
    "WEDNESDAY": 3,
    "TH": 4,
    "THU": 4,
    "THURSDAY": 4,
    "FR": 5,
    "FRI": 5,
    "FRIDAY": 5,
    "SA": 6,
    "SAT": 6,
    "SATURDAY": 6,
}


def next_day(date: Any, dayOfWeek: _str) -> Column:
    """The first date **after** `date` that falls on `dayOfWeek`.

    Strictly after: `next_day` of a Monday asking for Monday is seven days on, not the
    same day, which is why the modulo below maps a zero delta to 7.

    `dayOfWeek` must be a literal name -- `"Sun"`, `"SUNDAY"`, `"Su"` -- because the
    name is resolved to a number here rather than in SQL.
    """
    if not isinstance(dayOfWeek, _str):
        raise EngineTypeError(
            f"next_day() expects a literal day name such as 'Sun', got {type(dayOfWeek).__name__}."
        )
    target = _DAY_OF_WEEK.get(dayOfWeek.strip().upper())
    if target is None:
        raise EngineValueError(
            f"next_day() does not recognise the day name {dayOfWeek!r}. "
            f"Use one of Mon, Tue, Wed, Thu, Fri, Sat, Sun."
        )
    current = sg.Anonymous(this="dayofweek", expressions=[_col(date)])
    # (target - current + 7) % 7, then 0 -> 7 so that "next" is never "today".
    raw = sg.Mod(
        this=sg.Paren(
            this=sg.Add(
                this=sg.Paren(this=sg.Sub(this=to_literal(target), expression=current)),
                expression=to_literal(7),
            )
        ),
        expression=to_literal(7),
    )
    delta = sg.Case(
        ifs=[sg.If(this=sg.EQ(this=raw, expression=to_literal(0)), true=to_literal(7))],
        default=raw.copy(),
    )
    return _as_date(_fn("date_add", _col(date), _days(delta)))


def unix_date(col: Any) -> Column:
    """Whole days since 1970-01-01."""
    return _fn("date_diff", to_literal("day"), sg.cast(to_literal("1970-01-01"), "DATE"), _col(col))


def date_from_unix_date(col: Any) -> Column:
    """The inverse of `unix_date`: a day count back to a DATE."""
    return _as_date(_fn("date_add", sg.cast(to_literal("1970-01-01"), "DATE"), _days(_col(col))))


def unix_seconds(col: Any) -> Column:
    """Whole seconds since the epoch."""
    return Column(sg.cast(sg.Anonymous(this="epoch", expressions=[_col(col)]), "BIGINT"))


def unix_millis(col: Any) -> Column:
    """Whole milliseconds since the epoch."""
    return _fn("epoch_ms", _col(col))


def unix_micros(col: Any) -> Column:
    """Whole microseconds since the epoch."""
    return _fn("epoch_us", _col(col))


def timestamp_millis(col: Any) -> Column:
    """A millisecond epoch count as a timestamp -- the inverse of `unix_millis`."""
    return _fn("epoch_ms", _col(col))


def timestamp_micros(col: Any) -> Column:
    """A microsecond epoch count as a timestamp -- the inverse of `unix_micros`.

    DuckDB spells this `make_timestamp` when given a single integer, which is a
    different function from the six-argument `make_timestamp` below.
    """
    return _fn("make_timestamp", _col(col))


def make_timestamp(years: Any, months: Any, days: Any, hours: Any, mins: Any, secs: Any) -> Column:
    """A timestamp from its parts. `secs` may carry a fraction."""
    return _fn(
        "make_timestamp",
        _col(years),
        _col(months),
        _col(days),
        _col(hours),
        _col(mins),
        sg.cast(_col(secs), "DOUBLE"),
    )


#: The reference engine's other spelling of `unix_timestamp`.
to_unix_timestamp = unix_timestamp


#: Fields whose numbering the reference engine and DuckDB agree on. The day-of-week family is
#: deliberately absent: the reference engine numbers Sunday 1 and DuckDB numbers it 0, so passing
#: the field through would be off by one with nothing to show for it.
_DATE_PART_FIELDS = frozenset(
    {
        "YEAR",
        "Y",
        "YEARS",
        "YR",
        "YRS",
        "QUARTER",
        "QTR",
        "MONTH",
        "MON",
        "MONS",
        "MONTHS",
        "WEEK",
        "W",
        "WEEKS",
        "DAY",
        "D",
        "DAYS",
        "DAYOFYEAR",
        "DOY",
        "HOUR",
        "H",
        "HOURS",
        "HR",
        "HRS",
        "MINUTE",
        "M",
        "MIN",
        "MINS",
        "MINUTES",
        "SECOND",
        "S",
        "SEC",
        "SECONDS",
        "SECS",
    }
)


def date_part(field: _str, source: Any) -> Column:
    """One field of a date or timestamp, by name.

    The day-of-week fields are **refused** rather than translated: the reference engine numbers
    Sunday 1, DuckDB numbers it 0, and a silently-off-by-one weekday is precisely the
    kind of plausible wrong answer this library exists to avoid. Use `dayofweek` or
    `weekday`, whose numbering is explicit in the name.
    """
    if not isinstance(field, _str):
        raise EngineTypeError(
            f"date_part() expects a literal field name, got {type(field).__name__}."
        )
    normalized = field.strip().upper()
    if normalized not in _DATE_PART_FIELDS:
        raise EngineValueError(
            f"date_part() does not support the field {field!r}. "
            f"For the day of week use F.dayofweek() (Sunday is 1) or F.weekday() "
            f"(Monday is 0), whose numbering DuckDB's own does not match."
        )
    return _fn("date_part", to_literal(normalized.lower()), _col(source))


#: The reference engine's other spellings of `date_part`.
datepart = date_part
extract = date_part


def try_to_timestamp(col: Any, format: Any = None) -> Column:
    """Parse a timestamp, giving **NULL** rather than an error when it will not parse.

    Without a `format` this is a lenient cast. With one, the pattern is DuckDB's
    strftime syntax, not the reference engine's Java one -- the same caveat `date_format` carries.
    """
    if format is None:
        return Column(sg.TryCast(this=_col(col), to=sg.DataType.build("TIMESTAMP")))
    return _fn("try_strptime", _col(col), _col(format))


# ---------------------------------------------------------------------------
# Hashing -- second tranche
# ---------------------------------------------------------------------------


def sha(col: Any) -> Column:
    """The reference engine's other name for `sha1`."""
    return sha1(col)


def xxhash64(*cols: Any) -> Column:
    """A 64-bit hash. **Not the reference engine's xxHash64** -- see `hash` and divergence.md.

    The reference engine implements a specific xxHash64 seeded with 42; DuckDB's `hash` is its own
    algorithm. The values are stable within one query, which is what bucketing and
    sampling need, and they will not agree with a value another the reference engine job wrote down.
    Cast to `BIGINT` for the same reason `hash` is: DuckDB returns `UBIGINT`, which
    The reference engine has no type for.
    """
    if not cols:
        raise EngineValueError("xxhash64() needs at least one column.")
    return Column(sg.cast(_fn("hash", *[_col(c) for c in cols])._copy(), "BIGINT"))


# ---------------------------------------------------------------------------
# Misc and session
# ---------------------------------------------------------------------------


def current_user() -> Column:
    """The user the engine is running as."""
    return _fn("current_user")


#: The reference engine's other spellings of `current_user`.
user = current_user
session_user = current_user


def current_catalog() -> Column:
    return _fn("current_catalog")


def current_schema() -> Column:
    return _fn("current_schema")


#: The reference engine calls the current schema the current database.
current_database = current_schema


def equal_null(col1: Any, col2: Any) -> Column:
    """Null-safe equality -- the function form of `<=>`.

    Two NULLs are equal and a NULL against a value is false, where `=` would give
    NULL for both.
    """
    return Column(sg.NullSafeEQ(this=_col(col1), expression=_col(col2)))


def raise_error(errMsg: Any) -> Column:
    """Fail the query with `errMsg`."""
    return _fn("error", _col(errMsg))


def assert_true(col: Any, errMsg: Any = None) -> Column:
    """NULL when `col` is true, and a failed query when it is not.

    The reference engine returns NULL rather than true on success, which reads oddly until you
    remember it is a statement, not a predicate.
    """
    message = to_literal("assertion failed") if errMsg is None else _col(errMsg)
    return Column(
        sg.Case(
            ifs=[sg.If(this=_col(col), true=sg.null())],
            default=sg.Anonymous(this="error", expressions=[message]),
        )
    )


# ---------------------------------------------------------------------------
# Aggregates -- third tranche
# ---------------------------------------------------------------------------


def every(col: Any) -> Column:
    """True when every non-NULL row is true -- the reference engine's other name for `bool_and`."""
    return bool_and(col)


def some(col: Any) -> Column:
    """True when any non-NULL row is true -- the reference engine's other name for `bool_or`."""
    return bool_or(col)


#: The reference engine's third spelling of `bool_or`. `any_value` is a different function.
any = some


def bit_and(col: Any) -> Column:
    """Bitwise AND across the group."""
    return _fn("bit_and", _col(col))


def bit_or(col: Any) -> Column:
    """Bitwise OR across the group."""
    return _fn("bit_or", _col(col))


def bit_xor(col: Any) -> Column:
    """Bitwise XOR across the group."""
    return _fn("bit_xor", _col(col))


def std(col: Any) -> Column:
    """The reference engine's other name for `stddev`."""
    return stddev(col)


def percentile(col: Any, percentage: float) -> Column:
    """The **exact** percentile, interpolating between rows.

    `percentile_approx` is the sketch-based one; this reads every row. DuckDB spells
    the exact form `quantile_cont`.
    """
    if not isinstance(percentage, (int, float)) or not 0.0 <= percentage <= 1.0:
        raise EngineValueError("percentile() needs a percentage between 0 and 1.")
    return _fn("quantile_cont", _col(col), to_literal(float(percentage)))


def array_agg(col: Any) -> Column:
    """The reference engine's other name for `collect_list`."""
    return collect_list(col)


def _regression(name: _str, y: Any, x: Any) -> Column:
    return _fn(name, _col(y), _col(x))


def regr_count(y: Any, x: Any) -> Column:
    """Rows where neither `y` nor `x` is NULL."""
    return _regression("regr_count", y, x)


def regr_avgx(y: Any, x: Any) -> Column:
    return _regression("regr_avgx", y, x)


def regr_avgy(y: Any, x: Any) -> Column:
    return _regression("regr_avgy", y, x)


def regr_intercept(y: Any, x: Any) -> Column:
    return _regression("regr_intercept", y, x)


def regr_r2(y: Any, x: Any) -> Column:
    return _regression("regr_r2", y, x)


def regr_slope(y: Any, x: Any) -> Column:
    return _regression("regr_slope", y, x)


def regr_sxx(y: Any, x: Any) -> Column:
    return _regression("regr_sxx", y, x)


def regr_sxy(y: Any, x: Any) -> Column:
    return _regression("regr_sxy", y, x)


def regr_syy(y: Any, x: Any) -> Column:
    return _regression("regr_syy", y, x)


# ---------------------------------------------------------------------------
# Arrays -- second tranche
# ---------------------------------------------------------------------------


def array_append(col: Any, value: Any) -> Column:
    return _fn("list_append", _col(col), _value(value))


def array_prepend(col: Any, value: Any) -> Column:
    return _fn("list_prepend", _value(value), _col(col))


def array_compact(col: Any) -> Column:
    """The array without its NULLs."""
    return _fn(
        "list_filter",
        _col(col),
        sg.Lambda(
            this=sg.Not(this=sg.Is(this=sg.column("x"), expression=sg.null())),
            expressions=[sg.to_identifier("x")],
        ),
    )


def array_except(col1: Any, col2: Any) -> Column:
    """The elements of `col1` that are not in `col2`, **de-duplicated** as the reference is."""
    kept = sg.Anonymous(
        this="list_filter",
        expressions=[
            _col(col1),
            sg.Lambda(
                this=sg.Not(
                    this=sg.Anonymous(
                        this="list_contains", expressions=[_col(col2), sg.column("x")]
                    )
                ),
                expressions=[sg.to_identifier("x")],
            ),
        ],
    )
    return _fn("list_distinct", kept)


def array_repeat(col: Any, count: Any) -> Column:
    """An array holding `col` `count` times."""
    return _fn(
        "repeat",
        sg.Anonymous(this="list_value", expressions=[_col(col)]),
        _col(count),
    )


def array_size(col: Any) -> Column:
    """The number of elements, **NULL for a NULL array**.

    The reference engine's `size` answers -1 there by default; `array_size` answers NULL. The two
    exist to differ, so neither can be an alias of the other.
    """
    return _fn("len", _col(col))


#: The reference engine's other name for `size`.
cardinality = size


def get(col: Any, index: Any) -> Column:
    """The element at `index`, **0-indexed**, NULL when out of range.

    The reference engine's `get` counts from 0 where `element_at` counts from 1. Both are the
    reference engine's,
    and the off-by-one between them is the reference engine's too -- the `+ 1` here is what keeps
    `get` faithful to its own convention on top of DuckDB's 1-indexed lists.
    """
    return _fn(
        "list_extract",
        _col(col),
        sg.Paren(this=sg.Add(this=_col(index), expression=to_literal(1))),
    )

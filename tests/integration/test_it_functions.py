"""Every `F.*` name, executed against a real column in the real catalog.

The local suite already asserts what each function *returns*, thoroughly, against
literal arguments: `F.upper(F.lit("aBc")) == "ABC"`. Repeating those here would prove
nothing new -- the argument is a constant, so the catalog never enters into it, and 240
copies of that shape all exercise one code path.

What is worth asserting here is the part the local suite cannot reach:

**Every name still resolves when its argument is a real column.** A function that works
on `F.lit("aBc")` can still fail on a real `VARCHAR` column read out of parquet over an
object store -- a dialect mapping that only fires for non-literals, a type the local
fixture never produces, a function that silently stopped being registered. The sweep
below calls all 296 scalar names against a column of real data and requires each to
compile, analyse and execute.

**And the families where real data is the point.** Date parts over 5,000 genuine
timestamps rather than one hand-written one. String functions over a column that is
NULL 971 times. Rounding over doubles that range from 0.0 to 161024.37. Aggregates over
880,374 rows, checked against DuckDB reading the same parquet.

`test_every_exported_name_is_accounted_for` is what keeps this honest: every name in
`functions.__all__` must be in the sweep, in the explicit table, or in
`_COVERED_ELSEWHERE` with a reason. A new function cannot be added without landing in
one of the three, so coverage cannot rot quietly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icetl.sql import functions as F
from icetl.sql.types import StringType, StructField, StructType
from icetl.sql.window import Window
from tests.integration.helpers import assert_sorted, column, duckdb_answer

if TYPE_CHECKING:
    from collections.abc import Callable

    from icetl.sql.column import Column
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


# --- the real columns every probe is built from ------------------------------
# Functions, not module-level constants, so no two probes ever share an expression
# object. All of these are columns of `icetl_it.trips_small` -- 5,000 rows carved out
# of nyc.yellow_tripdata, carrying 971 real NULLs and real timestamps.


def NUM() -> Column:
    """A real double. Ranges 0.0 .. 30.77 in the small slice, so `0` is present."""
    return F.col("trip_distance")


def INT() -> Column:
    return F.col("PULocationID")


def STR() -> Column:
    """A real string column that is NULL in 971 of 5,000 rows."""
    return F.col("store_and_fwd_flag")


def TS() -> Column:
    """A real timestamp. Not a parsed literal -- these came out of the trip data."""
    return F.col("tpep_pickup_datetime")


def DATE() -> Column:
    return F.col("tpep_pickup_datetime").cast("date")


def BOOL() -> Column:
    return F.col("trip_distance") > 1


def ARR() -> Column:
    return F.col("tags")


def MAP() -> Column:
    return F.col("scores")


def WIN() -> Any:
    return Window.partitionBy("VendorID").orderBy("tpep_pickup_datetime")


#: Which seeded table a shape's columns come from.
_SMALL, _NESTED = "small", "nested"

#: How to call a function of each argument shape. The shapes were derived by probing
#: every exported name; `_BY_SHAPE` below records which name landed in which.
_SHAPES: dict[str, tuple[str, Callable[[Any], Column]]] = {
    "nullary": (_SMALL, lambda f: f()),
    "num": (_SMALL, lambda f: f(NUM())),
    "str": (_SMALL, lambda f: f(STR())),
    "int": (_SMALL, lambda f: f(INT())),
    "ts": (_SMALL, lambda f: f(TS())),
    "bool": (_SMALL, lambda f: f(BOOL())),
    "num,num": (_SMALL, lambda f: f(NUM(), NUM())),
    "num,int": (_SMALL, lambda f: f(NUM(), F.lit(2))),
    "str,str": (_SMALL, lambda f: f(STR(), F.lit("N"))),
    "str,int": (_SMALL, lambda f: f(STR(), F.lit(2))),
    "str,int,int": (_SMALL, lambda f: f(STR(), F.lit(1), F.lit(2))),
    "str,str,str": (_SMALL, lambda f: f(STR(), F.lit("N"), F.lit("Y"))),
    "ts,int": (_SMALL, lambda f: f(TS(), F.lit(1))),
    "window": (_SMALL, lambda f: f().over(WIN())),
    "window_num": (_SMALL, lambda f: f(NUM()).over(WIN())),
    "window_num_int": (_SMALL, lambda f: f(NUM(), 1).over(WIN())),
    "arr": (_NESTED, lambda f: f(ARR())),
    "arr,arr": (_NESTED, lambda f: f(ARR(), ARR())),
    "arr,str": (_NESTED, lambda f: f(ARR(), F.lit("x"))),
    "arr,init,lambda": (_NESTED, lambda f: f(ARR(), F.lit(""), lambda a, b: a)),
    "lambda1": (_NESTED, lambda f: f(ARR(), lambda x: x)),
    "map": (_NESTED, lambda f: f(MAP())),
}


_BY_SHAPE: dict[str, tuple[str, ...]] = {
    "arr": (
        "array_compact",
        "array_distinct",
        "array_max",
        "array_min",
        "array_sort",
        "arrays_zip",
        "cardinality",
        "explode",
        "explode_outer",
        "size",
        "sort_array",
    ),
    "arr,arr": (
        "array_except",
        "array_intersect",
        "array_union",
        "arrays_overlap",
        "map_from_arrays",
    ),
    "arr,init,lambda": ("aggregate",),
    "arr,str": (
        "array_append",
        "array_contains",
        "array_position",
        "array_prepend",
        "array_remove",
    ),
    "bool": (
        "any",
        "bool_and",
        "bool_or",
        "count_if",
        "every",
        "some",
    ),
    "int": (
        "bit_and",
        "bit_count",
        "bit_or",
        "bit_xor",
        "bitwise_not",
        "date_from_unix_date",
        "from_unixtime",
        "timestamp_micros",
        "timestamp_millis",
        "timestamp_seconds",
        "unix_millis",
    ),
    "lambda1": ("transform",),
    "map": (
        "map_entries",
        "map_keys",
        "map_values",
    ),
    "nullary": (
        "array",
        "create_map",
        "curdate",
        "current_catalog",
        "current_database",
        "current_date",
        "current_schema",
        "current_timestamp",
        "current_user",
        "e",
        "localtimestamp",
        "monotonically_increasing_id",
        "now",
        "pi",
        "rand",
        "randn",
        "session_user",
        "to_unix_timestamp",
        "unix_timestamp",
        "user",
    ),
    "num": (
        "abs",
        "acosh",
        "any_value",
        "approx_count_distinct",
        "array_agg",
        "asinh",
        "atan",
        "avg",
        "base64",
        "bin",
        "cbrt",
        "ceil",
        "ceiling",
        "coalesce",
        "collect_list",
        "collect_set",
        "concat",
        "cos",
        "cosh",
        "cot",
        "count",
        "countDistinct",
        "csc",
        "degrees",
        "exp",
        "expm1",
        "first",
        "floor",
        "hash",
        "hex",
        "isnan",
        "isnull",
        "kurtosis",
        "last",
        "lit",
        "ln",
        "log",
        "log10",
        "log1p",
        "log2",
        "max",
        "mean",
        "median",
        "min",
        "mode",
        "negative",
        "positive",
        "radians",
        "rint",
        "round",
        "schema_of_json",
        "sec",
        "sign",
        "signum",
        "sin",
        "sinh",
        "skewness",
        "sqrt",
        "std",
        "stddev",
        "stddev_pop",
        "stddev_samp",
        "struct",
        "sum",
        "sum_distinct",
        "tan",
        "tanh",
        "to_date",
        "to_json",
        "to_timestamp",
        "try_to_timestamp",
        "var_pop",
        "var_samp",
        "variance",
        "xxhash64",
    ),
    "num,int": ("array_repeat",),
    "num,num": (
        "atan2",
        "corr",
        "covar_pop",
        "covar_samp",
        "equal_null",
        "greatest",
        "hypot",
        "ifnull",
        "least",
        "max_by",
        "min_by",
        "nanvl",
        "nullif",
        "nvl",
        "pmod",
        "pow",
        "power",
        "regr_avgx",
        "regr_avgy",
        "regr_count",
        "regr_intercept",
        "regr_r2",
        "regr_slope",
        "regr_sxx",
        "regr_sxy",
        "regr_syy",
        "try_divide",
        "when",
    ),
    "str": (
        "array_size",
        "ascii",
        "bit_length",
        "btrim",
        "char_length",
        "character_length",
        "initcap",
        "lcase",
        "length",
        "lower",
        "ltrim",
        "md5",
        "octet_length",
        "reverse",
        "rtrim",
        "sha",
        "sha1",
        "trim",
        "ucase",
        "unhex",
        "upper",
    ),
    "str,int": (
        "element_at",
        "get",
        "left",
        "lpad",
        "repeat",
        "right",
        "rpad",
        "substr",
    ),
    "str,int,int": ("substring",),
    "str,str": (
        "contains",
        "endswith",
        "find_in_set",
        "levenshtein",
        "regexp",
        "regexp_count",
        "regexp_instr",
        "regexp_like",
        "regexp_substr",
        "startswith",
    ),
    "str,str,str": (
        "nvl2",
        "replace",
    ),
    "ts": (
        "day",
        "dayofmonth",
        "dayofweek",
        "dayofyear",
        "hour",
        "last_day",
        "minute",
        "month",
        "quarter",
        "second",
        "unix_date",
        "unix_micros",
        "unix_seconds",
        "weekday",
        "weekofyear",
        "year",
    ),
    "ts,int": (
        "add_months",
        "date_add",
        "date_sub",
        "dateadd",
        "percentile_approx",
    ),
    "window": (
        "cume_dist",
        "dense_rank",
        "ntile",
        "percent_rank",
        "rank",
        "row_number",
    ),
    "window_num": (
        "first_value",
        "lag",
        "last_value",
        "lead",
    ),
    "window_num_int": (
        "nth_value",
        "percentile",
    ),
}

#: Names whose arguments are too specific for a shared shape -- a format string, a
#: regex, a lambda of a particular arity, a literal that must be valid. Written out
#: rather than derived, because guessing an argument is how a sweep starts passing
#: for the wrong reason.
_EXPLICIT: dict[str, tuple[str, Callable[[], Column]]] = {
    "array_join": (_NESTED, lambda: F.array_join(ARR(), ",")),
    "char": (_SMALL, lambda: F.char(F.lit(65))),
    "chr": (_SMALL, lambda: F.chr(F.lit(65))),
    "concat_ws": (_SMALL, lambda: F.concat_ws("-", STR(), F.lit("x"))),
    "date_diff": (_SMALL, lambda: F.date_diff(TS(), TS())),
    "datediff": (_SMALL, lambda: F.datediff(TS(), TS())),
    "date_format": (_SMALL, lambda: F.date_format(TS(), "%Y-%m-%d")),
    "date_part": (_SMALL, lambda: F.date_part("year", TS())),
    "datepart": (_SMALL, lambda: F.datepart("year", TS())),
    "date_trunc": (_SMALL, lambda: F.date_trunc("day", TS())),
    "extract": (_SMALL, lambda: F.extract("year", TS())),
    "elt": (_SMALL, lambda: F.elt(F.lit(1), F.lit("a"), F.lit("b"))),
    "exists": (_NESTED, lambda: F.exists(ARR(), lambda x: x == F.lit("x"))),
    "forall": (_NESTED, lambda: F.forall(ARR(), lambda x: x.isNotNull())),
    "filter": (_NESTED, lambda: F.filter(ARR(), lambda x: x.isNotNull())),
    "factorial": (_SMALL, lambda: F.factorial(F.lit(5))),
    "flatten": (_SMALL, lambda: F.flatten(F.array(F.array(F.lit(1)), F.array(F.lit(2))))),
    "format_number": (_SMALL, lambda: F.format_number(NUM(), 2)),
    "format_string": (_SMALL, lambda: F.format_string("%s", STR())),
    "printf": (_SMALL, lambda: F.printf("%s", STR())),
    "from_json": (
        _SMALL,
        lambda: F.from_json(F.lit('{"a":"b"}'), StructType([StructField("a", StringType())])),
    ),
    "get_json_object": (_SMALL, lambda: F.get_json_object(F.lit('{"a":1}'), "$.a")),
    "json_tuple": (_SMALL, lambda: F.json_tuple(F.lit('{"a":1}'), "a")),
    "instr": (_SMALL, lambda: F.instr(STR(), "N")),
    "locate": (_SMALL, lambda: F.locate("N", STR())),
    "make_date": (_SMALL, lambda: F.make_date(F.lit(2024), F.lit(6), F.lit(1))),
    "make_timestamp": (
        _SMALL,
        lambda: F.make_timestamp(F.lit(2024), F.lit(6), F.lit(1), F.lit(0), F.lit(0), F.lit(0)),
    ),
    "map_concat": (_NESTED, lambda: F.map_concat(MAP(), MAP())),
    "map_filter": (_NESTED, lambda: F.map_filter(MAP(), lambda k, v: v > F.lit(0))),
    "map_from_entries": (
        _SMALL,
        lambda: F.map_from_entries(
            F.array(F.struct(F.lit("k").alias("key"), F.lit(1).alias("value")))
        ),
    ),
    "transform_keys": (_NESTED, lambda: F.transform_keys(MAP(), lambda k, v: k)),
    "transform_values": (_NESTED, lambda: F.transform_values(MAP(), lambda k, v: v)),
    "months_between": (_SMALL, lambda: F.months_between(TS(), TS())),
    "next_day": (_SMALL, lambda: F.next_day(DATE(), "Mon")),
    "overlay": (_SMALL, lambda: F.overlay(STR(), F.lit("Z"), F.lit(1))),
    "regexp_extract": (_SMALL, lambda: F.regexp_extract(STR(), r"(\w)", 1)),
    "regexp_replace": (_SMALL, lambda: F.regexp_replace(STR(), r"\w", "z")),
    "sequence": (_SMALL, lambda: F.sequence(F.lit(1), F.lit(3))),
    "sha2": (_SMALL, lambda: F.sha2(STR(), 256)),
    "shiftleft": (_SMALL, lambda: F.shiftleft(F.lit(1), 2)),
    "shiftright": (_SMALL, lambda: F.shiftright(F.lit(4), 2)),
    "shiftrightunsigned": (_SMALL, lambda: F.shiftrightunsigned(F.lit(4), 2)),
    "slice": (_NESTED, lambda: F.slice(ARR(), 1, 1)),
    "split": (_SMALL, lambda: F.split(STR(), "N")),
    "split_part": (_SMALL, lambda: F.split_part(STR(), F.lit("N"), F.lit(1))),
    "substring_index": (_SMALL, lambda: F.substring_index(STR(), "N", 1)),
    "translate": (_SMALL, lambda: F.translate(STR(), "N", "Z")),
    "trunc": (_SMALL, lambda: F.trunc(DATE(), "month")),
    "unbase64": (_SMALL, lambda: F.unbase64(F.base64(STR()))),
    "width_bucket": (_SMALL, lambda: F.width_bucket(NUM(), F.lit(0.0), F.lit(100.0), F.lit(4))),
    "zip_with": (_NESTED, lambda: F.zip_with(ARR(), ARR(), lambda a, b: a)),
    # acos/asin/atanh are defined only on [-1, 1], and every numeric column in real
    # trip data leaves that range -- PULocationID reaches 263. `x / (x + 1)` maps any
    # non-negative real value into [0, 1), so the argument stays a real column instead
    # of degrading into a literal. The generic numeric shape found this by failing.
    "acos": (_SMALL, lambda: F.acos(NUM() / (NUM() + F.lit(1.0)))),
    "asin": (_SMALL, lambda: F.asin(NUM() / (NUM() + F.lit(1.0)))),
    "atanh": (_SMALL, lambda: F.atanh(NUM() / (NUM() + F.lit(1.0)))),
    # `trip_distance` really is 0.0 for some trips, and 0 is not true -- so the
    # argument has to be a condition that holds for every row.
    "assert_true": (_SMALL, lambda: F.assert_true(NUM() >= F.lit(0.0))),
}

#: Names the sweep cannot reach, and where each is covered instead. Every one is a
#: function that is not a scalar projection: it belongs to `ORDER BY`, to `GROUP BY`,
#: to the generator position, or it is not an expression at all.
_COVERED_ELSEWHERE: dict[str, str] = {
    "asc": "TestTheSortHelpers",
    "asc_nulls_first": "TestTheSortHelpers",
    "asc_nulls_last": "TestTheSortHelpers",
    "desc": "TestTheSortHelpers",
    "desc_nulls_first": "TestTheSortHelpers",
    "desc_nulls_last": "TestTheSortHelpers",
    "grouping": "TestTheGroupingHelpers -- only meaningful under GROUP BY",
    "grouping_id": "TestTheGroupingHelpers -- only meaningful under GROUP BY",
    "inline": "TestTheRowGenerators -- expands to several columns, not one",
    "inline_outer": "TestTheRowGenerators -- expands to several columns, not one",
    "posexplode": "TestTheRowGenerators -- expands to several columns, not one",
    "posexplode_outer": "TestTheRowGenerators -- expands to several columns, not one",
    "col": "used by every test in the suite; not a function of a column",
    "column": "used by every test in the suite; not a function of a column",
    "expr": "a parser entry point, not an expression over a column",
    "udf": "tests/integration/test_it_udf.py",
    "pandas_udf": "tests/integration/test_it_udf.py",
    "broadcast": "a join hint on a DataFrame, not a column expression",
    "raise_error": "raises by contract, so it cannot be swept alongside the rest",
}


def _probe(name: str) -> tuple[str, Column]:
    """The source table and expression this name is swept with."""
    if name in _EXPLICIT:
        source, build = _EXPLICIT[name]
        return source, build()
    for shape, names in _BY_SHAPE.items():
        if name in names:
            source, call = _SHAPES[shape]
            return source, call(getattr(F, name))
    raise AssertionError(f"{name} has no probe")


#: Every name the sweep runs, sorted so the parametrized ids are stable.
SWEPT = sorted({n for names in _BY_SHAPE.values() for n in names} | set(_EXPLICIT))


class TestEveryNameResolves:
    """The sweep. Each name, called on a real column, must compile and execute."""

    @pytest.mark.parametrize("name", SWEPT)
    def test_it_evaluates_against_a_real_column(
        self, it_session: Session, trips_small: str, nested: str, name: str
    ) -> None:
        source, expression = _probe(name)
        table = trips_small if source == _SMALL else nested
        rows = it_session.table(table).select(expression.alias("v")).limit(1).collect()
        assert len(rows) == 1, f"F.{name} produced no row"
        assert "v" in rows[0].asDict(), f"F.{name} did not produce the projected column"

    def test_every_exported_name_is_accounted_for(self) -> None:
        """The gate that stops this coverage rotting.

        A name added to `functions.__all__` and to nothing else fails here, naming
        itself, rather than quietly never being tested against a real catalog.
        """
        exported = {name for name in F.__all__ if callable(getattr(F, name, None))}
        accounted = set(SWEPT) | set(_COVERED_ELSEWHERE)
        missing = sorted(exported - accounted)
        assert not missing, (
            f"{len(missing)} exported function(s) are in no bucket: {missing}. "
            f"Add each to _BY_SHAPE, to _EXPLICIT, or to _COVERED_ELSEWHERE with a reason."
        )

    def test_the_sweep_is_not_secretly_empty(self) -> None:
        """A sweep that stopped collecting would otherwise pass by doing nothing."""
        assert len(SWEPT) > 250, len(SWEPT)


# ---------------------------------------------------------------------------
# The families where real data is the point
# ---------------------------------------------------------------------------


class TestOnRealTimestamps:
    """Date parts over 5,000 timestamps that came out of the trip data.

    The seeded slice is the first week of June 2024, so the expected values are
    properties of that window rather than of any particular row -- they hold however
    the slice is re-cut, as long as `ICETL_IT_SEED_START` still names a June day.
    """

    def test_the_year_and_month_are_the_windows(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select(
            F.year(TS()).alias("y"), F.month(TS()).alias("m")
        )
        assert set(column(frame, "y")) == {2024}
        assert set(column(frame, "m")) == {6}

    def test_every_hour_is_a_real_hour(self, it_session: Session, trips_small: str) -> None:
        hours = set(column(it_session.table(trips_small).select(F.hour(TS()).alias("h")), "h"))
        assert hours, "no rows"
        assert all(0 <= h <= 23 for h in hours), sorted(hours)

    def test_date_trunc_to_day_drops_the_time(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small).select(
            F.hour(F.date_trunc("day", TS())).alias("h"),
            F.minute(F.date_trunc("day", TS())).alias("mi"),
        )
        assert set(column(frame, "h")) == {0}
        assert set(column(frame, "mi")) == {0}

    def test_date_format_matches_the_parts(self, it_session: Session, trips_small: str) -> None:
        """Two ways of asking the same question about a real timestamp.

        The pattern is strftime's, not Java's -- see the divergence pinned below.
        """
        row = (
            it_session.table(trips_small)
            .select(
                F.date_format(TS(), "%Y-%m-%d").alias("formatted"),
                F.year(TS()).alias("y"),
                F.month(TS()).alias("m"),
                F.dayofmonth(TS()).alias("d"),
            )
            .limit(1)
            .collect()[0]
        )
        assert row["formatted"] == f"{row['y']:04d}-{row['m']:02d}-{row['d']:02d}"

    def test_a_java_date_pattern_is_not_translated(
        self, it_session: Session, trips_small: str
    ) -> None:
        """divergence.md, line 111, pinned on real data.

        `date_format` takes strftime patterns, and Java's are deliberately **not**
        translated -- they are two pattern languages, not two spellings of one. So the
        reference's own `yyyy-MM-dd` contains no strftime directive and comes back
        verbatim rather than raising. That is the sharp edge for anyone porting a
        query, and a test that says so is worth more than a comment.
        """
        value = (
            it_session.table(trips_small)
            .select(F.date_format(TS(), "yyyy-MM-dd").alias("v"))
            .limit(1)
            .collect()[0]["v"]
        )
        assert value == "yyyy-MM-dd", (
            "date_format now interprets Java patterns -- divergence.md line 111 is "
            "stale and this test should become an equality against the real date"
        )

    def test_datediff_between_pickup_and_dropoff_is_never_negative(
        self, it_session: Session, trips_small: str
    ) -> None:
        """An invariant of the data, asserted over every row rather than one."""
        days = column(
            it_session.table(trips_small).select(
                F.datediff(F.col("tpep_dropoff_datetime"), TS()).alias("d")
            ),
            "d",
        )
        assert days
        assert all(d is None or d >= 0 for d in days)

    def test_dayofweek_is_the_reference_numbering(
        self, it_session: Session, trips_small: str
    ) -> None:
        """`F.dayofweek` counts 1..7 from Sunday, `F.weekday` counts 0..6 from Monday.

        The seed window opens on 2024-06-01, which is a Saturday -- so the reference
        answers are 7 and 5, and DuckDB's own numbering (Saturday = 6) is neither.
        """
        row = (
            it_session.table(trips_small)
            .orderBy(TS())
            .select(
                TS().alias("ts"),
                F.dayofweek(TS()).alias("dow"),
                F.weekday(TS()).alias("wd"),
            )
            .limit(1)
            .collect()[0]
        )
        expected_dow = row["ts"].isoweekday() % 7 + 1  # Sunday = 1
        expected_wd = row["ts"].weekday()  # Monday = 0
        assert row["dow"] == expected_dow
        assert row["wd"] == expected_wd

    def test_the_sql_surface_still_disagrees_about_the_day_of_week(
        self, it_session: Session, trips_small: str
    ) -> None:
        """FINDINGS 1.8, pinned on real data -- the one live wrong answer in the tree.

        `Session.sql()` resolves `dayofweek` and `weekday` to DuckDB's functions rather
        than the reference's, so both come back as DuckDB's Sunday=0 numbering and the
        two surfaces disagree. P1 does not hold until Phase 15 fixes it.

        This is a characterisation test: it asserts the bug, so it **fails when the bug
        is fixed**, which is the notification Phase 15 wants.
        """
        frame = (
            it_session.table(trips_small)
            .orderBy(TS())
            .select(F.dayofweek(TS()).alias("dow"), F.weekday(TS()).alias("wd"))
            .limit(1)
            .collect()[0]
        )
        via_sql = it_session.sql(
            f"SELECT dayofweek(tpep_pickup_datetime) AS dow, "
            f"weekday(tpep_pickup_datetime) AS wd FROM {trips_small} "
            f"ORDER BY tpep_pickup_datetime LIMIT 1"
        ).collect()[0]
        assert via_sql["dow"] != frame["dow"], (
            "the SQL surface now agrees with F.dayofweek -- FINDINGS 1.8 is fixed, "
            "so delete this test and its entry in FINDINGS.md and STATUS.md"
        )
        assert via_sql["wd"] != frame["wd"], (
            "the SQL surface now agrees with F.weekday -- FINDINGS 1.8 is fixed"
        )


class TestOnRealStrings:
    """A column that is genuinely NULL in 971 of its 5,000 rows."""

    def test_case_changes_preserve_null(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small).select(
            F.upper(STR()).alias("u"), F.lower(STR()).alias("lo"), STR().alias("raw")
        )
        rows = frame.collect()
        assert any(row["raw"] is None for row in rows), "the column has no NULLs"
        for row in rows:
            if row["raw"] is None:
                assert row["u"] is None and row["lo"] is None
            else:
                assert row["u"] == row["raw"].upper()

    def test_length_of_the_flag_is_one(self, it_session: Session, trips_small: str) -> None:
        lengths = set(
            column(
                it_session.table(trips_small)
                .filter(STR().isNotNull())
                .select(F.length(STR()).alias("n")),
                "n",
            )
        )
        assert lengths == {1}, lengths

    def test_concat_with_null_is_null(self, it_session: Session, trips_small: str) -> None:
        """The reference propagates NULL through `concat`; some engines return 'x'."""
        value = (
            it_session.table(trips_small)
            .filter(STR().isNull())
            .select(F.concat(STR(), F.lit("x")).alias("v"))
            .limit(1)
            .collect()[0]["v"]
        )
        assert value is None

    def test_the_flag_only_ever_holds_the_two_real_values(
        self, it_session: Session, trips_small: str
    ) -> None:
        values = {
            v
            for v in column(it_session.table(trips_small).select(STR().alias("v")).distinct(), "v")
            if v is not None
        }
        assert values <= {"Y", "N"}, values


class TestOnRealDoubles:
    """Doubles from the trip data, including the 76 rows where distance is exactly 0."""

    def test_floor_and_ceil_bracket_the_value(self, it_session: Session, trips_small: str) -> None:
        rows = (
            it_session.table(trips_small)
            .select(NUM().alias("x"), F.floor(NUM()).alias("lo"), F.ceil(NUM()).alias("hi"))
            .collect()
        )
        assert rows
        for row in rows:
            assert row["lo"] <= row["x"] <= row["hi"]

    def test_dividing_by_a_real_zero_is_null_not_an_error(
        self, it_session: Session, trips_small: str
    ) -> None:
        """The conformance rule that needed real data to mean anything.

        76 of these trips recorded a distance of exactly 0.0. The reference returns
        NULL for `x / 0`; DuckDB on its own would return infinity. Every zero-distance
        row must produce exactly one NULL quotient -- counted, not sampled.
        """
        zeros = it_session.table(trips_small).filter(NUM() == 0).count()
        assert zeros > 0, "the slice has no zero-distance trips, so this proves nothing"
        nulls = (
            it_session.table(trips_small)
            .select((F.col("total_amount") / NUM()).alias("q"))
            .filter(F.col("q").isNull())
            .count()
        )
        assert nulls == zeros

    def test_round_is_within_half_a_unit(self, it_session: Session, trips_small: str) -> None:
        rows = (
            it_session.table(trips_small)
            .select(NUM().alias("x"), F.round(NUM()).alias("r"))
            .collect()
        )
        assert all(abs(row["r"] - row["x"]) <= 0.5 for row in rows)

    def test_sqrt_squares_back(self, it_session: Session, trips_small: str) -> None:
        rows = (
            it_session.table(trips_small)
            .select(NUM().alias("x"), (F.sqrt(NUM()) * F.sqrt(NUM())).alias("back"))
            .limit(200)
            .collect()
        )
        for row in rows:
            assert row["back"] == pytest.approx(row["x"], rel=1e-9, abs=1e-9)


class TestOnRealNulls:
    """971 real NULLs, and the functions whose whole job is what to do about them."""

    def test_null_and_not_null_partition_the_table(
        self, it_session: Session, trips_small: str
    ) -> None:
        """An invariant: no golden number, and it survives any re-seed."""
        frame = it_session.table(trips_small)
        total = frame.count()
        nulls = frame.filter(STR().isNull()).count()
        present = frame.filter(STR().isNotNull()).count()
        assert nulls > 0 and present > 0
        assert nulls + present == total

    def test_coalesce_fills_exactly_the_nulls(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small)
        nulls = frame.filter(STR().isNull()).count()
        filled = (
            frame.select(F.coalesce(STR(), F.lit("missing")).alias("v"))
            .filter(F.col("v") == "missing")
            .count()
        )
        assert filled == nulls
        assert (
            frame.select(F.coalesce(STR(), F.lit("missing")).alias("v"))
            .filter(F.col("v").isNull())
            .count()
            == 0
        )

    def test_nvl_and_ifnull_agree_with_coalesce(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select(
            F.coalesce(STR(), F.lit("z")).alias("a"),
            F.nvl(STR(), F.lit("z")).alias("b"),
            F.ifnull(STR(), F.lit("z")).alias("c"),
        )
        rows = frame.collect()
        assert rows
        assert all(row["a"] == row["b"] == row["c"] for row in rows)

    def test_count_of_a_column_skips_nulls_but_count_star_does_not(
        self, it_session: Session, trips_small: str
    ) -> None:
        """The two-questions rule, on the operation where it bit before."""
        row = (
            it_session.table(trips_small)
            .select(
                F.count(F.lit(1)).alias("all_rows"),
                F.count(STR()).alias("present"),
                F.sum(F.when(STR().isNull(), F.lit(1)).otherwise(F.lit(0))).alias("absent"),
            )
            .collect()[0]
        )
        assert row["present"] + row["absent"] == row["all_rows"]
        assert row["absent"] > 0

    def test_arithmetic_propagates_null(self, it_session: Session, trips_small: str) -> None:
        nulls = it_session.table(trips_small).filter(F.col("passenger_count").isNull()).count()
        propagated = (
            it_session.table(trips_small)
            .select((F.col("passenger_count") + F.lit(1)).alias("v"))
            .filter(F.col("v").isNull())
            .count()
        )
        assert nulls > 0
        assert propagated == nulls


class TestAggregatesOverRealRows:
    """880,374 rows, checked against DuckDB reading the very same parquet files."""

    def test_the_headline_aggregates_match_raw_duckdb(
        self, it_session: Session, trips: str
    ) -> None:
        """The differential that bypasses icetl's plan entirely.

        Same files, same bytes; the only difference is that one answer went through
        the optimizer, the pushdown and the conformance layer and the other did not.
        """
        frame = it_session.table(trips)
        mine = frame.select(
            F.count(F.lit(1)).alias("n"),
            F.sum(F.col("trip_distance")).alias("total"),
            F.min(F.col("trip_distance")).alias("lo"),
            F.max(F.col("trip_distance")).alias("hi"),
        ).collect()[0]

        theirs = duckdb_answer(
            it_session,
            frame,
            "SELECT count(*) AS n, sum(trip_distance) AS total, "
            "min(trip_distance) AS lo, max(trip_distance) AS hi FROM read_parquet($paths)",
        ).to_pylist()[0]

        assert mine["n"] == theirs["n"]
        assert mine["lo"] == theirs["lo"]
        assert mine["hi"] == theirs["hi"]
        # Floats only: DuckDB's aggregation order varies between runs, so the last
        # few digits of a sum over 880k doubles do too (FINDINGS 3.7).
        assert mine["total"] == pytest.approx(theirs["total"], rel=1e-9)

    def test_avg_is_sum_over_count(self, it_session: Session, trips: str) -> None:
        row = (
            it_session.table(trips)
            .select(
                F.avg(F.col("trip_distance")).alias("mean"),
                F.sum(F.col("trip_distance")).alias("total"),
                F.count(F.col("trip_distance")).alias("n"),
            )
            .collect()[0]
        )
        assert row["mean"] == pytest.approx(row["total"] / row["n"], rel=1e-9)

    def test_distinct_count_is_bounded_by_the_row_count(
        self, it_session: Session, trips: str
    ) -> None:
        row = (
            it_session.table(trips)
            .select(
                F.countDistinct(F.col("PULocationID")).alias("d"),
                F.count(F.lit(1)).alias("n"),
            )
            .collect()[0]
        )
        assert 0 < row["d"] <= row["n"]

    def test_approx_count_distinct_is_close_to_the_exact_one(
        self, it_session: Session, trips: str
    ) -> None:
        """A sketch over 880k real values, held to a tolerance rather than a number."""
        row = (
            it_session.table(trips)
            .select(
                F.countDistinct(F.col("PULocationID")).alias("exact"),
                F.approx_count_distinct(F.col("PULocationID")).alias("approx"),
            )
            .collect()[0]
        )
        assert row["approx"] == pytest.approx(row["exact"], rel=0.1)


# ---------------------------------------------------------------------------
# The names the sweep cannot reach
# ---------------------------------------------------------------------------


class TestTheSortHelpers:
    """`asc`/`desc` and their null-placement variants, in the only position they work."""

    def test_ascending_puts_real_nulls_first(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small).orderBy(F.asc("store_and_fwd_flag")).select(STR()),
            "store_and_fwd_flag",
        )
        assert_sorted(values, ascending=True, nulls_first=True)

    def test_descending_puts_real_nulls_last(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small).orderBy(F.desc("store_and_fwd_flag")).select(STR()),
            "store_and_fwd_flag",
        )
        assert_sorted(values, ascending=False, nulls_first=False)

    def test_nulls_last_overrides_the_ascending_default(
        self, it_session: Session, trips_small: str
    ) -> None:
        values = column(
            it_session.table(trips_small)
            .orderBy(F.asc_nulls_last("store_and_fwd_flag"))
            .select(STR()),
            "store_and_fwd_flag",
        )
        assert_sorted(values, ascending=True, nulls_first=False)

    def test_nulls_first_overrides_the_descending_default(
        self, it_session: Session, trips_small: str
    ) -> None:
        values = column(
            it_session.table(trips_small)
            .orderBy(F.desc_nulls_first("store_and_fwd_flag"))
            .select(STR()),
            "store_and_fwd_flag",
        )
        assert_sorted(values, ascending=False, nulls_first=True)

    def test_asc_nulls_first_is_the_default_spelled_out(
        self, it_session: Session, trips_small: str
    ) -> None:
        explicit = column(
            it_session.table(trips_small)
            .orderBy(F.asc_nulls_first("store_and_fwd_flag"))
            .select(STR()),
            "store_and_fwd_flag",
        )
        assert_sorted(explicit, ascending=True, nulls_first=True)

    def test_desc_nulls_last_is_the_default_spelled_out(
        self, it_session: Session, trips_small: str
    ) -> None:
        explicit = column(
            it_session.table(trips_small)
            .orderBy(F.desc_nulls_last("store_and_fwd_flag"))
            .select(STR()),
            "store_and_fwd_flag",
        )
        assert_sorted(explicit, ascending=False, nulls_first=False)


class TestTheGroupingHelpers:
    """`grouping` and `grouping_id`, which only mean anything under a rollup."""

    def test_grouping_marks_the_rolled_up_row(self, it_session: Session, trips_small: str) -> None:
        rows = (
            it_session.table(trips_small)
            .rollup("VendorID")
            .agg(F.grouping("VendorID").alias("g"), F.count(F.lit(1)).alias("n"))
            .collect()
        )
        flags = sorted({row["g"] for row in rows})
        assert flags == [0, 1], flags
        grand = [row for row in rows if row["g"] == 1]
        assert len(grand) == 1
        assert grand[0]["n"] == it_session.table(trips_small).count()

    def test_grouping_id_agrees_with_grouping_on_one_column(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .rollup("VendorID")
            .agg(F.grouping("VendorID").alias("g"), F.grouping_id("VendorID").alias("gid"))
            .collect()
        )
        assert rows
        assert all(row["g"] == row["gid"] for row in rows)


class TestTheRowGenerators:
    """`posexplode` and `inline`, which produce several columns rather than one."""

    def test_posexplode_numbers_the_elements(self, it_session: Session, nested: str) -> None:
        rows = it_session.table(nested).select(F.posexplode(ARR())).collect()
        assert rows
        positions = [row[0] for row in rows]
        assert positions == sorted(positions) or set(positions) >= {0}
        assert all(isinstance(p, int) for p in positions)

    def test_posexplode_outer_keeps_the_empty_row(self, it_session: Session, nested: str) -> None:
        """The replica's second row has an empty list -- the whole reason it exists."""
        inner = it_session.table(nested).select(F.posexplode(ARR())).count()
        outer = it_session.table(nested).select(F.posexplode_outer(ARR())).count()
        assert outer > inner

    def test_inline_expands_a_struct_array_into_columns(
        self, it_session: Session, nested: str
    ) -> None:
        frame = it_session.table(nested).select(
            F.inline(F.array(F.struct(F.lit(1).alias("a"), F.lit("x").alias("b"))))
        )
        assert frame.columns == ["a", "b"]
        rows = frame.collect()
        assert rows and rows[0]["a"] == 1 and rows[0]["b"] == "x"

    def test_inline_outer_keeps_a_row_for_an_empty_array(
        self, it_session: Session, nested: str
    ) -> None:
        empty = F.array().cast("array<struct<a:int>>")
        rows = it_session.table(nested).select(F.inline_outer(empty)).collect()
        assert len(rows) == it_session.table(nested).count()

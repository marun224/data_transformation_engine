"""The second function tranche, executed against real rows.

Same rule as `test_functions.py`: assert on **values**, state what Spark returns.
Four of the first ninety functions generated plausible SQL and the wrong answer, so
nothing here is taken on trust.
"""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import PySparkValueError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.column import Column
    from icetl.sql.session import SparkSession


def _one(spark: SparkSession, column: Column) -> Any:
    return spark.table("fx.plain").select(column.alias("v")).limit(1).collect()[0]["v"]


class TestTrigonometry:
    def test_the_circle_functions(self, spark: SparkSession) -> None:
        assert _one(spark, F.sin(F.lit(0.0))) == 0.0
        assert _one(spark, F.cos(F.lit(0.0))) == 1.0
        assert math.isclose(_one(spark, F.tan(F.lit(0.0))), 0.0, abs_tol=1e-12)

    def test_the_inverses(self, spark: SparkSession) -> None:
        assert math.isclose(_one(spark, F.asin(F.lit(1.0))), math.pi / 2)
        assert math.isclose(_one(spark, F.acos(F.lit(1.0))), 0.0, abs_tol=1e-12)
        assert math.isclose(_one(spark, F.atan(F.lit(1.0))), math.pi / 4)
        assert math.isclose(_one(spark, F.atan2(F.lit(1.0), F.lit(1.0))), math.pi / 4)

    def test_hyperbolic(self, spark: SparkSession) -> None:
        assert math.isclose(_one(spark, F.sinh(F.lit(0.0))), 0.0, abs_tol=1e-12)
        assert math.isclose(_one(spark, F.cosh(F.lit(0.0))), 1.0)
        assert math.isclose(_one(spark, F.tanh(F.lit(0.0))), 0.0, abs_tol=1e-12)

    def test_degrees_and_radians(self, spark: SparkSession) -> None:
        assert math.isclose(_one(spark, F.degrees(F.lit(math.pi))), 180.0)
        assert math.isclose(_one(spark, F.radians(F.lit(180.0))), math.pi)


class TestMathBreadth:
    def test_cbrt_and_factorial(self, spark: SparkSession) -> None:
        assert math.isclose(_one(spark, F.cbrt(F.lit(27.0))), 3.0)
        assert _one(spark, F.factorial(F.lit(5))) == 120

    def test_hypot(self, spark: SparkSession) -> None:
        assert math.isclose(_one(spark, F.hypot(F.lit(3.0), F.lit(4.0))), 5.0)

    def test_pmod_is_never_negative(self, spark: SparkSession) -> None:
        """Spark's `pmod(-7, 3)` is 2. SQL's `%` keeps the dividend's sign and gives
        -1, which is the divergence this function exists to fix."""
        assert _one(spark, F.pmod(F.lit(-7), F.lit(3))) == 2
        assert _one(spark, F.pmod(F.lit(7), F.lit(3))) == 1

    def test_bit_shifts(self, spark: SparkSession) -> None:
        assert _one(spark, F.shiftleft(F.lit(1), 3)) == 8
        assert _one(spark, F.shiftright(F.lit(8), 3)) == 1

    def test_hex_unhex_bin(self, spark: SparkSession) -> None:
        assert _one(spark, F.hex(F.lit(255))) == "FF"
        assert _one(spark, F.bin(F.lit(5))) == "101"

    def test_negative_and_positive(self, spark: SparkSession) -> None:
        assert _one(spark, F.negative(F.lit(5))) == -5
        assert _one(spark, F.positive(F.lit(5))) == 5


class TestStringBreadth:
    def test_instr_is_one_indexed_and_zero_when_absent(self, spark: SparkSession) -> None:
        assert _one(spark, F.instr(F.lit("abc"), "b")) == 2
        assert _one(spark, F.instr(F.lit("abc"), "z")) == 0

    def test_translate(self, spark: SparkSession) -> None:
        assert _one(spark, F.translate(F.lit("abc"), "ab", "xy")) == "xyc"

    def test_levenshtein(self, spark: SparkSession) -> None:
        assert _one(spark, F.levenshtein(F.lit("kitten"), F.lit("sitting"))) == 3

    def test_left_and_right(self, spark: SparkSession) -> None:
        assert _one(spark, F.left(F.lit("hello"), 2)) == "he"
        assert _one(spark, F.right(F.lit("hello"), 2)) == "lo"

    def test_btrim(self, spark: SparkSession) -> None:
        assert _one(spark, F.btrim(F.lit("  x  "))) == "x"
        assert _one(spark, F.btrim(F.lit("xxaxx"), "x")) == "a"

    def test_substring_index_counting_from_the_left(self, spark: SparkSession) -> None:
        assert _one(spark, F.substring_index(F.lit("a.b.c.d"), ".", 2)) == "a.b"

    def test_substring_index_counting_from_the_right(self, spark: SparkSession) -> None:
        """A negative count counts delimiters from the right, as in Spark."""
        assert _one(spark, F.substring_index(F.lit("a.b.c.d"), ".", -2)) == "c.d"


class TestDateBreadth:
    def test_add_months_returns_a_date(self, spark: SparkSession) -> None:
        start = F.to_date(F.lit("2024-01-15"))
        assert _one(spark, F.add_months(start, 2)) == datetime.date(2024, 3, 15)

    def test_add_months_clamps_to_the_month_end(self, spark: SparkSession) -> None:
        start = F.to_date(F.lit("2024-01-31"))
        assert _one(spark, F.add_months(start, 1)) == datetime.date(2024, 2, 29)

    def test_last_day(self, spark: SparkSession) -> None:
        assert _one(spark, F.last_day(F.to_date(F.lit("2024-02-05")))) == datetime.date(2024, 2, 29)

    def test_weekofyear_is_iso(self, spark: SparkSession) -> None:
        assert _one(spark, F.weekofyear(F.to_date(F.lit("2024-01-04")))) == 1

    def test_months_between(self, spark: SparkSession) -> None:
        later, earlier = F.to_date(F.lit("2024-03-31")), F.to_date(F.lit("2024-01-31"))
        assert _one(spark, F.months_between(later, earlier)) == 2

    def test_trunc_returns_a_date(self, spark: SparkSession) -> None:
        assert _one(spark, F.trunc(F.to_date(F.lit("2024-03-17")), "month")) == datetime.date(
            2024, 3, 1
        )

    def test_make_date(self, spark: SparkSession) -> None:
        made = F.make_date(F.lit(2024), F.lit(3), F.lit(17))
        assert _one(spark, made) == datetime.date(2024, 3, 17)

    def test_timestamp_seconds(self, spark: SparkSession) -> None:
        assert _one(spark, F.timestamp_seconds(F.lit(1704067200))) == datetime.datetime(
            2024, 1, 1, 0, 0
        )

    def test_unix_timestamp_round_trips(self, spark: SparkSession) -> None:
        ts = F.to_timestamp(F.lit("2024-01-01 00:00:00"))
        assert _one(spark, F.unix_timestamp(ts)) == 1704067200

    def test_from_unixtime(self, spark: SparkSession) -> None:
        assert _one(spark, F.from_unixtime(F.lit(1704067200))) == "2024-01-01 00:00:00"

    def test_from_unixtime_refuses_a_java_pattern(self) -> None:
        """Spark's format patterns are Java's; silently reading them as strftime
        would produce plausible nonsense."""
        with pytest.raises(PySparkValueError):
            F.from_unixtime(F.lit(0), "yyyy-MM-dd")


class TestAggregateBreadth:
    def test_population_and_sample_forms_differ(self, spark: SparkSession) -> None:
        """`fx.plain.id` is 1..5, so the two denominators are visibly different."""
        row = (
            spark.table("fx.plain")
            .select(F.var_pop("id").alias("vp"), F.var_samp("id").alias("vs"))
            .collect()[0]
        )
        assert row["vp"] == pytest.approx(2.0)
        assert row["vs"] == pytest.approx(2.5)

    def test_stddev_forms(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.stddev_pop("id").alias("sp"), F.stddev_samp("id").alias("ss"))
            .collect()[0]
        )
        assert row["sp"] == pytest.approx(math.sqrt(2.0))
        assert row["ss"] == pytest.approx(math.sqrt(2.5))

    def test_median_and_mode(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.median("id").alias("m"), F.mode("vendor").alias("mo"))
            .collect()[0]
        )
        assert row["m"] == 3
        assert row["mo"] == "a"

    def test_count_if(self, spark: SparkSession) -> None:
        result = spark.table("fx.plain").select(F.count_if(F.col("id") > 3).alias("n"))
        assert result.collect()[0]["n"] == 2

    def test_bool_and_or(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.bool_and(F.col("id") > 0).alias("a"), F.bool_or(F.col("id") > 4).alias("o"))
            .collect()[0]
        )
        assert row["a"] is True
        assert row["o"] is True

    def test_max_by_and_min_by(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.max_by("id", "id").alias("mx"), F.min_by("id", "id").alias("mn"))
            .collect()[0]
        )
        assert (row["mx"], row["mn"]) == (5, 1)

    def test_corr_and_covar(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.corr("id", "id").alias("c"), F.covar_pop("id", "id").alias("cv"))
            .collect()[0]
        )
        assert row["c"] == pytest.approx(1.0)
        assert row["cv"] == pytest.approx(2.0)

    def test_percentile_approx(self, spark: SparkSession) -> None:
        result = spark.table("fx.plain").select(F.percentile_approx("id", 0.5).alias("p"))
        assert result.collect()[0]["p"] == 3

    def test_percentile_approx_refuses_an_accuracy_argument(self) -> None:
        with pytest.raises(PySparkValueError):
            F.percentile_approx("id", 0.5, accuracy=100)

    def test_skewness_and_kurtosis_run(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.skewness("id").alias("s"), F.kurtosis("id").alias("k"))
            .collect()[0]
        )
        assert row["s"] == pytest.approx(0.0, abs=1e-9)
        assert row["k"] is not None


class TestArrays:
    def _arr(self, *values: Any) -> Column:
        return F.array(*[F.lit(v) for v in values])

    def test_array_contains(self, spark: SparkSession) -> None:
        assert _one(spark, F.array_contains(self._arr(1, 2, 3), 2)) is True
        assert _one(spark, F.array_contains(self._arr(1, 2, 3), 9)) is False

    def test_array_distinct(self, spark: SparkSession) -> None:
        assert sorted(_one(spark, F.array_distinct(self._arr(1, 1, 2)))) == [1, 2]

    def test_array_position_is_one_indexed_and_zero_when_absent(self, spark: SparkSession) -> None:
        """DuckDB returns NULL for a missing element, which would read as "unknown"
        rather than Spark's "not there"."""
        assert _one(spark, F.array_position(self._arr(10, 20, 30), 20)) == 2
        assert _one(spark, F.array_position(self._arr(10, 20, 30), 99)) == 0

    def test_array_remove(self, spark: SparkSession) -> None:
        assert _one(spark, F.array_remove(self._arr(1, 2, 1, 3), 1)) == [2, 3]

    def test_array_sort_and_sort_array(self, spark: SparkSession) -> None:
        assert _one(spark, F.array_sort(self._arr(3, 1, 2))) == [1, 2, 3]
        assert _one(spark, F.sort_array(self._arr(3, 1, 2), asc=False)) == [3, 2, 1]

    def test_array_max_and_min(self, spark: SparkSession) -> None:
        assert _one(spark, F.array_max(self._arr(1, 5, 3))) == 5
        assert _one(spark, F.array_min(self._arr(1, 5, 3))) == 1

    def test_array_join(self, spark: SparkSession) -> None:
        assert _one(spark, F.array_join(self._arr("a", "b"), "-")) == "a-b"

    def test_array_union_deduplicates(self, spark: SparkSession) -> None:
        """Spark's `array_union` is a set union; a plain concat would keep the dupe."""
        assert sorted(_one(spark, F.array_union(self._arr(1, 2), self._arr(2, 3)))) == [1, 2, 3]

    def test_array_intersect_and_overlap(self, spark: SparkSession) -> None:
        assert sorted(_one(spark, F.array_intersect(self._arr(1, 2, 3), self._arr(2, 3, 4)))) == [
            2,
            3,
        ]
        assert _one(spark, F.arrays_overlap(self._arr(1, 2), self._arr(2, 9))) is True

    def test_element_at_is_one_indexed(self, spark: SparkSession) -> None:
        """Spark's own inconsistency: `element_at` is 1-based while `getItem` is 0-based."""
        assert _one(spark, F.element_at(self._arr("x", "y", "z"), 1)) == "x"
        assert _one(spark, F.element_at(self._arr("x", "y", "z"), 3)) == "z"

    def test_slice_is_one_indexed(self, spark: SparkSession) -> None:
        assert _one(spark, F.slice(self._arr(1, 2, 3, 4, 5), 2, 3)) == [2, 3, 4]

    def test_flatten(self, spark: SparkSession) -> None:
        nested = F.array(self._arr(1, 2), self._arr(3))
        assert _one(spark, F.flatten(nested)) == [1, 2, 3]

    def test_sequence_includes_its_endpoint(self, spark: SparkSession) -> None:
        """Spark's `sequence` is inclusive; DuckDB's `range` is not."""
        assert _one(spark, F.sequence(F.lit(1), F.lit(4))) == [1, 2, 3, 4]

    def test_size(self, spark: SparkSession) -> None:
        assert _one(spark, F.size(self._arr(1, 2, 3))) == 3


class TestMaps:
    def test_map_keys_and_values(self, spark: SparkSession) -> None:
        rows = (
            spark.table("fx.nested")
            .select(F.map_keys("scores").alias("k"), F.map_values("scores").alias("v"))
            .collect()
        )
        keys = sorted(key for row in rows for key in row["k"])
        assert keys == ["a", "b", "c"]
        assert sorted(v for row in rows for v in row["v"]) == [1, 2, 3]


class TestNullHandling:
    def test_ifnull(self, spark: SparkSession) -> None:
        assert _one(spark, F.ifnull(F.lit(None), F.lit(5))) == 5

    def test_nvl2(self, spark: SparkSession) -> None:
        assert _one(spark, F.nvl2(F.lit(1), F.lit("yes"), F.lit("no"))) == "yes"
        assert _one(spark, F.nvl2(F.lit(None), F.lit("yes"), F.lit("no"))) == "no"


class TestHash:
    def test_hash_is_stable_within_a_query(self, spark: SparkSession) -> None:
        row = (
            spark.table("fx.plain")
            .select(F.hash(F.lit("abc")).alias("a"), F.hash(F.lit("abc")).alias("b"))
            .limit(1)
            .collect()[0]
        )
        assert row["a"] == row["b"]

    def test_hash_is_documented_as_not_sparks(self) -> None:
        """Spark's `hash` is a specific Murmur3 variant; DuckDB's is its own. Anything
        persisted or compared across engines must not rely on this."""
        assert "Murmur3" in (F.hash.__doc__ or "")


class TestOutputNaming:
    """Spark's generated column names, which scripts index results by.

    The rule that matters here is case: Spark spells a function name in **lower**
    case in a generated name (`sum(amount)`), while keeping SQL keywords upper
    (`CAST(a AS INT)`). Getting it wrong does not fail a query -- it fails
    `row["sum(amount)"]` in someone's script, one layer away from the cause.
    """

    def test_an_aggregate_is_named_in_lower_case(self, spark: SparkSession) -> None:
        df = spark.table("fx.plain").select(F.sum("amount"), F.avg("amount"))
        assert df.columns == ["sum(amount)", "avg(amount)"]

    def test_count_star_is_named_count_one(self, spark: SparkSession) -> None:
        """Spark rewrites `count(*)` to `count(1)` before naming the column."""
        assert spark.table("fx.plain").select(F.count("*")).columns == ["count(1)"]

    def test_count_distinct_keeps_the_keyword_upper(self, spark: SparkSession) -> None:
        assert spark.table("fx.plain").select(F.countDistinct("vendor")).columns == [
            "count(DISTINCT vendor)"
        ]

    def test_a_scalar_function_is_named_in_lower_case(self, spark: SparkSession) -> None:
        assert spark.table("fx.plain").select(F.upper("vendor")).columns == ["upper(vendor)"]

    def test_a_cast_keeps_its_keyword_upper(self, spark: SparkSession) -> None:
        assert spark.table("fx.plain").select(F.col("id").cast("int")).columns == [
            "CAST(id AS INT)"
        ]

    def test_an_operator_is_parenthesised(self, spark: SparkSession) -> None:
        assert spark.table("fx.plain").select(F.col("id") + 1).columns == ["(id + 1)"]

    def test_a_nested_call(self, spark: SparkSession) -> None:
        assert spark.table("fx.plain").select(F.upper(F.trim("vendor"))).columns == [
            "upper(trim(vendor))"
        ]

    def test_an_explicit_alias_still_wins(self, spark: SparkSession) -> None:
        assert spark.table("fx.plain").select(F.sum("amount").alias("total")).columns == ["total"]

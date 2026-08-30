"""Every `F.*` function, executed against real rows.

Generated SQL is not evidence. `F.split` generated perfectly plausible SQL --
`STR_SPLIT(s, '[0-9]')` -- that returned the whole string as one element, because
DuckDB's `str_split` treats the pattern literally while the reference engine's `split` treats it as
a regex. Nothing raised. That is the failure mode every test here exists to catch, so
each asserts on **values**, and each states what the reference engine returns.

`_one` evaluates an expression against a single fixture row, which keeps a test to
one line of setup and makes the expected value the point of the test.
"""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import EngineTypeError, EngineValueError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.column import Column
    from icetl.sql.session import Session


def _one(session: Session, column: Column) -> Any:
    """Evaluate `column` against one row of `fx.plain` and return the value."""
    return session.table("fx.plain").select(column.alias("v")).limit(1).collect()[0]["v"]


class TestConditionals:
    def test_when_otherwise(self, session: Session) -> None:
        result = session.table("fx.plain").select(
            F.when(F.col("id") > 3, F.lit("big")).otherwise(F.lit("small")).alias("v"),
            F.col("id"),
        )
        by_id = {row["id"]: row["v"] for row in result.collect()}
        assert by_id[1] == "small"
        assert by_id[5] == "big"

    def test_when_without_otherwise_is_null(self, session: Session) -> None:
        """The reference engine leaves an unmatched row NULL, as SQL's CASE does."""
        result = session.table("fx.plain").select(
            F.col("id"), F.when(F.col("id") > 3, F.lit("big")).alias("v")
        )
        by_id = {row["id"]: row["v"] for row in result.collect()}
        assert by_id[1] is None

    def test_when_chained(self, session: Session) -> None:
        column = (
            F.when(F.col("id") == 1, F.lit("one"))
            .when(F.col("id") == 2, F.lit("two"))
            .otherwise(F.lit("many"))
        )
        result = session.table("fx.plain").select(F.col("id"), column.alias("v"))
        by_id = {row["id"]: row["v"] for row in result.collect()}
        assert [by_id[1], by_id[2], by_id[5]] == ["one", "two", "many"]

    def test_when_requires_a_column_condition(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            F.when(True, F.lit(1))  # type: ignore[arg-type]

    def test_otherwise_only_once(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            F.when(F.col("id") > 1, 1).otherwise(0).otherwise(2)

    def test_coalesce_takes_the_first_non_null(self, session: Session) -> None:
        assert _one(session, F.coalesce(F.lit(None), F.lit(None), F.lit(7))) == 7

    def test_nullif(self, session: Session) -> None:
        assert _one(session, F.nullif(F.lit(3), F.lit(3))) is None
        assert _one(session, F.nullif(F.lit(3), F.lit(4))) == 3

    def test_greatest_and_least_ignore_nulls(self, session: Session) -> None:
        """The reference engine's greatest/least skip NULLs rather than propagating them."""
        assert _one(session, F.greatest(F.lit(1), F.lit(None), F.lit(9))) == 9
        assert _one(session, F.least(F.lit(1), F.lit(None), F.lit(9))) == 1

    def test_greatest_needs_two_arguments(self) -> None:
        with pytest.raises(EngineValueError):
            F.greatest(F.lit(1))


class TestStrings:
    def test_upper_lower(self, session: Session) -> None:
        assert _one(session, F.upper(F.lit("aBc"))) == "ABC"
        assert _one(session, F.lower(F.lit("aBc"))) == "abc"

    def test_a_string_argument_names_a_column(self, session: Session) -> None:
        """The the reference API asymmetry: in `functions`, a string is a column name."""
        values = {
            row["v"]
            for row in session.table("fx.plain").select(F.upper("vendor").alias("v")).collect()
        }
        assert values == {"A", "B", "C", None}

    def test_length(self, session: Session) -> None:
        assert _one(session, F.length(F.lit("hello"))) == 5

    def test_trim_family(self, session: Session) -> None:
        assert _one(session, F.trim(F.lit("  x  "))) == "x"
        assert _one(session, F.ltrim(F.lit("  x"))) == "x"
        assert _one(session, F.rtrim(F.lit("x  "))) == "x"

    def test_concat_propagates_null(self, session: Session) -> None:
        """The reference engine's `concat` is NULL if any argument is NULL. DuckDB's `concat`
        silently skips NULLs, which is why this generates `||` instead."""
        assert _one(session, F.concat(F.lit("a"), F.lit(None), F.lit("b"))) is None
        assert _one(session, F.concat(F.lit("a"), F.lit("b"))) == "ab"

    def test_concat_ws_skips_null(self, session: Session) -> None:
        """Unlike `concat`, the reference engine's `concat_ws` drops NULLs -- and DuckDB agrees."""
        assert _one(session, F.concat_ws("-", F.lit("a"), F.lit(None), F.lit("b"))) == "a-b"

    def test_substring_is_one_indexed(self, session: Session) -> None:
        assert _one(session, F.substring(F.lit("hello"), 2, 3)) == "ell"

    def test_split_treats_the_pattern_as_a_regex(self, session: Session) -> None:
        """The bug this file exists for: DuckDB's `str_split` is literal, so this
        would come back as `['a1b2c']` with no error."""
        assert _one(session, F.split(F.lit("a1b2c"), "[0-9]")) == ["a", "b", "c"]

    def test_split_on_a_plain_separator_still_works(self, session: Session) -> None:
        assert _one(session, F.split(F.lit("a,b,c"), ",")) == ["a", "b", "c"]

    def test_regexp_replace_replaces_every_match(self, session: Session) -> None:
        """The reference replaces all occurrences; DuckDB without the `g` flag replaces one."""
        assert _one(session, F.regexp_replace(F.lit("a1b2c3"), "[0-9]", "-")) == "a-b-c-"

    def test_regexp_extract_returns_a_group(self, session: Session) -> None:
        assert _one(session, F.regexp_extract(F.lit("id-42"), r"id-(\d+)", 1)) == "42"

    def test_regexp_extract_is_empty_when_nothing_matches(self, session: Session) -> None:
        """The reference engine returns "" rather than NULL for a non-match."""
        assert _one(session, F.regexp_extract(F.lit("nope"), r"id-(\d+)", 1)) == ""

    def test_replace_is_literal(self, session: Session) -> None:
        assert _one(session, F.replace(F.lit("a.b"), F.lit("."), F.lit("-"))) == "a-b"

    def test_initcap_reverse_ascii(self, session: Session) -> None:
        assert _one(session, F.initcap(F.lit("hello world"))) == "Hello World"
        assert _one(session, F.reverse(F.lit("abc"))) == "cba"
        assert _one(session, F.ascii(F.lit("A"))) == 65

    def test_padding(self, session: Session) -> None:
        assert _one(session, F.lpad(F.lit("7"), 3, "0")) == "007"
        assert _one(session, F.rpad(F.lit("7"), 3, "0")) == "700"

    def test_repeat(self, session: Session) -> None:
        assert _one(session, F.repeat(F.lit("ab"), 3)) == "ababab"

    def test_locate_is_one_indexed_and_zero_when_absent(self, session: Session) -> None:
        assert _one(session, F.locate("b", F.lit("abc"))) == 2
        assert _one(session, F.locate("z", F.lit("abc"))) == 0


class TestMath:
    def test_abs_ceil_floor(self, session: Session) -> None:
        assert _one(session, F.abs(F.lit(-3))) == 3
        assert _one(session, F.ceil(F.lit(1.2))) == 2
        assert _one(session, F.floor(F.lit(1.8))) == 1

    def test_round_is_half_up(self, session: Session) -> None:
        """The reference engine rounds half away from zero, and so does DuckDB -- pinned in case
        that stops being true."""
        assert float(_one(session, F.round(F.lit(2.5)))) == 3.0
        assert float(_one(session, F.round(F.lit(-2.5)))) == -3.0

    def test_round_with_scale(self, session: Session) -> None:
        assert float(_one(session, F.round(F.lit(3.456), 2))) == 3.46

    def test_sqrt_exp_pow(self, session: Session) -> None:
        assert _one(session, F.sqrt(F.lit(9))) == 3.0
        assert math.isclose(_one(session, F.exp(F.lit(0))), 1.0)
        assert _one(session, F.pow(F.lit(2), 10)) == 1024.0

    def test_logs(self, session: Session) -> None:
        assert math.isclose(_one(session, F.log(F.lit(math.e))), 1.0)
        assert math.isclose(_one(session, F.log(2, F.lit(8))), 3.0)
        assert math.isclose(_one(session, F.log2(F.lit(8))), 3.0)
        assert math.isclose(_one(session, F.log10(F.lit(1000))), 3.0)

    def test_signum(self, session: Session) -> None:
        assert _one(session, F.signum(F.lit(-5))) == -1


class TestDateTime:
    def test_year_month_day(self, session: Session) -> None:
        date = F.to_date(F.lit("2024-03-17"))
        assert _one(session, F.year(date)) == 2024
        assert _one(session, F.month(date)) == 3
        assert _one(session, F.dayofmonth(date)) == 17
        assert _one(session, F.quarter(date)) == 1

    def test_dayofweek_numbers_sunday_one(self, session: Session) -> None:
        """The reference engine: Sunday is 1, Saturday is 7. DuckDB numbers Sunday 0, so every day
        would be off by one with nothing raising."""
        sunday = F.to_date(F.lit("2024-01-07"))
        monday = F.to_date(F.lit("2024-01-08"))
        saturday = F.to_date(F.lit("2024-01-13"))
        assert _one(session, F.dayofweek(sunday)) == 1
        assert _one(session, F.dayofweek(monday)) == 2
        assert _one(session, F.dayofweek(saturday)) == 7

    def test_dayofyear(self, session: Session) -> None:
        assert _one(session, F.dayofyear(F.to_date(F.lit("2024-02-01")))) == 32

    def test_time_parts(self, session: Session) -> None:
        ts = F.to_timestamp(F.lit("2024-01-08 13:45:56"))
        assert _one(session, F.hour(ts)) == 13
        assert _one(session, F.minute(ts)) == 45
        assert _one(session, F.second(ts)) == 56

    def test_to_date(self, session: Session) -> None:
        assert _one(session, F.to_date(F.lit("2024-01-08"))) == datetime.date(2024, 1, 8)

    def test_date_add_and_sub(self, session: Session) -> None:
        date = F.to_date(F.lit("2024-01-08"))
        assert _one(session, F.date_add(date, 5)) == datetime.date(2024, 1, 13)
        assert _one(session, F.date_sub(date, 5)) == datetime.date(2024, 1, 3)

    def test_datediff_is_end_minus_start(self, session: Session) -> None:
        """The reference engine's argument order is `datediff(end, start)`."""
        end, start = F.to_date(F.lit("2024-01-10")), F.to_date(F.lit("2024-01-01"))
        assert _one(session, F.datediff(end, start)) == 9

    def test_date_trunc(self, session: Session) -> None:
        ts = F.to_timestamp(F.lit("2024-03-17 13:45:56"))
        assert _one(session, F.date_trunc("month", ts)) == datetime.datetime(2024, 3, 1)

    def test_current_date_runs(self, session: Session) -> None:
        assert isinstance(_one(session, F.current_date()), datetime.date)


class TestHashing:
    def test_md5(self, session: Session) -> None:
        assert _one(session, F.md5(F.lit("abc"))) == "900150983cd24fb0d6963f7d28e17f72"

    def test_sha2_256(self, session: Session) -> None:
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        assert _one(session, F.sha2(F.lit("abc"), 256)) == expected

    def test_sha2_rejects_a_width_duckdb_lacks(self) -> None:
        with pytest.raises(EngineValueError):
            F.sha2(F.lit("abc"), 224)


class TestAggregates:
    def test_count_star_counts_rows(self, session: Session) -> None:
        assert session.table("fx.plain").select(F.count("*").alias("n")).collect()[0]["n"] == 5

    def test_count_of_a_column_skips_nulls(self, session: Session) -> None:
        """`vendor` has one NULL, and the reference engine's `count(col)` does not count it."""
        assert session.table("fx.plain").select(F.count("vendor").alias("n")).collect()[0]["n"] == 4

    def test_count_distinct(self, session: Session) -> None:
        result = session.table("fx.plain").select(F.countDistinct("vendor").alias("n"))
        assert result.collect()[0]["n"] == 3

    def test_sum_avg_min_max(self, session: Session) -> None:
        result = session.table("fx.plain").select(
            F.sum("id").alias("s"),
            F.avg("id").alias("a"),
            F.min("id").alias("mn"),
            F.max("id").alias("mx"),
        )
        row = result.collect()[0]
        assert (row["s"], row["a"], row["mn"], row["mx"]) == (15, 3.0, 1, 5)

    def test_aggregates_skip_nulls(self, session: Session) -> None:
        """`amount` has a NULL; the reference engine's `sum` and `avg` ignore it."""
        row = (
            session.table("fx.plain")
            .select(F.sum("amount").alias("s"), F.avg("amount").alias("a"))
            .collect()[0]
        )
        assert row["s"] == pytest.approx(110.75)
        assert row["a"] == pytest.approx(110.75 / 4)

    def test_collect_list_and_set(self, session: Session) -> None:
        row = (
            session.table("fx.plain")
            .select(F.collect_list("vendor").alias("lst"), F.collect_set("vendor").alias("st"))
            .collect()[0]
        )
        assert sorted(v for v in row["lst"] if v) == ["a", "a", "b", "c"]
        assert sorted(v for v in row["st"] if v) == ["a", "b", "c"]

    def test_stddev_and_variance_are_sample_not_population(self, session: Session) -> None:
        """The reference engine's `stddev`/`variance` are the *sample* forms."""
        row = (
            session.table("fx.plain")
            .select(F.stddev("id").alias("sd"), F.variance("id").alias("var"))
            .collect()[0]
        )
        assert row["var"] == pytest.approx(2.5)
        assert row["sd"] == pytest.approx(math.sqrt(2.5))

    def test_first_rejects_ignorenulls(self) -> None:
        """Refused rather than silently keeping NULLs."""
        with pytest.raises(EngineValueError):
            F.first("vendor", ignorenulls=True)


class TestCollections:
    def test_array_and_size(self, session: Session) -> None:
        assert _one(session, F.array(F.lit(1), F.lit(2), F.lit(3))) == [1, 2, 3]
        assert _one(session, F.size(F.array(F.lit(1), F.lit(2)))) == 2

    def test_struct_names_its_fields(self, session: Session) -> None:
        value = _one(session, F.struct("id", "vendor"))
        assert set(value) == {"id", "vendor"}

    def test_element_access_is_zero_indexed(self, session: Session) -> None:
        """The reference engine's arrays are 0-based; DuckDB's lists are 1-based."""
        arr = F.array(F.lit("x"), F.lit("y"), F.lit("z"))
        assert _one(session, arr[0]) == "x"
        assert _one(session, arr[2]) == "z"

    def test_get_field_on_a_struct(self, session: Session) -> None:
        assert _one(session, F.struct("id").getField("id")) == 1


class TestColumnMethods:
    def test_like_and_ilike(self, session: Session) -> None:
        assert _one(session, F.lit("hello").like("hel%")) is True
        assert _one(session, F.lit("HELLO").ilike("hel%")) is True

    def test_rlike(self, session: Session) -> None:
        assert _one(session, F.lit("abc123").rlike(r"\d+")) is True

    def test_contains_startswith_endswith(self, session: Session) -> None:
        value = F.lit("hello")
        assert _one(session, value.contains("ell")) is True
        assert _one(session, value.startswith("he")) is True
        assert _one(session, value.endswith("lo")) is True

    def test_substr(self, session: Session) -> None:
        assert _one(session, F.lit("hello").substr(2, 3)) == "ell"

    def test_eq_null_safe(self, session: Session) -> None:
        assert _one(session, F.lit(None).eqNullSafe(F.lit(None))) is True
        assert _one(session, F.lit(1).eqNullSafe(F.lit(None))) is False

    def test_bitwise(self, session: Session) -> None:
        assert _one(session, F.lit(6).bitwiseAND(F.lit(3))) == 2
        assert _one(session, F.lit(6).bitwiseOR(F.lit(3))) == 7
        assert _one(session, F.lit(6).bitwiseXOR(F.lit(3))) == 5

    def test_isnull_and_isnan(self, session: Session) -> None:
        assert _one(session, F.isnull(F.lit(None))) is True
        assert _one(session, F.lit(1.0).isNaN()) is False


class TestOrdering:
    def test_asc_puts_nulls_first(self, session: Session) -> None:
        """`F.asc` leaves null placement unstated; the conformance pass fills in
        The reference engine's default."""
        rows = session.sql("SELECT vendor FROM fx.plain ORDER BY vendor").collect()
        assert next(row[0] for row in rows) is None

    def test_asc_nulls_last_overrides(self, session: Session) -> None:
        column = F.asc_nulls_last("vendor")
        assert column._expression.args["nulls_first"] is False

    def test_desc_nulls_first_overrides(self, session: Session) -> None:
        assert F.desc_nulls_first("vendor")._expression.args["nulls_first"] is True

    def test_asc_leaves_it_to_the_conformance_pass(self) -> None:
        assert F.asc("vendor")._expression.args.get("nulls_first") is None


class TestArgumentErrors:
    def test_a_literal_where_a_column_is_expected(self) -> None:
        """The error names the fix, because this is the commonest mistake."""
        with pytest.raises(EngineTypeError, match=r"F\.lit"):
            F.upper(3)

    def test_coalesce_needs_an_argument(self) -> None:
        with pytest.raises(EngineValueError):
            F.coalesce()

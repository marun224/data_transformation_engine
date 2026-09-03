"""The Spark-3.5 semantics icetl preserves, checked where they actually bite.

`compat/divergence.md` records every place DuckDB's behaviour is a defensible choice
rather than the only one, and the conformance layer rewrites the tree so the reference's
answer is the one returned. The local suite proves the rewrite happens. What it cannot
prove is that the rewrite still holds once the data is real -- and these particular rules
are all about **edge values**, which is exactly what a five-row fixture is short of.

The seeded slice supplies them without anyone arranging it:

  * 76 trips of exactly **0.0** miles, so `x / 0` is a real query, not a contrived one.
  * 971 rows where `store_and_fwd_flag` is **NULL**, so null ordering and null-safe
    equality have something to order and compare.
  * A string column that never parses as a number, so a failed cast is reachable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import QueryExecutionException
from icetl.sql import functions as F
from tests.integration.helpers import assert_sorted, column

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestDivisionByZero:
    """`1/0` is NULL in the reference. DuckDB alone would give infinity."""

    def test_dividing_by_a_real_zero_is_null(self, it_session: Session, trips_small: str) -> None:
        zeros = it_session.table(trips_small).filter(F.col("trip_distance") == 0).count()
        assert zeros > 0, "no zero-distance trips, so this proves nothing"
        nulls = (
            it_session.table(trips_small)
            .select((F.col("fare_amount") / F.col("trip_distance")).alias("q"))
            .filter(F.col("q").isNull())
            .count()
        )
        assert nulls == zeros

    def test_the_non_zero_rows_still_divide(self, it_session: Session, trips_small: str) -> None:
        """The rule must not turn every division into NULL."""
        rows = (
            it_session.table(trips_small)
            .filter(F.col("trip_distance") > 0)
            .select(
                F.col("fare_amount").alias("f"),
                F.col("trip_distance").alias("d"),
                (F.col("fare_amount") / F.col("trip_distance")).alias("q"),
            )
            .limit(100)
            .collect()
        )
        assert rows
        for row in rows:
            assert row["q"] == pytest.approx(row["f"] / row["d"])

    def test_no_quotient_is_infinite(self, it_session: Session, trips_small: str) -> None:
        """The failure this rule prevents, stated as the absence of infinity."""
        import math

        quotients = column(
            it_session.table(trips_small).select(
                (F.col("fare_amount") / F.col("trip_distance")).alias("q")
            ),
            "q",
        )
        assert not any(v is not None and math.isinf(v) for v in quotients)

    def test_modulo_by_a_real_zero_gives_nan_not_null(
        self, it_session: Session, trips_small: str
    ) -> None:
        """FINDINGS 1.14 -- pinned. divergence.md line 60 says this needs no rule.

        It does. `/` by zero is rewritten to NULL and `%` is not, so on two real
        `DOUBLE` columns the reference's NULL comes back as **NaN** -- a value, not a
        NULL, so nothing downstream notices. The literal forms the divergence table was
        checked against (`5 % 0`, `5.0 % 0.0`) fold to DECIMAL and do give NULL, which
        is why it looked settled.

        A characterisation test: it fails when the rule is added.
        """
        import math

        zeros = it_session.table(trips_small).filter(F.col("trip_distance") == 0).count()
        assert zeros > 0

        remainders = column(
            it_session.table(trips_small)
            .filter(F.col("trip_distance") == 0)
            .select((F.col("fare_amount") % F.col("trip_distance")).alias("q")),
            "q",
        )
        assert remainders
        assert all(v is not None and math.isnan(v) for v in remainders), (
            "modulo by zero no longer gives NaN -- FINDINGS 1.14 is fixed, so this "
            "test should become an assertion that every remainder is NULL"
        )

    def test_division_by_zero_is_still_handled(self, it_session: Session, trips_small: str) -> None:
        """The control: `/` has the rule that `%` is missing."""
        quotients = column(
            it_session.table(trips_small)
            .filter(F.col("trip_distance") == 0)
            .select((F.col("fare_amount") / F.col("trip_distance")).alias("q")),
            "q",
        )
        assert quotients
        assert all(v is None for v in quotients)


class TestFailedCasts:
    """A cast that cannot succeed is NULL, not an error -- outside ANSI mode."""

    def test_an_unparseable_string_casts_to_null(
        self, it_session: Session, trips_small: str
    ) -> None:
        """`store_and_fwd_flag` holds 'Y' and 'N', neither of which is a number."""
        values = column(
            it_session.table(trips_small)
            .filter(F.col("store_and_fwd_flag").isNotNull())
            .select(F.col("store_and_fwd_flag").cast("int").alias("v")),
            "v",
        )
        assert values
        assert set(values) == {None}

    def test_the_query_does_not_fail(self, it_session: Session, trips_small: str) -> None:
        """The point of the rule: a bad cast must not kill the whole scan."""
        frame = it_session.table(trips_small).select(
            F.col("store_and_fwd_flag").cast("int").alias("v")
        )
        assert frame.count() == it_session.table(trips_small).count()

    def test_a_valid_cast_returns_a_number(self, it_session: Session, trips_small: str) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("trip_distance").alias("d"),
                F.col("trip_distance").cast("int").alias("i"),
            )
            .limit(100)
            .collect()
        )
        assert rows
        for row in rows:
            assert isinstance(row["i"], int)
            assert abs(row["i"] - row["d"]) <= 1

    def test_casting_a_double_to_an_int_rounds_instead_of_truncating(
        self, it_session: Session
    ) -> None:
        """FINDINGS 1.15 -- pinned. The reference truncates toward zero; this rounds.

        `CAST(3.94 AS INT)` is 3 in the reference and 4 here, and `CAST(-3.94 AS INT)`
        is -3 there and -4 here. Nothing raises and the result is a plausible integer,
        so every double-to-int cast in an ETL job is off by one for most inputs.

        Both surfaces agree with each other and disagree with the reference, so P1
        holds and the conformance rule is simply absent.

        A characterisation test: it fails when the rule is added.
        """
        for value, rounded, reference in [(3.94, 4, 3), (-3.94, -4, -3), (2.5, 3, 2)]:
            got = it_session.sql(f"SELECT CAST({value} AS INT) AS v").collect()[0]["v"]
            assert got == rounded, (
                f"CAST({value} AS INT) gave {got}, not the rounded {rounded} -- if it "
                f"is now the reference's {reference}, FINDINGS 1.15 is fixed"
            )
            assert got != reference

    def test_ansi_mode_raises_instead(self, ansi_session: Session, trips_small: str) -> None:
        """The other half of the rule: `icetl.ansiMode=true` makes it an error."""
        with pytest.raises(QueryExecutionException):
            ansi_session.table(trips_small).select(
                F.col("store_and_fwd_flag").cast("int").alias("v")
            ).collect()


class TestNullOrdering:
    """Nulls first ascending, last descending -- the reference's order, not DuckDB's."""

    def test_ascending_puts_the_real_nulls_first(
        self, it_session: Session, trips_small: str
    ) -> None:
        values = column(
            it_session.table(trips_small)
            .orderBy("store_and_fwd_flag")
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        assert None in values
        assert_sorted(values, ascending=True, nulls_first=True)

    def test_descending_puts_them_last(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small)
            .orderBy(F.col("store_and_fwd_flag").desc())
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        assert_sorted(values, ascending=False, nulls_first=False)

    def test_both_surfaces_order_nulls_alike(self, it_session: Session, trips_small: str) -> None:
        via_sql = column(
            it_session.sql(
                f"SELECT store_and_fwd_flag FROM {trips_small} ORDER BY store_and_fwd_flag"
            ),
            "store_and_fwd_flag",
        )
        via_frame = column(
            it_session.table(trips_small)
            .orderBy("store_and_fwd_flag")
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        assert via_sql[:20] == via_frame[:20]


class TestNullSemantics:
    """NULL is not false, and it is not equal to itself."""

    def test_equality_with_null_is_null_not_false(
        self, it_session: Session, trips_small: str
    ) -> None:
        """So a NULL row satisfies neither `= 'Y'` nor `<> 'Y'`."""
        frame = it_session.table(trips_small)
        nulls = frame.filter(F.col("store_and_fwd_flag").isNull()).count()
        equal = frame.filter(F.col("store_and_fwd_flag") == "Y").count()
        unequal = frame.filter(F.col("store_and_fwd_flag") != "Y").count()
        assert nulls > 0
        assert equal + unequal + nulls == frame.count()

    def test_null_safe_equality_matches_null_to_null(
        self, it_session: Session, trips_small: str
    ) -> None:
        """`<=>` is the operator that does treat two NULLs as equal."""
        frame = it_session.table(trips_small)
        nulls = frame.filter(F.col("store_and_fwd_flag").isNull()).count()
        matched = frame.filter(
            F.col("store_and_fwd_flag").eqNullSafe(F.lit(None).cast("string"))
        ).count()
        assert matched == nulls

    def test_not_of_a_null_predicate_is_still_null(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        positive = frame.filter(F.col("store_and_fwd_flag") == "Y").count()
        negated = frame.filter(~(F.col("store_and_fwd_flag") == "Y")).count()
        assert positive + negated < frame.count(), "the NULL rows were swept into one side"

    def test_an_aggregate_ignores_nulls(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small)
        row = frame.select(
            F.count(F.col("passenger_count")).alias("counted"),
            F.avg(F.col("passenger_count")).alias("mean"),
            F.sum(F.col("passenger_count")).alias("total"),
        ).collect()[0]
        assert row["counted"] < frame.count()
        # `sum` of a bigint column comes back as Decimal, so the comparison has to
        # be made in one number system rather than two.
        assert row["mean"] == pytest.approx(float(row["total"]) / row["counted"], rel=1e-9)


class TestPartitionColumnsComeFromTheData:
    """DuckDB would synthesise a typed column from the directory name."""

    def test_the_partition_column_reads_as_its_declared_type(
        self, it_session: Session, trips: str
    ) -> None:
        """`icetl_it.trips` is laid out as `.../VendorID=1/...` on MinIO.

        If DuckDB sourced the column from the path it would come back as a string, or
        as the wrong integer width.
        """
        values = column(it_session.table(trips).select("VendorID").limit(10), "VendorID")
        assert values
        assert all(isinstance(v, int) for v in values), [type(v).__name__ for v in values]

    def test_the_partition_values_match_the_rows(self, it_session: Session, trips: str) -> None:
        """Counting by the partition column must agree with counting the whole table."""
        per_vendor = it_session.table(trips).groupBy("VendorID").count().collect()
        assert sum(row["count"] for row in per_vendor) == it_session.table(trips).count()

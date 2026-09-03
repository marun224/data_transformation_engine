"""Window functions over real partitions.

The local suite deliberately builds its own `ties` and `gaps` frames, because the
distinction between a `ROWS` frame and a `RANGE` frame only shows up when the ordering
column has **duplicate values**, and a hand-made fixture is the easiest way to guarantee
some.

Real trip data guarantees them for free, and at a scale a fixture cannot: thousands of
trips share a pickup timestamp to the second. So the frame-semantics tests here are not
a copy of the local ones -- they are the same question asked where the ties arise
naturally, over 3 real vendor partitions and 5,000 rows.

The invariants do the work. `row_number()` over a partition of n rows is exactly 1..n,
whatever the rows are; a running sum ends at the partition total; `rank` and
`dense_rank` agree exactly when there are no ties. None of that needs a golden value.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from icetl.sql.window import Window

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


def _by_vendor() -> object:
    return Window.partitionBy("VendorID").orderBy("tpep_pickup_datetime")


class TestRowNumbering:
    """`row_number` over a partition of n rows is 1..n. Nothing else is acceptable."""

    def test_row_number_numbers_each_partition_from_one(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("VendorID"),
                F.row_number().over(_by_vendor()).alias("rn"),
            )
            .collect()
        )
        per_vendor: dict[object, list[int]] = {}
        for row in rows:
            per_vendor.setdefault(row["VendorID"], []).append(row["rn"])

        assert per_vendor, "no partitions"
        for vendor, numbers in per_vendor.items():
            assert sorted(numbers) == list(range(1, len(numbers) + 1)), (
                f"vendor {vendor} was not numbered 1..{len(numbers)}"
            )

    def test_the_partitions_cover_every_row(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small)
        numbered = frame.select(F.row_number().over(_by_vendor()).alias("rn"))
        assert numbered.count() == frame.count()

    def test_ntile_splits_a_partition_into_balanced_buckets(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(F.col("VendorID"), F.ntile(4).over(_by_vendor()).alias("q"))
            .collect()
        )
        assert {row["q"] for row in rows} == {1, 2, 3, 4}
        for vendor in {row["VendorID"] for row in rows}:
            sizes = Counter(row["q"] for row in rows if row["VendorID"] == vendor)
            if sum(sizes.values()) >= 4:
                assert max(sizes.values()) - min(sizes.values()) <= 1, sizes


class TestRankingWithRealTies:
    """Real timestamps repeat, so ranking has ties without anyone arranging them."""

    def test_the_ordering_column_really_does_have_ties(
        self, it_session: Session, trips_small: str
    ) -> None:
        """The premise of this whole class, asserted rather than assumed."""
        frame = it_session.table(trips_small)
        distinct = frame.select("tpep_pickup_datetime").distinct().count()
        assert distinct < frame.count(), "no repeated timestamps, so ties prove nothing"

    def test_rank_skips_after_a_tie_and_dense_rank_does_not(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.rank().over(_by_vendor()).alias("r"),
                F.dense_rank().over(_by_vendor()).alias("d"),
                F.row_number().over(_by_vendor()).alias("rn"),
            )
            .collect()
        )
        ranks = [row["r"] for row in rows]
        dense = [row["d"] for row in rows]
        assert max(ranks) > max(dense), (
            "rank never skipped a value, which means the ties were not seen"
        )
        assert all(row["d"] <= row["r"] <= row["rn"] for row in rows)

    def test_dense_rank_counts_the_distinct_keys(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("VendorID"),
                F.dense_rank().over(_by_vendor()).alias("d"),
            )
            .collect()
        )
        for vendor in {row["VendorID"] for row in rows}:
            highest = max(row["d"] for row in rows if row["VendorID"] == vendor)
            distinct = (
                it_session.table(trips_small)
                .filter(F.col("VendorID") == vendor)
                .select("tpep_pickup_datetime")
                .distinct()
                .count()
            )
            assert highest == distinct

    def test_percent_rank_and_cume_dist_stay_in_range(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.percent_rank().over(_by_vendor()).alias("p"),
                F.cume_dist().over(_by_vendor()).alias("c"),
            )
            .collect()
        )
        assert rows
        assert all(0.0 <= row["p"] <= 1.0 for row in rows)
        assert all(0.0 < row["c"] <= 1.0 for row in rows)


class TestRunningAggregates:
    """A running total has to end at the partition total. That is the whole test."""

    def test_a_running_sum_ends_at_the_partition_total(
        self, it_session: Session, trips_small: str
    ) -> None:
        running = (
            Window.partitionBy("VendorID")
            .orderBy("tpep_pickup_datetime")
            .rowsBetween(Window.unboundedPreceding, Window.currentRow)
        )
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("VendorID"),
                F.row_number().over(_by_vendor()).alias("rn"),
                F.sum(F.col("trip_distance")).over(running).alias("running"),
            )
            .collect()
        )
        totals = {
            row["VendorID"]: row["t"]
            for row in it_session.table(trips_small)
            .groupBy("VendorID")
            .agg(F.sum(F.col("trip_distance")).alias("t"))
            .collect()
        }
        for vendor, expected in totals.items():
            last = max((row for row in rows if row["VendorID"] == vendor), key=lambda r: r["rn"])
            assert last["running"] == pytest.approx(expected, rel=1e-9)

    def test_an_unpartitioned_window_covers_the_whole_table(
        self, it_session: Session, trips_small: str
    ) -> None:
        whole = Window.orderBy("tpep_pickup_datetime").rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )
        rows = (
            it_session.table(trips_small)
            .select(F.sum(F.col("trip_distance")).over(whole).alias("t"))
            .limit(5)
            .collect()
        )
        expected = (
            it_session.table(trips_small)
            .select(F.sum(F.col("trip_distance")).alias("t"))
            .collect()[0]["t"]
        )
        assert all(row["t"] == pytest.approx(expected, rel=1e-9) for row in rows)

    def test_a_windowed_count_matches_the_partition_size(
        self, it_session: Session, trips_small: str
    ) -> None:
        whole_partition = Window.partitionBy("VendorID").rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )
        rows = (
            it_session.table(trips_small)
            .select(F.col("VendorID"), F.count(F.lit(1)).over(whole_partition).alias("n"))
            .distinct()
            .collect()
        )
        expected = {
            row["VendorID"]: row["n"]
            for row in it_session.table(trips_small)
            .groupBy("VendorID")
            .agg(F.count(F.lit(1)).alias("n"))
            .collect()
        }
        assert {row["VendorID"]: row["n"] for row in rows} == expected


class TestOffsetFunctions:
    """`lag`, `lead`, `first_value`, `last_value` over a real ordering."""

    def test_lag_of_the_next_row_is_the_current_one(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("VendorID"),
                F.row_number().over(_by_vendor()).alias("rn"),
                F.col("trip_distance").alias("d"),
                F.lag(F.col("trip_distance")).over(_by_vendor()).alias("prev"),
            )
            .collect()
        )
        by_key = {(row["VendorID"], row["rn"]): row for row in rows}
        checked = 0
        for (vendor, rn), row in by_key.items():
            earlier = by_key.get((vendor, rn - 1))
            if earlier is not None:
                assert row["prev"] == earlier["d"]
                checked += 1
        assert checked > 0

    def test_the_first_row_of_a_partition_has_no_lag(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.row_number().over(_by_vendor()).alias("rn"),
                F.lag(F.col("trip_distance")).over(_by_vendor()).alias("prev"),
            )
            .filter(F.col("rn") == 1)
            .collect()
        )
        assert rows
        assert all(row["prev"] is None for row in rows)

    def test_lead_is_lag_from_the_other_direction(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("VendorID"),
                F.row_number().over(_by_vendor()).alias("rn"),
                F.col("trip_distance").alias("d"),
                F.lead(F.col("trip_distance")).over(_by_vendor()).alias("nxt"),
            )
            .collect()
        )
        by_key = {(row["VendorID"], row["rn"]): row for row in rows}
        checked = 0
        for (vendor, rn), row in by_key.items():
            later = by_key.get((vendor, rn + 1))
            if later is not None:
                assert row["nxt"] == later["d"]
                checked += 1
        assert checked > 0

    def test_first_value_is_constant_within_a_partition(
        self, it_session: Session, trips_small: str
    ) -> None:
        whole_partition = (
            Window.partitionBy("VendorID")
            .orderBy("tpep_pickup_datetime")
            .rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        )
        rows = (
            it_session.table(trips_small)
            .select(
                F.col("VendorID"),
                F.first(F.col("trip_distance")).over(whole_partition).alias("f"),
            )
            .distinct()
            .collect()
        )
        vendors = [row["VendorID"] for row in rows]
        assert len(vendors) == len(set(vendors)), "first_value varied within a partition"


class TestFrameSemanticsWithRealTies:
    """`ROWS` and `RANGE` differ exactly when the ordering column has peers."""

    def test_a_range_frame_includes_every_peer_and_a_rows_frame_does_not(
        self, it_session: Session, trips_small: str
    ) -> None:
        """The distinction the local suite needed a hand-built fixture to show.

        `RANGE ... CURRENT ROW` includes every row with the same ordering value; `ROWS
        ... CURRENT ROW` stops at this one. Real pickup timestamps repeat, so on some
        row the two must disagree.
        """
        order = Window.partitionBy("VendorID").orderBy("tpep_pickup_datetime")
        by_range = order.rangeBetween(Window.unboundedPreceding, Window.currentRow)
        by_rows = order.rowsBetween(Window.unboundedPreceding, Window.currentRow)

        rows = (
            it_session.table(trips_small)
            .select(
                F.count(F.lit(1)).over(by_range).alias("in_range"),
                F.count(F.lit(1)).over(by_rows).alias("in_rows"),
            )
            .collect()
        )
        assert rows
        assert all(row["in_range"] >= row["in_rows"] for row in rows)
        assert any(row["in_range"] > row["in_rows"] for row in rows), (
            "no row saw more peers under RANGE than under ROWS, so the two frames "
            "were not distinguished -- check the ordering column really has ties"
        )

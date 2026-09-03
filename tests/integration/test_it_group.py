"""`groupBy().agg()`, rollup, cube and pivot, over real cardinality.

The local fixture groups five rows into three buckets, which is enough to prove the SQL
is built correctly and not much else. Here the same operations run over 880,374 trips
across 3 vendors and 253 pickup zones, where the failures that matter show up:

  * a grouped aggregate whose parts do not sum to the whole,
  * a NULL grouping key that silently becomes its own bucket or silently vanishes,
  * a rollup whose grand-total row disagrees with the ungrouped aggregate.

Every count assertion here is a **conservation law** -- the groups must account for
every row -- rather than a number copied out of a previous run. That is what keeps it
meaningful when the seed window changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import agree_across_surfaces, column

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestTheGroupsAccountForEveryRow:
    """The conservation law. If the parts do not sum to the whole, something was lost."""

    def test_grouped_counts_sum_to_the_row_count(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips)
        per_vendor = frame.groupBy("VendorID").agg(F.count(F.lit(1)).alias("n"))
        assert sum(column(per_vendor, "n")) == frame.count()

    def test_grouped_sums_add_up_to_the_ungrouped_sum(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips)
        total = frame.select(F.sum(F.col("trip_distance")).alias("t")).collect()[0]["t"]
        parts = column(frame.groupBy("VendorID").agg(F.sum(F.col("trip_distance")).alias("t")), "t")
        assert sum(parts) == pytest.approx(total, rel=1e-9)

    def test_grouping_by_a_high_cardinality_key_still_conserves(
        self, it_session: Session, trips: str
    ) -> None:
        """253 real groups rather than three."""
        frame = it_session.table(trips)
        per_zone = frame.groupBy("PULocationID").agg(F.count(F.lit(1)).alias("n"))
        assert per_zone.count() == frame.select("PULocationID").distinct().count()
        assert sum(column(per_zone, "n")) == frame.count()

    def test_a_two_key_grouping_conserves_too(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips)
        grouped = frame.groupBy("VendorID", "PULocationID").agg(F.count(F.lit(1)).alias("n"))
        assert sum(column(grouped, "n")) == frame.count()


class TestNullKeys:
    """A NULL grouping key is a bucket, not a dropped row -- and the data has them."""

    def test_a_null_key_becomes_its_own_group(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small)
        nulls = frame.filter(F.col("store_and_fwd_flag").isNull()).count()
        assert nulls > 0, "no NULL keys in the slice, so this proves nothing"

        grouped = frame.groupBy("store_and_fwd_flag").agg(F.count(F.lit(1)).alias("n"))
        rows = grouped.collect()
        null_rows = [row for row in rows if row["store_and_fwd_flag"] is None]
        assert len(null_rows) == 1, "the NULL key did not form exactly one group"
        assert null_rows[0]["n"] == nulls

    def test_the_null_group_is_counted_in_the_total(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        grouped = frame.groupBy("store_and_fwd_flag").agg(F.count(F.lit(1)).alias("n"))
        assert sum(column(grouped, "n")) == frame.count()

    def test_an_aggregate_over_a_null_column_skips_the_nulls(
        self, it_session: Session, trips_small: str
    ) -> None:
        """`count(col)` is not `count(*)`, and the difference is exactly the NULLs."""
        frame = it_session.table(trips_small)
        row = frame.select(
            F.count(F.lit(1)).alias("all_rows"),
            F.count(F.col("passenger_count")).alias("present"),
        ).collect()[0]
        nulls = frame.filter(F.col("passenger_count").isNull()).count()
        assert row["all_rows"] - row["present"] == nulls


class TestTheAggregateShorthands:
    """`groupBy().count()`, `.sum()`, `.avg()`, `.min()`, `.max()`."""

    def test_count_shorthand_matches_the_explicit_aggregate(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips)
        short = {
            row["VendorID"]: row["count"] for row in frame.groupBy("VendorID").count().collect()
        }
        explicit = {
            row["VendorID"]: row["n"]
            for row in frame.groupBy("VendorID").agg(F.count(F.lit(1)).alias("n")).collect()
        }
        assert short == explicit

    def test_min_and_max_bracket_the_average(self, it_session: Session, trips: str) -> None:
        rows = (
            it_session.table(trips)
            .groupBy("VendorID")
            .agg(
                F.min(F.col("trip_distance")).alias("lo"),
                F.avg(F.col("trip_distance")).alias("mean"),
                F.max(F.col("trip_distance")).alias("hi"),
            )
            .collect()
        )
        assert rows
        for row in rows:
            assert row["lo"] <= row["mean"] <= row["hi"]

    def test_sum_shorthand_matches_the_explicit_one(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips)
        short = frame.groupBy("VendorID").sum("trip_distance").collect()
        explicit = frame.groupBy("VendorID").agg(F.sum(F.col("trip_distance")).alias("s")).collect()
        by_vendor = {row["VendorID"]: row["s"] for row in explicit}
        for row in short:
            value = next(v for k, v in row.asDict().items() if k != "VendorID")
            assert value == pytest.approx(by_vendor[row["VendorID"]], rel=1e-9)


class TestRollupAndCube:
    """The grand-total row has to agree with the ungrouped aggregate."""

    def test_a_rollups_total_row_matches_the_whole_table(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips)
        rows = (
            frame.rollup("VendorID")
            .agg(F.count(F.lit(1)).alias("n"), F.grouping("VendorID").alias("g"))
            .collect()
        )
        total = [row for row in rows if row["g"] == 1]
        assert len(total) == 1
        assert total[0]["n"] == frame.count()

    def test_a_rollups_detail_rows_match_the_plain_grouping(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips)
        rolled = {
            row["VendorID"]: row["n"]
            for row in frame.rollup("VendorID")
            .agg(F.count(F.lit(1)).alias("n"), F.grouping("VendorID").alias("g"))
            .collect()
            if row["g"] == 0
        }
        plain = {
            row["VendorID"]: row["n"]
            for row in frame.groupBy("VendorID").agg(F.count(F.lit(1)).alias("n")).collect()
        }
        assert rolled == plain

    def test_a_cube_covers_every_combination(self, it_session: Session, trips: str) -> None:
        """Two keys give four grouping levels: both, each alone, and neither."""
        frame = it_session.table(trips)
        rows = (
            frame.cube("VendorID", "store_and_fwd_flag")
            .agg(
                F.count(F.lit(1)).alias("n"),
                F.grouping("VendorID").alias("gv"),
                F.grouping("store_and_fwd_flag").alias("gs"),
            )
            .collect()
        )
        levels = {(row["gv"], row["gs"]) for row in rows}
        assert levels == {(0, 0), (0, 1), (1, 0), (1, 1)}

        grand = [row for row in rows if (row["gv"], row["gs"]) == (1, 1)]
        assert len(grand) == 1
        assert grand[0]["n"] == frame.count()

    def test_each_cube_level_conserves_the_row_count(self, it_session: Session, trips: str) -> None:
        """Every level is a complete partition of the table, on its own."""
        frame = it_session.table(trips)
        rows = (
            frame.cube("VendorID", "store_and_fwd_flag")
            .agg(
                F.count(F.lit(1)).alias("n"),
                F.grouping("VendorID").alias("gv"),
                F.grouping("store_and_fwd_flag").alias("gs"),
            )
            .collect()
        )
        total = frame.count()
        by_level: dict[tuple[int, int], int] = {}
        for row in rows:
            key = (row["gv"], row["gs"])
            by_level[key] = by_level.get(key, 0) + row["n"]
        for level, counted in by_level.items():
            assert counted == total, f"grouping level {level} lost rows"


class TestPivot:
    """A pivot over the real vendor values."""

    def test_a_pivot_conserves_the_row_count(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips)
        vendors = sorted(column(frame.select("VendorID").distinct(), "VendorID"))
        pivoted = (
            frame.groupBy("store_and_fwd_flag")
            .pivot("VendorID", vendors)
            .agg(F.count(F.lit(1)))
            .collect()
        )
        counted = sum(
            value
            for row in pivoted
            for key, value in row.asDict().items()
            if key != "store_and_fwd_flag" and value is not None
        )
        assert counted == frame.count()

    def test_a_pivot_makes_a_column_per_value(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips)
        vendors = sorted(column(frame.select("VendorID").distinct(), "VendorID"))
        pivoted = (
            frame.groupBy("store_and_fwd_flag").pivot("VendorID", vendors).agg(F.count(F.lit(1)))
        )
        assert len(pivoted.columns) == len(vendors) + 1


class TestBothSurfacesGroupAlike:
    def test_a_grouped_aggregate_agrees(self, it_session: Session, trips: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT VendorID, count(*) AS n, sum(trip_distance) AS d "
            f"FROM {trips} GROUP BY VendorID",
            it_session.table(trips)
            .groupBy("VendorID")
            .agg(F.count(F.lit(1)).alias("n"), F.sum(F.col("trip_distance")).alias("d")),
        )

    def test_a_having_style_filter_agrees(self, it_session: Session, trips: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT PULocationID, count(*) AS n FROM {trips} "
            f"GROUP BY PULocationID HAVING count(*) > 1000",
            it_session.table(trips)
            .groupBy("PULocationID")
            .agg(F.count(F.lit(1)).alias("n"))
            .filter(F.col("n") > 1000),
        )

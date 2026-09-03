"""Joins against the real catalog, on a key that really is in the data.

`icetl_it.zones` holds one row per pickup location that actually occurs in the trip
slice, so `trips ⋈ zones` is a real many-to-one join over 253 distinct keys and 880,374
rows -- not the 1:1 toy join a fixture gives you.

**The class to read first is `TestTheAntiJoinBug`.** Twice now a pushdown has turned an
anti-join into "return everything" (FINDINGS 1.10, then 1.12 -- the same defect reached
through `RIGHT JOIN` instead of `LEFT`). Both were silent: the query succeeded and
returned the wrong rows. The lesson STATUS.md draws from it is that a new pushdown path
has to be tested against *every join spelling*, not one, so that class runs the same
question through every spelling the surface offers and requires one answer.

The other classes assert the ordinary shapes, but on real cardinality -- where a join
that quietly duplicates rows shows up as a row count that moved.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import agree_across_surfaces, column, scans_of

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestTheAntiJoinBug:
    """FINDINGS 1.10 and 1.12, pinned against real data and every join spelling.

    The shape: a left/anti join whose right side matches *everything*. The correct
    answer is no rows. The defect returned every row, and did it without raising.
    """

    def test_an_anti_join_against_a_total_match_is_empty(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        """Every PULocationID in the trips has a zone, so nothing is unmatched."""
        left = it_session.table(trips_small)
        right = it_session.table(zones)
        unmatched = left.join(right, left.PULocationID == right.zone_id, "left_anti")
        assert unmatched.count() == 0

    def test_an_anti_join_against_a_partial_match_keeps_only_the_rest(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        """The other half of the same claim: it must not return *nothing* either."""
        left = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        unmatched = left.join(narrow, left.PULocationID == narrow.zone_id, "left_anti")

        expected = left.filter(~(F.col("PULocationID") < 100)).count()
        assert unmatched.count() == expected
        assert 0 < unmatched.count() < left.count()

    @pytest.mark.parametrize("how", ["left_semi", "leftsemi", "semi"])
    def test_every_semi_join_spelling_agrees(
        self, it_session: Session, trips_small: str, zones: str, how: str
    ) -> None:
        left = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        matched = left.join(narrow, left.PULocationID == narrow.zone_id, how)
        assert matched.count() == left.filter(F.col("PULocationID") < 100).count()

    @pytest.mark.parametrize("how", ["left_anti", "leftanti", "anti"])
    def test_every_anti_join_spelling_agrees(
        self, it_session: Session, trips_small: str, zones: str, how: str
    ) -> None:
        left = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        unmatched = left.join(narrow, left.PULocationID == narrow.zone_id, how)
        assert unmatched.count() == left.filter(~(F.col("PULocationID") < 100)).count()

    def test_semi_and_anti_partition_the_left_side(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        """The invariant that would have caught both defects on its own.

        Whatever the right side is, every left row is either matched or unmatched --
        so the two counts must sum to the left row count. The 1.10 defect made the
        anti side equal the whole table, which breaks this immediately.
        """
        left = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        condition = left.PULocationID == narrow.zone_id
        semi = left.join(narrow, condition, "left_semi").count()
        anti = left.join(narrow, condition, "left_anti").count()
        assert semi + anti == left.count()
        assert semi > 0 and anti > 0, "the split is degenerate, so it proves nothing"

    def test_the_right_spelling_reaches_the_same_answer(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        """FINDINGS 1.12 exactly: the same bug, through `RIGHT JOIN`.

        `a RIGHT JOIN b` and `b LEFT JOIN a` are the same relation. A pruning rule
        that only understands one of the two spellings gives different answers to the
        same question, which is what happened.
        """
        trips = it_session.table(trips_small)
        zone = it_session.table(zones)

        via_right = trips.join(zone, trips.PULocationID == zone.zone_id, "right").count()
        via_left = zone.join(trips, zone.zone_id == trips.PULocationID, "left").count()
        assert via_right == via_left

    def test_no_join_spelling_loses_the_null_padded_rows(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        """Pruning is only safe under `is_null_rejecting`, and this is why.

        On a left join the unmatched rows are null-padded. A predicate pushed into
        the right side that does not reject NULLs would drop them -- silently.
        """
        trips = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        outer = trips.join(narrow, trips.PULocationID == narrow.zone_id, "left")
        assert outer.count() == trips.count()
        padded = outer.filter(F.col("zone_name").isNull()).count()
        assert padded == trips.filter(~(F.col("PULocationID") < 100)).count()


class TestTheOrdinaryShapes:
    """Every join type, over real cardinality."""

    def test_an_inner_join_keeps_every_matching_row(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        trips = it_session.table(trips_small)
        zone = it_session.table(zones)
        joined = trips.join(zone, trips.PULocationID == zone.zone_id, "inner")
        # `zones` has one row per key, so a many-to-one join must not change the count.
        assert joined.count() == trips.count()

    def test_a_left_join_never_loses_a_row(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        trips = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        joined = trips.join(narrow, trips.PULocationID == narrow.zone_id, "left")
        assert joined.count() == trips.count()

    def test_a_full_outer_join_covers_both_sides(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        trips = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        joined = trips.join(narrow, trips.PULocationID == narrow.zone_id, "full")
        assert joined.count() >= trips.count()

    def test_the_joined_column_carries_the_right_value(
        self, it_session: Session, trips_small: str, zones: str
    ) -> None:
        """Not just the right number of rows -- the right data in them."""
        trips = it_session.table(trips_small)
        zone = it_session.table(zones)
        rows = (
            trips.join(zone, trips.PULocationID == zone.zone_id, "inner")
            .select("PULocationID", "zone_name")
            .limit(200)
            .collect()
        )
        assert rows
        for row in rows:
            assert row["zone_name"] == f"zone-{row['PULocationID']}"

    def test_a_cross_join_multiplies(self, it_session: Session, plain: str) -> None:
        """On the replica, because a cross join of real tables is 222 million rows."""
        left = it_session.table(plain)
        right = it_session.table(plain).select(F.col("id").alias("other"))
        assert left.crossJoin(right).count() == left.count() * right.count()

    def test_a_self_join_reads_the_table_twice(self, it_session: Session, trips_small: str) -> None:
        a = it_session.table(trips_small).select(
            F.col("PULocationID").alias("k"), F.col("trip_distance").alias("d")
        )
        b = it_session.table(trips_small).select(F.col("PULocationID").alias("k2"))
        joined = a.join(b, F.col("k") == F.col("k2"), "inner")
        assert joined.count() > 0


class TestBothSurfacesJoinAlike:
    """P1 over joins, which is where a plan rewrite is most likely to diverge."""

    def test_an_inner_join_agrees(self, it_session: Session, trips_small: str, zones: str) -> None:
        trips = it_session.table(trips_small)
        zone = it_session.table(zones)
        agree_across_surfaces(
            it_session,
            f"SELECT z.zone_name AS zone_name, count(*) AS n "
            f"FROM {trips_small} t JOIN {zones} z ON t.PULocationID = z.zone_id "
            f"GROUP BY z.zone_name",
            trips.join(zone, trips.PULocationID == zone.zone_id, "inner")
            .groupBy("zone_name")
            .agg(F.count(F.lit(1)).alias("n")),
        )

    def test_an_anti_join_agrees(self, it_session: Session, trips_small: str, zones: str) -> None:
        trips = it_session.table(trips_small)
        narrow = it_session.table(zones).filter(F.col("zone_id") < 100)
        via_sql = it_session.sql(
            f"SELECT count(*) AS n FROM {trips_small} t "
            f"WHERE NOT EXISTS ("
            f"  SELECT 1 FROM {zones} z WHERE z.zone_id < 100 AND z.zone_id = t.PULocationID)"
        ).collect()[0]["n"]
        via_frame = trips.join(narrow, trips.PULocationID == narrow.zone_id, "left_anti").count()
        assert via_sql == via_frame


class TestPushdownSurvivesTheJoin:
    """A join must not cost the pruning either side had on its own (FINDINGS 3.5)."""

    def test_a_filter_on_one_side_still_prunes_its_files(
        self, it_session: Session, trips: str, zones: str
    ) -> None:
        left = it_session.table(trips).filter(F.col("VendorID") == 1)
        right = it_session.table(zones)
        joined = left.join(right, left.PULocationID == right.zone_id, "inner")

        trip_scan = next(scan for scan in scans_of(joined) if scan.source.key.endswith("trips"))
        assert trip_scan.files_total is not None
        assert trip_scan.files_scanned < trip_scan.files_total, (
            "the VendorID filter stopped pruning once the frame was joined"
        )

    def test_the_join_still_returns_the_filtered_rows(
        self, it_session: Session, trips: str, zones: str
    ) -> None:
        """Pruning must change speed, not answers -- restated for the join path."""
        left = it_session.table(trips).filter(F.col("VendorID") == 1)
        right = it_session.table(zones)
        joined = left.join(right, left.PULocationID == right.zone_id, "inner")
        assert joined.count() == left.count()
        assert set(column(joined.select("VendorID").distinct(), "VendorID")) == {1}

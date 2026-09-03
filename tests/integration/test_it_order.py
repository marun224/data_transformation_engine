"""Ordering, distinct and de-duplication, over real values.

Ordering is the place the conformance layer earns its keep: DuckDB sorts NULLs **last**
ascending, the reference sorts them **first**, and a real column that is NULL 971 times
in 5,000 rows makes the difference visible instead of theoretical.

Every ordering assertion goes through `assert_sorted`, which checks the property rather
than a golden list -- the sequence is sorted, and the NULLs are at the end the reference
puts them at. That keeps working when the seed window moves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import agree_across_surfaces, assert_sorted, column

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestNullPlacement:
    """The divergence the conformance layer exists to preserve."""

    def test_ascending_puts_nulls_first(self, it_session: Session, trips_small: str) -> None:
        """DuckDB would put them last. The reference puts them first, so icetl does."""
        values = column(
            it_session.table(trips_small)
            .orderBy("store_and_fwd_flag")
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        assert None in values, "the column has no NULLs, so this proves nothing"
        assert_sorted(values, ascending=True, nulls_first=True)

    def test_descending_puts_nulls_last(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small)
            .orderBy(F.col("store_and_fwd_flag").desc())
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        assert_sorted(values, ascending=False, nulls_first=False)

    def test_the_ascending_flag_spells_the_same_thing(
        self, it_session: Session, trips_small: str
    ) -> None:
        by_flag = column(
            it_session.table(trips_small)
            .orderBy("store_and_fwd_flag", ascending=False)
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        by_method = column(
            it_session.table(trips_small)
            .orderBy(F.col("store_and_fwd_flag").desc())
            .select("store_and_fwd_flag"),
            "store_and_fwd_flag",
        )
        assert by_flag == by_method


class TestOrderingRealValues:
    def test_a_numeric_order_is_sorted(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small).orderBy("trip_distance").select("trip_distance"),
            "trip_distance",
        )
        assert_sorted(values, ascending=True)

    def test_a_timestamp_order_is_sorted(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small)
            .orderBy("tpep_pickup_datetime")
            .select("tpep_pickup_datetime"),
            "tpep_pickup_datetime",
        )
        assert_sorted(values, ascending=True)

    def test_ordering_does_not_change_the_row_count(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        assert frame.orderBy("trip_distance").count() == frame.count()

    def test_a_two_key_order_breaks_ties_by_the_second(
        self, it_session: Session, trips_small: str
    ) -> None:
        rows = (
            it_session.table(trips_small)
            .orderBy("VendorID", "trip_distance")
            .select("VendorID", "trip_distance")
            .collect()
        )
        pairs = [(row["VendorID"], row["trip_distance"]) for row in rows]
        assert pairs == sorted(pairs)

    def test_limit_after_order_takes_the_smallest(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        smallest = column(
            frame.orderBy("trip_distance").limit(10).select("trip_distance"), "trip_distance"
        )
        overall = frame.select(F.min(F.col("trip_distance")).alias("m")).collect()[0]["m"]
        assert len(smallest) == 10
        assert smallest[0] == overall
        assert_sorted(smallest, ascending=True)

    def test_ordering_by_an_output_alias_works(self, it_session: Session, trips: str) -> None:
        """FINDINGS 3.1: this spelling once disabled projection pushdown entirely.

        Here it is asserted for its *answer*; `test_it_pushdown.py` asserts that it
        still prunes.
        """
        rows = (
            it_session.table(trips)
            .groupBy("VendorID")
            .agg(F.count(F.lit(1)).alias("trips_taken"))
            .orderBy("trips_taken")
            .collect()
        )
        counts = [row["trips_taken"] for row in rows]
        assert counts == sorted(counts)


class TestDistinctAndDeduplication:
    def test_distinct_removes_exactly_the_duplicates(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        distinct = frame.distinct()
        assert distinct.count() == len(set(column(frame, "PULocationID")))
        assert distinct.count() < frame.count(), "no duplicates, so this proves nothing"

    def test_distinct_keeps_one_null(self, it_session: Session, trips_small: str) -> None:
        values = column(
            it_session.table(trips_small).select("store_and_fwd_flag").distinct(),
            "store_and_fwd_flag",
        )
        assert values.count(None) == 1

    def test_drop_duplicates_on_a_subset_keeps_one_row_per_key(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID", "trip_distance")
        deduped = frame.dropDuplicates(["PULocationID"])
        keys = column(deduped, "PULocationID")
        assert len(keys) == len(set(keys))
        assert set(keys) == set(column(frame, "PULocationID"))

    def test_drop_duplicates_with_no_subset_is_distinct(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("VendorID", "store_and_fwd_flag")
        assert frame.dropDuplicates().count() == frame.distinct().count()


class TestLocalData:
    """`take`, `head`, `first`, `toLocalIterator` -- the row-at-a-time surface."""

    def test_take_and_head_agree(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small).orderBy("tpep_pickup_datetime")
        assert frame.take(3) == frame.head(3)

    def test_first_is_the_first_of_take(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small).orderBy("tpep_pickup_datetime")
        assert frame.first() == frame.take(1)[0]

    def test_the_local_iterator_yields_every_row(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        streamed = sum(1 for _ in frame.toLocalIterator())
        assert streamed == frame.count()

    def test_the_local_iterator_yields_the_same_rows_as_collect(
        self, it_session: Session, zones: str
    ) -> None:
        frame = it_session.table(zones).orderBy("zone_id")
        assert list(frame.toLocalIterator()) == frame.collect()


class TestBothSurfacesOrderAlike:
    def test_an_ordered_projection_agrees(self, it_session: Session, zones: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT zone_id, zone_name FROM {zones} ORDER BY zone_id",
            it_session.table(zones).select("zone_id", "zone_name").orderBy("zone_id"),
        )

    def test_a_distinct_agrees(self, it_session: Session, trips_small: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT DISTINCT store_and_fwd_flag FROM {trips_small}",
            it_session.table(trips_small).select("store_and_fwd_flag").distinct(),
        )

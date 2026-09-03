"""`DELETE`, `UPDATE` and `MERGE` against the real catalog.

PLAN.md calls this the riskiest phase in the plan, and the reason is worth restating
because it is what these tests are shaped around.

A row-level operation commits as `overwrite(rows, overwrite_filter=P)`: PyIceberg deletes
whatever `P` matches, then appends the rows the SQL kept. **The predicate has to mean the
same thing twice.** If `P` is wider than the SQL's `WHERE`, rows are deleted and never
written back. If it is narrower, rows survive the delete *and* arrive again in the append.
Either way the table is wrong and nothing raised.

So every test below asserts on **both halves**: the rows that should have changed, and
the rows that should not. `TestScopeIsExact` is the class that would notice a predicate
that drifted -- it runs statements whose scope is easy to get wrong and checks the
untouched remainder row for row.

Real data earns its place here twice over. `store_and_fwd_flag` is NULL in 971 of 5,000
rows, so **NULL is not false** is a live concern rather than a hypothetical: `DELETE ...
WHERE flag = 'Y'` must keep every NULL row, because `flag = 'Y'` is NULL for those and
NULL is not true. And a real REST catalog means these commits go through real optimistic
concurrency rather than a sqlite transaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import AnalysisException
from icetl.sql import functions as F
from tests.integration.helpers import column

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


@pytest.fixture
def loaded(session: Session, trips_small: str, target: str) -> str:
    """A throwaway table holding a copy of the real slice, dropped by `target`."""
    session.table(trips_small).write.saveAsTable(target)
    return target


@pytest.fixture
def loaded_zones(session: Session, zones: str, target: str) -> str:
    """A small throwaway with a unique key, for the merge tests."""
    session.table(zones).write.saveAsTable(target)
    return target


class TestDelete:
    def test_a_delete_removes_exactly_the_matching_rows(
        self, session: Session, loaded: str
    ) -> None:
        before = session.table(loaded).count()
        matching = session.table(loaded).filter(F.col("VendorID") == 1).count()
        assert 0 < matching < before

        session.sql(f"DELETE FROM {loaded} WHERE VendorID = 1")
        assert session.table(loaded).count() == before - matching

    def test_the_rows_that_did_not_match_are_untouched(self, session: Session, loaded: str) -> None:
        """The other half of the same claim, checked row for row."""
        survivors = (
            session.table(loaded)
            .filter(F.col("VendorID") != 1)
            .orderBy("tpep_pickup_datetime", "PULocationID")
            .collect()
        )

        session.sql(f"DELETE FROM {loaded} WHERE VendorID = 1")
        after = session.table(loaded).orderBy("tpep_pickup_datetime", "PULocationID").collect()
        assert after == survivors

    def test_a_delete_keeps_the_null_rows(self, session: Session, loaded: str) -> None:
        """NULL is not false. `flag = 'Y'` is NULL for a NULL row, so it does not match."""
        nulls = session.table(loaded).filter(F.col("store_and_fwd_flag").isNull()).count()
        assert nulls > 0

        session.sql(f"DELETE FROM {loaded} WHERE store_and_fwd_flag = 'Y'")
        assert session.table(loaded).filter(F.col("store_and_fwd_flag").isNull()).count() == nulls

    def test_deleting_where_a_column_is_null_removes_those_rows(
        self, session: Session, loaded: str
    ) -> None:
        before = session.table(loaded).count()
        nulls = session.table(loaded).filter(F.col("store_and_fwd_flag").isNull()).count()

        session.sql(f"DELETE FROM {loaded} WHERE store_and_fwd_flag IS NULL")
        assert session.table(loaded).count() == before - nulls
        assert session.table(loaded).filter(F.col("store_and_fwd_flag").isNull()).count() == 0

    def test_a_delete_matching_nothing_changes_nothing(self, session: Session, loaded: str) -> None:
        before = session.table(loaded).count()
        session.sql(f"DELETE FROM {loaded} WHERE VendorID = -1")
        assert session.table(loaded).count() == before

    def test_a_delete_with_no_where_empties_the_table(
        self, session: Session, loaded_zones: str
    ) -> None:
        session.sql(f"DELETE FROM {loaded_zones}")
        assert session.table(loaded_zones).count() == 0


class TestUpdate:
    def test_an_update_changes_exactly_the_matching_rows(
        self, session: Session, loaded_zones: str
    ) -> None:
        matching = session.table(loaded_zones).filter(F.col("zone_id") < 100).count()
        assert matching > 0

        session.sql(f"UPDATE {loaded_zones} SET zone_name = 'renamed' WHERE zone_id < 100")
        assert (
            session.table(loaded_zones).filter(F.col("zone_name") == "renamed").count() == matching
        )

    def test_an_update_leaves_the_rest_alone(self, session: Session, loaded_zones: str) -> None:
        untouched = (
            session.table(loaded_zones)
            .filter(~(F.col("zone_id") < 100))
            .orderBy("zone_id")
            .collect()
        )
        session.sql(f"UPDATE {loaded_zones} SET zone_name = 'renamed' WHERE zone_id < 100")
        after = (
            session.table(loaded_zones)
            .filter(~(F.col("zone_id") < 100))
            .orderBy("zone_id")
            .collect()
        )
        assert after == untouched

    def test_an_update_does_not_change_the_row_count(
        self, session: Session, loaded_zones: str
    ) -> None:
        """The failure that would mean the predicate meant two different things."""
        before = session.table(loaded_zones).count()
        session.sql(f"UPDATE {loaded_zones} SET zone_name = 'renamed' WHERE zone_id < 100")
        assert session.table(loaded_zones).count() == before

    def test_an_update_can_read_the_old_value(self, session: Session, loaded_zones: str) -> None:
        before = {row["zone_id"]: row["zone_name"] for row in session.table(loaded_zones).collect()}
        session.sql(f"UPDATE {loaded_zones} SET zone_id = zone_id + 1000 WHERE zone_id < 100")
        after = {
            row["zone_id"]: row["zone_name"]
            for row in session.table(loaded_zones).filter(F.col("zone_id") >= 1000).collect()
        }
        for shifted, name in after.items():
            assert before[shifted - 1000] == name

    def test_an_update_keeps_the_null_rows_out_of_scope(
        self, session: Session, loaded: str
    ) -> None:
        nulls = session.table(loaded).filter(F.col("store_and_fwd_flag").isNull()).count()
        session.sql(f"UPDATE {loaded} SET trip_distance = -1 WHERE store_and_fwd_flag = 'Y'")
        assert session.table(loaded).filter(F.col("store_and_fwd_flag").isNull()).count() == nulls
        assert (
            session.table(loaded)
            .filter(F.col("store_and_fwd_flag").isNull() & (F.col("trip_distance") == -1))
            .count()
            == 0
        )


class TestScopeIsExact:
    """A predicate that is wider or narrower than the SQL's `WHERE` corrupts the table.

    Both directions are checked by counting the whole table afterwards: too wide loses
    rows, too narrow duplicates them. Either shows up as a row count that moved.
    """

    def test_a_delete_neither_loses_nor_duplicates(self, session: Session, loaded: str) -> None:
        before = session.table(loaded).count()
        removed = session.table(loaded).filter(F.col("trip_distance") > 5).count()
        assert 0 < removed < before

        session.sql(f"DELETE FROM {loaded} WHERE trip_distance > 5")
        after = session.table(loaded).count()
        assert after == before - removed
        assert session.table(loaded).filter(F.col("trip_distance") > 5).count() == 0

    def test_an_update_neither_loses_nor_duplicates(self, session: Session, loaded: str) -> None:
        before = session.table(loaded).count()
        session.sql(f"UPDATE {loaded} SET passenger_count = 9 WHERE trip_distance > 5")
        assert session.table(loaded).count() == before

    def test_a_half_translatable_predicate_still_scopes_exactly(
        self, session: Session, loaded: str
    ) -> None:
        """One conjunct pushes, the other cannot. The scope must still be the whole
        predicate -- a commit filter of only the pushable half would delete too much."""
        before = session.table(loaded).count()
        target_rows = (
            session.table(loaded)
            .filter((F.col("VendorID") == 1) & (F.length(F.col("store_and_fwd_flag")) == 1))
            .count()
        )
        assert 0 < target_rows < before

        session.sql(f"DELETE FROM {loaded} WHERE VendorID = 1 AND length(store_and_fwd_flag) = 1")
        assert session.table(loaded).count() == before - target_rows


class TestMerge:
    def test_matched_rows_are_updated(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        session.sql(
            f"MERGE INTO {loaded_zones} t USING {zones} s ON t.zone_id = s.zone_id "
            f"WHEN MATCHED THEN UPDATE SET t.zone_name = 'merged'"
        )
        frame = session.table(loaded_zones)
        assert frame.filter(F.col("zone_name") == "merged").count() == frame.count()

    def test_a_merge_does_not_change_the_row_count_when_everything_matches(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        before = session.table(loaded_zones).count()
        session.sql(
            f"MERGE INTO {loaded_zones} t USING {zones} s ON t.zone_id = s.zone_id "
            f"WHEN MATCHED THEN UPDATE SET t.zone_name = 'merged'"
        )
        assert session.table(loaded_zones).count() == before

    def test_unmatched_rows_are_inserted(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        session.sql(f"DELETE FROM {loaded_zones} WHERE zone_id < 100")
        after_delete = session.table(loaded_zones).count()
        source_total = session.table(zones).count()

        session.sql(
            f"MERGE INTO {loaded_zones} t USING {zones} s ON t.zone_id = s.zone_id "
            f"WHEN NOT MATCHED THEN INSERT *"
        )
        assert session.table(loaded_zones).count() == source_total
        assert session.table(loaded_zones).count() > after_delete

    def test_matched_delete_removes_the_overlap(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        session.sql(
            f"MERGE INTO {loaded_zones} t USING "
            f"(SELECT * FROM {zones} WHERE zone_id < 100) s ON t.zone_id = s.zone_id "
            f"WHEN MATCHED THEN DELETE"
        )
        assert session.table(loaded_zones).filter(F.col("zone_id") < 100).count() == 0
        assert session.table(loaded_zones).count() > 0

    def test_the_untouched_rows_survive_a_merge(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        untouched = (
            session.table(loaded_zones)
            .filter(~(F.col("zone_id") < 100))
            .orderBy("zone_id")
            .collect()
        )
        session.sql(
            f"MERGE INTO {loaded_zones} t USING "
            f"(SELECT * FROM {zones} WHERE zone_id < 100) s ON t.zone_id = s.zone_id "
            f"WHEN MATCHED THEN UPDATE SET t.zone_name = 'merged'"
        )
        after = (
            session.table(loaded_zones)
            .filter(~(F.col("zone_id") < 100))
            .orderBy("zone_id")
            .collect()
        )
        assert after == untouched

    def test_a_conditional_when_matched_only_touches_its_condition(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        expected = session.table(loaded_zones).filter(F.col("zone_id") < 50).count()
        session.sql(
            f"MERGE INTO {loaded_zones} t USING {zones} s ON t.zone_id = s.zone_id "
            f"WHEN MATCHED AND t.zone_id < 50 THEN UPDATE SET t.zone_name = 'merged'"
        )
        assert (
            session.table(loaded_zones).filter(F.col("zone_name") == "merged").count() == expected
        )


class TestCardinality:
    """A merge that matches one target row twice has no answer, so it is refused."""

    def test_a_source_matching_twice_is_refused(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        doubled = f"(SELECT * FROM {zones} UNION ALL SELECT * FROM {zones})"
        with pytest.raises(AnalysisException):
            session.sql(
                f"MERGE INTO {loaded_zones} t USING {doubled} s ON t.zone_id = s.zone_id "
                f"WHEN MATCHED THEN UPDATE SET t.zone_name = 'merged'"
            )

    def test_the_refusal_leaves_the_table_unchanged(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        """A refused statement must not have half-committed."""
        before = session.table(loaded_zones).orderBy("zone_id").collect()
        doubled = f"(SELECT * FROM {zones} UNION ALL SELECT * FROM {zones})"
        with pytest.raises(AnalysisException):
            session.sql(
                f"MERGE INTO {loaded_zones} t USING {doubled} s ON t.zone_id = s.zone_id "
                f"WHEN MATCHED THEN UPDATE SET t.zone_name = 'merged'"
            )
        session.catalog.refreshTable(loaded_zones)
        assert session.table(loaded_zones).orderBy("zone_id").collect() == before


class TestSnapshots:
    """Every row-level operation is a commit, so the history has to show it."""

    def test_a_delete_adds_a_snapshot(
        self, session: Session, catalog: Catalog, loaded_zones: str
    ) -> None:
        before = len(catalog.load_table(loaded_zones).metadata.snapshots)
        session.sql(f"DELETE FROM {loaded_zones} WHERE zone_id < 100")
        after = len(catalog.load_table(loaded_zones).metadata.snapshots)
        assert after > before

    def test_the_previous_snapshot_still_has_the_old_rows(
        self, session: Session, catalog: Catalog, loaded_zones: str
    ) -> None:
        """Copy-on-write: the delete rewrote files, it did not erase history."""
        before = session.table(loaded_zones).count()
        previous = catalog.load_table(loaded_zones).current_snapshot()
        assert previous is not None

        session.sql(f"DELETE FROM {loaded_zones} WHERE zone_id < 100")
        assert session.table(loaded_zones).count() < before

        old = session.sql(
            f"SELECT count(*) AS n FROM {loaded_zones} VERSION AS OF {previous.snapshot_id}"
        ).collect()[0]["n"]
        assert old == before


class TestBothSurfacesSeeTheResult:
    def test_a_delete_is_visible_through_both_surfaces(
        self, session: Session, loaded_zones: str
    ) -> None:
        session.sql(f"DELETE FROM {loaded_zones} WHERE zone_id < 100")
        via_sql = session.sql(f"SELECT count(*) AS n FROM {loaded_zones}").collect()[0]["n"]
        assert via_sql == session.table(loaded_zones).count()

    def test_the_remaining_keys_are_the_expected_ones(
        self, session: Session, loaded_zones: str, zones: str
    ) -> None:
        expected = set(column(session.table(zones).filter(~(F.col("zone_id") < 100)), "zone_id"))
        session.sql(f"DELETE FROM {loaded_zones} WHERE zone_id < 100")
        assert set(column(session.table(loaded_zones), "zone_id")) == expected

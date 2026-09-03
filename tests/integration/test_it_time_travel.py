"""Time travel and Iceberg's metadata tables, over history somebody else made.

`nyc.yellow_tripdata` carries **12 real APPEND snapshots**, each adding roughly three
million rows, committed seconds apart. That is history the tests did not create, which
makes it better material than a fixture: the snapshot ids, the timestamps and the row
counts are all facts about the warehouse, and every expected value below is read out of
the catalog in the test rather than written down.

The one thing worth stating plainly: **time travel is a property of the source key**, not
a filter applied afterwards. `t VERSION AS OF x` and `t` are different sources, so they
bind different schemas and cache separately. `TestTheCacheDistinguishesThem` is what
would notice if they ever collapsed into one.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest

from icetl.errors import AnalysisException
from tests.integration.conftest import REAL_TABLE

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table.snapshots import Snapshot

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


def _snapshots(catalog: Catalog, identifier: str) -> list[Snapshot]:
    """Every snapshot of a table, oldest first."""
    table = catalog.load_table(identifier)
    return sorted(table.metadata.snapshots, key=lambda s: s.timestamp_ms)


class TestTheHistoryIsReal:
    """The premise of the rest of the module, asserted rather than assumed."""

    def test_the_real_table_has_several_snapshots(self, catalog: Catalog) -> None:
        history = _snapshots(catalog, REAL_TABLE)
        assert len(history) > 1, "no history to travel through"

    def test_the_snapshots_are_ordered_in_time(self, catalog: Catalog) -> None:
        stamps = [s.timestamp_ms for s in _snapshots(catalog, REAL_TABLE)]
        assert stamps == sorted(stamps)

    def test_the_table_grew_over_its_history(self, catalog: Catalog) -> None:
        """Every snapshot is an APPEND, so the row count is monotonic."""
        counts = [
            int(str(s.summary["total-records"]))
            for s in _snapshots(catalog, REAL_TABLE)
            if s.summary is not None and "total-records" in s.summary
        ]
        assert len(counts) > 1
        assert counts == sorted(counts)


class TestVersionAsOf:
    def test_an_older_snapshot_has_fewer_rows(self, it_session: Session, catalog: Catalog) -> None:
        """The expected number comes from the snapshot summary, not from a literal."""
        history = _snapshots(catalog, REAL_TABLE)
        first = history[0]
        assert first.summary is not None
        expected = int(str(first.summary["total-records"]))

        travelled = it_session.sql(
            f"SELECT count(*) AS n FROM {REAL_TABLE} VERSION AS OF {first.snapshot_id}"
        ).collect()[0]["n"]
        assert travelled == expected
        assert travelled < it_session.table(REAL_TABLE).count()

    def test_the_latest_snapshot_matches_the_untravelled_table(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        latest = _snapshots(catalog, REAL_TABLE)[-1]
        travelled = it_session.sql(
            f"SELECT count(*) AS n FROM {REAL_TABLE} VERSION AS OF {latest.snapshot_id}"
        ).collect()[0]["n"]
        assert travelled == it_session.table(REAL_TABLE).count()

    def test_every_snapshot_reports_the_count_its_summary_claims(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        """Answered from the manifests, so travelling twelve snapshots is cheap."""
        for snapshot in _snapshots(catalog, REAL_TABLE):
            if snapshot.summary is None or "total-records" not in snapshot.summary:
                continue
            counted = it_session.sql(
                f"SELECT count(*) AS n FROM {REAL_TABLE} VERSION AS OF {snapshot.snapshot_id}"
            ).collect()[0]["n"]
            assert counted == int(str(snapshot.summary["total-records"])), snapshot.snapshot_id

    def test_travelling_reads_the_rows_not_just_the_count(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        first = _snapshots(catalog, REAL_TABLE)[0]
        rows = it_session.sql(
            f"SELECT VendorID FROM {REAL_TABLE} VERSION AS OF {first.snapshot_id} LIMIT 5"
        ).collect()
        assert len(rows) == 5

    def test_a_missing_snapshot_is_refused(self, it_session: Session) -> None:
        with pytest.raises(AnalysisException):
            it_session.sql(f"SELECT count(*) AS n FROM {REAL_TABLE} VERSION AS OF 1").collect()


class TestTimestampAsOf:
    def test_travelling_to_a_moment_finds_the_snapshot_of_that_moment(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        """Iceberg resolves a timestamp to the latest snapshot at or before it."""
        history = _snapshots(catalog, REAL_TABLE)
        target = history[0]
        assert target.summary is not None

        # A moment strictly between the first commit and the second, so exactly one
        # snapshot is current.
        moment = datetime.datetime.fromtimestamp(
            (target.timestamp_ms + 1) / 1000, datetime.UTC
        ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        counted = it_session.sql(
            f"SELECT count(*) AS n FROM {REAL_TABLE} TIMESTAMP AS OF '{moment}'"
        ).collect()[0]["n"]
        assert counted == int(str(target.summary["total-records"]))

    def test_travelling_to_now_matches_the_untravelled_table(self, it_session: Session) -> None:
        moment = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
        counted = it_session.sql(
            f"SELECT count(*) AS n FROM {REAL_TABLE} TIMESTAMP AS OF '{moment}'"
        ).collect()[0]["n"]
        assert counted == it_session.table(REAL_TABLE).count()

    def test_travelling_before_the_first_commit_is_refused(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        earliest = _snapshots(catalog, REAL_TABLE)[0]
        before = datetime.datetime.fromtimestamp(
            (earliest.timestamp_ms - 60_000) / 1000, datetime.UTC
        ).strftime("%Y-%m-%d %H:%M:%S")
        with pytest.raises(AnalysisException):
            it_session.sql(
                f"SELECT count(*) AS n FROM {REAL_TABLE} TIMESTAMP AS OF '{before}'"
            ).collect()


class TestTheCacheDistinguishesThem:
    """Time travel is part of the source key, so the two must not share a cache entry."""

    def test_the_travelled_and_untravelled_counts_differ(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        first = _snapshots(catalog, REAL_TABLE)[0]
        travelled = it_session.sql(
            f"SELECT count(*) AS n FROM {REAL_TABLE} VERSION AS OF {first.snapshot_id}"
        ).collect()[0]["n"]
        current = it_session.table(REAL_TABLE).count()
        assert travelled != current

    def test_asking_in_either_order_gives_the_same_two_answers(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        """A shared cache entry would make the second question return the first answer."""
        first = _snapshots(catalog, REAL_TABLE)[0]
        sql = f"SELECT count(*) AS n FROM {REAL_TABLE} VERSION AS OF {first.snapshot_id}"

        a_travelled = it_session.sql(sql).collect()[0]["n"]
        a_current = it_session.table(REAL_TABLE).count()
        b_current = it_session.table(REAL_TABLE).count()
        b_travelled = it_session.sql(sql).collect()[0]["n"]

        assert a_travelled == b_travelled
        assert a_current == b_current


class TestMetadataTables:
    """Iceberg's own metadata, addressed as a suffix on the table name."""

    def test_the_snapshots_table_lists_every_snapshot(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        listed = it_session.table(f"{REAL_TABLE}.snapshots").count()
        assert listed == len(_snapshots(catalog, REAL_TABLE))

    def test_the_history_table_is_readable(self, it_session: Session) -> None:
        assert it_session.table(f"{REAL_TABLE}.history").count() > 0

    def test_the_files_table_counts_the_data_files(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        current = catalog.load_table(REAL_TABLE).current_snapshot()
        assert current is not None and current.summary is not None
        expected = int(str(current.summary["total-data-files"]))
        assert it_session.table(f"{REAL_TABLE}.data_files").count() == expected

    def test_the_manifests_table_is_readable(self, it_session: Session) -> None:
        assert it_session.table(f"{REAL_TABLE}.manifests").count() > 0

    def test_a_metadata_table_can_be_queried_like_any_other(self, it_session: Session) -> None:
        """It is a frame, so the whole surface applies to it."""
        from icetl.sql import functions as F

        rows = (
            it_session.table(f"{REAL_TABLE}.snapshots")
            .select("snapshot_id", "operation")
            .filter(F.col("operation") == "append")
            .collect()
        )
        assert rows
        assert all(row["operation"] == "append" for row in rows)

    def test_both_surfaces_read_the_metadata_table_alike(self, it_session: Session) -> None:
        via_sql = it_session.sql(f"SELECT count(*) AS n FROM {REAL_TABLE}.snapshots").collect()[0][
            "n"
        ]
        assert via_sql == it_session.table(f"{REAL_TABLE}.snapshots").count()


class TestTravellingAThrowawayTable:
    """The same operations against history this suite made, so writes can be checked."""

    def test_an_earlier_snapshot_of_a_written_table_has_the_earlier_rows(
        self, session: Session, catalog: Catalog, zones: str, target: str
    ) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        first_snapshot = catalog.load_table(target).current_snapshot()
        assert first_snapshot is not None
        original = session.table(target).count()

        source.write.mode("append").saveAsTable(target)
        assert session.table(target).count() == original * 2

        travelled = session.sql(
            f"SELECT count(*) AS n FROM {target} VERSION AS OF {first_snapshot.snapshot_id}"
        ).collect()[0]["n"]
        assert travelled == original

    def test_a_deleted_row_is_still_there_in_the_previous_snapshot(
        self, session: Session, catalog: Catalog, zones: str, target: str
    ) -> None:
        session.table(zones).write.saveAsTable(target)
        before = session.table(target).count()
        snapshot = catalog.load_table(target).current_snapshot()
        assert snapshot is not None

        session.sql(f"DELETE FROM {target} WHERE zone_id < 100")
        assert session.table(target).count() < before

        travelled = session.sql(
            f"SELECT count(*) AS n FROM {target} VERSION AS OF {snapshot.snapshot_id}"
        ).collect()[0]["n"]
        assert travelled == before

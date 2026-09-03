"""Phase 11: compaction, snapshot expiry, and orphan-file cleanup on one node.

**Compaction is not a tidiness feature.** PLAN.md 3.6 leans on column-statistics file
pruning for the securities table, and statistics only prune when files are large and
sorted. Many small unsorted files defeat the read design, so this is the job that
keeps it working.

Every test here asserts on **the rows**, not only on the file count. A compaction that
halved the file count and lost a row would satisfy any count-based test, and it is the
single worst thing this module could do. `TestTheRowsSurvive` is the point of the
file; the counts are how you tell it did anything.

`TestDeletingOrphansIsSafe` is the other one worth reading. Removing a file that is
not really an orphan is unrecoverable, so the test does not check that the right files
were deleted — it checks that the table still reads afterwards.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from icetl.errors import EngineTypeError, EngineValueError, UnsupportedFeatureError

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.session import Session

#: Small enough that every fixture file counts as "small" and compaction engages.
SMALL = 10_000_000


def data_files(catalog: SqlCatalog, name: str) -> int:
    table = catalog.load_table(tuple(name.split(".")))
    return len(list(table.scan().plan_files()))


@pytest.fixture
def fragmented(session: Session, catalog: SqlCatalog) -> Iterator[str]:
    """A partitioned table written as six small appends: 6 files, 2 partitions."""
    name = f"wr.frag_{uuid.uuid4().hex[:8]}"
    session.sql(f"CREATE TABLE {name} (id BIGINT, part BIGINT, v STRING) PARTITIONED BY (part)")
    for batch in range(6):
        session.createDataFrame(
            [(batch * 10 + row, batch % 2, f"v{batch}{row}") for row in range(3)],
            ["id", "part", "v"],
        ).write.mode("append").saveAsTable(name)
    yield name
    with contextlib.suppress(Exception):
        catalog.drop_table(tuple(name.split(".")))


@pytest.fixture
def unpartitioned(session: Session, catalog: SqlCatalog) -> Iterator[str]:
    """The same shape with no partitioning: one group, several files."""
    name = f"wr.flat_{uuid.uuid4().hex[:8]}"
    for batch in range(4):
        session.createDataFrame(
            [(batch * 10 + row, f"v{batch}{row}") for row in range(3)], ["id", "v"]
        ).write.mode("append").saveAsTable(name)
    yield name
    with contextlib.suppress(Exception):
        catalog.drop_table(tuple(name.split(".")))


class TestTheRowsSurvive:
    """The only thing compaction may not change."""

    def test_every_row_is_still_there(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        before = sorted(row[0] for row in session.table(fragmented).select("id").collect())
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        after = sorted(row[0] for row in session.table(fragmented).select("id").collect())
        assert before == after == sorted(batch * 10 + row for batch in range(6) for row in range(3))

    def test_every_column_survives_too(self, session: Session, fragmented: str) -> None:
        before = sorted(tuple(row) for row in session.table(fragmented).collect())
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        assert sorted(tuple(row) for row in session.table(fragmented).collect()) == before

    def test_the_count_is_unchanged(self, session: Session, fragmented: str) -> None:
        assert session.table(fragmented).count() == 18
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        assert session.table(fragmented).count() == 18

    def test_rows_land_in_the_partition_they_started_in(
        self, session: Session, fragmented: str
    ) -> None:
        """A compaction that moved rows between partitions would still count right."""
        before = sorted(
            (row[0], row[1]) for row in session.table(fragmented).select("id", "part").collect()
        )
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        after = sorted(
            (row[0], row[1]) for row in session.table(fragmented).select("id", "part").collect()
        )
        assert before == after

    def test_an_unpartitioned_table_compacts(
        self, session: Session, catalog: SqlCatalog, unpartitioned: str
    ) -> None:
        before = sorted(row[0] for row in session.table(unpartitioned).select("id").collect())
        assert data_files(catalog, unpartitioned) == 4
        session.maintenance(unpartitioned).compact(targetFileSizeBytes=SMALL)
        assert data_files(catalog, unpartitioned) == 1
        after = sorted(row[0] for row in session.table(unpartitioned).select("id").collect())
        assert after == before


class TestItActuallyCompacts:
    def test_six_files_become_one_per_partition(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        assert data_files(catalog, fragmented) == 6
        result = session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        assert result.changed
        assert result.partitions_rewritten == 2
        assert data_files(catalog, fragmented) == 2

    def test_the_result_reports_what_it_did(self, session: Session, fragmented: str) -> None:
        result = session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        assert result.files_before == 6
        assert result.files_after == 2
        assert result.rows_rewritten == 18
        assert "6 -> 2 file(s)" in str(result)

    def test_compacting_twice_does_nothing_the_second_time(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        """A run that rewrote everything every time would cost a snapshot for nothing."""
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        settled = data_files(catalog, fragmented)
        again = session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        assert not again.changed
        assert again.skipped
        assert data_files(catalog, fragmented) == settled

    def test_min_input_files_holds_it_back(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        result = session.maintenance(fragmented).compact(
            targetFileSizeBytes=SMALL, minInputFiles=99
        )
        assert not result.changed
        assert data_files(catalog, fragmented) == 6

    def test_a_target_below_every_file_leaves_them_alone(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        """Nothing is 'small' against a target of one byte, so nothing is rewritten."""
        result = session.maintenance(fragmented).compact(targetFileSizeBytes=1)
        assert not result.changed
        assert data_files(catalog, fragmented) == 6

    def test_an_empty_table_is_not_an_error(self, session: Session, catalog: SqlCatalog) -> None:
        name = f"wr.empty_{uuid.uuid4().hex[:8]}"
        session.sql(f"CREATE TABLE {name} (id BIGINT)")
        try:
            result = session.maintenance(name).compact()
            assert not result.changed
            assert result.skipped == ("no data files",)
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_compaction_is_visible_to_the_session_that_did_it(
        self, session: Session, fragmented: str
    ) -> None:
        """A `ScanSource` pins a snapshot, so the commit has to invalidate it."""
        assert session.table(fragmented).count() == 18
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        frame = session.table(fragmented)
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        assert compiled.scans[0].files_scanned == 2

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [({"targetFileSizeBytes": 0}, "positive"), ({"minInputFiles": 0}, "at least 1")],
    )
    def test_nonsense_arguments_are_refused(
        self, session: Session, fragmented: str, kwargs: dict[str, int], message: str
    ) -> None:
        with pytest.raises(EngineValueError, match=message):
            session.maintenance(fragmented).compact(**kwargs)  # type: ignore[arg-type]


class TestExpireSnapshots:
    def test_retain_last_keeps_the_newest(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        table = catalog.load_table(tuple(fragmented.split(".")))
        # Six appends, six snapshots -- CREATE TABLE commits metadata, not a snapshot.
        assert len(list(table.snapshots())) == 6
        result = session.maintenance(fragmented).expireSnapshots(retainLast=2)
        assert len(result.expired) == 4
        assert len(list(catalog.load_table(tuple(fragmented.split("."))).snapshots())) == 2

    def test_the_table_still_reads_afterwards(self, session: Session, fragmented: str) -> None:
        session.maintenance(fragmented).expireSnapshots(retainLast=1)
        assert session.table(fragmented).count() == 18

    def test_the_current_snapshot_is_never_expired(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        """A table whose current snapshot was deleted is not readable."""
        table = catalog.load_table(tuple(fragmented.split(".")))
        current = table.current_snapshot()
        assert current is not None
        result = session.maintenance(fragmented).expireSnapshots(
            snapshotIds=[s.snapshot_id for s in table.snapshots()]
        )
        assert current.snapshot_id not in result.expired
        assert session.table(fragmented).count() == 18

    def test_older_than_expires_by_age(self, session: Session, fragmented: str) -> None:
        result = session.maintenance(fragmented).expireSnapshots(
            olderThan=datetime.now(UTC) + timedelta(days=1)
        )
        assert result.expired  # everything but the current one
        assert session.table(fragmented).count() == 18

    def test_a_cutoff_before_everything_expires_nothing(
        self, session: Session, fragmented: str
    ) -> None:
        result = session.maintenance(fragmented).expireSnapshots(
            olderThan=datetime.now(UTC) - timedelta(days=365)
        )
        assert result.expired == ()

    def test_specific_ids_can_be_named(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        table = catalog.load_table(tuple(fragmented.split(".")))
        oldest = sorted(table.snapshots(), key=lambda s: s.timestamp_ms)[0]
        result = session.maintenance(fragmented).expireSnapshots(snapshotIds=[oldest.snapshot_id])
        assert result.expired == (oldest.snapshot_id,)

    def test_expiring_with_no_policy_is_refused(self, session: Session, fragmented: str) -> None:
        """Expiring everything is never what was meant."""
        with pytest.raises(EngineValueError, match="olderThan"):
            session.maintenance(fragmented).expireSnapshots()

    def test_retain_last_below_one_is_refused(self, session: Session, fragmented: str) -> None:
        with pytest.raises(EngineValueError, match="at least 1"):
            session.maintenance(fragmented).expireSnapshots(retainLast=0)


class TestDeletingOrphansIsSafe:
    """The test is not "were the right files deleted" but "does the table still read".

    A file that looks orphaned and is not is unrecoverable, so the assertion that
    matters is the one about the data, taken after the deletion.
    """

    def test_a_fresh_table_has_no_orphans(self, session: Session, fragmented: str) -> None:
        scan = session.maintenance(fragmented).removeOrphanFiles(olderThan=datetime.now(UTC))
        assert scan.orphans == ()
        assert scan.referenced > 0

    def test_it_reports_without_deleting_by_default(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        session.maintenance(fragmented).expireSnapshots(retainLast=1)
        scan = session.maintenance(fragmented).removeOrphanFiles(olderThan=datetime.now(UTC))
        assert scan.orphans  # the files compaction replaced
        assert scan.deleted == ()
        assert session.table(fragmented).count() == 18

    def test_deleting_leaves_the_table_readable(self, session: Session, fragmented: str) -> None:
        before = sorted(tuple(row) for row in session.table(fragmented).collect())
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        session.maintenance(fragmented).expireSnapshots(retainLast=1)
        scan = session.maintenance(fragmented).removeOrphanFiles(
            olderThan=datetime.now(UTC), dryRun=False
        )
        assert scan.deleted
        assert sorted(tuple(row) for row in session.table(fragmented).collect()) == before

    def test_a_live_data_file_is_never_an_orphan(
        self, session: Session, catalog: SqlCatalog, fragmented: str
    ) -> None:
        """Every file the current snapshot references must be excluded by name."""
        table = catalog.load_table(tuple(fragmented.split(".")))
        live = {
            task.file.file_path.replace("\\", "/").split("/")[-1]
            for task in table.scan().plan_files()
        }
        scan = session.maintenance(fragmented).removeOrphanFiles(olderThan=datetime.now(UTC))
        orphaned = {path.split("/")[-1] for path in scan.orphans}
        assert live and not (live & orphaned)

    def test_recent_files_are_left_alone_by_default(
        self, session: Session, fragmented: str
    ) -> None:
        """A commit in flight has written its data and not yet referenced it."""
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        session.maintenance(fragmented).expireSnapshots(retainLast=1)
        scan = session.maintenance(fragmented).removeOrphanFiles()
        assert scan.orphans == ()

    def test_the_scan_reports_sizes(self, session: Session, fragmented: str) -> None:
        session.maintenance(fragmented).compact(targetFileSizeBytes=SMALL)
        session.maintenance(fragmented).expireSnapshots(retainLast=1)
        scan = session.maintenance(fragmented).removeOrphanFiles(olderThan=datetime.now(UTC))
        assert scan.bytes_orphaned > 0
        assert "orphan(s)" in str(scan)


class TestRefused:
    def test_rewrite_manifests_is_refused_by_name(self, session: Session, fragmented: str) -> None:
        """PyIceberg 0.11 exposes none, and a wrong manifest hides data silently."""
        with pytest.raises(UnsupportedFeatureError, match="rewriteManifests"):
            session.maintenance(fragmented).rewriteManifests()

    def test_the_refusal_says_what_does_work(self, session: Session, fragmented: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="compact"):
            session.maintenance(fragmented).rewriteManifests()

    def test_maintenance_needs_a_table_name(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.maintenance(42)  # type: ignore[arg-type]

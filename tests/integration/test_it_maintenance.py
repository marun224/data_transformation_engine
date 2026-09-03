"""Table maintenance against the real warehouse.

Compaction, snapshot expiry and orphan detection are the three operations that *delete
things*, so they are the ones where running against a real object store rather than a
temp directory matters most: an orphan scan lists a MinIO prefix, an expiry drops
manifests the catalog still points at, and a compaction commits a replacement for files
that other readers may hold.

Every table here is a throwaway the test builds itself, because these operations are
destructive by nature and the fixtures are shared.

Two behaviours from `divergence.md` are asserted rather than assumed:

  * **`removeOrphanFiles` reports by default.** Deleting a file that only looks like an
    orphan -- because the commit that wrote it has not landed yet -- is unrecoverable.
    `dryRun=False` is the caller saying they read the list.
  * **`rewriteManifests` is refused.** PyIceberg 0.11 has no such operation, and a
    maintenance call that silently does nothing is worse than one that says so.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest

from icetl.errors import UnsupportedFeatureError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


@pytest.fixture
def fragmented(session: Session, zones: str, target: str) -> str:
    """A table deliberately built out of many small files."""
    source = session.table(zones)
    source.write.saveAsTable(target)
    for _ in range(4):
        source.write.mode("append").saveAsTable(target)
    return target


def _file_count(catalog: Catalog, identifier: str) -> int:
    snapshot = catalog.load_table(identifier).current_snapshot()
    assert snapshot is not None and snapshot.summary is not None
    return int(str(snapshot.summary["total-data-files"]))


class TestCompaction:
    def test_the_fixture_really_is_fragmented(self, catalog: Catalog, fragmented: str) -> None:
        """The premise, asserted -- compacting one file would prove nothing."""
        assert _file_count(catalog, fragmented) >= 5

    def test_compaction_reduces_the_file_count(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        before = _file_count(catalog, fragmented)
        result = session.maintenance(fragmented).compact(minInputFiles=2)
        assert result.changed
        assert _file_count(catalog, fragmented) < before

    def test_compaction_does_not_change_the_rows(self, session: Session, fragmented: str) -> None:
        """The only thing that must not change. Fewer files, identical data."""
        before = session.table(fragmented).orderBy("zone_id", "zone_name").collect()
        session.maintenance(fragmented).compact(minInputFiles=2)
        session.catalog.refreshTable(fragmented)
        after = session.table(fragmented).orderBy("zone_id", "zone_name").collect()
        assert after == before

    def test_compaction_does_not_change_the_count(self, session: Session, fragmented: str) -> None:
        before = session.table(fragmented).count()
        session.maintenance(fragmented).compact(minInputFiles=2)
        session.catalog.refreshTable(fragmented)
        assert session.table(fragmented).count() == before

    def test_a_table_that_does_not_need_compacting_is_left_alone(
        self, session: Session, catalog: Catalog, zones: str, target: str
    ) -> None:
        """A compaction that rewrote everything regardless would be worse than none."""
        session.table(zones).write.saveAsTable(target)
        before = _file_count(catalog, target)
        result = session.maintenance(target).compact(minInputFiles=10)
        assert not result.changed
        assert _file_count(catalog, target) == before

    def test_compaction_adds_a_snapshot(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        before = len(catalog.load_table(fragmented).metadata.snapshots)
        session.maintenance(fragmented).compact(minInputFiles=2)
        assert len(catalog.load_table(fragmented).metadata.snapshots) > before


class TestExpireSnapshots:
    """One at a time works; a batch is refused by the catalog -- FINDINGS 2.12."""

    def test_expiring_a_single_snapshot_works(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        history = sorted(
            catalog.load_table(fragmented).metadata.snapshots, key=lambda s: s.timestamp_ms
        )
        before = len(history)
        session.maintenance(fragmented).expireSnapshots(snapshotIds=[history[0].snapshot_id])
        assert len(catalog.load_table(fragmented).metadata.snapshots) == before - 1

    def test_expiring_several_at_once_is_refused_by_a_rest_catalog(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        """FINDINGS 2.12 -- the second real-catalog-only defect this suite found.

        Iceberg's REST spec gives `remove-snapshots` exactly one `snapshot-id` per
        metadata update. PyIceberg 0.11.1 packs every id into a single update, so any
        expiry of more than one snapshot is rejected:

            IllegalArgumentException: Invalid set of snapshot ids to remove.
            Expected one value but received: [..., ..., ...]

        It comes back as `CommitStateUnknownException`, which for a *destructive*
        operation is the worst shape of failure: the caller cannot tell whether it
        landed. `retainLast` on any table with more than one expirable snapshot hits
        this, which is the ordinary case.

        A characterisation test: it fails when the bug is fixed, upstream or here.
        The fix on this side is to commit one id per call, as
        `test_expiring_a_single_snapshot_works` above shows already works.
        """
        from pyiceberg.exceptions import CommitStateUnknownException

        history = catalog.load_table(fragmented).metadata.snapshots
        assert len(history) > 2, "need several expirable snapshots"

        with pytest.raises(CommitStateUnknownException, match="Invalid set of snapshot ids"):
            session.maintenance(fragmented).expireSnapshots(retainLast=2)

    def test_expiry_never_drops_the_current_snapshot(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        current = catalog.load_table(fragmented).current_snapshot()
        assert current is not None

        session.maintenance(fragmented).expireSnapshots(snapshotIds=[current.snapshot_id])
        still = catalog.load_table(fragmented).current_snapshot()
        assert still is not None and still.snapshot_id == current.snapshot_id

    def test_the_table_still_reads_after_expiry(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        before = session.table(fragmented).count()
        history = sorted(
            catalog.load_table(fragmented).metadata.snapshots, key=lambda s: s.timestamp_ms
        )
        session.maintenance(fragmented).expireSnapshots(snapshotIds=[history[0].snapshot_id])
        session.catalog.refreshTable(fragmented)
        assert session.table(fragmented).count() == before

    def test_expiring_by_age_keeps_everything_recent(
        self, session: Session, catalog: Catalog, fragmented: str
    ) -> None:
        """Every snapshot was made seconds ago, so an hour-old cutoff drops none."""
        before = len(catalog.load_table(fragmented).metadata.snapshots)
        cutoff = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)
        session.maintenance(fragmented).expireSnapshots(olderThan=cutoff)
        assert len(catalog.load_table(fragmented).metadata.snapshots) == before


class TestRefusedRatherThanHalfDone:
    def test_rewrite_manifests_is_refused(self, session: Session, zones: str, target: str) -> None:
        """PyIceberg 0.11 has no such operation, and silence would be worse."""
        session.table(zones).write.saveAsTable(target)
        with pytest.raises(UnsupportedFeatureError):
            session.maintenance(target).rewriteManifests()


class TestMaintenanceOnRealisticData:
    """The same operations over a table carrying real trip rows."""

    def test_compacting_a_partitioned_real_table_preserves_every_row(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)
        source.write.mode("append").partitionBy("VendorID").saveAsTable(target)

        before = session.table(target).count()
        nulls = session.table(target).filter(F.col("store_and_fwd_flag").isNull()).count()

        session.maintenance(target).compact(minInputFiles=2)
        session.catalog.refreshTable(target)

        assert session.table(target).count() == before
        assert session.table(target).filter(F.col("store_and_fwd_flag").isNull()).count() == nulls

    def test_compaction_keeps_the_partition_layout(
        self, session: Session, catalog: Catalog, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)
        source.write.mode("append").partitionBy("VendorID").saveAsTable(target)

        session.maintenance(target).compact(minInputFiles=2)
        spec = catalog.load_table(target).spec()
        assert [field.name for field in spec.fields] == ["VendorID"]

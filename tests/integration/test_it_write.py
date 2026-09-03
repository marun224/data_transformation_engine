"""The write path, committing real rows to the real catalog.

Every table here is a throwaway in the integration namespace, created by the test and
dropped by the `target` fixture through `guard.safe_drop` -- which refuses any name
outside that namespace. The session-wide `Witness` re-reads `nyc` and `amazon` after the
last test and fails the run if either moved, so these tests are safe to run against a
warehouse holding real data.

What a real catalog adds over the local fixture:

  * **A REST commit, not a sqlite one.** Snapshot creation, optimistic concurrency and
    metadata-location updates all go through the catalog service.
  * **Data files land on MinIO**, so partition layout is a real object-store prefix
    rather than a directory, and every path has to survive `s3://` translation twice --
    once on the way out, once on the way back.
  * **Real rows.** The source is a slice of `nyc.yellow_tripdata`, so writes carry real
    timestamps, real NULLs and real doubles instead of five hand-typed values.

`TestPartitionOverwrite` is the class to read first, for the reason the local suite gives:
a `static` overwrite -- the default, here and in the reference -- replaces **every row in
the table**, while a `dynamic` one replaces only the partitions the incoming data
touches. The two differ by an option string and by the entire contents of the table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import AnalysisException, TableAlreadyExistsException
from icetl.sql import functions as F
from tests.integration.helpers import column

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestSaveAsTable:
    def test_a_missing_table_is_created_from_the_frame(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        source.write.saveAsTable(target)
        assert session.table(target).count() == source.count()

    def test_the_created_table_keeps_the_column_names(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        """Mixed case included -- `VendorID` is indexed on by name downstream."""
        source = session.table(trips_small)
        source.write.saveAsTable(target)
        assert session.table(target).columns == source.columns

    def test_the_written_values_come_back(self, session: Session, zones: str, target: str) -> None:
        source = session.table(zones).orderBy("zone_id")
        source.write.saveAsTable(target)
        written = session.table(target).orderBy("zone_id")
        assert written.collect() == source.collect()

    def test_real_nulls_survive_the_round_trip(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        """A NULL written as an empty string would be a silent corruption."""
        source = session.table(trips_small)
        expected = source.filter(F.col("store_and_fwd_flag").isNull()).count()
        assert expected > 0
        source.write.saveAsTable(target)
        assert (
            session.table(target).filter(F.col("store_and_fwd_flag").isNull()).count() == expected
        )

    def test_writing_over_an_existing_table_is_refused_by_default(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.table(zones).write.saveAsTable(target)
        with pytest.raises(TableAlreadyExistsException):
            session.table(zones).write.saveAsTable(target)


class TestSaveModes:
    def test_append_adds_to_what_is_there(self, session: Session, zones: str, target: str) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        source.write.mode("append").saveAsTable(target)
        assert session.table(target).count() == source.count() * 2

    def test_overwrite_replaces_what_is_there(
        self, session: Session, zones: str, target: str
    ) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        half = source.filter(F.col("zone_id") < 100)
        half.write.mode("overwrite").saveAsTable(target)
        assert session.table(target).count() == half.count()

    def test_ignore_leaves_the_table_alone(self, session: Session, zones: str, target: str) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        before = session.table(target).count()
        source.filter(F.col("zone_id") < 10).write.mode("ignore").saveAsTable(target)
        assert session.table(target).count() == before

    def test_error_mode_raises(self, session: Session, zones: str, target: str) -> None:
        session.table(zones).write.saveAsTable(target)
        with pytest.raises(TableAlreadyExistsException):
            session.table(zones).write.mode("error").saveAsTable(target)


class TestReadsSeeWrites:
    """FINDINGS 1.6: a write was once invisible to the session that made it.

    A `ScanSource` pins a PyIceberg table to the snapshot it was loaded at, so without
    invalidation the very session that appended could not see its own rows.
    """

    def test_the_writing_session_sees_its_own_append(
        self, session: Session, zones: str, target: str
    ) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        first = session.table(target).count()
        source.write.mode("append").saveAsTable(target)
        assert session.table(target).count() == first * 2

    def test_a_read_before_the_write_does_not_pin_the_result(
        self, session: Session, zones: str, target: str
    ) -> None:
        """Reading first is what made the stale pin observable."""
        source = session.table(zones)
        source.write.saveAsTable(target)
        session.table(target).count()  # pins the snapshot
        source.write.mode("append").saveAsTable(target)
        assert session.table(target).count() == source.count() * 2

    def test_another_session_sees_the_committed_rows(
        self, session: Session, it_session: Session, zones: str, target: str
    ) -> None:
        """The commit is in the catalog, not in the writer's memory."""
        source = session.table(zones)
        source.write.saveAsTable(target)
        it_session.catalog.refreshTable(target)
        assert it_session.table(target).count() == source.count()


class TestPartitionedCreation:
    def test_partition_by_creates_the_partition_spec(
        self, session: Session, catalog: Catalog, trips_small: str, target: str
    ) -> None:
        session.table(trips_small).write.partitionBy("VendorID").saveAsTable(target)
        spec = catalog.load_table(target).spec()
        assert [field.name for field in spec.fields] == ["VendorID"]

    def test_a_partitioned_write_produces_a_file_per_partition(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        vendors = source.select("VendorID").distinct().count()
        source.write.partitionBy("VendorID").saveAsTable(target)

        from tests.integration.helpers import scan_of

        scan = scan_of(session.table(target))
        assert scan.files_total == vendors

    def test_the_partitioned_table_holds_the_same_rows(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)
        assert session.table(target).count() == source.count()

    def test_a_partition_filter_prunes_the_new_table(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        """The write and the read have to agree about the partition layout."""
        from tests.integration.helpers import scan_of

        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)
        one = session.table(target).filter(F.col("VendorID") == 1)
        scan = scan_of(one)
        assert scan.files_total is not None
        assert scan.files_scanned < scan.files_total
        assert one.count() == source.filter(F.col("VendorID") == 1).count()


class TestPartitionOverwrite:
    """Static replaces the table; dynamic replaces only the partitions written."""

    def test_a_static_overwrite_replaces_every_row(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)

        one_vendor = source.filter(F.col("VendorID") == 1)
        one_vendor.write.mode("overwrite").partitionBy("VendorID").saveAsTable(target)

        assert session.table(target).count() == one_vendor.count()
        assert set(column(session.table(target).select("VendorID").distinct(), "VendorID")) == {1}

    def test_a_dynamic_overwrite_keeps_the_untouched_partitions(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)
        before = source.count()
        vendors = set(column(source.select("VendorID").distinct(), "VendorID"))
        assert len(vendors) > 1, "one partition cannot show the difference"

        one_vendor = source.filter(F.col("VendorID") == 1)
        (
            one_vendor.write.mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .partitionBy("VendorID")
            .saveAsTable(target)
        )

        after = session.table(target)
        assert after.count() == before, "a dynamic overwrite changed the untouched partitions"
        assert set(column(after.select("VendorID").distinct(), "VendorID")) == vendors

    def test_a_dynamic_overwrite_does_replace_the_partition_it_touches(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        """The other half: it must not simply append."""
        source = session.table(trips_small)
        source.write.partitionBy("VendorID").saveAsTable(target)

        half_of_one = source.filter(F.col("VendorID") == 1).limit(10)
        (
            half_of_one.write.mode("overwrite")
            .option("partitionOverwriteMode", "dynamic")
            .partitionBy("VendorID")
            .saveAsTable(target)
        )
        remaining = session.table(target).filter(F.col("VendorID") == 1).count()
        assert remaining == 10


class TestInsertInto:
    def test_insert_into_appends_by_position(
        self, session: Session, zones: str, target: str
    ) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        source.write.insertInto(target)
        assert session.table(target).count() == source.count() * 2

    def test_insert_into_with_overwrite_replaces(
        self, session: Session, zones: str, target: str
    ) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        half = source.filter(F.col("zone_id") < 100)
        half.write.insertInto(target, overwrite=True)
        assert session.table(target).count() == half.count()

    def test_a_sql_insert_appends(self, session: Session, zones: str, target: str) -> None:
        source = session.table(zones)
        source.write.saveAsTable(target)
        session.sql(f"INSERT INTO {target} SELECT * FROM {zones}")
        assert session.table(target).count() == source.count() * 2

    def test_inserting_the_wrong_shape_is_refused(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.table(zones).write.saveAsTable(target)
        with pytest.raises((AnalysisException, Exception)):
            session.table(zones).select("zone_id").write.insertInto(target)


class TestMergeSchema:
    def test_a_new_column_is_added_when_merge_schema_is_asked_for(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.table(zones).write.saveAsTable(target)
        widened = session.table(zones).withColumn("extra", F.lit(1))
        widened.write.mode("append").option("mergeSchema", "true").saveAsTable(target)

        assert "extra" in session.table(target).columns
        assert session.table(target).count() == session.table(zones).count() * 2

    def test_the_old_rows_are_null_in_the_new_column(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.table(zones).write.saveAsTable(target)
        original = session.table(zones).count()
        widened = session.table(zones).withColumn("extra", F.lit(1))
        widened.write.mode("append").option("mergeSchema", "true").saveAsTable(target)
        assert session.table(target).filter(F.col("extra").isNull()).count() == original

    def test_a_new_column_without_merge_schema_is_refused(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.table(zones).write.saveAsTable(target)
        widened = session.table(zones).withColumn("extra", F.lit(1))
        with pytest.raises(AnalysisException):
            widened.write.mode("append").saveAsTable(target)


class TestTheGuardItself:
    """The safety net, asserted rather than trusted."""

    def test_a_write_outside_the_namespace_would_be_refused(self) -> None:
        from tests.integration.conftest import REAL_TABLE
        from tests.integration.guard import ProtectedNamespaceError, safe_identifier

        with pytest.raises(ProtectedNamespaceError):
            safe_identifier(REAL_TABLE)

    def test_the_target_fixture_stays_inside_the_namespace(
        self, target: str, namespace: str
    ) -> None:
        assert target.startswith(f"{namespace}.")

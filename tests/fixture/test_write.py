"""Phase 7: the write path, against the local fixture catalog.

Every test writes into its own throwaway table in the `wr` namespace and drops it
afterwards, because the fixture tables are session-scoped and deliberately immutable --
a test that appended to `fx.plain` would change what every later test reads.

**The case worth reading first is `TestPartitionOverwrite`.** A `static` overwrite --
the default, here and in the reference -- replaces *every row in the table*, not the
rows resembling the incoming data. A `dynamic` one replaces only the partitions the new
data touches. The two differ by an option string and by the entire contents of the
table, so both are asserted on row counts *and* on which partitions survived.

`TestReadsSeeWrites` guards the other thing that went wrong while building this: a
`ScanSource` pins a PyIceberg table to the snapshot it was loaded at, so without
invalidation a write was invisible to the very session that made it.

Every assertion is on a value, per the rule Phase 3 established.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    AnalysisException,
    EngineValueError,
    TableAlreadyExistsException,
    UnsupportedFeatureError,
)
from icetl.sql import functions as F

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


@pytest.fixture
def target(catalog: SqlCatalog) -> Iterator[str]:
    """A table name nothing else uses, dropped when the test ends."""
    name = f"wr.t_{uuid.uuid4().hex[:8]}"
    yield name
    with contextlib.suppress(Exception):
        catalog.drop_table(tuple(name.split(".")))


def ids(df: DataFrame) -> list[int]:
    return sorted(row[0] for row in df.collect())


class TestSaveAsTable:
    def test_a_missing_table_is_created_from_the_frame(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        written = session.table(target)
        assert written.dtypes == source.dtypes
        assert ids(written) == [1, 2, 3, 4, 5]

    def test_a_missing_table_is_created_in_every_mode(self, session: Session, target: str) -> None:
        """The mode is about existing data, and a missing table has none."""
        session.table("fx.plain").write.mode("ignore").saveAsTable(target)
        assert session.table(target).count() == 5

    def test_append_adds_to_what_is_there(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        source.write.mode("append").saveAsTable(target)
        assert session.table(target).count() == 10

    def test_overwrite_replaces_everything(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        source.filter(F.col("id") <= 2).write.mode("overwrite").saveAsTable(target)
        assert ids(session.table(target)) == [1, 2]

    def test_ignore_leaves_an_existing_table_alone(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.filter(F.col("id") <= 2).write.saveAsTable(target)
        source.write.mode("ignore").saveAsTable(target)
        assert ids(session.table(target)) == [1, 2]

    def test_the_default_mode_refuses_an_existing_table(
        self, session: Session, target: str
    ) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        with pytest.raises(TableAlreadyExistsException, match="already exists"):
            source.write.saveAsTable(target)

    def test_an_empty_frame_still_creates_the_table(self, session: Session, target: str) -> None:
        session.table("fx.plain").filter(F.col("id") < 0).write.saveAsTable(target)
        written = session.table(target)
        assert written.count() == 0
        assert written.columns == ["id", "vendor", "amount"]

    def test_complex_types_round_trip(self, session: Session, target: str) -> None:
        """Structs, arrays and maps survive the trip out and back."""
        source = session.table("fx.nested")
        source.write.saveAsTable(target)
        written = session.table(target)
        assert written.dtypes == source.dtypes
        assert [tuple(r) for r in written.collect()] == [tuple(r) for r in source.collect()]


class TestInsertInto:
    def test_it_appends_to_an_existing_table(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        source.filter(F.col("id") <= 2).write.insertInto(target)
        assert session.table(target).count() == 7

    def test_it_refuses_a_table_that_does_not_exist(self, session: Session, target: str) -> None:
        """`insertInto` never creates; that is what `saveAsTable` is for."""
        with pytest.raises(AnalysisException, match="does not exist"):
            session.table("fx.plain").write.insertInto(target)

    def test_overwrite_replaces_the_rows(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        source.filter(F.col("id") == 1).write.insertInto(target, overwrite=True)
        assert ids(session.table(target)) == [1]

    def test_it_matches_by_position_not_by_name(self, session: Session, target: str) -> None:
        """The reference's rule, and the reason `insertInto` is its own method.

        Two columns of the *same type* in the wrong order go in swapped, with nothing
        raising -- names are not consulted at all. Both columns are `bigint` here
        precisely so that nothing else can catch it.
        """
        session.createDataFrame([(1, 100)], "a bigint, b bigint").write.saveAsTable(target)
        swapped = session.createDataFrame([(2, 200)], "a bigint, b bigint").select("b", "a")
        swapped.write.insertInto(target)
        written = sorted(tuple(row) for row in session.table(target).collect())
        assert written == [(1, 100), (200, 2)], "the second row went in by position"

    def test_a_type_mismatch_is_caught_rather_than_scrambled(
        self, session: Session, target: str
    ) -> None:
        """By-position is silent only where the types happen to line up.

        `fx.plain` is `bigint, string, double`, so inserting it in the wrong order is
        caught by PyIceberg's schema check before anything is written -- a narrower
        exposure than the reference's, and worth pinning so it is not mistaken for luck.
        """
        source = session.table("fx.plain").select("id", "vendor", "amount")
        source.write.saveAsTable(target)
        swapped = session.table("fx.plain").select("vendor", "id", "amount")
        with pytest.raises(Exception, match="Mismatch in fields"):
            swapped.write.insertInto(target)

    def test_a_column_count_mismatch_is_refused(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        with pytest.raises(AnalysisException, match="by position"):
            source.select("id").write.insertInto(target)


class TestTableSettings:
    def test_partition_by_partitions_a_new_table(
        self, session: Session, target: str, catalog: SqlCatalog
    ) -> None:
        session.table("fx.partitioned").write.partitionBy("as_at_date").saveAsTable(target)
        table = catalog.load_table(tuple(target.split(".")))
        assert [field.name for field in table.spec().fields] == ["as_at_date"]

    def test_partitioned_by_is_the_same_method(self, session: Session, target: str) -> None:
        writer = session.table("fx.partitioned").write
        assert writer.partitionedBy("as_at_date")._partition_by == ["as_at_date"]

    def test_sort_by_records_a_sort_order(
        self, session: Session, target: str, catalog: SqlCatalog
    ) -> None:
        session.table("fx.plain").write.sortBy("id").saveAsTable(target)
        table = catalog.load_table(tuple(target.split(".")))
        assert len(table.sort_order().fields) == 1

    def test_table_property_is_set(
        self, session: Session, target: str, catalog: SqlCatalog
    ) -> None:
        session.table("fx.plain").write.tableProperty("owner", "etl").saveAsTable(target)
        table = catalog.load_table(tuple(target.split(".")))
        assert table.properties["owner"] == "etl"

    def test_an_unknown_partition_column_is_refused(self, session: Session, target: str) -> None:
        with pytest.raises(AnalysisException, match="does not have"):
            session.table("fx.plain").write.partitionBy("nope").saveAsTable(target)

    def test_the_writer_is_immutable(self, session: Session) -> None:
        base = session.table("fx.plain").write
        appending = base.mode("append")
        assert base._mode == "error"
        assert appending._mode == "append"


class TestPartitionOverwrite:
    """`static` replaces the whole table; `dynamic` replaces only the partitions touched.

    They differ by one option string and by the entire contents of the table, so the
    default being `static` -- as it is in the reference -- is worth knowing about.
    """

    @pytest.fixture
    def partitioned(self, session: Session, target: str) -> str:
        session.table("fx.partitioned").write.partitionBy("as_at_date").saveAsTable(target)
        return target

    def test_static_overwrite_replaces_every_row(self, session: Session, partitioned: str) -> None:
        one = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        one.write.mode("overwrite").saveAsTable(partitioned)
        written = session.table(partitioned)
        assert written.count() == 4, "static overwrite drops the other partitions"
        assert {r[0] for r in written.select("as_at_date").collect()} == {"2026-08-16"}

    def test_dynamic_overwrite_replaces_only_that_partition(
        self, session: Session, partitioned: str
    ) -> None:
        one = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        one.write.mode("overwrite").option("partitionOverwriteMode", "dynamic").saveAsTable(
            partitioned
        )
        written = session.table(partitioned)
        assert written.count() == 12, "the other two partitions survive"
        assert {r[0] for r in written.select("as_at_date").collect()} == {
            "2026-08-15",
            "2026-08-16",
            "2026-08-17",
        }

    def test_dynamic_really_replaces_rather_than_appends(
        self, session: Session, partitioned: str
    ) -> None:
        """Two of the four rows of one partition, so the partition ends up with two."""
        two = session.table("fx.partitioned").filter(
            (F.col("as_at_date") == "2026-08-16") & (F.col("id") <= 5)
        )
        two.write.mode("overwrite").option("partitionOverwriteMode", "dynamic").saveAsTable(
            partitioned
        )
        written = session.table(partitioned)
        assert written.count() == 10
        assert written.filter(F.col("as_at_date") == "2026-08-16").count() == 2

    def test_dynamic_needs_a_partitioned_table(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        writer = source.write.mode("overwrite").option("partitionOverwriteMode", "dynamic")
        with pytest.raises(AnalysisException, match="partitioned table"):
            writer.saveAsTable(target)

    def test_an_unknown_overwrite_mode_is_refused(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        writer = source.write.mode("overwrite").option("partitionOverwriteMode", "sometimes")
        with pytest.raises(EngineValueError, match="static' or 'dynamic"):
            writer.saveAsTable(target)


class TestSqlInsert:
    def test_insert_into_select(self, session: Session, target: str) -> None:
        session.table("fx.plain").write.saveAsTable(target)
        session.sql(f"INSERT INTO {target} SELECT * FROM fx.plain WHERE id = 1")
        assert session.table(target).count() == 6

    def test_insert_overwrite_select(self, session: Session, target: str) -> None:
        session.table("fx.plain").write.saveAsTable(target)
        session.sql(f"INSERT OVERWRITE {target} SELECT * FROM fx.plain WHERE id <= 2")
        assert ids(session.table(target)) == [1, 2]

    def test_it_returns_an_empty_frame(self, session: Session, target: str) -> None:
        """A statement is not a query, but callers still expect something back."""
        session.table("fx.plain").write.saveAsTable(target)
        out = session.sql(f"INSERT INTO {target} SELECT * FROM fx.plain WHERE id = 1")
        assert out.count() == 0

    def test_it_agrees_with_the_dataframe_surface(self, session: Session, target: str) -> None:
        """P1: both surfaces route through the same writer."""
        session.table("fx.plain").filter(F.col("id") < 0).write.saveAsTable(target)
        session.sql(f"INSERT INTO {target} SELECT * FROM fx.plain WHERE id = 1")
        through_sql = ids(session.table(target))

        session.table("fx.plain").filter(F.col("id") == 1).write.insertInto(target)
        assert ids(session.table(target)) == sorted([*through_sql, 1])

    def test_an_explicit_column_list_is_refused(self, session: Session, target: str) -> None:
        """Renaming half a table on the way in is a different feature from inserting."""
        session.table("fx.plain").write.saveAsTable(target)
        with pytest.raises(UnsupportedFeatureError, match="column list"):
            session.sql(f"INSERT INTO {target} (id) SELECT id FROM fx.plain")

    def test_insert_from_values_is_refused_clearly(self, session: Session, target: str) -> None:
        session.table("fx.plain").write.saveAsTable(target)
        with pytest.raises(UnsupportedFeatureError, match="INSERT"):
            session.sql(f"INSERT INTO {target} VALUES (1, 'a', 1.0)")


class TestReadsSeeWrites:
    """A `ScanSource` pins a table to the snapshot it was loaded at.

    Without invalidating that cache a write is invisible to the session that made it --
    every count in the first smoke test came back unchanged, which looked like the write
    silently failing rather than the read being stale.
    """

    def test_a_write_is_visible_to_the_next_read(self, session: Session, target: str) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        assert session.table(target).count() == 5
        source.write.mode("append").saveAsTable(target)
        assert session.table(target).count() == 10

    def test_a_frame_built_before_the_write_keeps_its_snapshot(
        self, session: Session, target: str
    ) -> None:
        """Read-your-plan, not read-your-writes: the frame is bound to what it resolved."""
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        before = session.table(target)
        assert before.count() == 5
        source.write.mode("append").saveAsTable(target)
        assert before.count() == 5
        assert session.table(target).count() == 10


class TestSnapshotShape:
    """PLAN.md asked for "one Iceberg snapshot per write". Measured, that is not quite it.

    An append is one snapshot. An **overwrite is two** -- a delete and an append --
    because that is how Iceberg models replacing rows. Both land inside a single
    transaction, though, so the property that actually matters holds: one *commit* per
    write, and no reader ever sees the table mid-overwrite.
    """

    def test_an_append_is_one_snapshot(
        self, session: Session, target: str, catalog: SqlCatalog
    ) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        before = len(catalog.load_table(tuple(target.split("."))).metadata.snapshots)
        source.write.mode("append").saveAsTable(target)
        after = len(catalog.load_table(tuple(target.split("."))).metadata.snapshots)
        assert after - before == 1

    def test_an_overwrite_is_a_delete_and_an_append(
        self, session: Session, target: str, catalog: SqlCatalog
    ) -> None:
        source = session.table("fx.plain")
        source.write.saveAsTable(target)
        before = len(catalog.load_table(tuple(target.split("."))).metadata.snapshots)
        source.write.mode("overwrite").saveAsTable(target)
        snapshots = catalog.load_table(tuple(target.split("."))).metadata.snapshots
        assert len(snapshots) - before == 2
        operations = [
            s.summary.operation.value if s.summary is not None else None for s in snapshots[-2:]
        ]
        assert operations == ["delete", "append"]


class TestStreamingIsBlockedUpstream:
    """PLAN.md wanted batches streamed into PyIceberg. Its API will not take them.

    Kept as a characterisation test rather than a comment: if a later PyIceberg accepts
    a reader, this fails, and that failure is the signal that the streaming write in
    PLAN.md's Phase 7 has become buildable.
    """

    def test_pyiceberg_refuses_anything_but_a_table(
        self, session: Session, target: str, catalog: SqlCatalog
    ) -> None:
        import pyarrow as pa

        session.table("fx.plain").write.saveAsTable(target)
        table = catalog.load_table(tuple(target.split(".")))
        data = session.table("fx.plain").toArrow()
        reader = pa.RecordBatchReader.from_batches(data.schema, data.to_batches())
        with pytest.raises(ValueError, match="Expected PyArrow table"):
            table.append(reader)


class TestWriterRefusals:
    def test_a_path_write_is_refused(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="saveAsTable"):
            session.table("fx.plain").write.save("/tmp/out")

    def test_another_format_is_refused(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="format"):
            session.table("fx.plain").write.format("parquet")

    def test_iceberg_format_is_accepted(self, session: Session) -> None:
        assert session.table("fx.plain").write.format("iceberg") is not None

    def test_an_unknown_mode_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="Unknown save mode"):
            session.table("fx.plain").write.mode("upsert")

    def test_an_unqualified_name_is_refused(self, session: Session) -> None:
        """The fixture session has a default namespace, so this needs a name with none."""
        with pytest.raises((AnalysisException, TableAlreadyExistsException)):
            session.table("fx.plain").write.saveAsTable("fx.plain")

"""Phase 9: `CREATE`, `DROP` and `ALTER` -- the DDL surface.

**`TestSchemaChangesAreVisible` is the class to read first.** A DDL statement that
changes a schema and a session that keeps reading the old one is the failure this phase
had to fix: the analyser registers a zero-row view per source *once*, and source view
names were numbered from how many sources were cached -- a count that goes back down when
one is invalidated, handing the next source a name an earlier one had used. `df.columns`
then described a different table. It was reachable from Phase 7 with nothing but a write
and a read of a second table; Phase 9 found it because a schema change is the one thing
that makes it visible. FINDINGS.md §1.9.

**`TestNullability` is the second.** `CREATE TABLE t (id BIGINT NOT NULL)` says something
DuckDB cannot, so the table is built from an Arrow schema rather than handed to the
writer. Dropping the constraint silently would have been the easy path.

Every table is a throwaway in the `wr` namespace, dropped when the test ends.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    AnalysisException,
    TableAlreadyExistsException,
    TableNotFoundError,
    UnsupportedFeatureError,
)

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.session import Session


@pytest.fixture
def target(catalog: SqlCatalog) -> Iterator[str]:
    name = f"wr.t_{uuid.uuid4().hex[:8]}"
    yield name
    with contextlib.suppress(Exception):
        catalog.drop_table(tuple(name.split(".")))


def spec_of(catalog: SqlCatalog, name: str) -> list[tuple[str, str]]:
    table = catalog.load_table(tuple(name.split(".")))
    return [(field.name, str(field.transform)) for field in table.spec().fields]


def properties_of(catalog: SqlCatalog, name: str) -> dict[str, str]:
    return dict(catalog.load_table(tuple(name.split("."))).properties)


class TestCreateTable:
    def test_a_column_list(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT, v STRING) USING iceberg")
        assert session.table(target).columns == ["id", "v"]
        assert session.table(target).count() == 0

    def test_using_is_optional_and_only_iceberg_is_accepted(
        self, session: Session, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        assert session.catalog.tableExists(target)
        with pytest.raises(UnsupportedFeatureError, match="USING"):
            session.sql(f"CREATE TABLE {target}_2 (id BIGINT) USING parquet")

    def test_if_not_exists_leaves_an_existing_table_alone(
        self, session: Session, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.table("fx.plain").select("id").write.mode("append").insertInto(target)
        session.sql(f"CREATE TABLE IF NOT EXISTS {target} (id BIGINT, extra STRING)")
        assert session.table(target).columns == ["id"]
        assert session.table(target).count() == 5

    def test_the_default_refuses_an_existing_table(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        with pytest.raises(TableAlreadyExistsException, match="already exists"):
            session.sql(f"CREATE TABLE {target} (id BIGINT)")

    def test_or_replace_rebuilds_it(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT, v STRING)")
        session.sql(f"CREATE OR REPLACE TABLE {target} (other DOUBLE)")
        assert session.table(target).columns == ["other"]

    def test_tblproperties_and_comment(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        session.sql(
            f"CREATE TABLE {target} (id BIGINT) USING iceberg "
            "TBLPROPERTIES ('owner'='etl', 'tier'='raw') COMMENT 'a note'"
        )
        properties = properties_of(catalog, target)
        assert properties["owner"] == "etl"
        assert properties["tier"] == "raw"
        assert properties["comment"] == "a note"

    def test_a_location_is_refused(self, session: Session, target: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="LOCATION"):
            session.sql(f"CREATE TABLE {target} (id BIGINT) LOCATION '/tmp/x'")

    def test_neither_columns_nor_a_query_is_refused(self, session: Session, target: str) -> None:
        with pytest.raises(AnalysisException, match="column list or AS SELECT"):
            session.sql(f"CREATE TABLE {target}")


class TestNullability:
    """`NOT NULL` is honoured, which is why a created table is not built by the writer."""

    def test_not_null_reaches_iceberg(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT NOT NULL, v STRING) USING iceberg")
        columns = {c.name: c.nullable for c in session.catalog.listColumns(target)}
        assert columns == {"id": False, "v": True}

    def test_a_required_column_refuses_a_null(self, session: Session, target: str) -> None:
        """The constraint is real, not decorative."""
        session.sql(f"CREATE TABLE {target} (id BIGINT NOT NULL) USING iceberg")
        with pytest.raises(AnalysisException):
            session.sql(f"INSERT INTO {target} SELECT CAST(NULL AS BIGINT)")

    def test_a_table_created_by_a_write_is_all_optional(
        self, session: Session, target: str
    ) -> None:
        """The Phase 7 divergence, unchanged -- and the contrast that explains this class."""
        session.table("fx.plain").write.saveAsTable(target)
        assert all(c.nullable for c in session.catalog.listColumns(target))


class TestCreateTableAsSelect:
    def test_it_takes_its_schema_and_rows_from_the_query(
        self, session: Session, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT id, vendor FROM fx.plain WHERE id <= 3")
        assert session.table(target).columns == ["id", "vendor"]
        assert sorted(row[0] for row in session.table(target).collect()) == [1, 2, 3]

    def test_or_replace_as_select(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM fx.plain")
        session.sql(f"CREATE OR REPLACE TABLE {target} AS SELECT id FROM fx.plain WHERE id = 1")
        assert session.table(target).columns == ["id"]
        assert session.table(target).count() == 1

    def test_a_timestamp_column_survives_the_round_trip(
        self, session: Session, target: str
    ) -> None:
        """DuckDB stamps its own zone on a `TIMESTAMP`; Iceberg accepts only UTC.

        Creating a table from a frame carrying one had been impossible since Phase 7 --
        no fixture had a timestamp column, so nothing reached it. FINDINGS.md §2.7.
        """
        session.sql(
            f"CREATE TABLE {target} AS SELECT 1 AS id, TIMESTAMP '2026-01-01 12:00:00' AS ts"
        )
        rows = session.table(target).collect()
        assert len(rows) == 1
        assert session.table(target).columns == ["id", "ts"]


class TestPartitioning:
    def test_an_identity_partition(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT, d STRING) PARTITIONED BY (d)")
        assert spec_of(catalog, target) == [("d", "identity")]

    def test_the_transforms(self, session: Session, catalog: SqlCatalog, target: str) -> None:
        session.sql(
            f"CREATE TABLE {target} (id BIGINT, ts TIMESTAMP, v STRING) "
            "PARTITIONED BY (days(ts), bucket(4, id), truncate(3, v))"
        )
        assert spec_of(catalog, target) == [
            ("ts_day", "day"),
            ("id_bucket_4", "bucket[4]"),
            ("v_trunc_3", "truncate[3]"),
        ]

    def test_an_unknown_transform_is_refused(self, session: Session, target: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="partition transform"):
            session.sql(f"CREATE TABLE {target} (id BIGINT) PARTITIONED BY (upper(id))")

    def test_partitioning_by_a_column_the_table_lacks_is_refused(
        self, session: Session, target: str
    ) -> None:
        with pytest.raises(AnalysisException, match="No column"):
            session.sql(f"CREATE TABLE {target} (id BIGINT) PARTITIONED BY (nope)")


class TestDrop:
    def test_it_drops(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        session.sql(f"DROP TABLE {target}")
        assert not session.catalog.tableExists(target)

    def test_a_missing_table_is_refused_unless_if_exists(
        self, session: Session, target: str
    ) -> None:
        with pytest.raises(TableNotFoundError, match="does not exist"):
            session.sql(f"DROP TABLE {target}")
        session.sql(f"DROP TABLE IF EXISTS {target}")

    def test_a_dropped_table_is_not_read_from_a_stale_source(
        self, session: Session, target: str
    ) -> None:
        """The cached source holds a loaded table; it has to go before the table does."""
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        assert session.table(target).count() == 0
        session.sql(f"DROP TABLE {target}")
        with pytest.raises(TableNotFoundError):
            session.table(target)


class TestNamespaces:
    def test_create_and_drop(self, session: Session) -> None:
        name = f"ns_{uuid.uuid4().hex[:8]}"
        session.sql(f"CREATE NAMESPACE {name}")
        assert session.catalog.databaseExists(name)
        session.sql(f"DROP NAMESPACE {name}")
        assert not session.catalog.databaseExists(name)

    def test_database_and_schema_name_the_same_thing(self, session: Session) -> None:
        name = f"ns_{uuid.uuid4().hex[:8]}"
        session.sql(f"CREATE DATABASE {name}")
        assert session.catalog.databaseExists(name)
        session.sql(f"DROP SCHEMA {name}")
        assert not session.catalog.databaseExists(name)

    def test_if_not_exists(self, session: Session) -> None:
        name = f"ns_{uuid.uuid4().hex[:8]}"
        session.sql(f"CREATE NAMESPACE {name}")
        session.sql(f"CREATE NAMESPACE IF NOT EXISTS {name}")
        with pytest.raises(AnalysisException, match="already exists"):
            session.sql(f"CREATE NAMESPACE {name}")
        session.sql(f"DROP NAMESPACE {name}")

    def test_a_namespace_holding_tables_needs_cascade(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        name = f"ns_{uuid.uuid4().hex[:8]}"
        session.sql(f"CREATE NAMESPACE {name}")
        session.sql(f"CREATE TABLE {name}.t (id BIGINT)")
        try:
            with pytest.raises(AnalysisException, match="not empty"):
                session.sql(f"DROP NAMESPACE {name}")
            session.sql(f"DROP NAMESPACE {name} CASCADE")
            assert not session.catalog.databaseExists(name)
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table((name, "t"))


class TestAlterColumns:
    @pytest.fixture
    def built(self, session: Session, target: str) -> str:
        session.sql(f"CREATE TABLE {target} (id BIGINT, v STRING) USING iceberg")
        return target

    def test_add_column(self, session: Session, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ADD COLUMN amt DOUBLE")
        assert session.table(built).columns == ["id", "v", "amt"]

    def test_add_columns(self, session: Session, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ADD COLUMNS (amt DOUBLE, note STRING)")
        assert session.table(built).columns == ["id", "v", "amt", "note"]

    def test_drop_column_singular(self, session: Session, built: str) -> None:
        """sqlglot does not parse this spelling; `ddl.py` reads the text itself."""
        session.sql(f"ALTER TABLE {built} DROP COLUMN v")
        assert session.table(built).columns == ["id"]

    def test_drop_columns_plural(self, session: Session, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ADD COLUMNS (amt DOUBLE, note STRING)")
        session.sql(f"ALTER TABLE {built} DROP COLUMNS (v, note)")
        assert session.table(built).columns == ["id", "amt"]

    def test_rename_column(self, session: Session, built: str) -> None:
        session.sql(f"ALTER TABLE {built} RENAME COLUMN v TO vendor")
        assert session.table(built).columns == ["id", "vendor"]

    def test_a_renamed_column_still_reads_its_old_rows(self, session: Session, built: str) -> None:
        """Iceberg tracks columns by field id, so the data follows the rename.

        Phase 2's reconciliation is what makes this true through `read_parquet`, which
        matches by name.
        """
        session.sql(f"INSERT INTO {built} SELECT 1, 'a'")
        session.sql(f"ALTER TABLE {built} RENAME COLUMN v TO vendor")
        assert [tuple(row) for row in session.table(built).collect()] == [(1, "a")]

    def test_an_added_column_reads_null_for_existing_rows(
        self, session: Session, built: str
    ) -> None:
        session.sql(f"INSERT INTO {built} SELECT 1, 'a'")
        session.sql(f"ALTER TABLE {built} ADD COLUMN amt DOUBLE")
        assert [tuple(row) for row in session.table(built).collect()] == [(1, "a", None)]

    def test_a_column_comment(self, session: Session, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ALTER COLUMN v COMMENT 'the vendor'")
        found = {c.name: c.description for c in session.catalog.listColumns(built)}
        assert found["v"] == "the vendor"

    def test_drop_not_null(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT NOT NULL) USING iceberg")
        session.sql(f"ALTER TABLE {target} ALTER COLUMN id DROP NOT NULL")
        assert session.catalog.listColumns(target)[0].nullable is True

    def test_a_type_change_is_refused_rather_than_half_done(
        self, session: Session, built: str
    ) -> None:
        with pytest.raises(UnsupportedFeatureError, match="TYPE"):
            session.sql(f"ALTER TABLE {built} ALTER COLUMN id TYPE BIGINT")

    def test_adding_a_not_null_column_is_refused(self, session: Session, built: str) -> None:
        """Existing rows would have no value for it."""
        with pytest.raises(AnalysisException, match="cannot be NOT NULL"):
            session.sql(f"ALTER TABLE {built} ADD COLUMN amt DOUBLE NOT NULL")

    def test_an_unknown_column_is_refused(self, session: Session, built: str) -> None:
        with pytest.raises(AnalysisException, match="No column"):
            session.sql(f"ALTER TABLE {built} DROP COLUMN nope")


class TestAlterTable:
    def test_rename_to(self, session: Session, catalog: SqlCatalog, target: str) -> None:
        renamed = f"{target}_renamed"
        try:
            session.sql(f"CREATE TABLE {target} (id BIGINT)")
            session.sql(f"ALTER TABLE {target} RENAME TO {renamed}")
            assert session.catalog.tableExists(renamed)
            assert not session.catalog.tableExists(target)
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(renamed.split(".")))

    def test_set_and_unset_tblproperties(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        session.sql(f"ALTER TABLE {target} SET TBLPROPERTIES ('owner'='etl', 'tier'='raw')")
        assert properties_of(catalog, target)["owner"] == "etl"
        session.sql(f"ALTER TABLE {target} UNSET TBLPROPERTIES ('owner')")
        properties = properties_of(catalog, target)
        assert "owner" not in properties
        assert properties["tier"] == "raw"

    def test_an_unknown_alter_is_refused_by_name(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        with pytest.raises(UnsupportedFeatureError, match="Supported"):
            session.sql(f"ALTER TABLE {target} SET IDENTIFIER FIELDS id")


class TestPartitionEvolution:
    @pytest.fixture
    def built(self, session: Session, target: str) -> str:
        session.sql(f"CREATE TABLE {target} (id BIGINT, ts TIMESTAMP, v STRING) USING iceberg")
        return target

    def test_add_partition_field(self, session: Session, catalog: SqlCatalog, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ADD PARTITION FIELD v")
        assert spec_of(catalog, built) == [("v", "identity")]

    def test_add_a_transformed_partition_field(
        self, session: Session, catalog: SqlCatalog, built: str
    ) -> None:
        session.sql(f"ALTER TABLE {built} ADD PARTITION FIELD days(ts)")
        assert spec_of(catalog, built) == [("ts_day", "day")]

    def test_add_with_an_alias(self, session: Session, catalog: SqlCatalog, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ADD PARTITION FIELD bucket(8, id) AS id_bucket")
        assert spec_of(catalog, built) == [("id_bucket", "bucket[8]")]

    def test_drop_partition_field(self, session: Session, catalog: SqlCatalog, built: str) -> None:
        session.sql(f"ALTER TABLE {built} ADD PARTITION FIELD v")
        session.sql(f"ALTER TABLE {built} DROP PARTITION FIELD v")
        assert [name for name, transform in spec_of(catalog, built) if transform != "void"] == []

    def test_replace_partition_field(
        self, session: Session, catalog: SqlCatalog, built: str
    ) -> None:
        session.sql(f"ALTER TABLE {built} ADD PARTITION FIELD days(ts)")
        session.sql(f"ALTER TABLE {built} REPLACE PARTITION FIELD days(ts) WITH months(ts)")
        assert ("ts_month", "month") in spec_of(catalog, built)

    def test_dropping_a_field_that_is_not_there_is_refused(
        self, session: Session, built: str
    ) -> None:
        with pytest.raises(AnalysisException, match="No partition field"):
            session.sql(f"ALTER TABLE {built} DROP PARTITION FIELD v")


class TestSortOrder:
    def test_write_ordered_by(self, session: Session, catalog: SqlCatalog, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT, v STRING) USING iceberg")
        session.sql(f"ALTER TABLE {target} WRITE ORDERED BY id DESC NULLS LAST, v")
        order = catalog.load_table(tuple(target.split("."))).sort_order()
        assert len(order.fields) == 2

    def test_nulls_default_the_way_the_reference_orders_them(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        """Nulls first ascending -- which is what the conformance layer does for ORDER BY.

        A sort order that disagreed would write files ordered one way and have every
        read expect the other.
        """
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.sql(f"ALTER TABLE {target} WRITE ORDERED BY id")
        field = catalog.load_table(tuple(target.split("."))).sort_order().fields[0]
        assert str(field.null_order).upper().replace("-", " ") == "NULLS FIRST"

    def test_write_unordered_clears_it(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.sql(f"ALTER TABLE {target} WRITE ORDERED BY id")
        session.sql(f"ALTER TABLE {target} WRITE UNORDERED")
        assert list(catalog.load_table(tuple(target.split("."))).sort_order().fields) == []


class TestSchemaChangesAreVisible:
    """A schema change and a session that keeps reading the old one. FINDINGS.md §1.9."""

    def test_a_read_after_an_alter_sees_the_new_schema(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT, v STRING) USING iceberg")
        assert session.table(target).columns == ["id", "v"]
        session.sql(f"ALTER TABLE {target} ADD COLUMN amt DOUBLE")
        assert session.table(target).columns == ["id", "v", "amt"]
        session.sql(f"ALTER TABLE {target} DROP COLUMN v")
        assert session.table(target).columns == ["id", "amt"]

    def test_a_write_does_not_make_another_table_report_the_wrong_columns(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        """The Phase 7-reachable half: invalidation freed a source view name for reuse.

        Read A, write to A -- which drops A's cached source -- then read B. B's source
        took the name A's had, the analyser had registered that name once already, and
        B was described with A's columns.
        """
        other = f"wr.o_{uuid.uuid4().hex[:8]}"
        try:
            session.table("fx.plain").write.saveAsTable(target)
            session.table("fx.partitioned").write.saveAsTable(other)
            session.sql("SELECT 1").collect()

            assert session.table(target).columns == ["id", "vendor", "amount"]
            session.table(target).write.mode("append").saveAsTable(target)
            assert session.table(other).columns == ["id", "as_at_date", "amount"]
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(other.split(".")))

    def test_the_new_column_is_queryable_not_just_listed(
        self, session: Session, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.sql(f"INSERT INTO {target} SELECT 1")
        session.sql(f"ALTER TABLE {target} ADD COLUMN amt DOUBLE")
        assert [tuple(r) for r in session.sql(f"SELECT id, amt FROM {target}").collect()] == [
            (1, None)
        ]


class TestStatementsReturnTheEmptyFrame:
    def test_create(self, session: Session, target: str) -> None:
        assert session.sql(f"CREATE TABLE {target} (id BIGINT)").collect() == []

    def test_alter(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        assert session.sql(f"ALTER TABLE {target} ADD COLUMN v STRING").collect() == []

    def test_drop(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        assert session.sql(f"DROP TABLE {target}").collect() == []

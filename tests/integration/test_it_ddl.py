"""SQL DDL against the real REST catalog.

DDL is the one area where a local `SqlCatalog` and a REST catalog are genuinely
different systems rather than the same system with different storage. A `CREATE TABLE`
here is an HTTP request that allocates a metadata location; an `ALTER` is a commit with
a requirement assertion that another writer may have invalidated; a `DROP` is a catalog
operation that may or may not purge the data behind it. None of that is exercised by
sqlite.

Every table is a throwaway in the integration namespace, created and dropped through the
guard. Schema evolution is asserted **through a read** wherever possible -- a column that
the catalog reports but which cannot be selected has not really been added.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    AnalysisException,
    TableAlreadyExistsException,
    TableNotFoundError,
)
from icetl.sql import functions as F

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


@pytest.fixture
def created(session: Session, target: str) -> str:
    """An empty table with a known schema, dropped by `target`."""
    session.sql(f"CREATE TABLE {target} (id BIGINT, name STRING, amount DOUBLE)")
    return target


class TestCreateTable:
    def test_a_table_is_created_with_the_declared_columns(
        self, session: Session, created: str
    ) -> None:
        assert session.table(created).columns == ["id", "name", "amount"]

    def test_the_declared_types_survive_the_catalog(self, session: Session, created: str) -> None:
        types = dict(session.table(created).dtypes)
        assert types["id"] == "bigint"
        assert types["name"] == "string"
        assert types["amount"] == "double"

    def test_a_created_table_starts_empty(self, session: Session, created: str) -> None:
        assert session.table(created).count() == 0

    def test_creating_twice_is_refused(self, session: Session, created: str) -> None:
        with pytest.raises(TableAlreadyExistsException):
            session.sql(f"CREATE TABLE {created} (id BIGINT)")

    def test_if_not_exists_is_not_refused(self, session: Session, created: str) -> None:
        session.sql(f"CREATE TABLE IF NOT EXISTS {created} (id BIGINT)")
        assert session.table(created).columns == ["id", "name", "amount"]

    def test_a_not_null_column_is_recorded_as_required(
        self, session: Session, catalog: Catalog, target: str
    ) -> None:
        """Visible through Iceberg's schema, not through DuckDB's."""
        session.sql(f"CREATE TABLE {target} (id BIGINT NOT NULL, name STRING)")
        fields = {f.name: f.required for f in catalog.load_table(target).schema().fields}
        assert fields["id"] is True
        assert fields["name"] is False

    def test_create_table_as_select_copies_the_rows(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {zones}")
        assert session.table(target).count() == session.table(zones).count()

    def test_create_table_as_select_copies_real_data(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        """Real timestamps and NULLs through a CTAS on a REST catalog."""
        source = session.table(trips_small)
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        written = session.table(target)
        assert written.count() == source.count()
        assert (
            written.filter(F.col("store_and_fwd_flag").isNull()).count()
            == source.filter(F.col("store_and_fwd_flag").isNull()).count()
        )


class TestDropTable:
    def test_a_dropped_table_is_gone(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        session.sql(f"DROP TABLE {target}")
        with pytest.raises(TableNotFoundError):
            session.table(target).count()

    def test_dropping_a_missing_table_is_refused(self, session: Session, namespace: str) -> None:
        missing = f"{namespace}.never_created_{uuid.uuid4().hex[:8]}"
        with pytest.raises(TableNotFoundError):
            session.sql(f"DROP TABLE {missing}")

    def test_if_exists_is_not_refused(self, session: Session, namespace: str) -> None:
        missing = f"{namespace}.never_created_{uuid.uuid4().hex[:8]}"
        session.sql(f"DROP TABLE IF EXISTS {missing}")


class TestAddAndDropColumns:
    def test_an_added_column_can_be_selected(self, session: Session, created: str) -> None:
        session.sql(f"ALTER TABLE {created} ADD COLUMN note STRING")
        assert "note" in session.table(created).columns

    def test_an_added_column_reads_null_for_the_old_rows(
        self, session: Session, zones: str, target: str
    ) -> None:
        """FINDINGS 1.11: a column added by ALTER once made the table unreadable.

        The older data files have no such column, so the read has to synthesise it --
        and reading it as NULL is the correct answer, not an error.
        """
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {zones}")
        rows = session.table(target).count()
        session.sql(f"ALTER TABLE {target} ADD COLUMN note STRING")

        frame = session.table(target)
        assert frame.count() == rows
        assert frame.filter(F.col("note").isNull()).count() == rows

    def test_a_dropped_column_disappears(self, session: Session, created: str) -> None:
        """`name` is field-id 2 of 3, so dropping it leaves `last-column-id` alone."""
        session.sql(f"ALTER TABLE {created} DROP COLUMN name")
        assert "name" not in session.table(created).columns

    def test_the_other_columns_survive_a_drop(self, session: Session, created: str) -> None:
        session.sql(f"ALTER TABLE {created} DROP COLUMN name")
        assert session.table(created).columns == ["id", "amount"]

    def test_the_rows_survive_a_drop(self, session: Session, trips_small: str, target: str) -> None:
        """A middle column of a real 19-column table, so `last-column-id` is unmoved."""
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        rows = session.table(target).count()
        before = session.table(target).columns
        assert before[-1] != "store_and_fwd_flag", "pick a column that is not the last id"

        session.sql(f"ALTER TABLE {target} DROP COLUMN store_and_fwd_flag")
        after = session.table(target)
        assert after.count() == rows
        assert "store_and_fwd_flag" not in after.columns
        assert after.columns == [c for c in before if c != "store_and_fwd_flag"]

    def test_dropping_the_highest_field_id_is_refused_by_a_rest_catalog(
        self, session: Session, created: str
    ) -> None:
        """FINDINGS 2.11 -- a real-catalog-only failure, and the reason this suite exists.

        Iceberg's REST spec says `last-column-id` may never decrease. PyIceberg 0.11.1
        recomputes it from the surviving fields, so dropping the column that *holds*
        the highest id sends a lower value and the catalog rejects the commit:

            IllegalArgumentException: Invalid last column ID: 2 < 3

        Dropping any other column is fine, because the maximum does not move. The local
        `SqlCatalog` performs no such validation, which is why the offline suite passes
        and why this could only be found here.

        Pinned as a characterisation test: it **fails when the bug is fixed**, which is
        the notification wanted. The failure is upstream, so nothing in `src/` is at
        fault -- but note it also escapes icetl's error hierarchy untranslated, which
        is fixable here.
        """
        from pyiceberg.exceptions import BadRequestError

        with pytest.raises(BadRequestError, match="last column ID"):
            session.sql(f"ALTER TABLE {created} DROP COLUMN amount")


class TestRename:
    def test_a_renamed_column_reads_under_its_new_name(
        self, session: Session, zones: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {zones}")
        session.sql(f"ALTER TABLE {target} RENAME COLUMN zone_name TO label")
        frame = session.table(target)
        assert "label" in frame.columns
        assert "zone_name" not in frame.columns

    def test_the_data_follows_the_rename(self, session: Session, zones: str, target: str) -> None:
        """Iceberg renames by field-id, so the values written under the old name stay.

        This is the same hazard `icetl_it.renamed` exists for, reached through DDL.
        """
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {zones}")
        rows = session.table(target).count()
        session.sql(f"ALTER TABLE {target} RENAME COLUMN zone_name TO label")
        frame = session.table(target)
        assert frame.filter(F.col("label").isNull()).count() == 0
        assert frame.count() == rows

    def test_a_renamed_table_answers_to_the_new_name(
        self, session: Session, catalog: Catalog, namespace: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        renamed = f"{namespace}.r_{uuid.uuid4().hex[:8]}"
        try:
            session.sql(f"ALTER TABLE {target} RENAME TO {renamed}")
            assert session.table(renamed).columns == ["id"]
            with pytest.raises(TableNotFoundError):
                session.table(target).count()
        finally:
            from tests.integration.guard import safe_drop

            safe_drop(catalog, renamed)


class TestTableProperties:
    def test_a_property_is_set_and_read_back(
        self, session: Session, catalog: Catalog, created: str
    ) -> None:
        session.sql(f"ALTER TABLE {created} SET TBLPROPERTIES ('owner' = 'integration')")
        assert catalog.load_table(created).properties.get("owner") == "integration"

    def test_a_property_is_unset(self, session: Session, catalog: Catalog, created: str) -> None:
        session.sql(f"ALTER TABLE {created} SET TBLPROPERTIES ('owner' = 'integration')")
        session.sql(f"ALTER TABLE {created} UNSET TBLPROPERTIES ('owner')")
        assert "owner" not in catalog.load_table(created).properties


class TestPartitionEvolution:
    def test_a_partition_field_is_added(
        self, session: Session, catalog: Catalog, trips_small: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        session.sql(f"ALTER TABLE {target} ADD PARTITION FIELD VendorID")
        spec = catalog.load_table(target).spec()
        assert "VendorID" in [field.name for field in spec.fields]

    def test_the_rows_written_before_the_change_still_read(
        self, session: Session, trips_small: str, target: str
    ) -> None:
        """Partition evolution does not rewrite data, so old files keep the old spec."""
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        before = session.table(target).count()
        session.sql(f"ALTER TABLE {target} ADD PARTITION FIELD VendorID")
        assert session.table(target).count() == before

    def test_a_partition_field_is_dropped(
        self, session: Session, catalog: Catalog, trips_small: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        session.sql(f"ALTER TABLE {target} ADD PARTITION FIELD VendorID")
        session.sql(f"ALTER TABLE {target} DROP PARTITION FIELD VendorID")
        spec = catalog.load_table(target).spec()
        active = [f.name for f in spec.fields if f.transform.__class__.__name__ != "VoidTransform"]
        assert "VendorID" not in active


class TestSortOrder:
    def test_a_sort_order_is_recorded(
        self, session: Session, catalog: Catalog, trips_small: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        session.sql(f"ALTER TABLE {target} WRITE ORDERED BY VendorID")
        order = catalog.load_table(target).sort_order()
        assert order.fields, "no sort order was recorded"

    def test_a_sort_order_is_cleared(
        self, session: Session, catalog: Catalog, trips_small: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} AS SELECT * FROM {trips_small}")
        session.sql(f"ALTER TABLE {target} WRITE ORDERED BY VendorID")
        session.sql(f"ALTER TABLE {target} WRITE UNORDERED")
        assert not catalog.load_table(target).sort_order().fields


class TestNamespaces:
    @pytest.fixture
    def scratch_namespace(self, catalog: Catalog) -> Iterator[str]:
        name = f"icetl_it_ns_{uuid.uuid4().hex[:8]}"
        yield name
        with contextlib.suppress(Exception):
            catalog.drop_namespace(name)

    def test_a_namespace_is_created_and_dropped(
        self, session: Session, catalog: Catalog, scratch_namespace: str
    ) -> None:
        session.sql(f"CREATE NAMESPACE {scratch_namespace}")
        assert scratch_namespace in {".".join(n) for n in catalog.list_namespaces()}

        session.sql(f"DROP NAMESPACE {scratch_namespace}")
        assert scratch_namespace not in {".".join(n) for n in catalog.list_namespaces()}

    def test_creating_a_namespace_twice_is_refused(
        self, session: Session, scratch_namespace: str
    ) -> None:
        session.sql(f"CREATE NAMESPACE {scratch_namespace}")
        with pytest.raises(AnalysisException):
            session.sql(f"CREATE NAMESPACE {scratch_namespace}")

    def test_if_not_exists_is_not_refused(self, session: Session, scratch_namespace: str) -> None:
        session.sql(f"CREATE NAMESPACE {scratch_namespace}")
        session.sql(f"CREATE NAMESPACE IF NOT EXISTS {scratch_namespace}")

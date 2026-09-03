"""`session.catalog.*` against the real REST catalog.

This is the surface that answers "what is in the warehouse", and it is the one place
where the local suite is least representative: a sqlite `SqlCatalog` holds six tables in
one namespace that the test itself created, while the real catalog holds namespaces
somebody else made, tables with 317 snapshots, and column metadata that came back over
HTTP.

The tests are written to be **true of any warehouse**, not of this one. They assert that
the real namespaces are listed rather than that there are exactly three; that a table the
test just created shows up; that `listColumns` agrees with the frame's own schema. That
way the suite still means something when the warehouse is re-seeded.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

import pytest

from icetl.errors import NamespaceNotFoundError, TableNotFoundError
from tests.integration.conftest import NAMESPACE, REAL_TABLE, TABLE

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestListing:
    def test_the_real_namespace_is_listed(self, it_session: Session) -> None:
        names = {database.name for database in it_session.catalog.listDatabases()}
        assert NAMESPACE in names

    def test_the_integration_namespace_is_listed(
        self, it_session: Session, namespace: str, replicas: dict[str, object]
    ) -> None:
        names = {database.name for database in it_session.catalog.listDatabases()}
        assert namespace in names

    def test_the_real_table_is_listed_in_its_namespace(self, it_session: Session) -> None:
        names = {table.name for table in it_session.catalog.listTables(NAMESPACE)}
        assert TABLE in names

    def test_listing_a_missing_namespace_is_refused(self, it_session: Session) -> None:
        with pytest.raises(NamespaceNotFoundError):
            it_session.catalog.listTables(f"no_such_ns_{uuid.uuid4().hex[:8]}")

    def test_database_exists_agrees_with_the_listing(self, it_session: Session) -> None:
        assert it_session.catalog.databaseExists(NAMESPACE)
        assert not it_session.catalog.databaseExists(f"absent_{uuid.uuid4().hex[:8]}")

    def test_table_exists_agrees_with_the_listing(self, it_session: Session) -> None:
        assert it_session.catalog.tableExists(REAL_TABLE)
        assert not it_session.catalog.tableExists(f"{NAMESPACE}.absent_{uuid.uuid4().hex[:8]}")


class TestDescribing:
    def test_get_table_returns_the_real_table(self, it_session: Session) -> None:
        described = it_session.catalog.getTable(REAL_TABLE)
        assert described.name == TABLE
        assert described.database == NAMESPACE

    def test_list_columns_agrees_with_the_frames_schema(self, it_session: Session) -> None:
        """Two routes to the same fact, so a disagreement is visible."""
        from_catalog = [column.name for column in it_session.catalog.listColumns(REAL_TABLE)]
        from_frame = it_session.table(REAL_TABLE).columns
        assert from_catalog == from_frame

    def test_list_columns_keeps_the_mixed_case_spellings(self, it_session: Session) -> None:
        names = {column.name for column in it_session.catalog.listColumns(REAL_TABLE)}
        assert "VendorID" in names

    def test_list_columns_reports_the_partition_column(self, it_session: Session) -> None:
        """`tpep_pickup_datetime` is the source of the `month()` partition."""
        columns = {c.name: c for c in it_session.catalog.listColumns(REAL_TABLE)}
        assert "tpep_pickup_datetime" in columns

    def test_getting_a_missing_table_is_refused(self, it_session: Session) -> None:
        with pytest.raises(TableNotFoundError):
            it_session.catalog.getTable(f"{NAMESPACE}.absent_{uuid.uuid4().hex[:8]}")


class TestTheCurrentDatabase:
    def test_the_default_namespace_is_the_current_one(self, it_session: Session) -> None:
        assert it_session.catalog.currentDatabase() == NAMESPACE

    def test_setting_the_current_database_changes_resolution(
        self, session: Session, namespace: str, zones: str
    ) -> None:
        """An unqualified name resolves against whatever the current database is."""
        session.catalog.setCurrentDatabase(namespace)
        bare = zones.split(".", 1)[1]
        assert session.table(bare).count() == session.table(zones).count()

    def test_setting_a_missing_database_is_refused(self, session: Session) -> None:
        with pytest.raises(NamespaceNotFoundError):
            session.catalog.setCurrentDatabase(f"absent_{uuid.uuid4().hex[:8]}")


class TestCreateAndDrop:
    def test_a_created_table_appears_in_the_listing(
        self, session: Session, namespace: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        names = {table.name for table in session.catalog.listTables(namespace)}
        assert target.split(".", 1)[1] in names

    def test_a_dropped_table_leaves_the_listing(
        self, session: Session, namespace: str, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        assert session.catalog.dropTable(target)
        names = {table.name for table in session.catalog.listTables(namespace)}
        assert target.split(".", 1)[1] not in names

    def test_table_exists_tracks_creation_and_removal(self, session: Session, target: str) -> None:
        assert not session.catalog.tableExists(target)
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        assert session.catalog.tableExists(target)
        session.catalog.dropTable(target)
        assert not session.catalog.tableExists(target)


class TestRefresh:
    def test_refresh_picks_up_an_out_of_band_commit(
        self, session: Session, it_session: Session, zones: str, target: str
    ) -> None:
        """Two sessions, one catalog -- the second only sees the write after a refresh.

        This is the multi-writer case a REST catalog exists to support, and it cannot
        be demonstrated against a session that made the write itself.
        """
        session.table(zones).write.saveAsTable(target)
        assert it_session.table(target).count() == session.table(zones).count()

        session.table(zones).write.mode("append").saveAsTable(target)
        it_session.catalog.refreshTable(target)
        assert it_session.table(target).count() == session.table(zones).count() * 2


class TestTempViews:
    """Session-local, so they must **not** reach the catalog."""

    def test_a_temp_view_is_queryable(self, session: Session, zones: str) -> None:
        session.table(zones).createOrReplaceTempView("zone_view")
        assert session.table("zone_view").count() == session.table(zones).count()

    def test_a_temp_view_is_listed_but_marked_temporary(
        self, session: Session, namespace: str
    ) -> None:
        """`listTables` includes temp views, as the reference does -- flagged, not hidden.

        That is what makes `tableExists('v')` and `listTables()` agree with each other.
        """
        session.table(f"{namespace}.zones").createOrReplaceTempView("zone_view")
        listed = {table.name: table for table in session.catalog.listTables(namespace)}
        assert "zone_view" in listed
        assert listed["zone_view"].isTemporary is True
        assert listed["zone_view"].namespace is None

    def test_a_temp_view_never_reaches_the_iceberg_catalog(
        self, session: Session, catalog: Catalog, namespace: str
    ) -> None:
        """The distinction that matters: nothing was created in the warehouse."""
        session.table(f"{namespace}.zones").createOrReplaceTempView("zone_view")
        real = {".".join(identifier) for identifier in catalog.list_tables(namespace)}
        assert f"{namespace}.zone_view" not in real

    def test_a_real_table_is_not_marked_temporary(
        self, session: Session, namespace: str, target: str
    ) -> None:
        """The control for the flag above."""
        session.sql(f"CREATE TABLE {target} (id BIGINT)")
        listed = {table.name: table for table in session.catalog.listTables(namespace)}
        created = listed[target.split(".", 1)[1]]
        assert created.isTemporary is False
        assert created.namespace == [namespace]

    def test_dropping_a_temp_view_removes_it(self, session: Session, zones: str) -> None:
        session.table(zones).createOrReplaceTempView("zone_view")
        assert session.dropTempView("zone_view")
        with pytest.raises(TableNotFoundError):
            session.table("zone_view").count()

    def test_a_temp_view_shadows_nothing_in_another_session(
        self, session: Session, it_session: Session, zones: str
    ) -> None:
        session.table(zones).createOrReplaceTempView("only_here")
        with pytest.raises(TableNotFoundError):
            it_session.table("only_here").count()

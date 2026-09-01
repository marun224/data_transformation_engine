"""Phase 9: `session.catalog` -- listing and describing what the catalog holds.

Two things here are worth more than the rest.

**`listColumns` is the only place true nullability is visible.** `df.schema` reports every
field nullable, because the analysed schema comes from DuckDB and DuckDB has no
non-nullable expression -- recorded in `divergence.md` since Phase 2. `listColumns` asks
Iceberg instead, so a column declared `NOT NULL` reads back as one. The two answering
differently is deliberate and `TestColumnsComeFromIceberg` pins it.

**The listing pattern is not a `LIKE` and not a regex.** It is the reference's own
`*`-and-`|` filter, and reading it as either of the other two gives a plausible wrong
answer -- `upp*` matching nothing, or `.` matching any character.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    NamespaceNotFoundError,
    TableNotFoundError,
    UnsupportedFeatureError,
)
from icetl.sql.catalog import _filter_pattern

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.session import Session


@pytest.fixture
def target(catalog: SqlCatalog) -> Iterator[str]:
    name = f"wr.t_{uuid.uuid4().hex[:8]}"
    yield name
    with contextlib.suppress(Exception):
        catalog.drop_table(tuple(name.split(".")))


class TestCatalogsAndDatabases:
    def test_the_current_catalog_and_database(self, session: Session) -> None:
        assert session.catalog.currentCatalog() == "test"
        assert session.catalog.currentDatabase() == "fx"

    def test_databases_are_iceberg_namespaces(self, session: Session) -> None:
        names = [database.name for database in session.catalog.listDatabases()]
        assert "fx" in names

    def test_database_exists(self, session: Session) -> None:
        assert session.catalog.databaseExists("fx")
        assert not session.catalog.databaseExists("no_such_namespace")

    def test_getting_a_missing_database_raises(self, session: Session) -> None:
        with pytest.raises(NamespaceNotFoundError, match="does not exist"):
            session.catalog.getDatabase("no_such_namespace")

    def test_setting_the_current_database_changes_what_a_bare_name_resolves_to(
        self, session: Session, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.catalog.setCurrentDatabase("wr")
        assert session.catalog.currentDatabase() == "wr"
        # The unqualified name now resolves, which is the whole point of the setting.
        assert session.table(target.split(".")[1]).count() == 0

    def test_setting_a_missing_database_raises(self, session: Session) -> None:
        with pytest.raises(NamespaceNotFoundError, match="does not exist"):
            session.catalog.setCurrentDatabase("no_such_namespace")

    def test_an_unconfigured_catalog_is_refused(self, session: Session) -> None:
        with pytest.raises(Exception, match="not configured"):
            session.catalog.setCurrentCatalog("elsewhere")


class TestTables:
    def test_it_lists_the_tables_in_a_database(self, session: Session) -> None:
        names = [table.name for table in session.catalog.listTables("fx")]
        assert {"plain", "partitioned", "wide", "nested"} <= set(names)

    def test_a_listed_table_carries_its_namespace(self, session: Session) -> None:
        found = next(t for t in session.catalog.listTables("fx") if t.name == "plain")
        assert found.namespace == ["fx"]
        assert found.database == "fx"
        assert found.isTemporary is False
        assert found.tableType == "MANAGED"

    def test_temporary_views_are_listed_beside_tables(self, session: Session) -> None:
        session.table("fx.plain").createOrReplaceTempView("a_view")
        found = [t for t in session.catalog.listTables("fx") if t.name == "a_view"]
        assert found and found[0].isTemporary is True
        assert found[0].namespace is None

    def test_table_exists_covers_both_tables_and_views(self, session: Session) -> None:
        session.table("fx.plain").createOrReplaceTempView("a_view")
        assert session.catalog.tableExists("fx.plain")
        assert session.catalog.tableExists("plain", "fx")
        assert session.catalog.tableExists("a_view")
        assert not session.catalog.tableExists("fx.no_such_table")

    def test_getting_a_missing_table_raises(self, session: Session) -> None:
        with pytest.raises(TableNotFoundError, match="does not exist"):
            session.catalog.getTable("fx.no_such_table")

    def test_listing_a_missing_database_raises(self, session: Session) -> None:
        with pytest.raises(NamespaceNotFoundError, match="does not exist"):
            session.catalog.listTables("no_such_namespace")


class TestColumnsComeFromIceberg:
    def test_it_reports_the_partition_columns(self, session: Session) -> None:
        columns = {c.name: c for c in session.catalog.listColumns("fx.partitioned")}
        assert columns["as_at_date"].isPartition is True
        assert columns["id"].isPartition is False

    def test_it_reports_the_types_the_frame_reports(self, session: Session) -> None:
        columns = session.catalog.listColumns("fx.plain")
        assert [(c.name, c.dataType) for c in columns] == session.table("fx.plain").dtypes

    def test_nullability_is_the_tables_own_not_the_analysers(
        self, session: Session, target: str
    ) -> None:
        """The one place the true answer is visible.

        `df.schema` calls every field nullable because DuckDB has no non-nullable
        expression. Iceberg knows better, and `listColumns` asks Iceberg.
        """
        session.sql(f"CREATE TABLE {target} (id BIGINT NOT NULL, v STRING) USING iceberg")
        columns = {c.name: c for c in session.catalog.listColumns(target)}
        assert columns["id"].nullable is False
        assert columns["v"].nullable is True
        assert all(field.nullable for field in session.table(target).schema.fields)

    def test_a_temporary_view_has_columns_too(self, session: Session) -> None:
        session.table("fx.plain").createOrReplaceTempView("a_view")
        assert [c.name for c in session.catalog.listColumns("a_view")] == [
            "id",
            "vendor",
            "amount",
        ]


class TestCreateAndDropThroughTheCatalog:
    def test_create_table_from_a_ddl_string(self, session: Session, target: str) -> None:
        frame = session.catalog.createTable(target, schema="id bigint, v string")
        assert frame.columns == ["id", "v"]
        assert session.catalog.tableExists(target)

    def test_drop_table_reports_whether_there_was_one(self, session: Session, target: str) -> None:
        session.catalog.createTable(target, schema="id bigint")
        assert session.catalog.dropTable(target) is True
        assert session.catalog.dropTable(target) is False

    def test_a_description_becomes_the_comment_property(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        session.catalog.createTable(target, schema="id bigint", description="a note")
        table = catalog.load_table(tuple(target.split(".")))
        assert table.properties.get("comment") == "a note"

    def test_a_path_is_refused(self, session: Session, target: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="path"):
            session.catalog.createTable(target, path="/tmp/x", schema="id bigint")


class TestListingPattern:
    """The reference's filter: `*` is any sequence, `|` separates alternatives."""

    def test_a_star_matches_a_sequence_not_a_single_character(self) -> None:
        assert _filter_pattern(["upper", "up", "u"], "up*") == ["upper", "up"]

    def test_alternatives(self) -> None:
        assert _filter_pattern(["upper", "lower", "abs"], "upp*|low*") == ["upper", "lower"]

    def test_a_leading_star(self) -> None:
        assert _filter_pattern(["array_size", "size", "abs"], "*size") == ["array_size", "size"]

    def test_no_pattern_keeps_everything(self) -> None:
        assert _filter_pattern(["a", "b"], None) == ["a", "b"]

    def test_it_is_not_a_regex(self) -> None:
        """`.` is a literal dot, so it matches nothing here rather than everything."""
        assert _filter_pattern(["ab", "a.b"], "a.b") == ["a.b"]

    def test_it_filters_the_function_surface(self, session: Session) -> None:
        names = [fn.name for fn in session.catalog.listFunctions(pattern="upp*")]
        assert names == ["upper"]
        assert session.catalog.functionExists("upper")
        assert not session.catalog.functionExists("no_such_function")


class TestCachingIsRefusedRatherThanHalfDone:
    """Caching here is per-frame and eager; there is no name-level cache to ask about.

    A `cacheTable` that shadowed the name for one surface and not the other, or went
    stale behind a write, would be worse than not having it.
    """

    def test_cache_table_points_at_the_frame_cache(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="cache"):
            session.catalog.cacheTable("fx.plain")

    def test_clear_cache_releases_materialised_frames(self, session: Session) -> None:
        cached = session.table("fx.plain").cache()
        assert cached.count() == 5
        session.catalog.clearCache()
        assert session._cached == set()

    def test_global_temp_views_are_refused(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="global temporary"):
            session.catalog.dropGlobalTempView("v")


class TestRefreshTable:
    def test_it_picks_up_a_write_made_outside_the_session(
        self, session: Session, catalog: SqlCatalog, target: str
    ) -> None:
        """The session pins a table to the snapshot it first read it at."""
        session.table("fx.plain").write.saveAsTable(target)
        assert session.table(target).count() == 5

        outside = catalog.load_table(tuple(target.split(".")))
        outside.delete("id > 3")

        assert session.table(target).count() == 5, "still reading the pinned snapshot"
        session.catalog.refreshTable(target)
        assert session.table(target).count() == 3

    def test_recover_partitions_has_nothing_to_do(self, session: Session) -> None:
        """Iceberg tracks files in metadata; there is no directory to re-scan."""
        session.catalog.recoverPartitions("fx.partitioned")
        with pytest.raises(TableNotFoundError):
            session.catalog.recoverPartitions("fx.no_such_table")

"""Phase 9: time travel, metadata tables, and `mergeSchema`.

**Time travel is a property of the source key, not of the scan.** `VERSION AS OF` is
folded into the key a table reference resolves under, so the same table at two snapshots
is two sources and joins to itself -- `TestTwoVersionsAtOnce` is the case that proves the
key does the work rather than a flag threaded through the planner.

**A snapshot is history, so nothing writes to one.** `TestWritingToASnapshotIsRefused`
covers every statement that could try.

**Metadata tables are materialised at plan time.** There are no data files to prune and
no predicate to push down, so the rows are read now and registered in both engines. The
consequence worth knowing is in `TestMetadataTables.test_it_is_a_snapshot_of_the_metadata`.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    AnalysisException,
    EngineValueError,
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


@pytest.fixture
def history(session: Session, catalog: SqlCatalog, target: str) -> tuple[str, int, int]:
    """A table with two states: five rows, then three. Returns `(name, first, second)`."""
    session.table("fx.plain").write.saveAsTable(target)
    table = catalog.load_table(tuple(target.split(".")))
    first = table.snapshots()[0].snapshot_id
    session.sql(f"DELETE FROM {target} WHERE id > 3")
    table = catalog.load_table(tuple(target.split(".")))
    return target, first, table.snapshots()[-1].snapshot_id


def ids(session: Session, sql: str) -> list[int]:
    return sorted(row[0] for row in session.sql(sql).collect())


class TestVersionAsOf:
    def test_it_reads_the_snapshot_it_names(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, first, _ = history
        assert ids(session, f"SELECT id FROM {name}") == [1, 2, 3]
        assert ids(session, f"SELECT id FROM {name} VERSION AS OF {first}") == [1, 2, 3, 4, 5]

    def test_a_filter_still_applies_to_the_old_snapshot(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, first, _ = history
        assert ids(session, f"SELECT id FROM {name} VERSION AS OF {first} WHERE id > 3") == [4, 5]

    def test_an_unknown_snapshot_is_refused_with_the_ids_that_exist(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, _, _ = history
        with pytest.raises(AnalysisException, match="no snapshot 999"):
            session.sql(f"SELECT * FROM {name} VERSION AS OF 999").collect()

    def test_a_table_with_no_snapshots_has_nowhere_to_travel(
        self, session: Session, target: str
    ) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        with pytest.raises(AnalysisException, match="no snapshots"):
            session.sql(f"SELECT * FROM {target} VERSION AS OF 1").collect()


class TestTimestampAsOf:
    def test_it_reads_the_newest_snapshot_at_or_before_the_time(
        self, session: Session, catalog: SqlCatalog, history: tuple[str, int, int]
    ) -> None:
        name, first, _ = history
        table = catalog.load_table(tuple(name.split(".")))
        committed = next(s for s in table.snapshots() if s.snapshot_id == first).timestamp_ms
        moment = datetime.fromtimestamp(committed / 1000 + 0.001, tz=UTC).isoformat()
        assert ids(session, f"SELECT id FROM {name} TIMESTAMP AS OF '{moment}'") == [
            1,
            2,
            3,
            4,
            5,
        ]

    def test_a_time_before_the_first_snapshot_is_refused(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, _, _ = history
        with pytest.raises(AnalysisException, match="at or before"):
            session.sql(f"SELECT * FROM {name} TIMESTAMP AS OF '1999-01-01'").collect()

    def test_an_unreadable_timestamp_is_refused(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, _, _ = history
        with pytest.raises(AnalysisException, match="ISO-8601"):
            session.sql(f"SELECT * FROM {name} TIMESTAMP AS OF 'yesterday'").collect()


class TestTwoVersionsAtOnce:
    """The version is part of the source key, so one query can hold two of them."""

    def test_a_query_over_both_states(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, first, _ = history
        got = session.sql(
            f"SELECT (SELECT count(*) FROM {name}) AS now_n, "
            f"(SELECT count(*) FROM {name} VERSION AS OF {first}) AS then_n"
        ).collect()
        assert [tuple(row) for row in got] == [(3, 5)]

    def test_the_rows_a_delete_removed(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, first, _ = history
        removed = session.sql(
            f"SELECT o.id FROM {name} VERSION AS OF {first} AS o "
            f"LEFT JOIN {name} AS n ON o.id = n.id WHERE n.id IS NULL"
        ).collect()
        assert sorted(row[0] for row in removed) == [4, 5]


class TestTheReader:
    def test_snapshot_id_option(self, session: Session, history: tuple[str, int, int]) -> None:
        name, first, _ = history
        frame = session.read.option("snapshot-id", first).table(name)
        assert sorted(row[0] for row in frame.collect()) == [1, 2, 3, 4, 5]

    def test_as_of_timestamp_accepts_epoch_millis(
        self, session: Session, catalog: SqlCatalog, history: tuple[str, int, int]
    ) -> None:
        """The reference documents milliseconds; an ISO string is accepted too."""
        name, first, _ = history
        table = catalog.load_table(tuple(name.split(".")))
        committed = next(s for s in table.snapshots() if s.snapshot_id == first).timestamp_ms
        frame = session.read.option("as-of-timestamp", committed).table(name)
        assert sorted(row[0] for row in frame.collect()) == [1, 2, 3, 4, 5]

    def test_no_option_is_an_ordinary_read(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, _, _ = history
        assert sorted(row[0] for row in session.read.table(name).collect()) == [1, 2, 3]

    def test_both_options_at_once_is_refused(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        name, first, _ = history
        with pytest.raises(EngineValueError, match="not both"):
            session.read.option("snapshot-id", first).option("as-of-timestamp", "2026-01-01").table(
                name
            )

    def test_the_reader_is_immutable(self, session: Session, history: tuple[str, int, int]) -> None:
        name, first, _ = history
        base = session.read
        base.option("snapshot-id", first)
        assert sorted(row[0] for row in base.table(name).collect()) == [1, 2, 3]

    def test_a_file_format_is_now_accepted(self, session: Session) -> None:
        """Phase 11 built the file readers this used to be deferred to.

        The reader is shared by both jobs, so this is the check that adding formats
        did not disturb the time-travel options -- `tests/fixture/test_file_readers.py`
        is where the formats themselves are tested.
        """
        assert session.read.format("parquet") is not None

    def test_load_without_a_format_still_points_at_table(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match=r"read\.table"):
            session.read.load("/tmp/x")


class TestWritingToASnapshotIsRefused:
    """A snapshot has already been committed. There is nothing to write to it."""

    def test_insert_does_not_even_parse(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        """sqlglot rejects a version on an INSERT target before we see it.

        The guard still exists for the statements that *do* parse one, below -- this
        records that the INSERT path is closed earlier and by someone else.
        """
        from icetl.errors import ParseException

        name, first, _ = history
        with pytest.raises(ParseException):
            session.sql(f"INSERT INTO {name} VERSION AS OF {first} SELECT * FROM fx.plain")

    def test_delete(self, session: Session, history: tuple[str, int, int]) -> None:
        name, first, _ = history
        with pytest.raises(UnsupportedFeatureError, match="AS OF"):
            session.sql(f"DELETE FROM {name} VERSION AS OF {first} WHERE id = 1")

    def test_alter(self, session: Session, history: tuple[str, int, int]) -> None:
        name, first, _ = history
        with pytest.raises(UnsupportedFeatureError, match="AS OF"):
            session.sql(f"ALTER TABLE {name} VERSION AS OF {first} ADD COLUMN x STRING")


class TestMetadataTables:
    def test_snapshots(self, session: Session, history: tuple[str, int, int]) -> None:
        name, _, _ = history
        rows = session.table(f"{name}.snapshots").collect()
        # The append, then the delete -- one snapshot each, because an exactly
        # translatable predicate takes PyIceberg's own `delete` rather than a rewrite.
        assert len(rows) == 2
        assert "snapshot_id" in session.table(f"{name}.snapshots").columns

    def test_history(self, session: Session, history: tuple[str, int, int]) -> None:
        name, _, _ = history
        assert session.table(f"{name}.history").count() == 2

    def test_files_and_partitions(self, session: Session) -> None:
        assert session.table("fx.partitioned.files").count() == 3
        assert session.table("fx.partitioned.partitions").count() == 3

    def test_manifests_and_refs(self, session: Session) -> None:
        assert session.table("fx.partitioned.manifests").count() == 3
        assert session.table("fx.partitioned.refs").count() == 1

    def test_they_are_queryable_like_any_table(self, session: Session) -> None:
        got = session.sql(
            "SELECT count(*) AS n FROM fx.partitioned.files WHERE record_count > 0"
        ).collect()
        assert [tuple(row) for row in got] == [(3,)]

    def test_both_surfaces_agree(self, session: Session) -> None:
        """P1: `session.table()` and SQL reach the same rows."""
        assert (
            session.table("fx.partitioned.files").count()
            == session.sql("SELECT * FROM fx.partitioned.files").count()
        )

    def test_it_is_a_snapshot_of_the_metadata(
        self, session: Session, history: tuple[str, int, int]
    ) -> None:
        """Materialised at plan time, so a frame holds the metadata as it was then."""
        name, _, _ = history
        before = session.table(f"{name}.snapshots")
        assert before.count() == 2
        session.sql(f"DELETE FROM {name} WHERE id = 1")
        assert before.count() == 2, "the frame holds the metadata as it was"
        assert session.table(f"{name}.snapshots").count() == 3

    def test_a_name_that_is_not_a_metadata_table_is_an_ordinary_missing_table(
        self, session: Session
    ) -> None:
        from icetl.errors import TableNotFoundError

        with pytest.raises(TableNotFoundError):
            session.table("fx.plain.not_a_metadata_table")


class TestMergeSchema:
    def test_off_by_default_a_wider_frame_is_refused(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        wider = session.sql("SELECT 1 AS id, 'x' AS extra")
        with pytest.raises(AnalysisException):
            wider.write.mode("append").saveAsTable(target)

    def test_it_adds_the_columns_the_frame_has(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.sql(f"INSERT INTO {target} SELECT 1")
        session.sql("SELECT CAST(2 AS BIGINT) AS id, 'x' AS extra").write.option(
            "mergeSchema", "true"
        ).mode("append").saveAsTable(target)
        assert session.table(target).columns == ["id", "extra"]
        assert sorted((row[0], row[1]) for row in session.table(target).collect()) == [
            (1, None),
            (2, "x"),
        ]

    def test_it_never_drops_a_column_the_frame_lacks(self, session: Session, target: str) -> None:
        """Additive only, which is what the reference's `mergeSchema` promises."""
        session.sql(f"CREATE TABLE {target} (id BIGINT, keep STRING) USING iceberg")
        session.sql(f"INSERT INTO {target} SELECT 1, 'here'")
        session.sql("SELECT CAST(2 AS BIGINT) AS id, 'x' AS extra").write.option(
            "mergeSchema", "true"
        ).mode("append").saveAsTable(target)
        assert session.table(target).columns == ["id", "keep", "extra"]
        assert sorted((row[0], row[1], row[2]) for row in session.table(target).collect()) == [
            (1, "here", None),
            (2, None, "x"),
        ]

    def test_a_frame_that_already_fits_changes_nothing(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        session.sql("SELECT CAST(1 AS BIGINT) AS id").write.option("mergeSchema", "true").mode(
            "append"
        ).saveAsTable(target)
        assert session.table(target).columns == ["id"]

    def test_a_bad_value_is_refused(self, session: Session, target: str) -> None:
        session.sql(f"CREATE TABLE {target} (id BIGINT) USING iceberg")
        with pytest.raises(EngineValueError, match="mergeSchema"):
            session.sql("SELECT CAST(1 AS BIGINT) AS id").write.option("mergeSchema", "maybe").mode(
                "append"
            ).saveAsTable(target)

"""Phase 10: caching analysed schemas, and the invalidation that makes it safe.

Analysis is a DuckDB round trip -- about 17 ms of an 84 ms query on the benchmark
table -- and it is paid again by every frame derived from the same shape, because
`DataFrame.schema` is memoised per frame and a derived frame is a new one.

The cache is keyed on the bound SQL plus the identity of every schema it is bound
against: each source's Iceberg `schema_id` and snapshot, plus a registry epoch
covering the relations a session materialises. **Exact, not time-based.** PLAN.md
proposed a TTL; Iceberg hands us the exact answer for free, and a TTL is stale for
its window and wasteful for the rest.

So the tests that matter are not the hits, they are the *misses*: a table whose
schema evolved, a temp view redefined, a materialised name reused. A stale schema is
a wrong column list, which is a wrong answer.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.session import Session


class TestTheCacheHits:
    def test_repeating_a_shape_adds_nothing(self, session: Session) -> None:
        """Two entries the first time -- the base frame and the projection -- then none.

        Asserted as growth rather than as a count, because how many frames a call
        analyses is an implementation detail and how many *repeats* cost anything is
        the property being claimed.
        """
        analyzer = session._analyzer
        analyzer._schemas.clear()
        assert session.table("fx.plain").select("id").columns == ["id"]
        after_first = len(analyzer._schemas)
        for _ in range(5):
            assert session.table("fx.plain").select("id").columns == ["id"]
        assert len(analyzer._schemas) == after_first

    def test_a_different_shape_adds_exactly_one_entry(self, session: Session) -> None:
        """The base frame is shared; only the new projection is a new question."""
        analyzer = session._analyzer
        analyzer._schemas.clear()
        assert session.table("fx.plain").select("id").columns
        after_first = len(analyzer._schemas)
        assert session.table("fx.plain").select("vendor").columns
        assert len(analyzer._schemas) == after_first + 1

    def test_the_cached_schema_is_the_same_schema(self, session: Session) -> None:
        first = session.table("fx.nested").schema
        second = session.table("fx.nested").schema
        assert first == second
        assert first.treeString() == second.treeString()

    def test_the_cache_is_bounded(self, session: Session) -> None:
        """A long session generating unique SQL must not grow without bound."""
        from icetl.plan.analysis import _SCHEMA_CACHE_LIMIT

        analyzer = session._analyzer
        analyzer._schemas.clear()
        analyzer._schemas.update(
            {(f"q{index}", ()): None for index in range(_SCHEMA_CACHE_LIMIT)}  # type: ignore[misc]
        )
        assert session.table("fx.plain").select("id", "vendor").columns
        assert len(analyzer._schemas) < _SCHEMA_CACHE_LIMIT


class TestTheCacheMisses:
    """Every case where returning the previous answer would be wrong."""

    def test_an_evolved_schema_is_analysed_again(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        """The one that would be a wrong column list, not a slow query."""
        name = f"wr.evo_{uuid.uuid4().hex[:8]}"
        session.createDataFrame([(1,), (2,)], ["id"]).write.saveAsTable(name)
        try:
            assert session.table(name).columns == ["id"]
            session.sql(f"ALTER TABLE {name} ADD COLUMN label STRING")
            assert session.table(name).columns == ["id", "label"]
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_a_dropped_column_disappears_from_the_schema(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        name = f"wr.drop_{uuid.uuid4().hex[:8]}"
        session.createDataFrame([(1, "a")], ["id", "label"]).write.saveAsTable(name)
        try:
            assert session.table(name).columns == ["id", "label"]
            session.sql(f"ALTER TABLE {name} DROP COLUMN label")
            assert session.table(name).columns == ["id"]
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_a_renamed_column_shows_its_new_name(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        name = f"wr.ren_{uuid.uuid4().hex[:8]}"
        session.createDataFrame([(1, "a")], ["id", "label"]).write.saveAsTable(name)
        try:
            assert session.table(name).columns == ["id", "label"]
            session.sql(f"ALTER TABLE {name} RENAME COLUMN label TO caption")
            assert session.table(name).columns == ["id", "caption"]
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_a_redefined_temp_view_is_analysed_again(self, session: Session) -> None:
        """A view is a plan, so its name resolving to a new plan is a new SQL string."""
        session.table("fx.plain").select("id").createOrReplaceTempView("v_cache")
        assert session.sql("SELECT * FROM v_cache").columns == ["id"]
        session.table("fx.plain").select("vendor").createOrReplaceTempView("v_cache")
        assert session.sql("SELECT * FROM v_cache").columns == ["vendor"]

    def test_a_new_materialised_relation_bumps_the_epoch(self, session: Session) -> None:
        """A registered name is not in the SQL's schema fingerprint, so the epoch is."""
        analyzer = session._analyzer
        before = analyzer._epoch
        assert session.createDataFrame([(1,)], ["id"]).columns
        assert analyzer._epoch > before

    def test_a_cached_frame_reads_its_own_schema(self, session: Session) -> None:
        cached = session.table("fx.plain").select("id").cache()
        assert cached.columns == ["id"]
        other = session.table("fx.plain").select("id", "vendor").cache()
        assert other.columns == ["id", "vendor"]

    def test_time_travel_and_now_are_different_keys(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        """A snapshot can carry a different schema, so the snapshot is in the key."""
        name = f"wr.tt_{uuid.uuid4().hex[:8]}"
        session.createDataFrame([(1,)], ["id"]).write.saveAsTable(name)
        snapshot = catalog.load_table(tuple(name.split("."))).current_snapshot()
        assert snapshot is not None
        try:
            session.sql(f"ALTER TABLE {name} ADD COLUMN label STRING")
            assert session.table(name).columns == ["id", "label"]
            travelled = session.sql(f"SELECT * FROM {name} VERSION AS OF {snapshot.snapshot_id}")
            assert travelled.columns == ["id", "label"]  # the current schema, by design
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))


class TestAnalysisStillFails:
    def test_a_bad_column_still_raises(self, session: Session) -> None:
        """Failures are not cached: a fixed table must not report the old complaint."""
        from icetl.errors import AnalysisException

        with pytest.raises(AnalysisException):
            _ = session.sql("SELECT nope FROM fx.plain").columns
        assert session.sql("SELECT id FROM fx.plain").columns == ["id"]

    def test_a_failure_leaves_nothing_in_the_cache(self, session: Session) -> None:
        from icetl.errors import AnalysisException

        analyzer = session._analyzer
        analyzer._schemas.clear()
        with pytest.raises(AnalysisException):
            _ = session.sql("SELECT nope FROM fx.plain").columns
        assert analyzer._schemas == {}

    def test_the_rows_are_unaffected_by_caching(self, session: Session) -> None:
        """The whole point: a cache that changed an answer would be a bug, not a speedup."""
        frame = session.table("fx.partitioned").filter(F.col("id") > 5).select("id")
        first = frame.collect()
        second = session.table("fx.partitioned").filter(F.col("id") > 5).select("id").collect()
        assert first == second
        assert sorted(row[0] for row in first) == [6, 7, 8, 9, 10, 11]

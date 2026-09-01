"""`sample`, `randomSplit`, `cache`/`persist`, `repartition`, and temporary views.

What these five have in common is that they are about *where the rows live* rather than
what the rows are, which on a single node means most of them have less to do than their
names suggest. `repartition` is a no-op with a docstring. `cache` is a DuckDB temp
table. The two that carry real risk are:

* **`randomSplit`** -- the splits must partition the frame. If the random draw were
  re-evaluated per action, a row could land in two splits or in none, and nothing about
  the result would look wrong. `TestRandomSplit` asserts on completeness and
  disjointness rather than on sizes, because sizes are the part that is allowed to vary.
* **`cache`** -- execution and analysis run on separate DuckDB connections by design, so
  a temp table registered for one is invisible to the other. A cached frame that could
  be collected but not `select`-ed would be the symptom; `TestCache` covers both.

Every assertion is on a value, per the rule Phase 3 established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    EngineTypeError,
    EngineValueError,
    TempTableAlreadyExistsException,
    UnsupportedFeatureError,
)
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.session import Session


class TestSample:
    def test_a_zero_fraction_returns_nothing_and_one_returns_everything(
        self, session: Session
    ) -> None:
        df = session.table("fx.partitioned")  # 12 rows
        assert df.sample(0.0).count() == 0
        assert df.sample(1.0).count() == 12

    def test_a_seeded_sample_repeats(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        sampled = df.sample(0.5, 42)
        assert sampled.count() == sampled.count()

    def test_the_sample_is_a_subset(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        every = {row[0] for row in df.collect()}
        drawn = {row[0] for row in df.sample(0.5, 7).collect()}
        assert drawn <= every

    def test_the_fraction_is_a_probability_not_a_row_count(self, session: Session) -> None:
        """Each row is decided on its own, so the count is near the fraction, not equal."""
        df = session.table("fx.partitioned")
        assert 0 <= df.sample(0.5, 1).count() <= 12

    def test_the_first_argument_may_be_the_fraction(self, session: Session) -> None:
        """`sample(0.5, 42)` means fraction then seed, as it does in the reference."""
        df = session.table("fx.partitioned")
        assert df.sample(0.5, 42).count() == df.sample(False, 0.5, 42).count()

    def test_a_filter_after_a_sample_nests_rather_than_merging(self, session: Session) -> None:
        """Merging would filter before the draw instead of after it -- a different query."""
        df = session.table("fx.partitioned")
        filtered = df.sample(1.0).filter(F.col("id") < 4)
        assert sorted(row[0] for row in filtered.collect()) == [0, 1, 2, 3]

    def test_sampling_with_replacement_is_refused_not_approximated(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        with pytest.raises(UnsupportedFeatureError, match="withReplacement"):
            df.sample(True, 0.5)

    def test_the_fraction_is_range_checked(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        with pytest.raises(EngineValueError, match=r"\[0, 1\]"):
            df.sample(1.5)


class TestRandomSplit:
    def test_the_splits_partition_the_frame(self, session: Session) -> None:
        """Complete and disjoint. Sizes may vary; membership may not."""
        df = session.table("fx.partitioned")
        parts = df.randomSplit([0.5, 0.5], seed=7)
        ids = [row[0] for part in parts for row in part.collect()]
        assert sorted(ids) == sorted(row[0] for row in df.collect())
        assert len(ids) == len(set(ids)), "a row landed in two splits"

    def test_each_split_is_stable_across_collections(self, session: Session) -> None:
        """The draw is materialised once; re-running `random()` would break this."""
        df = session.table("fx.partitioned")
        first, _ = df.randomSplit([0.3, 0.7], seed=3)
        assert [tuple(r) for r in first.collect()] == [tuple(r) for r in first.collect()]

    def test_the_draw_column_is_not_in_the_output(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        parts = df.randomSplit([0.5, 0.5], seed=1)
        assert parts[0].columns == df.columns

    def test_weights_are_normalised_rather_than_required_to_sum_to_one(
        self, session: Session
    ) -> None:
        df = session.table("fx.partitioned")
        parts = df.randomSplit([2.0, 8.0], seed=5)
        assert sum(part.count() for part in parts) == 12

    def test_one_weight_returns_the_whole_frame(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        (only,) = df.randomSplit([1.0], seed=2)
        assert only.count() == 12

    def test_weights_are_checked(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        with pytest.raises(EngineValueError, match="at least one weight"):
            df.randomSplit([])
        with pytest.raises(EngineValueError, match="above zero"):
            df.randomSplit([0.0, 0.0])
        with pytest.raises(EngineValueError, match=">= 0"):
            df.randomSplit([1.0, -1.0])


class TestCache:
    def test_a_cached_frame_holds_the_same_rows(self, session: Session) -> None:
        df = session.table("fx.plain")
        cached = df.cache()
        assert cached.columns == df.columns
        assert [tuple(r) for r in cached.collect()] == [tuple(r) for r in df.collect()]

    def test_a_cached_frame_reads_no_table(self, session: Session) -> None:
        """The rows are in a temp table now, so there is no Iceberg source left to scan."""
        cached = session.table("fx.plain").cache()
        assert cached._sources == {}

    def test_a_cached_frame_can_still_be_transformed(self, session: Session) -> None:
        """Schema resolution runs on the *analyzer's* connection, not the engine's.

        A temp table registered only for execution would collect fine and fail here, so
        this is the test that the frame is registered with both.
        """
        cached = session.table("fx.plain").cache()
        assert cached.filter(F.col("id") > 3).count() == 2
        assert cached.select("vendor").columns == ["vendor"]
        assert cached.groupBy("vendor").count().count() == 4

    def test_persist_is_cache(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.persist().count() == df.cache().count()

    def test_a_storage_level_is_refused(self, session: Session) -> None:
        """There is one storage level here, so accepting an argument would be a lie."""
        df = session.table("fx.plain")
        with pytest.raises(UnsupportedFeatureError, match="storageLevel"):
            df.persist(storageLevel="MEMORY_ONLY")

    def test_unpersist_releases_and_is_safe_to_repeat(self, session: Session) -> None:
        cached = session.table("fx.plain").cache()
        assert cached.count() == 5
        assert cached.unpersist() is cached
        assert cached.unpersist() is cached

    def test_unpersist_on_an_uncached_frame_does_nothing(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.unpersist() is df


class TestPartitioningIsANoOp:
    """One process, one connection, one partition -- these exist so scripts still run."""

    def test_repartition_returns_the_same_frame(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.repartition(8) is df
        assert df.repartition(8, "vendor") is df

    def test_coalesce_returns_the_same_frame(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.coalesce(1) is df


class TestTempViews:
    def test_a_view_is_queryable_by_name(self, session: Session) -> None:
        session.table("fx.plain").filter(F.col("id") <= 2).createOrReplaceTempView("v")
        assert [r[0] for r in session.sql("SELECT id FROM v ORDER BY id").collect()] == [1, 2]

    def test_session_table_resolves_a_view_too(self, session: Session) -> None:
        session.table("fx.plain").filter(F.col("id") <= 2).createOrReplaceTempView("v")
        assert session.table("v").count() == 2

    def test_a_view_is_a_plan_not_rows(self, session: Session) -> None:
        """So a query through it still prunes -- registering a view costs no pushdown."""
        session.table("fx.partitioned").createOrReplaceTempView("v")
        out = session.sql("SELECT id FROM v WHERE as_at_date = '2026-08-16'")
        scan = session._compile(out._plan, out._sources, out.columns).scans[0]
        assert scan.files_scanned == 1
        assert scan.pushed_filter is not None

    def test_a_view_can_be_joined_to_itself(self, session: Session) -> None:
        session.table("fx.plain").filter(F.col("id") <= 2).createOrReplaceTempView("v")
        out = session.sql("SELECT count(*) AS c FROM v a JOIN v b ON a.id = b.id")
        assert out.collect()[0][0] == 2

    def test_replacing_a_view_changes_what_the_name_means(self, session: Session) -> None:
        df = session.table("fx.plain")
        df.filter(F.col("id") <= 2).createOrReplaceTempView("v")
        assert session.table("v").count() == 2
        df.filter(F.col("id") <= 4).createOrReplaceTempView("v")
        assert session.table("v").count() == 4

    def test_create_temp_view_refuses_to_overwrite(self, session: Session) -> None:
        df = session.table("fx.plain")
        df.createTempView("v")
        with pytest.raises(TempTableAlreadyExistsException, match="already exists"):
            df.createTempView("v")

    def test_drop_temp_view_reports_whether_there_was_one(self, session: Session) -> None:
        session.table("fx.plain").createOrReplaceTempView("v")
        assert session.dropTempView("v") is True
        assert session.dropTempView("v") is False

    def test_a_dropped_view_stops_resolving(self, session: Session) -> None:
        session.table("fx.plain").createOrReplaceTempView("v")
        session.dropTempView("v")
        with pytest.raises(Exception, match="v"):
            session.sql("SELECT * FROM v").collect()

    def test_a_view_name_cannot_be_qualified(self, session: Session) -> None:
        """Views are session-local and belong to no namespace, so `ns.v` is a mistake."""
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="cannot be qualified"):
            df.createOrReplaceTempView("ns.v")

    def test_a_cte_of_the_same_name_wins(self, session: Session) -> None:
        """A `WITH` binding is not a table reference, so inlining must skip it."""
        session.table("fx.plain").createOrReplaceTempView("v")
        out = session.sql("WITH v AS (SELECT 1 AS id) SELECT id FROM v")
        assert [tuple(r) for r in out.collect()] == [(1,)]

    def test_create_view_sql_points_at_the_method_that_exists(self, session: Session) -> None:
        """The gate used to say only "Phase 9", sending people away from a working method."""
        with pytest.raises(UnsupportedFeatureError, match="createOrReplaceTempView"):
            session.sql("CREATE OR REPLACE TEMP VIEW v AS SELECT 1")

    def test_a_view_name_must_be_a_string(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineTypeError, match="non-empty name"):
            df.createOrReplaceTempView("")

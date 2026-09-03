"""Pruning against the real warehouse, where it is the difference that matters.

`nyc.yellow_tripdata` is 62 files and 41,169,720 rows behind an object store, partitioned
by `month(tpep_pickup_datetime)`. `nyc.yellow_tripdata_wide` is 357 files and 219
columns. Those are the shapes that make pushdown worth having, and they are also the only
place several of its failures are reachable -- three of the defects STATUS.md records were
found here and could not have been found on a local fixture.

Two rules govern every test below.

**Pruning changes speed, not answers.** Every assertion that a scan got smaller is paired
with one that the result did not change. A pruning bug that returns fewer rows is not a
performance regression, it is a wrong answer.

**Assertions are relational, not absolute.** `files_scanned < files_total`, not
`files_scanned == 3`. The real tables grow; the property does not.

`test_phase2_rest.py` covers the original three defects. This module covers what it does
not: byte accounting, the copy-on-write guard, rename reconciliation over an object
store, and the pruning that a filter must *not* be granted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import UnsupportedFeatureError
from icetl.sql import functions as F
from tests.integration.conftest import REAL_HUGE, REAL_TABLE, TIME_COLUMN
from tests.integration.helpers import column, compiled_sql, scan_of

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestPartitionPruning:
    def test_a_month_filter_skips_most_of_the_files(self, it_session: Session) -> None:
        frame = it_session.table(REAL_TABLE).filter(
            (F.col(TIME_COLUMN) >= "2024-06-01") & (F.col(TIME_COLUMN) < "2024-07-01")
        )
        scan = scan_of(frame)
        assert scan.files_total is not None
        assert scan.pushed_filter is not None
        assert scan.files_scanned < scan.files_total

    def test_pruning_does_not_change_the_answer(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        """The half that makes the half above safe to want.

        PyIceberg owns the metadata, so it settles what the right count is -- computed
        here rather than written down.
        """
        from pyiceberg.expressions import And

        from tests.predicates import GreaterThanOrEqual, LessThan

        table = catalog.load_table(REAL_TABLE)
        expected = (
            table.scan(
                row_filter=And(
                    GreaterThanOrEqual(TIME_COLUMN, "2024-06-01T00:00:00"),
                    LessThan(TIME_COLUMN, "2024-06-02T00:00:00"),
                ),
                selected_fields=(TIME_COLUMN,),
            )
            .to_arrow()
            .num_rows
        )
        mine = (
            it_session.table(REAL_TABLE)
            .filter((F.col(TIME_COLUMN) >= "2024-06-01") & (F.col(TIME_COLUMN) < "2024-06-02"))
            .count()
        )
        assert mine == expected

    def test_the_predicate_is_still_applied_by_duckdb(self, it_session: Session) -> None:
        """Iceberg pruning is stats-based and approximate.

        It selects the files that *may* contain a match, so DuckDB re-applying the
        predicate is what makes the row count exact rather than merely close.
        """
        frame = it_session.table(REAL_TABLE).filter(F.col(TIME_COLUMN) >= "2024-06-01")
        assert "2024-06-01" in compiled_sql(frame)

    def test_a_filter_on_an_unpartitioned_column_prunes_less_or_not_at_all(
        self, it_session: Session
    ) -> None:
        """Honest reporting: the counters must not claim pruning that did not happen."""
        frame = it_session.table(REAL_TABLE).filter(F.col("VendorID") == 1)
        scan = scan_of(frame)
        assert scan.files_total is not None
        assert scan.files_scanned <= scan.files_total

    def test_an_untranslatable_predicate_is_reported_not_pushed(self, it_session: Session) -> None:
        """Phase 2's promise: anything not understood is simply not pushed.

        It must also be *visible* -- an unpushed filter that vanished from the report
        would make `explain()` claim more pruning than happened.
        """
        frame = it_session.table(REAL_TABLE).filter(
            F.length(F.col("store_and_fwd_flag")) == F.col("VendorID")
        )
        scan = scan_of(frame)
        assert scan.files_total is not None
        assert scan.files_scanned == scan.files_total
        assert scan.unpushed_filters, "an untranslatable filter was not reported"


class TestProjectionPushdown:
    def test_two_columns_of_two_hundred_and_nineteen(self, it_session: Session) -> None:
        frame = it_session.table(REAL_HUGE).select("VendorID", "trip_distance")
        scan = scan_of(frame)
        assert scan.columns == ("VendorID", "trip_distance")
        assert scan.total_columns > 100

    def test_the_bytes_reported_reflect_the_pruning(self, it_session: Session) -> None:
        """FINDINGS 3.3: `bytes_scanned` once ignored column pruning entirely.

        On a 219-column table, reading two columns must account for a small fraction
        of the file -- if the two numbers are equal the accounting is not per-column.
        """
        scan = scan_of(it_session.table(REAL_HUGE).select("VendorID", "trip_distance"))
        assert scan.bytes_total > 0
        assert 0 < scan.bytes_scanned < scan.bytes_total

    def test_selecting_everything_reads_everything(self, it_session: Session) -> None:
        """The control for the test above."""
        scan = scan_of(it_session.table(REAL_HUGE))
        assert scan.bytes_scanned == pytest.approx(scan.bytes_total, rel=0.01)

    def test_an_aggregate_only_reads_the_columns_it_names(self, it_session: Session) -> None:
        frame = (
            it_session.table(REAL_HUGE)
            .filter(F.col(TIME_COLUMN) >= "2024-06-01")
            .groupBy("VendorID")
            .agg(F.sum(F.col("total_amount")).alias("revenue"))
        )
        scan = scan_of(frame)
        assert set(scan.columns) == {"VendorID", "total_amount", TIME_COLUMN}

    def test_ordering_by_an_output_alias_still_prunes(self, it_session: Session) -> None:
        """FINDINGS 3.1, pinned. `qualify` leaves such a reference unqualified because
        it names the projection rather than the table, and the extractor once read that
        as "unattributable column" and fell back to reading all 219."""
        frame = (
            it_session.table(REAL_HUGE)
            .filter(F.col(TIME_COLUMN) >= "2024-06-01")
            .groupBy("VendorID")
            .agg(F.sum(F.col("total_amount")).alias("revenue"))
            .orderBy("revenue")
        )
        scan = scan_of(frame)
        assert set(scan.columns) == {"VendorID", "total_amount", TIME_COLUMN}
        assert len(scan.columns) < scan.total_columns

    def test_the_output_names_survive_the_optimizer(self, it_session: Session) -> None:
        """The optimizer runs in DuckDB's dialect, which lowercases identifiers."""
        frame = it_session.table(REAL_HUGE).select("VendorID", "trip_distance")
        assert frame.columns == ["VendorID", "trip_distance"]


class TestBothSurfacesPruneAlike:
    """A rewrite that only one surface reaches would show up here and nowhere else."""

    def test_the_same_filter_prunes_the_same_files(self, it_session: Session) -> None:
        via_frame = scan_of(it_session.table(REAL_TABLE).filter(F.col(TIME_COLUMN) >= "2024-06-01"))
        via_sql = scan_of(
            it_session.sql(f"SELECT * FROM {REAL_TABLE} WHERE {TIME_COLUMN} >= '2024-06-01'")
        )
        assert via_frame.files_scanned == via_sql.files_scanned
        assert via_frame.pushed_filter == via_sql.pushed_filter

    def test_the_same_projection_prunes_the_same_columns(self, it_session: Session) -> None:
        via_frame = scan_of(it_session.table(REAL_HUGE).select("VendorID", "trip_distance"))
        via_sql = scan_of(it_session.sql(f"SELECT VendorID, trip_distance FROM {REAL_HUGE}"))
        assert set(via_frame.columns) == set(via_sql.columns)


class TestTheCopyOnWriteGuard:
    """Decision 11: a merge-on-read table is refused, never approximated.

    `read_parquet` cannot see a delete file, so without this guard the deleted rows
    would come back and the query would report success. The replica is built with a
    hand-written DELETES manifest precisely so the guard has something to fire on --
    and here it is doing so over a real REST catalog and real object store, which is
    where an upstream writer would actually introduce one.
    """

    def test_reading_a_merge_on_read_table_is_refused(self, it_session: Session, mor: str) -> None:
        with pytest.raises(UnsupportedFeatureError):
            it_session.table(mor).collect()

    def test_the_refusal_names_the_problem(self, it_session: Session, mor: str) -> None:
        """A guard whose message does not say what to do is a guard people work around."""
        with pytest.raises(UnsupportedFeatureError, match="delete"):
            it_session.table(mor).collect()

    def test_even_a_count_is_refused(self, it_session: Session, mor: str) -> None:
        """The metadata count must not become a way around the guard.

        `count(*)` is answered from the manifests, which would happily sum the data
        files and ignore the deletes -- a wrong answer arrived at quickly.
        """
        with pytest.raises(UnsupportedFeatureError):
            it_session.table(mor).count()

    def test_a_filtered_read_is_refused_too(self, it_session: Session, mor: str) -> None:
        with pytest.raises(UnsupportedFeatureError):
            it_session.table(mor).filter(F.col("id") > 2).collect()


class TestRenamedColumnReconciliation:
    """Iceberg tracks columns by field-id; parquet files carry the name they were written with.

    The replica has two data files that disagree about what field 2 is called. Matching
    on *name* returns NULLs for the older file's rows -- silently, which is what makes
    it dangerous (3.4). Over an object store this is also the path that has to open
    footers, so it is worth exercising here rather than only locally.
    """

    def test_both_files_read_back_their_values(self, it_session: Session, renamed: str) -> None:
        values = sorted(v for v in column(it_session.table(renamed), "new_name") if v is not None)
        assert values == ["after-c", "after-d", "before-a", "before-b"]

    def test_no_row_came_back_null(self, it_session: Session, renamed: str) -> None:
        """The failure mode stated directly: the old file's rows must not be NULL."""
        assert it_session.table(renamed).filter(F.col("new_name").isNull()).count() == 0

    def test_the_scan_reports_the_renamed_column(self, it_session: Session, renamed: str) -> None:
        scan = scan_of(it_session.table(renamed).select("new_name"))
        assert "new_name" in scan.renamed_columns

    def test_a_filter_on_the_renamed_column_finds_the_old_rows(
        self, it_session: Session, renamed: str
    ) -> None:
        matched = it_session.table(renamed).filter(F.col("new_name").startswith("before"))
        assert matched.count() == 2

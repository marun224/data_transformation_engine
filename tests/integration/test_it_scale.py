"""The tests that actually read 41 million rows.

Everything else in the integration suite is bounded so the fast tier stays under a few
minutes: it asserts on plan structure, or runs against a seeded slice. This module is the
part that is deliberately not bounded, and it carries the extra `slow` marker so it is
opt-in:

    uv run pytest -m "integration and not slow"   # the fast tier
    uv run pytest -m integration                  # everything, including this

What is here is what only real scale can establish:

  * **Pruning is worth having.** Reading 3 files of 62 is not a plan-shape claim, it is
    a measured difference in work done. Asserted as a ratio of bytes rather than a
    stopwatch reading, because a timing assertion over an object store is a flaky test.
  * **A 219-column table is a different problem from a 19-column one.** Projection
    pushdown that looks fine on two columns of three has to hold on two of 219.
  * **Streaming keeps its promise.** Peak memory is one batch, over a result far larger
    than memory -- which cannot be shown on 500 fixture rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.conftest import REAL_HUGE, REAL_TABLE, TIME_COLUMN
from tests.integration.helpers import scan_of

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestFullScans:
    """Answers over the whole table, not a slice of it."""

    def test_counting_forty_one_million_rows_agrees_with_the_manifests(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        snapshot = catalog.load_table(REAL_TABLE).current_snapshot()
        assert snapshot is not None and snapshot.summary is not None
        assert it_session.table(REAL_TABLE).count() == int(str(snapshot.summary["total-records"]))

    def test_an_aggregate_over_the_whole_table_completes(self, it_session: Session) -> None:
        """The headline query of PLAN.md, at full size."""
        rows = (
            it_session.table(REAL_TABLE)
            .groupBy("VendorID")
            .agg(
                F.count(F.lit(1)).alias("trips"),
                F.sum(F.col("total_amount")).alias("revenue"),
            )
            .orderBy("VendorID")
            .collect()
        )
        assert rows
        assert sum(row["trips"] for row in rows) == it_session.table(REAL_TABLE).count()

    def test_the_grouped_totals_reconcile_with_the_ungrouped_one(self, it_session: Session) -> None:
        """A conservation law over 41M rows -- the strongest correctness signal here."""
        whole = (
            it_session.table(REAL_TABLE)
            .select(F.sum(F.col("total_amount")).alias("t"))
            .collect()[0]["t"]
        )
        parts = (
            it_session.table(REAL_TABLE)
            .groupBy("VendorID")
            .agg(F.sum(F.col("total_amount")).alias("t"))
            .collect()
        )
        assert sum(row["t"] for row in parts) == pytest.approx(whole, rel=1e-9)

    def test_a_window_over_the_whole_table_completes(self, it_session: Session) -> None:
        """Spills to disk, which is the point -- the spill directory is the test's own."""
        from icetl.sql.window import Window

        frame = it_session.table(REAL_TABLE).select(
            F.row_number().over(Window.partitionBy("VendorID").orderBy(TIME_COLUMN)).alias("rn")
        )
        assert frame.count() == it_session.table(REAL_TABLE).count()


class TestPruningIsWorthHaving:
    """Measured as work avoided, not as a plan shape."""

    def test_a_month_filter_reads_a_fraction_of_the_bytes(self, it_session: Session) -> None:
        whole = scan_of(it_session.table(REAL_TABLE))
        pruned = scan_of(
            it_session.table(REAL_TABLE).filter(
                (F.col(TIME_COLUMN) >= "2024-06-01") & (F.col(TIME_COLUMN) < "2024-07-01")
            )
        )
        assert pruned.files_scanned < whole.files_scanned
        assert pruned.bytes_scanned < whole.bytes_scanned / 2, (
            f"pruning saved less than half the bytes: "
            f"{pruned.bytes_scanned} of {whole.bytes_scanned}"
        )

    def test_the_pruned_query_gives_the_same_answer_as_the_unpruned_one(
        self, it_session: Session
    ) -> None:
        """Pruning changes speed, not answers -- at full scale.

        The unpruned form applies the predicate in DuckDB over all 62 files; the pruned
        form skips most of them in the planner. Both must count the same rows.
        """
        pruned = (
            it_session.table(REAL_TABLE)
            .filter((F.col(TIME_COLUMN) >= "2024-06-01") & (F.col(TIME_COLUMN) < "2024-07-01"))
            .count()
        )
        unpruned = (
            it_session.table(REAL_TABLE)
            .select(TIME_COLUMN)
            .filter(
                (F.col(TIME_COLUMN) >= F.lit("2024-06-01").cast("timestamp"))
                & (F.col(TIME_COLUMN) < F.lit("2024-07-01").cast("timestamp"))
            )
            .count()
        )
        assert pruned == unpruned

    def test_projection_pushdown_on_two_of_two_hundred_and_nineteen(
        self, it_session: Session
    ) -> None:
        narrow = scan_of(it_session.table(REAL_HUGE).select("VendorID", "trip_distance"))
        wide = scan_of(it_session.table(REAL_HUGE))
        assert narrow.total_columns > 200
        assert narrow.bytes_scanned < wide.bytes_scanned / 10, (
            f"two of {narrow.total_columns} columns read "
            f"{narrow.bytes_scanned} of {wide.bytes_scanned} bytes"
        )

    def test_the_narrow_read_returns_the_same_rows(self, it_session: Session) -> None:
        narrow = it_session.table(REAL_HUGE).select("VendorID", "trip_distance")
        assert narrow.count() == it_session.table(REAL_HUGE).count()


class TestStreamingAtScale:
    """Peak memory is one batch, over a result that would not fit as one table."""

    def test_a_full_table_streams_in_many_batches(self, it_session: Session) -> None:
        frame = it_session.table(REAL_TABLE).select("VendorID", "trip_distance")
        batches = 0
        rows = 0
        for batch in frame.toArrowBatches(batchSize=200_000):
            batches += 1
            rows += batch.num_rows
        assert batches > 10, batches
        assert rows == frame.count()

    def test_no_batch_exceeds_the_requested_size(self, it_session: Session) -> None:
        frame = it_session.table(REAL_TABLE).select("VendorID")
        for batch in frame.toArrowBatches(batchSize=200_000):
            assert batch.num_rows <= 200_000

    def test_a_streamed_aggregate_matches_the_engines_own(self, it_session: Session) -> None:
        """41M values summed in Python, against 41M summed in DuckDB."""
        frame = it_session.table(REAL_TABLE).select("trip_distance")
        streamed = 0.0
        for batch in frame.toArrowBatches(batchSize=500_000):
            streamed += sum(v for v in batch.column("trip_distance").to_pylist() if v is not None)
        engine = (
            it_session.table(REAL_TABLE)
            .select(F.sum(F.col("trip_distance")).alias("t"))
            .collect()[0]["t"]
        )
        # Different summation orders over 41M doubles, so the last digits differ
        # (FINDINGS 3.7) -- the tolerance is the finding, not a fudge.
        assert streamed == pytest.approx(engine, rel=1e-6)


class TestTheWideTable:
    """219 columns and 357 files, which is the shape PLAN.md 3.6 was written for."""

    def test_collect_on_a_wide_result_is_avoidable(self, it_session: Session) -> None:
        """FINDINGS 3.8: `collect()` builds a `Row` per row and dominates the wall time.

        Asserted as an equivalence rather than a timing: `toArrow()` must return the
        same rows, so reaching for it is never a correctness trade.
        """
        frame = it_session.table(REAL_HUGE).limit(5_000)
        arrow = frame.toArrow()
        rows = frame.collect()
        assert arrow.num_rows == len(rows)
        assert arrow.num_columns == len(frame.columns)

    def test_a_filtered_wide_read_prunes_both_ways(self, it_session: Session) -> None:
        """Rows and columns at once, on the table where both matter."""
        scan = scan_of(
            it_session.table(REAL_HUGE)
            .filter(F.col(TIME_COLUMN) >= "2024-06-01")
            .select("VendorID", "trip_distance", TIME_COLUMN)
        )
        assert scan.files_total is not None
        assert scan.files_scanned < scan.files_total
        assert len(scan.columns) < scan.total_columns
        assert scan.bytes_scanned < scan.bytes_total

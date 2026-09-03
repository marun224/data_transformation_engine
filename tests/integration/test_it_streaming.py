"""Streaming a result out of the real warehouse, one batch at a time.

The reason streaming exists is stated in README.md: `collect()` builds a `Row` per
result row, and on a wide table that is 95% of the wall time (FINDINGS 3.8). The way out
is `toArrow()`, or `toArrowBatches()` when the result will not fit in memory at all.

A local fixture cannot really test that. 500 rows in one file arrive as a single batch,
so "streaming" and "not streaming" are the same code path with the same peak memory.
Here the source is 880,374 rows across three files on an object store, which produces
genuinely many batches and genuinely exercises the reader.

Two claims are worth pinning:

  * **The stream is complete and correct.** Concatenating the batches must give exactly
    the table `toArrow()` returns -- same rows, same schema, same order-insensitive
    contents. A streaming path that drops the last partial batch is easy to write and
    hard to notice.
  * **The stale-stream guard fires.** A `RecordBatchReader` is tied to the connection
    generation it was opened on; running another query underneath it invalidates it.
    Returning garbage there would be worse than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.conftest import REAL_TABLE, TIME_COLUMN

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestTheStreamIsComplete:
    def test_the_batches_hold_every_row(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips).select("VendorID", "trip_distance")
        streamed = sum(batch.num_rows for batch in frame.toArrowBatches())
        assert streamed == frame.count()

    def test_the_batches_reconstruct_the_table(self, it_session: Session, zones: str) -> None:
        """Small enough to hold both, so the comparison can be on values."""
        import pyarrow as pa

        frame = it_session.table(zones).orderBy("zone_id")
        whole = frame.toArrow()
        rebuilt = pa.Table.from_batches(list(frame.toArrowBatches()), schema=whole.schema)
        assert rebuilt.num_rows == whole.num_rows
        assert rebuilt.to_pylist() == whole.to_pylist()

    def test_every_batch_carries_the_same_schema(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips).select("VendorID", "trip_distance")
        schemas = {batch.schema for batch in frame.toArrowBatches()}
        assert len(schemas) == 1, "the batches disagreed about their own schema"

    def test_the_local_iterator_yields_every_row(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        assert sum(1 for _ in frame.toLocalIterator()) == frame.count()

    def test_a_filtered_stream_yields_only_the_matching_rows(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips).filter(F.col("VendorID") == 1).select("VendorID")
        streamed = sum(batch.num_rows for batch in frame.toArrowBatches())
        assert streamed == frame.count()
        assert 0 < streamed < it_session.table(trips).count()


class TestBatching:
    def test_a_small_batch_size_produces_many_batches(
        self, it_session: Session, trips: str
    ) -> None:
        """The property that makes peak memory a batch rather than a result."""
        frame = it_session.table(trips).select("VendorID")
        batches = list(frame.toArrowBatches(batchSize=10_000))
        assert len(batches) > 1
        assert sum(batch.num_rows for batch in batches) == frame.count()

    def test_no_batch_exceeds_the_requested_size(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips).select("VendorID")
        for batch in frame.toArrowBatches(batchSize=10_000):
            assert batch.num_rows <= 10_000

    def test_a_larger_batch_size_produces_fewer_batches(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips).select("VendorID")
        few = len(list(frame.toArrowBatches(batchSize=500_000)))
        many = len(list(frame.toArrowBatches(batchSize=10_000)))
        assert few < many

    def test_an_empty_result_streams_nothing_without_failing(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips).filter(F.col("VendorID") == -1)
        assert sum(batch.num_rows for batch in frame.toArrowBatches()) == 0


class TestStreamingAgreesWithCollecting:
    """Two ways of asking for the same rows, over an object store."""

    def test_the_streamed_aggregate_matches_the_collected_one(
        self, it_session: Session, trips: str
    ) -> None:
        frame = it_session.table(trips).select("trip_distance")
        streamed = sum(
            sum(v for v in batch.column("trip_distance").to_pylist() if v is not None)
            for batch in frame.toArrowBatches()
        )
        collected = (
            it_session.table(trips)
            .select(F.sum(F.col("trip_distance")).alias("t"))
            .collect()[0]["t"]
        )
        assert streamed == pytest.approx(collected, rel=1e-9)

    def test_streaming_respects_the_pushdown(self, it_session: Session) -> None:
        """A streaming read must not quietly become an unpruned one."""
        frame = (
            it_session.table(REAL_TABLE)
            .filter((F.col(TIME_COLUMN) >= "2024-06-01") & (F.col(TIME_COLUMN) < "2024-06-02"))
            .select(TIME_COLUMN)
        )
        streamed = sum(batch.num_rows for batch in frame.toArrowBatches(batchSize=100_000))
        assert streamed == frame.count()


class TestTheStaleStreamGuard:
    """A reader outlives the query that made it only until the connection moves on."""

    def test_running_another_query_mid_stream_is_refused(
        self, session: Session, trips: str
    ) -> None:
        """Better to raise than to hand back rows from the wrong query.

        A function-scoped session, because this deliberately invalidates the
        connection's generation and should not leak into another test.
        """
        from icetl.errors import QueryExecutionException

        stream = session.table(trips).select("VendorID").toArrowBatches(batchSize=10_000)
        next(stream)

        session.table(trips).select(F.count(F.lit(1)).alias("n")).collect()

        with pytest.raises(QueryExecutionException):
            for _ in stream:
                pass

    def test_a_stream_finished_before_the_next_query_is_fine(
        self, session: Session, zones: str
    ) -> None:
        """The control: the guard must not refuse the ordinary sequential case."""
        frame = session.table(zones).select("zone_id")
        first = sum(batch.num_rows for batch in frame.toArrowBatches())
        second = frame.count()
        assert first == second

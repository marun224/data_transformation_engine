"""Phase 10: streaming a result, so a large one never has to fit in memory.

`toArrowBatches()` yields the result as DuckDB produces it and `toLocalIterator()`
turns those into `Row`s a batch at a time. Both compile through exactly the same path
as `collect()` -- pruning, pushdown and conformance are not things a streaming caller
opts out of -- so the interesting tests are not "does it return the rows" but the two
places streaming differs from collecting:

**The staleness guard.** DuckDB ends a result when its cursor runs the next query,
and it does so *silently*: a half-read stream simply stops, reporting the prefix it
had as the whole answer. Since iterating lazily is the entire point, running another
query mid-iteration is a thing callers do. `TestAStaleStreamRefuses` is the case that
matters.

**The batching itself**, which has to be observable or the memory claim is unfounded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from icetl.errors import EngineValueError, QueryExecutionException
from icetl.exec.engine import DEFAULT_BATCH_ROWS
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.session import Session


class TestItReturnsTheSameRows:
    """Streaming is an execution strategy, not a different query."""

    @pytest.mark.parametrize("reference", ["fx.plain", "fx.partitioned", "fx.wide", "fx.nested"])
    def test_the_streamed_rows_equal_the_collected_ones(
        self, session: Session, reference: str
    ) -> None:
        frame = session.table(reference)
        assert list(frame.toLocalIterator()) == frame.collect()

    def test_a_filter_still_prunes_and_still_filters(self, session: Session) -> None:
        frame = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        rows = list(frame.toLocalIterator())
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        assert compiled.scans[0].files_scanned == 1
        assert len(rows) == 4

    def test_nested_types_convert_as_collect_does(self, session: Session) -> None:
        """`to_rows` is the same call, so struct/map/list reshaping cannot drift."""
        frame = session.table("fx.nested")
        assert list(frame.toLocalIterator()) == frame.collect()

    def test_an_empty_result_streams_nothing(self, session: Session) -> None:
        frame = session.table("fx.plain").filter(F.col("id") < 0)
        assert list(frame.toLocalIterator()) == []

    def test_a_count_streams_its_one_row(self, session: Session) -> None:
        """The metadata fast path has no batches to stream, and still must yield one."""
        frame = session.sql("SELECT count(*) FROM fx.partitioned")
        assert [row[0] for row in frame.toLocalIterator()] == [12]

    def test_a_cached_frame_streams(self, session: Session) -> None:
        """A temp table is per-cursor, which is why the stream shares the cursor."""
        cached = session.table("fx.plain").cache()
        assert list(cached.toLocalIterator()) == cached.collect()

    def test_a_local_frame_streams(self, session: Session) -> None:
        """A registered Arrow table is per-cursor too."""
        frame = session.createDataFrame([(1, "a"), (2, "b")], ["id", "name"])
        assert [row[0] for row in frame.toLocalIterator()] == [1, 2]


class TestBatching:
    def test_a_small_batch_size_produces_several_batches(self, session: Session) -> None:
        """The memory claim rests on this: the result arrives in pieces."""
        batches = list(session.table("fx.wide").toArrowBatches(batchSize=100))
        assert len(batches) > 1
        assert sum(batch.num_rows for batch in batches) == 500

    def test_the_default_batch_size_gives_one_batch_here(self, session: Session) -> None:
        """500 rows is far under the default, so nothing is split needlessly."""
        batches = list(session.table("fx.wide").toArrowBatches())
        assert len(batches) == 1
        assert DEFAULT_BATCH_ROWS > 500

    def test_the_batches_carry_the_frame_s_schema(self, session: Session) -> None:
        frame = session.table("fx.plain").select("id")
        batch = next(iter(frame.toArrowBatches()))
        assert batch.schema.names == ["id"]

    def test_the_batches_reassemble_into_the_collected_table(self, session: Session) -> None:
        frame = session.table("fx.wide").select("id", "col_001")
        batches = list(frame.toArrowBatches(batchSize=64))
        assert pa.Table.from_batches(batches).equals(frame.toArrow())

    @pytest.mark.parametrize("size", [0, -1, "many", 1.5, True])
    def test_a_nonsense_batch_size_is_refused(self, session: Session, size: object) -> None:
        with pytest.raises(EngineValueError):
            list(session.table("fx.plain").toArrowBatches(batchSize=size))  # type: ignore[arg-type]


class TestAStaleStreamRefuses:
    """The guard. Without it, this returns a prefix and reports success.

    DuckDB's own behaviour here is a silent truncation -- a reader whose cursor has
    run another query yields no further batches and raises nothing -- which is
    precisely the class of failure this codebase treats as worst. So the stream
    refuses rather than ending early.
    """

    def test_a_query_mid_iteration_invalidates_the_stream(self, session: Session) -> None:
        stream = session.table("fx.wide").toArrowBatches(batchSize=50)
        next(iter(stream))  # start consuming, so a prefix has been handed out
        session.table("fx.plain").collect()  # the interfering query
        with pytest.raises(QueryExecutionException, match="invalidated by another query"):
            list(stream)

    def test_the_refusal_says_what_to_do_instead(self, session: Session) -> None:
        stream = session.table("fx.wide").toLocalIterator(batchSize=50)
        next(stream)
        session.sql("SELECT 1").collect()
        with pytest.raises(QueryExecutionException) as caught:
            list(stream)
        assert "toArrow()" in str(caught.value)

    def test_a_second_stream_invalidates_the_first(self, session: Session) -> None:
        first = session.table("fx.wide").toArrowBatches(batchSize=50)
        next(iter(first))
        second = session.table("fx.wide").toArrowBatches(batchSize=50)
        assert next(iter(second)).num_rows == 50
        with pytest.raises(QueryExecutionException):
            list(first)

    def test_finishing_first_is_fine(self, session: Session) -> None:
        """The ordinary use. Consume the stream, then carry on."""
        rows = list(session.table("fx.wide").toLocalIterator(batchSize=50))
        assert len(rows) == 500
        assert session.table("fx.plain").count() == 5

    def test_two_streams_consumed_in_turn_are_fine(self, session: Session) -> None:
        first = list(session.table("fx.plain").toLocalIterator())
        second = list(session.table("fx.plain").toLocalIterator())
        assert first == second


class TestWhatDoesNotInvalidateAStream:
    def test_a_metadata_count_leaves_a_stream_alone(self, session: Session) -> None:
        """It answers from manifests, so DuckDB's cursor never moves.

        Worth pinning: the guard keys on queries actually run, not on calls made, and
        the fast-path count is the one action that looks like a query and is not one.
        """
        stream = session.table("fx.wide").toArrowBatches(batchSize=50)
        first = next(iter(stream))
        assert session.table("fx.partitioned").count() == 12
        assert first.num_rows + sum(batch.num_rows for batch in stream) == 500

    def test_planning_alone_leaves_a_stream_alone(self, session: Session) -> None:
        """`explain()` compiles but does not execute."""
        stream = session.table("fx.wide").toArrowBatches(batchSize=50)
        first = next(iter(stream))
        session.table("fx.plain")._explain_text(verbose=False)
        assert first.num_rows + sum(batch.num_rows for batch in stream) == 500

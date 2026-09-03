"""Phase 10: `count(*)` answered from Iceberg's manifests instead of parquet footers.

Iceberg records a `record_count` on every data file, so an unfiltered count is a sum
over metadata `plan_files()` has already fetched. Handing the question to DuckDB
instead opens a footer per file -- roughly 2x on a 357-file table, and worse as the
file count grows (FINDINGS.md 3.4).

**`TestNoFileIsOpened` is the test that matters.** Timing would only show that the
fast path is quicker; deleting the parquet files and asking again shows that it reads
nothing at all, which is the actual claim. Everything else here is the boundary of
what may take that path, and every one of those cases asserts on the *value*, because
a count that is fast and wrong is the whole risk.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from icetl.errors import QueryExecutionException
from icetl.paths import engine_path
from icetl.plan.counting import countable_scan
from icetl.sql import functions as F

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def takes_fast_path(session: Session, frame: DataFrame) -> bool:
    """Whether `frame.count()` would be answered from metadata.

    Built the way `DataFrame.count()` builds it, so this asks about the plan that is
    actually executed rather than about a plan resembling it.
    """
    from sqlglot import exp

    counting = exp.select(exp.Count(this=exp.Star())).from_(
        exp.Subquery(
            this=frame._plan.copy(),
            alias=exp.TableAlias(this=exp.to_identifier("q")),
        )
    )
    return countable_scan(counting, frame._sources) is not None


class TestTheCountIsRight:
    """Correctness before speed: every count below is checked against the rows."""

    @pytest.mark.parametrize("reference", ["fx.plain", "fx.partitioned", "fx.wide", "fx.nested"])
    def test_it_agrees_with_the_rows_it_counts(self, session: Session, reference: str) -> None:
        frame = session.table(reference)
        assert frame.count() == len(frame.collect())

    def test_it_agrees_with_pyiceberg(self, session: Session, catalog: SqlCatalog) -> None:
        """The other engine's answer for the same table, read its own way."""
        table = catalog.load_table("fx.partitioned")
        assert session.table("fx.partitioned").count() == table.scan().to_arrow().num_rows

    def test_both_surfaces_give_the_same_number(self, session: Session) -> None:
        """P1: the DataFrame and the SQL spelling are one code path, so one answer."""
        api = session.table("fx.partitioned").count()
        sql = session.sql("SELECT count(*) FROM fx.partitioned").collect()[0][0]
        assert api == sql == 12

    def test_the_sql_surface_keeps_the_column_name_it_would_have_had(
        self, session: Session
    ) -> None:
        """The fast path builds Arrow directly, so the schema is ours to get right."""
        frame = session.sql("SELECT count(*) FROM fx.partitioned")
        assert frame.collect()[0].asDict() == {frame.columns[0]: 12}

    def test_an_aliased_count_keeps_its_alias(self, session: Session) -> None:
        frame = session.sql("SELECT count(*) AS n FROM fx.partitioned")
        assert frame.columns == ["n"]
        assert frame.collect()[0]["n"] == 12

    def test_the_count_is_a_bigint_as_duckdb_s_own_would_be(self, session: Session) -> None:
        assert session.sql("SELECT count(*) FROM fx.plain").toArrow().schema.field(0).type == (
            pa.int64()
        )


class TestNoFileIsOpened:
    """Delete the data, keep the metadata, and the count still answers.

    A timing test would show the fast path is quicker; this shows it reads nothing,
    which is the actual claim being made. The same query one line later, asking for
    the rows, fails -- so the table really is unreadable and the count really did
    come from the manifest.
    """

    @pytest.fixture
    def gutted(self, session: Session, catalog: SqlCatalog) -> Iterator[str]:
        """A table whose parquet files are deleted after it is written."""
        name = f"wr.count_{uuid.uuid4().hex[:8]}"
        session.createDataFrame(
            [(index, f"row-{index}") for index in range(7)], ["id", "label"]
        ).write.saveAsTable(name)

        table = catalog.load_table(tuple(name.split(".")))
        for task in table.scan().plan_files():
            Path(engine_path(task.file.file_path)).unlink()
        yield name
        with contextlib.suppress(Exception):
            catalog.drop_table(tuple(name.split(".")))

    def test_the_count_survives_the_files_being_deleted(
        self, session: Session, gutted: str
    ) -> None:
        assert session.table(gutted).count() == 7

    def test_reading_the_same_table_does_not(self, session: Session, gutted: str) -> None:
        """The control. Without this the test above could be passing by accident."""
        with pytest.raises(QueryExecutionException):
            session.table(gutted).collect()

    def test_the_sql_spelling_survives_too(self, session: Session, gutted: str) -> None:
        assert session.sql(f"SELECT count(*) FROM {gutted}").collect()[0][0] == 7


class TestWhatMayNotTakeTheFastPath:
    """The boundary. Each of these is a count metadata cannot answer exactly."""

    def test_a_filter_disqualifies_it(self, session: Session) -> None:
        """File pruning is an over-approximation, so the sum would be too big.

        `fx.partitioned` holds 12 rows in 3 files; the filter matches 6, and the file
        holding them holds more than 6. Answering from metadata would return the
        file's row count and be confidently wrong.
        """
        frame = session.table("fx.partitioned").filter(F.col("id") > 5)
        assert not takes_fast_path(session, frame)
        assert frame.count() == 6

    def test_a_filter_that_prunes_no_file_is_still_disqualified(self, session: Session) -> None:
        """The predicate need not prune anything to make the sum wrong."""
        frame = session.table("fx.plain").filter(F.col("id") == 3)
        assert not takes_fast_path(session, frame)
        assert frame.count() == 1

    def test_a_limit_disqualifies_it(self, session: Session) -> None:
        frame = session.table("fx.partitioned").limit(3)
        assert not takes_fast_path(session, frame)
        assert frame.count() == 3

    def test_distinct_disqualifies_it(self, session: Session) -> None:
        frame = session.table("fx.partitioned").select("as_at_date").distinct()
        assert not takes_fast_path(session, frame)
        assert frame.count() == 3

    def test_a_join_disqualifies_it(self, session: Session) -> None:
        frame = session.table("fx.plain").join(
            session.table("fx.partitioned").select("id"), on="id"
        )
        assert not takes_fast_path(session, frame)
        assert frame.count() == 5

    def test_a_generator_disqualifies_it(self, session: Session) -> None:
        """The case a rule reading only the FROM clause would get wrong.

        `explode` emits one row per element from what looks like an ordinary
        projection, so the table's row count is not the query's -- three elements per
        row over `fx.plain`'s five rows is fifteen.
        """
        frame = session.table("fx.plain").select(
            F.explode(F.array(F.lit(1), F.lit(2), F.lit(3))).alias("n")
        )
        assert not takes_fast_path(session, frame)
        assert frame.count() == len(frame.collect()) == 15

    def test_an_aggregate_disqualifies_it(self, session: Session) -> None:
        frame = session.table("fx.partitioned").groupBy("as_at_date").count()
        assert not takes_fast_path(session, frame)
        assert frame.count() == 3

    def test_count_of_a_column_is_not_count_star(self, session: Session) -> None:
        """`count(col)` skips NULLs, so a row count is the wrong number for it."""
        rows = session.sql("SELECT count(id) FROM fx.plain").collect()
        assert rows[0][0] == 5

    def test_count_distinct_is_not_count_star(self, session: Session) -> None:
        assert (
            session.sql("SELECT count(DISTINCT as_at_date) FROM fx.partitioned").collect()[0][0]
            == 3
        )

    def test_a_union_disqualifies_it(self, session: Session) -> None:
        frame = session.table("fx.plain").select("id").union(session.table("fx.plain").select("id"))
        assert not takes_fast_path(session, frame)
        assert frame.count() == 10

    def test_a_temp_view_of_a_frame_is_still_recognised(self, session: Session) -> None:
        """A view is a plan, so an unfiltered one is still a plain scan."""
        session.table("fx.partitioned").createOrReplaceTempView("v_all")
        assert session.sql("SELECT count(*) FROM v_all").collect()[0][0] == 12

    def test_a_filtered_temp_view_is_not(self, session: Session) -> None:
        session.table("fx.partitioned").filter(F.col("id") > 5).createOrReplaceTempView("v_some")
        assert session.sql("SELECT count(*) FROM v_some").collect()[0][0] == 6


class TestItStillWorksWhereMetadataCannotHelp:
    def test_a_cached_frame_counts_from_duckdb(self, session: Session) -> None:
        """A cached frame is a DuckDB temp table, not an Iceberg source."""
        cached = session.table("fx.plain").cache()
        assert not takes_fast_path(session, cached)
        assert cached.count() == 5

    def test_a_local_frame_counts(self, session: Session) -> None:
        frame = session.createDataFrame([(1,), (2,), (3,)], ["id"])
        assert frame.count() == 3

    def test_an_empty_table_counts_zero(self, session: Session, catalog: SqlCatalog) -> None:
        """No data files at all is an ordinary state, and its count is 0, not an error."""
        name = f"wr.empty_{uuid.uuid4().hex[:8]}"
        session.sql(f"CREATE TABLE {name} (id BIGINT)")
        try:
            assert session.table(name).count() == 0
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_a_time_travelled_count_reads_that_snapshot(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        """The source key carries the snapshot, so the fast path counts the right one."""
        name = f"wr.tt_{uuid.uuid4().hex[:8]}"
        session.createDataFrame([(1,), (2,)], ["id"]).write.saveAsTable(name)
        first = catalog.load_table(tuple(name.split("."))).current_snapshot()
        assert first is not None
        session.createDataFrame([(3,)], ["id"]).write.mode("append").saveAsTable(name)
        try:
            assert session.table(name).count() == 3
            assert (
                session.sql(
                    f"SELECT count(*) FROM {name} VERSION AS OF {first.snapshot_id}"
                ).collect()[0][0]
                == 2
            )
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_a_renamed_column_table_still_counts(self, session: Session) -> None:
        """Field-id reconciliation is a read concern; the row count is not affected."""
        assert session.table("fx.renamed").count() == len(session.table("fx.renamed").collect())

"""`count(*)` against the real catalog, asked two ways every time.

Two separate claims, and the suite has to hold both:

**Speed.** An unfiltered `count(*)` is a sum over Iceberg's manifests, not a parquet
footer per file (FINDINGS 3.4). On a 62-file table over an object store the difference
is 62 range requests against none, so this is the one optimisation whose absence is
visible to a person rather than a benchmark.

**Correctness.** `count()` returned 5 where `collect()` returned 15 for two whole
phases, and it survived that long because every test asked one question. So every count
here is asked twice -- once through the fast path, once by counting the rows that come
back -- and the test is the comparison, not either number.

The real tables are where the fast path earns its keep, and where it is most dangerous:
`nyc.yellow_tripdata` reports 41,169,720 rows in its manifests, and if that number were
ever wrong nothing downstream would notice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.plan.counting import countable_scan
from icetl.sql import functions as F
from tests.integration.conftest import REAL_TABLE, TIME_COLUMN
from tests.integration.helpers import pyiceberg_count, scan_of

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


def takes_fast_path(frame: DataFrame) -> bool:
    """Whether `frame.count()` would be answered from metadata.

    Built the way `DataFrame.count()` builds it, so this asks about the plan that is
    actually executed rather than one resembling it. Kept identical to the local
    suite's helper on purpose -- the two files should be diffable.
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
    """Correctness before speed. Every count is checked against the rows it counts."""

    def test_it_agrees_with_the_rows_on_every_replica(
        self, it_session: Session, plain: str, partitioned: str, wide: str, nested: str
    ) -> None:
        """The two-questions rule, over infrastructure the local suite cannot reach."""
        for reference in (plain, partitioned, wide, nested):
            frame = it_session.table(reference)
            assert frame.count() == len(frame.collect()), reference

    def test_it_agrees_with_the_rows_on_real_data(self, it_session: Session, zones: str) -> None:
        """Small enough to collect, real enough to matter."""
        frame = it_session.table(zones)
        assert frame.count() == len(frame.collect())

    def test_a_filtered_count_agrees_with_the_rows(self, it_session: Session, plain: str) -> None:
        # 20.5, 30.25 and 50.0 clear the bar; 10.0 does not, and the NULL is not
        # false -- it simply fails the predicate, which is the reference behaviour.
        frame = it_session.table(plain).filter(F.col("amount") > 20)
        assert frame.count() == len(frame.collect()) == 3

    def test_the_two_surfaces_count_the_same(self, it_session: Session, trips: str) -> None:
        """P1 on the operation that broke it."""
        from_sql = it_session.sql(f"SELECT count(*) AS n FROM {trips}").collect()[0]["n"]
        assert from_sql == it_session.table(trips).count()


class TestTheFastPathIsTaken:
    """Which plans reach the manifests, and which correctly do not."""

    def test_a_bare_count_of_a_real_table_is_answered_from_metadata(
        self, it_session: Session
    ) -> None:
        assert takes_fast_path(it_session.table(REAL_TABLE))

    def test_a_filtered_count_is_not(self, it_session: Session) -> None:
        """File pruning over-approximates, so summing record counts under a filter
        would give a number that is too big and look authoritative doing it."""
        frame = it_session.table(REAL_TABLE).filter(F.col(TIME_COLUMN) >= "2024-06-01")
        assert not takes_fast_path(frame)

    def test_a_limited_count_is_not(self, it_session: Session, trips: str) -> None:
        assert not takes_fast_path(it_session.table(trips).limit(10))

    def test_a_distinct_count_is_not(self, it_session: Session, trips: str) -> None:
        assert not takes_fast_path(it_session.table(trips).select("VendorID").distinct())

    def test_a_grouped_count_is_not(self, it_session: Session, trips: str) -> None:
        assert not takes_fast_path(it_session.table(trips).groupBy("VendorID").count())


class TestAgainstPyIceberg:
    """PyIceberg owns the metadata, so it is the authority on how many rows there are."""

    def test_the_real_tables_count_matches_its_manifests(
        self, it_session: Session, catalog: Catalog
    ) -> None:
        """No literal here: the expected number is read from the catalog in the test.

        41M rows, and neither side opens a data file to answer.
        """
        table = catalog.load_table(REAL_TABLE)
        snapshot = table.current_snapshot()
        assert snapshot is not None and snapshot.summary is not None
        recorded = snapshot.summary.get("total-records")
        assert recorded is not None, "the snapshot summary carries no row count"
        expected = int(recorded)
        assert it_session.table(REAL_TABLE).count() == expected

    def test_a_seeded_slice_matches_a_pyiceberg_scan(
        self, it_session: Session, catalog: Catalog, zones: str
    ) -> None:
        table = catalog.load_table(zones)
        assert it_session.table(zones).count() == pyiceberg_count(table)


class TestNoFileIsOpened:
    """The claim that makes the fast path worth having."""

    def test_the_count_plan_selects_no_files_at_all(self, it_session: Session) -> None:
        """A metadata count reads the manifests and stops.

        Asserted on the plan rather than by timing it: a timing assertion over a
        network object store is a flaky test waiting to happen.
        """
        frame = it_session.table(REAL_TABLE)
        scan = scan_of(frame)
        assert scan.metadata_row_count is not None
        assert scan.metadata_row_count > 0

    def test_a_pruned_scan_refuses_to_answer_from_metadata(self, it_session: Session) -> None:
        """The guard on the optimisation. Pruning is approximate; counting is not."""
        frame = it_session.table(REAL_TABLE).filter(
            (F.col(TIME_COLUMN) >= "2024-06-01") & (F.col(TIME_COLUMN) < "2024-07-01")
        )
        scan = scan_of(frame)
        assert scan.pushed_filter is not None
        assert scan.metadata_row_count is None, (
            "a filtered scan offered a metadata row count, which would be an "
            "over-approximation reported as an exact answer"
        )

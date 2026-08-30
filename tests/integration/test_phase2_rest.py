"""Phase 2 against the real REST catalog and MinIO.

The local fixtures prove the mechanisms; these prove they survive contact with a real
table -- 41M rows, 219 columns, partitioned by a `month()` transform, with mixed-case
column names. Three of the defects Phase 2 fixed were only reachable here:

  * a bare date compared to a `timestamp` column made PyIceberg's literal binding
    *raise* from inside `plan_files()`, killing the query;
  * `ORDER BY <output alias>` disabled projection pushdown entirely, so the headline
    aggregate read 219 of 219 columns;
  * DuckDB re-typed the partition column from the `pickup_month=...` directory name.

Deselected by default. Run with a `.env` pointing at the catalog:

    uv run pytest -m integration
"""

from __future__ import annotations

import datetime
import os
from typing import TYPE_CHECKING

import pytest
from pyiceberg.expressions import And

from icetl.catalog import CatalogRegistry, TableResolver
from icetl.conf import IcetlSettings, resolve_settings
from icetl.sql import SparkSession
from tests.predicates import GreaterThanOrEqual, LessThan

if TYPE_CHECKING:
    from icetl.exec.scan_planner import ScanPlan
    from icetl.sql.dataframe import DataFrame

pytestmark = pytest.mark.integration

NAMESPACE = os.environ.get("ICETL_TEST_NAMESPACE", "nyc")
TABLE = os.environ.get("ICETL_TEST_TABLE", "yellow_tripdata")
WIDE_TABLE = os.environ.get("ICETL_TEST_WIDE_TABLE", "wide_smoke")

# A month wholly inside the data, expressed the way anyone would actually write it --
# a bare date, which is exactly the form PyIceberg's binder rejects.
MONTH_START = os.environ.get("ICETL_TEST_MONTH_START", "2024-06-01")
MONTH_END = os.environ.get("ICETL_TEST_MONTH_END", "2024-07-01")
TIME_COLUMN = os.environ.get("ICETL_TEST_TIME_COLUMN", "tpep_pickup_datetime")


@pytest.fixture(scope="module")
def settings() -> IcetlSettings:
    return resolve_settings()


@pytest.fixture(scope="module")
def spark(settings: IcetlSettings) -> SparkSession:
    return SparkSession(settings=settings)


def scan_of(df: DataFrame) -> ScanPlan:
    scans = df._session._compile(df._plan, df._sources, df.columns).scans
    assert len(scans) == 1
    return scans[0]


class TestProjectionPushdown:
    def test_a_two_column_query_does_not_read_all_of_them(self, spark: SparkSession) -> None:
        df = spark.table(f"{NAMESPACE}.{WIDE_TABLE}").select("VendorID", "trip_distance")
        scan = scan_of(df)
        assert scan.columns == ("VendorID", "trip_distance")
        assert scan.total_columns > 100, "expected the wide fixture table"

    def test_mixed_case_column_names_survive_the_optimizer(self, spark: SparkSession) -> None:
        """The optimizer runs in DuckDB's dialect, which lowercases identifiers.
        Output names are restored from the analysed schema, so `VendorID` keeps its
        spelling -- scripts index on it."""
        df = spark.table(f"{NAMESPACE}.{WIDE_TABLE}").select("VendorID", "trip_distance")
        assert df.columns == ["VendorID", "trip_distance"]

    def test_an_aggregate_with_order_by_still_prunes_columns(self, spark: SparkSession) -> None:
        df = spark.sql(
            f"SELECT VendorID, sum(total_amount) AS revenue "
            f"FROM {NAMESPACE}.{TABLE} "
            f"WHERE {TIME_COLUMN} >= '{MONTH_START}' "
            f"GROUP BY VendorID ORDER BY VendorID"
        )
        scan = scan_of(df)
        assert set(scan.columns) == {"VendorID", "total_amount", TIME_COLUMN}
        assert df.columns == ["VendorID", "revenue"]


class TestPartitionPruning:
    def test_a_bare_date_against_a_timestamp_column_prunes(self, spark: SparkSession) -> None:
        """The filter anyone would actually write. PyIceberg rejects a bare date, so
        it is widened to full ISO-8601 before being pushed."""
        df = spark.sql(
            f"SELECT count(*) AS n FROM {NAMESPACE}.{TABLE} "
            f"WHERE {TIME_COLUMN} >= '{MONTH_START}' AND {TIME_COLUMN} < '{MONTH_END}'"
        )
        scan = scan_of(df)
        assert scan.pushed_filter is not None
        assert scan.files_total is not None
        assert scan.files_scanned < scan.files_total, "nothing was pruned"

    def test_the_filter_is_still_in_the_generated_sql(self, spark: SparkSession) -> None:
        """Iceberg pruning is stats-based and approximate; DuckDB re-applying the
        predicate is what makes the row count exact."""
        df = spark.sql(
            f"SELECT count(*) AS n FROM {NAMESPACE}.{TABLE} WHERE {TIME_COLUMN} >= '{MONTH_START}'"
        )
        sql = df._session._compile(df._plan, df._sources, df.columns).sql
        assert MONTH_START in sql


class TestPrunedResultsMatchPyIceberg:
    def test_a_pruned_count_equals_pyicebergs_own(
        self, spark: SparkSession, settings: IcetlSettings
    ) -> None:
        """The differential test that matters: pruning must change speed, not answers.

        PyIceberg is the authority -- it is the thing that owns the metadata -- so the
        two engines counting the same rows is the strongest correctness signal
        available without a second Spark.
        """
        resolver = TableResolver(
            CatalogRegistry(settings), default_namespace=tuple(NAMESPACE.split("."))
        )
        table = resolver.resolve(f"{NAMESPACE}.{TABLE}").table
        expected = (
            table.scan(
                row_filter=And(
                    GreaterThanOrEqual(TIME_COLUMN, f"{MONTH_START}T00:00:00"),
                    LessThan(TIME_COLUMN, f"{MONTH_END}T00:00:00"),
                ),
                selected_fields=(TIME_COLUMN,),
            )
            .to_arrow()
            .num_rows
        )

        df = spark.sql(
            f"SELECT count(*) AS n FROM {NAMESPACE}.{TABLE} "
            f"WHERE {TIME_COLUMN} >= '{MONTH_START}' AND {TIME_COLUMN} < '{MONTH_END}'"
        )
        assert df.collect()[0]["n"] == expected


class TestHivePartitioning:
    def test_the_partition_column_is_read_from_the_data_not_the_path(
        self, spark: SparkSession
    ) -> None:
        """The warehouse lays data out as `.../pickup_month=2024-12/...`, which DuckDB
        auto-detects as Hive partitioning and would synthesise a *typed* column from.
        The value has to come from the file."""
        df = spark.table(f"{NAMESPACE}.{TABLE}").select(TIME_COLUMN).limit(1)
        value = df.collect()[0][0]
        assert isinstance(value, datetime.datetime), (
            f"{TIME_COLUMN} came back as {type(value).__name__}, "
            f"which means DuckDB sourced it from the directory name"
        )

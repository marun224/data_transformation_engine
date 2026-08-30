"""Integration checks against the real REST catalog and MinIO.

Deselected by default. Run them on the Windows box once `.env` is filled in:

    uv run pytest -m integration

They read configuration from the environment exactly as the smoke script does, so a
green run here means a green `scripts/smoke_catalog.py`.
"""

from __future__ import annotations

import os

import pytest
from rich.console import Console

from icetl.catalog import CatalogRegistry, TableResolver
from icetl.conf import IcetlSettings, resolve_settings
from icetl.diagnostics.smoke import run_smoke_test
from icetl.exec import DuckDBEngine
from icetl.paths import engine_paths

pytestmark = pytest.mark.integration

NAMESPACE = os.environ.get("ICETL_TEST_NAMESPACE", "nyc")
TABLE = os.environ.get("ICETL_TEST_TABLE", "yellow_tripdata")


@pytest.fixture(scope="module")
def settings() -> IcetlSettings:
    return resolve_settings()


def test_smoke_test_passes_end_to_end(settings: IcetlSettings) -> None:
    result = run_smoke_test(
        namespace=NAMESPACE,
        table_name=TABLE,
        limit=10,
        settings=settings,
        console=Console(quiet=True),
    )
    assert result.ok, [(name, detail) for name, ok, detail in result.steps if not ok]
    assert result.rows.num_rows > 0


def test_catalog_lists_the_target_namespace(settings: IcetlSettings) -> None:
    catalog = CatalogRegistry(settings).get()
    namespaces = {".".join(n) for n in catalog.list_namespaces()}
    assert NAMESPACE in namespaces


def test_duckdb_and_pyiceberg_agree_on_row_count(settings: IcetlSettings) -> None:
    """The Phase 0 correctness claim: both engines see the same rows in the same files.

    Run over a single data file to keep it quick regardless of table size.
    """
    registry = CatalogRegistry(settings)
    resolved = TableResolver(registry, default_namespace=tuple(NAMESPACE.split("."))).resolve(
        f"{NAMESPACE}.{TABLE}"
    )

    tasks = [t for t in resolved.table.scan().plan_files() if not t.delete_files][:1]
    if not tasks:
        pytest.skip("table has no delete-free data files to compare")

    expected = tasks[0].file.record_count
    paths = engine_paths([tasks[0].file.file_path])

    engine = DuckDBEngine(settings)
    engine.ensure_object_store(paths)
    actual = engine.arrow("SELECT count(*) AS n FROM read_parquet($paths)", {"paths": paths})
    engine.close()

    assert actual.column("n").to_pylist() == [expected]


def test_object_store_paths_translate(settings: IcetlSettings) -> None:
    """Whatever scheme the catalog reports must survive translation into DuckDB."""
    registry = CatalogRegistry(settings)
    resolved = TableResolver(registry, default_namespace=tuple(NAMESPACE.split("."))).resolve(
        f"{NAMESPACE}.{TABLE}"
    )
    tasks = list(resolved.table.scan(limit=1).plan_files())
    if not tasks:
        pytest.skip("table has no data files")

    translated = engine_paths([t.file.file_path for t in tasks])
    assert all(not p.startswith(("s3a://", "s3n://", "file://")) for p in translated)

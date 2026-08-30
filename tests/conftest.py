"""Shared fixtures: a local Iceberg catalog and the generated fixture tables.

Everything here is offline. The catalog is sqlite, the warehouse is a temp directory,
and nothing loads DuckDB's httpfs extension -- so the default suite runs with no
network and no REST catalog. Tests needing the real thing are marked `integration`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from icetl.catalog import CatalogRegistry, TableResolver
from icetl.conf import CatalogSettings, EngineSettings, IcetlSettings, SqlSettings
from icetl.exec import DuckDBEngine
from icetl.plan.builder import ScanSource
from icetl.sql import Session
from tests.fixtures import FixtureTable, build_all, local_catalog, warehouse_uri

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog


@pytest.fixture(scope="session")
def warehouse_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("warehouse")


@pytest.fixture(scope="session")
def catalog(warehouse_root: Path) -> SqlCatalog:
    """A local SqlCatalog with every fixture table already built.

    Session-scoped: building the 200-column fixture is the slowest thing in the suite
    and nothing mutates these tables.
    """
    built = local_catalog(warehouse_root)
    build_all(built)
    return built


@pytest.fixture(scope="session")
def fixtures(catalog: SqlCatalog) -> dict[str, FixtureTable]:
    """The fixture tables, keyed by short name (`plain`, `wide`, `renamed`, ...)."""
    return {
        name: FixtureTable(
            identifier=f"fx.{name}",
            table=catalog.load_table(f"fx.{name}"),
            rows=0,
            note="",
        )
        for name in ("plain", "partitioned", "wide", "nested", "renamed", "mor")
    }


@pytest.fixture(scope="session")
def local_settings(warehouse_root: Path) -> IcetlSettings:
    """Settings pointing at the local catalog, with no S3 configured."""
    return IcetlSettings(
        catalog=CatalogSettings(
            name="test",
            type="sql",
            uri=f"sqlite:///{(warehouse_root / 'catalog.db').as_posix()}",
            warehouse=warehouse_uri(warehouse_root),
        ),
        default_namespace=("fx",),
    )


@pytest.fixture
def registry(local_settings: IcetlSettings, catalog: SqlCatalog) -> CatalogRegistry:
    """A registry wired to the local catalog, so nothing connects to a network."""
    built = CatalogRegistry(local_settings)
    built.register(local_settings.catalog.name, catalog)
    return built


@pytest.fixture
def resolver(registry: CatalogRegistry) -> TableResolver:
    return TableResolver(registry)


@pytest.fixture
def engine(local_settings: IcetlSettings, tmp_path: Path) -> Iterator[DuckDBEngine]:
    """A DuckDB engine spilling into the test's own temp directory."""
    settings = replace(
        local_settings, engine=EngineSettings(temp_directory=str(tmp_path / "spill"))
    )
    built = DuckDBEngine(settings)
    yield built
    built.close()


@pytest.fixture
def session(
    local_settings: IcetlSettings, catalog: SqlCatalog, tmp_path: Path
) -> Iterator[Session]:
    """A Session wired to the local fixture catalog.

    Built directly rather than through `Session.builder`, so tests never touch
    the process-wide active session and can run in any order.
    """
    settings = replace(
        local_settings, engine=EngineSettings(temp_directory=str(tmp_path / "spill"))
    )
    session = Session(settings=settings, catalog=catalog)
    yield session
    session.stop()


@pytest.fixture
def ansi_session(
    local_settings: IcetlSettings, catalog: SqlCatalog, tmp_path: Path
) -> Iterator[Session]:
    """A session with `icetl.ansiMode=true`, for the strict-cast cases."""
    settings = replace(
        local_settings,
        engine=EngineSettings(temp_directory=str(tmp_path / "spill")),
        sql=SqlSettings(ansi_mode=True),
    )
    session = Session(settings=settings, catalog=catalog)
    yield session
    session.stop()


#: The fixture tables, in the order the `sources` fixture numbers their views.
FIXTURE_NAMES = ("plain", "partitioned", "wide", "nested", "renamed", "mor")


@pytest.fixture
def sources(resolver: TableResolver) -> dict[str, ScanSource]:
    """Real `ScanSource`s for every fixture table, keyed as a plan would spell them.

    Built through the actual resolver rather than a stand-in, so anything testing the
    planner is testing it against the same objects the session hands it.
    """
    return {
        f"fx.{name}": ScanSource(
            key=f"fx.{name}",
            resolved=resolver.resolve(f"fx.{name}"),
            view=f"icetl_src_{index}",
        )
        for index, name in enumerate(FIXTURE_NAMES)
    }

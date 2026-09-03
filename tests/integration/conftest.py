"""Fixtures for the integration suite: a real REST catalog, real MinIO, real data.

Everything here is the opposite of `tests/conftest.py`. That one is deliberately
offline -- sqlite, a temp directory, no network. This one talks to the catalog named in
`.env` and the object store behind it, and every table it reads is one somebody actually
loaded.

Three things it is responsible for:

**Configuration comes from the environment, never from here.** `resolve_settings()` reads
`.env` exactly as `scripts/smoke_catalog.py` does, so a green run here means a green
smoke script. Table names, namespaces and the seed window are all env-overridable, so
pointing the suite at a different warehouse is configuration rather than editing.

**The seeded tables are built once.** Session-scoped and idempotent: a second run reuses
what the first built, at the cost of one metadata read per table. `--it-reseed` forces
the rebuild.

**The real tables are proved untouched.** `guard.Witness` reads every protected table's
snapshot id, row count and schema id before any test runs and again after the last one,
and fails the session if anything moved. Write tests are only safe here because that
check does not depend on the suite being right about where it writes.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pytest

from icetl.catalog import CatalogRegistry
from icetl.conf import EngineSettings, IcetlSettings, SqlSettings, resolve_settings
from icetl.sql import Session
from tests.integration.guard import Witness, it_namespace, safe_drop
from tests.integration.seed import SeededTable, seed_real_data, seed_replicas

if TYPE_CHECKING:
    from pathlib import Path

    from pyiceberg.catalog import Catalog

# --- what the suite points at ------------------------------------------------
# Every one of these is an env override with a working default, so nothing in the
# suite carries a literal table name.

NAMESPACE = os.environ.get("ICETL_TEST_NAMESPACE", "nyc")
TABLE = os.environ.get("ICETL_TEST_TABLE", "yellow_tripdata")
WIDE_TABLE = os.environ.get("ICETL_TEST_WIDE_TABLE", "wide_smoke")
HUGE_TABLE = os.environ.get("ICETL_TEST_HUGE_TABLE", "yellow_tripdata_wide")
TIME_COLUMN = os.environ.get("ICETL_TEST_TIME_COLUMN", "tpep_pickup_datetime")

#: Fully qualified, because that is how tests spell them.
REAL_TABLE = f"{NAMESPACE}.{TABLE}"
REAL_WIDE = f"{NAMESPACE}.{WIDE_TABLE}"
REAL_HUGE = f"{NAMESPACE}.{HUGE_TABLE}"


@pytest.fixture(scope="session")
def reseed(pytestconfig: pytest.Config) -> bool:
    """Whether `--it-reseed` was passed. The option is declared in `tests/conftest.py`,
    because pytest only honours `pytest_addoption` in an initial conftest."""
    return bool(pytestconfig.getoption("--it-reseed"))


@pytest.fixture(scope="session")
def base_settings() -> IcetlSettings:
    """Settings from `.env` / the environment, as the smoke script resolves them.

    Deliberately *not* named `settings`: the two pre-existing integration modules
    each define a module-scoped fixture by that name, which would shadow this one
    and give the session-scoped fixtures below a narrower scope than they declare.
    """
    return resolve_settings()


@pytest.fixture(scope="session")
def spill_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("it-spill")


@pytest.fixture(scope="session")
def it_settings(base_settings: IcetlSettings, spill_root: Path) -> IcetlSettings:
    """The settings every session fixture below is built from.

    Spill goes to the test's own temp directory rather than the developer's configured
    one, so a run that spills cannot fill a directory somebody cares about.
    """
    from dataclasses import replace

    return replace(base_settings, engine=EngineSettings(temp_directory=str(spill_root / "duckdb")))


@pytest.fixture(scope="session")
def catalog(it_settings: IcetlSettings) -> Iterator[Catalog]:
    registry = CatalogRegistry(it_settings)
    yield registry.get()
    registry.close()


@pytest.fixture(scope="session")
def namespace() -> str:
    """The namespace the suite owns. Raises if it resolves to a protected one."""
    return it_namespace()


@pytest.fixture(scope="session", autouse=True)
def witness(catalog: Catalog) -> Iterator[Witness]:
    """Prove the real tables were not touched. Autouse: this is not opt-in.

    Captured before the seeder runs, verified after the last test -- so it covers the
    seeding too, which is the one part of the suite that writes without a test watching.
    """
    watching = Witness.capture(catalog)
    yield watching
    watching.verify()


@pytest.fixture(scope="session")
def it_session(it_settings: IcetlSettings, catalog: Catalog) -> Iterator[Session]:
    """A session-scoped session, used for seeding and by the read-only tests.

    Built directly rather than through `Session.builder`, so the suite never touches
    the process-wide active session and tests can run in any order. Unlike the
    pre-existing integration modules, this one is closed.
    """
    session = Session(settings=it_settings, catalog=catalog)
    yield session
    session.stop()


@pytest.fixture(scope="session")
def replicas(
    catalog: Catalog, namespace: str, witness: Witness, reseed: bool
) -> dict[str, SeededTable]:
    """The six fixture tables, rebuilt in the real catalog.

    Depends on `witness` only for ordering: the protected tables must be read before
    anything writes anywhere.
    """
    return seed_replicas(catalog, namespace, reseed=reseed)


@pytest.fixture(scope="session")
def real(
    it_session: Session, catalog: Catalog, namespace: str, witness: Witness, reseed: bool
) -> dict[str, SeededTable]:
    """Slices of the real table, carved out by icetl's own write path."""
    return seed_real_data(
        it_session,
        catalog,
        namespace,
        source=REAL_TABLE,
        time_column=TIME_COLUMN,
        reseed=reseed,
    )


# --- convenience accessors ---------------------------------------------------
# Named for what they are, so a test signature says which data it needs.


@pytest.fixture(scope="session")
def plain(replicas: dict[str, SeededTable]) -> str:
    """The 5-row replica with NULLs in two different columns. The workhorse."""
    return replicas["plain"].identifier


@pytest.fixture(scope="session")
def partitioned(replicas: dict[str, SeededTable]) -> str:
    return replicas["partitioned"].identifier


@pytest.fixture(scope="session")
def wide(replicas: dict[str, SeededTable]) -> str:
    return replicas["wide"].identifier


@pytest.fixture(scope="session")
def nested(replicas: dict[str, SeededTable]) -> str:
    return replicas["nested"].identifier


@pytest.fixture(scope="session")
def renamed(replicas: dict[str, SeededTable]) -> str:
    return replicas["renamed"].identifier


@pytest.fixture(scope="session")
def mor(replicas: dict[str, SeededTable]) -> str:
    """The merge-on-read table. Nothing reads it successfully -- that is the test."""
    return replicas["mor"].identifier


@pytest.fixture(scope="session")
def trips(real: dict[str, SeededTable]) -> str:
    """A week of real trips, partitioned by VendorID. Real NULLs, real cardinality."""
    return real["trips"].identifier


@pytest.fixture(scope="session")
def trips_small(real: dict[str, SeededTable]) -> str:
    """5k real rows -- the template write and row-level tests copy."""
    return real["trips_small"].identifier


@pytest.fixture(scope="session")
def zones(real: dict[str, SeededTable]) -> str:
    """One row per real pickup location. The join partner with a real key."""
    return real["zones"].identifier


# --- per-test fixtures -------------------------------------------------------


@pytest.fixture
def session(it_settings: IcetlSettings, catalog: Catalog) -> Iterator[Session]:
    """A fresh session per test, for anything that writes or caches.

    Function-scoped because a write invalidates source pins and a UDF registration is
    session state -- sharing either between tests makes failures depend on order.
    """
    built = Session(settings=it_settings, catalog=catalog)
    yield built
    built.stop()


@pytest.fixture
def ansi_session(it_settings: IcetlSettings, catalog: Catalog) -> Iterator[Session]:
    """A session with `icetl.ansiMode=true`, for the strict-cast cases."""
    from dataclasses import replace

    built = Session(settings=replace(it_settings, sql=SqlSettings(ansi_mode=True)), catalog=catalog)
    yield built
    built.stop()


@pytest.fixture
def target(catalog: Catalog, namespace: str) -> Iterator[str]:
    """A table name nothing else uses, dropped when the test ends.

    The same convention the local write tests use, with the drop routed through
    `safe_drop` so it cannot reach outside the suite's own namespace.
    """
    import uuid

    name = f"{namespace}.t_{uuid.uuid4().hex[:8]}"
    yield name
    safe_drop(catalog, name)

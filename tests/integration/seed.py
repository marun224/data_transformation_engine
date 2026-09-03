"""The tables the integration suite builds for itself, in the real catalog.

Two layers, because the suite needs two different things from its data and no single
table gives both.

**Layer 1 -- replicas.** The same six fixtures the default suite uses
(`tests/fixtures/generator.py`), built into the integration namespace. Their contents
are known exactly, which is what lets a test assert `== 30.25` instead of comparing two
engines and hoping. What changes versus the local suite is everything underneath: a REST
catalog rather than sqlite, MinIO rather than a temp directory, `s3://` paths rather than
`file://` ones, and a real object store between DuckDB and the parquet. That is where
the defects have been -- STATUS.md records three found only against a real table -- so
running the *same* assertions over *different* infrastructure is the point.

They also supply, in one move, the three shapes no real table here has: nested types, a
column-rename history, and merge-on-read delete files.

**Layer 2 -- real data.** Slices of `nyc.yellow_tripdata`, carved out by icetl itself.
Real NULLs (June 2024 is ~12% null in `passenger_count`), real cardinality, real
timestamps, real float noise. Used where those properties *are* the thing under test,
and where an expectation is derived rather than written down.

Everything here is **idempotent**: a table that already exists with the expected row
count is reused, so a repeat run costs one metadata read per table instead of a rebuild.
`--it-reseed` forces the rebuild.

Nothing in this module writes outside the namespace `guard.it_namespace()` returns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests.fixtures import FIXTURE_BUILDERS, FixtureTable
from tests.integration.guard import ensure_namespace, safe_drop, safe_identifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session

__all__ = [
    "REPLICA_ROWS",
    "SeededTable",
    "seed_real_data",
    "seed_replicas",
    "trip_window",
]

#: Row counts the six replicas are built with. Used for the idempotency check, and as
#: the expected values in the tests that read them.
#:
#: `mor` is the odd one: 8 rows in two data files, of which a delete file removes two.
#: Nothing can read it -- the scan is refused (decision 11) -- so the count is recorded
#: here rather than queried.
REPLICA_ROWS = {
    "plain": 5,
    "partitioned": 12,
    "wide": 500,
    "nested": 2,
    "renamed": 4,
    "mor": 6,
}

#: Tables whose contents cannot be counted through icetl, and so are trusted rather
#: than verified when deciding whether to reuse them.
_UNREADABLE = frozenset({"mor"})


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def trip_window() -> tuple[str, str]:
    """The half-open date range the real-data seeds are carved from.

    A week rather than the whole month: it keeps the seed under a minute and the fast
    tier under its budget, while still being several hundred thousand rows of genuine
    data with genuine NULLs. Widen it with `ICETL_IT_SEED_START` / `_END` when a test
    wants more.
    """
    return _env("ICETL_IT_SEED_START", "2024-06-01"), _env("ICETL_IT_SEED_END", "2024-06-08")


@dataclass(frozen=True)
class SeededTable:
    """A table this module built or reused, and the facts tests assert against."""

    identifier: str
    rows: int
    note: str = ""
    reused: bool = False


# ---------------------------------------------------------------------------
# Layer 1 -- the fixture replicas
# ---------------------------------------------------------------------------


def seed_replicas(
    catalog: Catalog, namespace: str, *, reseed: bool = False
) -> dict[str, SeededTable]:
    """Build the six fixture tables into `namespace`, reusing what is already there.

    Each is checked individually rather than the set as a whole, so a partially-built
    namespace -- an interrupted run, a single dropped table -- repairs itself instead
    of needing a manual clean-up.
    """
    ensure_namespace(catalog, namespace)
    existing = {".".join(parts) for parts in catalog.list_tables(namespace)}

    seeded: dict[str, SeededTable] = {}
    for name, builder in FIXTURE_BUILDERS.items():
        identifier = safe_identifier(f"{namespace}.{name}")
        rows = REPLICA_ROWS[name]

        if identifier in existing:
            if not reseed and _replica_is_intact(catalog, identifier, name, rows):
                seeded[name] = SeededTable(identifier, rows, note="reused", reused=True)
                continue
            safe_drop(catalog, identifier)

        built: FixtureTable = builder(catalog, namespace=namespace)
        seeded[name] = SeededTable(built.identifier, built.rows, note=built.note)
    return seeded


def _replica_is_intact(catalog: Catalog, identifier: str, name: str, expected: int) -> bool:
    """Whether an existing replica still looks like the fixture it should be.

    Row count via PyIceberg's manifest summary, not a scan -- the check has to be
    cheap enough to run on every table at the start of every session, and reading the
    rows would defeat the point of caching them.
    """
    try:
        table = catalog.load_table(identifier)
    except Exception:
        return False
    if name in _UNREADABLE:
        # A delete file makes the summary's `total-records` disagree with the live row
        # count, and nothing may read the table to settle it. Existence is the check.
        return table.current_snapshot() is not None
    snapshot = table.current_snapshot()
    if snapshot is None or snapshot.summary is None:
        return False
    return snapshot.summary.get("total-records") == str(expected)


# ---------------------------------------------------------------------------
# Layer 2 -- slices of the real table
# ---------------------------------------------------------------------------


def seed_real_data(
    session: Session,
    catalog: Catalog,
    namespace: str,
    *,
    source: str,
    time_column: str,
    reseed: bool = False,
) -> dict[str, SeededTable]:
    """Carve the real-data tables out of `source`, using icetl's own write path.

    Built through `df.write` rather than PyIceberg directly, deliberately: the seed is
    then itself a test of the write path against a real catalog, and it fails loudly at
    session setup rather than subtly inside whichever test ran first.

    `trips` is partitioned by `VendorID` -- an identity partition over three real
    values, which gives several data files and therefore something for dynamic
    partition overwrite and file pruning to bite on.
    """
    ensure_namespace(catalog, namespace)
    start, end = trip_window()
    window = f"{time_column} >= '{start}' AND {time_column} < '{end}'"

    def build_trips(name: str) -> None:
        session.sql(f"SELECT * FROM {source} WHERE {window}").write.partitionBy(
            "VendorID"
        ).saveAsTable(name)

    def build_small(name: str) -> None:
        # A cheap template for the write and row-level tests to copy: those build a
        # throwaway table per test, and copying the full week each time would dominate
        # the run. Ordered, so the slice is the same rows on every seed.
        session.sql(
            f"SELECT * FROM {namespace}.trips ORDER BY {time_column}, PULocationID LIMIT 5000"
        ).write.saveAsTable(name)

    def build_zones(name: str) -> None:
        # A join partner keyed on a column that really is in the data, so the join
        # tests exercise a real key distribution rather than a synthetic 1:1 one.
        session.sql(
            f"SELECT DISTINCT PULocationID AS zone_id, "
            f"concat('zone-', CAST(PULocationID AS STRING)) AS zone_name "
            f"FROM {namespace}.trips"
        ).write.saveAsTable(name)

    # Order matters: `trips_small` and `zones` are both derived from `trips`.
    plans = [
        ("trips", build_trips, f"real rows, {start}..{end}, partitioned by VendorID"),
        ("trips_small", build_small, "5k real rows, unpartitioned"),
        ("zones", build_zones, "one row per real pickup location"),
    ]

    seeded: dict[str, SeededTable] = {}
    for name, build, note in plans:
        seeded[name] = _ensure(
            session, catalog, f"{namespace}.{name}", build, note=note, reseed=reseed
        )
    return seeded


def _ensure(
    session: Session,
    catalog: Catalog,
    identifier: str,
    build: Callable[[str], None],
    *,
    note: str,
    reseed: bool,
) -> SeededTable:
    """Build `identifier` unless it is already there, and report which happened.

    The row count comes back through `count()`, which icetl answers from the manifests
    rather than by opening a parquet footer -- so verifying a reused table costs a
    metadata read, not a scan.
    """
    safe_identifier(identifier)
    if reseed:
        safe_drop(catalog, identifier)
    reused = _exists(catalog, identifier)
    if not reused:
        build(identifier)
        session.catalog.refreshTable(identifier)
    return SeededTable(
        identifier,
        rows=session.table(identifier).count(),
        note=f"{note} (reused)" if reused else note,
        reused=reused,
    )


def _exists(catalog: Catalog, identifier: str) -> bool:
    try:
        catalog.load_table(identifier)
    except Exception:
        return False
    return True

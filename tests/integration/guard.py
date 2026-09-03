"""The rule that makes it safe to run write tests against a live catalog.

The default suite writes into a throwaway sqlite catalog, so a misdirected write costs
nothing. Here it would land in a REST catalog holding real tables -- 41M rows and 317
snapshots of data nobody wants rewritten by a test run.

Care is not a control. Every write path in the integration suite therefore goes through
this module, which knows exactly one thing: **which namespace the tests own**. Anything
outside it is refused before it reaches the catalog, and the refusal is an error the run
cannot swallow.

Three layers, in the order they fire:

  1. `it_namespace()` refuses to *resolve* to a protected namespace, so a stray
     `ICETL_IT_NAMESPACE=nyc` fails at collection rather than at teardown.
  2. `safe_drop()` and `safe_identifier()` refuse a table outside that namespace, so a
     typo'd fixture drops nothing.
  3. `Witness` reads the protected tables before and after the run and fails if
     anything moved -- the backstop for a write that got past the first two.

Layer 3 is the one that matters. The first two prevent the mistakes anyone anticipates;
the third notices the ones nobody did.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

__all__ = [
    "DEFAULT_IT_NAMESPACE",
    "PROTECTED_NAMESPACES",
    "ProtectedNamespaceError",
    "Witness",
    "ensure_namespace",
    "it_namespace",
    "safe_drop",
    "safe_identifier",
]

#: Namespaces the suite must never write to. Everything the user already had.
PROTECTED_NAMESPACES = frozenset(
    part.strip()
    for part in os.environ.get("ICETL_IT_PROTECTED", "nyc,amazon").split(",")
    if part.strip()
)

#: Where write tests build their tables. Created on demand, dropped at the end.
DEFAULT_IT_NAMESPACE = "icetl_it"


class ProtectedNamespaceError(AssertionError):
    """Raised when something addressed a namespace the suite does not own.

    An `AssertionError` rather than a custom hierarchy: this is a test-harness bug, not
    a condition any product code should catch, and pytest reports it as a failure
    without any registration.
    """


def it_namespace() -> str:
    """The namespace the integration suite owns, from the environment.

    Refuses to hand back a protected one. A misconfigured `.env` is the likeliest way
    this suite would ever touch real data, so it fails here -- at import, before a
    single table is built -- rather than somewhere inside a teardown.
    """
    name = os.environ.get("ICETL_IT_NAMESPACE", DEFAULT_IT_NAMESPACE).strip()
    if not name:
        raise ProtectedNamespaceError("ICETL_IT_NAMESPACE is set but empty")
    if name in PROTECTED_NAMESPACES:
        raise ProtectedNamespaceError(
            f"ICETL_IT_NAMESPACE={name!r} names a protected namespace. The integration "
            f"suite creates and drops tables in whatever it is given, so it must never "
            f"be pointed at one holding real data. Protected: "
            f"{', '.join(sorted(PROTECTED_NAMESPACES))}."
        )
    return name


def safe_identifier(name: str) -> str:
    """Return `name` unchanged, having checked the suite is allowed to write to it.

    Accepts the `namespace.table` spelling used everywhere in the tests. A bare table
    name is refused rather than assumed to be in the right namespace: guessing is how
    a write ends up somewhere unintended.
    """
    parts = name.split(".")
    if len(parts) != 2:
        raise ProtectedNamespaceError(
            f"{name!r} is not a `namespace.table` reference. Write targets must be "
            f"fully qualified so the namespace can be checked."
        )
    namespace = parts[0]
    if namespace != it_namespace():
        raise ProtectedNamespaceError(
            f"{name!r} is outside the integration namespace {it_namespace()!r}. "
            f"Write tests may only create and drop tables the suite owns."
        )
    return name


def safe_drop(catalog: Catalog, name: str) -> bool:
    """Drop a table the suite owns. Returns whether anything was dropped.

    The only sanctioned way to drop anything in the integration suite -- nothing calls
    `catalog.drop_table` directly. A missing table is not an error, because this runs
    in teardown where the test may have failed before creating it.
    """
    safe_identifier(name)
    try:
        catalog.drop_table(name)
    except Exception:
        # Includes NoSuchTableError. Teardown must not mask the failure that got us
        # here, so nothing is re-raised.
        return False
    return True


def ensure_namespace(catalog: Catalog, namespace: str) -> None:
    """Create `namespace` if the catalog does not already have it."""
    if namespace in PROTECTED_NAMESPACES:
        raise ProtectedNamespaceError(f"refusing to create protected namespace {namespace!r}")
    existing = {".".join(parts) for parts in catalog.list_namespaces()}
    if namespace not in existing:
        catalog.create_namespace(namespace)


@dataclass(frozen=True)
class _TableState:
    """What a protected table looked like at a point in time."""

    snapshot_id: int | None
    records: str | None
    schema_id: int


@dataclass
class Witness:
    """Before-and-after proof that the real tables were not touched.

    `capture()` at session start, `verify()` at session end. It reads the current
    snapshot id, the row count the snapshot summary reports, and the schema id -- the
    three things any write of any kind would change. All of it is metadata, so the
    check costs one catalog round trip per table and reads no data.

    This is the backstop, and the reason write tests can run here at all: it does not
    depend on the suite being correct about where it writes, only on the catalog being
    honest about what happened.
    """

    catalog: Catalog
    before: dict[str, _TableState]

    @classmethod
    def capture(cls, catalog: Catalog) -> Witness:
        return cls(catalog=catalog, before=_read_protected(catalog))

    def verify(self) -> None:
        """Fail if any protected table moved. Called once, at session teardown."""
        after = _read_protected(self.catalog)
        moved = [
            f"  {name}: {self.before[name]} -> {after[name]}"
            for name in sorted(self.before)
            if name in after and after[name] != self.before[name]
        ]
        vanished = sorted(set(self.before) - set(after))
        appeared = sorted(set(after) - set(self.before))
        problems = []
        if moved:
            problems.append("changed:\n" + "\n".join(moved))
        if vanished:
            problems.append(f"disappeared: {', '.join(vanished)}")
        if appeared:
            problems.append(f"appeared: {', '.join(appeared)}")
        if problems:
            raise ProtectedNamespaceError(
                "the integration run modified tables it does not own -- " + "; ".join(problems)
            )


def _read_protected(catalog: Catalog) -> dict[str, _TableState]:
    """Snapshot-level state of every table in every protected namespace."""
    state: dict[str, _TableState] = {}
    for parts in catalog.list_namespaces():
        namespace = ".".join(parts)
        if namespace not in PROTECTED_NAMESPACES:
            continue
        for identifier in catalog.list_tables(parts):
            name = ".".join(identifier)
            table = catalog.load_table(identifier)
            snapshot = table.current_snapshot()
            state[name] = _TableState(
                snapshot_id=snapshot.snapshot_id if snapshot else None,
                records=(
                    snapshot.summary.get("total-records")
                    if snapshot is not None and snapshot.summary is not None
                    else None
                ),
                schema_id=table.schema().schema_id,
            )
    return state

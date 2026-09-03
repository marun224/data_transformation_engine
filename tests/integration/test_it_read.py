"""The read path against the real catalog -- and the proof that the rig itself works.

This module is the one to run first after any change to `conftest.py`, `seed.py`,
`guard.py` or `helpers.py`. Everything else in the integration suite assumes what is
asserted here: that the replicas really were built into the REST catalog, that they hold
the values the local fixtures hold, that the real tables read back, and that the guard
would notice if any of it went wrong.

`TestTheReplicasAreReal` is the load-bearing class. It asserts the *same* values
`tests/fixture/test_dataframe.py` asserts, over completely different infrastructure --
REST rather than sqlite, MinIO rather than a temp directory, `s3://` rather than
`file://`. Identical expectations, different substrate, which is the whole design: when
one of these fails and its local twin passes, the difference is the infrastructure and
nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.conftest import REAL_TABLE, REAL_WIDE, TIME_COLUMN
from tests.integration.helpers import agree_across_surfaces, column, scan_of

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

    from icetl.sql.session import Session
    from tests.integration.guard import Witness
    from tests.integration.seed import SeededTable

pytestmark = pytest.mark.integration


class TestTheRigItself:
    """The seeding and safety machinery, asserted rather than assumed."""

    def test_every_replica_was_built_into_the_real_catalog(
        self, catalog: Catalog, namespace: str, replicas: dict[str, SeededTable]
    ) -> None:
        built = {".".join(parts) for parts in catalog.list_tables(namespace)}
        for name, seeded in replicas.items():
            assert seeded.identifier in built, f"{name} was not created in {namespace}"

    def test_the_replicas_live_on_the_object_store_not_a_local_disk(
        self, it_session: Session, plain: str
    ) -> None:
        """The point of building them here rather than locally.

        If these came back as `file://` paths the suite would be testing the local
        fixtures a second time, with extra steps.
        """
        scan = scan_of(it_session.table(plain))
        paths = [path for group in scan.groups for path in group.paths]
        assert paths, "the replica has no data files"
        assert all(path.startswith("s3://") for path in paths), paths

    def test_the_guard_refuses_a_protected_namespace(self) -> None:
        from tests.integration.guard import ProtectedNamespaceError, safe_identifier

        with pytest.raises(ProtectedNamespaceError, match="outside the integration namespace"):
            safe_identifier(f"{REAL_TABLE}")

    def test_the_guard_refuses_an_unqualified_name(self) -> None:
        from tests.integration.guard import ProtectedNamespaceError, safe_identifier

        with pytest.raises(ProtectedNamespaceError, match=r"not a `namespace\.table`"):
            safe_identifier("just_a_table")

    def test_the_witness_is_watching_the_real_tables(self, witness: Witness) -> None:
        """A witness watching nothing would pass every run and prove nothing."""
        assert witness.before, "the witness captured no protected tables"
        assert any(name.startswith("nyc.") for name in witness.before), sorted(witness.before)


class TestTheReplicasAreReal:
    """The local fixtures' own expectations, re-asserted over REST + MinIO."""

    def test_plain_holds_the_rows_the_local_fixture_holds(
        self, it_session: Session, plain: str
    ) -> None:
        rows = it_session.table(plain).orderBy("id").collect()
        assert [row["id"] for row in rows] == [1, 2, 3, 4, 5]
        assert [row["vendor"] for row in rows] == ["a", "b", "a", "c", None]
        assert [row["amount"] for row in rows] == [10.0, 20.5, 30.25, None, 50.0]

    def test_the_nulls_survived_the_round_trip(self, it_session: Session, plain: str) -> None:
        """NULL through a REST catalog, MinIO, parquet and DuckDB is still NULL.

        Worth its own test: a null that came back as an empty string or a 0.0 would
        make roughly 190 of the mirrored tests wrong in the same direction, quietly.
        """
        frame = it_session.table(plain)
        assert frame.filter(F.col("vendor").isNull()).count() == 1
        assert frame.filter(F.col("amount").isNull()).count() == 1
        assert frame.filter(F.col("vendor").isNull() & F.col("amount").isNull()).count() == 0

    def test_partitioned_has_its_three_partitions(
        self, it_session: Session, partitioned: str
    ) -> None:
        scan = scan_of(it_session.table(partitioned))
        assert it_session.table(partitioned).count() == 12
        assert scan.files_total == 3

    def test_wide_has_two_hundred_columns(self, it_session: Session, wide: str) -> None:
        assert len(it_session.table(wide).columns) == 200

    def test_nested_kept_its_complex_types(self, it_session: Session, nested: str) -> None:
        types = dict(it_session.table(nested).dtypes)
        assert types["person"].startswith("struct")
        assert types["tags"].startswith("array")
        assert types["scores"].startswith("map")

    def test_renamed_reads_both_sides_of_the_rename(
        self, it_session: Session, renamed: str
    ) -> None:
        """Field-id reconciliation, over an object store this time.

        Two data files disagree about what field 2 is called. Reading by name would
        give NULLs for the older two rows -- the silent-wrong-answer case (3.4).
        """
        values = sorted(v for v in column(it_session.table(renamed), "new_name") if v is not None)
        assert values == ["after-c", "after-d", "before-a", "before-b"]


class TestTheRealTables:
    """The tables the user actually loaded, read-only."""

    def test_the_real_table_reads(self, it_session: Session) -> None:
        rows = it_session.table(REAL_TABLE).select(TIME_COLUMN).limit(5).collect()
        assert len(rows) == 5

    def test_its_schema_keeps_the_mixed_case_spellings(self, it_session: Session) -> None:
        """Scripts index on these by name, so the casing is part of the contract."""
        columns = it_session.table(REAL_TABLE).columns
        assert "VendorID" in columns
        assert "Airport_fee" in columns

    def test_the_wide_table_is_wide(self, it_session: Session) -> None:
        assert len(it_session.table(REAL_WIDE).columns) > 100

    def test_a_seeded_slice_carries_real_nulls(self, it_session: Session, trips: str) -> None:
        """The property the real data was seeded for.

        June 2024 is roughly 12% null in `passenger_count`. A slice with no NULLs would
        mean the seed silently filtered them, and the null-semantics tests downstream
        would all be vacuous.
        """
        frame = it_session.table(trips)
        total = frame.count()
        nulls = frame.filter(F.col("passenger_count").isNull()).count()
        assert total > 0
        assert nulls > 0, "the seeded slice has no NULLs, so it is not real data"
        assert nulls < total

    def test_the_seeded_slice_has_the_real_vendors(self, it_session: Session, trips: str) -> None:
        vendors = sorted(
            v for v in column(it_session.table(trips).select("VendorID").distinct(), "VendorID")
        )
        assert len(vendors) >= 2, vendors

    def test_the_join_partner_has_one_row_per_location(
        self, it_session: Session, trips: str, zones: str
    ) -> None:
        distinct = it_session.table(trips).select("PULocationID").distinct().count()
        assert it_session.table(zones).count() == distinct


class TestBothSurfacesAgree:
    """P1, asked of the real catalog: the two surfaces are one code path."""

    def test_a_projection_agrees(self, it_session: Session, plain: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT id, vendor FROM {plain}",
            it_session.table(plain).select("id", "vendor"),
        )

    def test_a_filter_agrees(self, it_session: Session, plain: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT id FROM {plain} WHERE amount > 20",
            it_session.table(plain).filter(F.col("amount") > 20).select("id"),
        )

    def test_an_aggregate_over_real_data_agrees(self, it_session: Session, trips: str) -> None:
        agree_across_surfaces(
            it_session,
            f"SELECT VendorID, count(*) AS n FROM {trips} GROUP BY VendorID",
            it_session.table(trips).groupBy("VendorID").count().withColumnRenamed("count", "n"),
        )

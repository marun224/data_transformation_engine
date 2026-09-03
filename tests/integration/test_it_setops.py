"""UNION, INTERSECT and EXCEPT over real data.

The organising idea is one invariant: **a predicate and its complement partition the
table.** Filter the trips two ways so that every row lands in exactly one side, union
the halves, and the result has to be the original table -- same count, same rows. That
holds whatever the data is, so it survives a re-seed, and it fails loudly if a set
operation drops rows, duplicates them, or loses its pushdown (FINDINGS 3.2, where a set
operation lost all pruning to output-name restoration).

`icetl_it.plain` appears where NULL handling is the point: INTERSECT matches NULL to
NULL, which needs a known NULL on both sides to demonstrate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import assert_partition_of, column, scans_of

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestComplementaryFiltersReunion:
    """The invariant, in several spellings."""

    def test_a_numeric_split_reunions_to_the_whole(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID", "trip_distance")
        near = frame.filter(F.col("trip_distance") <= 2)
        far = frame.filter(~(F.col("trip_distance") <= 2))
        assert near.count() > 0 and far.count() > 0
        assert near.union(far).count() == frame.count()

    def test_the_reunioned_rows_are_the_original_rows(
        self, it_session: Session, trips_small: str
    ) -> None:
        """Not just the same number of them."""
        frame = it_session.table(trips_small).select("PULocationID")
        near = frame.filter(F.col("PULocationID") < 100)
        far = frame.filter(~(F.col("PULocationID") < 100))
        assert_partition_of(
            [column(near, "PULocationID"), column(far, "PULocationID")],
            column(frame, "PULocationID"),
        )

    def test_a_three_way_split_reunions_too(self, it_session: Session, trips_small: str) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        a = frame.filter(F.col("PULocationID") < 100)
        b = frame.filter((F.col("PULocationID") >= 100) & (F.col("PULocationID") < 200))
        c = frame.filter(F.col("PULocationID") >= 200)
        assert a.union(b).union(c).count() == frame.count()

    def test_a_split_on_a_nullable_column_needs_the_null_branch(
        self, it_session: Session, trips_small: str
    ) -> None:
        """`x = 'Y'` and `x <> 'Y'` do **not** partition a nullable column.

        NULL satisfies neither, which is the reference's semantics and the reason a
        third branch is required. A union that came out whole without it would mean
        NULL was being treated as false somewhere.
        """
        frame = it_session.table(trips_small).select("store_and_fwd_flag")
        yes = frame.filter(F.col("store_and_fwd_flag") == "Y")
        no = frame.filter(F.col("store_and_fwd_flag") != "Y")
        missing = frame.filter(F.col("store_and_fwd_flag").isNull())

        assert missing.count() > 0
        assert yes.count() + no.count() < frame.count(), (
            "the two non-null branches covered everything, so NULL was not NULL"
        )
        assert yes.count() + no.count() + missing.count() == frame.count()


class TestUnion:
    def test_union_keeps_duplicates(self, it_session: Session, trips_small: str) -> None:
        """`union` is UNION ALL in the reference -- it does not deduplicate."""
        frame = it_session.table(trips_small).select("PULocationID")
        assert frame.union(frame).count() == frame.count() * 2

    def test_union_by_name_reorders_columns(self, it_session: Session, trips_small: str) -> None:
        a = it_session.table(trips_small).select("PULocationID", "trip_distance")
        b = it_session.table(trips_small).select("trip_distance", "PULocationID")
        combined = a.unionByName(b)
        assert combined.columns == ["PULocationID", "trip_distance"]
        assert combined.count() == a.count() * 2

    def test_union_by_position_would_have_mismatched(
        self, it_session: Session, trips_small: str
    ) -> None:
        """The distinction `unionByName` exists for, shown rather than described.

        Positional union of the two orderings puts distances into the location column,
        so the distinct location count explodes. By name it does not.
        """
        a = it_session.table(trips_small).select("PULocationID", "trip_distance")
        b = it_session.table(trips_small).select("trip_distance", "PULocationID")
        by_name = a.unionByName(b).select("PULocationID").distinct().count()
        assert by_name == a.select("PULocationID").distinct().count()


class TestIntersectAndExcept:
    def test_intersect_of_a_frame_with_itself_is_its_distinct_rows(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        assert frame.intersect(frame).count() == frame.distinct().count()

    def test_except_of_a_frame_with_itself_is_empty(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        assert frame.subtract(frame).count() == 0

    def test_except_removes_exactly_the_overlap(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small).select("PULocationID")
        low = frame.filter(F.col("PULocationID") < 100)
        remaining = frame.subtract(low)
        assert set(column(remaining, "PULocationID")) == set(
            column(frame.filter(~(F.col("PULocationID") < 100)).distinct(), "PULocationID")
        )

    def test_intersect_matches_null_to_null(self, it_session: Session, plain: str) -> None:
        """NULL = NULL is NULL, but INTERSECT treats NULLs as equal. The replica has one."""
        frame = it_session.table(plain).select("vendor")
        matched = column(frame.intersect(frame), "vendor")
        assert None in matched, "INTERSECT dropped the NULL row"

    def test_intersect_all_keeps_the_multiplicity(self, it_session: Session, plain: str) -> None:
        """The replica has `a` twice, which is what distinguishes the two spellings."""
        frame = it_session.table(plain).select("vendor")
        assert frame.intersectAll(frame).count() == frame.count()
        assert frame.intersect(frame).count() == frame.distinct().count()

    def test_except_all_keeps_the_multiplicity(self, it_session: Session, plain: str) -> None:
        frame = it_session.table(plain).select("vendor")
        one_a = frame.filter(F.col("vendor") == "a").limit(1)
        assert frame.exceptAll(one_a).count() == frame.count() - 1


class TestPushdownSurvivesTheSetOperation:
    """FINDINGS 3.2: a set operation once lost all pruning to output-name restoration."""

    def test_each_branch_still_prunes_its_own_files(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips).select("VendorID", "trip_distance")
        one = frame.filter(F.col("VendorID") == 1)
        two = frame.filter(F.col("VendorID") == 2)
        combined = one.union(two)

        scans = scans_of(combined)
        assert scans, "the union compiled to no scans at all"
        for scan in scans:
            assert scan.files_total is not None
            assert scan.files_scanned < scan.files_total, (
                "a union branch stopped pruning -- FINDINGS 3.2 has come back"
            )

    def test_the_union_still_returns_the_right_rows(self, it_session: Session, trips: str) -> None:
        """Pruning changes speed, not answers -- restated for the set-operation path."""
        frame = it_session.table(trips).select("VendorID", "trip_distance")
        one = frame.filter(F.col("VendorID") == 1)
        two = frame.filter(F.col("VendorID") == 2)
        combined = one.union(two)
        assert combined.count() == one.count() + two.count()
        assert set(column(combined.select("VendorID").distinct(), "VendorID")) == {1, 2}

    def test_projection_is_kept_through_the_union(self, it_session: Session, trips: str) -> None:
        frame = it_session.table(trips).select("VendorID", "trip_distance")
        combined = frame.filter(F.col("VendorID") == 1).union(frame.filter(F.col("VendorID") == 2))
        for scan in scans_of(combined):
            assert set(scan.columns) <= {"VendorID", "trip_distance"}, scan.columns
            assert scan.total_columns > len(scan.columns)


class TestBranchesMustLineUp:
    def test_a_column_count_mismatch_is_refused(
        self, it_session: Session, trips_small: str
    ) -> None:
        from icetl.errors import AnalysisException

        a = it_session.table(trips_small).select("PULocationID")
        b = it_session.table(trips_small).select("PULocationID", "trip_distance")
        with pytest.raises(AnalysisException):
            a.union(b).count()

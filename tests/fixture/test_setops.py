"""Set operations end to end against the local fixture catalog.

`fx.plain` again, for its nulls: `id` is `1..5`, `vendor` is `a, b, a, c, NULL` and
`amount` is `10.0, 20.5, 30.25, NULL, 50.0`. Set operations are where NULL stops
behaving like it does in a `WHERE` clause -- `INTERSECT` matches NULL *to* NULL, where
`=` would not -- so the null row earns its place in most of these.

The suite carries no duplicate rows anywhere, which is inconvenient for the multiset
forms, so `plain.union(plain)` is used to manufacture them: every row exactly twice.

Two tests here are guarding wrong *answers* rather than errors, and are the reason the
branch-nesting rules in `_as_set_branch` exist:

* `TestPrecedence` -- DuckDB binds `INTERSECT` tighter than `UNION ALL`, so an inlined
  `a.union(b).intersect(c)` computes a different query that runs perfectly well.
* `TestNoteTen` -- a set operation whose branches needed renaming used to lose *every*
  optimization, so it read every file and every column while answering correctly.

Every assertion is on a **value**, per the rule Phase 3 established. Set operations do
not guarantee row order, so each test sorts before comparing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import AnalysisException, EngineTypeError, EngineValueError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.exec.scan_planner import ScanPlan
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def rows_of(df: DataFrame) -> list[tuple[Any, ...]]:
    """Rows as plain tuples, sorted with NULL last so the order is stable."""
    return sorted(
        (tuple(row) for row in df.collect()),
        key=lambda t: tuple((v is None, "" if v is None else str(v)) for v in t),
    )


def ids_of(df: DataFrame) -> list[Any]:
    return sorted(row[0] for row in df.collect())


def scan_of(df: DataFrame) -> ScanPlan:
    """The single scan a one-table query plans."""
    scans = df._session._compile(df._plan, df._sources, df.columns).scans
    assert len(scans) == 1
    return scans[0]


class TestUnionShape:
    def test_output_names_come_from_the_left_frame(self, session: Session) -> None:
        left = session.table("fx.plain").select(F.col("id").alias("key"))
        right = session.table("fx.plain").select(F.col("id").alias("other"))
        assert left.union(right).columns == ["key"]

    def test_union_all_is_the_same_method(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert rows_of(df.unionAll(df)) == rows_of(df.union(df))

    def test_union_matches_by_position_not_by_name(self, session: Session) -> None:
        """The reference engine's documented behaviour, and its sharpest edge.

        Two frames whose columns are the same names in a different order union into
        nonsense without complaint. Nothing about it is detectably wrong, which is
        exactly why `unionByName` exists -- so this pins the trap rather than a fix.

        Note what the widening does to the type: column one is `bigint` on the left and
        `string` on the right, so the union settles on `string` and the perfectly good
        `id` of 1 comes back as `'1'`. No error, no warning, both rows returned.
        """
        df = session.table("fx.plain").filter(F.col("id") == 1)
        swapped = df.select("vendor", "id")
        out = df.select("id", "vendor").union(swapped)
        assert out.dtypes == [("id", "string"), ("vendor", "string")]
        assert [tuple(row) for row in out.collect()] == [("1", "a"), ("a", "1")]


class TestUnionValues:
    def test_union_keeps_duplicates(self, session: Session) -> None:
        """`union` is `UNION ALL`. This is the one place the reference's naming misleads."""
        df = session.table("fx.plain")
        assert df.union(df).count() == 10
        assert ids_of(df.union(df)) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    def test_union_of_disjoint_filters_returns_both_halves(self, session: Session) -> None:
        df = session.table("fx.plain")
        low = df.filter(F.col("id") <= 2)
        high = df.filter(F.col("id") >= 4)
        assert ids_of(low.union(high)) == [1, 2, 4, 5]

    def test_a_union_branch_carrying_a_limit_is_nested(self, session: Session) -> None:
        """LIMIT binds to the whole set operation; DuckDB will not parse it mid-branch."""
        df = session.table("fx.plain")
        assert df.limit(2).union(df.limit(1)).count() == 3


class TestIntersectAndSubtract:
    def test_intersect_deduplicates(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert ids_of(df.union(df).intersect(df)) == [1, 2, 3, 4, 5]

    def test_intersect_all_keeps_the_smaller_multiplicity(self, session: Session) -> None:
        """Each row twice on the left, once on the right, so once out."""
        df = session.table("fx.plain")
        assert ids_of(df.union(df).intersectAll(df)) == [1, 2, 3, 4, 5]

    def test_except_all_subtracts_multiplicities(self, session: Session) -> None:
        """Twice on the left minus once on the right leaves one of each."""
        df = session.table("fx.plain")
        assert ids_of(df.union(df).exceptAll(df)) == [1, 2, 3, 4, 5]

    def test_except_all_of_a_frame_with_itself_is_empty(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.exceptAll(df).count() == 0

    def test_subtract_removes_the_matching_rows(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert ids_of(df.subtract(df.filter(F.col("id") <= 3))) == [4, 5]

    def test_subtract_deduplicates_where_except_all_would_not(self, session: Session) -> None:
        """The pair that makes the two spellings worth having separately."""
        df = session.table("fx.plain")
        doubled = df.union(df)
        assert doubled.subtract(df.filter(F.col("id") <= 3)).count() == 2
        assert doubled.exceptAll(df.filter(F.col("id") <= 3)).count() == 7


class TestNullMatchesNull:
    """`=` never matches NULL to NULL; set operations do. Both engines agree here."""

    def test_intersect_matches_the_null_row(self, session: Session) -> None:
        df = session.table("fx.plain")
        null_row = df.filter(F.col("vendor").isNull())
        assert null_row.count() == 1
        # If NULL did not match NULL this would be 0 rows, not 1.
        assert ids_of(df.intersect(null_row)) == [5]

    def test_subtract_removes_the_null_row(self, session: Session) -> None:
        df = session.table("fx.plain")
        remaining = df.subtract(df.filter(F.col("vendor").isNull()))
        assert ids_of(remaining) == [1, 2, 3, 4]


class TestUnionByName:
    def test_columns_are_lined_up_by_name_not_position(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") == 1)
        swapped = df.select("vendor", "id")
        out = df.select("id", "vendor").unionByName(swapped)
        assert out.columns == ["id", "vendor"]
        assert [tuple(row) for row in out.collect()] == [(1, "a"), (1, "a")]

    def test_names_match_case_insensitively(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") == 1)
        left = df.select(F.col("id").alias("Key"))
        right = df.select(F.col("id").alias("KEY"))
        assert left.unionByName(right).count() == 2

    def test_a_missing_column_raises_unless_it_is_allowed(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="allowMissingColumns"):
            df.select("id", "vendor").unionByName(df.select("id"))

    def test_allow_missing_columns_fills_with_null(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") == 1)
        wide = df.select("id", "vendor")
        narrow = df.select("id")
        out = wide.unionByName(narrow, allowMissingColumns=True)
        assert out.columns == ["id", "vendor"]
        assert rows_of(out) == [(1, "a"), (1, None)]

    def test_a_column_only_on_the_right_is_appended(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") == 1)
        out = df.select("id").unionByName(df.select("id", "amount"), allowMissingColumns=True)
        assert out.columns == ["id", "amount"]
        assert rows_of(out) == [(1, 10.0), (1, None)]

    def test_the_filled_column_keeps_the_other_branch_type(self, session: Session) -> None:
        """A bare NULL, not a cast one -- the set operation types it from the other side."""
        df = session.table("fx.plain").filter(F.col("id") == 1)
        out = df.select("id").unionByName(df.select("id", "amount"), allowMissingColumns=True)
        assert out.dtypes == [("id", "bigint"), ("amount", "double")]


class TestPrecedence:
    """`a.union(b).intersect(c)` must mean what the Python call order says.

    DuckDB binds `INTERSECT` tighter than `UNION ALL`, so inlining the branches would
    silently evaluate `a UNION ALL (b INTERSECT c)` instead -- a different query that
    raises nothing and answers wrongly. `_as_set_branch` nests to prevent it.
    """

    def test_union_then_intersect_groups_left_to_right(self, session: Session) -> None:
        df = session.table("fx.plain")
        first = df.filter(F.col("id") <= 2)  # {1, 2}
        second = df.filter(F.col("id") == 3)  # {3}
        third = df.filter(F.col("id").isin(2, 3))  # {2, 3}

        # ({1,2} UNION ALL {3}) INTERSECT {2,3} == {2,3}
        # The wrong grouping, {1,2} UNION ALL ({3} INTERSECT {2,3}), gives {1,2,3}.
        assert ids_of(first.union(second).intersect(third)) == [2, 3]

    def test_intersect_then_union_groups_left_to_right(self, session: Session) -> None:
        df = session.table("fx.plain")
        first = df.filter(F.col("id") <= 2)
        second = df.filter(F.col("id").isin(2, 3))
        third = df.filter(F.col("id") == 5)
        assert ids_of(first.intersect(second).union(third)) == [2, 5]


class TestRefusals:
    def test_a_non_dataframe_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineTypeError, match="expects a DataFrame"):
            df.union("fx.plain")  # type: ignore[arg-type]

    def test_a_frame_from_another_session_is_refused(
        self, session: Session, ansi_session: Session
    ) -> None:
        with pytest.raises(EngineValueError, match="different sessions"):
            session.table("fx.plain").union(ansi_session.table("fx.plain"))

    def test_mismatched_column_counts_are_reported_before_the_query_runs(
        self, session: Session
    ) -> None:
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="same number of columns"):
            df.select("id").union(df.select("id", "vendor"))

    def test_the_message_names_the_operation(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="Intersect can only be performed"):
            df.select("id").intersect(df.select("id", "vendor"))
        with pytest.raises(AnalysisException, match="Except can only be performed"):
            df.select("id").subtract(df.select("id", "vendor"))

    def test_allow_missing_columns_must_be_a_bool(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineTypeError, match="allowMissingColumns"):
            df.unionByName(df, allowMissingColumns="yes")  # type: ignore[arg-type]


class TestSqlSurfaceAgrees:
    """P1: the same set operation through `Session.sql()` gives the same answer."""

    def test_union_all(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert rows_of(session.sql("SELECT * FROM fx.plain UNION ALL SELECT * FROM fx.plain")) == (
            rows_of(df.union(df))
        )

    def test_intersect_is_no_longer_refused(self, session: Session) -> None:
        """It was rejected as unimplemented only because the allowlist named `Union`."""
        df = session.table("fx.plain")
        out = session.sql("SELECT * FROM fx.plain INTERSECT SELECT * FROM fx.plain WHERE id <= 3")
        assert rows_of(out) == rows_of(df.intersect(df.filter(F.col("id") <= 3)))

    def test_except_is_no_longer_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = session.sql("SELECT * FROM fx.plain EXCEPT SELECT * FROM fx.plain WHERE id <= 3")
        assert rows_of(out) == rows_of(df.subtract(df.filter(F.col("id") <= 3)))

    def test_intersect_all_and_except_all_parse_and_run(self, session: Session) -> None:
        both = "SELECT id FROM fx.plain {} SELECT id FROM fx.plain WHERE id <= 3"
        assert ids_of(session.sql(both.format("INTERSECT ALL"))) == [1, 2, 3]
        assert ids_of(session.sql(both.format("EXCEPT ALL"))) == [4, 5]


class TestNoteTen:
    """Carry-over note 10: a set operation needing a rename kept none of its pushdown.

    `qualify` renames an unaliased `SUM(amount)` to `_col_0`, which the optimizer must
    put back before the plan can be adopted. It used to decline outright for anything
    that was not a plain SELECT, discarding the whole pipeline -- so predicate *and*
    projection pushdown went with the rename, and the query read every file and every
    column while still answering correctly.
    """

    def _aggregate_union(self, session: Session) -> DataFrame:
        df = session.table("fx.partitioned")
        left = df.filter(F.col("as_at_date") == "2026-08-15").agg(F.sum("amount"))
        right = df.filter(F.col("as_at_date") == "2026-08-17").agg(F.sum("amount"))
        return left.union(right)

    def test_the_optimized_plan_is_adopted(self, session: Session) -> None:
        out = self._aggregate_union(session)
        compiled = session._compile(out._plan, out._sources, out.columns)
        assert compiled.optimized is not None
        assert compiled.optimized.applied, compiled.optimized.note

    def test_the_output_name_survives_the_rename(self, session: Session) -> None:
        """The reason the optimizer declined in the first place: `_col_0` is a wrong answer."""
        assert self._aggregate_union(session).columns == ["sum(amount)"]

    def test_both_predicates_are_pushed(self, session: Session) -> None:
        scan = scan_of(self._aggregate_union(session))
        assert scan.pushed_filter is not None
        assert scan.files_scanned == 2, "the middle partition should be pruned away"

    def test_the_unused_column_is_pruned(self, session: Session) -> None:
        scan = scan_of(self._aggregate_union(session))
        assert "id" not in scan.columns
        assert set(scan.columns) == {"as_at_date", "amount"}

    def test_the_answer_is_still_right(self, session: Session) -> None:
        """Pruning that changed the answer would be the only thing worse than not pruning."""
        # Partition 2026-08-15 holds amounts 0..3, and 2026-08-17 holds 20..23.
        assert rows_of(self._aggregate_union(session)) == [(6.0,), (86.0,)]

"""Phase 5: window functions, and the frame semantics PLAN.md flags as likeliest to drift.

Most of these run against a frame built by `createDataFrame` rather than a fixture
table, and that is the point of having built it: frame semantics turn on **ties, nulls
and gaps** in the ordering column, and a table whose values happen to be distinct cannot
tell a correct implementation from a broken one.

The `ties` frame below is the workhorse -- `x` is `10, 20, 20, 40`, so rows 2 and 3 tie:

    id | x
    ---+----
     1 | 10
     2 | 20
     3 | 20     <- ties with row 2
     4 | 40

**The single most important case here is `TestFrameDefaults`.** With an ordering and no
explicit frame, SQL uses `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which
includes every row tying with the current one -- so a running total *jumps* over a tie
rather than climbing through it. The `ROWS` equivalent climbs. Both are correct; picking
the wrong one silently changes every running aggregate in a query, and the two agree on
any column without duplicates, which is what makes it so easy to ship.

Every assertion is on a value, per the rule Phase 3 established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import EngineTypeError, EngineValueError
from icetl.sql import functions as F
from icetl.sql.window import Window, WindowSpec

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


@pytest.fixture
def ties(session: Session) -> DataFrame:
    """Four rows, with a tie at 20, so RANGE and ROWS frames disagree."""
    return session.createDataFrame(
        [(1, 10.0), (2, 20.0), (3, 20.0), (4, 40.0)], "id bigint, x double"
    )


@pytest.fixture
def gaps(session: Session) -> DataFrame:
    """Nulls in the value column, for the `ignoreNulls` and offset-default cases."""
    return session.createDataFrame(
        [(1, None), (2, "b"), (3, None), (4, "d")], "id bigint, v string"
    )


def by_id(df: DataFrame) -> list[tuple]:
    return [tuple(row) for row in sorted(df.collect(), key=lambda row: row[0])]


class TestRanking:
    """The five ranking functions ignore the frame -- a rank is a property of the order."""

    def test_row_number_never_ties(self, ties: DataFrame) -> None:
        out = ties.select("id", F.row_number().over(Window.orderBy("x")).alias("n"))
        assert by_id(out) == [(1, 1), (2, 2), (3, 3), (4, 4)]

    def test_rank_leaves_a_gap_after_a_tie(self, ties: DataFrame) -> None:
        """1, 2, 2, **4** -- the tie consumes rank 3."""
        out = ties.select("id", F.rank().over(Window.orderBy("x")).alias("r"))
        assert by_id(out) == [(1, 1), (2, 2), (3, 2), (4, 4)]

    def test_dense_rank_leaves_no_gap(self, ties: DataFrame) -> None:
        """1, 2, 2, **3** -- the difference from `rank`, and the reason both exist."""
        out = ties.select("id", F.dense_rank().over(Window.orderBy("x")).alias("r"))
        assert by_id(out) == [(1, 1), (2, 2), (3, 2), (4, 3)]

    def test_percent_rank_spans_zero_to_one(self, ties: DataFrame) -> None:
        out = ties.select("id", F.percent_rank().over(Window.orderBy("x")).alias("p"))
        assert by_id(out) == [
            (1, 0.0),
            (2, pytest.approx(1 / 3)),
            (3, pytest.approx(1 / 3)),
            (4, 1.0),
        ]

    def test_cume_dist_counts_rows_at_or_before(self, ties: DataFrame) -> None:
        """The tied pair both report 0.75: three of four rows are at or before them."""
        out = ties.select("id", F.cume_dist().over(Window.orderBy("x")).alias("c"))
        assert by_id(out) == [(1, 0.25), (2, 0.75), (3, 0.75), (4, 1.0)]

    def test_ntile_splits_into_buckets(self, ties: DataFrame) -> None:
        out = ties.select("id", F.ntile(2).over(Window.orderBy("id")).alias("b"))
        assert by_id(out) == [(1, 1), (2, 1), (3, 2), (4, 2)]

    def test_a_frame_does_not_change_a_rank(self, ties: DataFrame) -> None:
        """Ranking functions ignore the frame, so `rowsBetween` beside one does nothing."""
        plain = Window.orderBy("x")
        framed = Window.orderBy("x").rowsBetween(0, 0)
        assert by_id(ties.select("id", F.rank().over(plain).alias("r"))) == by_id(
            ties.select("id", F.rank().over(framed).alias("r"))
        )


class TestFrameDefaults:
    """The default frame is RANGE, not ROWS, and on a tie they give different answers."""

    def test_the_default_frame_includes_every_tied_row(self, ties: DataFrame) -> None:
        """10, then **50** for both tied rows -- the running total jumps over the tie."""
        out = ties.select("id", F.sum("x").over(Window.orderBy("x")).alias("run"))
        assert by_id(out) == [(1, 10.0), (2, 50.0), (3, 50.0), (4, 90.0)]

    def test_an_explicit_rows_frame_climbs_one_row_at_a_time(self, ties: DataFrame) -> None:
        """The same query, one word different: 10, **30**, 50, 90."""
        window = Window.orderBy("x").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        out = ties.select("id", F.sum("x").over(window).alias("run"))
        assert by_id(out) == [(1, 10.0), (2, 30.0), (3, 50.0), (4, 90.0)]

    def test_no_ordering_means_the_whole_partition(self, ties: DataFrame) -> None:
        out = ties.select("id", F.sum("x").over(Window.partitionBy(F.lit(1))).alias("total"))
        assert by_id(out) == [(1, 90.0), (2, 90.0), (3, 90.0), (4, 90.0)]

    def test_range_between_zero_and_zero_is_the_tied_rows(self, ties: DataFrame) -> None:
        """Which is why it is not the same as `rowsBetween(0, 0)`."""
        window = Window.orderBy("x").rangeBetween(0, 0)
        out = ties.select("id", F.count(F.lit(1)).over(window).alias("n"))
        assert by_id(out) == [(1, 1), (2, 2), (3, 2), (4, 1)]

    def test_rows_between_zero_and_zero_is_this_row_alone(self, ties: DataFrame) -> None:
        window = Window.orderBy("x").rowsBetween(0, 0)
        out = ties.select("id", F.count(F.lit(1)).over(window).alias("n"))
        assert by_id(out) == [(1, 1), (2, 1), (3, 1), (4, 1)]


class TestFrameBounds:
    def test_a_sliding_frame_is_centred_on_the_current_row(self, ties: DataFrame) -> None:
        """Negative is preceding, positive is following -- so (-1, 1) is three rows."""
        window = Window.orderBy("id").rowsBetween(-1, 1)
        out = ties.select("id", F.sum("x").over(window).alias("s"))
        assert by_id(out) == [(1, 30.0), (2, 50.0), (3, 80.0), (4, 60.0)]

    def test_unbounded_both_ways_is_the_whole_partition(self, ties: DataFrame) -> None:
        window = Window.orderBy("id").rowsBetween(
            Window.unboundedPreceding, Window.unboundedFollowing
        )
        out = ties.select("id", F.sum("x").over(window).alias("s"))
        assert by_id(out) == [(1, 90.0), (2, 90.0), (3, 90.0), (4, 90.0)]

    def test_a_trailing_frame_looks_backwards_only(self, ties: DataFrame) -> None:
        window = Window.orderBy("id").rowsBetween(-1, 0)
        out = ties.select("id", F.sum("x").over(window).alias("s"))
        assert by_id(out) == [(1, 10.0), (2, 30.0), (3, 40.0), (4, 60.0)]

    def test_a_value_range_uses_the_ordering_column(self, ties: DataFrame) -> None:
        """Within 10 of this row's `x`, which is a different set from "the row either side"."""
        window = Window.orderBy("x").rangeBetween(-10, 10)
        out = ties.select("id", F.count(F.lit(1)).over(window).alias("n"))
        assert by_id(out) == [(1, 3), (2, 3), (3, 3), (4, 1)]

    def test_a_backwards_frame_is_refused(self) -> None:
        with pytest.raises(EngineValueError, match="must not be after"):
            Window.orderBy("x").rowsBetween(1, -1)

    def test_frame_bounds_must_be_ints(self) -> None:
        with pytest.raises(EngineTypeError, match="start as an int"):
            Window.orderBy("x").rowsBetween("1", 2)  # type: ignore[arg-type]


class TestOffsetFunctions:
    def test_lag_and_lead_look_either_way(self, ties: DataFrame) -> None:
        window = Window.orderBy("id")
        out = ties.select(
            "id",
            F.lag("x").over(window).alias("prev"),
            F.lead("x").over(window).alias("next"),
        )
        assert by_id(out) == [
            (1, None, 20.0),
            (2, 10.0, 20.0),
            (3, 20.0, 40.0),
            (4, 20.0, None),
        ]

    def test_an_offset_reaches_further(self, ties: DataFrame) -> None:
        out = ties.select("id", F.lag("x", 2).over(Window.orderBy("id")).alias("p"))
        assert by_id(out) == [(1, None), (2, None), (3, 10.0), (4, 20.0)]

    def test_a_default_fills_past_the_edge(self, ties: DataFrame) -> None:
        out = ties.select("id", F.lag("x", 1, -1.0).over(Window.orderBy("id")).alias("p"))
        assert by_id(out) == [(1, -1.0), (2, 10.0), (3, 20.0), (4, 20.0)]

    def test_the_offset_must_be_an_int(self) -> None:
        with pytest.raises(EngineTypeError, match="offset as an int"):
            F.lag("x", 1.5)  # type: ignore[arg-type]


class TestValueFunctions:
    def test_first_value_and_the_last_value_surprise(self, ties: DataFrame) -> None:
        """`last_value` over the default frame is the **current row**, not the last row.

        The frame ends at the current row, so its last value is this one. SQL's rule,
        and the reference has it too; the fix is an explicit unbounded frame.
        """
        window = Window.orderBy("id")
        unbounded = window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)
        out = ties.select(
            "id",
            F.last_value("x").over(window).alias("dflt"),
            F.last_value("x").over(unbounded).alias("fixed"),
        )
        assert by_id(out) == [
            (1, 10.0, 40.0),
            (2, 20.0, 40.0),
            (3, 20.0, 40.0),
            (4, 40.0, 40.0),
        ]

    def test_first_value_is_the_start_of_the_frame(self, ties: DataFrame) -> None:
        out = ties.select("id", F.first_value("x").over(Window.orderBy("id")).alias("f"))
        assert by_id(out) == [(1, 10.0), (2, 10.0), (3, 10.0), (4, 10.0)]

    def test_nth_value_is_null_until_the_frame_is_long_enough(self, ties: DataFrame) -> None:
        out = ties.select("id", F.nth_value("x", 2).over(Window.orderBy("id")).alias("n"))
        assert by_id(out) == [(1, None), (2, 20.0), (3, 20.0), (4, 20.0)]

    def test_ignore_nulls_skips_them(self, gaps: DataFrame) -> None:
        """`v` is NULL, 'b', NULL, 'd' -- so the first non-null is 'b' from row 2 on."""
        window = Window.orderBy("id")
        out = gaps.select(
            "id",
            F.first_value("v").over(window).alias("plain"),
            F.first_value("v", True).over(window).alias("skipping"),
        )
        assert by_id(out) == [
            (1, None, None),
            (2, None, "b"),
            (3, None, "b"),
            (4, None, "b"),
        ]

    def test_nth_value_counts_from_one(self) -> None:
        with pytest.raises(EngineValueError, match="counts from 1"):
            F.nth_value("x", 0)


class TestPartitioning:
    def test_each_partition_is_windowed_on_its_own(self, session: Session) -> None:
        df = session.createDataFrame([("a", 1), ("a", 2), ("b", 1)], "g string, x bigint")
        out = df.select(
            "g",
            "x",
            F.row_number().over(Window.partitionBy("g").orderBy("x")).alias("n"),
        )
        assert sorted(tuple(row) for row in out.collect()) == [
            ("a", 1, 1),
            ("a", 2, 2),
            ("b", 1, 1),
        ]

    def test_a_running_total_restarts_per_partition(self, session: Session) -> None:
        df = session.createDataFrame([("a", 1.0), ("a", 2.0), ("b", 5.0)], "g string, x double")
        window = Window.partitionBy("g").orderBy("x")
        out = df.select("g", "x", F.sum("x").over(window).alias("run"))
        assert sorted(tuple(row) for row in out.collect()) == [
            ("a", 1.0, 1.0),
            ("a", 2.0, 3.0),
            ("b", 5.0, 5.0),
        ]

    def test_it_works_over_a_catalog_table(self, session: Session) -> None:
        """`fx.plain`'s vendor is a, b, a, c, NULL -- so `a` is the only partition with two."""
        df = session.table("fx.plain")
        window = Window.partitionBy("vendor").orderBy("id")
        out = df.select("id", F.row_number().over(window).alias("n"))
        assert by_id(out) == [(1, 1), (2, 1), (3, 2), (4, 1), (5, 1)]


class TestNullOrderingInsideWindows:
    """The conformance pass reaches inside an `OVER` clause without knowing windows exist.

    `_fix_null_ordering` walks every `exp.Ordered`, and a window's ordering is made of
    the same nodes as a top-level one -- so `window.py` spells no null placement at all
    and still gets the reference's.
    """

    def test_nulls_sort_first_ascending_inside_a_window(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.select("vendor", F.row_number().over(Window.orderBy("vendor")).alias("n"))
        assert [row[0] for row in sorted(out.collect(), key=lambda r: r[1])] == [
            None,
            "a",
            "a",
            "b",
            "c",
        ]

    def test_the_generated_sql_says_so(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.select(F.row_number().over(Window.orderBy("vendor")).alias("n"))
        sql = session._compile(out._plan, out._sources, out.columns).sql
        assert "NULLS FIRST" in sql

    def test_both_surfaces_agree(self, session: Session) -> None:
        """P1: the same window through `Session.sql()` gives the same answer."""
        df = session.table("fx.plain")
        through_sql = session.sql(
            "SELECT vendor, row_number() OVER (ORDER BY vendor) AS n FROM fx.plain"
        )
        built = df.select("vendor", F.row_number().over(Window.orderBy("vendor")).alias("n"))
        # Sorted by the row number, which is what both surfaces have to agree on.
        assert [tuple(r) for r in sorted(through_sql.collect(), key=lambda r: r[1])] == [
            tuple(r) for r in sorted(built.collect(), key=lambda r: r[1])
        ]


class TestPushdownSurvivesAWindow:
    """A window must not cost the pruning the same query would get without one.

    It reads the whole partition, which is easy to mistake for "reads everything" -- but
    the columns it needs are still only the ones named, and a filter outside the window
    still prunes files.
    """

    def test_a_predicate_still_prunes_files(self, session: Session) -> None:
        df = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        window = Window.partitionBy("as_at_date").orderBy("id")
        out = df.select("id", F.sum("amount").over(window).alias("run"))
        scan = session._compile(out._plan, out._sources, out.columns).scans[0]
        assert scan.files_scanned == 1
        assert scan.pushed_filter is not None

    def test_only_the_columns_the_window_names_are_read(self, session: Session) -> None:
        """One of two hundred, on the table built to make projection pushdown visible."""
        out = session.table("fx.wide").select(
            "id", F.row_number().over(Window.orderBy("id")).alias("n")
        )
        scan = session._compile(out._plan, out._sources, out.columns).scans[0]
        assert scan.total_columns == 200
        assert set(scan.columns) == {"id"}


class TestMonotonicallyIncreasingId:
    def test_it_is_unique_and_increasing(self, session: Session) -> None:
        df = session.table("fx.plain")
        values = [row[0] for row in df.select(F.monotonically_increasing_id().alias("i")).collect()]
        assert len(set(values)) == 5
        assert values == sorted(values)

    def test_it_needs_no_window_of_its_own(self, session: Session) -> None:
        """Already a window function, so it takes no `.over()`."""
        df = session.table("fx.plain")
        assert df.select(F.monotonically_increasing_id().alias("i")).count() == 5


class TestWindowSpecItself:
    def test_a_spec_is_immutable(self) -> None:
        """So one passed to two columns cannot be changed by either of them."""
        base = Window.partitionBy("g")
        derived = base.orderBy("x")
        assert base is not derived
        assert base._order == []
        assert derived._order != []

    def test_a_spec_holds_no_frame(self, session: Session) -> None:
        """It belongs to no DataFrame, so one spec serves several."""
        window = Window.orderBy("id")
        first = session.table("fx.plain").select(F.row_number().over(window).alias("n"))
        second = session.table("fx.partitioned").select(F.row_number().over(window).alias("n"))
        assert first.count() == 5
        assert second.count() == 12

    def test_the_repr_says_what_it_holds(self) -> None:
        assert "partitionBy" in repr(Window.partitionBy("g"))
        assert repr(WindowSpec()) == "WindowSpec(unbounded)"

    def test_over_refuses_anything_that_is_not_a_spec(self) -> None:
        with pytest.raises(EngineTypeError, match="expects a WindowSpec"):
            F.sum("x").over("ORDER BY x")

    def test_partition_by_needs_a_column(self) -> None:
        with pytest.raises(EngineValueError, match="at least one column"):
            Window.partitionBy()

    def test_ntile_needs_a_positive_bucket_count(self) -> None:
        with pytest.raises(EngineValueError, match="n >= 1"):
            F.ntile(0)

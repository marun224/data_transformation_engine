"""`rollup`, `cube`, `pivot`, and the `F.*` functions that make their output readable.

The trap all three share is that a **rolled-up key comes back as NULL**, and `fx.plain`
already has a real NULL in `vendor` (`a, b, a, c, NULL`). So a rollup over it produces
two rows whose vendor is NULL that mean entirely different things -- the group of rows
that had no vendor, and the grand total over every row. Nothing in the values tells them
apart; `F.grouping` is the only thing that does, and `TestRolledUpNullVsRealNull` is
where that is pinned.

Every assertion is on a **value**, per the rule Phase 3 established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import EngineTypeError, EngineValueError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def rows_of(df: DataFrame) -> list[tuple[Any, ...]]:
    """Rows as tuples, sorted with NULL last so the order is stable."""
    return sorted(
        (tuple(row) for row in df.collect()),
        key=lambda t: tuple((v is None, "" if v is None else str(v)) for v in t),
    )


class TestRollup:
    def test_a_rollup_adds_one_grand_total_row(self, session: Session) -> None:
        """Four vendor groups, plus the total: `n + 1` grouping sets for one key."""
        df = session.table("fx.plain")
        plain = df.groupBy("vendor").agg(F.sum("amount").alias("total"))
        rolled = df.rollup("vendor").agg(F.sum("amount").alias("total"))
        assert plain.count() == 4
        assert rolled.count() == 5

    def test_the_extra_row_is_the_total_over_everything(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.rollup("vendor").agg(
            F.sum("amount").alias("total"), F.grouping("vendor").alias("g")
        )
        totals = [row[1] for row in out.collect() if row[2] == 1]
        assert totals == [10.0 + 20.5 + 30.25 + 50.0]

    def test_two_keys_give_three_levels(self, session: Session) -> None:
        """(vendor, id), then (vendor), then the total."""
        df = session.table("fx.plain")
        out = df.rollup("vendor", "id").agg(F.count("*").alias("n"))
        assert out.count() == 5 + 4 + 1

    def test_grouping_by_position_is_preserved(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.rollup("vendor").agg(F.sum("amount").alias("total"))
        assert out.columns == ["vendor", "total"]

    def test_pushdown_survives_a_rollup(self, session: Session) -> None:
        """A grouping set must not cost the projection pruning a plain group keeps."""
        df = session.table("fx.plain")
        out = df.rollup("vendor").agg(F.sum("amount").alias("total"))
        scan = session._compile(out._plan, out._sources, out.columns).scans[0]
        assert set(scan.columns) == {"vendor", "amount"}


class TestCube:
    def test_a_cube_covers_every_combination(self, session: Session) -> None:
        """`2**2` grouping sets: (v, id), (v), (id), and the total."""
        df = session.table("fx.plain")
        out = df.cube("vendor", "id").agg(F.count("*").alias("n"))
        assert out.count() == 5 + 4 + 5 + 1

    def test_a_one_key_cube_is_a_one_key_rollup(self, session: Session) -> None:
        df = session.table("fx.plain")
        cubed = df.cube("vendor").agg(F.sum("amount").alias("total"))
        rolled = df.rollup("vendor").agg(F.sum("amount").alias("total"))
        assert rows_of(cubed) == rows_of(rolled)


class TestRolledUpNullVsRealNull:
    """The one thing about grouping sets that is a wrong answer rather than an error."""

    def test_two_null_vendor_rows_mean_different_things(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.rollup("vendor").agg(
            F.sum("amount").alias("total"), F.grouping("vendor").alias("g")
        )
        null_vendor = [row for row in out.collect() if row[0] is None]
        assert len(null_vendor) == 2, "one real NULL group, one grand total"
        assert sorted(row[2] for row in null_vendor) == [0, 1]

        # The row for vendor IS NULL holds that row's amount; the total holds every row's.
        by_flag = {row[2]: row[1] for row in null_vendor}
        assert by_flag[0] == 50.0
        assert by_flag[1] == 110.75

    def test_grouping_is_zero_for_every_real_key(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.groupBy("vendor").agg(F.grouping("vendor").alias("g"))
        assert {row[1] for row in out.collect()} == {0}

    def test_grouping_id_is_a_bit_vector(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.rollup("vendor", "id").agg(F.grouping_id("vendor", "id").alias("gid"))
        # (v, id) -> 0, (v) -> 1, () -> 3. Never 2: `id` cannot roll up before `vendor`.
        assert sorted({row[2] for row in out.collect()}) == [0, 1, 3]

    def test_grouping_id_needs_its_columns_named(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="needs the columns named"):
            F.grouping_id()


class TestPivot:
    def test_distinct_values_become_columns(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        out = df.groupBy("id").pivot("as_at_date").agg(F.sum("amount"))
        assert out.columns == ["id", "2026-08-15", "2026-08-16", "2026-08-17"]

    def test_a_cell_with_no_rows_is_null(self, session: Session) -> None:
        """Each id belongs to one partition, so two of its three cells are empty."""
        df = session.table("fx.partitioned")
        out = df.groupBy("id").pivot("as_at_date").agg(F.sum("amount"))
        first = next(row for row in out.collect() if row[0] == 0)
        assert tuple(first) == (0, 0.0, None, None)

    def test_explicit_values_fix_the_columns_and_their_order(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        out = (
            df.groupBy("id").pivot("as_at_date", ["2026-08-17", "2026-08-15"]).agg(F.sum("amount"))
        )
        assert out.columns == ["id", "2026-08-17", "2026-08-15"]

    def test_a_value_not_in_the_data_gives_an_empty_column(self, session: Session) -> None:
        df = session.table("fx.partitioned")
        out = df.groupBy("id").pivot("as_at_date", ["1999-01-01"]).agg(F.sum("amount"))
        assert out.columns == ["id", "1999-01-01"]
        assert {row[1] for row in out.collect()} == {None}

    def test_several_aggregates_are_named_value_underscore_aggregate(
        self, session: Session
    ) -> None:
        df = session.table("fx.partitioned")
        out = (
            df.groupBy("id")
            .pivot("as_at_date", ["2026-08-15"])
            .agg(F.sum("amount").alias("total"), F.count("amount").alias("n"))
        )
        assert out.columns == ["id", "2026-08-15_total", "2026-08-15_n"]

    def test_a_null_pivot_key_becomes_its_own_column(self, session: Session) -> None:
        """A NULL key is a group, not an absence -- so it gets a column called `null`."""
        df = session.table("fx.plain")
        out = df.groupBy("id").pivot("vendor").agg(F.count("*"))
        assert out.columns == ["id", "a", "b", "c", "null"]
        # id 5 is the row whose vendor is NULL; its `null` column must be the 1.
        row = next(r for r in out.collect() if r[0] == 5)
        assert tuple(row) == (5, 0, 0, 0, 1)

    def test_count_star_is_restricted_too(self, session: Session) -> None:
        """`count(*)` has no argument to wrap, so it counts a literal instead."""
        df = session.table("fx.plain")
        out = df.groupBy("id").pivot("vendor", ["a"]).agg(F.count("*"))
        assert sorted(row[1] for row in out.collect()) == [0, 0, 0, 1, 1]

    def test_pivot_refuses_a_second_pivot(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="twice"):
            df.groupBy("id").pivot("vendor").pivot("vendor")

    def test_pivot_refuses_to_combine_with_rollup(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="rollup"):
            df.rollup("id").pivot("vendor")

    def test_pivot_wants_a_column_name(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineTypeError, match="column name"):
            df.groupBy("id").pivot(F.col("vendor"))  # type: ignore[arg-type]


class TestBroadcast:
    def test_broadcast_returns_the_frame_unchanged(self, session: Session) -> None:
        """No executors, nothing to ship, nothing to hint at -- it exists to not fail."""
        df = session.table("fx.plain")
        assert F.broadcast(df) is df

    def test_a_broadcast_join_still_answers(self, session: Session) -> None:
        df = session.table("fx.plain")
        joined = df.alias("l").join(F.broadcast(df.alias("r")), on="id")
        assert joined.count() == 5

    def test_broadcast_refuses_a_column(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match="expects a DataFrame"):
            F.broadcast(F.col("id"))


class TestCountAcceptsAColumn:
    """`F.count(Column)` raised until Phase 4: `col == "*"` builds an expression, not a
    bool, so the `or` guarding it reached `Column.__bool__`. Only the string spellings
    were covered, which is why 882 tests passed with it broken."""

    def test_count_of_a_column_object(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.agg(F.count(F.col("amount")).alias("n")).collect()[0][0] == 4

    def test_count_of_a_literal(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.agg(F.count(F.lit(1)).alias("n")).collect()[0][0] == 5

    def test_count_of_a_star_column(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.agg(F.count(F.col("*")).alias("n")).collect()[0][0] == 5

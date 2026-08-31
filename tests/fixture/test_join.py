"""Joins end to end against the local fixture catalog.

`fx.plain` is the left side (`id` 1-5). The right side is a filtered projection of it
(`id` 1-3), so every join type has both matched and unmatched rows to prove itself on:
inner keeps 3, left keeps 5, anti keeps 2.

The case worth reading twice is `on="id"` versus `on=col("a.id") == col("b.id")`. The
first emits **one** `id` column, the second **two** -- that is the reference behaviour,
and it is why the string form is not sugar for the Column form.
"""

from __future__ import annotations

from typing import Any

import pytest

from icetl.errors import EngineTypeError, EngineValueError
from icetl.sql import functions as F
from icetl.sql.dataframe import DataFrame
from icetl.sql.session import Session


def rows(df: DataFrame) -> list[tuple[Any, ...]]:
    """Rows as tuples, sorted with NULL last so the order is stable."""
    return sorted(
        (tuple(row) for row in df.collect()),
        key=lambda t: tuple((v is None, v) for v in t),
    )


def right_of(session: Session) -> DataFrame:
    """`id`, `v2` for ids 1-3 -- a projection and a filter, so it must be nested."""
    return (
        session.table("fx.plain")
        .select("id", F.col("vendor").alias("v2"))
        .filter(F.col("id") < F.lit(4))
    )


class TestKeyColumns:
    def test_a_name_key_collapses_to_one_column(self, session: Session) -> None:
        """`USING` semantics: the key appears once, as the reference has it."""
        out = session.table("fx.plain").join(right_of(session), "id")
        assert out.columns == ["id", "vendor", "amount", "v2"]
        assert rows(out) == [
            (1, "a", 10.0, "a"),
            (2, "b", 20.5, "b"),
            (3, "a", 30.25, "a"),
        ]

    def test_a_list_of_names_behaves_the_same(self, session: Session) -> None:
        one = session.table("fx.plain").join(right_of(session), "id")
        many = session.table("fx.plain").join(right_of(session), ["id"])
        assert many.columns == one.columns
        assert rows(many) == rows(one)

    def test_a_column_condition_keeps_both_keys(self, session: Session) -> None:
        """`ON` semantics. The second `id` is disambiguated rather than duplicated."""
        left = session.table("fx.plain").alias("a")
        right = right_of(session).alias("b")
        out = left.join(right, F.col("a.id") == F.col("b.id"))
        assert out.columns == ["id", "vendor", "amount", "id_1", "v2"]


class TestJoinTypes:
    @pytest.mark.parametrize(
        ("how", "expected_rows"),
        [
            ("inner", 3),
            ("left", 5),
            ("leftouter", 5),
            ("left_outer", 5),
            ("right", 3),
            ("rightouter", 3),
            ("outer", 5),
            ("full", 5),
            ("fullouter", 5),
        ],
    )
    def test_row_counts(self, session: Session, how: str, expected_rows: int) -> None:
        out = session.table("fx.plain").join(right_of(session), "id", how)
        assert len(out.collect()) == expected_rows

    def test_left_join_fills_unmatched_with_null(self, session: Session) -> None:
        out = session.table("fx.plain").join(right_of(session), "id", "left")
        assert dict((r[0], r[3]) for r in rows(out)) == {
            1: "a",
            2: "b",
            3: "a",
            4: None,
            5: None,
        }

    @pytest.mark.parametrize("how", ["semi", "leftsemi", "left_semi"])
    def test_semi_join_keeps_matched_left_rows_only(self, session: Session, how: str) -> None:
        """A semi join filters; it does not widen. No right-hand column appears."""
        out = session.table("fx.plain").join(right_of(session), "id", how)
        assert out.columns == ["id", "vendor", "amount"]
        assert [r[0] for r in rows(out)] == [1, 2, 3]

    @pytest.mark.parametrize("how", ["anti", "leftanti", "left_anti"])
    def test_anti_join_keeps_unmatched_left_rows_only(self, session: Session, how: str) -> None:
        out = session.table("fx.plain").join(right_of(session), "id", how)
        assert out.columns == ["id", "vendor", "amount"]
        assert [r[0] for r in rows(out)] == [4, 5]

    def test_cross_join_is_the_cartesian_product(self, session: Session) -> None:
        out = session.table("fx.plain").crossJoin(right_of(session))
        assert len(out.collect()) == 5 * 3

    def test_a_keyless_join_is_a_cross_join(self, session: Session) -> None:
        out = session.table("fx.plain").join(right_of(session))
        assert len(out.collect()) == 15


class TestSelfJoin:
    def test_aliases_disambiguate_a_self_join(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.alias("x").join(df.alias("y"), F.col("x.id") == F.col("y.id"))
        assert out.columns == ["id", "vendor", "amount", "id_1", "vendor_1", "amount_1"]
        assert len(out.collect()) == 5

    def test_a_self_join_can_project_through_its_aliases(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = (
            df.alias("x")
            .join(df.alias("y"), F.col("x.id") == F.col("y.id"))
            .select(F.col("x.vendor").alias("lv"), F.col("y.amount").alias("ra"))
        )
        assert out.columns == ["lv", "ra"]
        assert rows(out) == [
            ("a", 10.0),
            ("a", 30.25),
            ("b", 20.5),
            ("c", None),
            (None, 50.0),
        ]


class TestComposition:
    def test_join_then_group(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .join(right_of(session), "id")
            .groupBy("vendor")
            .agg(F.count("*").alias("n"))
        )
        assert rows(out) == [("a", 2), ("b", 1)]

    def test_filter_then_join(self, session: Session) -> None:
        out = session.table("fx.plain").filter(F.col("id") > F.lit(1)).join(right_of(session), "id")
        assert [r[0] for r in rows(out)] == [2, 3]

    def test_join_then_filter(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .join(right_of(session), "id")
            .filter(F.col("v2") == F.lit("a"))
        )
        assert [r[0] for r in rows(out)] == [1, 3]

    def test_chained_joins(self, session: Session) -> None:
        base = session.table("fx.plain")
        out = base.join(right_of(session), "id").join(
            base.select("id", F.col("amount").alias("a2")), "id"
        )
        assert len(out.collect()) == 3

    def test_both_surfaces_agree(self, session: Session) -> None:
        """P1: the DataFrame API and SQL must give the same answer."""
        api = session.table("fx.plain").join(right_of(session), "id")
        sql = session.sql(
            "SELECT * FROM fx.plain JOIN "
            "(SELECT id, vendor AS v2 FROM fx.plain WHERE id < 4) AS r USING (id)"
        )
        assert rows(api) == rows(sql)


class TestRejections:
    def test_an_unknown_join_type_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="Unknown join type"):
            session.table("fx.plain").join(right_of(session), "id", "sideways")

    def test_a_keyless_outer_join_is_refused(self, session: Session) -> None:
        """Only inner and cross may be keyless; an outer join without keys is a bug."""
        with pytest.raises(EngineValueError, match="needs an `on`"):
            session.table("fx.plain").join(right_of(session), None, "left")

    def test_joining_a_non_dataframe_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match="DataFrame"):
            session.table("fx.plain").join("fx.plain", "id")  # type: ignore[arg-type]

    def test_a_bad_on_type_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match="column name"):
            session.table("fx.plain").join(right_of(session), 1)  # type: ignore[arg-type]

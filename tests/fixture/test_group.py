"""`groupBy().agg()` end to end against the local fixture catalog.

`fx.plain` is chosen for its nulls: `vendor` is `a, b, a, c, NULL` and `amount` is
`10.0, 20.5, 30.25, NULL, 50.0`. So it exercises the three things aggregation gets
wrong quietly -- NULL as a grouping key in its own right, an all-NULL group summing to
NULL rather than 0, and `count(*)` counting a row that `count(col)` skips.

Every assertion is on a **value**, per the rule Phase 3 established. Group order is not
guaranteed by SQL, so each test sorts before comparing.
"""

from __future__ import annotations

from typing import Any

import pytest

from icetl.errors import EngineTypeError, EngineValueError, UnsupportedFeatureError
from icetl.sql import functions as F
from icetl.sql.dataframe import DataFrame
from icetl.sql.group import GroupedData
from icetl.sql.session import Session


def by_key(df: DataFrame) -> list[tuple[Any, ...]]:
    """Rows as plain tuples, sorted with NULL last so the order is stable."""
    return sorted(
        (tuple(row) for row in df.collect()),
        key=lambda t: (t[0] is None, t[0]),
    )


class TestGroupByShape:
    def test_group_by_returns_grouped_data_and_runs_nothing(self, session: Session) -> None:
        """P3: `groupBy` alone is lazy -- no plan built, no schema resolved."""
        grouped = session.table("fx.plain").groupBy("vendor")
        assert isinstance(grouped, GroupedData)
        assert repr(grouped) == "GroupedData[vendor]"

    def test_keys_come_first_then_aggregates(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg(F.sum("amount").alias("total"))
        assert out.columns == ["vendor", "total"]

    def test_groupby_is_an_alias_of_group_by(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert by_key(df.groupby("vendor").count()) == by_key(df.groupBy("vendor").count())


class TestAggregateValues:
    def test_sum_per_group_and_null_key_is_its_own_group(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg(F.sum("amount").alias("total"))
        assert by_key(out) == [("a", 40.25), ("b", 20.5), ("c", None), (None, 50.0)]

    def test_an_all_null_group_sums_to_null_not_zero(self, session: Session) -> None:
        """Vendor `c` has one row whose amount is NULL. The reference gives NULL."""
        out = session.table("fx.plain").groupBy("vendor").agg(F.sum("amount").alias("total"))
        assert dict(by_key(out))["c"] is None

    def test_count_star_counts_rows_but_count_column_skips_nulls(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .groupBy("vendor")
            .agg(F.count("*").alias("rows"), F.count("amount").alias("amounts"))
        )
        assert by_key(out) == [("a", 2, 2), ("b", 1, 1), ("c", 1, 0), (None, 1, 1)]

    def test_avg_skips_nulls(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg(F.avg("amount").alias("mean"))
        assert dict(by_key(out)) == {"a": 20.125, "b": 20.5, "c": None, None: 50.0}

    def test_min_and_max(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .groupBy("vendor")
            .agg(F.min("amount").alias("lo"), F.max("amount").alias("hi"))
        )
        assert by_key(out) == [
            ("a", 10.0, 30.25),
            ("b", 20.5, 20.5),
            ("c", None, None),
            (None, 50.0, 50.0),
        ]

    def test_several_keys(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor", "id").agg(F.sum("amount").alias("total"))
        assert out.columns == ["vendor", "id", "total"]
        assert len(out.collect()) == 5  # every row is its own group

    def test_grouping_by_an_expression(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .groupBy((F.col("id") % F.lit(2)).alias("parity"))
            .agg(F.count("*").alias("rows"))
        )
        assert by_key(out) == [(0, 2), (1, 3)]


class TestGeneratedNames:
    def test_an_unaliased_aggregate_gets_the_generated_name(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg(F.sum("amount"))
        assert out.columns == ["vendor", "sum(amount)"]

    def test_count_shortcut_is_named_count(self, session: Session) -> None:
        """`gd.count()` is named `count`, not `count(1)`."""
        out = session.table("fx.plain").groupBy("vendor").count()
        assert out.columns == ["vendor", "count"]
        assert by_key(out) == [("a", 2), ("b", 1), ("c", 1), (None, 1)]


class TestInputForms:
    def test_dict_form(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg({"amount": "sum"})
        assert out.columns == ["vendor", "sum(amount)"]
        assert dict(by_key(out))["a"] == 40.25

    def test_keyword_form_names_the_column(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg(total=F.sum("amount"))
        assert out.columns == ["vendor", "total"]

    def test_list_form(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").agg([F.sum("amount").alias("t")])
        assert out.columns == ["vendor", "t"]

    def test_sum_shortcut_over_named_columns(self, session: Session) -> None:
        out = session.table("fx.plain").groupBy("vendor").sum("amount")
        assert dict(by_key(out))["a"] == 40.25


class TestWholeFrameAggregate:
    def test_agg_without_keys_gives_one_row(self, session: Session) -> None:
        out = session.table("fx.plain").agg(F.count("*").alias("rows"))
        assert [tuple(row) for row in out.collect()] == [(5,)]

    def test_group_by_with_no_keys_is_the_same(self, session: Session) -> None:
        df = session.table("fx.plain")
        keyless = [tuple(r) for r in df.groupBy().agg(F.sum("amount").alias("t")).collect()]
        direct = [tuple(r) for r in df.agg(F.sum("amount").alias("t")).collect()]
        assert keyless == direct == [(110.75,)]


class TestComposition:
    def test_filter_before_group(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .filter(F.col("amount") > F.lit(15))
            .groupBy("vendor")
            .agg(F.count("*").alias("rows"))
        )
        assert by_key(out) == [("a", 1), ("b", 1), (None, 1)]

    def test_filter_after_group_applies_to_the_groups(self, session: Session) -> None:
        """A predicate on an aggregate must filter the grouped result, not the rows."""
        out = (
            session.table("fx.plain")
            .groupBy("vendor")
            .agg(F.count("*").alias("rows"))
            .filter(F.col("rows") > F.lit(1))
        )
        assert by_key(out) == [("a", 2)]

    def test_select_after_group(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .groupBy("vendor")
            .agg(F.sum("amount").alias("total"))
            .select("total")
        )
        assert out.columns == ["total"]

    def test_group_then_group_again(self, session: Session) -> None:
        out = (
            session.table("fx.plain")
            .groupBy("vendor")
            .agg(F.count("*").alias("rows"))
            .groupBy("rows")
            .agg(F.count("*").alias("groups"))
        )
        assert by_key(out) == [(1, 3), (2, 1)]

    def test_both_surfaces_agree(self, session: Session) -> None:
        """P1: the DataFrame API and SQL must produce the same answer."""
        api = session.table("fx.plain").groupBy("vendor").agg(F.sum("amount").alias("total"))
        sql = session.sql("SELECT vendor, sum(amount) AS total FROM fx.plain GROUP BY vendor")
        assert by_key(api) == by_key(sql)


class TestRejections:
    def test_a_bare_string_in_agg_is_refused(self, session: Session) -> None:
        """`agg("amount")` names a column, not an aggregate -- the error says so."""
        with pytest.raises(EngineTypeError, match=r"F\.sum"):
            session.table("fx.plain").groupBy("vendor").agg("amount")

    def test_agg_needs_an_expression(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="at least one"):
            session.table("fx.plain").groupBy("vendor").agg()

    def test_positional_and_keyword_together_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="not both"):
            session.table("fx.plain").groupBy("vendor").agg(F.sum("amount"), total=F.sum("amount"))

    def test_an_unknown_dict_function_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="does not support"):
            session.table("fx.plain").groupBy("vendor").agg({"amount": "nope"})

    def test_group_by_rejects_a_non_column(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match="groupBy"):
            session.table("fx.plain").groupBy(1)

    def test_bare_numeric_shortcut_names_the_phase(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="Phase 4"):
            session.table("fx.plain").groupBy("vendor").sum()

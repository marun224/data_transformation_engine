"""`distinct`/`dropDuplicates`, `orderBy`/`sort`, and the two local-data constructors.

These four close Phase 4. `fx.plain` serves the first two as usual -- `vendor` is
`a, b, a, c, NULL`, so it has both a duplicate and a null, which is what makes
de-duplication and null ordering testable on the same table.

**The ordering tests are mostly about nulls**, and about the fact that this module
spells no null placement at all. `_fix_null_ordering` in `sql/conformance.py` is a tree
pass over every `exp.Ordered`, so a sort built by `orderBy` and one parsed from
`Session.sql()` arrive at the same node and come out identical. `TestOrderingIsShared`
is the P1 check that this is true rather than merely intended.

**The clause-precedence tests are about wrong answers.** SQL applies DISTINCT and ORDER
BY before LIMIT, so folding either into a frame that has already taken a LIMIT would
silently change which rows come back -- `df.limit(3).orderBy("id")` must sort those three
rows, not sort the table and take three different ones.

Every assertion is on a value, per the rule Phase 3 established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    EngineValueError,
)
from icetl.sql import functions as F
from icetl.types import Row, StringType, StructField, StructType

if TYPE_CHECKING:
    from icetl.sql.session import Session


class TestDistinct:
    def test_distinct_removes_duplicate_rows(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.union(df).count() == 10
        assert df.union(df).distinct().count() == 5

    def test_null_counts_as_equal_to_null(self, session: Session) -> None:
        """The row whose vendor is NULL must collapse with its own copy, not survive twice."""
        df = session.table("fx.plain").filter(F.col("vendor").isNull())
        assert df.union(df).distinct().count() == 1

    def test_distinct_on_an_already_distinct_frame_is_a_no_op(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.distinct().distinct().count() == 5

    def test_distinct_after_a_limit_applies_to_the_limited_rows(self, session: Session) -> None:
        """SQL de-duplicates before it limits, so this has to nest rather than merge."""
        df = session.table("fx.plain")
        assert df.limit(3).distinct().count() == 3


class TestDropDuplicates:
    def test_no_subset_is_distinct(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.union(df).dropDuplicates().count() == 5

    def test_a_subset_leaves_one_row_per_key(self, session: Session) -> None:
        """`vendor` is a, b, a, c, NULL -- four distinct values including the NULL."""
        df = session.table("fx.plain")
        out = df.dropDuplicates(["vendor"])
        assert out.count() == 4
        vendors = [row[1] for row in out.collect()]
        assert len(vendors) == len(set(vendors))

    def test_the_whole_row_survives_not_just_the_key(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.dropDuplicates(["vendor"])
        assert out.columns == df.columns
        assert all(row[0] is not None for row in out.collect())

    def test_a_string_subset_is_accepted(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.dropDuplicates("vendor").count() == 4

    def test_drop_duplicates_is_an_alias(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.drop_duplicates(["vendor"]).count() == df.dropDuplicates(["vendor"]).count()

    def test_an_unknown_column_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="does not exist"):
            df.dropDuplicates(["nope"])


class TestOrderBy:
    def test_ascending_puts_nulls_first(self, session: Session) -> None:
        """The reference's rule, and nothing in `orderBy` spells it -- the pass does."""
        df = session.table("fx.plain")
        assert [row[1] for row in df.orderBy("vendor").collect()] == [None, "a", "a", "b", "c"]

    def test_descending_puts_nulls_last(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [row[1] for row in df.orderBy("vendor", ascending=False).collect()] == [
            "c",
            "b",
            "a",
            "a",
            None,
        ]

    def test_a_column_can_carry_its_own_direction(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [row[0] for row in df.orderBy(F.col("id").desc()).collect()] == [5, 4, 3, 2, 1]

    def test_ascending_overrides_the_column(self, session: Session) -> None:
        """The reference's precedence: the argument wins where both are given."""
        df = session.table("fx.plain")
        out = df.orderBy(F.col("id").desc(), ascending=True)
        assert [row[0] for row in out.collect()] == [1, 2, 3, 4, 5]

    def test_one_flag_per_column(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.orderBy("vendor", "id", ascending=[True, False])
        assert [(row[1], row[0]) for row in out.collect()] == [
            (None, 5),
            ("a", 3),
            ("a", 1),
            ("b", 2),
            ("c", 4),
        ]

    def test_sort_is_the_same_method(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [r[0] for r in df.sort("id", ascending=False).collect()] == [5, 4, 3, 2, 1]

    def test_sort_within_partitions_sorts_the_frame(self, session: Session) -> None:
        """One partition here, so sorting within it is sorting."""
        df = session.table("fx.plain")
        assert [r[0] for r in df.sortWithinPartitions("id").collect()] == [1, 2, 3, 4, 5]

    def test_ordering_after_a_limit_orders_the_limited_rows(self, session: Session) -> None:
        """`limit(3)` takes ids 1-3; sorting them descending gives 3, 2, 1 -- not 5, 4, 3."""
        df = session.table("fx.plain")
        out = df.limit(3).orderBy("id", ascending=False)
        assert [row[0] for row in out.collect()] == [3, 2, 1]

    def test_a_later_order_supersedes_an_earlier_one(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [r[0] for r in df.orderBy("vendor").orderBy("id").collect()] == [1, 2, 3, 4, 5]

    def test_a_mismatched_flag_count_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="ascending flag"):
            df.orderBy("id", "vendor", ascending=[True])

    def test_no_columns_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="at least one column"):
            df.orderBy()


class TestOrderingIsShared:
    """P1: a sort built by `orderBy` and one parsed from SQL are the same node."""

    def test_both_surfaces_place_nulls_the_same_way(self, session: Session) -> None:
        df = session.table("fx.plain")
        through_sql = session.sql("SELECT vendor FROM fx.plain ORDER BY vendor")
        assert [r[0] for r in through_sql.collect()] == [
            r[1] for r in df.orderBy("vendor").collect()
        ]

    def test_both_surfaces_agree_descending_too(self, session: Session) -> None:
        df = session.table("fx.plain")
        through_sql = session.sql("SELECT vendor FROM fx.plain ORDER BY vendor DESC")
        assert [r[0] for r in through_sql.collect()] == [
            r[1] for r in df.orderBy("vendor", ascending=False).collect()
        ]


class TestRange:
    def test_one_argument_is_the_end(self, session: Session) -> None:
        assert [row[0] for row in session.range(5).collect()] == [0, 1, 2, 3, 4]

    def test_two_arguments_are_start_and_end(self, session: Session) -> None:
        assert [row[0] for row in session.range(2, 5).collect()] == [2, 3, 4]

    def test_a_step_skips(self, session: Session) -> None:
        assert [row[0] for row in session.range(2, 10, 3).collect()] == [2, 5, 8]

    def test_the_column_is_a_bigint_called_id(self, session: Session) -> None:
        assert session.range(3).dtypes == [("id", "bigint")]

    def test_a_range_reads_no_table(self, session: Session) -> None:
        """A table *function*, so nothing asks the catalog to resolve `range`."""
        assert session.range(3)._sources == {}

    def test_an_empty_range_is_allowed(self, session: Session) -> None:
        assert session.range(5, 5).count() == 0

    def test_a_range_composes_like_any_frame(self, session: Session) -> None:
        out = session.range(10).filter(F.col("id") % 2 == 0).orderBy("id", ascending=False)
        assert [row[0] for row in out.collect()] == [8, 6, 4, 2, 0]

    def test_a_zero_step_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="non-zero step"):
            session.range(0, 5, 0)

    def test_arguments_must_be_ints(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match="end as an int"):
            session.range(0, "5")  # type: ignore[arg-type]


class TestCreateDataFrame:
    def test_tuples_without_a_schema_are_named_positionally(self, session: Session) -> None:
        """`_1`, `_2`, ... is what the reference calls unnamed columns."""
        out = session.createDataFrame([(1, "a"), (2, "b")])
        assert out.columns == ["_1", "_2"]
        assert sorted(tuple(row) for row in out.collect()) == [(1, "a"), (2, "b")]

    def test_a_list_of_names_renames_the_columns(self, session: Session) -> None:
        out = session.createDataFrame([(1, "a")], ["id", "name"])
        assert out.columns == ["id", "name"]

    def test_a_ddl_schema_names_and_types(self, session: Session) -> None:
        out = session.createDataFrame([(1, "a")], "id bigint, name string")
        assert out.dtypes == [("id", "bigint"), ("name", "string")]

    def test_a_struct_type_schema_works_too(self, session: Session) -> None:
        schema = StructType([StructField("name", StringType())])
        out = session.createDataFrame([("a",)], schema)
        assert out.columns == ["name"]
        assert out.collect()[0][0] == "a"

    def test_a_typed_schema_casts_rather_than_reinterprets(self, session: Session) -> None:
        """So the cast obeys the conformance rules: a value that will not convert is NULL."""
        out = session.createDataFrame([("abc",), ("7",)], "n bigint")
        assert sorted(row[0] is None for row in out.collect()) == [False, True]
        assert sorted(row[0] for row in out.collect() if row[0] is not None) == [7]

    def test_dicts_take_their_names_from_the_keys(self, session: Session) -> None:
        out = session.createDataFrame([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}])
        assert out.columns == ["a", "b"]
        assert sorted(tuple(row) for row in out.collect()) == [(1, "x"), (2, "y")]

    def test_rows_keep_their_field_names(self, session: Session) -> None:
        out = session.createDataFrame([Row(id=1, name="a"), Row(id=2, name="b")])
        assert out.columns == ["id", "name"]

    def test_an_arrow_table_passes_straight_through(self, session: Session) -> None:
        import pyarrow as pa

        out = session.createDataFrame(pa.table({"id": pa.array([1, 2], pa.int64())}))
        assert out.columns == ["id"]
        assert sorted(row[0] for row in out.collect()) == [1, 2]

    def test_a_pandas_frame_is_accepted(self, session: Session) -> None:
        import pandas as pd

        out = session.createDataFrame(pd.DataFrame({"id": [1, 2]}))
        assert sorted(row[0] for row in out.collect()) == [1, 2]

    def test_the_frame_reads_no_table(self, session: Session) -> None:
        """Materialised into a temp table, so it outlives the Python objects it came from."""
        rows = [(1, "a")]
        out = session.createDataFrame(rows, ["id", "name"])
        rows.clear()
        assert out.count() == 1

    def test_it_composes_with_a_catalog_frame(self, session: Session) -> None:
        local = session.createDataFrame([(1,), (2,)], ["id"])
        joined = local.join(session.table("fx.plain"), on="id")
        assert sorted(row[0] for row in joined.collect()) == [1, 2]

    def test_an_empty_list_needs_a_schema(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="cannot infer a schema"):
            session.createDataFrame([])

    def test_an_empty_list_with_a_schema_is_an_empty_frame(self, session: Session) -> None:
        out = session.createDataFrame([], ["id", "name"])
        assert out.columns == ["id", "name"]
        assert out.count() == 0

    def test_a_wrong_name_count_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="column name"):
            session.createDataFrame([(1, "a")], ["only_one"])

    def test_ragged_rows_are_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="same width"):
            session.createDataFrame([(1, "a"), (2,)])

    def test_an_unusable_schema_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match="schema"):
            session.createDataFrame([(1,)], schema=42)

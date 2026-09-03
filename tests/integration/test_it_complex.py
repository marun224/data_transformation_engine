"""Struct, array and map columns, read back through a real object store.

None of the four real tables in this warehouse has a nested type, so the replica built by
`tests/fixtures/generator.py` supplies them -- but built into the REST catalog and stored
on MinIO, which is the part the local suite cannot reach. Nested parquet is where schema
reconciliation is hardest: a struct field has its own field-id, `union_by_name` has to
descend into it, and the Arrow round trip has more places to lose a NULL.

The replica is two rows, chosen so the interesting cases are present:

    id | person                  | tags     | scores
    ---+-------------------------+----------+----------------
     1 | {name: 'ada', age: 36}  | [x, y]   | {a: 1}
     2 | {name: NULL, age: NULL} | []       | {b: 2, c: 3}

Row 2 is what distinguishes `explode` from `explode_outer` (an **empty** list, not a NULL
one) and what proves a struct's fields can be NULL while the struct itself is not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import column

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestTheTypesSurvivedTheObjectStore:
    def test_the_declared_types_come_back(self, it_session: Session, nested: str) -> None:
        types = dict(it_session.table(nested).dtypes)
        assert types["person"].startswith("struct")
        assert types["tags"].startswith("array")
        assert types["scores"].startswith("map")

    def test_both_rows_read(self, it_session: Session, nested: str) -> None:
        assert it_session.table(nested).count() == 2


class TestStructs:
    def test_a_struct_field_is_reachable_by_name(self, it_session: Session, nested: str) -> None:
        names = column(
            it_session.table(nested).select(F.col("person.name").alias("n")).orderBy("n"), "n"
        )
        assert "ada" in names

    def test_a_null_struct_field_stays_null(self, it_session: Session, nested: str) -> None:
        """The struct is present; its fields are not. Losing that distinction is easy."""
        rows = (
            it_session.table(nested)
            .select(F.col("id"), F.col("person.name").alias("n"), F.col("person.age").alias("a"))
            .orderBy("id")
            .collect()
        )
        assert rows[0]["n"] == "ada" and rows[0]["a"] == 36
        assert rows[1]["n"] is None and rows[1]["a"] is None

    def test_getfield_is_the_same_as_dotted_access(self, it_session: Session, nested: str) -> None:
        dotted = column(it_session.table(nested).select(F.col("person.name").alias("v")), "v")
        by_field = column(
            it_session.table(nested).select(F.col("person").getField("name").alias("v")), "v"
        )
        assert dotted == by_field

    def test_a_struct_can_be_built_and_read_back(self, it_session: Session, nested: str) -> None:
        rows = (
            it_session.table(nested)
            .select(F.struct(F.col("id").alias("k"), F.lit("x").alias("v")).alias("s"))
            .collect()
        )
        assert rows
        assert rows[0]["s"]["v"] == "x"


class TestArrays:
    def test_the_element_values_come_back(self, it_session: Session, nested: str) -> None:
        rows = it_session.table(nested).select("id", "tags").orderBy("id").collect()
        assert rows[0]["tags"] == ["x", "y"]
        assert rows[1]["tags"] == []

    def test_size_counts_the_elements(self, it_session: Session, nested: str) -> None:
        sizes = column(
            it_session.table(nested).select(F.size(F.col("tags")).alias("n")).orderBy("n"), "n"
        )
        assert sorted(sizes) == [0, 2]

    def test_array_contains_finds_a_real_element(self, it_session: Session, nested: str) -> None:
        matched = it_session.table(nested).filter(F.array_contains(F.col("tags"), "x")).count()
        assert matched == 1

    def test_explode_drops_the_empty_row(self, it_session: Session, nested: str) -> None:
        """Two elements from row 1, none from row 2."""
        assert it_session.table(nested).select(F.explode(F.col("tags")).alias("t")).count() == 2

    def test_explode_outer_keeps_the_empty_row(self, it_session: Session, nested: str) -> None:
        """The distinction the empty list exists for -- one extra row, holding NULL."""
        rows = it_session.table(nested).select(F.explode_outer(F.col("tags")).alias("t")).collect()
        assert len(rows) == 3
        assert sum(1 for row in rows if row["t"] is None) == 1

    def test_exploding_alongside_another_column_repeats_it(
        self, it_session: Session, nested: str
    ) -> None:
        rows = (
            it_session.table(nested)
            .select(F.col("id"), F.explode(F.col("tags")).alias("t"))
            .collect()
        )
        assert len(rows) == 2
        assert {row["id"] for row in rows} == {1}

    def test_a_higher_order_function_runs_over_the_elements(
        self, it_session: Session, nested: str
    ) -> None:
        rows = (
            it_session.table(nested)
            .select(F.col("id"), F.transform(F.col("tags"), lambda t: F.upper(t)).alias("t"))
            .orderBy("id")
            .collect()
        )
        assert ["X", "Y"] in [row["t"] for row in rows]

    def test_filter_over_an_array_keeps_the_matching_elements(
        self, it_session: Session, nested: str
    ) -> None:
        rows = (
            it_session.table(nested)
            .select(F.filter(F.col("tags"), lambda t: t == F.lit("x")).alias("t"))
            .collect()
        )
        assert ["x"] in [row["t"] for row in rows]

    def test_exists_over_an_empty_list_is_false_not_null(
        self, it_session: Session, nested: str
    ) -> None:
        """FINDINGS 1.7: `exists`/`forall` over an empty list. The empty row is row 2."""
        values = column(
            it_session.table(nested)
            .orderBy("id")
            .select(F.exists(F.col("tags"), lambda t: t == F.lit("x")).alias("v")),
            "v",
        )
        assert values == [True, False]

    def test_forall_over_an_empty_list_is_true(self, it_session: Session, nested: str) -> None:
        """Vacuous truth: every element of nothing satisfies anything."""
        values = column(
            it_session.table(nested)
            .orderBy("id")
            .select(F.forall(F.col("tags"), lambda t: t.isNotNull()).alias("v")),
            "v",
        )
        assert values == [True, True]


class TestMaps:
    def test_the_entries_come_back(self, it_session: Session, nested: str) -> None:
        rows = it_session.table(nested).select("id", "scores").orderBy("id").collect()
        assert rows[0]["scores"] == {"a": 1}
        assert rows[1]["scores"] == {"b": 2, "c": 3}

    def test_map_keys_and_values_line_up(self, it_session: Session, nested: str) -> None:
        rows = (
            it_session.table(nested)
            .select(
                F.map_keys(F.col("scores")).alias("k"), F.map_values(F.col("scores")).alias("v")
            )
            .collect()
        )
        assert rows
        for row in rows:
            assert len(row["k"]) == len(row["v"])

    def test_a_key_is_reachable(self, it_session: Session, nested: str) -> None:
        values = column(
            it_session.table(nested).select(F.col("scores").getItem("a").alias("v")), "v"
        )
        assert 1 in values

    def test_a_missing_key_is_null(self, it_session: Session, nested: str) -> None:
        values = column(
            it_session.table(nested).select(F.col("scores").getItem("zzz").alias("v")), "v"
        )
        assert set(values) == {None}

    def test_map_size_counts_the_entries(self, it_session: Session, nested: str) -> None:
        """`size(map_keys(m))` is the spelling that works -- see the test below for why."""
        sizes = sorted(
            column(
                it_session.table(nested).select(F.size(F.map_keys(F.col("scores"))).alias("n")),
                "n",
            )
        )
        assert sizes == [1, 2]

    @pytest.mark.parametrize("name", ["size", "cardinality"])
    def test_size_does_not_accept_a_map(self, it_session: Session, nested: str, name: str) -> None:
        """FINDINGS 2.13 -- a conformance gap, pinned.

        The reference's `size()` takes an array **or a map**; both spellings here emit
        `ARRAY_LENGTH`, which DuckDB rejects for a MAP. `map_size` is not exported
        either, so `size(map_keys(m))` is the only route today.

        A loud failure rather than a wrong answer, and it fails when the gap is closed.
        """
        from icetl.errors import AnalysisException

        function = getattr(F, name)
        with pytest.raises(AnalysisException, match="array_length"):
            it_session.table(nested).select(function(F.col("scores")).alias("n")).collect()


class TestPushdownWithNestedColumns:
    """A nested column must not break the projection accounting."""

    def test_selecting_one_nested_column_prunes_the_others(
        self, it_session: Session, nested: str
    ) -> None:
        from tests.integration.helpers import scan_of

        scan = scan_of(it_session.table(nested).select("tags"))
        assert set(scan.columns) == {"tags"}
        assert scan.total_columns > 1

    def test_a_generator_is_not_dropped_by_the_optimizer(
        self, it_session: Session, nested: str
    ) -> None:
        """FINDINGS 1.13: the optimizer once dropped a generator, so `count()` counted
        the wrong table. The two questions have to agree."""
        frame = it_session.table(nested).select(F.explode(F.col("tags")).alias("t"))
        assert frame.count() == len(frame.collect()) == 2

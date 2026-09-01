"""Phase 6: struct, array and map columns end to end.

`fx.nested` is the table for all of it -- two rows, chosen so that every collection has
an **empty** counterpart:

    id | person                    | tags       | scores
    ---+---------------------------+------------+------------------
     1 | {name: 'ada', age: 36}    | ['x', 'y'] | {'a': 1}
     2 | {name: NULL, age: NULL}   | []         | {'b': 2, 'c': 3}

Row 2's empty `tags` is what makes `explode` and `explode_outer` distinguishable: the
plain form drops the row entirely and the outer form keeps it with a NULL. On a table
where every list had elements the two would agree, and the tests would prove nothing.

**The generators are the phase's real work.** `explode` yields one column for a list and
**two** for a map, and `inline` yields one per struct field -- so how many columns come
out depends on the *type* of what is exploded, which is why expansion happens in
`select` rather than in `F`. `TestExplodeShape` pins each shape.

Every assertion is on a value, per the rule Phase 3 established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import AnalysisException, EngineTypeError, EngineValueError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def rows(df: DataFrame) -> list[tuple[Any, ...]]:
    return [tuple(row) for row in df.collect()]


@pytest.fixture
def nested(session: Session) -> DataFrame:
    return session.table("fx.nested")


@pytest.fixture
def structs(session: Session) -> DataFrame:
    """An array of structs, which `fx.nested` has no column of -- for `inline`."""
    return session.createDataFrame(
        [(1, [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]), (2, [])],
        "id bigint, items array<struct<a: bigint, b: string>>",
    )


class TestReadingComplexTypes:
    def test_the_schema_reports_the_nested_shapes(self, nested: DataFrame) -> None:
        assert nested.dtypes == [
            ("id", "bigint"),
            ("person", "struct<name:string,age:bigint>"),
            ("tags", "array<string>"),
            ("scores", "map<string,bigint>"),
        ]

    def test_values_come_back_as_python_shapes(self, nested: DataFrame) -> None:
        """A struct is a Row, a list is a list, a map is a dict."""
        first = nested.filter(F.col("id") == 1).collect()[0]
        assert first[1].name == "ada"
        assert first[2] == ["x", "y"]
        assert first[3] == {"a": 1}

    def test_a_nested_field_is_reachable_by_path(self, nested: DataFrame) -> None:
        out = nested.select(F.col("person.name").alias("n"))
        assert rows(out) == [("ada",), (None,)]

    def test_get_field_and_get_item(self, nested: DataFrame) -> None:
        out = nested.select(
            F.col("person").getField("age").alias("age"),
            F.col("tags").getItem(0).alias("first"),
            F.col("scores").getItem("a").alias("a"),
        )
        assert rows(out) == [(36, "x", 1), (None, None, None)]


class TestPrintSchema:
    """`printSchema(level=...)` was the last thing still raising `phase="Phase 6"`.

    It only earns its keep once a schema has nesting to truncate, which is why it waited
    for this phase rather than landing with `printSchema` itself.
    """

    def test_the_full_tree_reaches_the_leaves(self, nested: DataFrame) -> None:
        tree = nested.schema.treeString()
        assert "-- person: struct (nullable = true)" in tree
        assert "-- name: string (nullable = true)" in tree

    def test_a_level_stops_the_walk(self, nested: DataFrame) -> None:
        tree = nested.schema.treeString(1)
        assert "-- person: struct (nullable = true)" in tree
        assert "name" not in tree

    def test_level_zero_is_refused(self, nested: DataFrame) -> None:
        with pytest.raises(EngineValueError, match="level >= 1"):
            nested.schema.treeString(0)

    def test_print_schema_takes_the_level(self, nested: DataFrame, capsys: Any) -> None:
        nested.printSchema(1)
        assert "name" not in capsys.readouterr().out


class TestExplodeShape:
    def test_a_list_explodes_to_one_column_called_col(self, nested: DataFrame) -> None:
        out = nested.select("id", F.explode("tags"))
        assert out.columns == ["id", "col"]
        assert rows(out) == [(1, "x"), (1, "y")]

    def test_an_empty_list_drops_the_row(self, nested: DataFrame) -> None:
        """Row 2's `tags` is empty, so `explode` loses it entirely."""
        assert nested.select("id", F.explode("tags")).count() == 2

    def test_explode_outer_keeps_it_with_a_null(self, nested: DataFrame) -> None:
        out = nested.select("id", F.explode_outer("tags").alias("tag"))
        assert rows(out) == [(1, "x"), (1, "y"), (2, None)]

    def test_a_map_explodes_to_two_columns(self, nested: DataFrame) -> None:
        """Key and value, which is why a generator cannot be a plain expression."""
        out = nested.select("id", F.explode("scores"))
        assert out.columns == ["id", "key", "value"]
        assert rows(out) == [(1, "a", 1), (2, "b", 2), (2, "c", 3)]

    def test_posexplode_adds_a_zero_based_position(self, nested: DataFrame) -> None:
        out = nested.select("id", F.posexplode("tags"))
        assert out.columns == ["id", "pos", "col"]
        assert rows(out) == [(1, 0, "x"), (1, 1, "y")]

    def test_posexplode_over_a_map_gives_three_columns(self, nested: DataFrame) -> None:
        out = nested.select("id", F.posexplode("scores"))
        assert out.columns == ["id", "pos", "key", "value"]
        assert rows(out) == [(1, 0, "a", 1), (2, 0, "b", 2), (2, 1, "c", 3)]

    def test_inline_gives_one_column_per_struct_field(self, structs: DataFrame) -> None:
        out = structs.select("id", F.inline("items"))
        assert out.columns == ["id", "a", "b"]
        assert rows(out) == [(1, 1, "x"), (1, 2, "y")]

    def test_inline_outer_keeps_the_empty_row(self, structs: DataFrame) -> None:
        out = structs.select("id", F.inline_outer("items"))
        assert rows(out) == [(1, 1, "x"), (1, 2, "y"), (2, None, None)]

    def test_the_positions_pair_with_their_elements(self, nested: DataFrame) -> None:
        """Two `unnest`es of the same list are correlated by DuckDB, not multiplied."""
        out = nested.select(F.posexplode("tags"))
        assert out.count() == 2, "a cross product would give four rows"


class TestExplodeNaming:
    def test_a_single_column_generator_takes_one_alias(self, nested: DataFrame) -> None:
        out = nested.select(F.explode("tags").alias("tag"))
        assert out.columns == ["tag"]

    def test_a_two_column_generator_takes_two(self, nested: DataFrame) -> None:
        out = nested.select(F.explode("scores").alias("k", "v"))
        assert out.columns == ["k", "v"]

    def test_the_wrong_number_of_aliases_is_refused(self, nested: DataFrame) -> None:
        with pytest.raises(EngineValueError, match="2 name"):
            nested.select(F.explode("scores").alias("only_one"))

    def test_exploding_a_scalar_is_refused(self, nested: DataFrame) -> None:
        with pytest.raises(AnalysisException, match="array or map"):
            nested.select(F.explode("id"))

    def test_inline_needs_an_array_of_structs(self, nested: DataFrame) -> None:
        with pytest.raises(AnalysisException, match="array of structs"):
            nested.select(F.inline("tags"))

    def test_two_generators_in_one_select_are_refused(self, nested: DataFrame) -> None:
        """They would have to agree on how many rows to produce, and cannot."""
        with pytest.raises(EngineValueError, match="at most one generator"):
            nested.select(F.explode("tags"), F.explode("scores"))


class TestStructFields:
    def test_with_field_adds_one(self, nested: DataFrame) -> None:
        out = nested.select(F.col("person").withField("z", F.lit(1)).alias("p"))
        assert rows(out)[0][0] == {"name": "ada", "age": 36, "z": 1}

    def test_with_field_replaces_an_existing_one(self, nested: DataFrame) -> None:
        """`struct_insert` refuses a duplicate name, so the struct is rebuilt instead."""
        out = nested.select(F.col("person").withField("name", F.lit("bob")).alias("p"))
        assert rows(out)[0][0]["name"] == "bob"

    def test_drop_fields_rebuilds_without_them(self, nested: DataFrame) -> None:
        out = nested.select(F.col("person").dropFields("age").alias("p"))
        assert out.dtypes == [("p", "struct<name:string>")]
        assert rows(out) == [({"name": "ada"},), ({"name": None},)]

    def test_dropping_an_absent_field_is_not_an_error(self, nested: DataFrame) -> None:
        out = nested.select(F.col("person").dropFields("nope").alias("p"))
        assert out.columns == ["p"]
        assert rows(out)[0][0] == {"name": "ada", "age": 36}

    def test_dropping_every_field_is_refused(self, nested: DataFrame) -> None:
        """An empty struct has no SQL spelling, so this cannot be built."""
        with pytest.raises(AnalysisException, match="empty struct"):
            nested.select(F.col("person").dropFields("name", "age"))

    def test_struct_builds_one_from_columns(self, nested: DataFrame) -> None:
        out = nested.select(F.struct("id", "tags").alias("s"))
        assert rows(out)[0][0]["id"] == 1

    def test_struct_takes_an_aliased_column(self, nested: DataFrame) -> None:
        """The alias names the field and is then dropped from the value.

        Keeping it emits `{'a': 1 AS "a"}`, which is not SQL -- and nothing caught it
        until Phase 6, because every earlier caller passed plain column names.
        """
        out = nested.select(F.struct(F.lit(1).alias("a")).alias("s"))
        assert rows(out)[0][0] == {"a": 1}


class TestHigherOrderFunctions:
    def test_transform_maps_every_element(self, nested: DataFrame) -> None:
        out = nested.select(F.transform("tags", lambda x: F.upper(x)).alias("t"))
        assert rows(out) == [(["X", "Y"],), ([],)]

    def test_transform_can_see_the_index(self, nested: DataFrame) -> None:
        out = nested.select(F.transform("tags", lambda x, i: i).alias("t"))
        assert rows(out) == [([1, 2],), ([],)]

    def test_filter_keeps_the_matching_elements(self, nested: DataFrame) -> None:
        out = nested.select(F.filter("tags", lambda x: x == F.lit("x")).alias("t"))
        assert rows(out) == [(["x"],), ([],)]

    def test_aggregate_folds_from_an_initial_value(self, nested: DataFrame) -> None:
        out = nested.select(
            F.aggregate("tags", F.lit(""), lambda acc, x: F.concat(acc, x)).alias("g")
        )
        assert rows(out) == [("xy",), ("",)]

    def test_aggregate_over_an_empty_list_is_the_initial_value(self, nested: DataFrame) -> None:
        """Which is why the zero is prepended rather than left to `list_reduce`."""
        out = nested.select(F.aggregate("tags", F.lit("z"), lambda a, x: F.concat(a, x)).alias("g"))
        assert rows(out)[1][0] == "z"

    def test_aggregate_can_finish(self, nested: DataFrame) -> None:
        out = nested.select(
            F.aggregate("tags", F.lit(""), lambda a, x: F.concat(a, x), lambda a: F.upper(a)).alias(
                "g"
            )
        )
        assert rows(out) == [("XY",), ("",)]

    def test_zip_with_pairs_two_lists(self, nested: DataFrame) -> None:
        out = nested.select(F.zip_with("tags", "tags", lambda a, b: F.concat(a, b)).alias("z"))
        assert rows(out) == [(["xx", "yy"],), ([],)]

    def test_a_lambda_must_return_a_column(self, nested: DataFrame) -> None:
        """It is called once to build an expression, not once per row."""
        with pytest.raises(EngineTypeError, match="must return a Column"):
            nested.select(F.transform("tags", lambda x: 1))

    def test_a_lambda_of_the_wrong_arity_is_refused(self, nested: DataFrame) -> None:
        with pytest.raises(EngineValueError, match="1 to 2"):
            nested.select(F.transform("tags", lambda a, b, c: a))


class TestExistsAndForall:
    """Three-valued, as the reference specifies -- and the empty list is the giveaway."""

    def test_exists_is_true_when_one_matches(self, nested: DataFrame) -> None:
        out = nested.select(F.exists("tags", lambda x: x == F.lit("x")).alias("e"))
        assert rows(out)[0][0] is True

    def test_exists_over_an_empty_list_is_false_not_null(self, nested: DataFrame) -> None:
        """`list_bool_or([])` is NULL; the reference says false, so the case is spelled out."""
        out = nested.select(F.exists("tags", lambda x: x == F.lit("x")).alias("e"))
        assert rows(out)[1][0] is False

    def test_forall_over_an_empty_list_is_true(self, nested: DataFrame) -> None:
        """Vacuously true, and again not what `list_bool_and` alone answers."""
        out = nested.select(F.forall("tags", lambda x: x == F.lit("nope")).alias("e"))
        assert rows(out)[1][0] is True

    def test_forall_is_false_when_one_fails(self, nested: DataFrame) -> None:
        out = nested.select(F.forall("tags", lambda x: x == F.lit("x")).alias("e"))
        assert rows(out)[0][0] is False


class TestMaps:
    def test_create_map_takes_alternating_keys_and_values(self, nested: DataFrame) -> None:
        out = nested.select(F.create_map(F.lit("a"), F.lit(1)).alias("m"))
        assert rows(out)[0][0] == {"a": 1}

    def test_an_odd_number_of_arguments_is_refused(self, nested: DataFrame) -> None:
        with pytest.raises(EngineValueError, match="even number"):
            F.create_map(F.lit("a"))

    def test_map_from_arrays_pairs_positionally(self, nested: DataFrame) -> None:
        out = nested.select(F.map_from_arrays(F.array(F.lit("k")), F.array(F.lit(9))).alias("m"))
        assert rows(out)[0][0] == {"k": 9}

    def test_entries_round_trip(self, nested: DataFrame) -> None:
        out = nested.select(F.map_from_entries(F.map_entries("scores")).alias("m"))
        assert rows(out) == [({"a": 1},), ({"b": 2, "c": 3},)]

    def test_map_concat_merges(self, nested: DataFrame) -> None:
        other = F.create_map(F.lit("z"), F.lit(9).cast("bigint"))
        out = nested.select(F.map_concat("scores", other).alias("m"))
        assert rows(out)[0][0] == {"a": 1, "z": 9}

    def test_map_concat_refuses_mismatched_value_types(self, nested: DataFrame) -> None:
        """The reference widens `int` to meet `bigint`; DuckDB refuses. Loudly, at least."""
        mixed = nested.select(F.map_concat("scores", F.create_map(F.lit("z"), F.lit(9))))
        with pytest.raises(AnalysisException, match="type of map differs"):
            mixed.columns  # noqa: B018 - resolving the schema is what raises

    def test_transform_values_rewrites_the_values(self, nested: DataFrame) -> None:
        out = nested.select(F.transform_values("scores", lambda k, v: v * F.lit(2)).alias("m"))
        assert rows(out) == [({"a": 2},), ({"b": 4, "c": 6},)]

    def test_transform_keys_rewrites_the_keys(self, nested: DataFrame) -> None:
        out = nested.select(F.transform_keys("scores", lambda k, v: F.upper(k)).alias("m"))
        assert rows(out) == [({"A": 1},), ({"B": 2, "C": 3},)]

    def test_map_filter_keeps_matching_entries(self, nested: DataFrame) -> None:
        out = nested.select(F.map_filter("scores", lambda k, v: v > F.lit(1)).alias("m"))
        assert rows(out) == [({},), ({"b": 2, "c": 3},)]

    def test_arrays_zip_names_its_fields(self, nested: DataFrame) -> None:
        """Unnamed structs cannot survive the trip back into Arrow, so they are named."""
        out = nested.select(F.arrays_zip("tags", "tags").alias("z"))
        assert rows(out)[0][0] == [{"0": "x", "1": "x"}, {"0": "y", "1": "y"}]


class TestJson:
    @pytest.fixture
    def doc(self, session: Session) -> DataFrame:
        return session.createDataFrame([('{"a": 1, "b": {"c": "z"}, "d": [1, 2]}',)], ["j"])

    def test_get_json_object_reads_a_path(self, doc: DataFrame) -> None:
        assert rows(doc.select(F.get_json_object("j", "$.b.c").alias("v"))) == [("z",)]

    def test_an_object_comes_back_as_its_json_text(self, doc: DataFrame) -> None:
        assert rows(doc.select(F.get_json_object("j", "$.b").alias("v"))) == [('{"c":"z"}',)]

    def test_json_tuple_gives_a_column_per_field(self, doc: DataFrame) -> None:
        out = doc.select(F.json_tuple("j", "a", "d"))
        assert out.columns == ["c0", "c1"]
        assert rows(out) == [("1", "[1,2]")]

    def test_json_tuple_adds_no_rows(self, doc: DataFrame) -> None:
        assert doc.select(F.json_tuple("j", "a", "d")).count() == 1

    def test_from_json_parses_into_a_struct(self, doc: DataFrame) -> None:
        out = doc.select(F.from_json("j", "a bigint").alias("s"))
        assert rows(out) == [({"a": 1},)]

    def test_from_json_ignores_keys_the_schema_omits(self, doc: DataFrame) -> None:
        """Which is why it is `from_json` and not a `CAST` -- a cast refuses them."""
        out = doc.select(F.from_json("j", "a bigint").alias("s"))
        assert out.dtypes == [("s", "struct<a:bigint>")]

    def test_from_json_handles_nested_shapes(self, doc: DataFrame) -> None:
        out = doc.select(F.from_json("j", "a bigint, d array<bigint>").alias("s"))
        assert rows(out) == [({"a": 1, "d": [1, 2]},)]

    def test_to_json_renders_a_struct(self, doc: DataFrame) -> None:
        out = doc.select(F.to_json(F.struct(F.lit(1).alias("a"))).alias("t"))
        assert rows(out) == [('{"a":1}',)]

    def test_from_json_needs_a_struct_schema(self, doc: DataFrame) -> None:
        with pytest.raises(EngineTypeError, match="DDL string or a StructType"):
            doc.select(F.from_json("j", 42))

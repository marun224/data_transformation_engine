"""The reference engine type hierarchy, its two naming schemes, printSchema, and `Row`."""

from __future__ import annotations

import pickle

import pytest

from icetl.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    NullType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)


class TestNaming:
    """`typeName` and `simpleString` differ, and both matter to different callers."""

    @pytest.mark.parametrize(
        ("data_type", "type_name", "simple"),
        [
            (NullType(), "void", "void"),
            (BooleanType(), "boolean", "boolean"),
            (ByteType(), "byte", "tinyint"),
            (ShortType(), "short", "smallint"),
            (IntegerType(), "integer", "int"),
            (LongType(), "long", "bigint"),
            (FloatType(), "float", "float"),
            (DoubleType(), "double", "double"),
            (StringType(), "string", "string"),
            (BinaryType(), "binary", "binary"),
            (DateType(), "date", "date"),
            (TimestampType(), "timestamp", "timestamp"),
            (TimestampNTZType(), "timestamp_ntz", "timestamp_ntz"),
        ],
    )
    def test_scalar_names(self, data_type: DataType, type_name: str, simple: str) -> None:
        assert data_type.typeName() == type_name
        assert data_type.simpleString() == simple

    def test_decimal_carries_precision_everywhere_but_type_name(self) -> None:
        decimal = DecimalType(10, 2)
        assert decimal.typeName() == "decimal"
        assert decimal.simpleString() == "decimal(10,2)"
        # printSchema shows the precision, which is why `treeName` is separate.
        assert decimal.treeName() == "decimal(10,2)"

    def test_nested_simple_strings(self) -> None:
        assert ArrayType(LongType()).simpleString() == "array<bigint>"
        assert MapType(StringType(), LongType()).simpleString() == "map<string,bigint>"
        assert (
            StructType(
                [StructField("a", LongType()), StructField("b", StringType())]
            ).simpleString()
            == "struct<a:bigint,b:string>"
        )


class TestValueSemantics:
    def test_equal_instances_compare_and_hash_equal(self) -> None:
        assert LongType() == LongType()
        assert hash(LongType()) == hash(LongType())
        assert len({LongType(), LongType(), StringType()}) == 2

    def test_subclasses_do_not_compare_equal_to_their_base(self) -> None:
        # IntegerType and LongType are both IntegralType; they must not be confused.
        assert LongType() != IntegerType()

    def test_decimal_equality_includes_precision_and_scale(self) -> None:
        assert DecimalType(10, 2) == DecimalType(10, 2)
        assert DecimalType(10, 2) != DecimalType(10, 3)

    @pytest.mark.parametrize(("precision", "scale"), [(0, 0), (39, 0), (5, 6), (5, -1)])
    def test_invalid_decimals_are_rejected(self, precision: int, scale: int) -> None:
        with pytest.raises(ValueError):
            DecimalType(precision, scale)

    def test_repr_round_trips_through_eval(self) -> None:
        original = StructType([StructField("a", ArrayType(LongType(), False), False)])
        assert eval(repr(original)) == original


class TestStructType:
    def test_add_chains_and_updates_names(self) -> None:
        schema = StructType().add("a", LongType()).add(StructField("b", StringType(), False))
        assert schema.fieldNames() == ["a", "b"]
        assert schema.names == ["a", "b"]
        assert len(schema) == 2

    def test_lookup_by_name_index_and_slice(self) -> None:
        schema = StructType([StructField("a", LongType()), StructField("b", StringType())])
        by_name, by_index = schema["a"], schema[1]
        assert isinstance(by_name, StructField) and isinstance(by_index, StructField)
        assert by_name.dataType == LongType()
        assert by_index.name == "b"
        assert isinstance(schema[0:1], StructType)
        with pytest.raises(KeyError):
            schema["missing"]

    def test_add_by_name_requires_a_type(self) -> None:
        with pytest.raises(ValueError, match="data type is required"):
            StructType().add("a")


class TestTreeString:
    """`printSchema()` output. The gutter and per-level flag names follow the reference."""

    def test_flat_schema(self) -> None:
        schema = StructType(
            [StructField("id", LongType()), StructField("vendor", StringType(), False)]
        )
        assert schema.treeString() == (
            "root\n |-- id: long (nullable = true)\n |-- vendor: string (nullable = false)\n"
        )

    def test_struct_array_and_map_use_their_own_flag_names(self) -> None:
        schema = StructType(
            [
                StructField("person", StructType([StructField("name", StringType())])),
                StructField("tags", ArrayType(StringType(), True)),
                StructField("scores", MapType(StringType(), LongType(), True)),
            ]
        )
        assert schema.treeString() == (
            "root\n"
            " |-- person: struct (nullable = true)\n"
            " |    |-- name: string (nullable = true)\n"
            " |-- tags: array (nullable = true)\n"
            " |    |-- element: string (containsNull = true)\n"
            " |-- scores: map (nullable = true)\n"
            " |    |-- key: string\n"
            " |    |-- value: long (valueContainsNull = true)\n"
        )

    def test_nesting_indents_one_level_per_depth(self) -> None:
        schema = StructType(
            [StructField("a", ArrayType(StructType([StructField("b", LongType())])))]
        )
        assert schema.treeString() == (
            "root\n"
            " |-- a: array (nullable = true)\n"
            " |    |-- element: struct (containsNull = true)\n"
            " |    |    |-- b: long (nullable = true)\n"
        )

    def test_decimal_shows_precision(self) -> None:
        schema = StructType([StructField("amount", DecimalType(12, 4))])
        assert " |-- amount: decimal(12,4) (nullable = true)" in schema.treeString()


class TestNeedConversion:
    """Only maps and structs containing them need reshaping out of Arrow."""

    def test_flat_schema_needs_none(self) -> None:
        assert not StructType([StructField("a", LongType())]).needConversion()

    def test_map_needs_conversion_even_when_nested(self) -> None:
        nested = StructType(
            [StructField("s", StructType([StructField("m", MapType(StringType(), LongType()))]))]
        )
        assert nested.needConversion()
        assert StructType(
            [StructField("a", ArrayType(MapType(StringType(), LongType())))]
        ).needConversion()


class TestRow:
    def test_named_row_behaves_as_tuple_and_record(self) -> None:
        row = Row(a=1, b="x")
        # `tuple(row)`, not `row`, on the left: an `==` against a tuple literal
        # narrows `row` to that tuple type for the rest of the function, and the
        # record half of the interface disappears with it.
        assert tuple(row) == (1, "x")
        assert row.a == 1
        assert row["b"] == "x"
        assert row[0] == 1
        assert row.asDict() == {"a": 1, "b": "x"}
        assert repr(row) == "Row(a=1, b='x')"

    def test_field_order_is_insertion_order(self) -> None:
        # The reference engine 3.0 stopped sorting kwargs; `Row(b=1, a=2)` keeps b first.
        assert Row(b=1, a=2).__fields__ == ("b", "a")

    def test_unnamed_row_is_a_plain_tuple(self) -> None:
        row = Row(1, 2)
        assert tuple(row) == (1, 2)
        assert row.__fields__ is None
        assert repr(row) == "<Row(1, 2)>"
        with pytest.raises(TypeError):
            row.asDict()

    def test_row_of_strings_is_a_factory(self) -> None:
        Person = Row("name", "age")
        built = Person("ada", 36)
        assert built.asDict() == {"name": "ada", "age": 36}
        assert repr(built) == "Row(name='ada', age=36)"

    def test_factory_rejects_non_string_fields(self) -> None:
        with pytest.raises(TypeError, match="field-name strings"):
            Row(1, 2)("a", "b")

    def test_mixing_positional_and_named_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            Row(1, a=2)

    def test_recursive_asdict_unwraps_nested_rows(self) -> None:
        row = Row(id=1, person=Row(name="ada"), tags=[Row(t=1)])
        assert row.asDict(recursive=True) == {
            "id": 1,
            "person": {"name": "ada"},
            "tags": [{"t": 1}],
        }
        # Without recursion the nested Row is left alone.
        assert isinstance(row.asDict()["person"], Row)

    def test_contains_checks_field_names_for_named_rows(self) -> None:
        assert "a" in Row(a=1)
        assert "z" not in Row(a=1)

    def test_missing_field_raises_the_right_error_kind(self) -> None:
        row = Row(a=1)
        with pytest.raises(AttributeError):
            _ = row.missing
        with pytest.raises(KeyError):
            _ = row["missing"]

    def test_pickles_with_its_field_names(self) -> None:
        row = Row(a=1, b="x")
        restored = pickle.loads(pickle.dumps(row))
        assert restored == row
        assert restored.__fields__ == ("a", "b")
        assert restored.asDict() == {"a": 1, "b": "x"}

    def test_unnamed_row_pickles_too(self) -> None:
        assert pickle.loads(pickle.dumps(Row(1, 2))) == (1, 2)

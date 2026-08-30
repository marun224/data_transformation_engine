"""Reading Spark types out of DDL and JSON.

These are the inverse of `simpleString()` and `jsonValue()`, so most of the value is
in the round-trips: anything the model can *write* it must be able to read back, or
`schema=` and schema serialisation quietly disagree with `printSchema`.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from icetl.errors import PySparkTypeError, PySparkValueError, UnsupportedFeatureError
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
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)


class TestAtomicDDL:
    @pytest.mark.parametrize(
        ("ddl", "expected"),
        [
            ("boolean", BooleanType()),
            ("tinyint", ByteType()),
            ("byte", ByteType()),
            ("smallint", ShortType()),
            ("short", ShortType()),
            ("int", IntegerType()),
            ("integer", IntegerType()),
            ("bigint", LongType()),
            ("long", LongType()),
            ("float", FloatType()),
            ("double", DoubleType()),
            ("string", StringType()),
            ("binary", BinaryType()),
            ("date", DateType()),
            ("void", NullType()),
        ],
    )
    def test_each_atomic_name(self, ddl: str, expected: DataType) -> None:
        assert DataType.fromDDL(ddl) == expected

    def test_case_is_ignored(self) -> None:
        assert DataType.fromDDL("BIGINT") == DataType.fromDDL("bigint")

    def test_surrounding_whitespace_is_ignored(self) -> None:
        assert DataType.fromDDL("  bigint  ") == LongType()


class TestTimestampSemantics:
    """Spark's `TIMESTAMP` is an instant; `TIMESTAMP_NTZ` is wall-clock.

    Reading these the wrong way round shifts every value by the session's UTC offset
    and nothing raises, which is why they get their own class.
    """

    def test_bare_timestamp_is_an_instant(self) -> None:
        assert DataType.fromDDL("timestamp") == TimestampType()

    def test_timestamp_ntz_is_wall_clock(self) -> None:
        assert DataType.fromDDL("timestamp_ntz") == TimestampNTZType()

    def test_the_two_are_not_interchangeable(self) -> None:
        assert TimestampType() != TimestampNTZType()

    def test_they_round_trip_through_ddl(self) -> None:
        for original in (TimestampType(), TimestampNTZType()):
            assert DataType.fromDDL(original.simpleString()) == original


class TestDecimal:
    def test_precision_and_scale(self) -> None:
        assert DataType.fromDDL("decimal(18,4)") == DecimalType(18, 4)

    def test_bare_decimal_is_spark_default(self) -> None:
        """Spark's `decimal` with no arguments is `decimal(10,0)`."""
        assert DataType.fromDDL("decimal") == DecimalType(10, 0)

    def test_precision_only(self) -> None:
        assert DataType.fromDDL("decimal(12)") == DecimalType(12, 0)


class TestNestedDDL:
    def test_array(self) -> None:
        assert DataType.fromDDL("array<string>") == ArrayType(StringType())

    def test_map(self) -> None:
        assert DataType.fromDDL("map<string,int>") == MapType(StringType(), IntegerType())

    def test_struct(self) -> None:
        assert DataType.fromDDL("struct<a:int,b:string>") == StructType(
            [StructField("a", IntegerType()), StructField("b", StringType())]
        )

    def test_deep_nesting(self) -> None:
        parsed = DataType.fromDDL("array<struct<a:int,b:array<map<string,double>>>>")
        assert parsed.simpleString() == "array<struct<a:int,b:array<map<string,double>>>>"

    def test_an_array_without_an_element_type_is_refused(self) -> None:
        with pytest.raises(PySparkValueError):
            DataType.fromDDL("array")


class TestStructDDL:
    """`StructType.fromDDL` takes the column-list form too -- what `schema=` gets."""

    def test_the_column_list_form(self) -> None:
        assert StructType.fromDDL("id BIGINT, name STRING") == StructType(
            [StructField("id", LongType()), StructField("name", StringType())]
        )

    def test_the_type_form(self) -> None:
        assert StructType.fromDDL("struct<id:bigint,name:string>") == StructType(
            [StructField("id", LongType()), StructField("name", StringType())]
        )

    def test_both_forms_agree(self) -> None:
        assert StructType.fromDDL("id BIGINT, name STRING") == StructType.fromDDL(
            "struct<id:bigint,name:string>"
        )

    def test_a_backtick_quoted_field_name(self) -> None:
        parsed = StructType.fromDDL("`odd name` INT")
        assert parsed.fieldNames() == ["odd name"]

    def test_a_field_named_like_a_keyword(self) -> None:
        assert StructType.fromDDL("`order` INT, `select` STRING").fieldNames() == [
            "order",
            "select",
        ]

    def test_nested_columns(self) -> None:
        parsed = StructType.fromDDL("p STRUCT<x:INT, y:STRUCT<z:DOUBLE>>")
        assert parsed.simpleString() == "struct<p:struct<x:int,y:struct<z:double>>>"

    def test_fields_are_nullable_as_spark_defaults(self) -> None:
        assert all(f.nullable for f in StructType.fromDDL("a INT, b STRING").fields)

    @pytest.mark.parametrize("bad", ["", "   ", "nonsense type here ("])
    def test_unparseable_ddl_is_refused(self, bad: str) -> None:
        with pytest.raises(PySparkValueError):
            StructType.fromDDL(bad)

    def test_a_non_string_is_refused(self) -> None:
        with pytest.raises(PySparkTypeError):
            StructType.fromDDL(123)  # type: ignore[arg-type]

    def test_an_unsupported_type_names_its_phase(self) -> None:
        with pytest.raises(UnsupportedFeatureError):
            DataType.fromDDL("interval year to month")


class TestJson:
    def test_an_atomic_type_is_a_bare_string(self) -> None:
        assert DataType.fromJson("long") == LongType()

    def test_decimal_from_json(self) -> None:
        assert DataType.fromJson("decimal(10,2)") == DecimalType(10, 2)

    def test_array_object(self) -> None:
        assert DataType.fromJson(
            {"type": "array", "elementType": "string", "containsNull": False}
        ) == ArrayType(StringType(), False)

    def test_map_object(self) -> None:
        parsed = DataType.fromJson(
            {
                "type": "map",
                "keyType": "string",
                "valueType": "long",
                "valueContainsNull": False,
            }
        )
        assert parsed == MapType(StringType(), LongType(), False)

    def test_struct_object_keeps_nullability_and_metadata(self) -> None:
        parsed = StructType.fromJson(
            {
                "type": "struct",
                "fields": [
                    {"name": "a", "type": "long", "nullable": False, "metadata": {"k": "v"}}
                ],
            }
        )
        assert parsed.fields[0].nullable is False
        assert parsed.fields[0].metadata == {"k": "v"}

    def test_a_field_missing_its_type_is_refused(self) -> None:
        with pytest.raises(PySparkValueError):
            StructType.fromJson({"type": "struct", "fields": [{"name": "a"}]})

    def test_an_unknown_type_object_is_refused(self) -> None:
        with pytest.raises(PySparkValueError):
            DataType.fromJson({"type": "matrix"})

    def test_a_nonsense_name_is_refused(self) -> None:
        with pytest.raises(PySparkValueError):
            DataType.fromJson("frobnicate")


class TestRoundTrips:
    """Anything the model can write, it must read back -- with one documented gap."""

    SCHEMAS: ClassVar[list[StructType]] = [
        StructType([StructField("id", LongType())]),
        StructType([StructField("a", DecimalType(18, 4)), StructField("b", BinaryType())]),
        StructType([StructField("tags", ArrayType(StringType(), False))]),
        StructType([StructField("m", MapType(StringType(), LongType(), False))]),
        StructType(
            [
                StructField(
                    "p",
                    StructType(
                        [StructField("x", IntegerType()), StructField("y", TimestampNTZType())]
                    ),
                )
            ]
        ),
        StructType([StructField("ts", TimestampType()), StructField("d", DateType())]),
    ]

    @pytest.mark.parametrize("schema", SCHEMAS)
    def test_json_round_trip(self, schema: StructType) -> None:
        assert StructType.fromJson(json.loads(json.dumps(schema.jsonValue()))) == schema

    @pytest.mark.parametrize("schema", SCHEMAS)
    def test_ddl_round_trip_preserves_names_and_types(self, schema: StructType) -> None:
        """Field names and types survive; see the nullability class below for what
        does not."""
        parsed = StructType.fromDDL(schema.simpleString())
        assert parsed.fieldNames() == schema.fieldNames()
        assert parsed.simpleString() == schema.simpleString()


class TestDDLLosesElementNullability:
    """Spark's DDL has nowhere to put `containsNull` / `valueContainsNull`.

    `ArrayType(StringType(), containsNull=False).simpleString()` is `array<string>`,
    so a DDL round-trip widens it back to nullable. That is Spark's behaviour, not
    ours -- but it is a real trap, so it is pinned here rather than left to be
    rediscovered, and `jsonValue()` is the form that keeps the flag.
    """

    def test_array_element_nullability_is_lost_through_ddl(self) -> None:
        original = ArrayType(StringType(), containsNull=False)
        restored = DataType.fromDDL(original.simpleString())
        assert isinstance(restored, ArrayType)
        assert restored.containsNull is True

    def test_map_value_nullability_is_lost_through_ddl(self) -> None:
        original = MapType(StringType(), LongType(), valueContainsNull=False)
        restored = DataType.fromDDL(original.simpleString())
        assert isinstance(restored, MapType)
        assert restored.valueContainsNull is True

    def test_json_keeps_what_ddl_drops(self) -> None:
        original = StructType([StructField("tags", ArrayType(StringType(), False))])
        assert StructType.fromJson(json.loads(json.dumps(original.jsonValue()))) == original

    def test_top_level_field_nullability_survives_json(self) -> None:
        original = StructType([StructField("a", LongType(), nullable=False)])
        assert StructType.fromJson(original.jsonValue()).fields[0].nullable is False

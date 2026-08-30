"""Arrow -> Row conversion, `show()` rendering, and the pandas dtype contract."""

from __future__ import annotations

import pyarrow as pa
import pytest

from icetl.exec.result import NULL_DISPLAY, format_show, to_pandas, to_rows
from icetl.plan.analysis import arrow_to_spark_schema, arrow_to_spark_type
from icetl.types import (
    ArrayType,
    BinaryType,
    BooleanType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    Row,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampNTZType,
    TimestampType,
)

FLAT = StructType(
    [
        StructField("id", LongType()),
        StructField("vendor", StringType()),
        StructField("amount", DoubleType()),
    ]
)


def rows(*values: tuple) -> list[Row]:
    return [Row._from_fields(("id", "vendor", "amount"), value) for value in values]


class TestArrowTypeMapping:
    @pytest.mark.parametrize(
        ("arrow_type", "spark_type"),
        [
            (pa.int32(), IntegerType()),
            (pa.int64(), LongType()),
            (pa.float64(), DoubleType()),
            (pa.string(), StringType()),
            (pa.large_string(), StringType()),
            (pa.bool_(), BooleanType()),
            (pa.binary(), BinaryType()),
            (pa.date32(), DateType()),
            (pa.decimal128(10, 2), DecimalType(10, 2)),
            # Unsigned types widen to the smallest signed type that holds them.
            (pa.uint8(), ShortType()),
            (pa.uint32(), LongType()),
        ],
    )
    def test_scalar_types(self, arrow_type: pa.DataType, spark_type: DataType) -> None:
        assert arrow_to_spark_type(arrow_type) == spark_type

    def test_timestamp_zone_decides_the_spark_type(self) -> None:
        # Iceberg's `timestamptz` vs `timestamp`, carried through Arrow.
        assert arrow_to_spark_type(pa.timestamp("us", tz="UTC")) == TimestampType()
        assert arrow_to_spark_type(pa.timestamp("us")) == TimestampNTZType()

    def test_nested_types(self) -> None:
        assert arrow_to_spark_type(pa.list_(pa.int64())) == ArrayType(LongType(), True)
        assert arrow_to_spark_type(pa.map_(pa.string(), pa.int64())) == MapType(
            StringType(), LongType(), True
        )
        assert arrow_to_spark_type(pa.struct([("a", pa.int64())])) == StructType(
            [StructField("a", LongType(), True)]
        )

    def test_schema_preserves_field_order_and_nullability(self) -> None:
        schema = pa.schema([pa.field("a", pa.int64(), nullable=False), pa.field("b", pa.string())])
        assert arrow_to_spark_schema(schema) == StructType(
            [StructField("a", LongType(), False), StructField("b", StringType(), True)]
        )


class TestToRows:
    def test_flat_table(self) -> None:
        table = pa.table({"id": [1, 2], "vendor": ["a", None], "amount": [1.5, 2.5]})
        assert to_rows(table, FLAT) == [
            Row(id=1, vendor="a", amount=1.5),
            Row(id=2, vendor=None, amount=2.5),
        ]

    def test_rows_carry_their_field_names(self) -> None:
        table = pa.table({"id": [1], "vendor": ["a"], "amount": [1.0]})
        assert to_rows(table, FLAT)[0].asDict() == {"id": 1, "vendor": "a", "amount": 1.0}

    def test_structs_become_nested_rows_and_maps_become_dicts(self) -> None:
        arrow_schema = pa.schema(
            [
                ("person", pa.struct([("name", pa.string())])),
                ("tags", pa.list_(pa.string())),
                ("scores", pa.map_(pa.string(), pa.int64())),
            ]
        )
        table = pa.table(
            {"person": [{"name": "ada"}], "tags": [["x"]], "scores": [[("a", 1)]]},
            schema=arrow_schema,
        )
        [row] = to_rows(table, arrow_to_spark_schema(arrow_schema))
        assert row.person == Row(name="ada")
        assert row.tags == ["x"]
        # Arrow hands maps back as pairs; Spark yields a dict.
        assert row.scores == {"a": 1}

    def test_nulls_survive_conversion(self) -> None:
        arrow_schema = pa.schema([("person", pa.struct([("name", pa.string())]))])
        table = pa.table({"person": [None]}, schema=arrow_schema)
        assert to_rows(table, arrow_to_spark_schema(arrow_schema))[0].person is None

    def test_empty_table_yields_no_rows(self) -> None:
        table = pa.schema(
            [("id", pa.int64()), ("vendor", pa.string()), ("amount", pa.float64())]
        ).empty_table()
        assert to_rows(table, FLAT) == []


class TestToPandas:
    def test_strings_come_back_as_object_dtype(self) -> None:
        # Spark's toPandas() yields `object`; pandas 3 would otherwise use `str`.
        frame = to_pandas(pa.table({"s": ["a", None], "i": pa.array([1, 2], pa.int64())}))
        assert frame["s"].dtype == object
        assert str(frame["i"].dtype) == "int64"

    def test_values_are_unchanged(self) -> None:
        frame = to_pandas(pa.table({"s": ["a", None]}))
        assert frame["s"].tolist() == ["a", None]


class TestFormatShow:
    def test_layout_matches_spark(self) -> None:
        out = format_show(
            rows((1, "a", 1.5)), FLAT, n=20, truncate=20, vertical=False, has_more=False
        )
        assert out == (
            "+---+------+------+\n"
            "| id|vendor|amount|\n"
            "+---+------+------+\n"
            "|  1|     a|   1.5|\n"
            "+---+------+------+\n"
        )

    def test_nulls_render_as_the_configured_display(self) -> None:
        out = format_show(
            rows((1, None, 1.5)), FLAT, n=20, truncate=20, vertical=False, has_more=False
        )
        assert f"|{NULL_DISPLAY:>6}|" in out

    def test_truncation_on_right_justifies_and_off_left_justifies(self) -> None:
        long_rows = [
            Row._from_fields(("id", "vendor", "amount"), (1, "a-very-long-vendor-name", 1.5))
        ]
        truncated = format_show(long_rows, FLAT, n=20, truncate=8, vertical=False, has_more=False)
        assert "|a-ver...|" in truncated

        untruncated = format_show(long_rows, FLAT, n=20, truncate=0, vertical=False, has_more=False)
        assert "|a-very-long-vendor-name|" in untruncated
        # truncate=0 switches Spark to left-justified cells.
        assert "|1  |" in untruncated

    def test_footer_appears_only_when_rows_were_dropped(self) -> None:
        many = rows((1, "a", 1.0), (2, "b", 2.0))
        assert "only showing top 1 row\n" in format_show(
            many, FLAT, n=1, truncate=20, vertical=False, has_more=True
        )
        assert "only showing" not in format_show(
            many, FLAT, n=20, truncate=20, vertical=False, has_more=False
        )

    def test_header_only_when_there_are_no_rows(self) -> None:
        out = format_show([], FLAT, n=20, truncate=20, vertical=False, has_more=False)
        assert out == (
            "+---+------+------+\n| id|vendor|amount|\n+---+------+------+\n+---+------+------+\n"
        )

    def test_booleans_render_lowercase(self) -> None:
        schema = StructType([StructField("flag", BooleanType())])
        out = format_show(
            [Row._from_fields(("flag",), (True,))],
            schema,
            n=20,
            truncate=20,
            vertical=False,
            has_more=False,
        )
        assert "|true|" in out

    def test_nested_values_use_sparks_braces_and_arrows(self) -> None:
        schema = StructType(
            [
                StructField("s", StructType([StructField("a", LongType())])),
                StructField("l", ArrayType(LongType())),
                StructField("m", MapType(StringType(), LongType())),
            ]
        )
        out = format_show(
            [Row._from_fields(("s", "l", "m"), (Row(a=1), [1, 2], {"k": 9}))],
            schema,
            n=20,
            truncate=0,
            vertical=False,
            has_more=False,
        )
        assert "{1}" in out
        assert "[1, 2]" in out
        assert "{k -> 9}" in out

    def test_vertical_mode(self) -> None:
        out = format_show(
            rows((1, "a", 1.5)), FLAT, n=20, truncate=20, vertical=True, has_more=False
        )
        assert out == "-RECORD 0\n     id | 1\n vendor | a\n amount | 1.5\n"

    def test_vertical_mode_with_no_rows(self) -> None:
        assert (
            format_show([], FLAT, n=20, truncate=20, vertical=True, has_more=False) == "(0 rows)\n"
        )

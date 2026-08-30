"""Reading Spark types back out of DDL and JSON -- the inverse of `simpleString` and
`jsonValue`.

Two entry points, both of which Spark scripts use constantly:

    StructType.fromDDL("id BIGINT, name STRING")     schema= on readers, UDF returns
    StructType.fromJson(json.loads(text))            schema round-tripping

**DDL is parsed with sqlglot's Spark dialect, not by hand.** Spark's type grammar has
more corners than it looks -- `decimal(10,2)`, `array<struct<a:int>>`,
`map<string,array<int>>`, backtick-quoted field names, `not null` -- and we already
depend on a parser that knows all of them. Hand-rolling it would mean maintaining a
second, worse grammar and finding the corners in production.

Living in its own module rather than in `types.py` keeps that dependency one-way:
`types.py` stays a plain data model with no import of sqlglot, and this is the only
place that knows how Spark spells a type in text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlglot import exp
from sqlglot import parse_one as _parse_one

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

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["parse_datatype_json", "parse_datatype_string", "parse_struct_ddl"]

# sqlglot's type token -> the Spark type it means. Several tokens collapse onto one
# Spark type, which is why this is a mapping rather than a name lookup: Spark has no
# unsigned types and no `TEXT`, so those widen to what Spark's own reader gives.
_TOKENS: dict[Any, DataType] = {
    exp.DataType.Type.BOOLEAN: BooleanType(),
    exp.DataType.Type.TINYINT: ByteType(),
    exp.DataType.Type.SMALLINT: ShortType(),
    exp.DataType.Type.INT: IntegerType(),
    exp.DataType.Type.BIGINT: LongType(),
    exp.DataType.Type.FLOAT: FloatType(),
    exp.DataType.Type.DOUBLE: DoubleType(),
    exp.DataType.Type.VARCHAR: StringType(),
    exp.DataType.Type.CHAR: StringType(),
    exp.DataType.Type.TEXT: StringType(),
    exp.DataType.Type.BINARY: BinaryType(),
    exp.DataType.Type.VARBINARY: BinaryType(),
    exp.DataType.Type.DATE: DateType(),
    # Spark's bare `TIMESTAMP` is an *instant*, and sqlglot's Spark dialect resolves
    # it to TIMESTAMPTZ accordingly -- so the TIMESTAMP token below is not what
    # `"timestamp"` parses to here. It is kept as the reading for an already-zoneless
    # token, which is what a DuckDB-parsed node would carry. Pinned by
    # `test_parse_types.py::TestTimestampSemantics`, because getting this backwards
    # shifts every value by the session offset and nothing would raise.
    exp.DataType.Type.TIMESTAMP: TimestampNTZType(),
    exp.DataType.Type.TIMESTAMPNTZ: TimestampNTZType(),
    exp.DataType.Type.TIMESTAMPTZ: TimestampType(),
    exp.DataType.Type.TIMESTAMPLTZ: TimestampType(),
    exp.DataType.Type.NULL: NullType(),
}

#: The names Spark's own JSON schema uses for its atomic types.
_JSON_ATOMIC: dict[str, DataType] = {
    "boolean": BooleanType(),
    "byte": ByteType(),
    "tinyint": ByteType(),
    "short": ShortType(),
    "smallint": ShortType(),
    "integer": IntegerType(),
    "int": IntegerType(),
    "long": LongType(),
    "bigint": LongType(),
    "float": FloatType(),
    "double": DoubleType(),
    "string": StringType(),
    "binary": BinaryType(),
    "date": DateType(),
    "timestamp": TimestampType(),
    "timestamp_ntz": TimestampNTZType(),
    "void": NullType(),
    "null": NullType(),
}


def _from_sqlglot(node: exp.DataType) -> DataType:
    """One sqlglot type node as a Spark `DataType`."""
    kind = node.this

    if kind == exp.DataType.Type.DECIMAL:
        # `decimal` with no arguments is Spark's `decimal(10,0)`.
        params = [int(p.name) for p in node.expressions if isinstance(p, exp.DataTypeParam)]
        if not params:
            return DecimalType()
        if len(params) == 1:
            return DecimalType(params[0], 0)
        return DecimalType(params[0], params[1])

    if kind in (exp.DataType.Type.ARRAY, exp.DataType.Type.LIST):
        if not node.expressions:
            raise PySparkValueError("array requires an element type, as in `array<int>`.")
        return ArrayType(_from_sqlglot(node.expressions[0]))

    if kind == exp.DataType.Type.MAP:
        if len(node.expressions) != 2:
            raise PySparkValueError("map requires a key and a value type, as in `map<string,int>`.")
        return MapType(_from_sqlglot(node.expressions[0]), _from_sqlglot(node.expressions[1]))

    if kind == exp.DataType.Type.STRUCT:
        return StructType([_struct_field(field) for field in node.expressions])

    simple = _TOKENS.get(kind)
    if simple is not None:
        return simple

    raise UnsupportedFeatureError(f"The Spark type {node.sql(dialect='spark')!r}", phase="Phase 6")


def _struct_field(node: exp.Expression) -> StructField:
    """One `name TYPE` pair inside a struct."""
    if isinstance(node, exp.ColumnDef):
        kind = node.args.get("kind")
        if kind is None:
            raise PySparkValueError(f"Struct field {node.name!r} has no type.")
        return StructField(node.name, _from_sqlglot(kind), nullable=True)
    if isinstance(node, exp.DataType):
        # `struct<int, string>` -- Spark names these col1, col2, ...
        raise PySparkValueError(
            "Struct fields need names, as in `struct<a:int>`; positional struct types "
            "are not part of Spark's DDL."
        )
    raise PySparkValueError(f"Could not read the struct field {node.sql(dialect='spark')!r}.")


#: Spark type names sqlglot's Spark dialect does not recognise. `void` is Spark's own
#: spelling of `NullType` -- it is what `NullType().simpleString()` returns, so
#: without this entry a schema could be written and not read back.
_SPARK_ONLY: dict[str, DataType] = {"void": NullType()}


def parse_datatype_string(ddl: str) -> DataType:
    """One Spark type from DDL: `"bigint"`, `"decimal(10,2)"`, `"array<string>"`."""
    if not isinstance(ddl, str):
        raise PySparkTypeError(f"Expected a DDL string, got {type(ddl).__name__}.")
    text = ddl.strip()
    if not text:
        raise PySparkValueError("Cannot parse an empty type string.")
    spark_only = _SPARK_ONLY.get(text.lower())
    if spark_only is not None:
        return spark_only
    try:
        built = exp.DataType.build(text, dialect="spark")
    except Exception as exc:
        raise PySparkValueError(f"{ddl!r} is not a recognised Spark type.") from exc
    return _from_sqlglot(built)


def parse_struct_ddl(ddl: str) -> StructType:
    """A `StructType` from Spark's two DDL spellings.

        "id BIGINT, name STRING"        the column-list form, as used by `schema=`
        "struct<id:bigint,name:string>" the type form, as used inside another type

    The first is what people write and is not a *type* at all -- it is a column list,
    which `exp.DataType.build` will not accept. Parsing it as the tail of a
    `CREATE TABLE` is how sqlglot is asked the same question.

    **DDL loses element nullability**, in Spark as much as here:
    `ArrayType(StringType(), containsNull=False).simpleString()` is `array<string>`,
    so reading it back gives `containsNull=True`. Round-trip a schema through
    `jsonValue()` / `fromJson()` when that flag matters -- JSON carries it, DDL cannot.
    """
    if not isinstance(ddl, str):
        raise PySparkTypeError(f"Expected a DDL string, got {type(ddl).__name__}.")
    text = ddl.strip()
    if not text:
        raise PySparkValueError("Cannot parse an empty schema string.")

    if text.lower().startswith("struct<"):
        parsed = parse_datatype_string(text)
        if not isinstance(parsed, StructType):  # pragma: no cover - defensive
            raise PySparkValueError(f"{ddl!r} is not a struct type.")
        return parsed

    # A bare `a INT, b STRING`. `CREATE TABLE _ (...)` gives sqlglot the context it
    # needs, and the column definitions come back in the same shape struct fields do.
    try:
        statement = _parse_one(f"CREATE TABLE _icetl_ddl ({text})", read="spark")
        schema = statement.find(exp.Schema)
        if schema is None:
            raise ValueError("no column list")
        fields = [f for f in schema.expressions if isinstance(f, exp.ColumnDef)]
        if not fields:
            raise ValueError("no columns")
    except Exception as exc:
        raise PySparkValueError(f"Could not parse the schema {ddl!r}: {exc}") from exc
    return StructType([_struct_field(field) for field in fields])


def parse_datatype_json(value: Any) -> DataType:
    """A type from Spark's JSON form -- the inverse of `DataType.jsonValue()`.

    Spark writes an atomic type as a bare string and everything else as an object
    with a `type` discriminator, so this dispatches on which one arrived.
    """
    if isinstance(value, str):
        lowered = value.lower()
        atomic = _JSON_ATOMIC.get(lowered)
        if atomic is not None:
            return atomic
        if lowered.startswith("decimal"):
            return parse_datatype_string(lowered)
        raise PySparkValueError(f"{value!r} is not a recognised Spark type name.")

    if not isinstance(value, dict):
        raise PySparkTypeError(f"Expected a type name or object, got {type(value).__name__}.")

    kind = value.get("type")
    if kind == "array":
        return ArrayType(
            parse_datatype_json(value["elementType"]),
            containsNull=bool(value.get("containsNull", True)),
        )
    if kind == "map":
        return MapType(
            parse_datatype_json(value["keyType"]),
            parse_datatype_json(value["valueType"]),
            valueContainsNull=bool(value.get("valueContainsNull", True)),
        )
    if kind == "struct":
        return StructType([_field_from_json(f) for f in value.get("fields", [])])
    raise PySparkValueError(f"Unknown type object {kind!r}.")


def _field_from_json(value: Mapping[str, Any]) -> StructField:
    if "name" not in value or "type" not in value:
        raise PySparkValueError(f"A struct field needs `name` and `type`; got {sorted(value)}.")
    return StructField(
        value["name"],
        parse_datatype_json(value["type"]),
        nullable=bool(value.get("nullable", True)),
        metadata=dict(value.get("metadata") or {}),
    )

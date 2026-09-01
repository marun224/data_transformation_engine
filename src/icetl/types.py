"""The reference engine's type hierarchy, plus `Row`.

Phase 1 carries the types the fixture tables and the thin slice actually produce.
Phase 3 completes the set (`fromDDL`, `fromJson`, the full mapping matrix) -- the
classes here are the ones it will extend, not throwaways.

Two of the reference engine's naming conventions are easy to confuse, so they are spelled out:

    typeName()      "long", "decimal", "struct"     -- what printSchema shows
    simpleString()  "bigint", "decimal(10,2)"       -- what df.dtypes and DDL show

`DecimalType` is the odd one out: its *tree* name carries precision and scale, which
is why `treeName()` exists rather than printSchema simply calling `typeName()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from icetl.errors import EngineValueError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "ArrayType",
    "AtomicType",
    "BinaryType",
    "BooleanType",
    "ByteType",
    "DataType",
    "DateType",
    "DecimalType",
    "DoubleType",
    "FloatType",
    "FractionalType",
    "IntegerType",
    "IntegralType",
    "LongType",
    "MapType",
    "NullType",
    "NumericType",
    "Row",
    "ShortType",
    "StringType",
    "StructField",
    "StructType",
    "TimestampNTZType",
    "TimestampType",
]


class DataType:
    """Base of every the reference engine type.

    The atomic types are value objects: two `LongType()` compare equal, which is what
    lets schema assertions in tests read naturally.
    """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __hash__(self) -> int:
        return hash(repr(self))

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    @classmethod
    def typeName(cls) -> str:
        """`long`, `struct`, `decimal` -- the name printSchema uses."""
        return cls.__name__.removesuffix("Type").lower()

    def simpleString(self) -> str:
        """`bigint`, `array<int>` -- the name `df.dtypes` and the reference engine DDL use."""
        return self.typeName()

    def treeName(self) -> str:
        """The name `printSchema` shows. Differs from `typeName` only for decimals."""
        return self.typeName()

    def jsonValue(self) -> str | dict[str, Any]:
        return self.typeName()

    def needConversion(self) -> bool:
        """True when Arrow's Python form of this type is not the reference engine's."""
        return False

    @classmethod
    def fromDDL(cls, ddl: str) -> DataType:
        """A type from the reference engine DDL: `"bigint"`, `"decimal(10,2)"`, `"array<string>"`.

        The parser lives in `icetl.parse_types`, imported here rather than at module
        scope so this module stays a plain data model with no dependency on sqlglot.
        """
        from icetl.parse_types import parse_datatype_string

        return parse_datatype_string(ddl)

    @classmethod
    def fromJson(cls, json: Any) -> DataType:
        """The inverse of `jsonValue()`. Accepts the parsed object, not the text."""
        from icetl.parse_types import parse_datatype_json

        return parse_datatype_json(json)


class AtomicType(DataType):
    """A type with no nested components."""


class NumericType(AtomicType):
    """Numeric types."""


class IntegralType(NumericType):
    """Whole-number types."""


class FractionalType(NumericType):
    """Types with a fractional part."""


class NullType(DataType):
    """The type of an untyped NULL. The reference engine spells it `void`."""

    @classmethod
    def typeName(cls) -> str:
        return "void"


class BooleanType(AtomicType):
    pass


class ByteType(IntegralType):
    def simpleString(self) -> str:
        return "tinyint"


class ShortType(IntegralType):
    def simpleString(self) -> str:
        return "smallint"


class IntegerType(IntegralType):
    def simpleString(self) -> str:
        return "int"


class LongType(IntegralType):
    def simpleString(self) -> str:
        return "bigint"


class FloatType(FractionalType):
    pass


class DoubleType(FractionalType):
    pass


class StringType(AtomicType):
    pass


class BinaryType(AtomicType):
    pass


class DateType(AtomicType):
    pass


class TimestampType(AtomicType):
    """The reference engine's `TIMESTAMP`: an instant, stored UTC. Iceberg's `timestamptz`."""


class TimestampNTZType(AtomicType):
    """The reference engine's `TIMESTAMP_NTZ`: wall-clock time, no zone. Iceberg's `timestamp`."""

    @classmethod
    def typeName(cls) -> str:
        return "timestamp_ntz"


class DecimalType(FractionalType):
    """Fixed-precision decimal. The reference engine's default is `DECIMAL(10, 0)`."""

    def __init__(self, precision: int = 10, scale: int = 0) -> None:
        if not 1 <= precision <= 38:
            raise ValueError(f"Decimal precision must be in 1..38, got {precision}.")
        if not 0 <= scale <= precision:
            raise ValueError(f"Decimal scale must be in 0..{precision}, got {scale}.")
        self.precision = precision
        self.scale = scale

    def __repr__(self) -> str:
        return f"DecimalType({self.precision}, {self.scale})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DecimalType)
            and self.precision == other.precision
            and self.scale == other.scale
        )

    def __hash__(self) -> int:
        return hash(("decimal", self.precision, self.scale))

    def simpleString(self) -> str:
        return f"decimal({self.precision},{self.scale})"

    def treeName(self) -> str:
        # The reference engine's DecimalType overrides `typeName` to carry precision and scale, so
        # printSchema shows `decimal(10,2)` rather than a bare `decimal`.
        return self.simpleString()

    def jsonValue(self) -> str:
        return self.simpleString()


class ArrayType(DataType):
    def __init__(self, elementType: DataType, containsNull: bool = True) -> None:
        self.elementType = elementType
        self.containsNull = containsNull

    def __repr__(self) -> str:
        return f"ArrayType({self.elementType!r}, {self.containsNull})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ArrayType)
            and self.elementType == other.elementType
            and self.containsNull == other.containsNull
        )

    def __hash__(self) -> int:
        return hash(("array", self.elementType, self.containsNull))

    def simpleString(self) -> str:
        return f"array<{self.elementType.simpleString()}>"

    def jsonValue(self) -> dict[str, Any]:
        return {
            "type": "array",
            "elementType": self.elementType.jsonValue(),
            "containsNull": self.containsNull,
        }

    def needConversion(self) -> bool:
        return self.elementType.needConversion()


class MapType(DataType):
    def __init__(
        self, keyType: DataType, valueType: DataType, valueContainsNull: bool = True
    ) -> None:
        self.keyType = keyType
        self.valueType = valueType
        self.valueContainsNull = valueContainsNull

    def __repr__(self) -> str:
        return f"MapType({self.keyType!r}, {self.valueType!r}, {self.valueContainsNull})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MapType)
            and self.keyType == other.keyType
            and self.valueType == other.valueType
            and self.valueContainsNull == other.valueContainsNull
        )

    def __hash__(self) -> int:
        return hash(("map", self.keyType, self.valueType, self.valueContainsNull))

    def simpleString(self) -> str:
        return f"map<{self.keyType.simpleString()},{self.valueType.simpleString()}>"

    def jsonValue(self) -> dict[str, Any]:
        return {
            "type": "map",
            "keyType": self.keyType.jsonValue(),
            "valueType": self.valueType.jsonValue(),
            "valueContainsNull": self.valueContainsNull,
        }

    def needConversion(self) -> bool:
        # Arrow hands a map back as a list of (key, value) pairs; the reference engine yields a
        # dict.
        return True


class StructField(DataType):
    """One named field of a `StructType`."""

    def __init__(
        self,
        name: str,
        dataType: DataType,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.dataType = dataType
        self.nullable = nullable
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"StructField('{self.name}', {self.dataType!r}, {self.nullable})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StructField)
            and self.name == other.name
            and self.dataType == other.dataType
            and self.nullable == other.nullable
        )

    def __hash__(self) -> int:
        return hash((self.name, self.dataType, self.nullable))

    def simpleString(self) -> str:
        return f"{self.name}:{self.dataType.simpleString()}"

    def jsonValue(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.dataType.jsonValue(),
            "nullable": self.nullable,
            "metadata": self.metadata,
        }

    def needConversion(self) -> bool:
        return self.dataType.needConversion()


class StructType(DataType):
    """An ordered list of named fields -- also the type of a DataFrame's rows."""

    def __init__(self, fields: list[StructField] | None = None) -> None:
        self.fields: list[StructField] = list(fields or [])
        self.names: list[str] = [f.name for f in self.fields]

    def add(
        self,
        field: str | StructField,
        data_type: DataType | None = None,
        nullable: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> StructType:
        """Append a field, returning self so calls chain as they do in the reference engine."""
        if isinstance(field, StructField):
            self.fields.append(field)
        elif data_type is None:
            raise ValueError("A data type is required when adding a field by name.")
        else:
            self.fields.append(StructField(field, data_type, nullable, metadata))
        self.names = [f.name for f in self.fields]
        return self

    def __repr__(self) -> str:
        return f"StructType([{', '.join(repr(f) for f in self.fields)}])"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StructType) and self.fields == other.fields

    def __hash__(self) -> int:
        return hash(tuple(self.fields))

    def __iter__(self) -> Iterator[StructField]:
        return iter(self.fields)

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, key: str | int | slice) -> StructField | StructType:
        if isinstance(key, str):
            for field in self.fields:
                if field.name == key:
                    return field
            raise KeyError(f"No StructField named {key}.")
        if isinstance(key, slice):
            return StructType(self.fields[key])
        return self.fields[key]

    @classmethod
    def fromDDL(cls, ddl: str) -> StructType:
        """A schema from either the reference engine DDL spelling.

            "id BIGINT, name STRING"          the column-list form
            "struct<id:bigint,name:string>"   the type form

        Overrides `DataType.fromDDL` because only a struct accepts the first, which is
        the one people actually write in `schema=`.
        """
        from icetl.parse_types import parse_struct_ddl

        return parse_struct_ddl(ddl)

    @classmethod
    def fromJson(cls, json: Any) -> StructType:
        from icetl.parse_types import parse_datatype_json

        parsed = parse_datatype_json(json)
        if not isinstance(parsed, StructType):
            raise TypeError(f"Expected a struct, got {parsed.simpleString()}.")
        return parsed

    def fieldNames(self) -> list[str]:
        return list(self.names)

    def simpleString(self) -> str:
        return f"struct<{','.join(f.simpleString() for f in self.fields)}>"

    def jsonValue(self) -> dict[str, Any]:
        return {"type": "struct", "fields": [f.jsonValue() for f in self.fields]}

    def needConversion(self) -> bool:
        return any(f.needConversion() for f in self.fields)

    def treeString(self, level: int | None = None) -> str:
        """The `printSchema()` rendering, matching the reference engine's layout.

        `level` stops the walk after that many levels of nesting, which only becomes
        useful once a schema *has* nesting -- a struct of structs of arrays prints a
        page otherwise. Level 1 is the top-level fields; None means all of it.
        """
        if level is not None and level < 1:
            raise EngineValueError(f"treeString() expects level >= 1, got {level}.")
        lines = ["root"]
        for field in self.fields:
            _tree_field(lines, field.name, field.dataType, field.nullable, " |", level)
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# printSchema rendering
#
# The reference engine's shape: every level indents four spaces and re-opens a `|` gutter, and the
# trailing parenthetical names the nullability flag that applies at that level --
# `nullable` for struct fields, `containsNull` for array elements, `valueContainsNull`
# for map values. A map *key* carries no flag at all, because keys are never null.
# ---------------------------------------------------------------------------


def _tree_field(
    lines: list[str],
    name: str,
    data_type: DataType,
    nullable: bool,
    prefix: str,
    depth: int | None = None,
) -> None:
    lines.append(f"{prefix}-- {name}: {data_type.treeName()} (nullable = {str(nullable).lower()})")
    _tree_children(lines, data_type, f"{prefix}    |", _deeper(depth))


def _tree_labelled(
    lines: list[str],
    label: str,
    data_type: DataType,
    prefix: str,
    flag: str,
    value: bool,
    depth: int | None = None,
) -> None:
    lines.append(f"{prefix}-- {label}: {data_type.treeName()} ({flag} = {str(value).lower()})")
    _tree_children(lines, data_type, f"{prefix}    |", _deeper(depth))


def _deeper(depth: int | None) -> int | None:
    """One level further in, or None when no limit was asked for."""
    return None if depth is None else depth - 1


def _tree_children(
    lines: list[str], data_type: DataType, prefix: str, depth: int | None = None
) -> None:
    if depth is not None and depth < 1:
        return
    if isinstance(data_type, StructType):
        for field in data_type.fields:
            _tree_field(lines, field.name, field.dataType, field.nullable, prefix, depth)
    elif isinstance(data_type, ArrayType):
        _tree_labelled(
            lines,
            "element",
            data_type.elementType,
            prefix,
            "containsNull",
            data_type.containsNull,
            depth,
        )
    elif isinstance(data_type, MapType):
        lines.append(f"{prefix}-- key: {data_type.keyType.treeName()}")
        _tree_children(lines, data_type.keyType, f"{prefix}    |", _deeper(depth))
        _tree_labelled(
            lines,
            "value",
            data_type.valueType,
            prefix,
            "valueContainsNull",
            data_type.valueContainsNull,
            depth,
        )


# ---------------------------------------------------------------------------
# Row
# ---------------------------------------------------------------------------


class Row(tuple):
    """A result row: a tuple that also answers to its field names.

    Mirrors `icetl.sql.Row`, including the two-step factory form:

        >>> Person = Row("name", "age")
        >>> Person("ada", 36)
        Row(name='ada', age=36)

    Field order is insertion order, as in the reference 3.0+ (older releases sorted kwargs).
    """

    __fields__: tuple[str, ...] | None

    def __new__(cls, *args: Any, **kwargs: Any) -> Row:
        if args and kwargs:
            raise ValueError("Row takes either positional values or named fields, not both.")
        if kwargs:
            row = super().__new__(cls, tuple(kwargs.values()))
            row.__fields__ = tuple(kwargs)
            return row
        row = super().__new__(cls, args)
        # A Row built from bare positional values is either an unnamed row or, when
        # every value is a string, a factory for rows with those field names. Which
        # one it is only becomes clear when it is called; see `__call__`.
        row.__fields__ = None
        return row

    @classmethod
    def _from_fields(cls, fields: tuple[str, ...], values: tuple[Any, ...]) -> Row:
        """Build a named row directly. The internal constructor the result path uses."""
        if len(fields) != len(values):
            raise ValueError(f"Row has {len(fields)} field(s) but {len(values)} value(s).")
        row = tuple.__new__(cls, values)
        row.__fields__ = fields
        return row

    def __call__(self, *args: Any) -> Row:
        """Use an unnamed Row of strings as a factory for named rows."""
        if self.__fields__ is not None:
            raise TypeError(f"{self!r} is already a named row and cannot be called.")
        if not all(isinstance(name, str) for name in self):
            raise TypeError("Only a Row of field-name strings can be used as a row factory.")
        return Row._from_fields(tuple(self), args)

    def asDict(self, recursive: bool = False) -> dict[str, Any]:
        """As a dict. With `recursive=True`, nested Rows become dicts too."""
        if self.__fields__ is None:
            raise TypeError("Cannot convert an unnamed Row to a dict.")

        def convert(value: Any) -> Any:
            if isinstance(value, Row):
                return value.asDict(recursive=True)
            if isinstance(value, list):
                return [convert(v) for v in value]
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value

        if recursive:
            return {name: convert(value) for name, value in zip(self.__fields__, self, strict=True)}
        return dict(zip(self.__fields__, self, strict=True))

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (int, slice)):
            return super().__getitem__(item)
        if self.__fields__ is None:
            raise KeyError(item)
        try:
            index = self.__fields__.index(item)
        except ValueError as exc:
            raise KeyError(item) from exc
        return super().__getitem__(index)

    def __getattr__(self, item: str) -> Any:
        # Only reached for names that are not real attributes, so `__fields__` itself
        # never routes through here.
        if item.startswith("__"):
            raise AttributeError(item)
        fields = self.__dict__.get("__fields__")
        if fields is None:
            raise AttributeError(item)
        try:
            index = fields.index(item)
        except ValueError as exc:
            raise AttributeError(item) from exc
        return tuple.__getitem__(self, index)

    def __contains__(self, item: object) -> bool:
        if self.__fields__ is not None and isinstance(item, str):
            return item in self.__fields__
        return tuple.__contains__(self, item)

    def __repr__(self) -> str:
        if self.__fields__ is None:
            return f"<Row({', '.join(repr(v) for v in self)})>"
        pairs = ", ".join(
            f"{name}={value!r}" for name, value in zip(self.__fields__, self, strict=True)
        )
        return f"Row({pairs})"

    def __reduce__(self) -> Any:
        """Pickle support -- rows travel into worker processes and notebooks."""
        if self.__fields__ is None:
            return (Row, tuple(self))
        return (_row_from_fields, (self.__fields__, tuple(self)))


def _row_from_fields(fields: tuple[str, ...], values: tuple[Any, ...]) -> Row:
    """Module-level unpickler for `Row.__reduce__`."""
    return Row._from_fields(fields, values)

"""Working out a plan's output schema, and failing early when it has none.

The reference engine analyses eagerly: `df.select("typo")` raises `AnalysisException` there and
then, not later at `collect()`. Reproducing that needs a binder, and we already
depend on one -- so rather than hand-rolling type inference, we run the plan against
**zero-row Arrow views** of its source tables on a throwaway DuckDB connection. No
files are opened, no network is touched, and the types come back from the very engine
that will execute the query, so analysis cannot drift from execution.

The cost is that the schema is DuckDB's opinion rather than Iceberg's. Two knock-on
effects, both recorded in `compat/divergence.md`:

  * **Nullability is lost.** DuckDB has no notion of a non-nullable expression, so
    every field comes back `nullable = true` even where Iceberg marked it required.
    Phase 2's schema binding (Iceberg -> `MappingSchema`) is what fixes this.
  * **`timestamp` vs `timestamptz`** survives, because pyiceberg's Arrow schema
    carries the zone and DuckDB preserves it.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa

from icetl.errors import AnalysisException, UnsupportedFeatureError
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

    from icetl.plan.builder import ScanSource

__all__ = ["PlanAnalyzer", "arrow_to_datatype", "arrow_to_struct_type"]


# Arrow types with no parameters, matched by identity against the singletons.
_SCALAR_TYPES: list[tuple[pa.DataType, DataType]] = [
    (pa.null(), NullType()),
    (pa.bool_(), BooleanType()),
    (pa.int8(), ByteType()),
    (pa.int16(), ShortType()),
    (pa.int32(), IntegerType()),
    (pa.int64(), LongType()),
    # The reference engine has no unsigned types; each widens to the smallest signed type that
    # holds it, which is what the reference engine's own Arrow reader does.
    (pa.uint8(), ShortType()),
    (pa.uint16(), IntegerType()),
    (pa.uint32(), LongType()),
    (pa.float16(), FloatType()),
    (pa.float32(), FloatType()),
    (pa.float64(), DoubleType()),
    (pa.string(), StringType()),
    (pa.large_string(), StringType()),
    (pa.string_view(), StringType()),
    (pa.binary(), BinaryType()),
    (pa.large_binary(), BinaryType()),
    (pa.binary_view(), BinaryType()),
    (pa.date32(), DateType()),
    (pa.date64(), DateType()),
]


def arrow_to_datatype(arrow_type: pa.DataType) -> DataType:
    """Map one Arrow type onto the reference engine's."""
    for candidate, datatype in _SCALAR_TYPES:
        if arrow_type.equals(candidate):
            return datatype

    if pa.types.is_decimal(arrow_type):
        return DecimalType(arrow_type.precision, arrow_type.scale)
    if pa.types.is_timestamp(arrow_type):
        # A zone means an instant (the reference engine `TIMESTAMP`); none means wall-clock
        # (`TIMESTAMP_NTZ`). This is the same split Iceberg makes between
        # `timestamptz` and `timestamp`.
        return TimestampType() if arrow_type.tz else TimestampNTZType()
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return ArrayType(arrow_to_datatype(arrow_type.value_type), arrow_type.value_field.nullable)
    if pa.types.is_fixed_size_list(arrow_type):
        return ArrayType(arrow_to_datatype(arrow_type.value_type), arrow_type.value_field.nullable)
    if pa.types.is_map(arrow_type):
        return MapType(
            arrow_to_datatype(arrow_type.key_type),
            arrow_to_datatype(arrow_type.item_type),
            arrow_type.item_field.nullable,
        )
    if pa.types.is_struct(arrow_type):
        return StructType(
            [
                StructField(field.name, arrow_to_datatype(field.type), field.nullable)
                for field in arrow_type
            ]
        )
    if pa.types.is_uint64(arrow_type):
        # Would need the reference engine's DecimalType(20, 0) to stay lossless. Iceberg cannot
        # produce one, so raising beats silently truncating.
        raise UnsupportedFeatureError(
            "Reading a uint64 column (there is no unsigned 64-bit type)", phase="Phase 3"
        )
    raise UnsupportedFeatureError(
        f"Mapping the Arrow type {arrow_type} to the reference engine", phase="Phase 3"
    )


def arrow_to_struct_type(schema: pa.Schema) -> StructType:
    """Map a whole Arrow schema onto a `StructType`."""
    return StructType(
        [StructField(field.name, arrow_to_datatype(field.type), field.nullable) for field in schema]
    )


class PlanAnalyzer:
    """Binds plans against zero-row stand-ins for their source tables.

    Owns its own in-memory DuckDB connection, deliberately separate from the
    execution engine's: it never loads httpfs, never sees a credential, and never
    opens a file, so analysis stays offline even when execution does not.
    """

    def __init__(self) -> None:
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._registered: set[str] = set()
        self._lock = threading.Lock()

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        if self._connection is None:
            with self._lock:
                if self._connection is None:
                    self._connection = duckdb.connect(":memory:")
        return self._connection

    def register(self, source: ScanSource) -> None:
        """Register a zero-row Arrow view of `source`, once per session."""
        if source.view in self._registered:
            return
        arrow_schema = source.resolved.table.schema().as_arrow()
        self.connection.register(source.view, arrow_schema.empty_table())
        self._registered.add(source.view)

    def register_view(self, name: str, value: Any) -> None:
        """Register an arbitrary relation under `name`, replacing any existing one.

        `register` above is for scan sources; this is for the frames a session
        materialises, whose schema comes from the data rather than from a catalog.
        """
        self.connection.register(name, value)
        self._registered.add(name)

    def unregister_view(self, name: str) -> None:
        if name in self._registered:
            self.connection.unregister(name)
            self._registered.discard(name)

    def analyze(self, sql: str, sources: Mapping[str, ScanSource]) -> StructType:
        """Return the output schema of `sql`, or raise `AnalysisException`.

        `sql` must already have its sources substituted for their analysis views.
        """
        for source in sources.values():
            self.register(source)
        # LIMIT 0 is belt and braces: the sources are already empty, but an aggregate
        # over nothing still produces a row, and we only ever want the schema.
        wrapped = f"SELECT * FROM ({sql}) AS icetl_analysis LIMIT 0"
        try:
            result = self.connection.execute(wrapped).to_arrow_table()
        except duckdb.Error as exc:
            raise AnalysisException(_analysis_message(exc)) from exc
        return arrow_to_struct_type(result.schema)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._registered.clear()


def _analysis_message(exc: duckdb.Error) -> str:
    """Turn a DuckDB binder complaint into something a user recognises.

    DuckDB prefixes its own error class and often appends a `LINE 1: ...` excerpt of
    our generated SQL, which is noise to someone who wrote a DataFrame call. The
    first line carries the actual complaint.
    """
    first_line = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    return first_line.removeprefix("Binder Error: ").removeprefix("Catalog Error: ").strip()

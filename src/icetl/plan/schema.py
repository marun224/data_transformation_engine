"""Schema binding: PyIceberg's `Schema` as something sqlglot's optimizer can use.

This is the keystone of PLAN.md 3.1. Every rule downstream depends on it:

  * `qualify` cannot expand `SELECT *` into a column list without knowing the
    columns, and without that expansion **projection pushdown does nothing**;
  * `pushdown_predicates` cannot decide which scan a conjunct belongs to without
    knowing which table owns each column;
  * the scan planner has nothing to hand PyIceberg as `selected_fields`.

So a wrong or missing binding does not produce a wrong answer -- it silently
produces *no optimisation*, which is the failure mode this module's tests are
written to catch.

The types are spelled in DuckDB's dialect because DuckDB is what ultimately runs
the query: binding to any other dialect would let the optimizer reason about a
type the engine does not have. Nested types are spelled out in full rather than
collapsed to something vague, because `qualify` uses them to resolve `s.field`
references inside structs.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from pyiceberg import types as ice
from sqlglot.schema import MappingSchema

from icetl.errors import UnsupportedFeatureError
from icetl.plan.builder import source_table

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyiceberg.schema import Schema as IcebergSchema

    from icetl.plan.builder import ScanSource

__all__ = ["SchemaBinder", "iceberg_columns", "iceberg_to_duckdb_type"]

# The dialect every bound type is spelled in, and the one the optimizer runs under.
DIALECT = "duckdb"

# Iceberg primitives that map to a fixed DuckDB type name.
_PRIMITIVES: dict[type[ice.IcebergType], str] = {
    ice.BooleanType: "BOOLEAN",
    ice.IntegerType: "INT",
    ice.LongType: "BIGINT",
    ice.FloatType: "FLOAT",
    ice.DoubleType: "DOUBLE",
    ice.DateType: "DATE",
    ice.TimeType: "TIME",
    ice.TimestampType: "TIMESTAMP",
    ice.TimestamptzType: "TIMESTAMPTZ",
    ice.StringType: "VARCHAR",
    ice.UUIDType: "UUID",
    ice.BinaryType: "BLOB",
}


def _quote(name: str) -> str:
    """Quote a struct field name for a DuckDB type literal."""
    return '"' + name.replace('"', '""') + '"'


def iceberg_to_duckdb_type(iceberg_type: ice.IcebergType) -> str:
    """Spell one Iceberg type as DuckDB SQL.

    Iceberg's `timestamp` / `timestamptz` split is preserved rather than flattened:
    it is the same distinction the reference engine draws between `TIMESTAMP_NTZ` and `TIMESTAMP`,
    and losing it here would lose it everywhere downstream.
    """
    fixed = _PRIMITIVES.get(type(iceberg_type))
    if fixed is not None:
        return fixed

    # `FixedType` and `DecimalType` carry parameters, so they cannot be table-driven.
    if isinstance(iceberg_type, ice.FixedType):
        return "BLOB"
    if isinstance(iceberg_type, ice.DecimalType):
        return f"DECIMAL({iceberg_type.precision}, {iceberg_type.scale})"
    if isinstance(iceberg_type, ice.ListType):
        return f"{iceberg_to_duckdb_type(iceberg_type.element_type)}[]"
    if isinstance(iceberg_type, ice.MapType):
        key = iceberg_to_duckdb_type(iceberg_type.key_type)
        value = iceberg_to_duckdb_type(iceberg_type.value_type)
        return f"MAP({key}, {value})"
    if isinstance(iceberg_type, ice.StructType):
        fields = ", ".join(
            f"{_quote(field.name)} {iceberg_to_duckdb_type(field.field_type)}"
            for field in iceberg_type.fields
        )
        return f"STRUCT({fields})"

    raise UnsupportedFeatureError(
        f"Binding the Iceberg type {iceberg_type} for the optimizer", phase="Phase 6"
    )


def iceberg_columns(schema: IcebergSchema) -> dict[str, str]:
    """A table's top-level columns as `{name: duckdb type}`."""
    return {field.name: iceberg_to_duckdb_type(field.field_type) for field in schema.fields}


class SchemaBinder:
    """Builds and caches the `MappingSchema` the optimizer binds against.

    Cached per table reference for the session's lifetime, keyed by the reference as
    the plan spells it (`fx.wide`) *and* the Iceberg schema id, so a table that
    evolves under a long-running session rebinds instead of going stale -- which is
    the DDL invalidation 3.1 asks for, without needing anyone to remember to call it.
    """

    def __init__(self) -> None:
        self._cache: dict[str, tuple[int, dict[str, str]]] = {}
        self._lock = threading.Lock()

    def columns_for(self, source: ScanSource) -> dict[str, str]:
        schema = source.resolved.table.schema()
        cached = self._cache.get(source.key)
        if cached is not None and cached[0] == schema.schema_id:
            return cached[1]
        columns = iceberg_columns(schema)
        with self._lock:
            self._cache[source.key] = (schema.schema_id, columns)
        return columns

    def bind(self, sources: Mapping[str, ScanSource]) -> MappingSchema:
        """A `MappingSchema` covering every source in `sources`.

        Each table is added under the exact reference the plan spells, so a plan
        saying `FROM fx.wide` and one saying `FROM wide` both resolve.
        """
        schema = MappingSchema(dialect=DIALECT)
        for key, source in sources.items():
            schema.add_table(source_table(key), self.columns_for(source), dialect=DIALECT)
        return schema

    def invalidate(self, key: str | None = None) -> None:
        """Drop one binding, or all of them, after DDL."""
        with self._lock:
            if key is None:
                self._cache.clear()
            else:
                self._cache.pop(key, None)

"""Schema binding -- PLAN.md 3.1 says this gets tested first and hardest.

Every optimizer rule reads the binding, and a wrong binding does not raise: it
silently produces no optimisation, or worse, a `SELECT *` that expands to the wrong
column list. So the type mapping is checked exhaustively in both the primitive and
the nested directions, and the cache is checked to actually invalidate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pyiceberg import types as ice
from pyiceberg.schema import Schema

from icetl.errors import UnsupportedFeatureError
from icetl.plan.builder import source_table
from icetl.plan.schema import DIALECT, SchemaBinder, iceberg_columns, iceberg_to_duckdb_type

if TYPE_CHECKING:
    from icetl.plan.builder import ScanSource


class TestPrimitiveTypes:
    @pytest.mark.parametrize(
        ("iceberg_type", "expected"),
        [
            (ice.BooleanType(), "BOOLEAN"),
            (ice.IntegerType(), "INT"),
            (ice.LongType(), "BIGINT"),
            (ice.FloatType(), "FLOAT"),
            (ice.DoubleType(), "DOUBLE"),
            (ice.DateType(), "DATE"),
            (ice.TimeType(), "TIME"),
            (ice.StringType(), "VARCHAR"),
            (ice.UUIDType(), "UUID"),
            (ice.BinaryType(), "BLOB"),
            (ice.FixedType(16), "BLOB"),
            (ice.DecimalType(18, 4), "DECIMAL(18, 4)"),
        ],
    )
    def test_each_primitive_maps(self, iceberg_type: ice.IcebergType, expected: str) -> None:
        assert iceberg_to_duckdb_type(iceberg_type) == expected

    def test_the_timestamp_zone_distinction_survives(self) -> None:
        """Iceberg's timestamp/timestamptz split is Spark's TIMESTAMP_NTZ/TIMESTAMP one.

        Flattening the two here would lose it for every rule downstream, and the loss
        would only show up as a silently wrong hour.
        """
        assert iceberg_to_duckdb_type(ice.TimestampType()) == "TIMESTAMP"
        assert iceberg_to_duckdb_type(ice.TimestamptzType()) == "TIMESTAMPTZ"


class TestNestedTypes:
    def test_list(self) -> None:
        listed = ice.ListType(element_id=2, element=ice.StringType(), element_required=False)
        assert iceberg_to_duckdb_type(listed) == "VARCHAR[]"

    def test_map(self) -> None:
        mapped = ice.MapType(
            key_id=2, key_type=ice.StringType(), value_id=3, value_type=ice.LongType()
        )
        assert iceberg_to_duckdb_type(mapped) == "MAP(VARCHAR, BIGINT)"

    def test_struct_names_its_fields(self) -> None:
        struct = ice.StructType(
            ice.NestedField(2, "name", ice.StringType(), required=False),
            ice.NestedField(3, "age", ice.LongType(), required=False),
        )
        assert iceberg_to_duckdb_type(struct) == 'STRUCT("name" VARCHAR, "age" BIGINT)'

    def test_a_struct_field_named_like_a_keyword_is_quoted(self) -> None:
        struct = ice.StructType(ice.NestedField(2, "select", ice.LongType(), required=False))
        assert iceberg_to_duckdb_type(struct) == 'STRUCT("select" BIGINT)'

    def test_nesting_composes(self) -> None:
        inner = ice.StructType(ice.NestedField(3, "x", ice.LongType(), required=False))
        listed = ice.ListType(element_id=2, element=inner, element_required=False)
        assert iceberg_to_duckdb_type(listed) == 'STRUCT("x" BIGINT)[]'

    def test_an_unmappable_type_names_the_phase_that_owns_it(self) -> None:
        class Unknown(ice.IcebergType):
            pass

        with pytest.raises(UnsupportedFeatureError):
            iceberg_to_duckdb_type(Unknown())


class TestColumnMapping:
    def test_top_level_columns_only(self) -> None:
        """sqlglot resolves struct fields from the column's own type, not the table's."""
        schema = Schema(
            ice.NestedField(1, "id", ice.LongType(), required=False),
            ice.NestedField(
                2,
                "person",
                ice.StructType(ice.NestedField(3, "name", ice.StringType(), required=False)),
                required=False,
            ),
        )
        assert iceberg_columns(schema) == {"id": "BIGINT", "person": 'STRUCT("name" VARCHAR)'}


class TestBinder:
    def test_binds_every_source_under_the_reference_the_plan_spells(
        self, sources: dict[str, ScanSource]
    ) -> None:
        source = sources["fx.plain"]
        bound = SchemaBinder().bind({source.key: source})
        names = bound.column_names(source_table("fx.plain"), dialect=DIALECT)
        assert names == ["id", "vendor", "amount"]

    def test_the_wide_table_binds_all_200_columns(self, sources: dict[str, ScanSource]) -> None:
        """`SELECT *` expansion -- and therefore all projection pushdown -- needs these."""
        source = sources["fx.wide"]
        bound = SchemaBinder().bind({source.key: source})
        assert len(bound.column_names(source_table("fx.wide"), dialect=DIALECT)) == 200

    def test_a_nested_column_binds_as_a_struct(self, sources: dict[str, ScanSource]) -> None:
        source = sources["fx.nested"]
        columns = SchemaBinder().columns_for(source)
        assert columns["person"].startswith("STRUCT(")
        assert columns["tags"] == "VARCHAR[]"
        assert columns["scores"] == "MAP(VARCHAR, BIGINT)"

    def test_columns_are_cached_per_schema_id(self, sources: dict[str, ScanSource]) -> None:
        binder = SchemaBinder()
        source = sources["fx.plain"]
        assert binder.columns_for(source) is binder.columns_for(source)

    def test_invalidate_drops_one_binding(self, sources: dict[str, ScanSource]) -> None:
        binder = SchemaBinder()
        source = sources["fx.plain"]
        first = binder.columns_for(source)
        binder.invalidate(source.key)
        assert binder.columns_for(source) is not first

    def test_invalidate_with_no_key_drops_everything(self, sources: dict[str, ScanSource]) -> None:
        binder = SchemaBinder()
        source = sources["fx.plain"]
        first = binder.columns_for(source)
        binder.invalidate()
        assert binder.columns_for(source) is not first

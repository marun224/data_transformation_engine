"""The DataFrame API, end to end against the local fixture catalog.

Real Iceberg metadata, real parquet files, real DuckDB -- no network. These are the
tests that would catch a break in the compile pipeline as a whole.
"""

from __future__ import annotations

import pytest
from pyiceberg.catalog.sql import SqlCatalog

from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    EngineValueError,
    TableNotFoundError,
    UnsupportedFeatureError,
)
from icetl.sql import DataFrame
from icetl.sql import functions as F
from icetl.sql.session import Session
from icetl.types import (
    ArrayType,
    DoubleType,
    LongType,
    MapType,
    Row,
    StringType,
    StructField,
    StructType,
)


class TestTableAndSchema:
    def test_table_reads_the_iceberg_schema(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.columns == ["id", "vendor", "amount"]
        assert df.dtypes == [("id", "bigint"), ("vendor", "string"), ("amount", "double")]

    def test_schema_is_a_struct_type(self, session: Session) -> None:
        assert session.table("fx.plain").schema == StructType(
            [
                StructField("id", LongType(), True),
                StructField("vendor", StringType(), True),
                StructField("amount", DoubleType(), True),
            ]
        )

    def test_schema_is_cached(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.schema is df.schema

    def test_default_namespace_lets_the_name_stand_alone(self, session: Session) -> None:
        # `local_settings` sets the default namespace to `fx`.
        assert session.table("plain").count() == 5

    def test_missing_table_fails_at_table_not_at_the_action(self, session: Session) -> None:
        with pytest.raises(TableNotFoundError):
            session.table("fx.does_not_exist")

    def test_print_schema(self, session: Session, capsys: pytest.CaptureFixture[str]) -> None:
        session.table("fx.plain").printSchema()
        assert capsys.readouterr().out == (
            "root\n"
            " |-- id: long (nullable = true)\n"
            " |-- vendor: string (nullable = true)\n"
            " |-- amount: double (nullable = true)\n"
        )

    def test_repr_lists_the_columns_and_types(self, session: Session) -> None:
        assert (
            repr(session.table("fx.plain"))
            == "DataFrame[id: bigint, vendor: string, amount: double]"
        )

    def test_nested_types_survive_the_round_trip(self, session: Session) -> None:
        schema = session.table("fx.nested").schema
        types = {field.name: field.dataType for field in schema.fields}
        assert types["person"] == StructType(
            [StructField("name", StringType(), True), StructField("age", LongType(), True)]
        )
        assert types["tags"] == ArrayType(StringType(), True)
        assert types["scores"] == MapType(StringType(), LongType(), True)


class TestActions:
    def test_count(self, session: Session) -> None:
        assert session.table("fx.plain").count() == 5
        assert session.table("fx.partitioned").count() == 12

    def test_collect_returns_named_rows(self, session: Session) -> None:
        rows = session.table("fx.plain").filter(F.col("id") == 1).collect()
        assert rows == [Row(id=1, vendor="a", amount=10.0)]
        assert rows[0].vendor == "a"

    def test_nulls_come_back_as_none(self, session: Session) -> None:
        row = session.table("fx.plain").filter(F.col("id") == 4).collect()[0]
        assert row.amount is None

    def test_take_and_first_and_head(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") <= 2)
        assert len(df.take(1)) == 1
        first = df.first()
        assert first is not None and first.id in (1, 2)
        assert isinstance(df.head(), Row)
        assert len(df.take(2)) == 2

    def test_first_on_an_empty_result_is_none(self, session: Session) -> None:
        assert session.table("fx.plain").filter(F.col("id") == 999).first() is None

    def test_to_arrow_and_to_pandas(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.toArrow().num_rows == 5
        frame = df.toPandas()
        assert list(frame.columns) == ["id", "vendor", "amount"]
        # The reference engine's dtype contract for strings.
        assert frame["vendor"].dtype == object
        assert frame["vendor"].tolist()[4] is None

    def test_nested_values_take_the_reference_python_shapes(self, session: Session) -> None:
        rows = {row.id: row for row in session.table("fx.nested").collect()}
        assert rows[1].person == Row(name="ada", age=36)
        assert rows[1].tags == ["x", "y"]
        assert rows[2].scores == {"b": 2, "c": 3}


class TestSelect:
    def test_select_by_name(self, session: Session) -> None:
        assert session.table("fx.plain").select("id", "vendor").columns == ["id", "vendor"]

    def test_select_accepts_a_list(self, session: Session) -> None:
        assert session.table("fx.plain").select(["id", "vendor"]).columns == ["id", "vendor"]

    def test_select_star(self, session: Session) -> None:
        assert session.table("fx.plain").select("*").columns == ["id", "vendor", "amount"]

    def test_expression_gets_a_generated_name(self, session: Session) -> None:
        df = session.table("fx.plain").select(F.col("amount") * 2)
        assert df.columns == ["(amount * 2)"]

    def test_alias_overrides_the_generated_name(self, session: Session) -> None:
        df = session.table("fx.plain").select((F.col("amount") * 2).alias("doubled"))
        assert df.columns == ["doubled"]

    def test_select_expr(self, session: Session) -> None:
        df = session.table("fx.plain").selectExpr("id", "amount * 2 AS doubled")
        assert df.columns == ["id", "doubled"]

    def test_unknown_column_fails_at_select_not_at_collect(self, session: Session) -> None:
        with pytest.raises(AnalysisException, match="typo"):
            session.table("fx.plain").select("typo")

    def test_select_requires_a_column(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            session.table("fx.plain").select()

    def test_select_rejects_non_columns(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.table("fx.plain").select(1)

    def test_column_names_are_matched_case_insensitively(self, session: Session) -> None:
        # Case-insensitive resolution, as the reference engine does by default.
        assert session.table("fx.plain").select("ID").columns == ["id"]


class TestFilter:
    def test_filter_with_a_column(self, session: Session) -> None:
        assert session.table("fx.plain").filter(F.col("id") > 3).count() == 2

    def test_filter_with_a_sql_string(self, session: Session) -> None:
        assert session.table("fx.plain").filter("id > 3").count() == 2

    def test_where_is_filter(self, session: Session) -> None:
        assert DataFrame.where is DataFrame.filter

    def test_chained_filters_are_anded(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") > 1).filter(F.col("id") < 4)
        assert sorted(row.id for row in df.collect()) == [2, 3]

    def test_filter_after_a_projection_sees_the_new_names(self, session: Session) -> None:
        # This is the case that forces nesting: the alias only exists downstream.
        df = (
            session.table("fx.plain")
            .select(F.col("amount").alias("value"))
            .filter(F.col("value") > 20)
        )
        assert df.count() == 3

    def test_filter_rejects_wrong_types(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.table("fx.plain").filter(1)  # type: ignore[arg-type]


class TestLimit:
    def test_limit(self, session: Session) -> None:
        assert session.table("fx.plain").limit(2).count() == 2

    def test_the_tighter_limit_wins_when_chained(self, session: Session) -> None:
        assert session.table("fx.plain").limit(2).limit(4).count() == 2
        assert session.table("fx.plain").limit(4).limit(2).count() == 2

    def test_limit_zero(self, session: Session) -> None:
        assert session.table("fx.plain").limit(0).collect() == []

    def test_limit_rejects_bad_input(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            session.table("fx.plain").limit(-1)
        with pytest.raises(EngineTypeError):
            session.table("fx.plain").limit("2")  # type: ignore[arg-type]


class TestWithColumn:
    def test_adds_a_column(self, session: Session) -> None:
        df = session.table("fx.plain").withColumn("doubled", F.col("amount") * 2)
        assert df.columns == ["id", "vendor", "amount", "doubled"]
        assert df.filter(F.col("id") == 1).collect()[0].doubled == 20.0

    def test_replaces_an_existing_column_in_place(self, session: Session) -> None:
        df = session.table("fx.plain").withColumn("amount", F.col("amount") * 2)
        assert df.columns == ["id", "vendor", "amount"]
        assert df.filter(F.col("id") == 1).collect()[0].amount == 20.0

    def test_replacement_is_case_insensitive_and_takes_the_new_spelling(
        self, session: Session
    ) -> None:
        df = session.table("fx.plain").withColumn("AMOUNT", F.lit(1))
        assert df.columns == ["id", "vendor", "AMOUNT"]

    def test_rejects_a_plain_value(self, session: Session) -> None:
        with pytest.raises(EngineTypeError, match=r"F\.lit"):
            session.table("fx.plain").withColumn("x", 1)  # type: ignore[arg-type]


class TestRenameAndDrop:
    def test_rename(self, session: Session) -> None:
        df = session.table("fx.plain").withColumnRenamed("vendor", "supplier")
        assert df.columns == ["id", "supplier", "amount"]
        first = df.first()
        assert first is not None and first.supplier == "a"

    def test_renaming_an_unknown_column_is_a_no_op(self, session: Session) -> None:
        # The reference engine does not raise here, and neither do we.
        df = session.table("fx.plain")
        assert df.withColumnRenamed("nope", "x").columns == df.columns

    def test_drop(self, session: Session) -> None:
        assert session.table("fx.plain").drop("vendor").columns == ["id", "amount"]

    def test_drop_several_and_ignore_unknown_names(self, session: Session) -> None:
        assert session.table("fx.plain").drop("vendor", "nope").columns == ["id", "amount"]

    def test_drop_accepts_columns(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.drop(df.vendor).columns == ["id", "amount"]

    def test_dropping_everything_is_refused_rather_than_silently_odd(
        self, session: Session
    ) -> None:
        df = session.table("fx.plain")
        with pytest.raises(UnsupportedFeatureError):
            df.drop(*df.columns)


class TestColumnAccess:
    def test_attribute_access(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.filter(df.amount > 20).count() == 3

    def test_unknown_attribute_raises_attribute_error(self, session: Session) -> None:
        with pytest.raises(AttributeError, match="not a column"):
            _ = session.table("fx.plain").nope

    def test_item_access_by_name_and_position(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert repr(df["vendor"]) == "Column<'vendor'>"
        assert repr(df[0]) == "Column<'id'>"

    def test_item_access_with_a_predicate_filters(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df[df.id < 3].count() == 2

    def test_item_access_with_a_list_selects(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df[["id", "vendor"]].columns == ["id", "vendor"]

    def test_unknown_name_raises_analysis_exception(self, session: Session) -> None:
        with pytest.raises(AnalysisException):
            session.table("fx.plain")["nope"]


class TestAliasAndQualification:
    def test_alias_lets_columns_be_qualified(self, session: Session) -> None:
        df = session.table("fx.plain").alias("p")
        assert df.select(F.col("p.id")).columns == ["id"]

    def test_alias_rejects_non_strings(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.table("fx.plain").alias(1)  # type: ignore[arg-type]


class TestShow:
    def test_show_matches_the_reference_layout(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.table("fx.plain").filter(F.col("id") == 1).show()
        assert capsys.readouterr().out == (
            "+---+------+------+\n"
            "| id|vendor|amount|\n"
            "+---+------+------+\n"
            "|  1|     a|  10.0|\n"
            "+---+------+------+\n"
        )

    def test_show_reports_when_rows_were_dropped(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.table("fx.plain").show(2)
        assert "only showing top 2 rows" in capsys.readouterr().out

    def test_show_does_not_report_when_everything_fitted(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.table("fx.plain").show(20)
        assert "only showing" not in capsys.readouterr().out

    def test_show_vertical(self, session: Session, capsys: pytest.CaptureFixture[str]) -> None:
        session.table("fx.plain").filter(F.col("id") == 1).show(vertical=True)
        assert capsys.readouterr().out == ("-RECORD 0\n     id | 1\n vendor | a\n amount | 10.0\n")

    def test_show_rejects_bad_arguments(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.table("fx.plain").show("2")  # type: ignore[arg-type]
        with pytest.raises(EngineTypeError):
            session.table("fx.plain").show(truncate="yes")  # type: ignore[arg-type]


class TestExplain:
    def test_explain_shows_the_sql_and_the_scan(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.table("fx.plain").filter(F.col("id") > 1).explain()
        out = capsys.readouterr().out
        assert "== Physical Plan (DuckDB SQL) ==" in out
        assert "read_parquet" in out.lower()
        assert "test.fx.plain: 1 of 1 file(s)" in out
        # The filter reached PyIceberg, and is reported rather than merely applied.
        assert "pushed filters: id > 1" in out

    def test_extended_adds_the_logical_plan_and_schema(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.table("fx.plain").explain(extended=True)
        out = capsys.readouterr().out
        assert "== Logical Plan (icetl) ==" in out
        assert "== Analysed Schema ==" in out

    def test_mode_string_is_accepted_positionally(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session.table("fx.plain").explain("extended")
        assert "== Logical Plan (icetl) ==" in capsys.readouterr().out

    def test_unknown_mode_is_refused(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError):
            session.table("fx.plain").explain(mode="cost")

    def test_file_paths_travel_as_a_parameter_not_inline(
        self, session: Session, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Keeps explain() readable for a table that resolves to thousands of files.
        session.table("fx.wide").explain()
        assert "$icetl_src_" in capsys.readouterr().out


class TestPlanShape:
    """The extend-vs-nest guards. Getting these wrong would give wrong answers."""

    def plan_of(self, df: DataFrame) -> str:
        """The *logical* plan, which is where the nest-vs-extend decision shows.

        Asserted on the plan rather than the generated SQL because the Phase 2
        optimizer flattens the nesting away again -- correctly, and after the
        decision under test has already been made.
        """
        return df._plan.sql(dialect="duckdb")

    def test_filter_and_select_merge_into_the_base_scan(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.col("id") > 1).select("id")
        assert "(SELECT" not in self.plan_of(df)

    def test_a_projection_forces_a_later_filter_to_nest(self, session: Session) -> None:
        df = session.table("fx.plain").select(F.col("id").alias("k")).filter(F.col("k") > 1)
        assert "(SELECT" in self.plan_of(df)

    def test_a_second_limit_nests(self, session: Session) -> None:
        df = session.table("fx.plain").limit(2).limit(4)
        assert "(SELECT" in self.plan_of(df)

    def test_a_filter_after_a_limit_nests(self, session: Session) -> None:
        # `LIMIT 2 WHERE ...` would filter before limiting, which is the wrong answer.
        df = session.table("fx.plain").limit(2).filter(F.col("id") > 0)
        assert "(SELECT" in self.plan_of(df)
        assert df.count() == 2


class TestEmptyTable:
    def test_a_table_with_no_data_files_reads_as_no_rows(
        self, session: Session, catalog: SqlCatalog
    ) -> None:
        from pyiceberg.schema import Schema
        from pyiceberg.types import LongType as IcebergLong
        from pyiceberg.types import NestedField

        catalog.create_table(
            "fx.empty_df", schema=Schema(NestedField(1, "id", IcebergLong(), required=False))
        )
        df = session.table("fx.empty_df")
        assert df.columns == ["id"]
        assert df.count() == 0
        assert df.collect() == []


class TestDeferredSurface:
    @pytest.mark.parametrize("attribute", ["rdd"])
    def test_later_phases_name_themselves(self, session: Session, attribute: str) -> None:
        df = session.table("fx.plain")
        with pytest.raises(UnsupportedFeatureError):
            getattr(df, attribute)

    def test_write_arrived_in_phase_7(self, session: Session) -> None:
        """It raised `phase="Phase 7"` until the write path landed; see test_write."""
        from icetl.sql.writer import DataFrameWriter

        assert isinstance(session.table("fx.plain").write, DataFrameWriter)

    def test_print_schema_takes_a_level_since_phase_6(self, session: Session) -> None:
        """Deferred until there were nested schemas worth truncating; see test_complex_types."""
        session.table("fx.plain").printSchema(level=2)

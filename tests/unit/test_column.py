"""`Column` and the `F.*` constructors: the SQL they build and the names they take.

These are pure-tree tests -- no catalog, no DuckDB. Generated SQL is asserted in the
DuckDB dialect, because that is the one that actually runs.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlglot import exp

from icetl.errors import ParseException, PySparkTypeError, PySparkValueError
from icetl.sql import functions as F
from icetl.sql.column import Column
from icetl.types import DecimalType, LongType


def sql(column: Column) -> str:
    """The DuckDB SQL for a column expression."""
    return column._expression.sql(dialect="duckdb")


class TestConstructors:
    def test_col_and_column_are_the_same_function(self) -> None:
        assert F.column is F.col

    def test_col_parses_qualifiers(self) -> None:
        assert sql(F.col("amount")) == "amount"
        assert sql(F.col("t.amount")) == "t.amount"

    def test_col_star_forms(self) -> None:
        assert sql(F.col("*")) == "*"
        assert sql(F.col("t.*")) == "t.*"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, "1"), (1.5, "1.5"), ("x", "'x'"), (True, "TRUE"), (False, "FALSE"), (None, "NULL")],
    )
    def test_lit_renders_python_values(self, value: object, expected: str) -> None:
        assert sql(F.lit(value)) == expected

    def test_lit_distinguishes_bool_from_int(self) -> None:
        # bool is an int subclass; `lit(True)` must not become `1`.
        assert sql(F.lit(True)) == "TRUE"
        assert sql(F.lit(1)) == "1"

    def test_lit_passes_columns_through(self) -> None:
        column = F.col("a")
        assert F.lit(column) is column

    def test_lit_rejects_unsupported_types(self) -> None:
        with pytest.raises(PySparkTypeError, match="Cannot make a literal"):
            F.lit(object())

    def test_expr_parses_spark_sql(self) -> None:
        assert sql(F.expr("amount * 2")) == "amount * 2"

    def test_expr_rejects_unparseable_text(self) -> None:
        with pytest.raises(ParseException):
            F.expr("SELECT FROM WHERE ***")

    def test_constructors_reject_wrong_types(self) -> None:
        with pytest.raises(PySparkTypeError):
            F.col(1)  # type: ignore[arg-type]
        with pytest.raises(PySparkTypeError):
            F.expr(1)  # type: ignore[arg-type]


class TestOperators:
    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda c: c == 1, "a = 1"),
            (lambda c: c != 1, "a <> 1"),
            (lambda c: c < 1, "a < 1"),
            (lambda c: c <= 1, "a <= 1"),
            (lambda c: c > 1, "a > 1"),
            (lambda c: c >= 1, "a >= 1"),
            (lambda c: c + 1, "a + 1"),
            (lambda c: c - 1, "a - 1"),
            (lambda c: c * 2, "a * 2"),
            (lambda c: c % 2, "a % 2"),
            (lambda c: -c, "-a"),
            (lambda c: 1 + c, "1 + a"),
            (lambda c: 1 - c, "1 - a"),
            (lambda c: 10 % c, "10 % a"),
        ],
    )
    def test_operator_sql(self, build: Callable[[Column], Column], expected: str) -> None:
        assert sql(build(F.col("a"))) == expected

    def test_string_operand_is_a_literal_not_a_column(self) -> None:
        # The asymmetry with `select("b")`, where a string names a column.
        assert sql(F.col("a") == "b") == "a = 'b'"

    def test_logical_operators_parenthesise_correctly(self) -> None:
        combined = (F.col("a") > 1) & (F.col("b") < 2)
        assert sql(combined) == "a > 1 AND b < 2"
        assert sql((F.col("a") > 1) | (F.col("b") < 2)) == "a > 1 OR b < 2"
        assert sql(~(F.col("a") > 1)) == "NOT (a > 1)"

    def test_column_cannot_be_used_as_a_bool(self) -> None:
        with pytest.raises(PySparkValueError, match="Cannot convert a Column into a bool"):
            bool(F.col("a") == 1)

    def test_operands_are_copied_so_a_column_can_be_reused(self) -> None:
        base = F.col("a")
        first, second = base + 1, base * 2
        assert sql(first) == "a + 1"
        assert sql(second) == "a * 2"
        assert sql(base) == "a"


class TestDivision:
    """Spark's `/` yields NULL on a zero divisor; DuckDB 1.5 would yield `inf`."""

    def test_division_by_a_column_is_guarded(self) -> None:
        assert sql(F.col("a") / F.col("b")) == "a / NULLIF(b, 0)"

    def test_reverse_division_is_guarded(self) -> None:
        assert sql(1 / F.col("b")) == "1 / NULLIF(b, 0)"

    def test_a_nonzero_literal_divisor_needs_no_guard(self) -> None:
        assert sql(F.col("a") / 2) == "a / 2"
        assert sql(F.col("a") / 2.5) == "a / 2.5"

    def test_a_zero_literal_divisor_keeps_the_guard(self) -> None:
        assert sql(F.col("a") / 0) == "a / NULLIF(0, 0)"

    def test_the_sql_surface_produces_the_same_node(self) -> None:
        # P1: `spark.sql` and the DataFrame API must agree, and they do because
        # sqlglot's Spark parser sets the same `safe` flag we do.
        assert sql(F.expr("a / b")) == sql(F.col("a") / F.col("b"))


class TestNullsAndMembership:
    def test_null_checks(self) -> None:
        assert sql(F.col("a").isNull()) == "a IS NULL"
        assert sql(F.col("a").isNotNull()) == "NOT a IS NULL"

    def test_isin_accepts_varargs_and_iterables(self) -> None:
        assert sql(F.col("a").isin(1, 2)) == "a IN (1, 2)"
        assert sql(F.col("a").isin([1, 2])) == "a IN (1, 2)"

    def test_between(self) -> None:
        assert sql(F.col("a").between(1, 5)) == "a BETWEEN 1 AND 5"


class TestNamingAndCasting:
    def test_alias_sets_the_output_name(self) -> None:
        aliased = F.col("a").alias("b")
        assert sql(aliased) == 'a AS "b"'
        assert aliased._output_name == "b"

    def test_name_is_an_alias_for_alias(self) -> None:
        assert Column.name is Column.alias

    def test_alias_rejects_multiple_names_until_phase_6(self) -> None:
        with pytest.raises(PySparkValueError, match="exactly one name"):
            F.col("a").alias("x", "y")

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            (lambda: F.col("amount"), "amount"),
            (lambda: F.col("t.amount"), "amount"),
            (lambda: F.col("a") * 2, "(a * 2)"),
            (lambda: F.col("a") == 1, "(a = 1)"),
            (lambda: (F.col("a") > 1) & (F.col("b") > 2), "((a > 1) AND (b > 2))"),
            (lambda: F.lit(1), "1"),
            (lambda: F.lit("x"), "x"),
        ],
    )
    def test_spark_output_names(self, expression: Callable[[], Column], expected: str) -> None:
        assert expression()._output_name == expected

    def test_cast_accepts_a_datatype_or_ddl_string(self) -> None:
        assert sql(F.col("a").cast(LongType())) == "CAST(a AS BIGINT)"
        assert sql(F.col("a").cast("int")) == "CAST(a AS INT)"
        assert sql(F.col("a").cast(DecimalType(10, 2))) == "CAST(a AS DECIMAL(10, 2))"

    def test_astype_is_cast(self) -> None:
        assert Column.astype is Column.cast

    def test_cast_rejects_nonsense(self) -> None:
        with pytest.raises(PySparkTypeError):
            F.col("a").cast(1)  # type: ignore[arg-type]


class TestRepr:
    def test_repr_matches_pyspark_shape(self) -> None:
        assert repr(F.col("a")) == "Column<'a'>"
        assert repr(F.col("a") + 1) == "Column<'a + 1'>"

    def test_repr_strips_the_quoting_we_add_internally(self) -> None:
        quoted = Column(exp.column("order", quoted=True))
        assert repr(quoted) == "Column<'order'>"

    def test_column_rejects_non_expressions(self) -> None:
        with pytest.raises(PySparkTypeError, match="sqlglot expression"):
            Column("a")  # type: ignore[arg-type]

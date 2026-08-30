"""The §3.5 conformance rules against real rows, on both surfaces.

`tests/unit/test_conformance.py` pins the rewrite; this pins the *answer*. Each test
states what the reference engine returns, because that is the specification (P5) -- and every one of
these would pass just as quietly with the wrong rows if it only checked the SQL.

`fx.plain` is the fixture throughout: five rows, with a NULL in `vendor` and a NULL
in `amount`, which is what makes the null-ordering and null-propagation cases real.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import QueryExecutionException
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql import Session


class TestNullOrdering:
    """The reference engine sorts NULLs first ascending and last descending. DuckDB does neither."""

    def test_ascending_puts_nulls_first(self, session: Session) -> None:
        rows = session.sql("SELECT vendor FROM fx.plain ORDER BY vendor").collect()
        assert [row[0] for row in rows] == [None, "a", "a", "b", "c"]

    def test_descending_puts_nulls_last(self, session: Session) -> None:
        rows = session.sql("SELECT vendor FROM fx.plain ORDER BY vendor DESC").collect()
        assert [row[0] for row in rows] == ["c", "b", "a", "a", None]

    def test_an_explicit_nulls_last_is_honoured(self, session: Session) -> None:
        rows = session.sql("SELECT vendor FROM fx.plain ORDER BY vendor NULLS LAST").collect()
        assert [row[0] for row in rows] == ["a", "a", "b", "c", None]

    def test_a_second_sort_key_is_ordered_too(self, session: Session) -> None:
        rows = session.sql("SELECT vendor, amount FROM fx.plain ORDER BY vendor, amount").collect()
        assert [row[0] for row in rows] == [None, "a", "a", "b", "c"]


class TestCasting:
    """The reference engine's default cast yields NULL on failure. DuckDB's raises."""

    def test_an_impossible_cast_is_null_not_an_error(self, session: Session) -> None:
        rows = session.sql("SELECT CAST('abc' AS INT) AS v FROM fx.plain LIMIT 1").collect()
        assert rows[0]["v"] is None

    def test_the_dataframe_surface_agrees(self, session: Session) -> None:
        """P1: both surfaces must give the same answer, and they build different nodes."""
        df = session.table("fx.plain").select(F.lit("abc").cast("int").alias("v")).limit(1)
        assert df.collect()[0]["v"] is None

    def test_a_possible_cast_still_works(self, session: Session) -> None:
        rows = session.sql("SELECT CAST('42' AS INT) AS v FROM fx.plain LIMIT 1").collect()
        assert rows[0]["v"] == 42

    def test_casting_a_real_column_is_unaffected(self, session: Session) -> None:
        rows = session.sql("SELECT CAST(id AS STRING) AS v FROM fx.plain ORDER BY id").collect()
        assert [row["v"] for row in rows] == ["1", "2", "3", "4", "5"]

    def test_ansi_mode_raises_instead(self, ansi_session: Session) -> None:
        """`icetl.ansiMode=true` opts into strict-cast mode."""
        with pytest.raises(QueryExecutionException):
            ansi_session.sql("SELECT CAST('abc' AS INT) AS v FROM fx.plain LIMIT 1").collect()

    def test_ansi_mode_leaves_a_valid_cast_alone(self, ansi_session: Session) -> None:
        rows = ansi_session.sql("SELECT CAST('42' AS INT) AS v FROM fx.plain LIMIT 1").collect()
        assert rows[0]["v"] == 42

    def test_an_explicit_try_cast_survives_ansi_mode(self, ansi_session: Session) -> None:
        rows = ansi_session.sql(
            "SELECT TRY_CAST('abc' AS INT) AS v FROM fx.plain LIMIT 1"
        ).collect()
        assert rows[0]["v"] is None


class TestDivision:
    """The reference engine returns NULL for `x / 0`; DuckDB returns `inf`."""

    def test_sql_surface(self, session: Session) -> None:
        assert session.sql("SELECT 1 / 0 AS v FROM fx.plain LIMIT 1").collect()[0]["v"] is None

    def test_dataframe_surface(self, session: Session) -> None:
        df = session.table("fx.plain").select((F.lit(1) / F.lit(0)).alias("v")).limit(1)
        assert df.collect()[0]["v"] is None

    def test_dividing_by_a_column_that_holds_zero(self, session: Session) -> None:
        """The literal case is easy; the guard has to survive a runtime zero too."""
        rows = session.sql("SELECT 10 / (id - 3) AS v FROM fx.plain ORDER BY id").collect()
        assert rows[2]["v"] is None, "id = 3 makes the divisor zero"
        assert rows[0]["v"] == -5.0

    def test_a_decimal_literal_divisor_is_null_on_both_surfaces(self, session: Session) -> None:
        """`1.0 / 0.0`, not `1 / 0`.

        The integer spelling never exercised the optimizer's constant folder --
        sqlglot's `simplify` declines to fold integer division, calling it unsafe
        across engines. The decimal spelling *is* folded, in Python's `decimal`, which
        raises `DivisionByZero` from inside the rule. That escaped `optimize_plan`'s
        guard, so a plain `SELECT 1.0 / 0.0` crashed rather than answering NULL.
        """
        assert session.sql("SELECT 1.0 / 0.0 AS v FROM fx.plain LIMIT 1").collect()[0]["v"] is None
        df = session.table("fx.plain").select((F.lit(1.0) / F.lit(0.0)).alias("v")).limit(1)
        assert df.collect()[0]["v"] is None

    def test_ordinary_division_is_unharmed(self, session: Session) -> None:
        rows = session.sql("SELECT 10 / 4 AS v FROM fx.plain LIMIT 1").collect()
        assert rows[0]["v"] == 2.5


class TestNullSafeEquality:
    def test_null_equals_null(self, session: Session) -> None:
        """`<=>` is true for two NULLs, where `=` is NULL."""
        rows = session.sql(
            "SELECT vendor <=> NULL AS safe, vendor = NULL AS plain "
            "FROM fx.plain WHERE vendor IS NULL"
        ).collect()
        assert rows[0]["safe"] is True
        assert rows[0]["plain"] is None

"""The reference engine-vs-DuckDB conformance rules of PLAN.md 3.5, at the tree level.

P5 makes the reference engine the spec, so each rule here is a claim about what DuckDB would
otherwise get wrong. `tests/fixture/test_conformance.py` runs the same rules against
real data; these pin the rewrite itself, including the cases it must *not* touch.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from icetl.plan.builder import as_expression
from icetl.sql.conformance import apply_compat_semantics


def conform(sql: str, *, ansi_mode: bool = False) -> exp.Expression:
    """Parse then conform -- the real pipeline order, stopping at the tree."""
    parsed = as_expression(sqlglot.parse_one(sql, read="spark"))
    return apply_compat_semantics(parsed, ansi_mode=ansi_mode)


def compat_sql(sql: str, *, ansi_mode: bool = False) -> str:
    """As `conform`, rendered as DuckDB.

    Note that sqlglot *elides* a `NULLS LAST` that matches the target dialect's own
    default, so the generated text is not where to assert null ordering -- the tree
    is. `nulls_of` exists for that.
    """
    return conform(sql, ansi_mode=ansi_mode).sql(dialect="duckdb")


def nulls_of(sql: str) -> list[bool | None]:
    """`nulls_first` for each ORDER BY term, in order."""
    return [node.args.get("nulls_first") for node in conform(sql).find_all(exp.Ordered)]


class TestNullOrdering:
    """The reference engine: ASC nulls first, DESC nulls last. DuckDB 1.5: nulls last for both.

    So ASC is a genuine divergence and DESC merely happens to agree. Both are made
    explicit, because "happens to agree" is a property of this DuckDB release rather
    than a promise, and the cost of being explicit is nothing.
    """

    def test_ascending_gets_nulls_first(self) -> None:
        assert nulls_of("SELECT x FROM t ORDER BY x") == [True]

    def test_ascending_says_so_in_the_generated_sql(self) -> None:
        """The divergent direction, so this one is visible in the SQL: without it
        DuckDB would sort the NULLs to the other end."""
        assert "NULLS FIRST" in compat_sql("SELECT x FROM t ORDER BY x")

    def test_descending_gets_nulls_last(self) -> None:
        assert nulls_of("SELECT x FROM t ORDER BY x DESC") == [False]

    def test_an_explicit_choice_is_left_alone(self) -> None:
        """The user asking for the non-default order must get the non-default
        order."""
        assert nulls_of("SELECT x FROM t ORDER BY x NULLS LAST") == [False]
        assert nulls_of("SELECT x FROM t ORDER BY x DESC NULLS FIRST") == [True]

    def test_every_term_of_a_multi_key_sort_is_covered(self) -> None:
        assert nulls_of("SELECT x FROM t ORDER BY a ASC, b DESC") == [True, False]

    def test_ordering_inside_a_subquery_is_covered(self) -> None:
        assert nulls_of("SELECT * FROM (SELECT x FROM t ORDER BY x) AS q") == [True]


class TestCasting:
    """The reference engine's default (non-ANSI) cast yields NULL on failure; DuckDB's raises.

    The two surfaces arrive spelled differently -- sqlglot's the reference engine dialect already
    parses `CAST` into `TryCast`, while `Column.cast` builds a plain `Cast` -- which
    is exactly why this is a tree pass rather than a rule inside `Column`.
    """

    def test_a_cast_is_lenient_by_default(self) -> None:
        assert "TRY_CAST" in compat_sql("SELECT CAST(x AS INT) FROM t")

    def test_a_dataframe_style_cast_is_lenient_too(self) -> None:
        """`Column.cast` builds `exp.Cast`; it must end up the same as the SQL form."""
        plan = as_expression(sqlglot.parse_one("SELECT x FROM t", read="duckdb"))
        plan.set("expressions", [exp.cast(exp.column("x"), "INT", dialect="duckdb")])
        out = apply_compat_semantics(plan).sql(dialect="duckdb")
        assert "TRY_CAST" in out

    def test_ansi_mode_makes_a_cast_strict(self) -> None:
        out = compat_sql("SELECT CAST(x AS INT) FROM t", ansi_mode=True)
        assert "TRY_CAST" not in out
        assert "CAST" in out

    def test_an_explicit_try_cast_stays_lenient_even_under_ansi(self) -> None:
        """`try_cast(...)` means "be lenient because I asked", not "because the reference is",
        so `ansi_mode` has no business changing it."""
        assert "TRY_CAST" in compat_sql("SELECT TRY_CAST(x AS INT) FROM t", ansi_mode=True)

    def test_a_nested_cast_is_covered(self) -> None:
        out = compat_sql("SELECT CAST(CAST(x AS STRING) AS INT) FROM t")
        assert out.count("TRY_CAST") == 2

    def test_the_target_type_survives(self) -> None:
        assert "DECIMAL(10, 2)" in compat_sql("SELECT CAST(x AS DECIMAL(10,2)) FROM t")


class TestAlreadyMatching:
    """Behaviours DuckDB gets right unasked.

    Tested rather than implemented: a rewrite that changes nothing costs readability
    in every generated query. Each of these is a claim that could stop being true in
    a future DuckDB, and the suite is where we would find out.
    """

    def test_division_by_zero_is_guarded_by_the_parser(self) -> None:
        """The reference returns NULL for `1/0`; DuckDB returns `inf`. sqlglot's spark dialect
        already emits the `NULLIF` guard, so no rule of ours is needed."""
        assert "NULLIF" in compat_sql("SELECT 1 / 0")

    def test_null_safe_equality_translates(self) -> None:
        assert "IS NOT DISTINCT FROM" in compat_sql("SELECT a <=> b FROM t")

    def test_modulo_by_zero_needs_no_guard(self) -> None:
        """DuckDB already returns NULL for `x % 0`, as the reference engine does."""
        assert "NULLIF" not in compat_sql("SELECT x % 0 FROM t")


class TestPassThrough:
    def test_a_plan_with_nothing_to_fix_is_unchanged(self) -> None:
        before = "SELECT a, b FROM t WHERE a > 1"
        assert compat_sql(before) == sqlglot.parse_one(before, read="spark").sql(dialect="duckdb")

    def test_the_input_tree_is_not_mutated(self) -> None:
        parsed = as_expression(
            sqlglot.parse_one("SELECT CAST(x AS INT) FROM t ORDER BY x", read="spark")
        )
        before = parsed.sql(dialect="spark")
        apply_compat_semantics(parsed)
        assert parsed.sql(dialect="spark") == before

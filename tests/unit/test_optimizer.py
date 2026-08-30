"""The optimizer pipeline and, above all, the guarantee that makes it safe to run.

`qualify` renames unaliased projections to `_col_0`. For the reference engine that is a wrong
answer -- the column is called `sum(amount)` and scripts index on it -- so
`optimize_plan` is contractually required either to restore the names analysis
computed, or to hand back the original plan untouched. The tests here are mostly
about that contract; the pruning it enables is tested end to end in
`tests/fixture/test_pushdown.py`.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.schema import MappingSchema

from icetl.plan.builder import as_expression
from icetl.plan.optimizer import RULES, optimize_plan

SCHEMA = MappingSchema(
    {
        "fx": {
            "plain": {"id": "BIGINT", "vendor": "VARCHAR", "amount": "DOUBLE"},
            "wide": {"id": "BIGINT", **{f"col_{i:03d}": "DOUBLE" for i in range(1, 200)}},
        }
    },
    dialect="duckdb",
)


def parse(sql: str) -> exp.Expression:
    return as_expression(sqlglot.parse_one(sql, read="duckdb"))


def names(expression: exp.Expression) -> list[str]:
    """The output column names of a query node."""
    assert isinstance(expression, exp.Query)
    return list(expression.named_selects)


class TestFlatteningAndPruning:
    def test_the_dataframe_nesting_collapses_to_one_scan(self) -> None:
        """A chain of DataFrame calls arrives as layers of `SELECT * FROM (...)`.
        Nothing can be pushed down until that is flattened."""
        plan = parse(
            'SELECT "id", "col_001" FROM (SELECT * FROM "fx"."wide") AS _q1 WHERE "id" < 10'
        )
        result = optimize_plan(plan, SCHEMA, ["id", "col_001"])

        assert result.applied
        assert result.stages == RULES
        assert not list(result.optimized.find_all(exp.Subquery))

    def test_star_expansion_narrows_to_the_columns_actually_used(self) -> None:
        plan = parse('SELECT "id" FROM (SELECT * FROM "fx"."wide") AS _q1')
        result = optimize_plan(plan, SCHEMA, ["id"])

        columns = {column.name for column in result.optimized.find_all(exp.Column)}
        assert columns == {"id"}

    def test_a_filter_added_after_a_projection_reaches_the_scan(self) -> None:
        plan = parse(
            'SELECT * FROM (SELECT "id", "col_001" FROM "fx"."wide") AS _q1 WHERE "col_001" > 3'
        )
        result = optimize_plan(plan, SCHEMA, ["id", "col_001"])
        assert result.applied
        # The predicate now sits on the SELECT that reads the table directly.
        assert isinstance(result.optimized, exp.Select)
        assert result.optimized.args.get("where") is not None


class TestOutputNames:
    def test_an_unaliased_expression_keeps_the_name_analysis_promised(self) -> None:
        """Left alone, `qualify` would call this `_col_0`."""
        plan = parse('SELECT "amount" + 1 FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["(amount + 1)"])

        assert result.applied
        assert names(result.optimized) == ["(amount + 1)"]

    def test_an_aggregate_keeps_its_generated_name(self) -> None:
        plan = parse('SELECT "vendor", SUM("amount") FROM "fx"."plain" GROUP BY "vendor"')
        result = optimize_plan(plan, SCHEMA, ["vendor", "sum(amount)"])

        assert result.applied
        assert names(result.optimized) == ["vendor", "sum(amount)"]

    def test_original_case_is_restored_even_though_duckdb_normalises(self) -> None:
        schema = MappingSchema({"fx": {"t": {"VendorID": "BIGINT"}}}, dialect="duckdb")
        plan = parse('SELECT * FROM "fx"."t"')
        result = optimize_plan(plan, schema, ["VendorID"])

        assert result.applied
        assert names(result.optimized) == ["VendorID"]

    def test_a_plan_whose_names_cannot_be_reconciled_is_discarded(self) -> None:
        """Belt and braces: if the optimized tree does not have the promised number
        of output columns, running it would be a wrong answer, so it is not run."""
        plan = parse('SELECT "id", "vendor" FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["id"])

        assert not result.applied
        assert result.optimized is plan
        assert "could not be reconciled" in (result.note or "")

    def test_names_already_correct_are_left_alone(self) -> None:
        plan = parse('SELECT "id" FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["id"])
        assert result.applied
        assert names(result.optimized) == ["id"]


class TestFallback:
    def test_an_unbindable_table_falls_back_to_the_original_plan(self) -> None:
        """A plan we cannot bind still has to run -- reading more than it needed to,
        but returning the same rows."""
        plan = parse('SELECT "x" FROM "fx"."not_bound"')
        result = optimize_plan(plan, SCHEMA, ["x"])

        assert not result.applied
        assert result.optimized is plan
        assert result.note

    def test_the_note_says_which_rule_stopped_it(self) -> None:
        plan = parse('SELECT "x" FROM "fx"."not_bound"')
        result = optimize_plan(plan, SCHEMA, ["x"])
        assert "qualify" in (result.note or "")

    def test_the_original_plan_is_never_mutated(self) -> None:
        plan = parse('SELECT * FROM "fx"."plain"')
        before = plan.sql(dialect="duckdb")
        optimize_plan(plan, SCHEMA, ["id", "vendor", "amount"])
        assert plan.sql(dialect="duckdb") == before


class TestUnions:
    def test_a_union_is_optimized_only_when_its_names_already_match(self) -> None:
        """A UNION's output names come from its first branch several levels down.
        Rather than reach in and rewrite that, the pipeline declines."""
        plan = parse('SELECT "id" FROM "fx"."plain" UNION ALL SELECT "id" FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["id"])
        assert names(result.optimized) == ["id"]

    def test_a_union_needing_a_rename_is_discarded(self) -> None:
        plan = parse(
            'SELECT "amount" + 1 FROM "fx"."plain" UNION ALL SELECT "amount" FROM "fx"."plain"'
        )
        result = optimize_plan(plan, SCHEMA, ["(amount + 1)"])
        assert not result.applied


class TestArithmeticInsideARule:
    """`simplify` constant-folds literal arithmetic, and the arithmetic can fail.

    The pipeline's promise is that a rule which cannot handle a plan costs only that
    rule. An `ArithmeticError` raised *by* a rule broke that promise, because the
    guard listed only `OptimizeError`, `KeyError`, `ValueError` and `TypeError`.
    """

    def test_a_literal_division_by_zero_degrades_instead_of_raising(self) -> None:
        """Python's `decimal` raises `DivisionByZero` when `simplify` folds this."""
        plan = parse('SELECT 1.0 / 0.0 AS "v" FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["v"])
        assert "DivisionByZero" in (result.note or "")

    def test_the_rules_before_the_failure_are_kept(self) -> None:
        """ "Costs only that rule" is literal: the stages that ran before `simplify`
        are still applied, and the plan they produced is the one that runs."""
        plan = parse('SELECT 1.0 / 0.0 AS "v" FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["v"])
        assert result.applied
        assert "simplify" not in result.stages
        assert "qualify" in result.stages

    def test_integer_division_by_zero_is_left_alone_by_the_folder(self) -> None:
        """The spelling that hid the bug: `simplify` refuses to fold integer division
        at all, so this path never reached the failing arithmetic."""
        plan = parse('SELECT 1 / 0 AS "v" FROM "fx"."plain"')
        result = optimize_plan(plan, SCHEMA, ["v"])
        assert result.applied

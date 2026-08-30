"""Source nodes, source keys, and the substitution pass that rewrites them."""

from __future__ import annotations

import pytest
import sqlglot
from sqlglot import exp

from icetl.plan.builder import (
    ScanSource,
    as_expression,
    collect_source_keys,
    source_key,
    source_table,
    substitute_sources,
    wrap_as_subquery,
)


def parse(sql: str, read: str = "spark") -> exp.Expression:
    """Parse, narrowed to `Expression`.

    sqlglot 30 annotates `parse_one` with its looser `Expr` trait base, so the
    narrowing has to happen somewhere; once, here, keeps the tests uncluttered.
    """
    return as_expression(sqlglot.parse_one(sql, read=read))


def fake_source(key: str, view: str = "v0") -> ScanSource:
    """A ScanSource with no live table behind it -- substitution never touches one."""
    return ScanSource(key=key, resolved=None, view=view)  # type: ignore[arg-type]


class TestSourceTable:
    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            ("plain", '"plain"'),
            ("fx.plain", '"fx"."plain"'),
            ("cat.fx.plain", '"cat"."fx"."plain"'),
        ],
    )
    def test_builds_a_quoted_reference(self, reference: str, expected: str) -> None:
        assert source_table(reference).sql(dialect="duckdb") == expected

    def test_round_trips_through_source_key(
        self,
    ) -> None:
        for reference in ("plain", "fx.plain", "cat.fx.plain"):
            assert source_key(source_table(reference)) == reference

    def test_rejects_more_than_three_parts(self) -> None:
        with pytest.raises(ValueError, match="more than three parts"):
            source_table("a.b.c.d")

    def test_quoting_survives_a_keyword_name(self) -> None:
        assert source_table("fx.order").sql(dialect="duckdb") == '"fx"."order"'


class TestSourceKey:
    def test_ignores_the_alias(self) -> None:
        parsed = parse("SELECT * FROM nyc.trips AS t", "spark")
        table = parsed.find(exp.Table)
        assert table is not None
        assert source_key(table) == "nyc.trips"

    def test_parsed_and_built_nodes_agree(self) -> None:
        # The DataFrame API and the SQL parser must produce the same key, or
        # substitution would miss one of the two surfaces.
        parsed = parse("SELECT * FROM fx.plain", "spark").find(exp.Table)
        assert parsed is not None
        assert source_key(parsed) == source_key(source_table("fx.plain"))


class TestCollectSourceKeys:
    def test_finds_tables_in_first_seen_order(self) -> None:
        query = parse("SELECT * FROM b.t2, a.t1, b.t2", "spark")
        assert collect_source_keys(query) == ["b.t2", "a.t1"]

    def test_skips_cte_names(self) -> None:
        # A CTE reference parses as a table; sending it to the catalog would fail.
        query = parse("WITH c AS (SELECT * FROM fx.plain) SELECT * FROM c", "spark")
        assert collect_source_keys(query) == ["fx.plain"]

    def test_skips_table_functions(self) -> None:
        query = parse("SELECT * FROM read_parquet('x.parquet')", "duckdb")
        assert collect_source_keys(query) == []

    def test_finds_tables_inside_subqueries(self) -> None:
        query = parse("SELECT * FROM (SELECT * FROM fx.plain) AS q", "spark")
        assert collect_source_keys(query) == ["fx.plain"]

    def test_returns_nothing_for_a_sourceless_query(self) -> None:
        assert collect_source_keys(parse("SELECT 1")) == []


class TestSubstituteSources:
    def test_replaces_a_matching_table(self) -> None:
        plan = exp.select(exp.Star()).from_(source_table("fx.plain"))
        sources = {"fx.plain": fake_source("fx.plain")}
        out = substitute_sources(plan, sources, lambda s: exp.to_identifier(s.view, quoted=True))
        assert out.sql(dialect="duckdb") == 'SELECT * FROM "v0" AS "plain"'

    def test_invents_an_alias_from_the_table_name(self) -> None:
        # So `plain.amount` still resolves after the swap.
        plan = parse("SELECT plain.amount FROM fx.plain", "spark")
        out = substitute_sources(
            plan, {"fx.plain": fake_source("fx.plain")}, lambda s: exp.to_identifier(s.view)
        )
        assert 'AS "plain"' in out.sql(dialect="duckdb")

    def test_preserves_an_explicit_alias(self) -> None:
        plan = parse("SELECT t.amount FROM fx.plain AS t", "spark")
        out = substitute_sources(
            plan, {"fx.plain": fake_source("fx.plain")}, lambda s: exp.to_identifier(s.view)
        )
        sql = out.sql(dialect="duckdb")
        assert 'AS "t"' in sql
        assert "t.amount" in sql

    def test_leaves_unknown_tables_alone(self) -> None:
        plan = exp.select(exp.Star()).from_(source_table("other.thing"))
        out = substitute_sources(plan, {"fx.plain": fake_source("fx.plain")}, lambda s: exp.null())
        assert out.sql(dialect="duckdb") == 'SELECT * FROM "other"."thing"'

    def test_does_not_mutate_the_input(self) -> None:
        plan = exp.select(exp.Star()).from_(source_table("fx.plain"))
        before = plan.sql(dialect="duckdb")
        substitute_sources(
            plan, {"fx.plain": fake_source("fx.plain")}, lambda s: exp.to_identifier(s.view)
        )
        assert plan.sql(dialect="duckdb") == before

    def test_reaches_tables_nested_in_subqueries(self) -> None:
        plan = parse("SELECT * FROM (SELECT * FROM fx.plain) AS q", "spark")
        out = substitute_sources(
            plan, {"fx.plain": fake_source("fx.plain")}, lambda s: exp.to_identifier(s.view)
        )
        assert "fx" not in out.sql(dialect="duckdb")


class TestWrapAsSubquery:
    def test_wraps_and_aliases(self) -> None:
        plan = exp.select(exp.column("a")).from_(exp.to_table("t"))
        assert (
            wrap_as_subquery(plan, "_q1").sql(dialect="duckdb")
            == "SELECT * FROM (SELECT a FROM t) AS _q1"
        )

    def test_the_wrapper_can_take_a_new_projection(self) -> None:
        wrapped = wrap_as_subquery(exp.select(exp.column("a")).from_(exp.to_table("t")), "_q1")
        wrapped.set("expressions", [exp.column("a")])
        assert wrapped.sql(dialect="duckdb") == "SELECT a FROM (SELECT a FROM t) AS _q1"

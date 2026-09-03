"""Phase 10: a generator in a select list is not a scalar expression.

Phase 6 puts `unnest` in the *select list* rather than the FROM clause, which is what
makes `explode` free and lets repeated copies correlate (`sql/generators.py`). The
price is that a set-returning function sits exactly where every optimizer rule
expects a scalar, and three of sqlglot's rules moved it as if it were one:

    pushdown_projections   dropped it when nothing referenced it, so `count()` over
                           an exploded frame returned the *table's* row count
    pushdown_predicates    inlined it into a WHERE, which DuckDB rejects outright
    merge_subqueries       merged its defining scope away and took it with it

The first was a **silent wrong answer**, reachable since Phase 6 through the most
ordinary call there is -- `df.select(F.explode(...)).count()` -- and found while
building Phase 10's metadata count. FINDINGS.md 1.13.

Every case here asserts on a **row count that is arithmetic**: `fx.plain` holds 5
rows, so a three-element explode over it is 15 and nothing else. A count that agrees
with `len(collect())` is not enough on its own -- both go through the same optimizer,
and the bug moved them together for `count()` alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.plan.cardinality import contains_generator
from icetl.plan.optimizer import UNSAFE_WITH_GENERATORS, optimize_plan
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session

#: `fx.plain` has 5 rows; each explodes into 3. Written out because the whole point
#: is to assert against a number derived from the fixture, not from another query.
PLAIN_ROWS = 5
ELEMENTS = 3
EXPLODED = PLAIN_ROWS * ELEMENTS


@pytest.fixture
def exploded(session: Session) -> DataFrame:
    """`fx.plain` with a three-element array exploded over it: 15 rows."""
    return session.table("fx.plain").select(
        F.col("id"), F.explode(F.array(F.lit(1), F.lit(2), F.lit(3))).alias("n")
    )


class TestCountOverAGenerator:
    """The silent one. `count()` returned 5 where `collect()` returned 15."""

    def test_the_count_is_the_exploded_count(self, exploded: DataFrame) -> None:
        assert exploded.count() == EXPLODED

    def test_it_is_not_the_source_table_s_count(self, exploded: DataFrame) -> None:
        """The exact wrong answer, named so a regression cannot look like a near miss."""
        assert exploded.count() != PLAIN_ROWS

    def test_count_and_collect_agree(self, exploded: DataFrame) -> None:
        assert exploded.count() == len(exploded.collect())

    def test_the_sql_spelling_agrees_too(self, session: Session) -> None:
        rows = session.sql(
            "SELECT count(*) FROM (SELECT explode(array(1, 2, 3)) AS n FROM fx.plain)"
        ).collect()
        assert rows[0][0] == EXPLODED

    def test_an_outer_scope_referencing_nothing_still_sees_every_row(
        self, session: Session
    ) -> None:
        """The shape that made it silent: the generated column is never referenced."""
        rows = session.sql(
            "SELECT 1 AS k FROM (SELECT explode(array(1, 2, 3)) AS n FROM fx.plain)"
        ).collect()
        assert len(rows) == EXPLODED

    def test_exploding_a_real_column_counts_its_elements(self, session: Session) -> None:
        """`fx.nested` holds `['x', 'y']` and `[]`, so explode gives two rows, not two."""
        frame = session.table("fx.nested").select(F.explode(F.col("tags")).alias("tag"))
        assert frame.count() == 2
        assert sorted(row[0] for row in frame.collect()) == ["x", "y"]

    def test_explode_outer_keeps_the_empty_row(self, session: Session) -> None:
        """The count has to follow the generator's own semantics, not the table's."""
        frame = session.table("fx.nested").select(F.explode_outer(F.col("tags")).alias("tag"))
        assert frame.count() == 3


class TestFilteringOverAGenerator:
    """The loud one. `pushdown_predicates` inlined the generator into a WHERE."""

    def test_a_filter_over_a_generated_column_runs(self, exploded: DataFrame) -> None:
        """`n > 1` keeps 2 and 3 from each of the five rows."""
        assert len(exploded.filter(F.col("n") > 1).collect()) == PLAIN_ROWS * 2

    def test_it_keeps_the_right_values(self, exploded: DataFrame) -> None:
        rows = exploded.filter(F.col("n") > 1).collect()
        assert sorted({row["n"] for row in rows}) == [2, 3]

    def test_a_filter_on_the_source_column_still_prunes(
        self, session: Session, exploded: DataFrame
    ) -> None:
        """The table's own columns are unaffected: `id = 3` is still one row, times 3."""
        assert len(exploded.filter(F.col("id") == 3).collect()) == ELEMENTS

    def test_an_aggregate_over_generated_rows_sums_them_all(self, session: Session) -> None:
        rows = session.sql(
            "SELECT sum(n) FROM (SELECT explode(array(1, 2, 3)) AS n FROM fx.plain)"
        ).collect()
        assert rows[0][0] == (1 + 2 + 3) * PLAIN_ROWS

    def test_distinct_over_generated_rows(self, exploded: DataFrame) -> None:
        assert exploded.select("n").distinct().count() == ELEMENTS


class TestTheGuardItself:
    def test_a_plan_with_a_generator_is_recognised(self, exploded: DataFrame) -> None:
        assert contains_generator(exploded._plan)

    def test_an_ordinary_plan_is_not(self, session: Session) -> None:
        assert not contains_generator(session.table("fx.plain").filter("id > 2")._plan)

    def test_the_unsafe_rules_are_skipped_and_said_to_be(
        self, session: Session, exploded: DataFrame
    ) -> None:
        schema = session._binder.bind(exploded._sources)
        result = optimize_plan(exploded._plan, schema, exploded.columns)
        assert result.applied
        assert not (set(result.stages) & UNSAFE_WITH_GENERATORS)
        assert result.note is not None and "row-generating" in result.note

    def test_an_ordinary_plan_still_gets_every_rule(self, session: Session) -> None:
        """The guard costs nothing on the queries that do not explode anything."""
        frame = session.table("fx.plain").filter("id > 2").select("id")
        schema = session._binder.bind(frame._sources)
        result = optimize_plan(frame._plan, schema, frame.columns)
        assert set(result.stages) >= UNSAFE_WITH_GENERATORS

    def test_scan_pruning_survives_the_skipped_rules(self, session: Session) -> None:
        """What the guard costs is flattening, not pruning.

        `plan/pushdown.py` reads the qualified tree itself rather than relying on
        sqlglot's pushdown rules, so an exploded query over a partitioned table still
        prunes to the one file its filter selects.
        """
        filtered = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        filtered = filtered.select(F.col("id"), F.explode(F.array(F.lit(1), F.lit(2))).alias("n"))
        compiled = session._compile(filtered._plan, filtered._sources, filtered.columns)
        scan = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert scan.files_scanned == 1
        # That one file holds 4 rows, each exploded into 2.
        assert len(filtered.collect()) == 8

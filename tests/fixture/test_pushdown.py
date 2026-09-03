"""Phase 2 end to end: pruning you can observe, and the correctness it must not cost.

PLAN.md's "done when" for this phase is that pruning becomes *visible* -- `explain()`
reporting "3 of 4096 files, 12 of 214 columns" rather than a number with nothing to
compare it against. That is what the first class here checks.

The rest guard the ways pushdown could have bought speed by returning the wrong rows:
a filter pruned out of the SQL (3.2), a renamed column read by name (3.4), and a
delete file `read_parquet` cannot see. Every one of them is a silent failure in the
wild, so each is asserted on the data, not just on the plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import UnsupportedFeatureError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.exec.scan_planner import ScanPlan
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def compile_of(df: DataFrame) -> object:
    """The compiled plan, which is what `explain()` renders."""
    return df._session._compile(df._plan, df._sources, df.columns)


def scan_of(df: DataFrame) -> ScanPlan:
    """The single scan a one-table query plans."""
    scans = compile_of(df).scans  # type: ignore[attr-defined]
    assert len(scans) == 1
    return scans[0]


def explain_text(df: DataFrame, *, extended: bool = False) -> str:
    return df._explain_text(verbose=extended)


class TestPruningIsObservable:
    """PLAN.md 4, Phase 2: "pruning is observable"."""

    def test_the_wide_table_reports_columns_scanned_of_columns_available(
        self, session: Session
    ) -> None:
        df = session.table("fx.wide").select("id", "col_001")
        assert "columns: 2 of 200" in explain_text(df)

    def test_the_partitioned_table_reports_files_scanned_of_files_available(
        self, session: Session
    ) -> None:
        df = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        assert "1 of 3 file(s)" in explain_text(df)

    def test_the_pushed_predicate_is_named_readably(self, session: Session) -> None:
        df = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        assert "pushed filters: as_at_date = '2026-08-16'" in explain_text(df)

    def test_extended_mode_shows_the_optimized_tree_and_the_rules(self, session: Session) -> None:
        df = session.table("fx.wide").select("id").filter(F.col("id") < 10)
        out = explain_text(df, extended=True)
        assert "== Optimized Plan ==" in out
        assert "rules: qualify" in out
        assert "pushdown_projections" in out

    def test_an_untranslatable_filter_is_reported_rather_than_hidden(
        self, session: Session
    ) -> None:
        """It still runs in SQL, so the answer is right -- but it bought no pruning,
        which is what someone looking at a slow query needs told."""
        df = session.table("fx.plain").filter(F.expr("upper(vendor) = 'A'"))
        assert "kept in SQL only:" in explain_text(df)


class TestProjectionPushdown:
    def test_only_the_referenced_columns_reach_read_parquet(self, session: Session) -> None:
        """3.6: never `SELECT *` on the 200-column table."""
        df = session.table("fx.wide").select("id", "col_001")
        assert scan_of(df).columns == ("id", "col_001")

    def test_a_column_used_only_in_a_filter_is_still_read(self, session: Session) -> None:
        df = session.table("fx.wide").filter(F.col("col_002") > 0).select("id")
        assert set(scan_of(df).columns) == {"id", "col_002"}

    def test_count_star_reads_one_column_not_two_hundred(self, session: Session) -> None:
        """`count(*)` references no column at all; reading one is the cheapest way to
        get the row count and still have a legal projection."""
        df = session.table("fx.wide")
        assert df.count() == 500

    def test_an_order_by_on_an_output_alias_still_prunes(self, session: Session) -> None:
        """`GROUP BY x ORDER BY x` is the commonest analytic shape there is, and
        `qualify` deliberately leaves the ORDER BY reference unqualified because it
        names the *output* column. Reading that as an unattributable table column
        gave up projection pushdown entirely -- 219 of 219 on the real table."""
        df = session.sql("SELECT id, count(*) AS n FROM fx.wide GROUP BY id ORDER BY id")
        assert scan_of(df).columns == ("id",)
        assert df.columns == ["id", "n"]

    def test_an_aggregate_over_a_filter_reads_only_the_three_columns_involved(
        self, session: Session
    ) -> None:
        df = session.sql(
            "SELECT id, sum(col_001) AS total FROM fx.wide "
            "WHERE col_002 > 0 GROUP BY id ORDER BY id"
        )
        assert set(scan_of(df).columns) == {"id", "col_001", "col_002"}

    def test_selecting_everything_reads_everything(self, session: Session) -> None:
        df = session.table("fx.wide")
        assert len(scan_of(df).columns) == 200

    def test_the_rows_are_unchanged_by_pruning(self, session: Session) -> None:
        rows = session.table("fx.wide").select("id", "col_001").filter(F.col("id") < 3).collect()
        assert sorted(row["id"] for row in rows) == [0, 1, 2]
        assert {row["col_001"] for row in rows} == {2.0}


class TestPredicatePushdown:
    def test_a_partition_predicate_prunes_files(self, session: Session) -> None:
        df = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        assert scan_of(df).files_scanned == 1
        assert scan_of(df).files_total == 3

    def test_the_pruned_query_returns_the_right_rows(self, session: Session) -> None:
        rows = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16").collect()
        assert len(rows) == 4
        assert {row["as_at_date"] for row in rows} == {"2026-08-16"}

    def test_the_filter_is_always_kept_in_the_sql(self, session: Session) -> None:
        """The invariant of 3.2. PyIceberg's pruning is stats-based and therefore
        approximate: a file whose min/max straddles the predicate is kept, and its
        non-matching rows come back. Removing the filter from the SQL because it was
        "already applied" is how that becomes a wrong answer."""
        df = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        sql = compile_of(df).sql  # type: ignore[attr-defined]
        assert "'2026-08-16'" in sql
        assert scan_of(df).pushed_filter is not None

    def test_an_untranslatable_filter_still_filters(self, session: Session) -> None:
        df = session.table("fx.plain").filter(F.expr("upper(vendor) = 'A'"))
        assert scan_of(df).pushed_filter is None
        assert sorted(row["id"] for row in df.collect()) == [1, 3]

    def test_a_partial_and_pushes_the_half_it_understands(self, session: Session) -> None:
        df = session.table("fx.partitioned").filter(
            F.expr("as_at_date = '2026-08-16' AND upper(cast(id as varchar)) <> 'X'")
        )
        scan = scan_of(df)
        assert scan.files_scanned == 1
        assert scan.unpushed_filters
        assert len(df.collect()) == 4

    def test_a_predicate_matching_no_partition_reads_no_files(self, session: Session) -> None:
        df = session.table("fx.partitioned").filter(F.col("as_at_date") == "1999-01-01")
        assert scan_of(df).files_scanned == 0
        assert df.collect() == []

    def test_a_partition_column_reads_its_real_value_not_the_directory_name(
        self, session: Session
    ) -> None:
        """DuckDB auto-enables `hive_partitioning` when it sees `key=value` directories,
        which an Iceberg warehouse is full of. Left on, it synthesises the column from
        the *path* and type-casts it, so this `string` column came back as `DATE`.
        For Iceberg the directory holds the *transformed* value, not the column's, so
        the only correct source is the file."""
        rows = session.table("fx.partitioned").collect()
        assert {type(row["as_at_date"]) for row in rows} == {str}
        assert "2026-08-16" in {row["as_at_date"] for row in rows}

    def test_sql_and_the_dataframe_api_prune_identically(self, session: Session) -> None:
        """P1: one IR, so both surfaces get the same plan and the same pruning."""
        from_api = session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16")
        from_sql = session.sql("SELECT * FROM fx.partitioned WHERE as_at_date = '2026-08-16'")

        assert scan_of(from_api).files_scanned == scan_of(from_sql).files_scanned
        assert scan_of(from_api).pushed_filter == scan_of(from_sql).pushed_filter


class TestRenamedColumns:
    """PLAN.md 3.4 -- the sharp edge, now blunted.

    `tests/fixture/test_fixture_tables.py` pins what the naive `read_parquet` does to
    this table: the two rows written before the rename come back NULL. These check
    that going through icetl does not.
    """

    def test_the_renamed_column_reads_correctly(self, session: Session) -> None:
        values = [row["new_name"] for row in session.table("fx.renamed").collect()]
        assert sorted(values) == ["after-c", "after-d", "before-a", "before-b"]

    def test_no_value_comes_back_null(self, session: Session) -> None:
        """The specific shape of the bug: silence, not an error."""
        values = [row["new_name"] for row in session.table("fx.renamed").collect()]
        assert None not in values

    def test_it_agrees_with_pyiceberg_which_reads_by_field_id(
        self, session: Session, fixtures: dict[str, object]
    ) -> None:
        table = fixtures["renamed"].table  # type: ignore[attr-defined]
        expected = table.scan().to_arrow().column("new_name").to_pylist()
        actual = [row["new_name"] for row in session.table("fx.renamed").collect()]
        assert sorted(actual) == sorted(expected)

    def test_the_files_are_grouped_by_the_names_they_hold(self, session: Session) -> None:
        scan = scan_of(session.table("fx.renamed"))
        assert scan.renamed_columns == ("new_name",)
        assert len(scan.groups) == 2, "one group per stored spelling"

    def test_the_reconciliation_is_reported(self, session: Session) -> None:
        assert "field-id reconciliation: new_name" in explain_text(session.table("fx.renamed"))

    def test_a_table_that_never_renamed_takes_the_fast_path(self, session: Session) -> None:
        """Detection is O(schemas), so it must cost nothing on ordinary tables."""
        scan = scan_of(session.table("fx.plain"))
        assert scan.renamed_columns == ()
        assert len(scan.groups) == 1

    def test_filtering_on_a_renamed_column_still_works(self, session: Session) -> None:
        rows = session.table("fx.renamed").filter(F.col("new_name") == "before-a").collect()
        assert [row["id"] for row in rows] == [1]


class TestCopyOnWriteInvariant:
    """Decision 11: these tables are copy-on-write, so a scan is exactly its data files.

    That is an assumption about the *writers*, not something Iceberg enforces -- the
    table is shared, and another engine could add merge-on-read deletes to one we only
    read. The assumption is therefore asserted, because the failure mode if it breaks
    is the worst kind: `read_parquet` cannot see a delete file, so the deleted rows
    would come back and the query would report success.

    The `mor` fixture exists solely to prove the guard fires. It is the only way to
    build a table with delete files -- PyIceberg's own `delete()` is copy-on-write --
    and a guard with no test is a guard that stops working quietly.
    """

    def test_reading_a_table_with_delete_files_is_refused(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="merge-on-read"):
            session.table("fx.mor").collect()

    def test_the_refusal_says_which_table_and_how_many_files(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match=r"fx\.mor.*1 data file"):
            session.table("fx.mor").count()

    def test_it_is_refused_rather_than_silently_answered_wrongly(
        self, session: Session, fixtures: dict[str, object]
    ) -> None:
        """The point of the guard, stated as the thing it prevents.

        Iceberg says this table has 6 rows; the files on disk hold 8. Anything that
        returns 8 -- or returns anything at all -- has lost the deletes.
        """
        table = fixtures["mor"].table  # type: ignore[attr-defined]
        assert table.scan().to_arrow().num_rows == 6
        assert sum(t.file.record_count for t in table.scan().plan_files()) == 8

        with pytest.raises(UnsupportedFeatureError):
            session.table("fx.mor").toArrow()

    def test_planning_is_refused_too_not_just_execution(self, session: Session) -> None:
        """`explain()` must not print a plan that would be wrong to run."""
        with pytest.raises(UnsupportedFeatureError):
            explain_text(session.table("fx.mor"))

    def test_an_ordinary_table_is_unaffected(self, session: Session) -> None:
        assert len(scan_of(session.table("fx.plain")).groups) == 1


class TestUnbindablePlansStillRun:
    def test_an_aggregate_keeps_its_column_name_through_the_optimizer(
        self, session: Session
    ) -> None:
        df = session.sql("SELECT vendor, count(*) AS n FROM fx.plain GROUP BY vendor")
        assert df.columns == ["vendor", "n"]
        assert sum(row["n"] for row in df.collect()) == 5

    def test_a_nested_table_reads_unchanged(self, session: Session) -> None:
        """Struct, list and map columns bind as real DuckDB types, so `SELECT *`
        expansion has to survive them."""
        rows = session.table("fx.nested").collect()
        assert len(rows) == 2
        assert rows[0]["person"]["name"] == "ada"

    @pytest.mark.parametrize("reference", ["fx.plain", "fx.partitioned", "fx.wide"])
    def test_every_fixture_still_reads_end_to_end(self, session: Session, reference: str) -> None:
        assert session.table(reference).count() > 0


class TestOuterJoinsAreNotPrunedByTheirOwnNullChecks:
    """The anti-join idiom, and the one case where pruning changed the answer.

    `LEFT JOIN b ... WHERE b.id IS NULL` selects the rows where the join found nothing.
    Pushing that conjunct into `b`'s Iceberg scan pruned away every file -- no data file
    holds a NULL id -- so `b` read as empty, *every* left row survived the anti-join, and
    the query returned all of them instead of none.

    Reachable since Phase 4, silent, and it needs two bare table references: with a
    subquery on the right, the conjunct is not in that scope's WHERE and never reaches
    the scan. FINDINGS.md §1.10.
    """

    def test_an_antijoin_over_two_tables_answers_correctly(self, session: Session) -> None:
        """`fx.partitioned` holds ids 0-11, so every id in `fx.plain` matches."""
        rows = session.sql(
            "SELECT a.id FROM fx.plain AS a "
            "LEFT JOIN fx.partitioned AS b ON a.id = b.id "
            "WHERE b.id IS NULL"
        ).collect()
        assert [row[0] for row in rows] == []

    def test_it_finds_the_rows_that_really_are_unmatched(self, session: Session) -> None:
        rows = session.sql(
            "SELECT a.id FROM fx.partitioned AS a "
            "LEFT JOIN fx.plain AS b ON a.id = b.id "
            "WHERE b.id IS NULL"
        ).collect()
        assert sorted(row[0] for row in rows) == [0, 6, 7, 8, 9, 10, 11]

    def test_the_null_padded_side_is_not_pruned(self, session: Session) -> None:
        frame = session.sql(
            "SELECT a.id FROM fx.plain AS a "
            "LEFT JOIN fx.partitioned AS b ON a.id = b.id "
            "WHERE b.id IS NULL"
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        padded = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert padded.pushed_filter is None
        assert padded.files_scanned == 3

    def test_a_null_rejecting_conjunct_still_prunes_the_padded_side(self, session: Session) -> None:
        """Only the conjuncts a row of NULLs could satisfy are held back.

        `b.as_at_date = '2026-08-16'` cannot be true of a manufactured NULL row, so it
        prunes as it always did -- the fix costs nothing on the ordinary outer join.
        """
        frame = session.sql(
            "SELECT a.id FROM fx.plain AS a "
            "LEFT JOIN fx.partitioned AS b ON a.id = b.id "
            "WHERE b.as_at_date = '2026-08-16'"
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        padded = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert padded.pushed_filter is not None
        assert padded.files_scanned == 1

    def test_an_inner_join_prunes_through_its_on_clause(self, session: Session) -> None:
        """The conjunct is in `ON` by the time we see it, and prunes from there.

        `pushdown_predicates` folds a WHERE conjunct over an inner-joined table into
        that join's ON clause. Reading only the scope's WHERE meant the same predicate
        pruned one file as a LEFT JOIN and three -- every file -- as an INNER JOIN.
        `join_predicates` reads the ON clause too, so both spellings now prune alike.
        FINDINGS.md 3.5.
        """
        frame = session.sql(
            "SELECT a.id FROM fx.plain AS a "
            "INNER JOIN fx.partitioned AS b ON a.id = b.id "
            "WHERE b.as_at_date = '2026-08-16'"
        )
        assert sorted(row[0] for row in frame.collect()) == [4, 5]
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        inner = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert inner.pushed_filter == "as_at_date = '2026-08-16'"
        assert inner.files_scanned == 1

    def test_the_two_join_spellings_now_prune_identically(self, session: Session) -> None:
        """The measurement 3.5 recorded, inverted: 3 of 3 against 1 of 3, now 1 and 1."""
        counts = []
        for side in ("INNER", "LEFT"):
            frame = session.sql(
                f"SELECT a.id FROM fx.plain AS a "
                f"{side} JOIN fx.partitioned AS b ON a.id = b.id "
                f"WHERE b.as_at_date = '2026-08-16'"
            )
            compiled = session._compile(frame._plan, frame._sources, frame.columns)
            scan = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
            counts.append(scan.files_scanned)
        assert counts == [1, 1]

    def test_an_on_conjunct_prunes_the_null_padded_side(self, session: Session) -> None:
        """No `is_null_rejecting` gate here, and none is needed.

        A LEFT JOIN's ON clause is applied to rows read from the right table, before
        the join manufactures any NULL row -- so a right-side conjunct there filters
        real rows and prunes safely, which a WHERE conjunct over the same alias does
        not.
        """
        frame = session.sql(
            "SELECT a.id, b.id FROM fx.plain AS a "
            "LEFT JOIN fx.partitioned AS b "
            "ON a.id = b.id AND b.as_at_date = '2026-08-16'"
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        padded = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert padded.files_scanned == 1
        # Every left row survives; only the match is narrowed.
        rows = frame.collect()
        assert len(rows) == 5  # fx.plain holds ids 1-5
        assert sorted(row[1] for row in rows if row[1] is not None) == [4, 5]

    def test_the_preserved_side_is_not_pruned_by_its_own_on_conjunct(
        self, session: Session
    ) -> None:
        """`LEFT JOIN ... ON a.id > 3` keeps every row of `a`, so `a` may not prune.

        The unsafe half of the same fix: on the preserved side an ON conjunct decides
        whether a row *matches*, not whether it is read, and pruning by it would drop
        rows the query must return.
        """
        frame = session.sql(
            "SELECT a.id, b.id FROM fx.partitioned AS a "
            "LEFT JOIN fx.plain AS b ON a.id = b.id AND a.as_at_date = '2026-08-16'"
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        preserved = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert preserved.pushed_filter is None
        assert preserved.files_scanned == 3
        assert len(frame.collect()) == 12

    def test_a_full_join_prunes_neither_side_by_its_on_clause(self, session: Session) -> None:
        """A FULL join preserves both sides, so its ON clause filters neither."""
        frame = session.sql(
            "SELECT a.id, b.id FROM fx.plain AS a "
            "FULL JOIN fx.partitioned AS b "
            "ON a.id = b.id AND b.as_at_date = '2026-08-16'"
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        for scan in compiled.scans:
            assert scan.pushed_filter is None
        both = next(s for s in compiled.scans if s.source.key == "fx.partitioned")
        assert both.files_scanned == 3
        # All 12 partitioned rows appear, matched or null-padded.
        assert len([row for row in frame.collect() if row[1] is not None]) == 12


class TestTheAntiJoinBugReachedThroughARightJoin:
    """The same wrong answer as 1.10, through a spelling the fix for it never saw.

    `null_padded_aliases` read `args["from"]`, which sqlglot 30 spells `from_`, so it
    always saw a query with no FROM clause. Nothing broke for a LEFT JOIN -- the
    padded side is the join's own right, which needs no FROM -- but a RIGHT or FULL
    join null-pads the side that *is* the FROM clause, and that side was therefore
    never held back. `WHERE a.id IS NULL` pruned `a` to zero files, so the anti-join
    matched nothing, and every row came back instead of the unmatched ones.

    Found while building Phase 10; silent, and a wrong answer rather than a slow one.
    FINDINGS.md 1.12.
    """

    ANTI_JOIN = (
        "SELECT b.id FROM fx.plain AS a RIGHT JOIN fx.partitioned AS b "
        "ON a.id = b.id WHERE a.id IS NULL"
    )

    def test_a_right_join_anti_join_answers_correctly(self, session: Session) -> None:
        """`fx.plain` holds ids 1-5, so the unmatched ids are 0 and 6-11."""
        rows = session.sql(self.ANTI_JOIN).collect()
        assert sorted(row[0] for row in rows) == [0, 6, 7, 8, 9, 10, 11]

    def test_the_from_side_of_a_right_join_is_not_pruned(self, session: Session) -> None:
        frame = session.sql(self.ANTI_JOIN)
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        padded = next(s for s in compiled.scans if s.source.key == "fx.plain")
        assert padded.pushed_filter is None
        assert padded.files_scanned == 1

    def test_a_full_join_pads_both_sides(self, session: Session) -> None:
        frame = session.sql(
            "SELECT a.id, b.id FROM fx.plain AS a FULL JOIN fx.partitioned AS b "
            "ON a.id = b.id WHERE a.id IS NULL"
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        for scan in compiled.scans:
            assert scan.pushed_filter is None
        assert sorted(row[1] for row in frame.collect()) == [0, 6, 7, 8, 9, 10, 11]

    def test_the_left_join_spelling_was_never_wrong(self, session: Session) -> None:
        """Both directions of the same anti-join now agree, which is the real check."""
        left = session.sql(
            "SELECT a.id FROM fx.partitioned AS a LEFT JOIN fx.plain AS b "
            "ON a.id = b.id WHERE b.id IS NULL"
        ).collect()
        right = session.sql(self.ANTI_JOIN).collect()
        assert sorted(row[0] for row in left) == sorted(row[0] for row in right)

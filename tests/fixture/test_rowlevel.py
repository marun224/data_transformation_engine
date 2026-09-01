"""Phase 8: `DELETE`, `UPDATE` and `MERGE`, against the local fixture catalog.

PLAN.md calls this the riskiest phase in the plan, and these are the three ways it goes
wrong. Each has a class below whose whole job is to hold the line on it.

**The predicate has to mean the same thing twice.** A row-level operation commits as
`overwrite(rows, overwrite_filter=P)`: PyIceberg deletes what `P` matches, then appends
the rows the SQL kept. If `P` is wider than the SQL's `WHERE`, rows are deleted and
never written back; if it is narrower, rows survive the delete *and* arrive again in the
append. `TestScopeIsExact` is the one that would notice either -- it runs statements
whose predicate is deliberately half-translatable, and checks the untouched half of the
table row for row.

**NULL is not false.** `DELETE ... WHERE amount > 20` must keep the row whose amount is
NULL, and `NOT (amount > 20)` is NULL for exactly that row. Every class here uses
`fx.plain`, which was built with a NULL in `vendor` and a NULL in `amount`, for this
reason.

**A merge that matches twice has no answer.** `TestCardinality` pins the refusal.

Every table is a throwaway in the `wr` namespace, dropped when the test ends: the `fx.*`
fixtures are session-scoped and deliberately immutable, so a test that deleted from one
would change what every later test reads. And every assertion is on a value, per the
rule Phase 3 established.
"""

from __future__ import annotations

import contextlib
import uuid
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import (
    AnalysisException,
    ParseException,
    QueryExecutionException,
    TableNotFoundError,
    UnsupportedFeatureError,
)
from icetl.sql import functions as F

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.session import Session


@pytest.fixture
def target(catalog: SqlCatalog) -> Iterator[str]:
    """A table name nothing else uses, dropped when the test ends."""
    name = f"wr.t_{uuid.uuid4().hex[:8]}"
    yield name
    with contextlib.suppress(Exception):
        catalog.drop_table(tuple(name.split(".")))


@pytest.fixture
def plain(session: Session, target: str) -> str:
    """`fx.plain` copied into a table this test may write to.

    | id | vendor | amount |
    |---|---|---|
    | 1 | a | 10.0 |
    | 2 | b | 20.5 |
    | 3 | a | 30.25 |
    | 4 | c | **NULL** |
    | 5 | **NULL** | 50.0 |
    """
    session.table("fx.plain").write.saveAsTable(target)
    return target


@pytest.fixture
def partitioned(session: Session, target: str) -> str:
    """`fx.partitioned` copied: ids 0-11 over three `as_at_date` partitions."""
    session.table("fx.partitioned").write.partitionBy("as_at_date").saveAsTable(target)
    return target


def rows(session: Session, name: str) -> list[tuple[Any, ...]]:
    return sorted((tuple(row) for row in session.table(name).collect()), key=repr)


def ids(session: Session, name: str) -> list[Any]:
    return sorted(row[0] for row in session.table(name).collect())


def files(catalog: SqlCatalog, name: str) -> set[str]:
    table = catalog.load_table(tuple(name.split(".")))
    return {task.file.file_path for task in table.scan().plan_files()}


def snapshots(catalog: SqlCatalog, name: str) -> int:
    return len(catalog.load_table(tuple(name.split("."))).history())


# -- DELETE ----------------------------------------------------------------------


class TestDelete:
    def test_it_removes_the_rows_the_predicate_selects(self, session: Session, plain: str) -> None:
        session.sql(f"DELETE FROM {plain} WHERE id > 3")
        assert ids(session, plain) == [1, 2, 3]

    def test_a_null_survives_a_comparison_that_does_not_select_it(
        self, session: Session, plain: str
    ) -> None:
        """`amount > 20` is NULL for row 4, and NULL is not selected -- so row 4 stays.

        The bug this exists to catch is computing the survivors as `NOT (amount > 20)`,
        which is also NULL for row 4 and would drop it from both sides at once.
        """
        session.sql(f"DELETE FROM {plain} WHERE amount > 20")
        assert rows(session, plain) == sorted([(1, "a", 10.0), (4, "c", None)], key=repr)

    def test_the_rewrite_path_agrees_with_the_native_one(
        self, session: Session, plain: str, target: str, catalog: SqlCatalog
    ) -> None:
        """`abs(amount) > 20` selects the same rows and cannot be translated.

        So it takes the other route entirely -- survivors computed here and swapped in,
        rather than handed to PyIceberg's own `delete`. The answers have to match,
        NULL and all.
        """
        other = f"wr.t_{uuid.uuid4().hex[:8]}"
        session.table("fx.plain").write.saveAsTable(other)
        try:
            session.sql(f"DELETE FROM {plain} WHERE amount > 20")
            session.sql(f"DELETE FROM {other} WHERE abs(amount) > 20")
            assert rows(session, plain) == rows(session, other)
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(other.split(".")))

    def test_no_where_empties_the_table(self, session: Session, plain: str) -> None:
        session.sql(f"DELETE FROM {plain}")
        assert session.table(plain).count() == 0
        assert session.table(plain).columns == ["id", "vendor", "amount"]

    def test_a_predicate_matching_nothing_leaves_every_row(
        self, session: Session, plain: str
    ) -> None:
        session.sql(f"DELETE FROM {plain} WHERE id > 100")
        assert ids(session, plain) == [1, 2, 3, 4, 5]

    def test_it_answers_like_the_select_that_describes_it(
        self, session: Session, plain: str
    ) -> None:
        """P1, for a statement: the survivors are what a `SELECT` for them returns."""
        expected = sorted(
            (
                tuple(row)
                for row in session.sql(f"SELECT * FROM {plain} WHERE NOT (id > 3)").collect()
            ),
            key=repr,
        )
        session.sql(f"DELETE FROM {plain} WHERE id > 3")
        assert rows(session, plain) == expected

    def test_the_table_may_be_aliased(self, session: Session, plain: str) -> None:
        session.sql(f"DELETE FROM {plain} AS p WHERE p.vendor = 'a'")
        assert ids(session, plain) == [2, 4, 5]

    def test_a_read_after_the_delete_sees_it(self, session: Session, plain: str) -> None:
        """The Phase 7 trap: a cached `ScanSource` pins the snapshot it was loaded at."""
        assert session.table(plain).count() == 5
        session.sql(f"DELETE FROM {plain} WHERE id = 1")
        assert session.table(plain).count() == 4

    def test_a_frame_built_before_the_delete_keeps_its_snapshot(
        self, session: Session, plain: str
    ) -> None:
        """Read-your-plan, as it is for writes: the frame holds its own source."""
        before = session.table(plain)
        session.sql(f"DELETE FROM {plain} WHERE id = 1")
        assert before.count() == 5
        assert session.table(plain).count() == 4

    def test_it_is_one_snapshot(self, session: Session, plain: str, catalog: SqlCatalog) -> None:
        base = snapshots(catalog, plain)
        session.sql(f"DELETE FROM {plain} WHERE id = 1")
        assert snapshots(catalog, plain) - base == 1

    def test_using_is_refused_rather_than_guessed_at(self, session: Session, plain: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="USING"):
            session.sql(f"DELETE FROM {plain} USING fx.plain WHERE {plain}.id = fx.plain.id")

    def test_complex_types_survive_a_delete(self, session: Session, target: str) -> None:
        source = session.table("fx.nested")
        source.write.saveAsTable(target)
        session.sql(f"DELETE FROM {target} WHERE id = 1")
        assert session.table(target).count() == 1
        assert session.table(target).dtypes == source.dtypes


# -- UPDATE ----------------------------------------------------------------------


class TestUpdate:
    def test_it_sets_the_column_for_the_rows_selected(self, session: Session, plain: str) -> None:
        session.sql(f"UPDATE {plain} SET vendor = 'z' WHERE id <= 2")
        assert rows(session, plain) == sorted(
            [
                (1, "z", 10.0),
                (2, "z", 20.5),
                (3, "a", 30.25),
                (4, "c", None),
                (5, None, 50.0),
            ],
            key=repr,
        )

    def test_a_row_whose_condition_is_null_is_not_updated(
        self, session: Session, plain: str
    ) -> None:
        """`amount > 20` is NULL for row 4, so the `CASE` falls to `ELSE` and it keeps its value."""
        session.sql(f"UPDATE {plain} SET vendor = 'hit' WHERE amount > 20")
        assert sorted((row[0], row[1]) for row in session.table(plain).collect()) == sorted(
            [(1, "a"), (2, "hit"), (3, "hit"), (4, "c"), (5, "hit")]
        )

    def test_every_right_hand_side_sees_the_old_row(self, session: Session, plain: str) -> None:
        """`SET id = id + 100, amount = id` -- `amount` gets the *old* id, not the new one.

        Every assignment is a projection of the same input row, so this is structural
        rather than something the order of the SET list could break.
        """
        session.sql(f"UPDATE {plain} SET id = id + 100, amount = id")
        assert sorted((row[0], row[2]) for row in session.table(plain).collect()) == [
            (101, 1.0),
            (102, 2.0),
            (103, 3.0),
            (104, 4.0),
            (105, 5.0),
        ]

    def test_no_where_updates_every_row(self, session: Session, plain: str) -> None:
        session.sql(f"UPDATE {plain} SET vendor = 'all'")
        assert {row[1] for row in session.table(plain).collect()} == {"all"}
        assert session.table(plain).count() == 5

    def test_the_value_is_cast_to_the_column_s_own_type(self, session: Session, plain: str) -> None:
        """DuckDB types `7` as a 32-bit integer; the column is 64-bit.

        Without the cast back to the table's schema, PyIceberg's schema check rejects
        the write over a type difference the user never asked about.
        """
        session.sql(f"UPDATE {plain} SET id = 7 WHERE vendor = 'b'")
        assert ids(session, plain) == [1, 3, 4, 5, 7]
        assert session.table(plain).dtypes == session.table("fx.plain").dtypes

    def test_an_expression_over_other_columns(self, session: Session, plain: str) -> None:
        session.sql(f"UPDATE {plain} SET amount = amount * 2 WHERE vendor = 'a'")
        assert sorted((row[0], row[2]) for row in session.table(plain).collect()) == [
            (1, 20.0),
            (2, 20.5),
            (3, 60.5),
            (4, None),
            (5, 50.0),
        ]

    def test_an_untranslatable_predicate_still_works(self, session: Session, plain: str) -> None:
        session.sql(f"UPDATE {plain} SET vendor = 'z' WHERE upper(vendor) = 'A'")
        assert sorted((row[0], row[1]) for row in session.table(plain).collect()) == [
            (1, "z"),
            (2, "b"),
            (3, "z"),
            (4, "c"),
            (5, None),
        ]

    def test_it_is_one_commit_of_two_snapshots(
        self, session: Session, plain: str, catalog: SqlCatalog
    ) -> None:
        """An overwrite is a delete then an append, as Phase 7 found -- but one commit."""
        base = snapshots(catalog, plain)
        session.sql(f"UPDATE {plain} SET vendor = 'q' WHERE id = 2")
        assert snapshots(catalog, plain) - base == 2

    def test_an_unknown_column_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="no column"):
            session.sql(f"UPDATE {plain} SET nope = 1")

    def test_a_column_assigned_twice_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="more than once"):
            session.sql(f"UPDATE {plain} SET vendor = 'a', vendor = 'b'")

    def test_a_nested_field_assignment_is_refused_rather_than_half_done(
        self, session: Session, target: str
    ) -> None:
        session.table("fx.nested").write.saveAsTable(target)
        with pytest.raises(UnsupportedFeatureError, match="nested field"):
            session.sql(f"UPDATE {target} SET person.name = 'x'")

    def test_from_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="FROM"):
            session.sql(f"UPDATE {plain} SET vendor = 'z' FROM fx.plain")


# -- the predicate that has to mean the same thing twice --------------------------


class TestScopeIsExact:
    """The commit deletes what PyIceberg matches and writes back what the SQL kept.

    Every test here mixes a conjunct that translates with one that does not, so the
    scope is genuinely narrower than the table and genuinely wider than the condition.
    The assertion is always on the rows *outside* the scope: they are the ones a wrong
    predicate would silently take away or duplicate.
    """

    def test_a_half_translatable_delete_touches_only_its_own_partition(
        self, session: Session, partitioned: str, catalog: SqlCatalog
    ) -> None:
        before = files(catalog, partitioned)
        session.sql(
            f"DELETE FROM {partitioned} WHERE as_at_date = '2026-08-16' AND abs(amount) > 11"
        )
        assert ids(session, partitioned) == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
        # The other two partitions' files are the same files, not rewritten copies.
        assert len(before & files(catalog, partitioned)) == 2

    def test_a_half_translatable_update_leaves_the_rest_of_the_table_identical(
        self, session: Session, partitioned: str
    ) -> None:
        untouched = [row for row in session.table(partitioned).collect() if row[1] != "2026-08-16"]
        session.sql(
            f"UPDATE {partitioned} SET amount = -1.0 "
            "WHERE as_at_date = '2026-08-16' AND abs(amount) > 11"
        )
        after = [row for row in session.table(partitioned).collect() if row[1] != "2026-08-16"]
        assert sorted(tuple(r) for r in after) == sorted(tuple(r) for r in untouched)
        assert sorted(
            (row[0], row[2])
            for row in session.table(partitioned).collect()
            if row[1] == "2026-08-16"
        ) == [(4, 10.0), (5, 11.0), (6, -1.0), (7, -1.0)]

    def test_no_row_is_lost_or_duplicated(self, session: Session, partitioned: str) -> None:
        session.sql(
            f"UPDATE {partitioned} SET amount = 0.0 WHERE as_at_date >= '2026-08-16' "
            "AND abs(amount) > 100000"
        )
        assert ids(session, partitioned) == list(range(12))
        assert session.table(partitioned).count() == 12

    def test_a_wholly_untranslatable_predicate_rewrites_everything_correctly(
        self, session: Session, partitioned: str
    ) -> None:
        session.sql(f"DELETE FROM {partitioned} WHERE abs(amount) IN (0.0, 21.0)")
        assert ids(session, partitioned) == [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]


# -- MERGE, insert-only ----------------------------------------------------------


class TestMergeInsertOnly:
    """`WHEN NOT MATCHED` alone changes no existing row, so it is an append."""

    def test_it_adds_the_unmatched_source_rows(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING "
            "(SELECT 1 AS id, 'x' AS vendor, 1.0 AS amount "
            " UNION ALL SELECT 9, 'new', 9.0) AS s "
            "ON t.id = s.id WHEN NOT MATCHED THEN INSERT *"
        )
        assert ids(session, plain) == [1, 2, 3, 4, 5, 9]
        assert [row[1] for row in session.table(plain).collect() if row[0] == 1] == ["a"]

    def test_it_is_one_snapshot_because_it_appends(
        self, session: Session, plain: str, catalog: SqlCatalog
    ) -> None:
        base = snapshots(catalog, plain)
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 9 AS id, 'n' AS vendor, 1.0 AS amount) AS s "
            "ON t.id = s.id WHEN NOT MATCHED THEN INSERT *"
        )
        assert snapshots(catalog, plain) - base == 1

    def test_a_column_list_leaves_the_rest_null(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 42 AS k) AS s ON t.id = s.k "
            "WHEN NOT MATCHED THEN INSERT (id, vendor) VALUES (s.k, 'new')"
        )
        assert [tuple(r) for r in session.table(plain).collect() if r[0] == 42] == [
            (42, "new", None)
        ]

    def test_values_with_no_column_list_matches_by_position(
        self, session: Session, plain: str
    ) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 42 AS k) AS s ON t.id = s.k "
            "WHEN NOT MATCHED THEN INSERT VALUES (s.k, 'p', 9.5)"
        )
        assert [tuple(r) for r in session.table(plain).collect() if r[0] == 42] == [(42, "p", 9.5)]

    def test_the_first_clause_whose_condition_holds_wins(
        self, session: Session, plain: str
    ) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING "
            "(SELECT 20 AS k UNION ALL SELECT 30 UNION ALL SELECT 40) AS s ON t.id = s.k "
            "WHEN NOT MATCHED AND s.k <= 30 THEN INSERT (id, vendor) VALUES (s.k, 'low') "
            "WHEN NOT MATCHED AND s.k = 30 THEN INSERT (id, vendor) VALUES (s.k, 'never')"
        )
        assert sorted(
            (row[0], row[1]) for row in session.table(plain).collect() if row[0] >= 20
        ) == [(20, "low"), (30, "low")]

    def test_a_source_row_matching_no_clause_inserts_nothing(
        self, session: Session, plain: str
    ) -> None:
        """Not a row of NULLs -- no row at all. `40` above is the same case."""
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 40 AS k) AS s ON t.id = s.k "
            "WHEN NOT MATCHED AND s.k < 0 THEN INSERT (id) VALUES (s.k)"
        )
        assert ids(session, plain) == [1, 2, 3, 4, 5]

    def test_a_source_key_absent_from_the_target_never_reads_the_whole_table(
        self, session: Session, partitioned: str, catalog: SqlCatalog
    ) -> None:
        """The append path leaves every existing file exactly where it was."""
        before = files(catalog, partitioned)
        session.sql(
            f"MERGE INTO {partitioned} AS t "
            "USING (SELECT 99 AS id, '2026-08-15' AS as_at_date, 1.0 AS amount) AS s "
            "ON t.id = s.id WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT *"
        )
        assert before <= files(catalog, partitioned)
        assert ids(session, partitioned) == [*range(12), 99]


# -- MERGE, acting on target rows ------------------------------------------------


class TestMergeMatched:
    def test_update_set(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id, 'Z' AS v) AS s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.vendor = s.v"
        )
        assert rows(session, plain) == sorted(
            [
                (1, "Z", 10.0),
                (2, "b", 20.5),
                (3, "a", 30.25),
                (4, "c", None),
                (5, None, 50.0),
            ],
            key=repr,
        )

    def test_update_set_star_copies_every_column(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t "
            "USING (SELECT 1 AS id, 'Z' AS vendor, 99.0 AS amount) AS s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET *"
        )
        assert [tuple(r) for r in session.table(plain).collect() if r[0] == 1] == [(1, "Z", 99.0)]

    def test_delete(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id UNION ALL SELECT 3) AS s "
            "ON t.id = s.id WHEN MATCHED THEN DELETE"
        )
        assert ids(session, plain) == [2, 4, 5]

    def test_the_first_clause_that_fires_wins(self, session: Session, plain: str) -> None:
        """Three clauses, three different fates, decided by order and not by specificity."""
        session.sql(
            f"MERGE INTO {plain} AS t "
            "USING (SELECT 1 AS id UNION ALL SELECT 2 UNION ALL SELECT 3) AS s ON t.id = s.id "
            "WHEN MATCHED AND t.id = 1 THEN UPDATE SET t.vendor = 'one' "
            "WHEN MATCHED AND t.id <= 2 THEN DELETE "
            "WHEN MATCHED THEN UPDATE SET t.vendor = 'rest'"
        )
        assert rows(session, plain) == sorted(
            [
                (1, "one", 10.0),
                (3, "rest", 30.25),
                (4, "c", None),
                (5, None, 50.0),
            ],
            key=repr,
        )

    def test_a_target_row_no_clause_claims_keeps_its_values(
        self, session: Session, plain: str
    ) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id UNION ALL SELECT 2) AS s "
            "ON t.id = s.id WHEN MATCHED AND t.id = 1 THEN UPDATE SET t.vendor = 'one'"
        )
        assert [tuple(r) for r in session.table(plain).collect() if r[0] == 2] == [(2, "b", 20.5)]

    def test_the_source_may_be_a_table(self, session: Session, plain: str) -> None:
        session.sql(f"DELETE FROM {plain} WHERE id > 2")
        session.sql(
            f"MERGE INTO {plain} AS t USING fx.plain AS s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.amount = s.amount * 2 "
            "WHEN NOT MATCHED THEN INSERT *"
        )
        assert sorted((row[0], row[2]) for row in session.table(plain).collect()) == [
            (1, 20.0),
            (2, 41.0),
            (3, 30.25),
            (4, None),
            (5, 50.0),
        ]

    def test_both_halves_of_an_upsert_in_one_statement(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING "
            "(SELECT 1 AS id, 'upd' AS vendor, 1.0 AS amount "
            " UNION ALL SELECT 9, 'ins', 9.0) AS s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET * "
            "WHEN NOT MATCHED THEN INSERT *"
        )
        assert rows(session, plain) == sorted(
            [
                (1, "upd", 1.0),
                (2, "b", 20.5),
                (3, "a", 30.25),
                (4, "c", None),
                (5, None, 50.0),
                (9, "ins", 9.0),
            ],
            key=repr,
        )


class TestMergeNotMatchedBySource:
    """Clauses that act on the rows the source never mentions.

    They also widen the scope to the whole table -- there is no narrowing that could
    describe "everything the source does *not* name" -- which the last test pins.
    """

    def test_update(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
            "WHEN NOT MATCHED BY SOURCE THEN UPDATE SET t.vendor = 'stale'"
        )
        assert sorted((row[0], row[1]) for row in session.table(plain).collect()) == [
            (1, "a"),
            (2, "stale"),
            (3, "stale"),
            (4, "stale"),
            (5, "stale"),
        ]

    def test_delete(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id UNION ALL SELECT 2) AS s "
            "ON t.id = s.id WHEN NOT MATCHED BY SOURCE THEN DELETE"
        )
        assert ids(session, plain) == [1, 2]

    def test_a_condition_narrows_which_unmatched_rows_it_claims(
        self, session: Session, plain: str
    ) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
            "WHEN MATCHED THEN DELETE "
            "WHEN NOT MATCHED BY SOURCE AND t.id = 5 THEN UPDATE SET t.vendor = 'last'"
        )
        assert rows(session, plain) == sorted(
            [(2, "b", 20.5), (3, "a", 30.25), (4, "c", None), (5, "last", 50.0)],
            key=repr,
        )

    def test_set_star_has_no_source_row_to_copy_from(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="no source row"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
                "WHEN NOT MATCHED BY SOURCE THEN UPDATE SET *"
            )

    def test_it_reaches_partitions_the_source_never_names(
        self, session: Session, partitioned: str
    ) -> None:
        session.sql(
            f"MERGE INTO {partitioned} AS t USING (SELECT 4 AS id) AS s ON t.id = s.id "
            "WHEN NOT MATCHED BY SOURCE AND t.as_at_date = '2026-08-17' THEN DELETE"
        )
        assert ids(session, partitioned) == [0, 1, 2, 3, 4, 5, 6, 7]


class TestCardinality:
    """One target row, two source rows, and no defensible answer.

    Whichever clause won would depend on which of the two the engine looked at, so the
    reference refuses it and so does this. The check is a count against a count: an
    inner join emits one row per pair, which exceeds the number of matched target rows
    exactly when some target row has more than one partner.
    """

    def test_two_source_rows_for_one_target_row_is_refused(
        self, session: Session, plain: str
    ) -> None:
        with pytest.raises(AnalysisException, match="more than one row of the source"):
            session.sql(
                f"MERGE INTO {plain} AS t "
                "USING (SELECT 1 AS id, 'x' AS v UNION ALL SELECT 1, 'y') AS s ON t.id = s.id "
                "WHEN MATCHED THEN UPDATE SET t.vendor = s.v"
            )

    def test_the_table_is_left_untouched_when_it_is_refused(
        self, session: Session, plain: str
    ) -> None:
        with contextlib.suppress(AnalysisException):
            session.sql(
                f"MERGE INTO {plain} AS t "
                "USING (SELECT 1 AS id, 'x' AS v UNION ALL SELECT 1, 'y') AS s ON t.id = s.id "
                "WHEN MATCHED THEN DELETE"
            )
        assert ids(session, plain) == [1, 2, 3, 4, 5]

    def test_duplicate_source_rows_that_match_nothing_are_fine(
        self, session: Session, plain: str
    ) -> None:
        """The rule is about the *target* row, so two unmatched duplicates just insert twice."""
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 9 AS k UNION ALL SELECT 9) AS s "
            "ON t.id = s.k WHEN MATCHED THEN DELETE "
            "WHEN NOT MATCHED THEN INSERT (id) VALUES (s.k)"
        )
        assert ids(session, plain) == [1, 2, 3, 4, 5, 9, 9]


class TestMergeScope:
    def test_only_the_partitions_the_keys_touch_are_rewritten(
        self, session: Session, partitioned: str, catalog: SqlCatalog
    ) -> None:
        """The join keys the source holds become an `IN` list, which prunes the scan.

        Without it a merge of one row into a partitioned table rewrites every file.
        """
        before = files(catalog, partitioned)
        session.sql(
            f"MERGE INTO {partitioned} AS t USING (SELECT 4 AS id) AS s ON t.id = s.id "
            "WHEN MATCHED THEN UPDATE SET t.amount = -1.0"
        )
        assert len(before & files(catalog, partitioned)) == 2
        assert sorted((row[0], row[2]) for row in session.table(partitioned).collect()) == [
            (0, 0.0),
            (1, 1.0),
            (2, 2.0),
            (3, 3.0),
            (4, -1.0),
            (5, 11.0),
            (6, 12.0),
            (7, 13.0),
            (8, 20.0),
            (9, 21.0),
            (10, 22.0),
            (11, 23.0),
        ]

    def test_an_empty_source_changes_nothing(self, session: Session, plain: str) -> None:
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id WHERE FALSE) AS s ON t.id = s.id "
            "WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT (id) VALUES (s.id)"
        )
        assert ids(session, plain) == [1, 2, 3, 4, 5]

    def test_an_empty_source_commits_nothing_at_all(
        self, session: Session, plain: str, catalog: SqlCatalog
    ) -> None:
        base = snapshots(catalog, plain)
        session.sql(
            f"MERGE INTO {plain} AS t USING (SELECT 1 AS id WHERE FALSE) AS s ON t.id = s.id "
            "WHEN MATCHED THEN DELETE"
        )
        assert snapshots(catalog, plain) == base

    def test_a_null_key_in_the_target_matches_nothing(self, session: Session, target: str) -> None:
        """NULL equals nothing, so a NULL-keyed row is unmatched -- and `IN` agrees."""
        session.table("fx.plain").write.saveAsTable(target)
        session.sql(f"UPDATE {target} SET id = NULL WHERE id = 3")
        session.sql(
            f"MERGE INTO {target} AS t USING (SELECT 1 AS id UNION ALL SELECT 3) AS s "
            "ON t.id = s.id WHEN MATCHED THEN DELETE"
        )
        assert sorted(
            (row[0] for row in session.table(target).collect()), key=lambda v: (v is None, v)
        ) == [2, 4, 5, None]

    def test_an_extra_target_only_condition_in_the_on_still_prunes(
        self, session: Session, partitioned: str
    ) -> None:
        session.sql(
            f"MERGE INTO {partitioned} AS t USING (SELECT 4 AS id UNION ALL SELECT 8) AS s "
            "ON t.id = s.id AND t.as_at_date = '2026-08-16' "
            "WHEN MATCHED THEN DELETE"
        )
        assert ids(session, partitioned) == [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]


class TestMergeRefusals:
    """Statements that are well-formed SQL and have no meaning here."""

    def test_not_matched_cannot_delete(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="only INSERT"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
                "WHEN NOT MATCHED THEN DELETE"
            )

    def test_matched_cannot_insert(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="cannot INSERT"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
                "WHEN MATCHED THEN INSERT (id) VALUES (s.id)"
            )

    def test_a_subquery_source_needs_an_alias(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="needs an alias"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) ON t.id = id "
                "WHEN MATCHED THEN DELETE"
            )

    def test_insert_star_needs_a_source_column_for_each_target_column(
        self, session: Session, plain: str
    ) -> None:
        with pytest.raises(AnalysisException, match="none called 'amount'"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 9 AS id, 'v' AS vendor) AS s "
                "ON t.id = s.id WHEN NOT MATCHED THEN INSERT *"
            )

    def test_insert_naming_an_unknown_column_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="does not have"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 9 AS id) AS s ON t.id = s.id "
                "WHEN NOT MATCHED THEN INSERT (nope) VALUES (s.id)"
            )

    def test_a_value_count_mismatch_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match="by position"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 9 AS id) AS s ON t.id = s.id "
                "WHEN NOT MATCHED THEN INSERT VALUES (s.id)"
            )

    def test_a_missing_table_is_refused_before_anything_is_read(self, session: Session) -> None:
        with pytest.raises(TableNotFoundError, match="does_not_exist"):
            session.sql("DELETE FROM wr.does_not_exist WHERE id = 1")


class TestStatementsReturnTheEmptyFrame:
    """A statement is not a query, and the reference still hands one back."""

    def test_delete(self, session: Session, plain: str) -> None:
        assert session.sql(f"DELETE FROM {plain} WHERE id = 1").collect() == []

    def test_update(self, session: Session, plain: str) -> None:
        assert session.sql(f"UPDATE {plain} SET vendor = 'z'").collect() == []

    def test_merge(self, session: Session, plain: str) -> None:
        assert (
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
                "WHEN MATCHED THEN DELETE"
            ).collect()
            == []
        )

    def test_two_statements_at_once_are_still_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(ParseException, match="exactly one statement"):
            session.sql(f"DELETE FROM {plain} WHERE id = 1; DELETE FROM {plain} WHERE id = 2")


class TestAgainstTheDataFrameSurface:
    """The rewritten table is what the equivalent DataFrame expression describes (P1)."""

    def test_delete_matches_a_filter(self, session: Session, plain: str) -> None:
        expected = sorted(
            (
                tuple(row)
                for row in session.table("fx.plain")
                .filter(~F.coalesce(F.col("amount") > 20, F.lit(False)))
                .collect()
            ),
            key=repr,
        )
        session.sql(f"DELETE FROM {plain} WHERE amount > 20")
        assert rows(session, plain) == expected

    def test_update_matches_a_when_otherwise(self, session: Session, plain: str) -> None:
        expected = sorted(
            (
                tuple(row)
                for row in session.table("fx.plain")
                .select(
                    "id",
                    F.when(F.col("id") <= 2, F.lit("z")).otherwise(F.col("vendor")).alias("vendor"),
                    "amount",
                )
                .collect()
            ),
            key=repr,
        )
        session.sql(f"UPDATE {plain} SET vendor = 'z' WHERE id <= 2")
        assert rows(session, plain) == expected


class TestConcurrency:
    """Optimistic commit, conflict detection, bounded retry.

    Iceberg commits are optimistic, so the rows a statement computes can be stale by the
    time it writes them. The window that matters is between the *read* and the commit --
    PyIceberg's own `CommitFailedException` covers only the narrower one inside the
    commit itself, and would not notice at all here: an `overwrite` with a stale result
    is a perfectly valid commit that happens to erase someone else's rows.

    So the snapshot id is checked immediately before committing, and a statement whose
    table moved is planned again from scratch. These tests open the window by hand.
    """

    def _interloper(self, catalog: SqlCatalog, name: str, row_id: int) -> None:
        import pyarrow as pa

        table = catalog.load_table(tuple(name.split(".")))
        table.append(
            pa.table(
                {
                    "id": pa.array([row_id], pa.int64()),
                    "vendor": pa.array(["late"], pa.string()),
                    "amount": pa.array([5.0], pa.float64()),
                },
                schema=table.schema().as_arrow(),
            )
        )

    def test_a_write_that_lands_mid_statement_causes_a_replan(
        self,
        session: Session,
        plain: str,
        catalog: SqlCatalog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without the replan this deletes row 99 -- a row the statement never saw."""
        from icetl.sql import rowlevel

        original = rowlevel._select
        attempts: list[int] = []

        def racing(session_: Session, plan: object) -> object:
            result = original(session_, plan)  # type: ignore[arg-type]
            attempts.append(1)
            if len(attempts) == 1:
                self._interloper(catalog, plain, 99)
            return result

        monkeypatch.setattr(rowlevel, "_select", racing)
        session.sql(f"DELETE FROM {plain} WHERE abs(amount) > 20")

        assert len(attempts) > 1, "the statement should have been planned again"
        assert ids(session, plain) == [1, 4, 99]

    def test_it_gives_up_rather_than_committing_against_metadata_it_never_read(
        self,
        session: Session,
        plain: str,
        catalog: SqlCatalog,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from icetl.sql import rowlevel

        original = rowlevel._select
        next_id = [100]

        def always_racing(session_: Session, plan: object) -> object:
            result = original(session_, plan)  # type: ignore[arg-type]
            self._interloper(catalog, plain, next_id[0])
            next_id[0] += 1
            return result

        monkeypatch.setattr(rowlevel, "_select", always_racing)
        monkeypatch.setattr(rowlevel, "_BACKOFF_SECONDS", 0.0)
        with pytest.raises(QueryExecutionException, match="Gave up after"):
            session.sql(f"DELETE FROM {plain} WHERE abs(amount) > 20")

        # Nothing was committed, so every row the interloper added is still there.
        assert set(ids(session, plain)) >= {1, 2, 3, 4, 5}


class TestBySourceCannotReadTheSource:
    """`WHEN NOT MATCHED BY SOURCE` acts on rows no source row matched.

    So there is no source row to read, and the query is built without the source
    relation at all. Left to binding it fails as `Referenced table "s" not found`, which
    is true and explains nothing.
    """

    def test_a_condition_naming_the_source_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match=r"cannot read 's\.id'"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id) AS s ON t.id = s.id "
                "WHEN NOT MATCHED BY SOURCE AND s.id = 1 THEN DELETE"
            )

    def test_an_assignment_naming_the_source_is_refused(self, session: Session, plain: str) -> None:
        with pytest.raises(AnalysisException, match=r"cannot read 's\.v'"):
            session.sql(
                f"MERGE INTO {plain} AS t USING (SELECT 1 AS id, 'x' AS v) AS s ON t.id = s.id "
                "WHEN NOT MATCHED BY SOURCE THEN UPDATE SET t.vendor = s.v"
            )

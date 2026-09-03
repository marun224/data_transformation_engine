"""Predicate translation and projection extraction -- PLAN.md 3.2 and 3.6.

The translator's contract has two halves, and only one of them is about producing a
PyIceberg expression:

  * what it *does* translate must mean the same thing Iceberg means, or files that
    hold wanted rows get pruned away and the query silently loses data;
  * what it *cannot* translate must come back as `None`, never as a guess.

So the negative cases below matter as much as the positive ones, and the invariant
test in `tests/fixture/test_pushdown.py` closes the loop by checking the filter is
still in the SQL either way.
"""

from __future__ import annotations

import sqlglot
from pyiceberg import types as ice
from pyiceberg.expressions import AlwaysTrue, And, BooleanExpression, Not, Or
from pyiceberg.schema import Schema

from icetl.plan.builder import as_expression
from icetl.plan.describe import describe_predicate
from icetl.plan.pushdown import (
    ColumnResolver,
    binds_against,
    conjuncts,
    is_exactly_translatable,
    join_predicates,
    scope_predicate,
    translate_predicate,
)
from tests.predicates import (
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNull,
    LessThanOrEqual,
    StartsWith,
)

COLUMNS: dict[str, ice.IcebergType] = {
    "id": ice.LongType(),
    "vendor": ice.StringType(),
    "amount": ice.DoubleType(),
    "as_at_date": ice.StringType(),
    "is_active": ice.BooleanType(),
    "picked_up": ice.TimestampType(),
    "picked_up_tz": ice.TimestamptzType(),
}

SCHEMA = Schema(
    *[
        ice.NestedField(index, name, field_type, required=False)
        for index, (name, field_type) in enumerate(COLUMNS.items(), start=1)
    ]
)


def translate(sql: str, alias: str = "t") -> BooleanExpression | None:
    """Translate the WHERE clause of `SELECT * FROM t WHERE <sql>`."""
    parsed = sqlglot.parse_one(f"SELECT * FROM t AS {alias} WHERE {sql}", read="duckdb")
    where = parsed.args["where"].this
    return translate_predicate(where, ColumnResolver(alias, COLUMNS))


def described(sql: str) -> str:
    translated = translate(sql)
    assert translated is not None, f"expected {sql!r} to translate"
    return describe_predicate(translated)


class TestComparisons:
    def test_equality(self) -> None:
        assert translate("id = 1") == EqualTo("id", 1)

    def test_inequality(self) -> None:
        assert described("id != 1") == "id != 1"

    def test_ordering(self) -> None:
        assert translate("id > 1") == GreaterThan("id", 1)
        assert translate("id <= 1") == LessThanOrEqual("id", 1)

    def test_a_reversed_comparison_is_flipped_not_dropped(self) -> None:
        """`100 > amount` means `amount < 100`; translating it the other way would
        prune exactly the files the query wants."""
        assert described("100 > id") == "id < 100"
        assert described("100 <= id") == "id >= 100"

    def test_a_string_literal_keeps_its_type(self) -> None:
        assert translate("vendor = 'a'") == EqualTo("vendor", "a")

    def test_a_float_literal(self) -> None:
        assert translate("amount >= 2.5") is not None

    def test_a_negative_literal(self) -> None:
        assert described("amount > -1") == "amount > -1"

    def test_a_cast_literal_is_unwrapped_for_pyiceberg_to_bind(self) -> None:
        """PyIceberg binds the value against the field's Iceberg type, so handing it
        the raw string means we never disagree with it about what a date is."""
        assert translate("as_at_date = CAST('2026-08-17' AS DATE)") == EqualTo(
            "as_at_date", "2026-08-17"
        )

    def test_two_columns_compared_do_not_translate(self) -> None:
        assert translate("id = amount") is None


class TestSetAndNullPredicates:
    def test_in(self) -> None:
        assert translate("vendor IN ('a', 'b')") == In("vendor", ["a", "b"])

    def test_in_with_a_subquery_does_not_translate(self) -> None:
        assert translate("id IN (SELECT id FROM other)") is None

    def test_in_with_a_non_literal_does_not_translate(self) -> None:
        assert translate("id IN (1, amount)") is None

    def test_is_null(self) -> None:
        assert translate("vendor IS NULL") == IsNull("vendor")

    def test_is_not_null(self) -> None:
        assert translate("vendor IS NOT NULL") == Not(IsNull("vendor"))

    def test_between_becomes_two_bounds(self) -> None:
        assert translate("id BETWEEN 1 AND 5") == And(
            GreaterThanOrEqual("id", 1), LessThanOrEqual("id", 5)
        )

    def test_a_bare_boolean_column(self) -> None:
        assert translate("is_active") == EqualTo("is_active", True)


class TestLike:
    def test_a_trailing_wildcard_becomes_starts_with(self) -> None:
        assert translate("vendor LIKE 'ab%'") == StartsWith("vendor", "ab")

    def test_a_leading_wildcard_does_not_translate(self) -> None:
        """Iceberg has no `ENDS WITH`, and guessing would prune wanted files."""
        assert translate("vendor LIKE '%ab'") is None

    def test_an_underscore_wildcard_does_not_translate(self) -> None:
        assert translate("vendor LIKE 'a_b%'") is None

    def test_a_bare_wildcard_does_not_translate(self) -> None:
        assert translate("vendor LIKE '%'") is None


class TestConnectives:
    def test_and_of_two_translatable_terms(self) -> None:
        assert translate("id > 1 AND vendor = 'a'") == And(
            GreaterThan("id", 1), EqualTo("vendor", "a")
        )

    def test_and_keeps_the_half_it_understands(self) -> None:
        """`A AND B` never needs a file that `A` alone rejects, so a partial push is
        sound -- and on a partitioned table it is usually the half that matters."""
        assert translate("id > 1 AND upper(vendor) = 'A'") == GreaterThan("id", 1)

    def test_or_is_all_or_nothing(self) -> None:
        """Dropping half an OR narrows the predicate, which would reject files the
        query needs. Pushing nothing is the only safe answer."""
        assert translate("id > 1 OR upper(vendor) = 'A'") is None

    def test_or_of_two_translatable_terms(self) -> None:
        assert translate("id > 1 OR vendor = 'a'") == Or(
            GreaterThan("id", 1), EqualTo("vendor", "a")
        )

    def test_not(self) -> None:
        assert translate("NOT (id > 1)") == Not(GreaterThan("id", 1))

    def test_parentheses_are_transparent(self) -> None:
        assert translate("((id > 1))") == GreaterThan("id", 1)


class TestUntranslatable:
    def test_a_function_call(self) -> None:
        assert translate("upper(vendor) = 'A'") is None

    def test_an_unknown_column(self) -> None:
        assert translate("nope = 1") is None

    def test_a_column_qualified_with_another_table(self) -> None:
        """A predicate on a different table cannot prune this one."""
        parsed = sqlglot.parse_one("SELECT * FROM t AS t WHERE o.id = 1", read="duckdb")
        where = parsed.args["where"].this
        assert translate_predicate(where, ColumnResolver("t", COLUMNS)) is None

    def test_arithmetic_on_a_column(self) -> None:
        assert translate("id + 1 = 2") is None


class TestColumnResolver:
    def test_matching_is_case_insensitive_because_the_optimizer_normalises(self) -> None:
        """sqlglot works in DuckDB's dialect, which lowercases identifiers for
        lookup, so `VendorID` can reach us as `vendorid`."""
        resolver = ColumnResolver("t", {"VendorID": ice.LongType()})
        column = sqlglot.parse_one("SELECT vendorid FROM t", read="duckdb").expressions[0]
        assert resolver.name(column) == "VendorID"

    def test_a_struct_field_access_is_left_for_duckdb(self) -> None:
        resolver = ColumnResolver("t", {"person": ice.StringType()})
        column = sqlglot.parse_one("SELECT t.person.name FROM t", read="duckdb").expressions[0]
        assert resolver.name(column) is None

    def test_owns_every_column_is_false_for_a_mixed_predicate(self) -> None:
        resolver = ColumnResolver("t", COLUMNS)
        parsed = sqlglot.parse_one("SELECT * FROM t WHERE t.id = 1 AND o.x = 2", read="duckdb")
        assert not resolver.owns_every_column(parsed.args["where"].this)

    def test_owns_every_column_is_false_for_a_constant(self) -> None:
        """A predicate mentioning no column of ours is not ours to push."""
        resolver = ColumnResolver("t", COLUMNS)
        parsed = sqlglot.parse_one("SELECT * FROM t WHERE 1 = 1", read="duckdb")
        assert not resolver.owns_every_column(parsed.args["where"].this)


class TestConjuncts:
    def test_splits_nested_ands(self) -> None:
        parsed = sqlglot.parse_one("SELECT * FROM t WHERE a = 1 AND b = 2 AND c = 3")
        assert len(conjuncts(parsed.args["where"])) == 3

    def test_does_not_split_an_or(self) -> None:
        parsed = sqlglot.parse_one("SELECT * FROM t WHERE a = 1 OR b = 2")
        assert len(conjuncts(parsed.args["where"])) == 1

    def test_no_where_is_no_conjuncts(self) -> None:
        assert conjuncts(None) == []


class TestDescribePredicate:
    def test_always_true(self) -> None:
        assert describe_predicate(AlwaysTrue()) == "true"

    def test_a_conjunction_reads_as_sql(self) -> None:
        assert described("id > 1 AND vendor = 'a'") == "(id > 1 AND vendor = 'a')"

    def test_a_set_predicate_lists_its_values(self) -> None:
        assert described("vendor IN ('b', 'a')") == "vendor IN ('a', 'b')"

    def test_a_unary_predicate(self) -> None:
        assert described("vendor IS NULL") == "vendor IS NULL"

    def test_starts_with(self) -> None:
        assert described("vendor LIKE 'ab%'") == "vendor STARTS WITH 'ab'"


class TestTimestampLiterals:
    """A bare date against a timestamp column -- the most ordinary filter there is.

    SQL reads `picked_up >= '2024-01-01'` as midnight that day, and so do the reference engine and
    DuckDB. PyIceberg refuses anything short of full ISO-8601, and refuses it by
    *raising* from inside `plan_files()`, so this is both a correctness hazard and
    the difference between pruning a time-partitioned table and scanning all of it.
    """

    def test_a_bare_date_is_widened_for_a_timestamp_column(self) -> None:
        assert translate("picked_up >= '2024-01-01'") == GreaterThanOrEqual(
            "picked_up", "2024-01-01T00:00:00"
        )

    def test_the_widened_literal_actually_binds(self) -> None:
        translated = translate("picked_up >= '2024-01-01'")
        assert translated is not None
        assert binds_against(translated, SCHEMA)

    def test_a_space_separated_timestamp_is_accepted(self) -> None:
        translated = translate("picked_up >= '2024-01-01 06:30:00'")
        assert translated is not None
        assert binds_against(translated, SCHEMA)

    def test_a_full_timestamp_is_left_alone(self) -> None:
        assert translate("picked_up >= '2024-01-01T06:30:00'") == GreaterThanOrEqual(
            "picked_up", "2024-01-01T06:30:00"
        )

    def test_a_timestamptz_column_is_deliberately_not_widened(self) -> None:
        """A bare date has no zone; DuckDB resolves it against the session's, and
        guessing UTC could move the boundary and prune away a file holding wanted
        rows. Pruning less is free, pruning wrongly is not -- so this one is dropped
        by `binds_against` and stays in the SQL."""
        translated = translate("picked_up_tz >= '2024-01-01'")
        assert translated is not None
        assert not binds_against(translated, SCHEMA)

    def test_between_widens_both_bounds(self) -> None:
        translated = translate("picked_up BETWEEN '2024-01-01' AND '2024-02-01'")
        assert translated is not None
        assert binds_against(translated, SCHEMA)

    def test_in_widens_every_value(self) -> None:
        translated = translate("picked_up IN ('2024-01-01', '2024-02-01')")
        assert translated is not None
        assert binds_against(translated, SCHEMA)

    def test_a_string_column_is_untouched(self) -> None:
        assert translate("as_at_date = '2024-01-01'") == EqualTo("as_at_date", "2024-01-01")


class TestBindValidation:
    """The rule that makes "never push what you do not understand" true.

    Translation happens against sqlglot nodes; PyIceberg only inspects the literal
    when the scan is planned, deep inside `plan_files()` where nothing can decline
    gracefully. Binding here moves that failure somewhere it costs only pruning.
    """

    def test_a_well_formed_predicate_binds(self) -> None:
        translated = translate("id > 1")
        assert translated is not None
        assert binds_against(translated, SCHEMA)

    def test_a_predicate_naming_an_unknown_column_does_not_bind(self) -> None:
        assert not binds_against(GreaterThan("nope", 1), SCHEMA)

    def test_a_predicate_with_an_unparseable_literal_does_not_bind(self) -> None:
        assert not binds_against(GreaterThan("picked_up", "not-a-timestamp"), SCHEMA)

    def test_binding_never_raises(self) -> None:
        """Whatever pyiceberg throws, the answer is False, not an exception."""
        assert not binds_against(EqualTo("id", "not-a-number"), SCHEMA)


class TestExactTranslation:
    """The gate the write path stands on (Phase 8).

    Read pushdown may translate a predicate into something *wider* than the SQL: the
    SQL re-applies the filter, so a file that need not have been read costs only I/O.
    A row-level write has no second chance -- `overwrite(rows, overwrite_filter=P)`
    deletes what `P` matches and writes back what the SQL kept -- so there the two must
    agree row for row. `is_exactly_translatable` is the whitelist that says which node
    types have been read and found to.
    """

    def exact(self, sql: str) -> bool:
        parsed = sqlglot.parse_one(f"SELECT * FROM t AS t WHERE {sql}", read="duckdb")
        return is_exactly_translatable(parsed.args["where"].this)

    def test_comparisons_and_membership_are_exact(self) -> None:
        for sql in (
            "id = 1",
            "id <> 1",
            "amount >= 2.5",
            "id IN (1, 2, 3)",
            "id BETWEEN 1 AND 4",
            "vendor IS NULL",
            "vendor IS NOT NULL",
            "is_active",
        ):
            assert self.exact(sql), sql

    def test_and_or_not_are_exact_when_both_sides_are(self) -> None:
        assert self.exact("id = 1 AND vendor = 'a'")
        assert self.exact("id = 1 OR vendor = 'a'")
        assert self.exact("NOT (id = 1)")
        assert not self.exact("id = 1 OR vendor LIKE 'a%'")

    def test_like_is_not_exact_even_though_it_translates(self) -> None:
        """`StartsWith` prunes correctly and is still not a row-for-row equivalent.

        The deliberate omission: it is the one node whose translation is a *pattern*
        rewritten into a different operator, and escapes are spelled differently on
        each side. Good enough to prune with, not good enough to delete with.
        """
        assert translate("vendor LIKE 'a%'") is not None
        assert not self.exact("vendor LIKE 'a%'")

    def test_it_gates_the_shape_and_leaves_the_rest_to_the_translator(self) -> None:
        """`upper(v) = 'A'` is an `EQ`, so it passes here -- and is still not pushed.

        The two checks are separate on purpose. This one answers "would a translation
        of this shape be exact"; `translate_predicate` answers "is there one at all",
        and declines because neither side of the comparison is a bare column. A caller
        needs both, which is what `scope_predicate` does.
        """
        assert self.exact("upper(vendor) = 'A'")
        assert translate("upper(vendor) = 'A'") is None


class TestScopePredicate:
    """`scope_predicate` hands back both languages, built from the same nodes."""

    def scope(
        self, sql: str, *, exact_only: bool
    ) -> tuple[BooleanExpression, list[str], list[str]]:
        parsed = sqlglot.parse_one(f"SELECT * FROM t AS t WHERE {sql}", read="duckdb")
        predicate, kept, dropped = scope_predicate(
            conjuncts(parsed.args["where"]),
            ColumnResolver("t", COLUMNS),
            SCHEMA,
            exact_only=exact_only,
        )
        return predicate, [term.sql() for term in kept], [term.sql() for term in dropped]

    def test_the_kept_terms_are_the_ones_the_predicate_was_built_from(self) -> None:
        predicate, kept, dropped = self.scope("id = 1 AND upper(vendor) = 'A'", exact_only=True)
        assert kept == ["id = 1"]
        assert dropped == ["UPPER(vendor) = 'A'"]
        assert predicate == EqualTo("id", 1)

    def test_dropping_a_conjunct_only_ever_widens_the_scope(self) -> None:
        """`A AND B` -> `A`: more rows, never fewer. That is what makes it safe."""
        wide, _, _ = self.scope("id = 1 AND abs(amount) > 1", exact_only=True)
        assert wide == EqualTo("id", 1)

    def test_nothing_translatable_gives_the_whole_table(self) -> None:
        predicate, kept, _ = self.scope("upper(vendor) = 'A'", exact_only=True)
        assert isinstance(predicate, AlwaysTrue)
        assert kept == []

    def test_exact_only_refuses_what_plain_pushdown_accepts(self) -> None:
        loose, kept_loose, _ = self.scope("vendor LIKE 'a%'", exact_only=False)
        strict, kept_strict, _ = self.scope("vendor LIKE 'a%'", exact_only=True)
        assert kept_loose and not isinstance(loose, AlwaysTrue)
        assert kept_strict == [] and isinstance(strict, AlwaysTrue)

    def test_a_term_naming_another_table_is_not_this_table_s_business(self) -> None:
        predicate, kept, dropped = self.scope("id = 1 AND other.x = 2", exact_only=True)
        assert kept == ["id = 1"]
        assert dropped == ["other.x = 2"]
        assert predicate == EqualTo("id", 1)


class TestJoinPredicates:
    """Which side of a join its `ON` clause is allowed to prune.

    The rule is one sentence -- a join filters the side it does not preserve -- and
    every case below is that sentence applied. FINDINGS.md 3.5.
    """

    @staticmethod
    def terms(sql: str) -> dict[str, list[str]]:
        parsed = as_expression(sqlglot.parse_one(sql, read="duckdb"))
        return {
            alias: [term.sql(dialect="duckdb") for term in found]
            for alias, found in join_predicates(parsed).items()
        }

    def test_an_inner_join_filters_both_sides(self) -> None:
        found = self.terms("SELECT * FROM a INNER JOIN b ON a.id = b.id AND b.d = '1'")
        assert set(found) == {"a", "b"}
        assert sorted(found["b"]) == ["a.id = b.id", "b.d = '1'"]

    def test_a_left_join_filters_only_the_right(self) -> None:
        found = self.terms("SELECT * FROM a LEFT JOIN b ON a.id = b.id AND b.d = '1'")
        assert set(found) == {"b"}

    def test_a_right_join_filters_only_the_left(self) -> None:
        found = self.terms("SELECT * FROM a RIGHT JOIN b ON a.id = b.id")
        assert set(found) == {"a"}

    def test_a_full_join_filters_neither(self) -> None:
        assert self.terms("SELECT * FROM a FULL JOIN b ON a.id = b.id") == {}

    def test_an_earlier_table_is_filtered_by_a_later_inner_join(self) -> None:
        """`a` survives the second join only if it matched, so its ON filters `a` too."""
        found = self.terms("SELECT * FROM a JOIN b ON a.id = b.id JOIN c ON a.k = c.k")
        assert sorted(found["a"]) == ["a.id = b.id", "a.k = c.k"]

    def test_a_left_join_after_an_inner_one_still_preserves_its_left(self) -> None:
        found = self.terms("SELECT * FROM a JOIN b ON a.id = b.id LEFT JOIN c ON a.k = c.k")
        assert found["c"] == ["a.k = c.k"]
        assert found["a"] == ["a.id = b.id"]

    def test_a_cross_join_has_no_on_clause_to_read(self) -> None:
        assert self.terms("SELECT * FROM a CROSS JOIN b") == {}

    def test_an_unrecognised_join_shape_yields_nothing(self) -> None:
        """Conservative by default: no terms costs pruning, never correctness."""
        assert self.terms("SELECT * FROM a SEMI JOIN b ON a.id = b.id") == {}

    def test_a_plan_with_no_join_yields_nothing(self) -> None:
        assert self.terms("SELECT * FROM a WHERE a.id = 1") == {}

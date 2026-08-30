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

from icetl.plan.describe import describe_predicate
from icetl.plan.pushdown import (
    ColumnResolver,
    binds_against,
    conjuncts,
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

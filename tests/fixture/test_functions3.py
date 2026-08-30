"""The third function tranche, executed against real rows.

Same rule as `test_functions.py` and `test_functions2.py`: assert on **values**, and
say what the reference engine returns. Every expected value here is the reference engine's
documented result, written
down before the test was run -- which is the only reason a failure means anything.
"""

from __future__ import annotations

import datetime
import math
from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import EngineTypeError, EngineValueError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.sql.column import Column
    from icetl.sql.session import Session


def _one(session: Session, column: Column) -> Any:
    return session.table("fx.plain").select(column.alias("v")).limit(1).collect()[0]["v"]


class TestStringLengths:
    def test_char_and_chr_are_the_same_function(self, session: Session) -> None:
        assert _one(session, F.char(F.lit(65))) == "A"
        assert _one(session, F.chr(F.lit(97))) == "a"

    def test_char_length_counts_characters(self, session: Session) -> None:
        assert _one(session, F.char_length(F.lit("abc"))) == 3
        assert _one(session, F.character_length(F.lit("héllo"))) == 5

    def test_octet_length_counts_bytes(self, session: Session) -> None:
        """The two disagree on non-ASCII, which is the whole point of having both:
        `héllo` is 5 characters and 6 bytes in UTF-8."""
        assert _one(session, F.octet_length(F.lit("abc"))) == 3
        assert _one(session, F.octet_length(F.lit("héllo"))) == 6
        assert _one(session, F.char_length(F.lit("héllo"))) == 5

    def test_bit_length_is_octet_length_times_eight(self, session: Session) -> None:
        assert _one(session, F.bit_length(F.lit("abc"))) == 24
        assert _one(session, F.bit_length(F.lit("héllo"))) == 48


class TestStringCasingAndSearch:
    def test_ucase_and_lcase(self, session: Session) -> None:
        assert _one(session, F.ucase(F.lit("aBc"))) == "ABC"
        assert _one(session, F.lcase(F.lit("aBc"))) == "abc"

    def test_startswith_endswith_contains(self, session: Session) -> None:
        assert _one(session, F.startswith(F.lit("Basic SQL"), F.lit("Basic"))) is True
        assert _one(session, F.endswith(F.lit("Basic SQL"), F.lit("SQL"))) is True
        assert _one(session, F.contains(F.lit("Basic SQL"), F.lit("asi"))) is True
        assert _one(session, F.startswith(F.lit("Basic SQL"), F.lit("SQL"))) is False

    def test_search_functions_propagate_null(self, session: Session) -> None:
        """The reference engine returns NULL when either argument is NULL, not false."""
        assert _one(session, F.startswith(F.lit("Basic"), F.lit(None))) is None
        assert _one(session, F.contains(F.lit(None), F.lit("a"))) is None

    def test_substr(self, session: Session) -> None:
        assert _one(session, F.substr(F.lit("Basic SQL"), F.lit(7))) == "SQL"
        assert _one(session, F.substr(F.lit("Basic SQL"), F.lit(1), F.lit(5))) == "Basic"

    def test_find_in_set(self, session: Session) -> None:
        assert _one(session, F.find_in_set(F.lit("b"), F.lit("abc,b,ab,c,def"))) == 2
        assert _one(session, F.find_in_set(F.lit("zz"), F.lit("a,b,c"))) == 0

    def test_find_in_set_refuses_to_match_across_fields(self, session: Session) -> None:
        """The reference engine returns 0 when the needle contains a comma, rather than matching a
        run of fields."""
        assert _one(session, F.find_in_set(F.lit("a,b"), F.lit("a,b,c"))) == 0


class TestStringFormatting:
    def test_overlay_defaults_to_the_length_of_the_replacement(self, session: Session) -> None:
        assert _one(session, F.overlay(F.lit("Basic SQL"), F.lit("_"), F.lit(6))) == "Basic_SQL"

    def test_overlay_with_an_explicit_length(self, session: Session) -> None:
        assert _one(session, F.overlay(F.lit("Basic SQL"), F.lit("CORE"), F.lit(7))) == "Basic CORE"
        assert (
            _one(session, F.overlay(F.lit("Basic SQL"), F.lit("ANSI "), F.lit(7), 0))
            == "Basic ANSI SQL"
        )

    def test_format_string(self, session: Session) -> None:
        assert _one(session, F.format_string("%s-%d", F.lit("a"), F.lit(3))) == "a-3"
        assert _one(session, F.printf("%d/%d", F.lit(1), F.lit(2))) == "1/2"

    def test_format_string_wants_a_literal_format(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            F.format_string(F.lit("%s"), F.lit("a"))  # type: ignore[arg-type]

    def test_format_number_groups_thousands(self, session: Session) -> None:
        assert _one(session, F.format_number(F.lit(12345.678), 2)) == "12,345.68"
        assert _one(session, F.format_number(F.lit(5.0), 0)) == "5"

    def test_format_number_rejects_a_negative_scale(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            F.format_number(F.lit(1.0), -1)

    def test_split_part(self, session: Session) -> None:
        assert _one(session, F.split_part(F.lit("11.12.13"), F.lit("."), F.lit(3))) == "13"

    def test_split_part_indexes_from_the_end_with_a_negative(self, session: Session) -> None:
        assert _one(session, F.split_part(F.lit("11.12.13"), F.lit("."), F.lit(-1))) == "13"

    def test_split_part_out_of_range_is_the_empty_string(self, session: Session) -> None:
        """The reference engine gives '' where DuckDB's own list indexing would give NULL."""
        assert _one(session, F.split_part(F.lit("a.b"), F.lit("."), F.lit(9))) == ""

    def test_split_part_rejects_index_zero(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            F.split_part(F.lit("a.b"), F.lit("."), 0)

    def test_elt_is_one_indexed(self, session: Session) -> None:
        assert _one(session, F.elt(F.lit(1), F.lit("scala"), F.lit("java"))) == "scala"
        assert _one(session, F.elt(F.lit(2), F.lit("scala"), F.lit("java"))) == "java"

    def test_elt_out_of_range_is_null(self, session: Session) -> None:
        assert _one(session, F.elt(F.lit(3), F.lit("scala"), F.lit("java"))) is None

    def test_elt_needs_a_value(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            F.elt(F.lit(1))


class TestRegexpFamily:
    def test_regexp_count(self, session: Session) -> None:
        """The reference engine's own documented example for this function."""
        subject = F.lit("Steven Jones and Stephen Smith")
        assert _one(session, F.regexp_count(subject, F.lit(r"Ste(v|ph)en"))) == 2

    def test_regexp_count_with_no_match_is_zero(self, session: Session) -> None:
        assert _one(session, F.regexp_count(F.lit("abc"), F.lit("[0-9]"))) == 0

    def test_regexp_substr(self, session: Session) -> None:
        subject = F.lit("Steven Jones and Stephen Smith")
        assert _one(session, F.regexp_substr(subject, F.lit(r"Ste(v|ph)en"))) == "Steven"
        assert _one(session, F.regexp_substr(F.lit("a1b2"), F.lit(r"[0-9]"))) == "1"

    def test_regexp_substr_with_no_match_is_null(self, session: Session) -> None:
        """The reference engine gives NULL for no match. DuckDB's `regexp_extract` gives '', which
        The reference engine reserves for a pattern that matched an empty string -- so returning ''
        for both would make the two indistinguishable."""
        assert _one(session, F.regexp_substr(F.lit("abc"), F.lit(r"[0-9]"))) is None

    def test_regexp_instr(self, session: Session) -> None:
        assert _one(session, F.regexp_instr(F.lit("a1b2"), F.lit(r"[0-9]"))) == 2

    def test_regexp_instr_with_no_match_is_zero(self, session: Session) -> None:
        assert _one(session, F.regexp_instr(F.lit("abc"), F.lit(r"[0-9]"))) == 0

    def test_regexp_like_and_its_alias(self, session: Session) -> None:
        assert _one(session, F.regexp_like(F.lit("a1b2"), F.lit(r"[0-9]"))) is True
        assert _one(session, F.regexp(F.lit("abc"), F.lit(r"[0-9]"))) is False


class TestBase64:
    def test_base64_round_trip(self, session: Session) -> None:
        assert _one(session, F.base64(F.lit("Basic"))) == "QmFzaWM="

    def test_unbase64_returns_binary(self, session: Session) -> None:
        """The reference engine's `unbase64` gives bytes, not a string."""
        assert _one(session, F.unbase64(F.lit("QmFzaWM="))) == b"Basic"

    def test_base64_of_unbase64_is_the_identity(self, session: Session) -> None:
        assert _one(session, F.base64(F.unbase64(F.lit("U3Bhcms=")))) == "U3Bhcms="


class TestMathThirdTranche:
    def test_e_and_pi_take_no_arguments(self, session: Session) -> None:
        assert math.isclose(_one(session, F.e()), math.e)
        assert math.isclose(_one(session, F.pi()), math.pi)

    def test_ln(self, session: Session) -> None:
        assert math.isclose(_one(session, F.ln(F.lit(math.e))), 1.0)
        assert _one(session, F.ln(F.lit(1.0))) == 0.0

    def test_log1p_and_expm1_are_inverses(self, session: Session) -> None:
        assert math.isclose(_one(session, F.log1p(F.lit(0.5))), math.log1p(0.5))
        assert math.isclose(_one(session, F.expm1(F.lit(0.5))), math.expm1(0.5))
        assert math.isclose(_one(session, F.log1p(F.expm1(F.lit(0.5)))), 0.5)

    def test_rint_rounds_halves_to_even(self, session: Session) -> None:
        """Java's `Math.rint`, which the reference engine uses: 2.5 goes to 2 and 3.5 to 4, because
        the tie breaks toward the even neighbour. DuckDB's `round` would give 3 and 4.
        """
        assert _one(session, F.rint(F.lit(2.5))) == 2.0
        assert _one(session, F.rint(F.lit(3.5))) == 4.0

    def test_rint_breaks_negative_ties_to_even_too(self, session: Session) -> None:
        """-2.5 goes to -2, where `round` would give -3."""
        assert _one(session, F.rint(F.lit(-2.5))) == -2.0
        assert _one(session, F.rint(F.lit(-3.5))) == -4.0

    def test_rint_leaves_a_non_tie_alone(self, session: Session) -> None:
        assert _one(session, F.rint(F.lit(2.4))) == 2.0
        assert _one(session, F.rint(F.lit(2.6))) == 3.0
        assert _one(session, F.rint(F.lit(-2.6))) == -3.0

    def test_cot_csc_sec(self, session: Session) -> None:
        assert math.isclose(_one(session, F.cot(F.lit(1.0))), 1 / math.tan(1.0))
        assert math.isclose(_one(session, F.csc(F.lit(1.0))), 1 / math.sin(1.0))
        assert math.isclose(_one(session, F.sec(F.lit(1.0))), 1 / math.cos(1.0))

    def test_inverse_hyperbolics(self, session: Session) -> None:
        assert math.isclose(_one(session, F.acosh(F.lit(1.0))), 0.0, abs_tol=1e-12)
        assert math.isclose(_one(session, F.asinh(F.lit(1.0))), math.asinh(1.0))
        assert math.isclose(_one(session, F.atanh(F.lit(0.5))), math.atanh(0.5))

    def test_sign_and_power_are_aliases(self, session: Session) -> None:
        assert _one(session, F.sign(F.lit(-3.0))) == -1.0
        assert _one(session, F.power(F.lit(2.0), F.lit(10.0))) == 1024.0


class TestBitwise:
    def test_bit_count(self, session: Session) -> None:
        assert _one(session, F.bit_count(F.lit(7))) == 3
        assert _one(session, F.bit_count(F.lit(0))) == 0

    def test_bitwise_not(self, session: Session) -> None:
        assert _one(session, F.bitwise_not(F.lit(5))) == -6

    def test_shiftrightunsigned_treats_the_sign_bit_as_data(self, session: Session) -> None:
        """Java's `>>>`. `-8 >>> 1` is not -4: the vacated high bit is filled with a
        zero, so the result is a large positive number."""
        assert _one(session, F.shiftrightunsigned(F.lit(-8), 1)) == 9223372036854775804

    def test_shiftrightunsigned_on_a_positive_is_an_ordinary_shift(self, session: Session) -> None:
        assert _one(session, F.shiftrightunsigned(F.lit(8), 1)) == 4

    def test_shiftrightunsigned_rejects_an_out_of_range_shift(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            F.shiftrightunsigned(F.lit(8), 64)


class TestTryDivideAndBuckets:
    def test_try_divide_by_zero_is_null(self, session: Session) -> None:
        assert _one(session, F.try_divide(F.lit(1.0), F.lit(0.0))) is None

    def test_try_divide_otherwise_divides(self, session: Session) -> None:
        assert _one(session, F.try_divide(F.lit(10.0), F.lit(4.0))) == 2.5

    def test_width_bucket(self, session: Session) -> None:
        """The reference engine's own documented example."""
        column = F.width_bucket(F.lit(5.35), F.lit(0.024), F.lit(10.06), F.lit(5))
        assert _one(session, column) == 3

    def test_width_bucket_below_the_range_is_zero(self, session: Session) -> None:
        column = F.width_bucket(F.lit(-1.0), F.lit(0.0), F.lit(10.0), F.lit(5))
        assert _one(session, column) == 0

    def test_width_bucket_at_or_above_the_range_is_one_past_the_last(
        self, session: Session
    ) -> None:
        column = F.width_bucket(F.lit(10.0), F.lit(0.0), F.lit(10.0), F.lit(5))
        assert _one(session, column) == 6

    def test_width_bucket_with_descending_bounds(self, session: Session) -> None:
        """The reference engine allows min > max, which counts buckets downward from min."""
        column = F.width_bucket(F.lit(2.0), F.lit(10.0), F.lit(0.0), F.lit(5))
        assert _one(session, column) == 5


class TestRandom:
    def test_rand_is_in_the_unit_interval(self, session: Session) -> None:
        value = _one(session, F.rand())
        assert 0.0 <= value < 1.0

    def test_rand_varies_between_rows(self, session: Session) -> None:
        rows = session.table("fx.plain").select(F.rand().alias("v")).collect()
        assert len({row["v"] for row in rows}) > 1

    def test_randn_produces_a_finite_double(self, session: Session) -> None:
        assert math.isfinite(_one(session, F.randn()))

    def test_a_seed_is_refused_rather_than_ignored(self, session: Session) -> None:
        """Accepting a seed we cannot honour would break the one property the
        argument exists for."""
        with pytest.raises(EngineValueError):
            F.rand(42)
        with pytest.raises(EngineValueError):
            F.randn(42)


class TestCurrentMoment:
    def test_now_and_curdate_are_aliases(self, session: Session) -> None:
        assert isinstance(_one(session, F.now()), datetime.datetime)
        assert isinstance(_one(session, F.curdate()), datetime.date)

    def test_localtimestamp_has_no_timezone(self, session: Session) -> None:
        """The reference engine's `localtimestamp` is TIMESTAMP_NTZ; `current_timestamp` is not."""
        value = _one(session, F.localtimestamp())
        assert isinstance(value, datetime.datetime)
        assert value.tzinfo is None


class TestDayNumbering:
    """The reference engine has three day-of-week numberings and DuckDB has a fourth."""

    def test_day_is_dayofmonth(self, session: Session) -> None:
        assert _one(session, F.day(F.lit(datetime.date(2026, 8, 30)))) == 30

    def test_dayofweek_numbers_sunday_one(self, session: Session) -> None:
        """2026-08-30 is a Sunday."""
        assert _one(session, F.dayofweek(F.lit(datetime.date(2026, 8, 30)))) == 1
        assert _one(session, F.dayofweek(F.lit(datetime.date(2026, 8, 31)))) == 2

    def test_weekday_numbers_monday_zero(self, session: Session) -> None:
        """The same two days under the reference engine's other numbering: Monday 0, Sunday 6."""
        assert _one(session, F.weekday(F.lit(datetime.date(2026, 8, 30)))) == 6
        assert _one(session, F.weekday(F.lit(datetime.date(2026, 8, 31)))) == 0


class TestNextDay:
    def test_next_day(self, session: Session) -> None:
        """The reference engine's own documented example: 2015-07-27 is a Monday."""
        column = F.next_day(F.lit(datetime.date(2015, 7, 27)), "Sun")
        assert _one(session, column) == datetime.date(2015, 8, 2)

    def test_next_day_is_strictly_after(self, session: Session) -> None:
        """Asking a Monday for the next Monday gives the following week, not itself."""
        column = F.next_day(F.lit(datetime.date(2015, 7, 27)), "Monday")
        assert _one(session, column) == datetime.date(2015, 8, 3)

    def test_next_day_accepts_short_day_names(self, session: Session) -> None:
        monday = F.lit(datetime.date(2015, 7, 27))
        assert _one(session, F.next_day(monday, "TU")) == datetime.date(2015, 7, 28)
        assert _one(session, F.next_day(monday, "saturday")) == datetime.date(2015, 8, 1)

    def test_next_day_returns_a_date_not_a_timestamp(self, session: Session) -> None:
        """`date_add` widens to TIMESTAMP in DuckDB; the reference engine's `next_day` is a DATE."""
        value = _one(session, F.next_day(F.lit(datetime.date(2015, 7, 27)), "Sun"))
        assert type(value) is datetime.date

    def test_next_day_rejects_a_name_it_does_not_know(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            F.next_day(F.lit(datetime.date(2015, 7, 27)), "Caturday")


class TestEpochConversions:
    def test_unix_date_and_back(self, session: Session) -> None:
        day = datetime.date(2026, 8, 30)
        days = (day - datetime.date(1970, 1, 1)).days
        assert _one(session, F.unix_date(F.lit(day))) == days
        assert _one(session, F.date_from_unix_date(F.lit(days))) == day

    def test_unix_date_of_the_epoch_is_zero(self, session: Session) -> None:
        assert _one(session, F.unix_date(F.lit(datetime.date(1970, 1, 1)))) == 0

    def test_unix_seconds_millis_micros(self, session: Session) -> None:
        stamp = datetime.datetime(2026, 8, 30, 0, 0, 0)
        seconds = 1788048000
        assert _one(session, F.unix_seconds(F.lit(stamp))) == seconds
        assert _one(session, F.unix_millis(F.lit(stamp))) == seconds * 1000
        assert _one(session, F.unix_micros(F.lit(stamp))) == seconds * 1_000_000

    def test_timestamp_millis_and_micros_are_the_inverses(self, session: Session) -> None:
        stamp = datetime.datetime(2026, 8, 30, 0, 0, 0)
        seconds = 1788048000
        assert _one(session, F.timestamp_millis(F.lit(seconds * 1000))) == stamp
        assert _one(session, F.timestamp_micros(F.lit(seconds * 1_000_000))) == stamp

    def test_make_timestamp(self, session: Session) -> None:
        column = F.make_timestamp(F.lit(2026), F.lit(8), F.lit(30), F.lit(1), F.lit(2), F.lit(3.0))
        assert _one(session, column) == datetime.datetime(2026, 8, 30, 1, 2, 3)


class TestDatePart:
    def test_date_part_and_its_aliases(self, session: Session) -> None:
        day = F.lit(datetime.date(2026, 8, 30))
        assert _one(session, F.date_part("YEAR", day)) == 2026
        assert _one(session, F.datepart("month", day)) == 8
        assert _one(session, F.extract("DAY", day)) == 30

    def test_date_part_on_a_timestamp(self, session: Session) -> None:
        stamp = F.lit(datetime.datetime(2026, 8, 30, 13, 45, 7))
        assert _one(session, F.date_part("hour", stamp)) == 13
        assert _one(session, F.date_part("minute", stamp)) == 45

    def test_date_part_refuses_the_day_of_week_family(self, session: Session) -> None:
        """The reference engine numbers Sunday 1 here and DuckDB numbers it 0. Passing the field
        through would be off by one and look perfectly fine, so it is refused and the
        error names the two functions whose numbering is unambiguous."""
        with pytest.raises(EngineValueError, match="dayofweek"):
            F.date_part("DOW", F.lit(datetime.date(2026, 8, 30)))

    def test_date_part_wants_a_literal_field(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            F.date_part(F.lit("YEAR"), F.lit(datetime.date(2026, 8, 30)))  # type: ignore[arg-type]


class TestTryToTimestamp:
    def test_try_to_timestamp_parses(self, session: Session) -> None:
        column = F.try_to_timestamp(F.lit("2026-08-30 01:02:03"))
        assert _one(session, column) == datetime.datetime(2026, 8, 30, 1, 2, 3)

    def test_try_to_timestamp_gives_null_rather_than_raising(self, session: Session) -> None:
        assert _one(session, F.try_to_timestamp(F.lit("not a timestamp"))) is None

    def test_try_to_timestamp_with_a_format(self, session: Session) -> None:
        column = F.try_to_timestamp(F.lit("2026-08-30"), F.lit("%Y-%m-%d"))
        assert _one(session, column) == datetime.datetime(2026, 8, 30, 0, 0, 0)


class TestHashingSecondTranche:
    def test_sha_is_sha1(self, session: Session) -> None:
        expected = "a9993e364706816aba3e25717850c26c9cd0d89d"
        assert _one(session, F.sha(F.lit("abc"))) == expected

    def test_xxhash64_is_stable_and_fits_a_bigint(self, session: Session) -> None:
        """The value is *not* the reference engine's -- DuckDB hashes differently -- so what is
        asserted is what callers may actually rely on: the same input hashes the same
        way within a query, and the result is a signed 64-bit integer the reference engine can hold.
        """
        first = _one(session, F.xxhash64(F.lit("abc")))
        second = _one(session, F.xxhash64(F.lit("abc")))
        assert first == second
        assert -(2**63) <= first < 2**63
        assert _one(session, F.xxhash64(F.lit("abc"))) != _one(session, F.xxhash64(F.lit("abd")))

    def test_xxhash64_needs_a_column(self, session: Session) -> None:
        with pytest.raises(EngineValueError):
            F.xxhash64()


class TestSessionFunctions:
    def test_current_user_and_its_aliases(self, session: Session) -> None:
        value = _one(session, F.current_user())
        assert isinstance(value, str) and value
        assert _one(session, F.user()) == value
        assert _one(session, F.session_user()) == value

    def test_current_catalog_and_schema(self, session: Session) -> None:
        assert isinstance(_one(session, F.current_catalog()), str)
        schema = _one(session, F.current_schema())
        assert isinstance(schema, str)
        assert _one(session, F.current_database()) == schema


class TestEqualNullAndAssertions:
    def test_equal_null_treats_two_nulls_as_equal(self, session: Session) -> None:
        assert _one(session, F.equal_null(F.lit(None), F.lit(None))) is True

    def test_equal_null_is_false_not_null_against_a_value(self, session: Session) -> None:
        """`=` would give NULL here, which is the whole reason the function exists."""
        assert _one(session, F.equal_null(F.lit(None), F.lit(1))) is False

    def test_equal_null_on_two_values(self, session: Session) -> None:
        assert _one(session, F.equal_null(F.lit(1), F.lit(1))) is True
        assert _one(session, F.equal_null(F.lit(1), F.lit(2))) is False

    def test_assert_true_passes_quietly_as_null(self, session: Session) -> None:
        """The reference engine returns NULL on success, not true."""
        assert _one(session, F.assert_true(F.lit(1) < F.lit(2))) is None

    def test_assert_true_fails_the_query(self, session: Session) -> None:
        with pytest.raises(Exception, match="id must be positive"):
            session.table("fx.plain").select(
                F.assert_true(F.col("id") < F.lit(0), F.lit("id must be positive")).alias("v")
            ).collect()

    def test_raise_error_fails_the_query(self, session: Session) -> None:
        with pytest.raises(Exception, match="boom"):
            session.table("fx.plain").select(F.raise_error(F.lit("boom")).alias("v")).collect()


class TestAggregatesThirdTranche:
    def test_every_some_and_any(self, session: Session) -> None:
        df = session.table("fx.plain")
        row = df.select(
            F.every(F.col("id") > F.lit(0)).alias("all_positive"),
            F.some(F.col("id") > F.lit(4)).alias("any_big"),
            F.any(F.col("id") > F.lit(99)).alias("any_huge"),
        ).collect()[0]
        assert row["all_positive"] is True
        assert row["any_big"] is True
        assert row["any_huge"] is False

    def test_bit_aggregates(self, session: Session) -> None:
        """fx.plain holds ids 1..5, so the AND of all is 0, the OR is 7, and the XOR
        of 1^2^3^4^5 is 1."""
        row = (
            session.table("fx.plain")
            .select(
                F.bit_and(F.col("id")).alias("a"),
                F.bit_or(F.col("id")).alias("o"),
                F.bit_xor(F.col("id")).alias("x"),
            )
            .collect()[0]
        )
        assert (row["a"], row["o"], row["x"]) == (0, 7, 1)

    def test_std_is_stddev(self, session: Session) -> None:
        df = session.table("fx.plain")
        row = df.select(
            F.std(F.col("id")).alias("s"), F.stddev(F.col("id")).alias("expected")
        ).collect()[0]
        assert row["s"] == row["expected"]

    def test_percentile_is_exact_and_interpolates(self, session: Session) -> None:
        """ids are 1..5, so the median is 3 and the 25th percentile is 2."""
        row = (
            session.table("fx.plain")
            .select(
                F.percentile(F.col("id"), 0.5).alias("median"),
                F.percentile(F.col("id"), 0.25).alias("q1"),
            )
            .collect()[0]
        )
        assert row["median"] == 3.0
        assert row["q1"] == 2.0

    def test_percentile_rejects_a_percentage_outside_the_unit_interval(
        self, session: Session
    ) -> None:
        with pytest.raises(EngineValueError):
            F.percentile(F.col("id"), 50)

    def test_array_agg_is_collect_list(self, session: Session) -> None:
        values = session.table("fx.plain").select(F.array_agg(F.col("id")).alias("v")).collect()
        assert sorted(values[0]["v"]) == [1, 2, 3, 4, 5]

    def test_regression_aggregates(self, session: Session) -> None:
        """`amount` is 10.0, 20.5, 30.25, NULL, 50.0 against ids 1..5, so the two
        rows with a NULL on either side are dropped: regr_count is 4."""
        df = session.table("fx.plain")
        row = df.select(
            F.regr_count(F.col("amount"), F.col("id")).alias("n"),
            F.regr_avgx(F.col("amount"), F.col("id")).alias("avgx"),
            F.regr_slope(F.col("amount"), F.col("id")).alias("slope"),
            F.regr_r2(F.col("amount"), F.col("id")).alias("r2"),
        ).collect()[0]
        assert row["n"] == 4
        assert row["avgx"] == pytest.approx((1 + 2 + 3 + 5) / 4)
        assert row["slope"] > 0
        assert 0.0 <= row["r2"] <= 1.0

    def test_the_remaining_regression_aggregates_run(self, session: Session) -> None:
        df = session.table("fx.plain")
        row = df.select(
            F.regr_avgy(F.col("amount"), F.col("id")).alias("avgy"),
            F.regr_intercept(F.col("amount"), F.col("id")).alias("intercept"),
            F.regr_sxx(F.col("amount"), F.col("id")).alias("sxx"),
            F.regr_sxy(F.col("amount"), F.col("id")).alias("sxy"),
            F.regr_syy(F.col("amount"), F.col("id")).alias("syy"),
        ).collect()[0]
        assert row["avgy"] == pytest.approx((10.0 + 20.5 + 30.25 + 50.0) / 4)
        assert row["sxx"] > 0
        assert row["syy"] > 0


class TestArraysSecondTranche:
    def test_array_append_and_prepend(self, session: Session) -> None:
        base = F.array(F.lit(1), F.lit(2))
        assert _one(session, F.array_append(base, 3)) == [1, 2, 3]
        assert _one(session, F.array_prepend(base, 0)) == [0, 1, 2]

    def test_array_compact_drops_nulls(self, session: Session) -> None:
        column = F.array(F.lit(1), F.lit(None), F.lit(2))
        assert _one(session, F.array_compact(column)) == [1, 2]

    def test_array_except_removes_and_deduplicates(self, session: Session) -> None:
        """The reference engine's `array_except` de-duplicates what it keeps."""
        left = F.array(F.lit(1), F.lit(2), F.lit(3), F.lit(2))
        right = F.array(F.lit(2))
        assert _one(session, F.array_except(left, right)) == [1, 3]

    def test_array_repeat(self, session: Session) -> None:
        assert _one(session, F.array_repeat(F.lit("ab"), F.lit(3))) == ["ab", "ab", "ab"]

    def test_array_size_is_null_for_a_null_array(self, session: Session) -> None:
        """The one thing that separates `array_size` from `size`, which answers -1."""
        assert _one(session, F.array_size(F.array(F.lit(1), F.lit(2)))) == 2
        assert _one(session, F.array_size(F.lit(None))) is None

    def test_cardinality_is_size(self, session: Session) -> None:
        column = F.array(F.lit(1), F.lit(2), F.lit(3))
        assert _one(session, F.cardinality(column)) == 3

    def test_get_is_zero_indexed(self, session: Session) -> None:
        """`get` counts from 0 and `element_at` counts from 1 -- both are the reference engine's."""
        column = F.array(F.lit(10), F.lit(20), F.lit(30))
        assert _one(session, F.get(column, F.lit(0))) == 10
        assert _one(session, F.element_at(column, F.lit(1))) == 10

    def test_get_out_of_range_is_null(self, session: Session) -> None:
        column = F.array(F.lit(10), F.lit(20))
        assert _one(session, F.get(column, F.lit(5))) is None

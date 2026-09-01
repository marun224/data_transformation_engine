"""`df.na.*` and `df.stat.*` against the local fixture catalog.

`fx.plain` is the right table for both: `id` is `1..5`, `vendor` is `a, b, a, c, NULL`
and `amount` is `10.0, 20.5, 30.25, NULL, 50.0`. Two columns with nulls in *different*
rows is what makes `drop(how=...)` and `thresh` distinguishable at all -- with the nulls
in one row, "any" and "all" would agree and the tests would prove nothing.

**The type rule is the point of the fill tests.** `fill(0)` must leave a NULL string
alone and `fill("?")` must leave a NULL double alone. Getting that wrong is a wrong
answer, not an error, and a test that only ever filled a numeric column would pass
either way -- so each fill test asserts on *both* columns, including the one that must
not have changed.

Every assertion is on a value, per the rule Phase 3 established.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from icetl.errors import AnalysisException, EngineTypeError, EngineValueError

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def rows_of(df: DataFrame) -> list[tuple[Any, ...]]:
    return sorted((tuple(row) for row in df.collect()), key=lambda t: str(t[0]))


class TestDrop:
    def test_any_drops_a_row_with_any_null(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [row[0] for row in df.na.drop().collect()] == [1, 2, 3]

    def test_all_drops_only_rows_that_are_entirely_null(self, session: Session) -> None:
        """No row here is all-null, so `how='all'` drops nothing -- which is the point."""
        df = session.table("fx.plain")
        assert df.na.drop(how="all").count() == 5

    def test_subset_narrows_which_columns_count(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [row[0] for row in df.na.drop(subset=["amount"]).collect()] == [1, 2, 3, 5]
        assert [row[0] for row in df.na.drop(subset="vendor").collect()] == [1, 2, 3, 4]

    def test_thresh_counts_non_nulls_and_overrides_how(self, session: Session) -> None:
        """`thresh` wins over `how`: every row here has at least two non-null values."""
        df = session.table("fx.plain")
        assert df.na.drop(how="any", thresh=2).count() == 5
        assert df.na.drop(how="any", thresh=3).count() == 3

    def test_an_unknown_subset_column_is_ignored_not_refused(self, session: Session) -> None:
        """`subset` filters the columns; it does not assert about them."""
        df = session.table("fx.plain")
        assert df.na.drop(subset=["nope"]).count() == 5

    def test_dropna_is_the_same_method(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert rows_of(df.dropna()) == rows_of(df.na.drop())

    def test_how_is_checked(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="'any' or 'all'"):
            df.na.drop(how="some")


class TestFill:
    def test_a_number_fills_numeric_columns_only(self, session: Session) -> None:
        """The row that matters is id 4: amount filled, vendor untouched."""
        df = session.table("fx.plain")
        row = next(r for r in df.na.fill(0).collect() if r[0] == 4)
        assert tuple(row) == (4, "c", 0.0)
        # ... and id 5, whose vendor is NULL, must still be NULL.
        assert next(r for r in df.na.fill(0).collect() if r[0] == 5)[1] is None

    def test_a_string_fills_string_columns_only(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert next(r for r in df.na.fill("?").collect() if r[0] == 5)[1] == "?"
        assert next(r for r in df.na.fill("?").collect() if r[0] == 4)[2] is None

    def test_a_bool_reaches_neither_here(self, session: Session) -> None:
        """`bool` is an `int` subclass in Python; it must not leak into the numeric branch."""
        df = session.table("fx.plain")
        assert rows_of(df.na.fill(True)) == rows_of(df)

    def test_the_dict_form_names_its_own_columns(self, session: Session) -> None:
        df = session.table("fx.plain")
        filled = df.na.fill({"vendor": "z", "amount": -1.0})
        assert next(r for r in filled.collect() if r[0] == 4)[2] == -1.0
        assert next(r for r in filled.collect() if r[0] == 5)[1] == "z"

    def test_subset_limits_the_scalar_form(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert next(r for r in df.na.fill(0, subset=["id"]).collect() if r[0] == 4)[2] is None

    def test_the_dict_form_refuses_an_unknown_column(self, session: Session) -> None:
        """Unlike `subset`, a dict key naming a missing column can only be a mistake."""
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="which this frame does not have"):
            df.na.fill({"nope": 0})

    def test_fillna_is_the_same_method(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert rows_of(df.fillna(0)) == rows_of(df.na.fill(0))


class TestReplace:
    def test_one_value_for_another(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert [r[1] for r in rows_of(df.na.replace("a", "A"))] == ["A", "b", "A", "c", None]

    def test_the_list_form_is_pairwise(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.na.replace(["a", "b"], ["X", "Y"])
        assert [r[1] for r in rows_of(out)] == ["X", "Y", "X", "c", None]

    def test_the_dict_form_says_the_same_thing(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert rows_of(df.na.replace({"a": "X", "b": "Y"})) == rows_of(
            df.na.replace(["a", "b"], ["X", "Y"])
        )

    def test_a_replacement_only_reaches_columns_of_its_type(self, session: Session) -> None:
        """A numeric pair reaches `id`; `vendor` comes back untouched beside it."""
        df = session.table("fx.plain")
        before = {r[0]: r[1] for r in df.collect()}
        out = {r[0]: r[1] for r in df.na.replace(1, 99).collect()}
        assert sorted(out) == [2, 3, 4, 5, 99]
        # Every vendor is exactly what it was; only the key it hangs off moved.
        assert out[99] == before[1]
        assert {k: v for k, v in out.items() if k != 99} == {
            k: v for k, v in before.items() if k != 1
        }

    def test_replacing_with_null_is_allowed(self, session: Session) -> None:
        """Two `a`s become NULL, joining the one that was already NULL: three in all."""
        df = session.table("fx.plain")
        out = df.na.replace(["a"], [None])
        assert sum(row[1] is None for row in out.collect()) == 3
        assert sorted(row[1] for row in out.collect() if row[1] is not None) == ["b", "c"]

    def test_mismatched_list_lengths_are_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="same length"):
            df.na.replace(["a", "b"], ["X"])

    def test_replace_is_also_on_the_frame(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert rows_of(df.replace("a", "A")) == rows_of(df.na.replace("a", "A"))


class TestApproxQuantile:
    def test_exact_quantiles_are_observed_values(self, session: Session) -> None:
        """`relativeError=0` returns values from the data, never an interpolation."""
        df = session.table("fx.plain")
        assert df.stat.approxQuantile("amount", [0.0, 0.5, 1.0]) == [10.0, 20.5, 50.0]

    def test_a_list_of_columns_returns_a_list_per_column(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.stat.approxQuantile(["id", "amount"], [0.5]) == [[3.0], [20.5]]

    def test_nulls_are_ignored(self, session: Session) -> None:
        """Four non-null amounts, so the maximum is the largest of those four."""
        df = session.table("fx.plain")
        assert df.stat.approxQuantile("amount", [1.0]) == [50.0]

    def test_probabilities_are_range_checked(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match=r"\[0, 1\]"):
            df.stat.approxQuantile("amount", [1.5])

    def test_a_non_numeric_column_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="needs a numeric column"):
            df.stat.approxQuantile("vendor", [0.5])


class TestCorrAndCov:
    def test_correlation_of_a_column_with_itself_is_one(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.stat.corr("id", "id") == pytest.approx(1.0)

    def test_covariance_is_the_sample_form(self, session: Session) -> None:
        """`covar_samp` divides by n-1. Ids 1..5 have sample variance 2.5, not 2.0."""
        df = session.table("fx.plain")
        assert df.stat.cov("id", "id") == pytest.approx(2.5)

    def test_an_unknown_method_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="pearson"):
            df.stat.corr("id", "id", method="spearman")

    def test_a_non_numeric_column_is_refused(self, session: Session) -> None:
        """DuckDB would cast and answer; a number that means nothing is worse than an error."""
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="needs a numeric column"):
            df.stat.corr("id", "vendor")


class TestCrosstab:
    def test_the_shape_is_labels_down_the_side_and_across_the_top(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.stat.crosstab("vendor", "id")
        assert out.columns == ["vendor_id", "1", "2", "3", "4", "5"]

    def test_an_absent_pair_counts_zero_not_null(self, session: Session) -> None:
        """Vendor `a` occurs at ids 1 and 3; everywhere else it is an observed zero."""
        df = session.table("fx.plain")
        row = next(r for r in df.stat.crosstab("vendor", "id").collect() if r[0] == "a")
        assert tuple(row) == ("a", 1, 0, 1, 0, 0)

    def test_null_is_labelled_the_same_on_both_axes(self, session: Session) -> None:
        df = session.table("fx.plain")
        labels = {row[0] for row in df.stat.crosstab("vendor", "id").collect()}
        assert "null" in labels


class TestFreqItems:
    def test_items_above_the_support_threshold(self, session: Session) -> None:
        """`a` occurs twice in five rows, so it clears 0.3; nothing else does."""
        df = session.table("fx.plain")
        out = df.stat.freqItems(["vendor"], support=0.3)
        assert out.columns == ["vendor_freqItems"]
        assert out.collect()[0][0] == ["a"]

    def test_nothing_clears_an_impossible_threshold(self, session: Session) -> None:
        df = session.table("fx.plain")
        assert df.stat.freqItems("vendor", support=0.9).collect()[0][0] == []

    def test_one_row_out_however_many_columns_in(self, session: Session) -> None:
        df = session.table("fx.plain")
        out = df.stat.freqItems(["vendor", "id"], support=0.3)
        assert out.columns == ["vendor_freqItems", "id_freqItems"]
        assert out.count() == 1

    def test_the_result_is_independent_of_its_source(self, session: Session) -> None:
        """Folded into a literal plan, so it reads no table and collects the same twice."""
        df = session.table("fx.plain")
        out = df.stat.freqItems(["vendor"], support=0.3)
        assert out._sources == {}
        assert out.collect() == out.collect()

    def test_support_is_range_checked(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineValueError, match="0 < support <= 1"):
            df.stat.freqItems("vendor", support=0.0)

    def test_an_unknown_column_is_refused(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(AnalysisException, match="does not have"):
            df.stat.freqItems("nope")


class TestSubsetTypeChecking:
    def test_subset_takes_strings(self, session: Session) -> None:
        df = session.table("fx.plain")
        with pytest.raises(EngineTypeError, match="column names"):
            df.na.drop(subset=[1])  # type: ignore[list-item]

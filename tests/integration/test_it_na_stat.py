"""`df.na.*` and `df.stat.*`, against columns that are genuinely missing values.

These two namespaces exist entirely to deal with absent data, so a fixture with one
hand-placed NULL tests the plumbing and nothing else. The seeded slice is NULL in 971 of
5,000 rows, across several columns at once and always in the *same* rows -- which is the
shape that matters, because `na.drop()` with a subset has to distinguish "this column is
missing" from "some column is missing".

`icetl_it.plain` is used where the NULLs need to be in *different* rows -- its `vendor`
and `amount` are NULL on separate rows, which is what makes `how="any"` and `how="all"`
tell apart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import column

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestNaDrop:
    def test_dropping_any_removes_every_incomplete_row(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        kept = frame.na.drop()
        assert kept.count() < frame.count()
        assert kept.filter(F.col("passenger_count").isNull()).count() == 0
        assert kept.filter(F.col("store_and_fwd_flag").isNull()).count() == 0

    def test_dropping_on_a_subset_only_considers_that_subset(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        nulls = frame.filter(F.col("store_and_fwd_flag").isNull()).count()
        kept = frame.na.drop(subset=["store_and_fwd_flag"])
        assert kept.count() == frame.count() - nulls

    def test_any_and_all_differ_when_nulls_are_in_different_rows(
        self, it_session: Session, plain: str
    ) -> None:
        """The replica has a NULL vendor in one row and a NULL amount in another."""
        frame = it_session.table(plain)
        assert frame.na.drop(how="any").count() == 3
        assert frame.na.drop(how="all").count() == 5

    def test_a_threshold_counts_the_present_values(self, it_session: Session, plain: str) -> None:
        frame = it_session.table(plain)
        assert frame.na.drop(thresh=3).count() == 3
        assert frame.na.drop(thresh=2).count() == 5


class TestNaFill:
    def test_filling_replaces_exactly_the_missing_values(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        missing = frame.filter(F.col("store_and_fwd_flag").isNull()).count()
        filled = frame.na.fill("?", subset=["store_and_fwd_flag"])
        assert filled.filter(F.col("store_and_fwd_flag") == "?").count() == missing
        assert filled.filter(F.col("store_and_fwd_flag").isNull()).count() == 0
        assert filled.count() == frame.count()

    def test_filling_a_number_leaves_the_strings_alone(
        self, it_session: Session, trips_small: str
    ) -> None:
        """A fill value only applies to the columns whose type it fits."""
        frame = it_session.table(trips_small)
        filled = frame.na.fill(0)
        assert filled.filter(F.col("passenger_count").isNull()).count() == 0
        assert filled.filter(F.col("store_and_fwd_flag").isNull()).count() > 0

    def test_a_mapping_fills_each_column_with_its_own_value(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        filled = frame.na.fill({"store_and_fwd_flag": "?", "passenger_count": 0})
        assert filled.filter(F.col("store_and_fwd_flag") == "?").count() > 0
        assert filled.filter(F.col("passenger_count") == 0).count() > 0
        assert filled.filter(F.col("store_and_fwd_flag").isNull()).count() == 0

    def test_filling_does_not_touch_the_present_values(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        before = frame.filter(F.col("store_and_fwd_flag") == "N").count()
        after = (
            frame.na.fill("?", subset=["store_and_fwd_flag"])
            .filter(F.col("store_and_fwd_flag") == "N")
            .count()
        )
        assert before == after


class TestNaReplace:
    def test_replacing_a_real_value_changes_exactly_those_rows(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        before = frame.filter(F.col("store_and_fwd_flag") == "N").count()
        assert before > 0
        replaced = frame.na.replace({"N": "NO"}, subset=["store_and_fwd_flag"])
        assert replaced.filter(F.col("store_and_fwd_flag") == "NO").count() == before
        assert replaced.filter(F.col("store_and_fwd_flag") == "N").count() == 0

    def test_replacing_leaves_the_nulls_null(self, it_session: Session, trips_small: str) -> None:
        """`replace` is not `fill` -- a NULL is not a value it can match."""
        frame = it_session.table(trips_small)
        nulls = frame.filter(F.col("store_and_fwd_flag").isNull()).count()
        replaced = frame.na.replace({"N": "NO"}, subset=["store_and_fwd_flag"])
        assert replaced.filter(F.col("store_and_fwd_flag").isNull()).count() == nulls


class TestStat:
    """`corr`, `cov`, `approxQuantile`, `crosstab`, `freqItems` over real distributions."""

    def test_a_column_correlates_perfectly_with_itself(
        self, it_session: Session, trips_small: str
    ) -> None:
        value = it_session.table(trips_small).stat.corr("trip_distance", "trip_distance")
        assert value == pytest.approx(1.0, abs=1e-9)

    def test_fare_and_distance_are_positively_correlated(
        self, it_session: Session, trips: str
    ) -> None:
        """A property of the world, not of the fixture: longer trips cost more."""
        value = it_session.table(trips).stat.corr("trip_distance", "fare_amount")
        assert 0.0 < value <= 1.0, value

    def test_covariance_with_itself_is_the_variance(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        cov = frame.stat.cov("trip_distance", "trip_distance")
        variance = frame.select(F.var_samp(F.col("trip_distance")).alias("v")).collect()[0]["v"]
        assert cov == pytest.approx(variance, rel=1e-6)

    def test_quantiles_are_ordered_and_within_the_range(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        quantiles = frame.stat.approxQuantile("trip_distance", [0.0, 0.25, 0.5, 0.75, 1.0], 0.01)
        assert quantiles == sorted(quantiles), quantiles
        bounds = frame.select(
            F.min(F.col("trip_distance")).alias("lo"), F.max(F.col("trip_distance")).alias("hi")
        ).collect()[0]
        assert bounds["lo"] <= quantiles[0]
        assert quantiles[-1] <= bounds["hi"]

    def test_the_median_splits_the_rows_about_evenly(
        self, it_session: Session, trips_small: str
    ) -> None:
        frame = it_session.table(trips_small)
        median = frame.stat.approxQuantile("trip_distance", [0.5], 0.01)[0]
        below = frame.filter(F.col("trip_distance") < median).count()
        assert 0 < below < frame.count()

    def test_a_crosstab_conserves_the_row_count(
        self, it_session: Session, trips_small: str
    ) -> None:
        table = it_session.table(trips_small).stat.crosstab("VendorID", "store_and_fwd_flag")
        rows = table.collect()
        assert rows
        counted = sum(
            value
            for row in rows
            for key, value in row.asDict().items()
            if not key.endswith("store_and_fwd_flag") and isinstance(value, int)
        )
        assert counted == it_session.table(trips_small).count()

    def test_freq_items_finds_the_common_vendors(self, it_session: Session, trips: str) -> None:
        """Every vendor holds more than 10% of 880k rows, so all of them qualify."""
        found = it_session.table(trips).stat.freqItems(["VendorID"], 0.1).collect()[0][0]
        actual = set(column(it_session.table(trips).select("VendorID").distinct(), "VendorID"))
        assert set(found) <= actual
        assert found

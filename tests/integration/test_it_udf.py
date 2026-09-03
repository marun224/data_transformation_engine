"""Python UDFs applied to real rows.

The interesting part is not that a Python function runs -- it is what happens at the
edges, and real data supplies edges a fixture does not: a column that is NULL in 971 of
5,000 rows, doubles that are exactly 0.0, and strings that are sometimes absent.

Two behaviours from `divergence.md` are the point of this module:

  * **A NULL argument does not reach the function.** Neither of DuckDB's own null-handling
    modes was usable on its own (FINDINGS 2.9), so icetl guards the call: a UDF sees only
    non-NULL arguments, and a NULL in means a NULL out. Without that, every UDF ever
    written here would need its own `if x is None` and most would forget.
  * **A UDF is opaque to the optimizer.** Nothing about a Python function can be pushed
    into a scan, so a filter over one must not prune -- claiming otherwise would be a
    wrong answer, not a slow one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.sql import functions as F
from tests.integration.helpers import column, scan_of

if TYPE_CHECKING:
    from icetl.sql.session import Session

pytestmark = pytest.mark.integration


class TestScalarUdfs:
    def test_a_udf_is_applied_to_every_row(self, session: Session, trips_small: str) -> None:
        session.udf.register("double_it", lambda x: x * 2, "double")
        rows = session.sql(
            f"SELECT trip_distance, double_it(trip_distance) AS doubled FROM {trips_small}"
        ).collect()
        assert rows
        for row in rows:
            if row["trip_distance"] is not None:
                assert row["doubled"] == pytest.approx(row["trip_distance"] * 2)

    def test_a_udf_works_on_the_dataframe_surface_too(
        self, session: Session, trips_small: str
    ) -> None:
        """P1: one registration, reachable from both surfaces.

        `register` hands back the callable, so the *same* registration answers
        `Session.sql()` and the DataFrame API -- not two functions that happen to
        agree.
        """
        doubled = session.udf.register("double_it", lambda x: x * 2, "double")
        via_sql = session.sql(
            f"SELECT sum(double_it(trip_distance)) AS t FROM {trips_small}"
        ).collect()[0]["t"]
        via_frame = (
            session.table(trips_small)
            .select(doubled("trip_distance").alias("d"))
            .select(F.sum(F.col("d")).alias("t"))
            .collect()[0]["t"]
        )
        assert via_sql == pytest.approx(via_frame, rel=1e-9)

    def test_f_udf_needs_an_active_session(self, session: Session) -> None:
        """The documented constraint, pinned.

        `F.udf` registers against the *active* session, and a session built directly
        -- as every fixture here does, so tests can run in any order -- is not it.
        `session.udf.register` is the form that needs no ambient state.
        """
        from icetl.errors import UnsupportedFeatureError

        with pytest.raises(UnsupportedFeatureError, match="without a Session"):
            F.udf(lambda x: x * 2, "double")

    def test_a_string_udf_runs_over_real_strings(self, session: Session, trips_small: str) -> None:
        session.udf.register("shout", lambda s: s.upper() + "!", "string")
        values = {
            v
            for v in column(
                session.sql(f"SELECT shout(store_and_fwd_flag) AS v FROM {trips_small}"), "v"
            )
        }
        assert values <= {"N!", "Y!", None}
        assert values - {None}, "the UDF never ran on a non-null row"

    def test_a_udf_taking_two_columns(self, session: Session, trips_small: str) -> None:
        session.udf.register("ratio", lambda a, b: a / b if b else None, "double")
        rows = session.sql(
            f"SELECT trip_distance, fare_amount, ratio(fare_amount, trip_distance) AS r "
            f"FROM {trips_small} LIMIT 200"
        ).collect()
        assert rows
        for row in rows:
            if row["trip_distance"]:
                assert row["r"] == pytest.approx(row["fare_amount"] / row["trip_distance"])


class TestNullHandling:
    """divergence.md: a NULL argument does not reach the function."""

    def test_the_function_never_sees_a_null(self, session: Session, trips_small: str) -> None:
        """If a NULL reached it, this UDF would raise rather than return.

        The column is NULL in roughly a fifth of the rows, so a guard that did not
        work would fail loudly and immediately.
        """

        def must_not_be_null(value: str) -> str:
            assert value is not None, "a NULL reached the UDF"
            return value.lower()

        session.udf.register("lower_it", must_not_be_null, "string")
        rows = session.sql(f"SELECT lower_it(store_and_fwd_flag) AS v FROM {trips_small}").collect()
        assert len(rows) == session.table(trips_small).count()

    def test_a_null_in_means_a_null_out(self, session: Session, trips_small: str) -> None:
        session.udf.register("lower_it", lambda s: s.lower(), "string")
        nulls_in = session.table(trips_small).filter(F.col("store_and_fwd_flag").isNull()).count()
        assert nulls_in > 0
        nulls_out = sum(
            1
            for v in column(
                session.sql(f"SELECT lower_it(store_and_fwd_flag) AS v FROM {trips_small}"), "v"
            )
            if v is None
        )
        assert nulls_out == nulls_in

    def test_the_non_null_rows_still_get_their_answer(
        self, session: Session, trips_small: str
    ) -> None:
        """The guard must skip NULLs, not skip everything."""
        session.udf.register("lower_it", lambda s: s.lower(), "string")
        present = session.table(trips_small).filter(F.col("store_and_fwd_flag").isNotNull()).count()
        answered = sum(
            1
            for v in column(
                session.sql(f"SELECT lower_it(store_and_fwd_flag) AS v FROM {trips_small}"), "v"
            )
            if v is not None
        )
        assert answered == present


class TestVectorisedUdfs:
    def test_a_vectorised_udf_gives_the_same_answers(
        self, session: Session, trips_small: str
    ) -> None:
        import pandas as pd

        session.udf.register("scalar_double", lambda x: x * 2, "double")
        session.udf.registerVectorised("vector_double", lambda s: pd.Series(s) * 2, "double")
        row = session.sql(
            f"SELECT sum(scalar_double(trip_distance)) AS a, "
            f"sum(vector_double(trip_distance)) AS b FROM {trips_small}"
        ).collect()[0]
        assert row["a"] == pytest.approx(row["b"], rel=1e-9)

    def test_a_vectorised_udf_runs_over_every_row(self, session: Session, trips_small: str) -> None:
        import pandas as pd

        session.udf.registerVectorised("vector_double", lambda s: pd.Series(s) * 2, "double")
        counted = session.sql(
            f"SELECT count(vector_double(trip_distance)) AS n FROM {trips_small}"
        ).collect()[0]["n"]
        assert (
            counted == session.table(trips_small).filter(F.col("trip_distance").isNotNull()).count()
        )


class TestRegistration:
    def test_a_registered_udf_is_listed(self, session: Session) -> None:
        session.udf.register("double_it", lambda x: x * 2, "double")
        assert "double_it" in session.udf.registered

    def test_re_registering_a_name_replaces_it(self, session: Session, trips_small: str) -> None:
        """divergence.md: the second registration wins, silently, as the reference does."""
        session.udf.register("f", lambda x: x * 2, "double")
        first = session.sql(f"SELECT sum(f(trip_distance)) AS t FROM {trips_small}").collect()[0][
            "t"
        ]
        session.udf.register("f", lambda x: x * 3, "double")
        second = session.sql(f"SELECT sum(f(trip_distance)) AS t FROM {trips_small}").collect()[0][
            "t"
        ]
        assert second == pytest.approx(first * 1.5, rel=1e-9)

    def test_a_udf_does_not_leak_between_sessions(
        self, session: Session, it_session: Session, trips_small: str
    ) -> None:
        """Registration is session state, so another session must not see it."""
        session.udf.register("only_mine", lambda x: x, "double")
        assert "only_mine" not in it_session.udf.registered

    def test_the_return_type_is_declared_not_inferred(
        self, session: Session, trips_small: str
    ) -> None:
        """divergence.md: nothing here guesses what a Python function returns."""
        session.udf.register("as_text", lambda x: str(x), "string")
        types = dict(session.sql(f"SELECT as_text(trip_distance) AS v FROM {trips_small}").dtypes)
        assert types["v"] == "string"


class TestAUdfNeverPrunes:
    """A Python function is opaque, so a filter over one cannot be pushed."""

    def test_a_filter_on_a_udf_does_not_prune_files(self, session: Session, trips: str) -> None:
        session.udf.register("is_long", lambda d: d > 5, "boolean")
        frame = session.sql(f"SELECT * FROM {trips} WHERE is_long(trip_distance)")
        scan = scan_of(frame)
        assert scan.files_total is not None
        assert scan.files_scanned == scan.files_total, (
            "a UDF predicate was pushed into the scan, which it cannot be"
        )

    def test_the_udf_filter_still_returns_the_right_rows(
        self, session: Session, trips_small: str
    ) -> None:
        """Not pruning is correct; not filtering would not be."""
        session.udf.register("is_long", lambda d: d > 5, "boolean")
        via_udf = session.sql(
            f"SELECT count(*) AS n FROM {trips_small} WHERE is_long(trip_distance)"
        ).collect()[0]["n"]
        directly = session.table(trips_small).filter(F.col("trip_distance") > 5).count()
        assert via_udf == directly

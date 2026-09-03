"""Phase 11: Python UDFs, on both surfaces.

One registration serves `df.select(f(...))` and `session.sql("SELECT f(...)")`,
because a UDF is an ordinary function call in the plan and neither surface can tell
which one built it (P1). `TestBothSurfaces` is the test of that claim.

Three things about the wiring are easy to get wrong and each has its own class here:

  * **Two connections.** The engine executes on one DuckDB connection and the schema
    analyzer binds on another. A UDF registered only on the first fails at *analysis*
    — before a row is read, with an error naming a catalog rather than a function.
  * **NULL does not reach the function**, unless asked. The reference calls a UDF
    with `None`; DuckDB's default skips it. Matching the reference needs SPECIAL null
    handling, and SPECIAL over `read_parquet` invents a NULL call that the data never
    contained (FINDINGS §2.9) — which would crash the ordinary `lambda x: x * 2` on
    every Iceberg scan. So DuckDB's default stands and `callOnNull=True` opts in.
    `TestNullHandling` pins both halves.
  * **A UDF must never prune.** Iceberg cannot evaluate Python, so a predicate over a
    UDF has to be reported as unpushed and read every file. Pruning by a predicate the
    catalog cannot evaluate would be a wrong answer, which is `TestAUdfNeverPrunes`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    EngineValueError,
    QueryExecutionException,
)
from icetl.sql import functions as F
from icetl.sql.udf import duckdb_type_of
from icetl.types import ArrayType, LongType, StringType

if TYPE_CHECKING:
    from icetl.sql.session import Session


def double(value: int | None) -> int | None:
    return None if value is None else value * 2


class TestBothSurfaces:
    def test_one_registration_serves_both(self, session: Session) -> None:
        udf = session.udf.register("dbl", double, "bigint")
        api = [row[0] for row in session.table("fx.plain").select(udf("id")).collect()]
        sql = [row[0] for row in session.sql("SELECT dbl(id) FROM fx.plain").collect()]
        assert sorted(api) == sorted(sql) == [2, 4, 6, 8, 10]

    def test_the_returned_callable_builds_a_column(self, session: Session) -> None:
        udf = session.udf.register("dbl2", double, "bigint")
        frame = session.table("fx.plain").select(udf("id").alias("d"))
        assert frame.columns == ["d"]
        assert sorted(row["d"] for row in frame.collect()) == [2, 4, 6, 8, 10]

    def test_the_declared_return_type_is_the_column_type(self, session: Session) -> None:
        """DuckDB needs the type before the first row, so it is declared, not inferred."""
        session.udf.register("as_text", lambda value: str(value), "string")
        frame = session.sql("SELECT as_text(id) AS t FROM fx.plain")
        assert frame.schema.simpleString() == "struct<t:string>"

    def test_the_default_return_type_is_string(self, session: Session) -> None:
        """As the reference's is. Not inferred: a wrong guess is a mistyped column."""
        udf = session.udf.register("plain_default", lambda value: f"{value}")
        assert udf.returnType == StringType()
        assert (
            session.sql("SELECT plain_default(id) FROM fx.plain").schema.fields[0].dataType
            == StringType()
        )

    def test_a_udf_of_two_arguments(self, session: Session) -> None:
        session.udf.register("addup", lambda a, b: (a or 0) + (b or 0), "bigint")
        rows = session.sql("SELECT addup(id, id) FROM fx.plain").collect()
        assert sorted(row[0] for row in rows) == [2, 4, 6, 8, 10]

    def test_a_udf_over_a_string_column(self, session: Session) -> None:
        session.udf.register("shout", lambda v: f"<{v}>", "string")
        rows = session.sql("SELECT shout(vendor) FROM fx.plain WHERE id = 1").collect()
        assert rows[0][0].startswith("<")

    def test_a_udf_composes_with_the_function_library(self, session: Session) -> None:
        udf = session.udf.register("dbl3", double, "bigint")
        frame = session.table("fx.plain").select((udf("id") + F.lit(1)).alias("n"))
        assert sorted(row["n"] for row in frame.collect()) == [3, 5, 7, 9, 11]


class TestTheAnalyzerConnectionToo:
    """A UDF registered only on the engine fails before a single row is read."""

    def test_the_schema_binds_without_running_the_query(self, session: Session) -> None:
        udf = session.udf.register("dbl4", double, "bigint")
        frame = session.table("fx.plain").select(udf("id").alias("d"))
        # `.schema` goes through the analyzer alone -- no execution at all.
        assert frame.schema.simpleString() == "struct<d:bigint>"

    def test_explain_works_without_executing(self, session: Session) -> None:
        udf = session.udf.register("dbl5", double, "bigint")
        frame = session.table("fx.plain").select(udf("id"))
        assert "DBL5" in frame._explain_text(verbose=False).upper()

    def test_an_unregistered_name_is_an_analysis_error(self, session: Session) -> None:
        with pytest.raises(AnalysisException):
            _ = session.sql("SELECT never_registered(id) FROM fx.plain").columns


class TestNullHandling:
    """A NULL argument does not reach the function unless `callOnNull` asks.

    A divergence chosen under duress, not preference: see FINDINGS §2.9. The opt-in
    exists because "turn NULL into a default" is a real thing to write, and such a
    function handles `None` by construction — which is what makes the spurious call
    harmless for exactly the functions that want it.
    """

    def test_by_default_the_function_is_not_called_for_a_null(self, session: Session) -> None:
        seen: list[object] = []

        def probe(value: object) -> str:
            seen.append(value)
            return "called"

        session.udf.register("probe", probe, "string")
        assert session.sql("SELECT probe(NULL)").collect()[0][0] is None
        assert seen == []

    def test_a_naive_udf_survives_a_real_table(self, session: Session) -> None:
        """The case that decided the default. `vendor` has a NULL in it.

        Under SPECIAL this raises `TypeError: unsupported operand` -- and would raise
        even on a column with no NULLs at all, because SPECIAL over `read_parquet`
        invents one.
        """
        session.udf.register("naive", lambda v: v * 2, "string")
        rows = session.sql("SELECT naive(vendor) AS v FROM fx.plain").collect()
        assert sorted(row["v"] or "" for row in rows) == ["", "aa", "aa", "bb", "cc"]

    def test_a_naive_numeric_udf_survives_too(self, session: Session) -> None:
        session.udf.register("naive_n", lambda v: v * 2, "bigint")
        rows = session.sql("SELECT naive_n(id) FROM fx.plain").collect()
        assert sorted(row[0] for row in rows) == [2, 4, 6, 8, 10]

    def test_call_on_null_lets_the_function_see_none(self, session: Session) -> None:
        seen: list[object] = []

        def probe(value: object) -> str:
            seen.append(value)
            return "called"

        session.udf.register("probe_on_null", probe, "string", callOnNull=True)
        assert session.sql("SELECT probe_on_null(NULL)").collect()[0][0] == "called"
        assert None in seen

    def test_call_on_null_can_replace_a_null_with_a_value(self, session: Session) -> None:
        """What the opt-in is for, and it must handle None to be worth opting into."""
        session.udf.register("or_zero", lambda v: 0 if v is None else v, "bigint", callOnNull=True)
        assert session.sql("SELECT or_zero(NULL)").collect()[0][0] == 0

    def test_call_on_null_over_a_real_table(self, session: Session) -> None:
        """`amount` holds a NULL; the default is substituted for it and nothing else."""
        session.udf.register(
            "amount_or", lambda v: -1.0 if v is None else v, "double", callOnNull=True
        )
        rows = session.sql("SELECT amount_or(amount) AS a FROM fx.plain").collect()
        assert sorted(row["a"] for row in rows) == [-1.0, 10.0, 20.5, 30.25, 50.0]

    def test_a_udf_may_still_return_null(self, session: Session) -> None:
        session.udf.register("nuller", lambda v: None, "bigint")
        assert session.sql("SELECT nuller(1)").collect()[0][0] is None


class TestAUdfNeverPrunes:
    """Iceberg cannot evaluate Python, so a UDF predicate must read every file."""

    def test_a_udf_predicate_is_reported_unpushed(self, session: Session) -> None:
        udf = session.udf.register("dbl6", double, "bigint")
        frame = session.table("fx.partitioned").filter(udf("id") > 6)
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        scan = compiled.scans[0]
        assert scan.pushed_filter is None
        assert scan.unpushed_filters
        assert scan.files_scanned == scan.files_total == 3

    def test_the_udf_predicate_still_filters(self, session: Session) -> None:
        """Unpushed is not unapplied: DuckDB evaluates it, so the answer is right."""
        udf = session.udf.register("dbl7", double, "bigint")
        rows = session.table("fx.plain").filter(udf("id") > 6).select("id").collect()
        assert sorted(row[0] for row in rows) == [4, 5]

    def test_a_real_predicate_beside_a_udf_still_prunes(self, session: Session) -> None:
        udf = session.udf.register("dbl8", double, "bigint")
        frame = session.table("fx.partitioned").filter(
            (F.col("as_at_date") == "2026-08-16") & (udf("id") > 0)
        )
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        assert compiled.scans[0].files_scanned == 1
        assert len(frame.collect()) == 4


class TestVectorised:
    """`pandas_udf`: called once per vector, through DuckDB's Arrow UDF protocol."""

    def test_a_vectorised_udf_returns_the_same_values(self, session: Session) -> None:
        session.udf.registerVectorised("vdbl", lambda series: series * 2, "bigint")
        rows = session.sql("SELECT vdbl(id) FROM fx.plain").collect()
        assert sorted(row[0] for row in rows) == [2, 4, 6, 8, 10]

    def test_it_is_called_once_per_vector_not_once_per_row(self, session: Session) -> None:
        """The whole reason it exists, and observable by counting calls."""
        calls: list[int] = []

        def vectorised(series: object) -> object:
            calls.append(len(series))  # type: ignore[arg-type]
            return series * 3  # type: ignore[operator]

        session.udf.registerVectorised("vtriple", vectorised, "bigint")
        rows = session.sql("SELECT vtriple(id) FROM fx.plain").collect()
        assert sorted(row[0] for row in rows) == [3, 6, 9, 12, 15]
        # One call carrying every row, rather than one call per row. The count is the
        # claim; the vector length is DuckDB's business.
        assert len(calls) == 1 and calls[0] >= 5

    def test_the_function_receives_a_pandas_series(self, session: Session) -> None:
        """The reference's signature is the one people write, so it is the one given."""
        kinds: list[str] = []

        def vectorised(series: object) -> object:
            kinds.append(type(series).__name__)
            return series

        session.udf.registerVectorised("vkind", vectorised, "bigint")
        session.sql("SELECT vkind(id) FROM fx.plain").collect()
        assert kinds and set(kinds) == {"Series"}

    def test_a_vectorised_udf_over_strings(self, session: Session) -> None:
        """`vendor` holds a NULL, which arrives as a missing value in the Series."""
        session.udf.registerVectorised("vupper", lambda s: s.str.upper(), "string")
        rows = session.sql("SELECT vupper(vendor) AS v FROM fx.plain").collect()
        assert sorted(row["v"] or "" for row in rows) == ["", "A", "A", "B", "C"]


class TestRegistrationLifecycle:
    def test_re_registering_a_name_replaces_it(self, session: Session) -> None:
        """The reference allows it; DuckDB raises, so the old one is removed first."""
        session.udf.register("swap", lambda v: v * 2, "bigint")
        assert session.sql("SELECT swap(5)").collect()[0][0] == 10
        session.udf.register("swap", lambda v: v * 10, "bigint")
        assert session.sql("SELECT swap(5)").collect()[0][0] == 50

    def test_re_registering_with_a_new_return_type_is_not_cached(self, session: Session) -> None:
        """Identical SQL, different schema -- the case a SQL-keyed cache gets wrong."""
        session.udf.register("shifty", lambda v: v, "bigint")
        assert (
            session.sql("SELECT shifty(id) AS s FROM fx.plain").schema.fields[0].dataType
            == LongType()
        )
        session.udf.register("shifty", lambda v: str(v), "string")
        assert (
            session.sql("SELECT shifty(id) AS s FROM fx.plain").schema.fields[0].dataType
            == StringType()
        )

    def test_unregister_removes_it_from_both_connections(self, session: Session) -> None:
        session.udf.register("gone", lambda v: v, "bigint")
        assert session.sql("SELECT gone(1)").collect()[0][0] == 1
        assert session.udf.unregister("gone") is True
        with pytest.raises(AnalysisException):
            _ = session.sql("SELECT gone(1)").columns

    def test_unregistering_an_unknown_name_is_false(self, session: Session) -> None:
        assert session.udf.unregister("never_there") is False

    def test_the_registry_lists_what_is_registered(self, session: Session) -> None:
        session.udf.register("listed", lambda v: v, "bigint")
        assert "listed" in session.udf.registered
        assert session.udf.registered["listed"].returnType == LongType()

    def test_the_registry_copy_cannot_be_mutated_into_the_session(self, session: Session) -> None:
        session.udf.registered["smuggled"] = None  # type: ignore[assignment]
        assert "smuggled" not in session.udf.registered

    def test_the_same_registration_object_is_reused(self, session: Session) -> None:
        assert session.udf is session.udf


class TestComplexReturnTypes:
    def test_a_udf_returning_an_array(self, session: Session) -> None:
        udf = session.udf.register("pair", lambda v: [v, v], "array<bigint>")
        assert udf.returnType == ArrayType(LongType())
        assert session.sql("SELECT pair(3)").collect()[0][0] == [3, 3]

    def test_a_udf_returning_a_struct(self, session: Session) -> None:
        session.udf.register(
            "boxed", lambda v: {"n": v, "label": str(v)}, "struct<n:bigint,label:string>"
        )
        row = session.sql("SELECT boxed(7) AS b").collect()[0]
        assert row["b"]["n"] == 7
        assert row["b"]["label"] == "7"

    def test_a_udf_returning_a_map(self, session: Session) -> None:
        session.udf.register("mapped", lambda v: {"k": v}, "map<string,bigint>")
        assert session.sql("SELECT mapped(2) AS m").collect()[0]["m"] == {"k": 2}

    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            ("bigint", "BIGINT"),
            ("string", "TEXT"),
            ("array<string>", "TEXT[]"),
            ("map<string,bigint>", "MAP(TEXT, BIGINT)"),
            ("struct<a:bigint>", "STRUCT(a BIGINT)"),
            ("decimal(10,2)", "DECIMAL(10, 2)"),
        ],
    )
    def test_the_type_translation(self, spec: str, expected: str) -> None:
        assert duckdb_type_of(spec) == expected

    def test_a_datatype_object_works_as_well_as_a_string(self) -> None:
        assert duckdb_type_of(LongType()) == duckdb_type_of("bigint")


class TestRefusals:
    def test_a_non_callable_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.udf.register("bad", "not a function")  # type: ignore[arg-type]

    def test_an_empty_name_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.udf.register("", double, "bigint")

    def test_a_nonsense_return_type_is_refused(self, session: Session) -> None:
        with pytest.raises((EngineValueError, EngineTypeError, Exception)):
            session.udf.register("weird", double, "not_a_type")

    def test_a_non_type_return_type_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            duckdb_type_of(42)  # type: ignore[arg-type]

    def test_an_exception_in_the_udf_surfaces(self, session: Session) -> None:
        """It fails the query rather than quietly returning NULL, as the reference does."""

        def boom(value: object) -> object:
            raise ValueError("no")

        session.udf.register("boom", boom, "bigint")
        with pytest.raises(QueryExecutionException, match=r"(?i)udf"):
            session.sql("SELECT boom(1)").collect()


class TestUdfsWithTheRestOfTheEngine:
    def test_count_over_a_udf_projection_agrees_with_collect(self, session: Session) -> None:
        """The metadata fast path must not fire: a UDF is not a plain column."""
        udf = session.udf.register("dbl9", double, "bigint")
        frame = session.table("fx.plain").select(udf("id").alias("d"))
        assert frame.count() == len(frame.collect()) == 5

    def test_a_udf_result_streams(self, session: Session) -> None:
        udf = session.udf.register("dbl10", double, "bigint")
        frame = session.table("fx.plain").select(udf("id").alias("d"))
        assert sorted(row["d"] for row in frame.toLocalIterator()) == [2, 4, 6, 8, 10]

    def test_a_udf_in_a_group_by_aggregate(self, session: Session) -> None:
        """`id` has no NULLs, and under the default null handling none is invented.

        Under SPECIAL this same query called the UDF a sixth time with `None` and
        raised — FINDINGS §2.9, and the reason this test is worth having.
        """
        session.udf.register("bucket", lambda v: "hi" if v > 3 else "lo", "string")
        rows = session.sql(
            "SELECT bucket(id) AS b, count(*) AS n FROM fx.plain GROUP BY bucket(id)"
        ).collect()
        assert sorted((row["b"], row["n"]) for row in rows) == [("hi", 2), ("lo", 3)]

    def test_a_udf_writes_through_to_a_table(self, session: Session, catalog) -> None:  # type: ignore[no-untyped-def]
        import contextlib
        import uuid

        udf = session.udf.register("dbl11", double, "bigint")
        name = f"wr.udf_{uuid.uuid4().hex[:8]}"
        try:
            session.table("fx.plain").select(udf("id").alias("d")).write.saveAsTable(name)
            assert sorted(row[0] for row in session.table(name).collect()) == [2, 4, 6, 8, 10]
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

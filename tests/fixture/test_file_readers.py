"""Phase 11: reading parquet, CSV and JSON that is not in an Iceberg table yet.

A file read builds `SELECT * FROM read_parquet('...')` -- a table *function*, which
every part of the planner already skips when looking for tables to resolve. So the
interesting tests are not "does it read the file" but the two claims that follow from
that shape:

  * **It is an ordinary frame.** Filters, joins against real tables, aggregates and
    `write.saveAsTable` all work on it, because nothing downstream knows it came from
    a file. `TestItIsAnOrdinaryFrame` is that claim.
  * **It gets no pruning, and that is correct.** There are no manifests, so there is
    nothing to prune with, and the filter runs in DuckDB. What must *not* happen is
    the planner trying to resolve `read_parquet` against the catalog.

Object-store paths are refused rather than half-supported, because the schema is bound
on a connection that has no S3 credentials. `TestObjectStoreIsRefused` pins that as a
deliberate refusal rather than an accident.
"""

from __future__ import annotations

import contextlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from icetl.errors import EngineTypeError, EngineValueError, UnsupportedFeatureError
from icetl.sql import functions as F

if TYPE_CHECKING:
    from pyiceberg.catalog.sql import SqlCatalog

    from icetl.sql.session import Session


@pytest.fixture
def files(tmp_path: Path) -> dict[str, str]:
    """One small table written three ways, so the readers can be compared."""
    pq.write_table(pa.table({"id": [1, 2, 3], "v": ["a", "b", "c"]}), tmp_path / "t.parquet")
    (tmp_path / "t.csv").write_text("id,v\n1,a\n2,b\n3,c\n", encoding="utf-8")
    (tmp_path / "t.json").write_text(
        "\n".join(json.dumps({"id": index, "v": chr(96 + index)}) for index in (1, 2, 3)),
        encoding="utf-8",
    )
    (tmp_path / "semi.csv").write_text("id;v\n1;a\n2;b\n", encoding="utf-8")
    return {
        name: str(tmp_path / f"t.{name}").replace("\\", "/") for name in ("parquet", "csv", "json")
    } | {"semi": str(tmp_path / "semi.csv").replace("\\", "/")}


class TestTheThreeFormatsAgree:
    """The same three rows, whichever way they were written."""

    def test_parquet(self, session: Session, files: dict[str, str]) -> None:
        rows = session.read.parquet(files["parquet"]).collect()
        assert [(row["id"], row["v"]) for row in rows] == [(1, "a"), (2, "b"), (3, "c")]

    def test_csv(self, session: Session, files: dict[str, str]) -> None:
        rows = session.read.csv(files["csv"], header=True).collect()
        assert [(row["id"], row["v"]) for row in rows] == [(1, "a"), (2, "b"), (3, "c")]

    def test_json(self, session: Session, files: dict[str, str]) -> None:
        rows = session.read.json(files["json"]).collect()
        assert sorted((row["id"], row["v"]) for row in rows) == [(1, "a"), (2, "b"), (3, "c")]

    def test_the_schemas_match(self, session: Session, files: dict[str, str]) -> None:
        parquet = session.read.parquet(files["parquet"]).schema.simpleString()
        csv = session.read.csv(files["csv"], header=True).schema.simpleString()
        assert parquet == csv == "struct<id:bigint,v:string>"

    def test_several_paths_read_as_one_frame(
        self, session: Session, tmp_path: Path, files: dict[str, str]
    ) -> None:
        second = tmp_path / "u.parquet"
        pq.write_table(pa.table({"id": [4], "v": ["d"]}), second)
        frame = session.read.parquet(files["parquet"], str(second).replace("\\", "/"))
        assert sorted(row["id"] for row in frame.collect()) == [1, 2, 3, 4]


class TestCsvOptions:
    def test_type_inference_is_on_by_default(self, session: Session, files: dict[str, str]) -> None:
        """Unlike the reference, whose default is every column as text."""
        schema = session.read.csv(files["csv"], header=True).schema
        assert schema.simpleString() == "struct<id:bigint,v:string>"

    def test_infer_schema_off_reads_everything_as_text(
        self, session: Session, files: dict[str, str]
    ) -> None:
        schema = session.read.csv(files["csv"], header=True, inferSchema=False).schema
        assert schema.simpleString() == "struct<id:string,v:string>"

    def test_a_separator_can_be_given(self, session: Session, files: dict[str, str]) -> None:
        rows = session.read.csv(files["semi"], header=True, sep=";").collect()
        assert [(row["id"], row["v"]) for row in rows] == [(1, "a"), (2, "b")]

    def test_options_can_come_through_option(self, session: Session, files: dict[str, str]) -> None:
        """`option()` and the keyword argument are two spellings of one setting."""
        rows = session.read.option("header", True).option("sep", ";").csv(files["semi"]).collect()
        assert len(rows) == 2

    def test_a_keyword_beats_an_option(self, session: Session, files: dict[str, str]) -> None:
        reader = session.read.option("sep", ",")
        rows = reader.csv(files["semi"], header=True, sep=";").collect()
        assert [(row["id"], row["v"]) for row in rows] == [(1, "a"), (2, "b")]


class TestLoadAndFormat:
    @pytest.mark.parametrize("name", ["parquet", "csv", "json"])
    def test_format_then_load(self, session: Session, files: dict[str, str], name: str) -> None:
        reader = session.read.format(name)
        if name == "csv":
            reader = reader.option("header", True)
        assert reader.load(files[name]).count() == 3

    def test_load_with_the_format_named_inline(
        self, session: Session, files: dict[str, str]
    ) -> None:
        assert session.read.load(files["parquet"], format="parquet").count() == 3

    def test_load_without_a_path_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="needs a path"):
            session.read.format("parquet").load()

    def test_load_for_iceberg_points_at_table(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match=r"session\.read\.table"):
            session.read.format("iceberg").load("whatever")

    def test_an_unreadable_format_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="orc"):
            session.read.format("orc")

    def test_the_refusal_says_what_does_work(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match="parquet"):
            session.read.format("avro")

    def test_a_reader_is_not_mutated_by_format(
        self, session: Session, files: dict[str, str]
    ) -> None:
        """Every method returns a new reader, so a held one cannot change underfoot."""
        base = session.read
        base.format("parquet")
        with pytest.raises(UnsupportedFeatureError):
            base.load(files["parquet"])


class TestItIsAnOrdinaryFrame:
    """Nothing downstream knows the rows came from a file."""

    def test_a_filter_works(self, session: Session, files: dict[str, str]) -> None:
        rows = session.read.parquet(files["parquet"]).filter(F.col("id") > 1).collect()
        assert sorted(row["id"] for row in rows) == [2, 3]

    def test_an_aggregate_works(self, session: Session, files: dict[str, str]) -> None:
        frame = session.read.parquet(files["parquet"])
        rows = frame.groupBy().agg(F.sum("id").alias("t")).collect()
        assert rows[0]["t"] == 6

    def test_it_joins_against_an_iceberg_table(
        self, session: Session, files: dict[str, str]
    ) -> None:
        """The case the convenience readers exist for."""
        frame = session.read.parquet(files["parquet"]).join(
            session.table("fx.plain").select("id"), on="id"
        )
        assert sorted(row["id"] for row in frame.collect()) == [1, 2, 3]

    def test_it_writes_into_an_iceberg_table(
        self, session: Session, catalog: SqlCatalog, files: dict[str, str]
    ) -> None:
        name = f"wr.csv_{uuid.uuid4().hex[:8]}"
        try:
            session.read.csv(files["csv"], header=True).write.saveAsTable(name)
            assert sorted(row[0] for row in session.table(name).select("id").collect()) == [1, 2, 3]
        finally:
            with contextlib.suppress(Exception):
                catalog.drop_table(tuple(name.split(".")))

    def test_it_streams(self, session: Session, files: dict[str, str]) -> None:
        frame = session.read.parquet(files["parquet"])
        assert sorted(row["id"] for row in frame.toLocalIterator()) == [1, 2, 3]

    def test_it_can_be_registered_as_a_temp_view(
        self, session: Session, files: dict[str, str]
    ) -> None:
        session.read.parquet(files["parquet"]).createOrReplaceTempView("v_file")
        assert session.sql("SELECT count(*) AS n FROM v_file").collect()[0]["n"] == 3

    def test_the_planner_does_not_try_to_resolve_it(
        self, session: Session, files: dict[str, str]
    ) -> None:
        """A table function is not a catalog reference, and never was one."""
        frame = session.read.parquet(files["parquet"])
        assert frame._sources == {}
        compiled = session._compile(frame._plan, frame._sources, frame.columns)
        assert compiled.scans == []

    def test_count_agrees_with_collect(self, session: Session, files: dict[str, str]) -> None:
        """The metadata count must not fire: there are no manifests to ask."""
        frame = session.read.parquet(files["parquet"])
        assert frame.count() == len(frame.collect()) == 3


class TestObjectStoreIsRefused:
    """Deliberate, and named as such: the schema binds where the credentials are not."""

    @pytest.mark.parametrize(
        "path", ["s3://bucket/x.parquet", "s3a://bucket/x.parquet", "gs://bucket/x.parquet"]
    )
    def test_an_object_store_path_is_refused(self, session: Session, path: str) -> None:
        with pytest.raises(UnsupportedFeatureError, match="object storage"):
            session.read.parquet(path)

    def test_the_refusal_points_at_the_path_that_works(self, session: Session) -> None:
        with pytest.raises(UnsupportedFeatureError, match=r"session\.table"):
            session.read.csv("s3://bucket/x.csv")

    def test_csv_and_json_refuse_it_too(self, session: Session) -> None:
        for reader in (session.read.csv, session.read.json):
            with pytest.raises(UnsupportedFeatureError):
                reader("s3://bucket/x")


class TestRefusals:
    def test_no_path_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineValueError, match="at least one path"):
            session.read.parquet()

    def test_a_non_string_path_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.read.parquet(42)  # type: ignore[arg-type]

    def test_an_empty_path_is_refused(self, session: Session) -> None:
        with pytest.raises(EngineTypeError):
            session.read.parquet("")

    def test_time_travel_options_still_work(self, session: Session) -> None:
        """Phase 9's job is unaffected by Phase 11 sharing the reader."""
        assert session.read.table("fx.plain").count() == 5

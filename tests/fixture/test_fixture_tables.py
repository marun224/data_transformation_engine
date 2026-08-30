"""The generated fixture tables, and what they prove about the paths ahead.

These run against the local sqlite catalog, entirely offline.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from icetl.exec import DuckDBEngine
from icetl.paths import engine_path, engine_paths
from tests.fixtures import FixtureTable


def _paths(table: object) -> list[str]:
    return engine_paths([task.file.file_path for task in table.scan().plan_files()])  # type: ignore[attr-defined]


class TestFixturesBuild:
    def test_all_fixtures_exist(self, fixtures: dict[str, FixtureTable]) -> None:
        assert set(fixtures) == {"plain", "partitioned", "wide", "nested", "renamed", "mor"}

    @pytest.mark.parametrize(
        ("name", "rows"),
        [
            ("plain", 5),
            ("partitioned", 12),
            ("wide", 500),
            ("nested", 2),
            ("renamed", 4),
            ("mor", 6),
        ],
    )
    def test_row_counts(self, fixtures: dict[str, FixtureTable], name: str, rows: int) -> None:
        assert fixtures[name].table.scan().to_arrow().num_rows == rows

    def test_wide_table_has_200_columns(self, fixtures: dict[str, FixtureTable]) -> None:
        assert len(fixtures["wide"].table.schema().fields) == 200

    def test_partitioned_table_has_one_file_per_partition(
        self, fixtures: dict[str, FixtureTable]
    ) -> None:
        """Pruning in Phase 2 needs several files to prune between."""
        tasks = list(fixtures["partitioned"].table.scan().plan_files())
        assert len(tasks) == 3

    def test_only_the_mor_fixture_has_delete_files(self, fixtures: dict[str, FixtureTable]) -> None:
        """Only the fixture built to trip the copy-on-write guard may have deletes.

        `read_parquet` cannot honour delete files, so any fixture that grew one by
        accident would read as silently wrong rather than being refused.
        """
        for name, fixture in fixtures.items():
            with_deletes = [t for t in fixture.table.scan().plan_files() if t.delete_files]
            if name == "mor":
                tasks = list(fixture.table.scan().plan_files())
                assert len(with_deletes) == 1, "the merge-on-read fixture lost its deletes"
                assert len(tasks) == 2, "a mixture of clean and deleted files"
            else:
                assert not with_deletes, f"{name} unexpectedly has delete files"


class TestDuckDBReadsWhatIcebergPlanned:
    def test_paths_translate_and_read(
        self, fixtures: dict[str, FixtureTable], engine: DuckDBEngine
    ) -> None:
        """The end-to-end Phase 0 claim: plan in PyIceberg, read in DuckDB."""
        paths = _paths(fixtures["plain"].table)
        result = engine.arrow(
            "SELECT id, vendor, amount FROM read_parquet($paths) ORDER BY id", {"paths": paths}
        )
        assert result.num_rows == 5
        assert result.column("vendor").to_pylist() == ["a", "b", "a", "c", None]

    def test_local_reads_need_no_httpfs(
        self, fixtures: dict[str, FixtureTable], engine: DuckDBEngine
    ) -> None:
        """Loading httpfs can hit the network; local paths must never trigger it."""
        paths = _paths(fixtures["plain"].table)
        engine.ensure_object_store(paths)
        loaded = engine.arrow(
            "SELECT count(*) AS n FROM duckdb_extensions() "
            "WHERE extension_name = 'httpfs' AND loaded"
        )
        assert loaded.column("n").to_pylist() == [0]

    def test_projection_reads_only_named_columns(
        self, fixtures: dict[str, FixtureTable], engine: DuckDBEngine
    ) -> None:
        """The Phase 2 invariant, checked early: never `SELECT *` on the wide table."""
        paths = _paths(fixtures["wide"].table)
        result = engine.arrow(
            "SELECT id, col_001 FROM read_parquet($paths) WHERE id < 10", {"paths": paths}
        )
        assert result.column_names == ["id", "col_001"]
        assert result.num_rows == 10


class TestSchemaEvolutionHazard:
    """Characterisation tests for PLAN.md 3.4 -- the renamed-column sharp edge.

    The `renamed` fixture holds one file written as `old_name` and one written as
    `new_name`, both carrying field-id 2. These tests pin the behaviour of the *raw*
    tools, which is what motivated the design: `read_parquet(union_by_name = true)`
    is silently wrong here and always will be, and PyIceberg is right.

    They are kept, unchanged, now that Phase 2 has fixed the pipeline -- because the
    hazard they document is a property of DuckDB, not of our code, and the day
    someone reaches for the "simpler" naive read again these say why not.
    `tests/fixture/test_pushdown.py` is where the fix itself is asserted.
    """

    def test_pyiceberg_reads_by_field_id_and_is_correct(
        self, fixtures: dict[str, FixtureTable]
    ) -> None:
        values = fixtures["renamed"].table.scan().to_arrow().column("new_name").to_pylist()
        assert sorted(values) == ["after-c", "after-d", "before-a", "before-b"]

    def test_naive_read_parquet_by_name_is_silently_wrong(
        self, fixtures: dict[str, FixtureTable], engine: DuckDBEngine
    ) -> None:
        """This is the bug PLAN.md 3.4 warns about, reproduced.

        `union_by_name` matches on name, so the two rows written before the rename
        come back as NULL instead of their values. No error, no warning -- which is
        exactly why detection has to be mandatory rather than opportunistic.

        This is deliberately the naive read, not ours: `icetl` groups the files by
        field-id before it gets here, which is what makes the same query correct.
        """
        paths = _paths(fixtures["renamed"].table)
        result = engine.arrow(
            "SELECT new_name FROM read_parquet($paths, union_by_name = true)", {"paths": paths}
        )
        values = result.column("new_name").to_pylist()

        assert None in values, "expected the pre-rename rows to come back NULL"
        assert sorted(v for v in values if v) == ["after-c", "after-d"]
        assert "before-a" not in values

    def test_parquet_files_carry_field_ids_for_detection(
        self, fixtures: dict[str, FixtureTable]
    ) -> None:
        """The detection mechanism 3.4 proposes needs field-ids in the footers."""
        for task in fixtures["renamed"].table.scan().plan_files():
            schema = pq.ParquetFile(engine_path(task.file.file_path)).schema_arrow
            field = schema.field(1)
            assert field.metadata is not None
            assert field.metadata[b"PARQUET:field_id"] == b"2"

    def test_schema_history_reveals_the_rename_without_touching_files(
        self, fixtures: dict[str, FixtureTable]
    ) -> None:
        """The cheaper detection: field-id 2 has had two names across the schema history.

        This is O(schemas), not O(files), so the fast path can be cleared without
        opening a single parquet footer.
        """
        names_by_field_id: dict[int, set[str]] = {}
        for schema in fixtures["renamed"].table.schemas().values():
            for field in schema.fields:
                names_by_field_id.setdefault(field.field_id, set()).add(field.name)

        assert names_by_field_id[2] == {"old_name", "new_name"}
        assert names_by_field_id[1] == {"id"}

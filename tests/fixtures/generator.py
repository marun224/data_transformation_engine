"""Build the fixture tables the test suite runs against.

A local PyIceberg `SqlCatalog` (sqlite metadata, local-filesystem warehouse) stands in
for the REST catalog. The tables here are chosen to reproduce the cases PLAN.md calls
out as risky, so those paths have coverage before the code that handles them exists:

    plain          the baseline
    partitioned    identity partitioning, for manifest-level pruning (3.2)
    wide           200 columns, for projection pushdown (3.6)
    nested         struct/list/map, for Phase 6
    renamed        a column renamed after data was written -- the sharp edge in 3.4,
                   where `read_parquet` matching by name silently returns NULLs
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.manifest import (
    POSITIONAL_DELETE_SCHEMA,
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestWriterV2,
    write_manifest_list,
)
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import Snapshot
from pyiceberg.table.metadata import TableMetadataV2
from pyiceberg.table.snapshots import Operation, Summary
from pyiceberg.table.update import (
    AddSnapshotUpdate,
    AssertRefSnapshotId,
    SetSnapshotRefUpdate,
)
from pyiceberg.transforms import IdentityTransform
from pyiceberg.typedef import Record
from pyiceberg.types import (
    DoubleType,
    ListType,
    LongType,
    MapType,
    NestedField,
    StringType,
    StructType,
)

# PyIceberg's snapshot models are pydantic classes whose fields carry hyphenated
# aliases, so mypy synthesises `__init__(**{"snapshot-id": ...})` and never sees the
# underscore spelling that `populate_by_name` accepts at runtime. Naming them through
# `Any` keeps the fixture below readable instead of `type: ignore`-per-argument.
_Snapshot: Any = Snapshot
_AddSnapshot: Any = AddSnapshotUpdate
_SetSnapshotRef: Any = SetSnapshotRefUpdate
_AssertRefSnapshotId: Any = AssertRefSnapshotId


class _DeleteManifestWriter(ManifestWriterV2):
    """A manifest writer that marks its content as DELETES.

    PyIceberg's `write_manifest` always produces a DATA manifest -- reasonably, since
    it cannot write delete files -- and the content flag is what tells a reader that
    the entries describe deletions. Overriding it is the whole difference.
    """

    def content(self) -> ManifestContent:
        return ManifestContent.DELETES


if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

__all__ = [
    "FIXTURE_BUILDERS",
    "FixtureTable",
    "build_all",
    "local_catalog",
    "warehouse_uri",
]

NAMESPACE = "fx"
WIDE_COLUMN_COUNT = 200


@dataclass(frozen=True)
class FixtureTable:
    """A built fixture, with the facts a test needs to assert against."""

    identifier: str
    table: Table
    rows: int
    note: str = ""


def warehouse_uri(root: Path) -> str:
    """Build a warehouse location PyIceberg can use on this platform.

    A bare `file://` prefix is correct on both platforms, because `as_posix()` already
    supplies the right number of slashes:

        Windows   C:/wh        ->  file://C:/wh     (two slashes; the drive is the netloc)
        POSIX     /tmp/wh      ->  file:///tmp/wh   (three slashes; empty netloc)

    The Windows form matters: PyIceberg rebuilds the path as `netloc + path`, so
    `file:///C:/wh` would collapse to `/C:/wh`, which the OS rejects. See `icetl.paths`
    for the matching translation on the DuckDB side, which needs the opposite spelling.
    """
    return f"file://{root.as_posix()}"


def local_catalog(root: Path, name: str = "test") -> SqlCatalog:
    """A sqlite-backed catalog over a local-filesystem warehouse."""
    root.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        name,
        **{
            "uri": f"sqlite:///{(root / 'catalog.db').as_posix()}",
            "warehouse": warehouse_uri(root),
        },
    )
    if (NAMESPACE,) not in catalog.list_namespaces():
        catalog.create_namespace(NAMESPACE)
    return catalog


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_plain(catalog: Catalog) -> FixtureTable:
    """Baseline: a few columns, one snapshot, no partitioning, no deletes."""
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "vendor", StringType(), required=False),
        NestedField(3, "amount", DoubleType(), required=False),
    )
    identifier = f"{NAMESPACE}.plain"
    table = catalog.create_table(identifier, schema=schema)
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
            "vendor": pa.array(["a", "b", "a", "c", None], pa.string()),
            "amount": pa.array([10.0, 20.5, 30.25, None, 50.0], pa.float64()),
        },
        schema=schema.as_arrow(),
    )
    table.append(data)
    return FixtureTable(identifier, table, rows=5, note="nulls in vendor and amount")


def build_partitioned(catalog: Catalog) -> FixtureTable:
    """Identity-partitioned, several partitions, several files."""
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "as_at_date", StringType(), required=False),
        NestedField(3, "amount", DoubleType(), required=False),
    )
    spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="as_at_date")
    )
    identifier = f"{NAMESPACE}.partitioned"
    table = catalog.create_table(identifier, schema=schema, partition_spec=spec)

    dates = ["2026-08-15", "2026-08-16", "2026-08-17"]
    rows = 0
    for index, date in enumerate(dates):
        # A separate append per date, so each partition also gets its own snapshot.
        count = 4
        table.append(
            pa.table(
                {
                    "id": pa.array(range(index * count, (index + 1) * count), pa.int64()),
                    "as_at_date": pa.array([date] * count, pa.string()),
                    "amount": pa.array([float(index * 10 + i) for i in range(count)], pa.float64()),
                },
                schema=schema.as_arrow(),
            )
        )
        rows += count
    return FixtureTable(identifier, table, rows=rows, note=f"{len(dates)} partitions")


def build_wide(catalog: Catalog) -> FixtureTable:
    """200 columns. Reading this with `SELECT *` is the anti-pattern (3.6)."""
    fields = [NestedField(1, "id", LongType(), required=False)]
    fields += [
        NestedField(i, f"col_{i - 1:03d}", DoubleType(), required=False)
        for i in range(2, WIDE_COLUMN_COUNT + 1)
    ]
    schema = Schema(*fields)
    identifier = f"{NAMESPACE}.wide"
    table = catalog.create_table(identifier, schema=schema)

    row_count = 500
    columns: dict[str, pa.Array] = {"id": pa.array(range(row_count), pa.int64())}
    for i in range(2, WIDE_COLUMN_COUNT + 1):
        columns[f"col_{i - 1:03d}"] = pa.array([float(i)] * row_count, pa.float64())
    table.append(pa.table(columns, schema=schema.as_arrow()))
    return FixtureTable(identifier, table, rows=row_count, note=f"{WIDE_COLUMN_COUNT} columns")


def build_nested(catalog: Catalog) -> FixtureTable:
    """Struct, list, and map columns, for the Phase 6 work."""
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(
            2,
            "person",
            StructType(
                NestedField(5, "name", StringType(), required=False),
                NestedField(6, "age", LongType(), required=False),
            ),
            required=False,
        ),
        NestedField(
            3,
            "tags",
            ListType(element_id=7, element=StringType(), element_required=False),
            required=False,
        ),
        NestedField(
            4,
            "scores",
            MapType(
                key_id=8,
                key_type=StringType(),
                value_id=9,
                value_type=LongType(),
                value_required=False,
            ),
            required=False,
        ),
    )
    identifier = f"{NAMESPACE}.nested"
    table = catalog.create_table(identifier, schema=schema)
    # Plain Python values with an explicit schema: inference would read the map column
    # as a list of tuples rather than a map.
    data = pa.table(
        {
            "id": [1, 2],
            "person": [{"name": "ada", "age": 36}, {"name": None, "age": None}],
            "tags": [["x", "y"], []],
            "scores": [[("a", 1)], [("b", 2), ("c", 3)]],
        },
        schema=schema.as_arrow(),
    )
    table.append(data)
    return FixtureTable(identifier, table, rows=2, note="struct, list, map")


def build_renamed(catalog: Catalog) -> FixtureTable:
    """A column renamed between two appends -- the silent-wrong-results case (3.4).

    After this runs the table holds two data files: the first written when field 2 was
    called `old_name`, the second when it was called `new_name`. Both carry field-id 2.
    Reading by *name* returns NULLs for the first file's rows; reading by *field-id*
    returns the data. Any fast path that cannot tell the difference is broken.
    """
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "old_name", StringType(), required=False),
    )
    identifier = f"{NAMESPACE}.renamed"
    table = catalog.create_table(identifier, schema=schema)
    table.append(
        pa.table(
            {"id": pa.array([1, 2], pa.int64()), "old_name": pa.array(["before-a", "before-b"])},
            schema=schema.as_arrow(),
        )
    )

    with table.update_schema() as update:
        update.rename_column("old_name", "new_name")

    table.append(
        pa.table(
            {"id": pa.array([3, 4], pa.int64()), "new_name": pa.array(["after-c", "after-d"])},
            schema=table.schema().as_arrow(),
        )
    )
    return FixtureTable(identifier, table, rows=4, note="2 rows written as old_name, 2 as new_name")


def build_mor(catalog: Catalog) -> FixtureTable:
    """A table with merge-on-read positional deletes -- the case decision 11 forbids.

    icetl assumes copy-on-write, so nothing here reads this table successfully. It
    exists to prove the *guard* fires: `read_parquet` cannot see a delete file, so
    without a check the deleted rows would come back and the query would report
    success, and a guard with no test is a guard that stops working quietly.

    PyIceberg cannot write delete files either -- `table.delete()` is copy-on-write,
    so it rewrites data files instead of recording deletions. Building one by hand is
    the only way to get this table, which is what this does:

        1. write a positional-delete parquet naming (file_path, pos) pairs
        2. write a manifest whose content is DELETES rather than DATA
        3. write a manifest list carrying the existing data manifests plus that one
        4. commit it as a new snapshot on `main`

    Every step uses PyIceberg's own writers, so the result is a table any Iceberg
    reader would agree about -- and `plan_files()` duly hands back a task carrying a
    delete file, which is the input the guard exists to catch.
    """
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "vendor", StringType(), required=False),
        NestedField(3, "amount", DoubleType(), required=False),
    )
    identifier = f"{NAMESPACE}.mor"
    table = catalog.create_table(identifier, schema=schema)
    table.append(
        pa.table(
            {
                "id": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
                "vendor": pa.array(["a", "b", "c", "d", "e", "f"]),
                "amount": pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], pa.float64()),
            },
            schema=schema.as_arrow(),
        )
    )
    table = catalog.load_table(identifier)

    snapshot = table.current_snapshot()
    assert snapshot is not None
    # Positional deletes only exist from format version 2, which is what PyIceberg
    # creates; the cast is what lets `last_sequence_number` be read without a guard
    # for the v1 case that cannot occur here.
    metadata = cast(TableMetadataV2, table.metadata)
    assert metadata.format_version >= 2
    data_path = next(iter(table.scan().plan_files())).file.file_path
    new_snapshot_id = snapshot.snapshot_id + 1

    # Rows at positions 1 and 3 of the data file -- ids 2 and 4.
    deletes = pa.table(
        {"file_path": pa.array([data_path, data_path]), "pos": pa.array([1, 3], pa.int64())},
        schema=POSITIONAL_DELETE_SCHEMA.as_arrow(),
    )
    delete_path = f"{table.location()}/data/positional-deletes-{uuid.uuid4()}.parquet"
    output = table.io.new_output(delete_path)
    with output.create(overwrite=True) as handle:
        pq.write_table(deletes, handle)
    delete_size = len(table.io.new_input(delete_path))

    manifest_path = f"{table.location()}/metadata/{uuid.uuid4()}-deletes.avro"
    with _DeleteManifestWriter(
        spec=table.spec(),
        schema=table.schema(),
        output_file=table.io.new_output(manifest_path),
        snapshot_id=new_snapshot_id,
        avro_compression="null",
    ) as writer:
        writer.add_entry(
            ManifestEntry.from_args(
                status=ManifestEntryStatus.ADDED,
                snapshot_id=new_snapshot_id,
                data_file=DataFile.from_args(
                    content=DataFileContent.POSITION_DELETES,
                    file_path=delete_path,
                    file_format=FileFormat.PARQUET,
                    partition=Record(),
                    record_count=deletes.num_rows,
                    file_size_in_bytes=delete_size,
                    spec_id=metadata.default_spec_id,
                ),
            )
        )
    delete_manifest = writer.to_manifest_file()

    manifest_list_path = f"{table.location()}/metadata/snap-{new_snapshot_id}-deletes.avro"
    with write_manifest_list(
        format_version=metadata.format_version,
        output_file=table.io.new_output(manifest_list_path),
        snapshot_id=new_snapshot_id,
        parent_snapshot_id=snapshot.snapshot_id,
        sequence_number=metadata.last_sequence_number + 1,
        avro_compression="null",
    ) as manifest_list:
        manifest_list.add_manifests([*snapshot.manifests(table.io), delete_manifest])

    catalog.commit_table(
        table,
        (_AssertRefSnapshotId(snapshot_id=snapshot.snapshot_id, ref="main"),),
        (
            _AddSnapshot(
                snapshot=_Snapshot(
                    snapshot_id=new_snapshot_id,
                    parent_snapshot_id=snapshot.snapshot_id,
                    sequence_number=metadata.last_sequence_number + 1,
                    timestamp_ms=snapshot.timestamp_ms + 1,
                    manifest_list=manifest_list_path,
                    summary=Summary(operation=Operation.OVERWRITE, **{"total-data-files": "1"}),
                    schema_id=table.schema().schema_id,
                )
            ),
            _SetSnapshotRef(ref_name="main", type="branch", snapshot_id=new_snapshot_id),
        ),
    )
    # A second, delete-free file, so the table is a realistic mixture rather than
    # wholly unreadable: Iceberg sees 6 live rows where the files hold 8, which is the
    # gap a missed delete file would silently expose.
    table = catalog.load_table(identifier)
    table.append(
        pa.table(
            {
                "id": pa.array([7, 8], pa.int64()),
                "vendor": pa.array(["g", "h"]),
                "amount": pa.array([7.0, 8.0], pa.float64()),
            },
            schema=schema.as_arrow(),
        )
    )
    return FixtureTable(
        identifier,
        catalog.load_table(identifier),
        rows=6,
        note="8 rows in 2 files; ids 2 and 4 removed by a delete file. Refused on read.",
    )


FIXTURE_BUILDERS: dict[str, Callable[[Catalog], FixtureTable]] = {
    "plain": build_plain,
    "partitioned": build_partitioned,
    "wide": build_wide,
    "nested": build_nested,
    "renamed": build_renamed,
    "mor": build_mor,
}


def build_all(catalog: Catalog) -> dict[str, FixtureTable]:
    """Build every fixture table into `catalog`."""
    return {name: builder(catalog) for name, builder in FIXTURE_BUILDERS.items()}

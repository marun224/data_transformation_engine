"""Turning an Iceberg table plus a pruning request into files DuckDB can read.

PyIceberg plans, DuckDB executes (P2). Everything here is metadata work: manifests,
partition summaries, column statistics, parquet footers. No row is read.

Two things make this more than "list the files":

  * **Pruning (3.2, 3.6).** The predicate and column list the optimizer worked out
    are handed to `table.scan()`, which skips manifests, partitions and whole data
    files without opening them, and fetches statistics for fewer columns.

  * **Renamed columns (3.4).** Iceberg tracks columns by field-id; parquet files
    carry whatever name they had when written. `union_by_name` matches on *name*, so
    a column renamed after data was written reads back as NULLs from the older
    files -- silently, which is what makes it dangerous. Detection is cheap (the
    schema history says which field-ids have ever changed name) and only when it
    fires do we open footers and group files by the names they actually hold.

**Copy-on-write only** (decision 11). Every writer of these tables rewrites data
files rather than recording deletions, so a scan is exactly its data files and the
merge-on-read hybrid split of 3.3 is not built. It is *deferred*, not dropped --
Phase 12 owns it, and the guard below is the branch point it will grow from.

That is an assumption about the *writers*, though, not something the format
enforces: an Iceberg table is shared, and nothing stops another engine adding
merge-on-read deletes to one we only read. So the assumption is asserted rather than
trusted. `read_parquet` cannot see a delete file, so it would return the deleted rows
and report success, and a silently wrong answer is worse than a refused one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow.parquet as pq
from pyiceberg.expressions import AlwaysTrue

from icetl.errors import UnsupportedFeatureError
from icetl.paths import engine_paths
from icetl.plan.annotations import ScanRequest
from icetl.plan.describe import describe_predicate

if TYPE_CHECKING:
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table import Table

    from icetl.plan.builder import ScanSource

__all__ = ["ColumnAlias", "FileGroup", "ScanPlan", "plan_scan"]

# The parquet footer key Iceberg writes its field-ids under.
_FIELD_ID_KEY = b"PARQUET:field_id"


@dataclass(frozen=True)
class ColumnAlias:
    """How one output column is obtained from the files in a group.

    `stored` is the name the column has *in these parquet files*, which is not
    necessarily the name Iceberg calls it today. `None` means the files predate the
    column entirely, so it is projected as a typed NULL -- the same answer Iceberg
    gives, and the reason column *addition* was never the dangerous case.
    """

    output: str
    stored: str | None
    duckdb_type: str


@dataclass(frozen=True)
class FileGroup:
    """Files that share a column naming, and can therefore be read as one call."""

    paths: tuple[str, ...]
    total_bytes: int
    # None on the fast path: the files already spell every selected column the way
    # Iceberg does, so no aliasing is needed.
    projection: tuple[ColumnAlias, ...] | None = None

    @property
    def needs_aliasing(self) -> bool:
        return self.projection is not None


@dataclass(frozen=True)
class ScanPlan:
    """Everything the compile step needs to read one table reference.

    The counters exist so `explain()` can report pruning as a ratio -- "3 of 4096
    files, 12 of 214 columns" -- because a number with nothing to compare it to
    cannot tell you whether pushdown worked.
    """

    source: ScanSource
    groups: tuple[FileGroup, ...] = ()
    columns: tuple[str, ...] = ()
    total_columns: int = 0
    files_scanned: int = 0
    files_total: int | None = None
    bytes_scanned: int = 0
    pushed_filter: str | None = None
    unpushed_filters: tuple[str, ...] = ()
    renamed_columns: tuple[str, ...] = ()

    @property
    def file_count(self) -> int:
        return self.files_scanned

    @property
    def is_empty(self) -> bool:
        """True when nothing at all will be read.

        An ordinary state, not an error: a table can be newly created, fully deleted,
        or pruned down to nothing by a predicate that matches no partition.
        """
        return not self.groups

    def describe(self) -> str:
        """The one-line summary `explain()` prints."""
        if self.is_empty:
            return "no data files"
        files = (
            f"{self.files_scanned} of {self.files_total} file(s)"
            if self.files_total is not None
            else f"{self.files_scanned} file(s)"
        )
        return f"{files}, {self.bytes_scanned / 1e6:.1f} MB"

    def describe_columns(self) -> str:
        return f"{len(self.columns)} of {self.total_columns}: {', '.join(self.columns)}"


def _selected_columns(schema: IcebergSchema, request: ScanRequest) -> tuple[str, ...]:
    """The columns to read, in schema order.

    A request for *no* columns is real -- `count(*)` references nothing -- but a
    projection has to name something, so the cheapest single column stands in. The
    row count is what the query wanted and the row count is what it gets.
    """
    names = [f.name for f in schema.fields]
    if request.columns is None:
        return tuple(names)
    wanted = {name.lower() for name in request.columns}
    selected = tuple(name for name in names if name.lower() in wanted)
    return selected or tuple(names[:1])


def _names_by_field_id(table: Table) -> dict[int, set[str]]:
    """Every name each field-id has ever had, across the table's whole schema history.

    O(schemas), not O(files) -- so the rename check costs nothing on the tables that
    have never renamed anything, which is nearly all of them.
    """
    history: dict[int, set[str]] = {}
    for schema in table.schemas().values():
        # Top-level fields only: `union_by_name` matches the column names in the
        # parquet root, so a nested field's rename cannot confuse it.
        for nested in schema.fields:
            history.setdefault(nested.field_id, set()).add(nested.name)
    return history


def _field_ids(table: Table) -> dict[str, int]:
    """`{column name: field id}` for the table's current top-level columns."""
    return {nested.name: nested.field_id for nested in table.schema().fields}


def _renamed_columns(table: Table, columns: tuple[str, ...]) -> tuple[str, ...]:
    """Which of `columns` have ever gone by another name."""
    ids = _field_ids(table)
    history = _names_by_field_id(table)
    return tuple(
        name
        for name in columns
        if (field_id := ids.get(name)) is not None and len(history.get(field_id, set())) > 1
    )


def _footer_names(table: Table, path: str) -> dict[int, str]:
    """`{field_id: column name}` as one parquet file actually spells them.

    Opened through the table's own `FileIO`, so this works against object storage
    with the catalog's credentials rather than needing its own.
    """
    with table.io.new_input(path).open() as handle:
        arrow_schema = pq.ParquetFile(handle).schema_arrow
    names: dict[int, str] = {}
    for arrow_field in arrow_schema:
        metadata = arrow_field.metadata or {}
        raw = metadata.get(_FIELD_ID_KEY)
        if raw is not None:
            names[int(raw)] = arrow_field.name
    return names


def _grouped_by_stored_names(
    table: Table, tasks: list, columns: tuple[str, ...], duckdb_types: dict[str, str]
) -> tuple[FileGroup, ...]:
    """Group files by the names they hold, and alias each group back to today's names.

    Only reached when a selected column is known to have been renamed. Each file's
    footer is read once; the result is `SELECT old AS new, ...` per group, which
    turns the silently-wrong read of 3.4 into a correct one without leaving DuckDB.
    """
    ids = _field_ids(table)
    field_ids = [ids[name] for name in columns]

    buckets: dict[tuple[str | None, ...], list[object]] = {}
    for task in tasks:
        stored = _footer_names(table, task.file.file_path)
        key = tuple(stored.get(field_id) for field_id in field_ids)
        buckets.setdefault(key, []).append(task)

    groups = []
    for key, bucket in buckets.items():
        projection = tuple(
            ColumnAlias(output=name, stored=key[index], duckdb_type=duckdb_types[name])
            for index, name in enumerate(columns)
        )
        groups.append(
            FileGroup(
                paths=tuple(engine_paths([t.file.file_path for t in bucket])),  # type: ignore[attr-defined]
                total_bytes=sum(t.file.file_size_in_bytes for t in bucket),  # type: ignore[attr-defined]
                projection=projection,
            )
        )
    return tuple(groups)


def _total_data_files(table: Table) -> int | None:
    """How many data files the current snapshot has, from its summary.

    Read from the summary rather than counted, so asking "how much did we prune?"
    never costs a second manifest scan. Returns None when the summary does not say.
    """
    snapshot = table.current_snapshot()
    if snapshot is None or snapshot.summary is None:
        return None
    raw = snapshot.summary.get("total-data-files")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _assert_copy_on_write(source: ScanSource, tasks: list) -> None:
    """Refuse a table that turns out to carry merge-on-read delete files.

    Decision 11 says these tables are copy-on-write, so this should never fire. It
    exists because the cost of being wrong is asymmetric: a delete file is invisible
    to `read_parquet`, so the rows Iceberg marks deleted would come back and the
    query would report success.

    Cheap, too. `delete_files` is already on the task, so this is a list scan over
    metadata that has been fetched either way.
    """
    dirty = [task for task in tasks if task.delete_files]
    if not dirty:
        return
    # `feature` is a noun phrase: the exception renders it as "<feature> is not
    # implemented yet." Everything explanatory belongs in the hint.
    raise UnsupportedFeatureError(
        f"Reading {source.resolved.qualified_name!r}, which has {len(dirty)} data "
        f"file(s) carrying merge-on-read delete files",
        phase="Phase 12",
        hint=(
            "icetl assumes copy-on-write (decision 11), and read_parquet cannot see a "
            "delete file -- it would return the rows Iceberg marks deleted and report "
            "success, so the scan is refused rather than answered wrongly. Either "
            "compact the table with an engine that materialises deletes, or pick up "
            "Phase 12, which restores the merge-on-read split of PLAN.md 3.3"
        ),
    )


def plan_scan(source: ScanSource, request: ScanRequest | None = None) -> ScanPlan:
    """Plan one table reference's scan, honouring whatever pruning `request` asks for.

    With no request the whole table is read, which is what a plan the optimizer could
    not bind falls back to -- less efficient, never less correct.
    """
    from icetl.plan.schema import iceberg_to_duckdb_type

    table = source.resolved.table
    request = request or ScanRequest(source=source)
    schema = table.schema()
    columns = _selected_columns(schema, request)
    duckdb_types = {f.name: iceberg_to_duckdb_type(f.field_type) for f in schema.fields}

    reads_every_column = len(columns) == len(schema.fields)
    scan = table.scan(
        row_filter=request.predicate,
        selected_fields=("*",) if reads_every_column else columns,
    )
    tasks = list(scan.plan_files())
    _assert_copy_on_write(source, tasks)

    renamed = _renamed_columns(table, columns)
    if not tasks:
        groups: tuple[FileGroup, ...] = ()
    elif renamed:
        groups = _grouped_by_stored_names(table, tasks, columns, duckdb_types)
    else:
        groups = (
            FileGroup(
                paths=tuple(engine_paths([task.file.file_path for task in tasks])),
                total_bytes=sum(task.file.file_size_in_bytes for task in tasks),
                projection=tuple(
                    ColumnAlias(output=name, stored=name, duckdb_type=duckdb_types[name])
                    for name in columns
                ),
            ),
        )

    return ScanPlan(
        source=source,
        groups=groups,
        columns=columns,
        total_columns=len(schema.fields),
        files_scanned=len(tasks),
        files_total=_total_data_files(table),
        bytes_scanned=sum(task.file.file_size_in_bytes for task in tasks),
        pushed_filter=(
            None
            if isinstance(request.predicate, AlwaysTrue)
            else describe_predicate(request.predicate)
        ),
        unpushed_filters=request.unpushed,
        renamed_columns=renamed,
    )

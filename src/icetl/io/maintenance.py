"""Compaction, snapshot expiry, and orphan-file cleanup -- on one node, no Spark.

    session.maintenance("nyc.trips").compact()
    session.maintenance("nyc.trips").expireSnapshots(retainLast=5)
    session.maintenance("nyc.trips").removeOrphanFiles()      # reports; deletes on request

**Compaction is not a tidiness feature here.** PLAN.md 3.6 leans on column-statistics
file pruning for the securities table, because its queries never filter the second
partition level -- and stats only prune when files are large and sorted. Many small
unsorted files defeat the whole read design, so this is the job that keeps it working.

**What PyIceberg 0.11 provides is one of the four.** `expire_snapshots` exists as a
builder; `rewrite_data_files`, `rewrite_manifests` and orphan cleanup do not, so the
first is wrapped and the rest are built or refused. `rewrite_manifests` is refused --
see `rewriteManifests` for why building it badly is worse than not building it.

**Everything that deletes is opt-in.** `removeOrphanFiles` reports by default and
deletes only when asked, because a file that looks orphaned and is not is
unrecoverable data loss, and the ways to look orphaned include "written by a commit
that is still in flight".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from icetl.errors import EngineValueError, UnsupportedFeatureError
from icetl.paths import engine_path

if TYPE_CHECKING:
    from pyiceberg.table import Table

    from icetl.sql.session import Session

__all__ = [
    "CompactionResult",
    "ExpiryResult",
    "OrphanScan",
    "TableMaintenance",
]

#: Files at or above this are left alone by `compact()`. Iceberg's own default target,
#: and large enough that the per-file overhead of opening a footer disappears.
DEFAULT_TARGET_FILE_SIZE = 512 * 1024 * 1024

#: A partition with fewer small files than this is not worth a rewrite: the commit
#: costs a snapshot and the read saves almost nothing.
DEFAULT_MIN_INPUT_FILES = 2

#: How old a file must be before `removeOrphanFiles` will consider it orphaned. A
#: commit in flight has already written its data files and not yet referenced them,
#: so anything recent is assumed to belong to one.
DEFAULT_ORPHAN_AGE = timedelta(days=3)


@dataclass(frozen=True)
class CompactionResult:
    """What one `compact()` did, in the units someone would check."""

    partitions_rewritten: int = 0
    files_before: int = 0
    files_after: int = 0
    bytes_rewritten: int = 0
    rows_rewritten: int = 0
    #: Partitions considered and skipped, with the reason -- so a compaction that did
    #: nothing says why rather than looking like a failure.
    skipped: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.partitions_rewritten > 0

    def __str__(self) -> str:
        if not self.changed:
            return f"compact: nothing to do ({len(self.skipped)} partition(s) already fine)"
        return (
            f"compact: {self.files_before} -> {self.files_after} file(s) across "
            f"{self.partitions_rewritten} partition(s), {self.rows_rewritten} row(s), "
            f"{self.bytes_rewritten / 1e6:.1f} MB rewritten"
        )


@dataclass(frozen=True)
class ExpiryResult:
    """Which snapshots were expired, and which were kept."""

    expired: tuple[int, ...] = ()
    retained: tuple[int, ...] = ()

    def __str__(self) -> str:
        return f"expireSnapshots: {len(self.expired)} expired, {len(self.retained)} retained"


@dataclass(frozen=True)
class OrphanScan:
    """Files under the table's location that no snapshot references."""

    orphans: tuple[str, ...] = ()
    referenced: int = 0
    scanned: int = 0
    deleted: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bytes_orphaned(self) -> int:
        from pathlib import Path

        total = 0
        for path in self.orphans:
            candidate = Path(engine_path(path))
            if candidate.is_file():
                total += candidate.stat().st_size
        return total

    def __str__(self) -> str:
        verb = f"{len(self.deleted)} deleted" if self.deleted else "none deleted"
        return (
            f"removeOrphanFiles: {len(self.orphans)} orphan(s) of {self.scanned} file(s) "
            f"scanned, {self.referenced} referenced, {verb}"
        )


class TableMaintenance:
    """What `session.maintenance("ns.table")` returns."""

    def __init__(self, session: Session, reference: str) -> None:
        self._session = session
        self._reference = reference

    def __repr__(self) -> str:
        return f"TableMaintenance[{self._reference}]"

    @property
    def _table(self) -> Table:
        """The table, freshly resolved.

        A property rather than a field: every operation here commits, and a commit
        invalidates the metadata the previous one read. Holding one `Table` across two
        operations is how a maintenance script gets a stale-metadata failure on its
        second call.
        """
        return self._session._resolver.resolve(self._reference).table

    # -- compaction --------------------------------------------------------

    def compact(
        self,
        *,
        targetFileSizeBytes: int = DEFAULT_TARGET_FILE_SIZE,
        minInputFiles: int = DEFAULT_MIN_INPUT_FILES,
        sort: bool = True,
    ) -> CompactionResult:
        """Rewrite each partition's small files into fewer, larger, sorted ones.

        The unit of work is a **partition**, because that is the unit Iceberg lets a
        single-node writer replace atomically: `overwrite` with a filter naming the
        partition is one commit that deletes its files and writes the replacements. A
        partition whose small files number fewer than `minInputFiles` is skipped, and
        so is one with no small files at all -- a compaction that rewrote everything
        every time would cost a snapshot per run for no gain.

        `sort` applies the table's own sort order, and is the half that makes column
        statistics prune (PLAN.md 3.6). A table with no sort order is compacted
        unsorted, which still helps: fewer files, fewer footers.

        **Concurrency.** Each partition is one commit, retried on conflict. Rows
        written to a partition by another writer *between* the read and the commit
        would be replaced by what was read -- Iceberg's optimistic concurrency
        catches a conflicting snapshot, not a conflicting intent -- so compaction
        wants a quiet table, as it does in every engine.
        """
        if targetFileSizeBytes <= 0:
            raise EngineValueError(
                f"targetFileSizeBytes must be positive, got {targetFileSizeBytes}."
            )
        if minInputFiles < 1:
            raise EngineValueError(f"minInputFiles must be at least 1, got {minInputFiles}.")

        table = self._table
        groups = self._partition_groups(table, targetFileSizeBytes)
        if not groups:
            return CompactionResult(skipped=("no data files",))

        rewritten = 0
        before = after = 0
        rewritten_bytes = rewritten_rows = 0
        skipped: list[str] = []

        for label, tasks in sorted(groups.items()):
            small = [task for task in tasks if task.file.file_size_in_bytes < targetFileSizeBytes]
            if len(small) < minInputFiles:
                skipped.append(f"{label}: {len(small)} small file(s), below minInputFiles")
                continue

            result = self._compact_one(table, label, tasks, sort=sort)
            if result is None:
                skipped.append(f"{label}: no rows to rewrite")
                continue
            rows, size, count = result
            rewritten += 1
            before += count
            after += 1
            rewritten_bytes += size
            rewritten_rows += rows
            table = self._table  # the commit moved the metadata on

        return CompactionResult(
            partitions_rewritten=rewritten,
            files_before=before,
            files_after=after,
            bytes_rewritten=rewritten_bytes,
            rows_rewritten=rewritten_rows,
            skipped=tuple(skipped),
        )

    def _partition_groups(self, table: Table, target: int) -> dict[str, list[Any]]:
        """The table's data files, grouped by the partition they belong to.

        An unpartitioned table is one group under `""`, so the loop above has no
        special case for it.
        """
        groups: dict[str, list[Any]] = {}
        for task in table.scan().plan_files():
            key = _partition_label(table, task.file)
            groups.setdefault(key, []).append(task)
        return groups

    def _compact_one(
        self, table: Table, label: str, tasks: list[Any], *, sort: bool
    ) -> tuple[int, int, int] | None:
        """Rewrite one partition. Returns `(rows, bytes, files replaced)`."""
        from pyiceberg.expressions import AlwaysTrue

        from icetl.sql.writer import commit_with_retry

        predicate = _partition_predicate(table, tasks[0].file)
        rows = table.scan(row_filter=predicate).to_arrow()
        if rows.num_rows == 0:
            return None
        if sort:
            rows = _sorted_by_table_order(table, rows)

        size = sum(task.file.file_size_in_bytes for task in tasks)
        count = len(tasks)

        def operation() -> None:
            fresh = self._session._resolver.resolve(self._reference).table
            if isinstance(predicate, AlwaysTrue):
                fresh.overwrite(rows)
            else:
                fresh.overwrite(rows, overwrite_filter=predicate)

        commit_with_retry(operation)
        self._session._invalidate_source(self._session._resolver.parse(self._reference))
        return rows.num_rows, size, count

    # -- snapshots ---------------------------------------------------------

    def expireSnapshots(
        self,
        *,
        olderThan: datetime | None = None,
        retainLast: int | None = None,
        snapshotIds: list[int] | None = None,
    ) -> ExpiryResult:
        """Drop old snapshots, keeping the current one and whatever is asked for.

        `retainLast` is icetl's, not PyIceberg's -- its builder expires by age or by
        id and has no notion of "keep the newest N", which is the form a retention
        policy is usually written in. It is applied here by working out the ids and
        handing those to `by_ids`, so the commit is still PyIceberg's.

        The current snapshot is never expired, whatever the arguments say: a table
        whose current snapshot has been deleted is not readable.
        """
        table = self._table
        snapshots = sorted(table.snapshots(), key=lambda s: s.timestamp_ms)
        if not snapshots:
            return ExpiryResult()

        current = table.current_snapshot()
        protected = {current.snapshot_id} if current is not None else set()

        if snapshotIds is not None:
            doomed = {int(identifier) for identifier in snapshotIds}
        else:
            doomed = set()
            if olderThan is not None:
                cutoff = _as_millis(olderThan)
                doomed |= {s.snapshot_id for s in snapshots if s.timestamp_ms < cutoff}
            if retainLast is not None:
                if retainLast < 1:
                    raise EngineValueError(f"retainLast must be at least 1, got {retainLast}.")
                doomed |= {s.snapshot_id for s in snapshots[:-retainLast]}
            if olderThan is None and retainLast is None:
                raise EngineValueError(
                    "expireSnapshots() needs one of olderThan, retainLast or snapshotIds -- "
                    "expiring everything is never what was meant."
                )

        doomed -= protected
        if not doomed:
            return ExpiryResult(retained=tuple(s.snapshot_id for s in snapshots))

        from icetl.sql.writer import commit_with_retry

        def operation() -> None:
            fresh = self._session._resolver.resolve(self._reference).table
            fresh.maintenance.expire_snapshots().by_ids(sorted(doomed)).commit()

        commit_with_retry(operation)
        self._session._invalidate_source(self._session._resolver.parse(self._reference))
        return ExpiryResult(
            expired=tuple(sorted(doomed)),
            retained=tuple(s.snapshot_id for s in snapshots if s.snapshot_id not in doomed),
        )

    # -- orphans -----------------------------------------------------------

    def removeOrphanFiles(
        self,
        *,
        olderThan: datetime | None = None,
        dryRun: bool = True,
    ) -> OrphanScan:
        """Find files under the table's location that no snapshot references.

        **Reports by default.** Deleting a file that is not really an orphan is
        unrecoverable, and the ways to look like one include being written by a commit
        that has not landed yet. `dryRun=False` is the caller saying they have read
        the list.

        `olderThan` defaults to three days ago for the same reason: a data file
        belonging to an in-flight commit exists before the metadata that references
        it, and a cleanup that raced one would delete live data.

        **Local filesystems only.** Listing an object store is a different operation
        with different costs, and doing it badly here would be worse than refusing.
        """
        from pathlib import Path

        table = self._table
        location = engine_path(table.location())
        root = Path(location)
        if not root.is_dir():
            raise UnsupportedFeatureError(
                f"Scanning {table.location()!r} for orphan files",
                hint=(
                    "removeOrphanFiles walks a local directory; an object-store "
                    "location needs a listing API this does not use yet"
                ),
            )

        horizon = olderThan or (datetime.now(UTC) - DEFAULT_ORPHAN_AGE)
        cutoff = _as_millis(horizon)
        referenced = _referenced_files(table)

        orphans: list[str] = []
        scanned = 0
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            scanned += 1
            resolved = candidate.as_posix()
            if resolved in referenced:
                continue
            if candidate.stat().st_mtime * 1000 >= cutoff:
                continue
            orphans.append(resolved)

        deleted: tuple[str, ...] = ()
        if not dryRun and orphans:
            for path in orphans:
                Path(path).unlink(missing_ok=True)
            deleted = tuple(orphans)

        return OrphanScan(
            orphans=tuple(sorted(orphans)),
            referenced=len(referenced),
            scanned=scanned,
            deleted=deleted,
        )

    # -- refused -----------------------------------------------------------

    def rewriteManifests(self) -> None:
        """Refused, and deliberately.

        PyIceberg 0.11 exposes no manifest rewriting, so building this means writing
        manifest lists by hand -- and a manifest that is subtly wrong does not fail,
        it makes files invisible to every reader of the table, which is data loss
        wearing the costume of a successful commit.

        `compact()` already rewrites the manifests that matter, because replacing a
        partition's data files writes new manifest entries for them. What is left is
        the case where manifests are fragmented but data files are fine, and that is
        worth its own phase rather than a guess at the end of this one.
        """
        raise UnsupportedFeatureError(
            "rewriteManifests()",
            hint=(
                "PyIceberg 0.11 exposes no manifest rewriting, and hand-written "
                "manifests fail silently by hiding data files. compact() rewrites the "
                "manifests for the partitions it touches"
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_millis(moment: datetime) -> int:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


def _partition_label(table: Table, data_file: Any) -> str:
    """A stable name for the partition a data file belongs to."""
    spec = table.spec()
    if not spec.fields:
        return ""
    record = data_file.partition
    parts = []
    for index, spec_field in enumerate(spec.fields):
        value = record[index] if record is not None else None
        parts.append(f"{spec_field.name}={value}")
    return "/".join(parts)


def _partition_predicate(table: Table, data_file: Any) -> Any:
    """A PyIceberg predicate selecting exactly this file's partition.

    Identity transforms only. Under any other transform the partition value is not
    the column's value -- a bucket number, a truncated string, a year -- so a
    predicate built from it would select the wrong rows, and `AlwaysTrue` (rewrite the
    whole table in one commit) is the answer that stays correct.
    """
    from pyiceberg.expressions import AlwaysTrue, And, EqualTo, IsNull
    from pyiceberg.transforms import IdentityTransform

    # Through `Any` for the reason `plan/pushdown.py` documents: PyIceberg's predicates
    # define a positional `__init__(term, literal)` on their base class, and mypy
    # synthesises a keyword-only one per pydantic subclass instead, so it never sees
    # the constructor Python actually calls.
    equal_to: Any = EqualTo
    is_null: Any = IsNull

    spec = table.spec()
    if not spec.fields:
        return AlwaysTrue()

    schema = table.schema()
    predicate: Any = AlwaysTrue()
    for index, spec_field in enumerate(spec.fields):
        if not isinstance(spec_field.transform, IdentityTransform):
            return AlwaysTrue()
        source = schema.find_field(spec_field.source_id)
        value = data_file.partition[index] if data_file.partition is not None else None
        term: Any = is_null(source.name) if value is None else equal_to(source.name, value)
        predicate = term if isinstance(predicate, AlwaysTrue) else And(predicate, term)
    return predicate


def _sorted_by_table_order(table: Table, rows: Any) -> Any:
    """Sort an Arrow table by the Iceberg table's sort order, if it has one."""
    order = table.sort_order()
    if order is None or not order.fields:
        return rows

    schema = table.schema()
    keys = []
    for sort_field in order.fields:
        name = schema.find_field(sort_field.source_id).name
        descending = str(getattr(sort_field, "direction", "")).lower().endswith("desc")
        keys.append((name, "descending" if descending else "ascending"))
    if not keys:
        return rows
    return rows.sort_by(keys)


def _referenced_files(table: Table) -> set[str]:
    """Every file any snapshot of the table refers to, as filesystem paths.

    Metadata as well as data: a metadata or manifest file is just as much a live file
    as a parquet one, and deleting one because it is not a data file would destroy the
    table rather than tidy it.
    """
    referenced: set[str] = set()

    def add(location: str | None) -> None:
        if location:
            referenced.add(engine_path(location).replace("\\", "/"))

    add(table.metadata_location)
    for entry in table.metadata.metadata_log:
        add(entry.metadata_file)

    for snapshot in table.snapshots():
        add(snapshot.manifest_list)
        try:
            manifests = snapshot.manifests(table.io)
        except Exception:  # pragma: no cover - a manifest list already gone
            continue
        for manifest in manifests:
            add(manifest.manifest_path)
            for manifest_entry in manifest.fetch_manifest_entry(table.io, discard_deleted=False):
                add(manifest_entry.data_file.file_path)

    return referenced

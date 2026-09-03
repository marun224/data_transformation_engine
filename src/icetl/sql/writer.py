"""`df.write` -- the write path.

The shape mirrors the reference's `DataFrameWriter`: a builder that collects a mode,
options and table settings, and does nothing until `saveAsTable` or `insertInto` is
called. Every method returns a **new** writer, so one held in a variable cannot be
changed by a later call on a derivative of it.

**Three things about this path are worth knowing before reading the code.**

*The result is materialised.* PyIceberg 0.11's `append` opens with
`if not isinstance(df, pa.Table): raise ValueError`, so there is no batch or reader form
to hand it -- the whole result becomes one Arrow table before any of it is written.
PLAN.md's Phase 7 asked for streaming; the API cannot take it today, and pretending
otherwise by chunking would trade the one-snapshot-per-write guarantee for it. Recorded
in `divergence.md` rather than worked around.

*Nullability comes from the data, not from the plan.* The analysed schema calls every
column nullable, because it comes from DuckDB, which has no non-nullable expression. So
a table **created** here has all-optional fields. Writing *into* an existing table is
safe regardless: PyIceberg's `_check_pyarrow_schema_compatible` validates the incoming
schema against the table's own, and rejects a NULL heading for a `required` field. The
looseness is confined to `saveAsTable` on a table that does not exist yet.

*A commit can lose a race.* Iceberg commits are optimistic, so a concurrent writer can
invalidate ours between reading the metadata and writing it. `_commit` retries a
`CommitFailedException` after refreshing, which is the whole of the "retry on commit
conflict" requirement -- the operation is rebuilt against the new metadata each time,
not replayed blindly.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    EngineValueError,
    QueryExecutionException,
    TableAlreadyExistsException,
    UnsupportedFeatureError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import pyarrow as pa
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

    from icetl.catalog.resolver import TableRef
    from icetl.sql.dataframe import DataFrame

__all__ = ["DataFrameWriter", "commit_with_retry", "iceberg_ready"]

#: The reference's save modes, and the spellings it accepts for each.
_MODES = {
    "append": "append",
    "overwrite": "overwrite",
    "error": "error",
    "errorifexists": "error",
    "default": "error",
    "ignore": "ignore",
}

#: How many times a lost commit race is retried before giving up.
_COMMIT_ATTEMPTS = 4
_COMMIT_BACKOFF_SECONDS = 0.1


class DataFrameWriter:
    """What `df.write` returns. Immutable; every method returns a new writer."""

    def __init__(
        self,
        df: DataFrame,
        *,
        mode: str = "error",
        options: Mapping[str, str] | None = None,
        partition_by: Sequence[str] = (),
        sort_by: Sequence[str] = (),
        properties: Mapping[str, str] | None = None,
    ) -> None:
        self._df = df
        self._mode = mode
        self._options = dict(options or {})
        self._partition_by = list(partition_by)
        self._sort_by = list(sort_by)
        self._properties = dict(properties or {})

    def __repr__(self) -> str:
        return f"DataFrameWriter[mode={self._mode}]"

    def _derive(self, **changes: Any) -> DataFrameWriter:
        settings: dict[str, Any] = {
            "mode": self._mode,
            "options": self._options,
            "partition_by": self._partition_by,
            "sort_by": self._sort_by,
            "properties": self._properties,
        }
        settings.update(changes)
        return DataFrameWriter(self._df, **settings)

    # -- builder ------------------------------------------------------------

    def format(self, source: str) -> DataFrameWriter:
        """Only `"iceberg"`. Accepted so a script written against the reference runs."""
        if not isinstance(source, str):
            raise EngineTypeError(f"format() expects a string, got {type(source).__name__}.")
        if source.lower() not in ("iceberg", "org.apache.iceberg.spark.source.icebergsource"):
            raise UnsupportedFeatureError(
                f"format({source!r})",
                hint="Iceberg is the only format this engine writes",
            )
        return self

    def mode(self, saveMode: str) -> DataFrameWriter:
        """`append`, `overwrite`, `error`/`errorifexists` (the default), or `ignore`."""
        if not isinstance(saveMode, str):
            raise EngineTypeError(f"mode() expects a string, got {type(saveMode).__name__}.")
        normalised = _MODES.get(saveMode.strip().lower())
        if normalised is None:
            raise EngineValueError(
                f"Unknown save mode {saveMode!r}. Known: {', '.join(sorted(set(_MODES)))}."
            )
        return self._derive(mode=normalised)

    def option(self, key: str, value: Any) -> DataFrameWriter:
        if not isinstance(key, str):
            raise EngineTypeError(f"option() expects a string key, got {type(key).__name__}.")
        return self._derive(options={**self._options, key: str(value)})

    def options(self, **options: Any) -> DataFrameWriter:
        merged = {**self._options, **{key: str(value) for key, value in options.items()}}
        return self._derive(options=merged)

    def partitionBy(self, *cols: Any) -> DataFrameWriter:
        """Partition a **newly created** table by these columns, with identity transforms.

        Ignored when the table already exists: an existing table's partitioning is its
        own, and silently re-partitioning someone else's table on write would be a far
        larger act than a write. `Table.update_spec()` is the deliberate way to change it.
        """
        return self._derive(partition_by=_names("partitionBy", cols))

    #: The reference's V2 writer spells it this way.
    partitionedBy = partitionBy

    def sortBy(self, *cols: Any) -> DataFrameWriter:
        """Record a sort order on a newly created table. Ignored when it already exists."""
        return self._derive(sort_by=_names("sortBy", cols))

    def tableProperty(self, key: str, value: Any) -> DataFrameWriter:
        """Set an Iceberg table property on a newly created table."""
        if not isinstance(key, str):
            raise EngineTypeError(
                f"tableProperty() expects a string key, got {type(key).__name__}."
            )
        return self._derive(properties={**self._properties, key: str(value)})

    # -- actions -------------------------------------------------------------

    def saveAsTable(self, name: str) -> None:
        """Write to `name`, creating the table if it does not exist.

        The mode decides what happens when it **does**: `append` adds, `overwrite`
        replaces the data, `ignore` does nothing, and `error` -- the default -- raises.
        A missing table is created in every mode, which is the reference's behaviour:
        the mode is about existing data, and there is none.
        """
        self._save(name, insert_into=False, overwrite=self._mode == "overwrite")

    def insertInto(self, tableName: str, overwrite: bool = False) -> None:
        """Write into an existing table, matching columns **by position**.

        By position, not by name -- that is the reference's rule for `insertInto`, and
        the reason it is a different method from `saveAsTable` rather than a flag on it.
        A frame whose columns are the right names in the wrong order inserts scrambled
        data without complaint, so the column count is checked and the order is not
        second-guessed.

        The table must already exist; `insertInto` never creates one.
        """
        if not isinstance(overwrite, bool):
            raise EngineTypeError(
                f"insertInto() expects overwrite as a bool, got {type(overwrite).__name__}."
            )
        self._save(tableName, insert_into=True, overwrite=overwrite or self._mode == "overwrite")

    def save(self, path: str | None = None) -> None:
        """Not available: this engine writes tables, not paths."""
        raise UnsupportedFeatureError(
            "df.write.save()",
            hint=(
                "There is no path-based write here -- data lives in Iceberg tables. "
                "Use saveAsTable('namespace.table')"
            ),
        )

    # -- the write itself ----------------------------------------------------

    def _save(self, name: str, *, insert_into: bool, overwrite: bool) -> None:
        if not isinstance(name, str):
            raise EngineTypeError(f"A table name must be a string, got {type(name).__name__}.")
        session = self._df._session
        ref = session._resolver.parse(name)
        if not ref.namespace:
            raise AnalysisException(
                f"Table reference {name!r} has no namespace and no current namespace is "
                f"set. Use a qualified name like 'nyc.{ref.name}'."
            )
        catalog = session._registry.get(ref.catalog)
        exists = catalog.table_exists(ref.identifier)

        if not exists:
            if insert_into:
                raise AnalysisException(
                    f"Table {name!r} does not exist. insertInto() writes into an existing "
                    f"table; use saveAsTable() to create one."
                )
            self._create_and_fill(catalog, ref, name)
            return

        if not insert_into:
            if self._mode == "error":
                raise TableAlreadyExistsException(
                    f"Table {name!r} already exists. Choose a mode: "
                    f".mode('append'), .mode('overwrite') or .mode('ignore')."
                )
            if self._mode == "ignore":
                return

        table = catalog.load_table(ref.identifier)
        if self._merges_schema():
            table = self._merge_schema(catalog, ref, table)
        data = self._arrow_for(table, name, positional=insert_into)
        if overwrite:
            self._overwrite(table, data)
        else:
            commit_with_retry(lambda: table.append(data))
        session._invalidate_source(ref)

    def _create_and_fill(self, catalog: Catalog, ref: TableRef, name: str) -> None:
        """Create the table from the frame's own schema, then write the rows into it.

        The schema is the **Arrow** schema of the executed result, handed to PyIceberg
        as-is: it knows how to turn one into an Iceberg schema, and doing it here would
        be a second type mapping to keep in step with the first.
        """
        data = iceberg_ready(self._df.toArrow())
        # Best effort: some catalogs manage namespaces themselves and refuse to be told
        # about them, which is not a reason to fail the write.
        with contextlib.suppress(Exception):
            catalog.create_namespace_if_not_exists(ref.namespace)
        try:
            table = catalog.create_table(
                ref.identifier, schema=data.schema, properties=self._properties
            )
        except Exception as exc:
            raise QueryExecutionException(f"Could not create table {name!r}: {exc}") from exc

        # Partitioning and sorting are applied by *name* after creation, so nothing here
        # has to know which field ids PyIceberg assigned to the schema it just built.
        if self._partition_by:
            self._apply_partitioning(table, name)
        if self._sort_by:
            self._apply_sort_order(table, name)
        if data.num_rows:
            table = catalog.load_table(ref.identifier)
            commit_with_retry(lambda: table.append(data))
        self._df._session._invalidate_source(ref)

    def _apply_partitioning(self, table: Table, name: str) -> None:
        available = {field.name for field in table.schema().fields}
        with table.update_spec() as update:
            for column in self._partition_by:
                if column not in available:
                    raise AnalysisException(
                        f"partitionBy({column!r}) names a column {name!r} does not have. "
                        f"Columns: {', '.join(sorted(available))}."
                    )
                update.add_identity(column)

    def _apply_sort_order(self, table: Table, name: str) -> None:
        available = {field.name for field in table.schema().fields}
        from pyiceberg.transforms import IdentityTransform

        with table.update_sort_order() as update:
            for column in self._sort_by:
                if column not in available:
                    raise AnalysisException(
                        f"sortBy({column!r}) names a column {name!r} does not have. "
                        f"Columns: {', '.join(sorted(available))}."
                    )
                update.asc(column, IdentityTransform())

    def _overwrite(self, table: Table, data: pa.Table) -> None:
        """Replace the whole table, or only the partitions the data touches."""
        if self._dynamic_partitions():
            if not table.spec().fields:
                raise AnalysisException(
                    "partitionOverwriteMode=dynamic needs a partitioned table; this one "
                    "has no partition spec, so there are no partitions to replace."
                )
            commit_with_retry(lambda: table.dynamic_partition_overwrite(data))
            return
        commit_with_retry(lambda: table.overwrite(data))

    def _merges_schema(self) -> bool:
        """True when `mergeSchema` asks for the table to be widened to fit the frame."""
        for key, value in self._options.items():
            if key.split(".")[-1].lower() != "mergeschema":
                continue
            setting = value.strip().lower()
            if setting not in ("true", "false"):
                raise EngineValueError(f"mergeSchema must be true or false, got {value!r}.")
            return setting == "true"
        return False

    def _merge_schema(self, catalog: Catalog, ref: TableRef, table: Table) -> Table:
        """Add the columns the frame has and the table does not, then reload it.

        Only ever **additive**: `union_by_name` adds what is missing and never drops or
        retypes what is there, which is what the reference's `mergeSchema` promises. A
        column the table has and the frame does not is untouched and lands NULL.

        A merge that changes nothing still costs a metadata read, which is why it is off
        unless asked for -- and why an ordinary write that does not fit still fails
        loudly rather than quietly reshaping the table.
        """
        incoming = iceberg_ready(self._df.toArrow()).schema
        existing = {field.name for field in table.schema().fields}
        missing = [field for field in incoming if field.name not in existing]
        if not missing:
            return table
        import pyarrow as arrow

        with table.update_schema() as update:
            update.union_by_name(arrow.schema(missing))
        self._df._session._invalidate_source(ref)
        return catalog.load_table(ref.identifier)

    def _dynamic_partitions(self) -> bool:
        """True when `partitionOverwriteMode` asks for a per-partition overwrite.

        The default is `static`, as it is in the reference -- and the default is the
        dangerous one, which is why it is the one you have to *not* ask for: a static
        overwrite replaces every row in the table, not only the ones the new data
        resembles.
        """
        for key, value in self._options.items():
            if key.split(".")[-1].lower() != "partitionoverwritemode":
                continue
            setting = value.strip().lower()
            if setting not in ("static", "dynamic"):
                raise EngineValueError(
                    f"partitionOverwriteMode must be 'static' or 'dynamic', got {value!r}."
                )
            return setting == "dynamic"
        return False

    def _arrow_for(self, table: Table, name: str, *, positional: bool) -> pa.Table:
        """The rows to write, checked against the table they are going into."""
        data = iceberg_ready(self._df.toArrow())
        expected = [field.name for field in table.schema().fields]
        if positional:
            if len(data.schema.names) != len(expected):
                raise AnalysisException(
                    f"insertInto({name!r}) matches columns by position, and this frame "
                    f"has {len(data.schema.names)} column(s) where the table has "
                    f"{len(expected)}."
                )
            # Renamed, not reordered: by position is by position, and quietly reordering
            # to match the names would make the method the one it is not.
            data = data.rename_columns(expected)
        return data


def iceberg_ready(data: pa.Table) -> pa.Table:
    """Retype the timestamp columns to something Iceberg will accept.

    DuckDB stamps a `TIMESTAMP WITH TIME ZONE` with the **session's own** zone --
    `timestamp[us, tz=Asia/Calcutta]` on the machine this was found on -- and Iceberg's
    `timestamptz` is UTC by definition, so PyIceberg refuses anything else outright:
    *Column 'ts' has an unsupported type*. Nanoseconds are refused for the same reason,
    Iceberg storing microseconds.

    Neither is a conversion of the data. A zone-aware Arrow timestamp is an instant, and
    so is Iceberg's, so this rewrites the label and leaves every value where it was. It
    runs on the way *into* Iceberg only -- what comes back out is untouched.

    Reaching this at all needs a timestamp column, which no `fx.*` fixture had until
    Phase 9 added one; `saveAsTable` had been unable to create a table from a frame
    carrying one since Phase 7.
    """
    import pyarrow as arrow

    fields = list(data.schema)
    changed = False
    for index, field in enumerate(fields):
        replacement = _iceberg_timestamp(field.type, arrow)
        if replacement is not None:
            fields[index] = field.with_type(replacement)
            changed = True
    if not changed:
        return data
    return data.cast(arrow.schema(fields, metadata=data.schema.metadata))


def _iceberg_timestamp(field_type: Any, arrow: Any) -> Any | None:
    """The type Iceberg wants for `field_type`, or None when it is already fine."""
    if not arrow.types.is_timestamp(field_type):
        return None
    unit = "us" if field_type.unit == "ns" else field_type.unit
    zone = "UTC" if field_type.tz is not None else None
    if unit == field_type.unit and zone == field_type.tz:
        return None
    return arrow.timestamp(unit, tz=zone)


def _names(method: str, cols: tuple[Any, ...]) -> list[str]:
    items = list(cols)
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = list(items[0])
    for item in items:
        if not isinstance(item, str):
            raise EngineTypeError(
                f"{method}() takes column names as strings, got {type(item).__name__}."
            )
    return items


def commit_with_retry(operation: Callable[[], None]) -> None:
    """Run an Iceberg commit, retrying if a concurrent writer wins the race.

    Iceberg commits are optimistic: the metadata read at the start may be stale by the
    time it is written. PyIceberg raises `CommitFailedException` for exactly that, and
    the operation is re-run against refreshed metadata rather than replayed -- a retry
    that reused the stale table object would fail the same way forever.
    """
    from pyiceberg.exceptions import CommitFailedException

    for attempt in range(_COMMIT_ATTEMPTS):
        try:
            operation()
            return
        except CommitFailedException:
            if attempt == _COMMIT_ATTEMPTS - 1:
                raise
            time.sleep(_COMMIT_BACKOFF_SECONDS * (2**attempt))
        except UnicodeEncodeError as exc:
            # PyIceberg renders a schema mismatch as a `rich` table of tick and cross
            # marks, printed to stdout *before* it raises. On a Windows console using
            # cp1252 that print raises `UnicodeEncodeError` first -- so the mismatch
            # never gets reported and the caller is told about a codec instead, having
            # watched half a table scroll past. FINDINGS.md 2.10.
            #
            # The real error cannot be recovered, because PyIceberg was still building
            # it. Saying what happened is what is available, and it is a great deal
            # better than a charmap complaint.
            raise AnalysisException(
                "The data does not match the table's schema. PyIceberg reports the "
                "difference as a table of Unicode marks, and printing it failed on "
                "this console's encoding, so its own message is unavailable -- "
                f"({exc}). Compare df.schema against the table's, or re-run with "
                "PYTHONIOENCODING=utf-8 to see PyIceberg's report."
            ) from exc
        except ValueError as exc:
            # PyIceberg validates the incoming schema against the table's and reports a
            # mismatch as a bare `ValueError` -- a required field given a NULL, a column
            # the table does not have, a type that will not fit. Every one of those is
            # an analysis failure in this engine's vocabulary, and a caller that catches
            # `AnalysisException` should not have to also catch `ValueError` to hear
            # about it. The message is PyIceberg's own, which is the useful part.
            raise AnalysisException(str(exc)) from exc

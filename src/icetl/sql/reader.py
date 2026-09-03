"""`session.read` -- a table as it was, or a file that is not in a table at all.

Two jobs, added in two phases. **Time travel** (Phase 9) is where the reference puts
it, and **file formats** (Phase 11) are the convenience readers for data that has not
been loaded into Iceberg yet -- a CSV to be cleaned up and appended, a parquet dump to
join against:

    session.read.parquet("data/trips.parquet")
    session.read.csv("data/trips.csv", header=True)
    session.read.json("data/events.json")
    session.read.format("parquet").load("data/trips.parquet")

A file read builds `SELECT * FROM read_parquet('...')` -- a table *function*, which
every part of the planner already skips when looking for tables to resolve, so it
needs no special case in the catalog, the scan planner or pushdown. It also gets no
pruning: there are no manifests, so there is nothing to prune with. The filter runs in
DuckDB, which for a file is the only place it could run anyway.

**Local paths only.** Schema analysis happens on the analyzer's own DuckDB connection,
which has neither httpfs nor the S3 credentials the engine's connection carries, so an
object-store path would bind against nothing and fail confusingly. It is refused
plainly instead, naming `session.table()` for data that lives in the warehouse.

What this exists for today is time travel, because that is where the reference puts it:

    session.read.option("snapshot-id", 8271497619288662701).table("nyc.trips")
    session.read.option("as-of-timestamp", "2026-08-16T00:00:00").table("nyc.trips")

Both build the same plan `SELECT ... FROM t VERSION AS OF ...` builds, and go down the
same path from there (P1) -- the option is turned into a version on the table node and
nothing downstream can tell which surface asked.

Every method returns a **new** reader, so one held in a variable cannot be changed by a
later call on a derivative of it.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, Any

from sqlglot import exp

from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    EngineValueError,
    UnsupportedFeatureError,
)
from icetl.paths import engine_path, is_object_store
from icetl.plan.builder import source_table

if TYPE_CHECKING:
    from collections.abc import Mapping

    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session

__all__ = ["DataFrameReader"]

#: The reference's Iceberg options for reading an earlier state of a table. Both are
#: spelled with hyphens, and both are also accepted under a `spark.`-style prefix so a
#: migrated script's fully-qualified key still lands.
_SNAPSHOT_ID = "snapshot-id"
_AS_OF = "as-of-timestamp"

#: Where `format()` stashes its choice, so `load()` can read it back. Prefixed so it
#: cannot collide with an option a caller passes through.
_FORMAT_KEY = "icetl.read.format"

#: File formats DuckDB reads directly, and therefore so do we.
_READABLE_FORMATS = ("parquet", "csv", "json")

#: Formats that would need a reader we do not have.
_UNSUPPORTED_FORMATS = ("orc", "avro", "text", "jdbc", "delta", "xml")


class DataFrameReader:
    """What `session.read` returns. Immutable; every method returns a new reader."""

    def __init__(self, session: Session, options: Mapping[str, str] | None = None) -> None:
        self._session = session
        self._options = dict(options or {})

    def __repr__(self) -> str:
        return f"DataFrameReader[{', '.join(sorted(self._options)) or 'no options'}]"

    def format(self, source: str) -> DataFrameReader:
        """The format `load()` will use. Iceberg, parquet, csv or json."""
        if not isinstance(source, str):
            raise EngineTypeError(f"format() expects a string, got {type(source).__name__}.")
        lowered = source.lower()
        if lowered in _READABLE_FORMATS:
            return self.option(_FORMAT_KEY, lowered)
        if lowered in _UNSUPPORTED_FORMATS:
            raise UnsupportedFeatureError(
                f"session.read.format({source!r})",
                hint=(
                    "icetl reads iceberg, parquet, csv and json. Load anything else "
                    "with its own library and pass the Arrow table to createDataFrame"
                ),
            )
        if lowered not in ("iceberg", "org.apache.iceberg.spark.source.icebergsource"):
            raise UnsupportedFeatureError(
                f"session.read.format({source!r})",
                hint="icetl reads iceberg, parquet, csv and json",
            )
        return self.option(_FORMAT_KEY, "iceberg")

    def option(self, key: str, value: Any) -> DataFrameReader:
        if not isinstance(key, str):
            raise EngineTypeError(f"option() expects a string key, got {type(key).__name__}.")
        return DataFrameReader(self._session, {**self._options, key: str(value)})

    def options(self, **options: Any) -> DataFrameReader:
        merged = {**self._options, **{key: str(value) for key, value in options.items()}}
        return DataFrameReader(self._session, merged)

    def schema(self, schema: Any) -> DataFrameReader:
        """Not available: an Iceberg table's schema is the table's, not the reader's."""
        raise UnsupportedFeatureError(
            "session.read.schema()",
            hint=(
                "An Iceberg table carries its own schema. To reshape the result, "
                "select and cast on the frame"
            ),
        )

    def table(self, tableName: str) -> DataFrame:
        """The table, as it is now or as it was at the snapshot the options name."""
        if not isinstance(tableName, str):
            raise EngineTypeError(f"table() expects a name, got {type(tableName).__name__}.")
        version = self._version()
        if version is None:
            return self._session.table(tableName)

        node = source_table(tableName)
        node.set("version", version)
        plan = exp.select(exp.Star()).from_(node)
        return self._session._frame_for(plan)

    def load(self, path: str | None = None, format: str | None = None, **options: Any) -> DataFrame:
        """Read `path` in whichever format `format()` or `format=` named."""
        reader = self.options(**options) if options else self
        # Read directly, not through `_lookup`: that matches an option by its last
        # dotted part, which is how a `spark.`-prefixed key still lands, and our own
        # key ends in "format" -- so a caller's `option("format", ...)` would win.
        named = (format or reader._options.get(_FORMAT_KEY) or "").lower()
        if path is None:
            raise EngineValueError("load() needs a path. For an Iceberg table, use table().")
        if named in ("", "iceberg"):
            raise UnsupportedFeatureError(
                "session.read.load() for an Iceberg table",
                hint="An Iceberg table is addressed by name: session.read.table('ns.table')",
            )
        if named == "parquet":
            return reader.parquet(path)
        if named == "csv":
            return reader.csv(path)
        if named == "json":
            return reader.json(path)
        raise UnsupportedFeatureError(
            f"session.read.load(format={named!r})",
            hint="icetl reads iceberg, parquet, csv and json",
        )

    def parquet(self, *paths: str) -> DataFrame:
        """Read one or more parquet files, without Iceberg being involved.

        `union_by_name` is on, as it is for an Iceberg scan, so a set of files whose
        columns differ reads as their union rather than failing.
        """
        return self._file_source(
            "read_parquet",
            paths,
            {"union_by_name": exp.true(), "hive_partitioning": exp.false()},
        )

    def csv(
        self,
        *paths: str,
        header: bool | None = None,
        sep: str | None = None,
        inferSchema: bool | None = None,
        nullValue: str | None = None,
    ) -> DataFrame:
        """Read one or more CSV files.

        The reference's option names, mapped onto DuckDB's `read_csv`. Type inference
        is DuckDB's sniffer and is on by default; `inferSchema=False` reads every
        column as text, which is the reference's default and rarely what anyone wants
        -- so unlike the reference, inference stays on unless it is turned off.
        """
        arguments: dict[str, exp.Expression] = {}
        header_value = self._flag("header", header)
        if header_value is not None:
            arguments["header"] = exp.true() if header_value else exp.false()
        separator = sep if sep is not None else self._lookup("sep") or self._lookup("delimiter")
        if separator is not None:
            arguments["delim"] = exp.Literal.string(separator)
        infer = self._flag("inferschema", inferSchema)
        if infer is False:
            arguments["all_varchar"] = exp.true()
        missing = nullValue if nullValue is not None else self._lookup("nullvalue")
        if missing is not None:
            arguments["nullstr"] = exp.Literal.string(missing)
        return self._file_source("read_csv", paths, arguments)

    def json(self, *paths: str) -> DataFrame:
        """Read one or more JSON files -- newline-delimited or an array of objects."""
        return self._file_source("read_json", paths, {})

    def _flag(self, key: str, given: bool | None) -> bool | None:
        """A boolean from the argument, else from `option()`, else unset."""
        if given is not None:
            return bool(given)
        raw = self._lookup(key)
        if raw is None:
            return None
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")

    def _file_source(
        self, function: str, paths: tuple[str, ...], arguments: dict[str, exp.Expression]
    ) -> DataFrame:
        """`SELECT * FROM <function>([paths], key => value, ...)` as a frame.

        The paths are inlined as literals rather than bound as a parameter, unlike an
        Iceberg scan's file list: there are a handful of them rather than thousands,
        and inlining is what lets the *analyzer* -- which binds on its own connection,
        with no parameters -- work out the schema by reading the file's own header.
        """
        if not paths:
            raise EngineValueError(f"{function}() needs at least one path.")
        located = [_readable_path(path) for path in paths]

        expressions: list[exp.Expression] = [
            exp.Array(expressions=[exp.Literal.string(path) for path in located])
        ]
        expressions += [
            exp.Kwarg(this=exp.var(key), expression=value) for key, value in arguments.items()
        ]
        call = exp.Anonymous(this=function, expressions=expressions)
        plan = exp.select(exp.Star()).from_(call)
        return self._session._frame_for(plan)

    def _version(self) -> exp.Version | None:
        """The `VERSION AS OF` the options ask for, or None for an ordinary read."""
        snapshot = self._lookup(_SNAPSHOT_ID)
        as_of = self._lookup(_AS_OF)
        if snapshot is not None and as_of is not None:
            raise EngineValueError(
                f"Give one of {_SNAPSHOT_ID!r} and {_AS_OF!r}, not both -- they name "
                f"different snapshots."
            )
        if snapshot is not None:
            try:
                identifier = int(snapshot)
            except ValueError:
                raise AnalysisException(
                    f"{_SNAPSHOT_ID!r} must be a snapshot id, got {snapshot!r}."
                ) from None
            return exp.Version(
                this="VERSION", expression=exp.Literal.number(identifier), kind="AS OF"
            )
        if as_of is not None:
            return exp.Version(
                this="TIMESTAMP", expression=exp.Literal.string(_as_iso(as_of)), kind="AS OF"
            )
        return None

    def _lookup(self, name: str) -> str | None:
        """Find an option by its last dotted part, so a prefixed key still lands."""
        for key, value in self._options.items():
            if key.split(".")[-1].lower() == name:
                return value
        return None


def _as_iso(value: str) -> str:
    """`as-of-timestamp` as an ISO-8601 string.

    The reference documents it as **epoch milliseconds**, and people write an ISO
    timestamp anyway. Both are accepted, because the two cannot be confused: a bare
    integer is milliseconds and anything else is a timestamp to parse.
    """
    from datetime import datetime

    text = value.strip()
    if text.lstrip("-").isdigit():
        return datetime.fromtimestamp(int(text) / 1000, tz=UTC).isoformat()
    return text


def _readable_path(path: str) -> str:
    """A path DuckDB can open, or a refusal saying why it cannot.

    Object storage is refused rather than half-supported: the schema is worked out on
    the analyzer's connection, which carries neither httpfs nor the S3 credentials the
    engine's connection is given, so a working execution would sit behind a failing
    analysis. Iceberg data in object storage is read by name through `session.table()`,
    which is the path that has those credentials.
    """
    if not isinstance(path, str) or not path:
        raise EngineTypeError(f"Expected a file path, got {path!r}.")
    if is_object_store(path):
        raise UnsupportedFeatureError(
            f"Reading {path!r} directly from object storage",
            hint=(
                "the file readers bind their schema on a connection without S3 "
                "credentials. Iceberg data in object storage is read by name with "
                "session.table('ns.table')"
            ),
        )
    return engine_path(path)

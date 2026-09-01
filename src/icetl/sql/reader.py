"""`session.read` -- reading a table, optionally as it was. Phase 9.

The reference's `DataFrameReader` is mostly about *formats*: parquet, csv, json, jdbc.
None of those is here yet -- Phase 11 owns them -- and `session.table()` already covers
the ordinary Iceberg read. What this exists for today is **time travel**, because that is
where the reference puts it:

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

_FILE_FORMATS = ("parquet", "csv", "json", "orc", "avro", "text", "jdbc")


class DataFrameReader:
    """What `session.read` returns. Immutable; every method returns a new reader."""

    def __init__(self, session: Session, options: Mapping[str, str] | None = None) -> None:
        self._session = session
        self._options = dict(options or {})

    def __repr__(self) -> str:
        return f"DataFrameReader[{', '.join(sorted(self._options)) or 'no options'}]"

    def format(self, source: str) -> DataFrameReader:
        """Only `"iceberg"`. Accepted so a script naming it explicitly runs."""
        if not isinstance(source, str):
            raise EngineTypeError(f"format() expects a string, got {type(source).__name__}.")
        if source.lower() in _FILE_FORMATS:
            raise UnsupportedFeatureError(f"session.read.format({source!r})", phase="Phase 11")
        if source.lower() not in ("iceberg", "org.apache.iceberg.spark.source.icebergsource"):
            raise UnsupportedFeatureError(
                f"session.read.format({source!r})",
                hint="Iceberg is the only format this engine reads",
            )
        return self

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
        raise UnsupportedFeatureError(
            "session.read.load()",
            phase="Phase 11",
            hint="There is no path-based read here. Use session.read.table('ns.table')",
        )

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

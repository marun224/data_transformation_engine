"""The DuckDB connection: lifecycle, S3/httpfs setup, spill configuration.

Two decisions worth knowing about:

**httpfs is loaded lazily.** Loading it can hit the network to install the extension,
so we only do that when a location actually needs it. The local-fixture test suite
therefore runs completely offline, which is what lets it run anywhere.

**Threads and memory are left to DuckDB.** P7 says "use every core, spill when memory
runs out", and DuckDB's own defaults already do the first part. We only ever *narrow*
them, when the user asked us to. Spill is the exception: DuckDB will not spill at all
without a temp directory, so that one is always configured.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from icetl.conf import IcetlSettings
from icetl.errors import QueryExecutionException
from icetl.paths import is_object_store

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pyarrow as pa

__all__ = ["DEFAULT_BATCH_ROWS", "DuckDBEngine"]

logger = logging.getLogger(__name__)

# One secret name, replaced rather than accumulated, so credentials cannot pile up
# in a long-lived session.
_S3_SECRET_NAME = "icetl_s3"

#: Rows per batch when streaming. Large enough that per-batch overhead disappears,
#: small enough that one batch of a wide row is megabytes rather than gigabytes --
#: 200 columns of 8 bytes at this size is about 200 MB, which is the number that
#: matters for the table Phase 10 exists for.
DEFAULT_BATCH_ROWS = 131_072


def _quote(value: str) -> str:
    """Single-quote a SQL literal.

    Needed because DuckDB's CREATE SECRET takes no bind parameters, so the credential
    has to be inlined. Nothing built by this function is ever logged.
    """
    return "'" + value.replace("'", "''") + "'"


class DuckDBEngine:
    """Owns one DuckDB connection for the lifetime of a session.

    DuckDB connections are not safe to use from several threads at once. `cursor()`
    hands out a thread-local child connection that shares the same database and
    configuration, which is the supported way to run concurrent queries.
    """

    def __init__(self, settings: IcetlSettings, *, database: str = ":memory:") -> None:
        self._settings = settings
        self._database = database
        self._connection: duckdb.DuckDBPyConnection | None = None
        self._httpfs_loaded = False
        self._lock = threading.Lock()
        self._local = threading.local()

    # -- lifecycle ---------------------------------------------------------

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """The root connection, opened and configured on first access."""
        if self._connection is None:
            with self._lock:
                if self._connection is None:
                    self._connection = self._open()
        return self._connection

    def _open(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(self._database)
        engine = self._settings.engine

        if engine.threads is not None:
            connection.execute(f"SET threads = {int(engine.threads)}")
        if engine.memory_limit:
            connection.execute(f"SET memory_limit = {_quote(engine.memory_limit)}")

        temp_directory = Path(engine.resolved_temp_directory())
        temp_directory.mkdir(parents=True, exist_ok=True)
        connection.execute(f"SET temp_directory = {_quote(str(temp_directory))}")

        return connection

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """A connection safe to use from the calling thread."""
        existing = getattr(self._local, "cursor", None)
        if existing is None:
            existing = self.connection.cursor()
            self._local.cursor = existing
            self._local.generation = 0
        return existing

    def _next_generation(self) -> int:
        """Record that this thread's cursor is about to run a new query.

        Per-thread, not per-engine: cursors are thread-local, so a query on one
        thread does not invalidate a stream being read on another, and counting them
        together would refuse a stream that was never in danger.
        """
        generation = getattr(self._local, "generation", 0) + 1
        self._local.generation = generation
        return generation

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            self._httpfs_loaded = False
            self._local = threading.local()

    def __enter__(self) -> DuckDBEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- object store ------------------------------------------------------

    def ensure_object_store(self, locations: list[str] | None = None) -> None:
        """Load httpfs and install the S3 secret, if any location needs it.

        Idempotent. Call before reading paths that may live in object storage.
        """
        if self._httpfs_loaded:
            return
        if locations is not None and not any(is_object_store(loc) for loc in locations):
            return

        connection = self.connection
        try:
            connection.execute("INSTALL httpfs")
            connection.execute("LOAD httpfs")
        except duckdb.Error as exc:
            raise QueryExecutionException(
                "Could not load DuckDB's httpfs extension, which is required to read "
                f"from object storage. If this machine is offline, pre-install it with "
                f"`duckdb -c 'INSTALL httpfs'`. Original error: {exc}"
            ) from exc

        self._apply_s3_secret(connection)
        self._httpfs_loaded = True

    def _apply_s3_secret(self, connection: duckdb.DuckDBPyConnection) -> None:
        s3 = self._settings.s3
        if not s3.configured:
            logger.debug("No S3 credentials configured; relying on DuckDB's own chain.")
            return

        clauses = ["TYPE s3"]
        if s3.access_key_id:
            clauses.append(f"KEY_ID {_quote(s3.access_key_id)}")
        if s3.secret_access_key:
            clauses.append(f"SECRET {_quote(s3.secret_access_key)}")
        if s3.session_token:
            clauses.append(f"SESSION_TOKEN {_quote(s3.session_token)}")
        if s3.region:
            clauses.append(f"REGION {_quote(s3.region)}")

        endpoint = s3.duckdb_endpoint()
        if endpoint is not None:
            host, use_ssl = endpoint
            clauses.append(f"ENDPOINT {_quote(host)}")
            clauses.append(f"USE_SSL {'true' if use_ssl else 'false'}")
        clauses.append(f"URL_STYLE {_quote('path' if s3.path_style_access else 'vhost')}")

        # Logged without the clause list -- it holds the secret key.
        logger.debug("Installing DuckDB S3 secret %s", _S3_SECRET_NAME)
        connection.execute(f"CREATE OR REPLACE SECRET {_S3_SECRET_NAME} ({', '.join(clauses)})")

    # -- queries -----------------------------------------------------------

    def register(self, name: str, value: Any) -> None:
        """Expose an in-memory object (an Arrow table, say) under a SQL name.

        Registered on the calling thread's cursor, which is where `execute` runs, so
        the name is visible to the very next query.
        """
        self.cursor().register(name, value)

    def execute(
        self, sql: str, parameters: dict[str, Any] | list[Any] | None = None
    ) -> duckdb.DuckDBPyConnection:
        """Run `sql`, translating DuckDB errors into the engine error hierarchy."""
        cursor = self.cursor()
        self._next_generation()
        try:
            return (
                cursor.execute(sql, parameters) if parameters is not None else cursor.execute(sql)
            )
        except duckdb.Error as exc:
            raise QueryExecutionException(f"{type(exc).__name__}: {exc}") from exc

    def arrow(self, sql: str, parameters: dict[str, Any] | list[Any] | None = None) -> pa.Table:
        """Run `sql` and return the whole result as an Arrow table.

        `to_arrow_table`, not `.arrow()`: as of duckdb 1.5 the latter returns a
        RecordBatchReader, and `fetch_arrow_table` is deprecated.
        """
        return self.execute(sql, parameters).to_arrow_table()

    def record_batches(
        self,
        sql: str,
        parameters: dict[str, Any] | list[Any] | None = None,
        *,
        batch_size: int = DEFAULT_BATCH_ROWS,
    ) -> Iterator[pa.RecordBatch]:
        """Run `sql` and yield the result in batches, so a large one never materialises.

        Runs on the calling thread's shared cursor, which it has to: a registered
        Arrow table and a `CREATE TEMP TABLE` are both **per-cursor** in DuckDB, so a
        dedicated connection would not see a cached frame, a `createDataFrame`, an
        inlined metadata table, or the empty-table stand-in a pruned-to-nothing scan
        substitutes. A stream that cannot read half the session's relations is not a
        stream worth having.

        Sharing the cursor costs this: DuckDB invalidates a result when the next
        query runs on the same cursor, and it does so **silently** -- the reader
        simply stops, and a partially consumed one reports zero further rows rather
        than raising. Iterating lazily is the entire point of a stream, so running
        another query mid-iteration is a thing callers will do.

        So it is guarded rather than documented. Every `execute` bumps a counter; each
        batch checks it, and a stream whose cursor has moved on refuses instead of
        ending early. A truncated result that reports success is the failure mode this
        codebase treats as worst (FINDINGS.md 6, rule 4).
        """
        cursor = self.cursor()
        generation = self._next_generation()
        try:
            result = (
                cursor.execute(sql, parameters) if parameters is not None else cursor.execute(sql)
            )
            reader = result.to_arrow_reader(batch_size)
        except duckdb.Error as exc:
            raise QueryExecutionException(f"{type(exc).__name__}: {exc}") from exc
        return self._guarded(reader, generation)

    def _guarded(self, reader: pa.RecordBatchReader, generation: int) -> Iterator[pa.RecordBatch]:
        """Yield from `reader`, refusing to continue if the cursor has been reused.

        The check happens **before** each batch is pulled, not after one is handed
        out. An invalidated reader does not error and does not yield -- it simply
        reports that it is finished -- so a check inside the loop body is never
        reached in the very case it exists for. Ask first, then pull.
        """
        while True:
            if getattr(self._local, "generation", -1) != generation:
                raise QueryExecutionException(
                    "This stream was invalidated by another query on the same session. "
                    "DuckDB ends a result when its cursor runs the next query, and it "
                    "does so without an error -- so the rows already yielded are a "
                    "prefix of the answer, not the answer. Either finish iterating "
                    "before running anything else, or collect the result first with "
                    "toArrow()."
                )
            try:
                batch = next(reader)
            except StopIteration:
                return
            yield batch

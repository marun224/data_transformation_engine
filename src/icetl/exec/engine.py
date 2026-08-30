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
    import pyarrow as pa

__all__ = ["DuckDBEngine"]

logger = logging.getLogger(__name__)

# One secret name, replaced rather than accumulated, so credentials cannot pile up
# in a long-lived session.
_S3_SECRET_NAME = "icetl_s3"


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
        return existing

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
        """Run `sql`, translating DuckDB errors into the PySpark hierarchy."""
        cursor = self.cursor()
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
        batch_size: int = 1_000_000,
    ) -> pa.RecordBatchReader:
        """Run `sql` and stream the result, so large results never materialise."""
        return self.execute(sql, parameters).to_arrow_reader(batch_size)

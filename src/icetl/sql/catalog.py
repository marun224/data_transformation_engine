"""`session.catalog` -- listing, describing and managing what the catalog holds.

Phase 9. The shape mirrors the reference's `Catalog`, down to the value types it hands
back, so a script that walks `listDatabases()` and `listColumns()` runs unaltered.

**Three things are worth knowing before reading the code.**

*A namespace is a database.* Iceberg namespaces nest -- `("nyc", "raw")` is one
namespace, not two -- and the reference has a flat `database` with a single name. They
are joined with dots here, so `nyc.raw` names the nested namespace and round-trips
through `setCurrentDatabase`. A namespace whose own name contains a dot cannot be
addressed, which is recorded in `divergence.md` rather than worked around.

*Column facts come from Iceberg, not from the analyser.* `listColumns` reads the table's
own schema, so `nullable` reports what Iceberg actually says -- unlike `df.schema`, where
every field is nullable because DuckDB has no non-nullable expression. That is the one
place in the codebase where the true nullability is visible, and it is deliberate: the
question `listColumns` answers is about the *table*, not about a query over it.

*Temporary views are listed alongside tables*, as the reference lists them, with
`isTemporary=True`. They are session-local and belong to no namespace, so they appear
whatever database is current.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from icetl.errors import (
    AnalysisException,
    EngineTypeError,
    NamespaceNotFoundError,
    TableNotFoundError,
    UnsupportedFeatureError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from icetl.sql.session import Session
    from icetl.types import StructType

__all__ = [
    "Catalog",
    "CatalogMetadata",
    "Column",
    "Database",
    "Function",
    "Table",
]


# -- what the listing methods hand back ------------------------------------------
#
# Field names are the reference's, `camelCase` included, because callers index them.


@dataclass(frozen=True)
class CatalogMetadata:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Database:
    name: str
    catalog: str | None
    description: str | None
    locationUri: str


@dataclass(frozen=True)
class Table:
    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    tableType: str
    isTemporary: bool

    @property
    def database(self) -> str | None:
        """The reference keeps this beside `namespace`, deprecated but still read."""
        return ".".join(self.namespace) if self.namespace else None


@dataclass(frozen=True)
class Column:
    name: str
    description: str | None
    dataType: str
    nullable: bool
    isPartition: bool
    isBucket: bool
    isCluster: bool = False


@dataclass(frozen=True)
class Function:
    name: str
    catalog: str | None
    namespace: list[str] | None
    description: str | None
    className: str
    isTemporary: bool


#: What `listTables` reports for a table the catalog owns. Iceberg tables in a REST or
#: SQL catalog are managed by that catalog -- it holds the metadata pointer and a `DROP`
#: removes it -- so `MANAGED` is the honest answer of the reference's four.
_TABLE_TYPE = "MANAGED"

_TEMPORARY = "TEMPORARY"


def _filter_pattern(names: Iterable[str], pattern: str | None) -> list[str]:
    """The reference's listing filter: `*` matches anything, `|` separates alternatives.

    Not SQL `LIKE` and not a regex -- `listTables("nyc*|raw*")` is the documented
    spelling, and anything else in the pattern is matched literally.
    """
    values = list(names)
    if pattern is None:
        return values
    if not isinstance(pattern, str):
        raise EngineTypeError(f"A listing pattern must be a string, got {type(pattern).__name__}.")
    alternatives = [
        re.compile("".join(".*" if part == "*" else re.escape(part) for part in _split_stars(alt)))
        for alt in pattern.split("|")
    ]
    return [value for value in values if any(alt.fullmatch(value) for alt in alternatives)]


def _split_stars(pattern: str) -> list[str]:
    """`ab*c` -> `['ab', '*', 'c']`, so the star can be replaced and the rest escaped."""
    return [piece for piece in re.split(r"(\*)", pattern) if piece]


class Catalog:
    """What `session.catalog` returns."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def __repr__(self) -> str:
        return f"<Catalog {self.currentCatalog()!r} database={self.currentDatabase()!r}>"

    # -- catalogs ----------------------------------------------------------

    def currentCatalog(self) -> str:
        return self._session._registry.default_name

    def setCurrentCatalog(self, catalogName: str) -> None:
        """Only the configured catalog. Accepted so a script naming it explicitly runs."""
        if not isinstance(catalogName, str):
            raise EngineTypeError(
                f"setCurrentCatalog() expects a name, got {type(catalogName).__name__}."
            )
        if not self._session._registry.is_known(catalogName):
            known = ", ".join(self._session._registry.names())
            raise AnalysisException(
                f"Catalog {catalogName!r} is not configured. Known catalogs: {known}."
            )

    def listCatalogs(self, pattern: str | None = None) -> list[CatalogMetadata]:
        names = _filter_pattern(self._session._registry.names(), pattern)
        return [CatalogMetadata(name=name, description=None) for name in names]

    # -- databases ---------------------------------------------------------

    def currentDatabase(self) -> str:
        return ".".join(self._session._resolver.current_namespace)

    def setCurrentDatabase(self, dbName: str) -> None:
        """Set the namespace an unqualified table reference resolves against."""
        if not isinstance(dbName, str):
            raise EngineTypeError(
                f"setCurrentDatabase() expects a name, got {type(dbName).__name__}."
            )
        self._session._resolver.set_current_namespace(dbName)

    def listDatabases(self, pattern: str | None = None) -> list[Database]:
        catalog = self._session._registry.get()
        names = sorted(".".join(namespace) for namespace in catalog.list_namespaces())
        return [self._database(name) for name in _filter_pattern(names, pattern)]

    def databaseExists(self, dbName: str) -> bool:
        catalog = self._session._registry.get()
        return bool(catalog.namespace_exists(_namespace(dbName)))

    def getDatabase(self, dbName: str) -> Database:
        if not self.databaseExists(dbName):
            raise NamespaceNotFoundError(f"Database {dbName!r} does not exist.")
        return self._database(dbName)

    def _database(self, name: str) -> Database:
        catalog = self._session._registry.get()
        try:
            properties = dict(catalog.load_namespace_properties(_namespace(name)))
        except Exception:  # pragma: no cover - a catalog that will not describe one
            properties = {}
        return Database(
            name=name,
            catalog=self.currentCatalog(),
            description=properties.get("comment"),
            locationUri=properties.get("location", ""),
        )

    # -- tables ------------------------------------------------------------

    def listTables(self, dbName: str | None = None, pattern: str | None = None) -> list[Table]:
        """Tables in `dbName` (default: the current database), plus the temporary views.

        Temporary views belong to no namespace, so they are listed whatever database is
        asked for -- which is what the reference does, and what makes
        `tableExists('v')` and `listTables()` agree.
        """
        namespace = _namespace(dbName) if dbName else self._session._resolver.current_namespace
        catalog = self._session._registry.get()
        if not catalog.namespace_exists(namespace):
            raise NamespaceNotFoundError(f"Database {'.'.join(namespace)!r} does not exist.")

        tables = [
            Table(
                name=identifier[-1],
                catalog=self.currentCatalog(),
                namespace=list(identifier[:-1]),
                description=None,
                tableType=_TABLE_TYPE,
                isTemporary=False,
            )
            for identifier in sorted(catalog.list_tables(namespace))
        ]
        tables += [
            Table(
                name=name,
                catalog=None,
                namespace=None,
                description=None,
                tableType=_TEMPORARY,
                isTemporary=True,
            )
            for name in sorted(self._session._temp_views)
        ]
        keep = set(_filter_pattern((table.name for table in tables), pattern))
        return [table for table in tables if table.name in keep]

    def tableExists(self, tableName: str, dbName: str | None = None) -> bool:
        if not isinstance(tableName, str):
            raise EngineTypeError(f"tableExists() expects a name, got {type(tableName).__name__}.")
        if dbName is None and tableName in self._session._temp_views:
            return True
        reference = f"{dbName}.{tableName}" if dbName else tableName
        try:
            ref = self._session._resolver.parse(reference)
        except Exception:
            return False
        if not ref.namespace:
            return False
        try:
            return bool(self._session._registry.get(ref.catalog).table_exists(ref.identifier))
        except Exception:
            return False

    def getTable(self, tableName: str) -> Table:
        if tableName in self._session._temp_views:
            return Table(
                name=tableName,
                catalog=None,
                namespace=None,
                description=None,
                tableType=_TEMPORARY,
                isTemporary=True,
            )
        ref = self._session._resolver.parse(tableName)
        if not self.tableExists(tableName):
            raise TableNotFoundError(f"Table or view {tableName!r} does not exist.")
        return Table(
            name=ref.name,
            catalog=ref.catalog or self.currentCatalog(),
            namespace=list(ref.namespace),
            description=None,
            tableType=_TABLE_TYPE,
            isTemporary=False,
        )

    def listColumns(self, tableName: str, dbName: str | None = None) -> list[Column]:
        """The table's columns, with Iceberg's own nullability and partition facts.

        A temporary view has neither, so its columns come from analysing the view's plan
        and report `nullable=True` throughout -- the same schema `df.schema` gives.
        """
        if dbName is None and tableName in self._session._temp_views:
            return [
                Column(
                    name=field_.name,
                    description=None,
                    dataType=field_.dataType.simpleString(),
                    nullable=field_.nullable,
                    isPartition=False,
                    isBucket=False,
                )
                for field_ in self._session.table(tableName).schema.fields
            ]

        reference = f"{dbName}.{tableName}" if dbName else tableName
        resolved = self._session._resolver.resolve(reference)
        schema = resolved.table.schema()
        partitioned = {field_.source_id for field_ in resolved.table.spec().fields}
        from icetl.plan.analysis import arrow_to_datatype

        return [
            Column(
                name=nested.name,
                description=nested.doc,
                dataType=arrow_to_datatype(arrow.type).simpleString(),
                nullable=not nested.required,
                isPartition=nested.field_id in partitioned,
                isBucket=False,
            )
            for nested, arrow in zip(schema.fields, schema.as_arrow(), strict=True)
        ]

    def createTable(
        self,
        tableName: str,
        path: str | None = None,
        source: str | None = None,
        schema: StructType | str | None = None,
        description: str | None = None,
        **options: Any,
    ) -> Any:
        """Create an empty Iceberg table and return a frame over it.

        Routed through the same `DataFrameWriter` `saveAsTable` uses, so a table created
        here and a table created by a write are created the same way (P1) -- including
        the all-optional fields recorded in `divergence.md`.

        `path` is refused: data lives in Iceberg tables here, not at a path.
        """
        if path is not None:
            raise UnsupportedFeatureError(
                "catalog.createTable(path=...)",
                hint="There is no path-based table here -- the catalog owns the location",
            )
        if source is not None and source.lower() != "iceberg":
            raise UnsupportedFeatureError(
                f"catalog.createTable(source={source!r})",
                hint="Iceberg is the only source this engine creates",
            )
        if schema is None:
            raise AnalysisException(
                "catalog.createTable() needs a schema: a StructType or a DDL string "
                "like 'id bigint, name string'."
            )
        writer = self._session.createDataFrame([], schema).write
        if description is not None:
            writer = writer.tableProperty("comment", description)
        for key, value in options.items():
            writer = writer.tableProperty(key, value)
        writer.saveAsTable(tableName)
        return self._session.table(tableName)

    def dropTable(self, tableName: str) -> bool:
        """Drop a table. False when there was nothing to drop.

        Not in the reference's `Catalog` -- it is `DROP TABLE` there -- but PLAN.md asks
        for it beside `createTable`, and a symmetric pair is worth the small addition.
        """
        from icetl.sql.ddl import drop_table

        return drop_table(self._session, tableName, if_exists=True, purge=False)

    def refreshTable(self, tableName: str) -> None:
        """Forget the snapshot cached for `tableName`, so the next read reloads it.

        The session pins a table to the snapshot it was first read at, which is what
        makes repeated reads cheap. Writes made *through this session* invalidate it
        themselves; this is how you pick up a write made by someone else.
        """
        ref = self._session._resolver.parse(tableName)
        self._session._invalidate_source(ref)

    def recoverPartitions(self, tableName: str) -> None:
        """Nothing to recover: Iceberg tracks files in metadata, not by listing paths.

        The reference's version re-scans the filesystem for Hive-style directories that
        the metastore has not been told about. Iceberg has no such gap -- a file is in
        the table when a manifest says so -- so this would have nothing to do. It is
        accepted rather than refused so a migrated script runs.
        """
        if not self.tableExists(tableName):
            raise TableNotFoundError(f"Table {tableName!r} does not exist.")

    def refreshByPath(self, path: str) -> None:
        raise UnsupportedFeatureError(
            "catalog.refreshByPath()",
            hint="Tables are addressed by name here. Use refreshTable('ns.table')",
        )

    # -- temporary views ---------------------------------------------------

    def dropTempView(self, viewName: str) -> bool:
        return self._session.dropTempView(viewName)

    def dropGlobalTempView(self, viewName: str) -> bool:
        raise UnsupportedFeatureError(
            "global temporary views",
            hint=(
                "There is one session here, so a cross-session view has no one to share "
                "with. Use createOrReplaceTempView()"
            ),
        )

    # -- caching -----------------------------------------------------------

    def cacheTable(self, tableName: str, storageLevel: Any = None) -> None:
        raise UnsupportedFeatureError(
            "catalog.cacheTable()",
            hint=(
                "Caching here is per-frame and eager: cached = session.table(name).cache(), "
                "then query `cached`. A name-level cache would have to shadow the table for "
                "both surfaces and go stale behind a write"
            ),
        )

    def uncacheTable(self, tableName: str) -> None:
        raise UnsupportedFeatureError(
            "catalog.uncacheTable()", hint="Use frame.unpersist() on the frame cache() returned"
        )

    def isCached(self, tableName: str) -> bool:
        raise UnsupportedFeatureError(
            "catalog.isCached()", hint="Caching here is per-frame; there is no name to ask about"
        )

    def clearCache(self) -> None:
        """Release every materialised frame this session is holding."""
        for name in list(self._session._cached):
            self._session._release(name)

    # -- functions ---------------------------------------------------------

    def listFunctions(
        self, dbName: str | None = None, pattern: str | None = None
    ) -> list[Function]:
        """The `F.*` surface.

        ⚠️ **This is the DataFrame surface's function list**, which is not yet the same as
        what `Session.sql()` resolves -- decision 16, deferred to Phase 15. A name listed
        here is one `F.*` provides; most but not all are reachable from SQL, and
        `weekday`/`dayofweek` answer differently through SQL. `divergence.md` has the list.
        """
        from icetl.sql import functions as F

        names = _filter_pattern(sorted(F.__all__), pattern)
        return [
            Function(
                name=name,
                catalog=None,
                namespace=None,
                description=(getattr(F, name).__doc__ or "").strip().split("\n")[0] or None,
                className=f"icetl.sql.functions.{name}",
                isTemporary=True,
            )
            for name in names
        ]

    def functionExists(self, functionName: str, dbName: str | None = None) -> bool:
        from icetl.sql import functions as F

        return functionName in set(F.__all__)

    def getFunction(self, functionName: str) -> Function:
        found = [fn for fn in self.listFunctions() if fn.name == functionName]
        if not found:
            raise AnalysisException(f"Function {functionName!r} does not exist.")
        return found[0]

    def registerFunction(self, name: str, f: Any, returnType: Any = None) -> None:
        raise UnsupportedFeatureError("catalog.registerFunction()", phase="Phase 11")


def _namespace(name: str) -> tuple[str, ...]:
    """`nyc.raw` -> `("nyc", "raw")`, the Iceberg namespace it addresses."""
    if not isinstance(name, str):
        raise EngineTypeError(f"A database name must be a string, got {type(name).__name__}.")
    return tuple(part for part in name.split(".") if part)

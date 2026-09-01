"""`Session` -- the entry point, and the only thing that knows how to run a plan.

The session owns the three long-lived pieces: the catalog registry, the DuckDB
engine, and the analyzer. A DataFrame owns none of them; it holds a plan and asks the
session to analyse or execute it. That keeps the compile pipeline in one place, which
matters because both user surfaces run through it:

    plan  ->  substitute sources  ->  generate DuckDB SQL  ->  execute

`Session.sql()` and the DataFrame API differ only in how the plan is *built*. From
the substitution step down there is one path (P1).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from icetl import __version__
from icetl.catalog import CatalogRegistry, TableResolver
from icetl.compat import SQL_DIALECT
from icetl.conf import IcetlConf, IcetlSettings, resolve_settings
from icetl.errors import (
    EngineTypeError,
    EngineValueError,
    ParseException,
    TempTableAlreadyExistsException,
    UnsupportedFeatureError,
)
from icetl.exec import DuckDBEngine
from icetl.exec.scan_planner import ScanPlan, plan_scan
from icetl.exec.source_sql import build_source
from icetl.plan.analysis import PlanAnalyzer
from icetl.plan.annotations import PlanAnnotations
from icetl.plan.builder import (
    ScanSource,
    collect_source_keys,
    source_key,
    source_table,
    substitute_sources,
)
from icetl.plan.optimizer import OptimizedPlan, optimize_plan
from icetl.plan.pushdown import extract_scan_requests
from icetl.plan.schema import SchemaBinder
from icetl.sql.conformance import apply_compat_semantics
from icetl.sql.dataframe import DataFrame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa
    from pyiceberg.catalog import Catalog

    from icetl.catalog.resolver import TableRef
    from icetl.types import StructType

__all__ = ["CompiledPlan", "Session"]

#: The release whose semantics the conformance rules are checked against.
#: Reported by `Session.reference_semantics`, not by `Session.version`.
REFERENCE_SEMANTICS_VERSION = "3.5.0"

# Statement types Phase 1 can run. Everything else has a phase that owns it.
_DDL_PHASES: dict[type[exp.Expression], tuple[str, str]] = {
    exp.Create: ("CREATE", "Phase 9"),
    exp.Drop: ("DROP", "Phase 9"),
    exp.Alter: ("ALTER", "Phase 9"),
}


@dataclass(frozen=True)
class CompiledPlan:
    """One plan, ready to run, with everything `explain()` needs to describe it."""

    sql: str
    parameters: dict[str, Any] = field(default_factory=dict)
    scans: list[ScanPlan] = field(default_factory=list)
    optimized: OptimizedPlan | None = None


class RuntimeConfig:
    """`Session.conf` -- get and set configuration on a live session."""

    def __init__(self, conf: IcetlConf) -> None:
        self._conf = conf

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._conf.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._conf.set(key, value)

    def unset(self, key: str) -> None:
        self._conf.remove(key)

    def isModifiable(self, key: str) -> bool:
        """Always True: nothing here is fixed at startup the way a cluster would be."""
        return True

    def getAll(self) -> dict[str, str]:
        return dict(self._conf.getAll())


class Session:
    """A session over an Iceberg catalog, executed by DuckDB."""

    _active: Session | None = None
    _active_lock = threading.Lock()

    def __init__(
        self,
        settings: IcetlSettings | None = None,
        conf: IcetlConf | None = None,
        *,
        catalog: Catalog | None = None,
    ) -> None:
        self._conf = conf or IcetlConf()
        self._settings = settings or resolve_settings(self._conf)
        self._registry = CatalogRegistry(self._settings)
        if catalog is not None:
            # How tests -- and anyone who built their own catalog -- inject one.
            self._registry.register(self._settings.catalog.name, catalog)
        self._resolver = TableResolver(self._registry)
        self._engine = DuckDBEngine(self._settings)
        self._analyzer = PlanAnalyzer()
        self._binder = SchemaBinder()
        self._sources: dict[str, ScanSource] = {}
        self._counter = 0
        self._cached: set[str] = set()
        self._temp_views: dict[str, exp.Expression] = {}
        self._lock = threading.Lock()
        self._stopped = False

    # -- builder -----------------------------------------------------------

    class Builder:
        """`Session.builder.appName(...).config(...).getOrCreate()`."""

        def __init__(self) -> None:
            self._conf = IcetlConf()
            self._catalog: Catalog | None = None

        def appName(self, name: str) -> Session.Builder:
            self._conf.set("icetl.appName", name)
            return self

        def config(
            self,
            key: str | None = None,
            value: Any = None,
            conf: IcetlConf | None = None,
            *,
            map: Mapping[str, Any] | None = None,
        ) -> Session.Builder:
            if conf is not None:
                self._conf.setAll(dict(conf.getAll()))
            if map is not None:
                self._conf.setAll(map)
            if key is not None:
                if value is None:
                    raise EngineTypeError(f"config({key!r}) needs a value.")
                self._conf.set(key, value)
            return self

        def withCatalog(self, catalog: Catalog) -> Session.Builder:
            """Use an already-built PyIceberg catalog."""
            self._catalog = catalog
            return self

        def getOrCreate(self) -> Session:
            """Return the active session, or build one.

            Matching the reference engine, configuration given here is applied to an existing
            session rather than replacing it -- but note that catalog settings are
            read when the session is built, so changing one afterwards has no effect
            until `stop()`.
            """
            with Session._active_lock:
                active = Session._active
                if active is not None and not active._stopped:
                    active._conf.setAll(dict(self._conf.getAll()))
                    return active
                session = Session(
                    settings=resolve_settings(self._conf),
                    conf=self._conf,
                    catalog=self._catalog,
                )
                Session._active = session
                return session

        def create(self) -> Session:
            """Build a new session even if one is already active."""
            session = Session(
                settings=resolve_settings(self._conf), conf=self._conf, catalog=self._catalog
            )
            with Session._active_lock:
                Session._active = session
            return session

    builder = Builder()

    @classmethod
    def getActiveSession(cls) -> Session | None:
        active = cls._active
        return None if active is None or active._stopped else active

    @classmethod
    def active(cls) -> Session:
        session = cls.getActiveSession()
        if session is None:
            raise UnsupportedFeatureError(
                "Using icetl without a Session",
                hint="Build one with Session.builder.getOrCreate()",
            )
        return session

    # -- properties --------------------------------------------------------

    @property
    def version(self) -> str:
        """icetl's own version."""
        return __version__

    @property
    def reference_semantics(self) -> str:
        """The release whose behaviour the conformance rules are checked against.

        Nothing here executes on that engine; it names the specification only.
        """
        return REFERENCE_SEMANTICS_VERSION

    @property
    def conf(self) -> RuntimeConfig:
        return RuntimeConfig(self._conf)

    @property
    def settings(self) -> IcetlSettings:
        """The resolved settings."""
        return self._settings

    @property
    def catalog(self) -> Any:
        raise UnsupportedFeatureError("Session.catalog", phase="Phase 9")

    @property
    def read(self) -> Any:
        raise UnsupportedFeatureError(
            "Session.read",
            phase="Phase 11",
            hint="Use Session.table() for Iceberg tables",
        )

    @property
    def udf(self) -> Any:
        raise UnsupportedFeatureError("Session.udf", phase="Phase 11")

    def createDataFrame(self, data: Any, schema: Any = None) -> DataFrame:
        """Build a frame from local data: rows, dicts, Rows, pandas or Arrow.

        `schema` may be omitted (types are inferred by Arrow), a list of column names, a
        DDL string (`"id bigint, name string"`), or a `StructType`. When it carries
        types, the columns are **cast** to them rather than reinterpreted, so the cast
        obeys the same conformance rules as any other -- a value that will not convert
        becomes NULL, or raises under `icetl.ansiMode`.

        The rows are materialised into a DuckDB temp table, exactly as `cache()` does,
        which is what makes the frame independent of this process's Python objects.
        `unpersist()` releases it.
        """
        import pyarrow as arrow

        fields = _schema_fields(schema)
        names = [field.name for field in fields] if fields else _schema_names(schema)
        table = _arrow_table(data, names, arrow)

        name = self._materialize(table)
        plan = exp.select(exp.Star()).from_(exp.to_identifier(name, quoted=True))
        frame = DataFrame(self, plan, {})
        frame._cache_name = name
        if fields is None:
            return frame

        if len(fields) != len(frame.columns):
            raise EngineValueError(
                f"createDataFrame() was given {len(fields)} field(s) but the data has "
                f"{len(frame.columns)} column(s)."
            )
        from icetl.sql.column import Column

        casts = [
            Column(exp.column(actual, quoted=True)).cast(field.dataType).alias(field.name)
            for actual, field in zip(frame.columns, fields, strict=True)
        ]
        return frame.select(*casts)

    def range(
        self,
        start: int,
        end: int | None = None,
        step: int = 1,
        numPartitions: int | None = None,
    ) -> DataFrame:
        """A frame of one `bigint` column `id`, counting from `start` up to but not
        including `end`.

        `range(5)` is `0..4`, as in the reference: with one argument, it is the *end*.
        `numPartitions` is accepted and ignored -- there is one partition here.

        Built on DuckDB's `range` table function, so nothing is materialised: a
        billion-row range costs nothing until something reads it.
        """
        for label, value in (("start", start), ("end", end), ("step", step)):
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise EngineTypeError(
                    f"range() expects {label} as an int, got {type(value).__name__}."
                )
        if end is None:
            start, end = 0, start
        if step == 0:
            raise EngineValueError("range() needs a non-zero step.")

        # A table *function*, so `collect_source_keys` skips it and nothing asks the
        # catalog to resolve a table called `range`.
        source = exp.Table(
            this=exp.Anonymous(
                this="range",
                expressions=[
                    exp.Literal.number(str(start)),
                    exp.Literal.number(str(end)),
                    exp.Literal.number(str(step)),
                ],
            ),
            alias=exp.TableAlias(this=exp.to_identifier("icetl_range", quoted=True)),
        )
        column = exp.column("range", table="icetl_range", quoted=True)
        plan = exp.select(exp.alias_(column, "id", quoted=True)).from_(source)
        return DataFrame(self, plan, {})

    # -- entry points ------------------------------------------------------

    def table(self, tableName: str) -> DataFrame:
        """`SELECT * FROM tableName` as a DataFrame.

        The reference is resolved against the catalog now, so a missing table fails
        here rather than at the first action.
        """
        if not isinstance(tableName, str):
            raise EngineTypeError(f"table() expects a name, got {type(tableName).__name__}.")
        view = self._temp_views.get(tableName)
        if view is not None:
            # A temporary view shadows a catalog table of the same name, as it does in
            # the reference -- registering one is how you override a table for a session.
            plan = view.copy()
            return DataFrame(
                self, plan, {k: self._source_for(k) for k in collect_source_keys(plan)}
            )
        source = self._source_for(tableName)
        plan = exp.select(exp.Star()).from_(source_table(tableName))
        return DataFrame(self, plan, {source.key: source})

    def sql(self, sqlQuery: str, **kwargs: Any) -> DataFrame:
        """Run a SQL query, producing the same kind of plan the DataFrame API builds."""
        if kwargs:
            raise UnsupportedFeatureError(
                f"Session.sql() named-argument substitution ({', '.join(sorted(kwargs))})",
                phase="Phase 4",
            )
        if not isinstance(sqlQuery, str):
            raise EngineTypeError(f"sql() expects a string, got {type(sqlQuery).__name__}.")
        try:
            statements = sqlglot.parse(sqlQuery, read=SQL_DIALECT)
        except Exception as exc:
            raise ParseException(f"Could not parse the query: {exc}") from exc

        parsed = [s for s in statements if s is not None]
        if len(parsed) != 1:
            raise ParseException(f"Session.sql() takes exactly one statement, got {len(parsed)}.")
        plan = parsed[0]

        if isinstance(plan, exp.Create) and str(plan.args.get("kind") or "").upper() == "VIEW":
            # Worth its own message: the rest of CREATE really is Phase 9, but a view is
            # available today through the DataFrame, and "scheduled for Phase 9" would
            # send someone away from a method that already exists.
            raise UnsupportedFeatureError(
                "CREATE VIEW statements",
                phase="Phase 9",
                hint=(
                    "For a session-local view, use "
                    "df.createOrReplaceTempView('name') and then query it by name"
                ),
            )
        if isinstance(plan, exp.Insert):
            return self._insert(plan)
        # Row-level operations (Phase 8). Each rewrites data files and returns nothing,
        # so they are statements in the same sense INSERT is.
        if isinstance(plan, (exp.Delete, exp.Update, exp.Merge)):
            return self._row_level(plan)
        for node_type, (keyword, phase) in _DDL_PHASES.items():
            if isinstance(plan, node_type):
                raise UnsupportedFeatureError(f"{keyword} statements", phase=phase)
        # `exp.SetOperation` rather than `exp.Union`: INTERSECT and EXCEPT are its
        # siblings, and naming only the one had them rejected as unimplemented when
        # sqlglot parsed them and DuckDB ran them perfectly well.
        if not isinstance(plan, (exp.Select, exp.SetOperation, exp.Subquery)):
            raise UnsupportedFeatureError(
                f"{type(plan).__name__.upper()} statements", phase="Phase 4"
            )

        plan = self._inline_temp_views(plan)
        sources = {key: self._source_for(key) for key in collect_source_keys(plan)}
        return DataFrame(self, plan, sources)

    def _insert(self, plan: exp.Insert) -> DataFrame:
        """Run `INSERT INTO t SELECT ...` / `INSERT OVERWRITE t SELECT ...`.

        Routed through the same `DataFrameWriter` the DataFrame surface uses, so the two
        agree on modes, on column matching and on what a commit conflict does (P1). SQL
        `INSERT` matches **by position**, which is what `insertInto` already does.

        Returns an empty frame, as the reference does -- a statement is not a query, but
        callers still expect something back.
        """
        target = plan.this
        if isinstance(target, exp.Schema):
            # `INSERT INTO t (a, b) SELECT ...` -- the column list is a rename, and
            # renaming half a table on the way in is a different feature from inserting.
            raise UnsupportedFeatureError(
                "INSERT with an explicit column list",
                hint=(
                    "SQL INSERT matches by position here. Project the SELECT into the "
                    "table's column order instead"
                ),
            )
        if not isinstance(target, exp.Table):
            raise ParseException(
                f"INSERT needs a table to write into, got {type(target).__name__}."
            )

        query = plan.expression
        if not isinstance(query, (exp.Select, exp.SetOperation, exp.Subquery)):
            raise UnsupportedFeatureError(
                f"INSERT from {type(query).__name__.upper()}",
                hint="INSERT INTO t SELECT ... is the supported form",
            )

        source = self._frame_for(query)
        name = source_key(target)
        overwrite = bool(plan.args.get("overwrite"))
        source.write.mode("overwrite" if overwrite else "append").insertInto(name)
        return self._empty_frame()

    def _row_level(self, plan: exp.Delete | exp.Update | exp.Merge) -> DataFrame:
        """Run `DELETE` / `UPDATE` / `MERGE`, then hand back the empty frame.

        Temporary views are inlined first, so a merge can read from one -- the *target*
        is still resolved against the catalog, because a view has nothing to write to.
        """
        from icetl.sql.rowlevel import run_delete, run_merge, run_update

        inlined = self._inline_temp_views(plan)
        if isinstance(inlined, exp.Delete):
            run_delete(self, inlined)
        elif isinstance(inlined, exp.Update):
            run_update(self, inlined)
        else:
            assert isinstance(inlined, exp.Merge)
            run_merge(self, inlined)
        return self._empty_frame()

    def _frame_for(self, query: exp.Expression) -> DataFrame:
        """A DataFrame over an already-parsed query, with its sources resolved."""
        plan = self._inline_temp_views(query)
        sources = {key: self._source_for(key) for key in collect_source_keys(plan)}
        return DataFrame(self, plan, sources)

    def _empty_frame(self) -> DataFrame:
        """The no-column, no-row frame a statement returns."""
        plan = exp.select(exp.alias_(exp.Null(), "_", quoted=True)).where(exp.false())
        return DataFrame(self, plan, {})

    # -- source resolution -------------------------------------------------

    def _source_for(self, reference: str) -> ScanSource:
        """Resolve a table reference, caching the result for the session's lifetime."""
        cached = self._sources.get(reference)
        if cached is not None:
            return cached
        resolved = self._resolver.resolve(reference)
        with self._lock:
            # Re-check inside the lock: two threads resolving the same reference must
            # end up with one view name, not two.
            existing = self._sources.get(reference)
            if existing is not None:
                return existing
            source = ScanSource(
                key=reference, resolved=resolved, view=f"icetl_src_{len(self._sources)}"
            )
            self._sources[reference] = source
        return source

    def _invalidate_source(self, ref: TableRef) -> None:
        """Forget any cached binding for the table `ref` names.

        A `ScanSource` holds a PyIceberg `Table` pinned to the snapshot it was loaded at,
        which is what makes repeated reads cheap -- and what makes them **stale** the
        moment something writes. So a write drops the entry, and the next `table()` call
        loads the table again and sees the new snapshot.

        Matched on the resolved identifier rather than the reference string, because
        `nyc.trips` and `trips` are the same table under two spellings and both have to
        go. A DataFrame built *before* the write keeps its own source and so keeps its
        snapshot, which is the read-your-plan behaviour rather than a leak.
        """
        with self._lock:
            stale = [
                key
                for key, source in self._sources.items()
                if source.resolved.ref.identifier == ref.identifier
            ]
            for key in stale:
                del self._sources[key]

    def _next_alias(self) -> str:
        """A subquery alias unique within this session, so nesting never collides."""
        with self._lock:
            self._counter += 1
            return f"_q{self._counter}"

    # -- materialised frames -----------------------------------------------

    def _materialize(self, table: pa.Table) -> str:
        """Store `table` as a DuckDB temp table and return the name it is known by.

        Registered in **both** engines, which is the part worth explaining. Execution
        and analysis run on deliberately separate DuckDB connections -- the analyzer
        never loads httpfs, never sees a credential and never opens a file -- so a temp
        table created for execution is invisible to schema resolution. The engine gets
        the real rows; the analyzer gets a zero-row view of the same Arrow schema, which
        is all it ever needs.
        """
        with self._lock:
            self._counter += 1
            name = f"icetl_cache_{self._counter}"
        staging = f"{name}_arrow"
        self._engine.register(staging, table)
        self._engine.execute(f'CREATE TEMP TABLE "{name}" AS SELECT * FROM "{staging}"')
        self._analyzer.register_view(name, table.schema.empty_table())
        with self._lock:
            self._cached.add(name)
        return name

    def _release(self, name: str) -> None:
        """Drop a materialised frame from both engines. Safe to call twice."""
        with self._lock:
            if name not in self._cached:
                return
            self._cached.discard(name)
        if not self._stopped:
            self._engine.execute(f'DROP TABLE IF EXISTS "{name}"')
            self._analyzer.unregister_view(name)

    # -- temporary views ---------------------------------------------------

    def _register_temp_view(self, name: str, plan: exp.Expression, *, replace: bool) -> None:
        if not isinstance(name, str) or not name:
            raise EngineTypeError("A temporary view needs a non-empty name.")
        if "." in name:
            raise EngineValueError(
                f"A temporary view name cannot be qualified, got {name!r}. Views are "
                f"session-local and belong to no namespace."
            )
        with self._lock:
            if not replace and name in self._temp_views:
                raise TempTableAlreadyExistsException(
                    f"Temporary view {name!r} already exists. Use "
                    f"createOrReplaceTempView() to replace it."
                )
            self._temp_views[name] = plan.copy()

    def dropTempView(self, viewName: str) -> bool:
        """Forget a temporary view. True when one was there, False when not.

        The reference puts this on `spark.catalog`, which is Phase 9 here; until then it
        lives on the session so a registered view can actually be removed again.
        """
        with self._lock:
            return self._temp_views.pop(viewName, None) is not None

    def _inline_temp_views(self, plan: exp.Expression) -> exp.Expression:
        """Replace references to temporary views with the plans they stand for.

        Done here, before source keys are collected, so everything downstream sees one
        ordinary plan: the optimizer, pushdown and the scan planner never learn that
        views exist. CTE names are skipped for the same reason `collect_source_keys`
        skips them -- a `WITH` binding is not a table reference.
        """
        if not self._temp_views:
            return plan
        bound = {cte.alias_or_name for cte in plan.find_all(exp.CTE) if cte.alias_or_name}

        def replace(node: exp.Expression) -> exp.Expression:
            if not isinstance(node, exp.Table) or isinstance(node.this, exp.Func):
                return node
            if node.catalog or node.db:
                return node
            name = node.name
            if name in bound or name not in self._temp_views:
                return node
            alias = exp.TableAlias(this=exp.to_identifier(node.alias or name, quoted=True))
            return exp.Subquery(this=self._temp_views[name].copy(), alias=alias)

        return plan.transform(replace, copy=True)

    # -- the compile pipeline ---------------------------------------------

    def _analyze(self, plan: exp.Expression, sources: Mapping[str, ScanSource]) -> StructType:
        """The plan's output schema, bound against zero-row views of its sources."""
        bound = substitute_sources(
            plan, sources, lambda source: exp.to_identifier(source.view, quoted=True)
        )
        return self._analyzer.analyze(bound.sql(dialect="duckdb"), sources)

    def _compile(
        self,
        plan: exp.Expression,
        sources: Mapping[str, ScanSource],
        output_names: Sequence[str],
    ) -> CompiledPlan:
        """Optimize the plan, prune the scans, and generate the DuckDB SQL.

        The order matters and is the whole of Phase 2: bind a schema, optimize
        against it, read the pruning facts off the optimized tree, plan each scan
        with those facts, and only then substitute the sources. Every step degrades
        to the Phase 1 behaviour rather than failing, so a plan the optimizer cannot
        bind still runs -- it just reads more than it needed to.
        """
        # The reference engine semantics first, so the optimizer -- and therefore pushdown --
        # reasons
        # about the tree that will actually execute (PLAN.md 3.5).
        conformed = apply_compat_semantics(plan, ansi_mode=self._settings.sql.ansi_mode)
        schema = self._binder.bind(sources)
        optimized = optimize_plan(conformed, schema, output_names)

        annotations = (
            extract_scan_requests(optimized.optimized, sources)
            if optimized.applied
            else PlanAnnotations(conformed)
        )
        requests = annotations.merged()
        scans = {key: plan_scan(source, requests.get(key)) for key, source in sources.items()}

        parameters: dict[str, Any] = {}
        bound = substitute_sources(
            optimized.optimized,
            sources,
            lambda source: build_source(
                scans[source.key], parameters=parameters, register=self._engine.register
            ),
        )
        return CompiledPlan(
            sql=bound.sql(dialect="duckdb"),
            parameters=parameters,
            scans=list(scans.values()),
            optimized=optimized,
        )

    def _execute(
        self,
        plan: exp.Expression,
        sources: Mapping[str, ScanSource],
        output_names: Sequence[str],
    ) -> pa.Table:
        """Compile and run a plan, returning Arrow."""
        if self._stopped:
            raise UnsupportedFeatureError("Using a Session after stop()")
        compiled = self._compile(plan, sources, output_names)
        paths = [path for scan in compiled.scans for group in scan.groups for path in group.paths]
        self._engine.ensure_object_store(paths)
        return self._engine.arrow(compiled.sql, compiled.parameters or None)

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        """Close the engine and the analyzer, and clear the active session."""
        self._engine.close()
        self._analyzer.close()
        self._registry.close()
        self._stopped = True
        with Session._active_lock:
            if Session._active is self:
                Session._active = None

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        name = self._conf.get("icetl.appName", "icetl")
        return f"<Session app={name!r} catalog={self._settings.catalog.name!r}>"


def _schema_fields(schema: Any) -> list[Any] | None:
    """The fields `schema` describes, or None when it carries no types.

    A list of plain names is *not* a schema in this sense -- it renames columns and
    leaves Arrow's inference to do the typing, which is what the reference does too.
    """
    from icetl.parse_types import parse_struct_ddl
    from icetl.types import StructType

    if schema is None:
        return None
    if isinstance(schema, StructType):
        return list(schema.fields)
    if isinstance(schema, str):
        return list(parse_struct_ddl(schema).fields)
    if isinstance(schema, (list, tuple)):
        return None
    raise EngineTypeError(
        f"createDataFrame() expects schema as a DDL string, a StructType, a list of "
        f"column names, or None -- got {type(schema).__name__}."
    )


def _schema_names(schema: Any) -> list[str] | None:
    """Column names taken from a list-of-names schema, or None."""
    if not isinstance(schema, (list, tuple)):
        return None
    names = list(schema)
    for name in names:
        if not isinstance(name, str):
            raise EngineTypeError(
                f"createDataFrame() expects column names as strings, got {type(name).__name__}."
            )
    return names


def _arrow_table(data: Any, names: list[str] | None, arrow: Any) -> Any:
    """Coerce local data into an Arrow table, naming the columns if `names` says so."""
    table = _as_arrow(data, names, arrow)
    if names is None:
        return table
    if len(names) != table.num_columns:
        raise EngineValueError(
            f"createDataFrame() was given {len(names)} column name(s) but the data has "
            f"{table.num_columns} column(s)."
        )
    return table.rename_columns(names)


def _as_arrow(data: Any, names: list[str] | None, arrow: Any) -> Any:
    if isinstance(data, arrow.Table):
        return data
    if hasattr(data, "to_dict") and hasattr(data, "columns") and not isinstance(data, dict):
        # A pandas DataFrame, without importing pandas to find out.
        return arrow.Table.from_pandas(data, preserve_index=False)
    if not isinstance(data, (list, tuple)):
        raise EngineTypeError(
            f"createDataFrame() expects a list of rows, a pandas DataFrame or an Arrow "
            f"table, got {type(data).__name__}."
        )

    rows = list(data)
    if not rows:
        if names is None:
            raise EngineValueError(
                "createDataFrame() cannot infer a schema from no rows. Pass a schema."
            )
        return arrow.table({name: arrow.array([], arrow.null()) for name in names})

    if all(isinstance(row, dict) for row in rows):
        return arrow.Table.from_pylist(rows)

    tuples = [_row_values(row) for row in rows]
    width = len(tuples[0])
    if any(len(row) != width for row in tuples):
        raise EngineValueError("createDataFrame() needs every row to be the same width.")
    # Built with the data's own column count, never with `names`: naming happens in
    # `_arrow_table`, so that a `names` list of the wrong length is *caught* there
    # rather than silently truncating the rows to fit it.
    columns = _row_names(rows[0], width)
    return arrow.table(
        {
            column: arrow.array([row[index] for row in tuples])
            for index, column in enumerate(columns)
        }
    )


def _row_values(row: Any) -> tuple[Any, ...]:
    if isinstance(row, (list, tuple)):
        return tuple(row)
    # A bare scalar is a one-column row, as it is in the reference.
    return (row,)


def _row_names(row: Any, width: int) -> list[str]:
    fields = getattr(row, "__fields__", None)
    if fields and len(fields) == width:
        return list(fields)
    return [f"_{index + 1}" for index in range(width)]

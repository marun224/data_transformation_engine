"""`SparkSession` -- the entry point, and the only thing that knows how to run a plan.

The session owns the three long-lived pieces: the catalog registry, the DuckDB
engine, and the analyzer. A DataFrame owns none of them; it holds a plan and asks the
session to analyse or execute it. That keeps the compile pipeline in one place, which
matters because both user surfaces run through it:

    plan  ->  substitute sources  ->  generate DuckDB SQL  ->  execute

`spark.sql()` and the DataFrame API differ only in how the plan is *built*. From the
substitution step down there is one path (P1).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from icetl import __version__
from icetl.catalog import CatalogRegistry, TableResolver
from icetl.conf import IcetlConf, IcetlSettings, resolve_settings
from icetl.errors import ParseException, PySparkTypeError, UnsupportedFeatureError
from icetl.exec import DuckDBEngine
from icetl.exec.scan_planner import ScanPlan, plan_scan
from icetl.exec.source_sql import build_source
from icetl.plan.analysis import PlanAnalyzer
from icetl.plan.annotations import PlanAnnotations
from icetl.plan.builder import ScanSource, collect_source_keys, source_table, substitute_sources
from icetl.plan.optimizer import OptimizedPlan, optimize_plan
from icetl.plan.pushdown import extract_scan_requests
from icetl.plan.schema import SchemaBinder
from icetl.sql.conformance import apply_spark_semantics
from icetl.sql.dataframe import DataFrame

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pyarrow as pa
    from pyiceberg.catalog import Catalog

    from icetl.types import StructType

__all__ = ["CompiledPlan", "SparkSession"]

# The PySpark version whose semantics we target. Scripts branch on it.
SPARK_VERSION = "3.5.0"

# Statement types Phase 1 can run. Everything else has a phase that owns it.
_DDL_PHASES: dict[type[exp.Expression], tuple[str, str]] = {
    exp.Create: ("CREATE", "Phase 9"),
    exp.Drop: ("DROP", "Phase 9"),
    exp.Alter: ("ALTER", "Phase 9"),
    exp.Insert: ("INSERT", "Phase 7"),
    exp.Update: ("UPDATE", "Phase 8"),
    exp.Delete: ("DELETE", "Phase 8"),
    exp.Merge: ("MERGE", "Phase 8"),
}


@dataclass(frozen=True)
class CompiledPlan:
    """One plan, ready to run, with everything `explain()` needs to describe it."""

    sql: str
    parameters: dict[str, Any] = field(default_factory=dict)
    scans: list[ScanPlan] = field(default_factory=list)
    optimized: OptimizedPlan | None = None


class RuntimeConfig:
    """`spark.conf` -- get and set configuration on a live session."""

    def __init__(self, conf: IcetlConf) -> None:
        self._conf = conf

    def get(self, key: str, default: str | None = None) -> str | None:
        return self._conf.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._conf.set(key, value)

    def unset(self, key: str) -> None:
        self._conf.remove(key)

    def isModifiable(self, key: str) -> bool:
        """Always True: nothing here is fixed at startup the way Spark's cluster is."""
        return True

    def getAll(self) -> dict[str, str]:
        return dict(self._conf.getAll())


class SparkSession:
    """A session over an Iceberg catalog, executed by DuckDB."""

    _active: SparkSession | None = None
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
        self._lock = threading.Lock()
        self._stopped = False

    # -- builder -----------------------------------------------------------

    class Builder:
        """`SparkSession.builder.appName(...).config(...).getOrCreate()`."""

        def __init__(self) -> None:
            self._conf = IcetlConf()
            self._catalog: Catalog | None = None

        def appName(self, name: str) -> SparkSession.Builder:
            self._conf.set("spark.app.name", name)
            return self

        def master(self, master: str) -> SparkSession.Builder:
            """Accepted and ignored -- there is no cluster to point at."""
            self._conf.set("spark.master", master)
            return self

        def config(
            self,
            key: str | None = None,
            value: Any = None,
            conf: IcetlConf | None = None,
            *,
            map: Mapping[str, Any] | None = None,
        ) -> SparkSession.Builder:
            if conf is not None:
                self._conf.setAll(dict(conf.getAll()))
            if map is not None:
                self._conf.setAll(map)
            if key is not None:
                if value is None:
                    raise PySparkTypeError(f"config({key!r}) needs a value.")
                self._conf.set(key, value)
            return self

        def enableHiveSupport(self) -> SparkSession.Builder:
            """Accepted and ignored -- catalogs are configured through `icetl.conf`."""
            return self

        def remote(self, url: str) -> SparkSession.Builder:
            raise UnsupportedFeatureError(
                f"Spark Connect (remote={url!r})",
                hint="icetl runs in this process, so there is nothing to connect to",
            )

        def withCatalog(self, catalog: Catalog) -> SparkSession.Builder:
            """Use an already-built PyIceberg catalog. Not part of PySpark's API."""
            self._catalog = catalog
            return self

        def getOrCreate(self) -> SparkSession:
            """Return the active session, or build one.

            Matching Spark, configuration given here is applied to an existing
            session rather than replacing it -- but note that catalog settings are
            read when the session is built, so changing one afterwards has no effect
            until `stop()`.
            """
            with SparkSession._active_lock:
                active = SparkSession._active
                if active is not None and not active._stopped:
                    active._conf.setAll(dict(self._conf.getAll()))
                    return active
                session = SparkSession(
                    settings=resolve_settings(self._conf),
                    conf=self._conf,
                    catalog=self._catalog,
                )
                SparkSession._active = session
                return session

        def create(self) -> SparkSession:
            """Build a new session even if one is already active."""
            session = SparkSession(
                settings=resolve_settings(self._conf), conf=self._conf, catalog=self._catalog
            )
            with SparkSession._active_lock:
                SparkSession._active = session
            return session

    builder = Builder()

    @classmethod
    def getActiveSession(cls) -> SparkSession | None:
        active = cls._active
        return None if active is None or active._stopped else active

    @classmethod
    def active(cls) -> SparkSession:
        session = cls.getActiveSession()
        if session is None:
            raise UnsupportedFeatureError(
                "Using icetl without a SparkSession",
                hint="Build one with SparkSession.builder.getOrCreate()",
            )
        return session

    # -- properties --------------------------------------------------------

    @property
    def version(self) -> str:
        """The Spark version whose semantics we target, not icetl's own version."""
        return SPARK_VERSION

    @property
    def icetlVersion(self) -> str:
        return __version__

    @property
    def conf(self) -> RuntimeConfig:
        return RuntimeConfig(self._conf)

    @property
    def settings(self) -> IcetlSettings:
        """The resolved settings. Not part of PySpark's API."""
        return self._settings

    @property
    def catalog(self) -> Any:
        raise UnsupportedFeatureError("spark.catalog", phase="Phase 9")

    @property
    def read(self) -> Any:
        raise UnsupportedFeatureError(
            "spark.read",
            phase="Phase 11",
            hint="Use spark.table() for Iceberg tables",
        )

    @property
    def sparkContext(self) -> Any:
        raise UnsupportedFeatureError(
            "spark.sparkContext",
            hint="There is no RDD layer here, and there will not be one",
        )

    @property
    def udf(self) -> Any:
        raise UnsupportedFeatureError("spark.udf", phase="Phase 11")

    def createDataFrame(self, *args: Any, **kwargs: Any) -> DataFrame:
        raise UnsupportedFeatureError("spark.createDataFrame", phase="Phase 4")

    def range(self, *args: Any, **kwargs: Any) -> DataFrame:
        raise UnsupportedFeatureError("spark.range", phase="Phase 4")

    # -- entry points ------------------------------------------------------

    def table(self, tableName: str) -> DataFrame:
        """`SELECT * FROM tableName` as a DataFrame.

        The reference is resolved against the catalog now, so a missing table fails
        here rather than at the first action.
        """
        if not isinstance(tableName, str):
            raise PySparkTypeError(f"table() expects a name, got {type(tableName).__name__}.")
        source = self._source_for(tableName)
        plan = exp.select(exp.Star()).from_(source_table(tableName))
        return DataFrame(self, plan, {source.key: source})

    def sql(self, sqlQuery: str, **kwargs: Any) -> DataFrame:
        """Run a Spark SQL query, producing the same kind of plan the API builds."""
        if kwargs:
            raise UnsupportedFeatureError(
                f"spark.sql() named-argument substitution ({', '.join(sorted(kwargs))})",
                phase="Phase 4",
            )
        if not isinstance(sqlQuery, str):
            raise PySparkTypeError(f"sql() expects a string, got {type(sqlQuery).__name__}.")
        try:
            statements = sqlglot.parse(sqlQuery, read="spark")
        except Exception as exc:
            raise ParseException(f"Could not parse the query: {exc}") from exc

        parsed = [s for s in statements if s is not None]
        if len(parsed) != 1:
            raise ParseException(f"spark.sql() takes exactly one statement, got {len(parsed)}.")
        plan = parsed[0]

        for node_type, (keyword, phase) in _DDL_PHASES.items():
            if isinstance(plan, node_type):
                raise UnsupportedFeatureError(f"{keyword} statements", phase=phase)
        if not isinstance(plan, (exp.Select, exp.Union, exp.Subquery)):
            raise UnsupportedFeatureError(
                f"{type(plan).__name__.upper()} statements", phase="Phase 4"
            )

        sources = {key: self._source_for(key) for key in collect_source_keys(plan)}
        return DataFrame(self, plan, sources)

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

    def _next_alias(self) -> str:
        """A subquery alias unique within this session, so nesting never collides."""
        with self._lock:
            self._counter += 1
            return f"_q{self._counter}"

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
        # Spark semantics first, so the optimizer -- and therefore pushdown -- reasons
        # about the tree that will actually execute (PLAN.md 3.5).
        conformed = apply_spark_semantics(plan, ansi_mode=self._settings.sql.ansi_mode)
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
            raise UnsupportedFeatureError("Using a SparkSession after stop()")
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
        with SparkSession._active_lock:
            if SparkSession._active is self:
                SparkSession._active = None

    def __enter__(self) -> SparkSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def __repr__(self) -> str:
        name = self._conf.get("spark.app.name", "icetl")
        return f"<SparkSession app={name!r} catalog={self._settings.catalog.name!r}>"

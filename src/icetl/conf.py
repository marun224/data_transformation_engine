"""Configuration: a SparkConf-alike plus resolution into typed settings.

Four layers, highest priority first:

    1. `SparkSession.builder.config(...)` / `IcetlConf.set(...)`
    2. process environment
    3. `.env` file
    4. built-in defaults

Spark-style keys are accepted verbatim so an existing script's `.config(...)` calls
keep meaning what they meant:

    spark.sql.catalog.<name>.uri            ==  ICETL_CATALOG_URI
    spark.sql.catalog.<name>.warehouse      ==  ICETL_CATALOG_WAREHOUSE
    spark.sql.catalog.<name>.s3.endpoint    ==  ICETL_S3_ENDPOINT
    spark.sql.defaultCatalog                ==  ICETL_CATALOG_NAME

Per P7 ("no config knobs"), DuckDB threads and memory are left unset by default so
DuckDB sizes them from the machine. Only spill-to-disk is configured up front,
because DuckDB will not spill at all without a temp directory.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import dotenv_values

from icetl.errors import ConfigurationError

__all__ = [
    "CatalogSettings",
    "EngineSettings",
    "IcetlConf",
    "IcetlSettings",
    "S3Settings",
    "resolve_settings",
]

_TRUTHY = frozenset({"1", "true", "yes", "y", "on"})
_FALSEY = frozenset({"0", "false", "no", "n", "off"})

# Spark-style prefixes we understand.
_SPARK_CATALOG_PREFIX = "spark.sql.catalog."
_SPARK_DEFAULT_CATALOG = "spark.sql.defaultCatalog"

_DEFAULT_CATALOG_NAME = "rest"
_DEFAULT_CATALOG_TYPE = "rest"
_DEFAULT_CATALOG_URI = "http://localhost:8182"


def _as_bool(value: str | bool | None, *, key: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSEY:
        return False
    raise ConfigurationError(
        f"{key}={value!r} is not a boolean. Use one of {sorted(_TRUTHY | _FALSEY)}."
    )


def _as_int(value: str | int | None, *, key: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key}={value!r} is not an integer.") from exc


class IcetlConf:
    """A mutable key/value bag with SparkConf's method names.

    `SparkSession.builder.config(...)` writes here in Phase 1. Values are stored as
    strings, as Spark does, so `.get()` round-trips whatever was set.
    """

    def __init__(self, pairs: Mapping[str, Any] | None = None) -> None:
        self._data: dict[str, str] = {}
        if pairs:
            for key, value in pairs.items():
                self.set(key, value)

    def set(self, key: str, value: Any) -> IcetlConf:
        if isinstance(value, bool):
            # Spark renders booleans lowercase; `str(True)` would give "True".
            self._data[key] = "true" if value else "false"
        else:
            self._data[key] = str(value)
        return self

    def setIfMissing(self, key: str, value: Any) -> IcetlConf:
        if key not in self._data:
            self.set(key, value)
        return self

    def setAll(self, pairs: Mapping[str, Any]) -> IcetlConf:
        for key, value in pairs.items():
            self.set(key, value)
        return self

    def get(self, key: str, defaultValue: str | None = None) -> str | None:
        return self._data.get(key, defaultValue)

    def getAll(self) -> list[tuple[str, str]]:
        return sorted(self._data.items())

    def contains(self, key: str) -> bool:
        return key in self._data

    def remove(self, key: str) -> IcetlConf:
        self._data.pop(key, None)
        return self

    def copy(self) -> IcetlConf:
        return IcetlConf(self._data)

    def toDebugString(self) -> str:
        return "\n".join(f"{k}={v}" for k, v in self.getAll())

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self.getAll())

    def __repr__(self) -> str:
        return f"IcetlConf({len(self._data)} entries)"


# ---------------------------------------------------------------------------
# Typed settings
# ---------------------------------------------------------------------------

# Redacted in every debug rendering. Substring match, so `s3.secret-access-key`
# and `ICETL_S3_SECRET_ACCESS_KEY` are both covered.
_SECRET_MARKERS = ("secret", "password", "token", "access-key", "access_key", "credential")


def redact(key: str, value: str) -> str:
    """Mask a value if its key looks like a credential."""
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_MARKERS):
        return "***" if value else ""
    return value


@dataclass(frozen=True)
class S3Settings:
    """Object-store access, shared by PyIceberg (reads metadata) and DuckDB (reads data)."""

    endpoint: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    region: str | None = None
    # MinIO addresses buckets path-style; real AWS S3 uses virtual-host style.
    path_style_access: bool = True

    @property
    def configured(self) -> bool:
        """True when anything S3-related was set, which is what turns httpfs on."""
        return any([self.endpoint, self.access_key_id, self.secret_access_key, self.session_token])

    def pyiceberg_properties(self) -> dict[str, str]:
        """Map to PyIceberg's `s3.*` property names.

        PyIceberg has no `path-style-access`; it has the inverse,
        `s3.force-virtual-addressing`.
        """
        props: dict[str, str] = {}
        if self.endpoint:
            props["s3.endpoint"] = self.endpoint
        if self.access_key_id:
            props["s3.access-key-id"] = self.access_key_id
        if self.secret_access_key:
            props["s3.secret-access-key"] = self.secret_access_key
        if self.session_token:
            props["s3.session-token"] = self.session_token
        if self.region:
            props["s3.region"] = self.region
        props["s3.force-virtual-addressing"] = "false" if self.path_style_access else "true"
        return props

    def duckdb_endpoint(self) -> tuple[str, bool] | None:
        """Split the endpoint into DuckDB's `(host[:port], use_ssl)` form.

        DuckDB's ENDPOINT excludes the scheme and carries SSL as a separate flag.
        """
        if not self.endpoint:
            return None
        parsed = urlparse(self.endpoint if "//" in self.endpoint else f"//{self.endpoint}")
        host = parsed.netloc or parsed.path
        return host, parsed.scheme == "https"


@dataclass(frozen=True)
class CatalogSettings:
    """One named Iceberg catalog."""

    name: str = _DEFAULT_CATALOG_NAME
    type: str = _DEFAULT_CATALOG_TYPE
    uri: str | None = _DEFAULT_CATALOG_URI
    warehouse: str | None = None
    # Anything else under `spark.sql.catalog.<name>.*`, passed through untouched.
    extra: Mapping[str, str] = field(default_factory=dict)

    def pyiceberg_properties(self, s3: S3Settings) -> dict[str, str]:
        """Build the property dict for `pyiceberg.catalog.load_catalog`."""
        props: dict[str, str] = {"type": self.type}
        if self.uri:
            props["uri"] = self.uri
        if self.warehouse:
            props["warehouse"] = self.warehouse
        props.update(s3.pyiceberg_properties())
        # Explicit `extra` wins: it is the escape hatch for properties we do not model.
        props.update(self.extra)
        return props


@dataclass(frozen=True)
class EngineSettings:
    """DuckDB engine knobs. All optional -- unset means "let DuckDB decide" (P7)."""

    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | None = None

    def resolved_temp_directory(self) -> str:
        """Where DuckDB spills. Always set: without it DuckDB will not spill at all."""
        if self.temp_directory:
            return self.temp_directory
        return str(Path(tempfile.gettempdir()) / "icetl-duckdb-spill")


@dataclass(frozen=True)
class SqlSettings:
    """Spark semantics that a session can be asked to change."""

    #: `spark.sql.ansi.enabled`. Off by default, as in Spark: a failed cast gives
    #: NULL rather than raising. Turning it on opts into strict casting only -- it
    #: does not make integer overflow wrap, which DuckDB cannot do. See
    #: `compat/divergence.md`.
    ansi_mode: bool = False


@dataclass(frozen=True)
class IcetlSettings:
    """Everything resolved, ready to hand to the catalog registry and the engine."""

    catalog: CatalogSettings = field(default_factory=CatalogSettings)
    s3: S3Settings = field(default_factory=S3Settings)
    engine: EngineSettings = field(default_factory=EngineSettings)
    sql: SqlSettings = field(default_factory=SqlSettings)
    default_namespace: tuple[str, ...] = ()

    def with_catalog(self, **changes: Any) -> IcetlSettings:
        return replace(self, catalog=replace(self.catalog, **changes))

    def debug_pairs(self) -> list[tuple[str, str]]:
        """Flat, secret-redacted view for `--verbose` and error messages."""
        pairs: list[tuple[str, str]] = [
            ("catalog.name", self.catalog.name),
            ("catalog.type", self.catalog.type),
            ("catalog.uri", self.catalog.uri or ""),
            ("catalog.warehouse", self.catalog.warehouse or ""),
            ("default_namespace", ".".join(self.default_namespace) or "(none)"),
            ("s3.endpoint", self.s3.endpoint or ""),
            ("s3.region", self.s3.region or ""),
            ("s3.path_style_access", str(self.s3.path_style_access).lower()),
            ("s3.access_key_id", redact("access-key", self.s3.access_key_id or "")),
            ("s3.secret_access_key", redact("secret", self.s3.secret_access_key or "")),
            ("sql.ansi_mode", str(self.sql.ansi_mode).lower()),
            ("engine.threads", str(self.engine.threads) if self.engine.threads else "(auto)"),
            ("engine.memory_limit", self.engine.memory_limit or "(auto)"),
            ("engine.temp_directory", self.engine.resolved_temp_directory()),
        ]
        for key, value in sorted(self.catalog.extra.items()):
            pairs.append((f"catalog.extra.{key}", redact(key, value)))
        return pairs


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class _Layers:
    """Reads a key across conf -> env -> .env -> default, in that order."""

    def __init__(self, conf: IcetlConf, env: Mapping[str, str], dotenv: Mapping[str, str]) -> None:
        self._conf = conf
        self._env = env
        self._dotenv = dotenv

    def conf_key(self, *keys: str) -> str | None:
        for key in keys:
            value = self._conf.get(key)
            if value is not None:
                return value
        return None

    def env_key(self, *keys: str) -> str | None:
        for source in (self._env, self._dotenv):
            for key in keys:
                value = source.get(key)
                if value is not None and value != "":
                    return value
        return None

    def lookup(
        self, conf_keys: tuple[str, ...], env_keys: tuple[str, ...], default: str | None = None
    ) -> str | None:
        return self.conf_key(*conf_keys) or self.env_key(*env_keys) or default


def _resolve_catalog_name(layers: _Layers, conf: IcetlConf) -> str:
    explicit = layers.lookup(
        (_SPARK_DEFAULT_CATALOG, "icetl.catalog.name"), ("ICETL_CATALOG_NAME",)
    )
    if explicit:
        return explicit
    # Fall back to whichever catalog the script configured, e.g. a lone
    # `spark.sql.catalog.prod.uri` implies the catalog is called "prod".
    named = {
        key[len(_SPARK_CATALOG_PREFIX) :].split(".", 1)[0]
        for key, _ in conf.getAll()
        if key.startswith(_SPARK_CATALOG_PREFIX)
    }
    if len(named) == 1:
        return named.pop()
    if len(named) > 1:
        raise ConfigurationError(
            f"Several catalogs are configured ({', '.join(sorted(named))}) but none is "
            f"marked as the default. Set `{_SPARK_DEFAULT_CATALOG}` or ICETL_CATALOG_NAME."
        )
    return _DEFAULT_CATALOG_NAME


def _catalog_scoped(conf: IcetlConf, catalog_name: str) -> dict[str, str]:
    """Collect `spark.sql.catalog.<name>.<rest>` into `{<rest>: value}`."""
    prefix = f"{_SPARK_CATALOG_PREFIX}{catalog_name}."
    return {key[len(prefix) :]: value for key, value in conf.getAll() if key.startswith(prefix)}


# Scoped keys consumed into typed fields; everything else flows into `extra`.
_MODELLED_SCOPED_KEYS = frozenset(
    {
        "type",
        "uri",
        "warehouse",
        "s3.endpoint",
        "s3.access-key-id",
        "s3.secret-access-key",
        "s3.session-token",
        "s3.region",
        "s3.path-style-access",
    }
)


def resolve_settings(
    conf: IcetlConf | None = None,
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> IcetlSettings:
    """Collapse the four configuration layers into typed settings.

    `env` and `dotenv_path` are injectable so tests never touch the real environment
    or the developer's `.env`.
    """
    conf = conf or IcetlConf()
    env = os.environ if env is None else env

    if dotenv_path is None:
        candidate = Path.cwd() / ".env"
        dotenv: Mapping[str, str] = (
            {k: v for k, v in dotenv_values(candidate).items() if v is not None}
            if candidate.is_file()
            else {}
        )
    elif Path(dotenv_path).is_file():
        dotenv = {k: v for k, v in dotenv_values(dotenv_path).items() if v is not None}
    else:
        dotenv = {}

    layers = _Layers(conf, env, dotenv)
    catalog_name = _resolve_catalog_name(layers, conf)
    scoped = _catalog_scoped(conf, catalog_name)

    def pick(scoped_key: str, *env_keys: str, default: str | None = None) -> str | None:
        return scoped.get(scoped_key) or layers.env_key(*env_keys) or default

    s3 = S3Settings(
        endpoint=pick("s3.endpoint", "ICETL_S3_ENDPOINT", "AWS_ENDPOINT_URL"),
        access_key_id=pick("s3.access-key-id", "ICETL_S3_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"),
        secret_access_key=pick(
            "s3.secret-access-key", "ICETL_S3_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"
        ),
        session_token=pick("s3.session-token", "ICETL_S3_SESSION_TOKEN", "AWS_SESSION_TOKEN"),
        region=pick("s3.region", "ICETL_S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"),
        path_style_access=_as_bool(
            pick("s3.path-style-access", "ICETL_S3_PATH_STYLE_ACCESS"),
            key="s3.path-style-access",
            default=True,
        ),
    )

    catalog = CatalogSettings(
        name=catalog_name,
        type=pick("type", "ICETL_CATALOG_TYPE", default=_DEFAULT_CATALOG_TYPE)
        or _DEFAULT_CATALOG_TYPE,
        uri=pick("uri", "ICETL_CATALOG_URI", default=_DEFAULT_CATALOG_URI),
        warehouse=pick("warehouse", "ICETL_CATALOG_WAREHOUSE"),
        extra={k: v for k, v in scoped.items() if k not in _MODELLED_SCOPED_KEYS},
    )

    engine = EngineSettings(
        threads=_as_int(
            layers.lookup(("icetl.duckdb.threads",), ("ICETL_DUCKDB_THREADS",)),
            key="icetl.duckdb.threads",
        ),
        memory_limit=layers.lookup(("icetl.duckdb.memoryLimit",), ("ICETL_DUCKDB_MEMORY_LIMIT",)),
        temp_directory=layers.lookup(
            ("icetl.duckdb.tempDirectory",), ("ICETL_DUCKDB_TEMP_DIRECTORY",)
        ),
    )

    sql = SqlSettings(
        ansi_mode=_as_bool(
            layers.lookup(("spark.sql.ansi.enabled", "icetl.ansiMode"), ("ICETL_ANSI_MODE",)),
            key="spark.sql.ansi.enabled",
            default=False,
        )
    )

    namespace = layers.lookup(
        ("icetl.defaultNamespace", "spark.sql.defaultDatabase"), ("ICETL_DEFAULT_NAMESPACE",)
    )

    return IcetlSettings(
        catalog=catalog,
        s3=s3,
        engine=engine,
        sql=sql,
        default_namespace=tuple(p for p in (namespace or "").split(".") if p),
    )

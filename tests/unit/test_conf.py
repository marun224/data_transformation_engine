"""Configuration layering, Spark-key compatibility, and secret redaction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from icetl.conf import IcetlConf, IcetlSettings, S3Settings, resolve_settings
from icetl.errors import ConfigurationError

# Every test injects `env` and `dotenv_path`, so the developer's real environment and
# `.env` can never leak into a result.
NO_ENV: Mapping[str, str] = {}
NO_DOTENV = Path("does-not-exist.env")


def resolve(
    conf: IcetlConf | None = None,
    *,
    env: Mapping[str, str] = NO_ENV,
    dotenv_path: Path = NO_DOTENV,
) -> IcetlSettings:
    return resolve_settings(conf, env=env, dotenv_path=dotenv_path)


class TestIcetlConf:
    def test_set_get_roundtrip(self) -> None:
        conf = IcetlConf().set("a.b", "1")
        assert conf.get("a.b") == "1"
        assert conf.get("missing", "fallback") == "fallback"

    def test_booleans_render_spark_style(self) -> None:
        """Spark writes `true`, not Python's `True`; scripts compare against strings."""
        assert IcetlConf().set("k", True).get("k") == "true"
        assert IcetlConf().set("k", False).get("k") == "false"

    def test_set_if_missing_does_not_overwrite(self) -> None:
        conf = IcetlConf().set("k", "first").setIfMissing("k", "second")
        assert conf.get("k") == "first"

    def test_get_all_is_sorted(self) -> None:
        conf = IcetlConf({"z": 1, "a": 2})
        assert [k for k, _ in conf.getAll()] == ["a", "z"]


class TestLayering:
    def test_defaults_when_nothing_is_set(self) -> None:
        settings = resolve()
        assert settings.catalog.name == "rest"
        assert settings.catalog.type == "rest"
        assert settings.catalog.uri == "http://localhost:8182"

    def test_env_overrides_default(self) -> None:
        settings = resolve(env={"ICETL_CATALOG_URI": "http://box:9999"})
        assert settings.catalog.uri == "http://box:9999"

    def test_conf_overrides_env(self) -> None:
        """`.config(...)` is the highest layer -- an explicit call always wins."""
        conf = IcetlConf({"spark.sql.catalog.rest.uri": "http://from-conf:1"})
        settings = resolve(conf, env={"ICETL_CATALOG_URI": "http://from-env:2"})
        assert settings.catalog.uri == "http://from-conf:1"

    def test_dotenv_is_read_and_env_wins_over_it(self, tmp_path: Path) -> None:
        dotenv = tmp_path / ".env"
        dotenv.write_text("ICETL_CATALOG_URI=http://from-dotenv:3\nICETL_S3_REGION=eu-west-1\n")

        settings = resolve(env={"ICETL_CATALOG_URI": "http://from-env:2"}, dotenv_path=dotenv)
        assert settings.catalog.uri == "http://from-env:2"
        # Not shadowed by the environment, so the .env value survives.
        assert settings.s3.region == "eu-west-1"


class TestCatalogNaming:
    def test_spark_default_catalog_key(self) -> None:
        conf = IcetlConf({"spark.sql.defaultCatalog": "prod"})
        assert resolve(conf).catalog.name == "prod"

    def test_single_configured_catalog_is_inferred(self) -> None:
        """A lone `spark.sql.catalog.prod.uri` names the catalog without repeating it."""
        conf = IcetlConf({"spark.sql.catalog.prod.uri": "http://prod:8182"})
        settings = resolve(conf)
        assert settings.catalog.name == "prod"
        assert settings.catalog.uri == "http://prod:8182"

    def test_ambiguous_catalogs_are_rejected(self) -> None:
        """Two catalogs and no default is a setup mistake worth failing loudly on."""
        conf = IcetlConf(
            {"spark.sql.catalog.a.uri": "http://a", "spark.sql.catalog.b.uri": "http://b"}
        )
        with pytest.raises(ConfigurationError, match="none is marked as the default"):
            resolve(conf)

    def test_unmodelled_scoped_keys_flow_into_extra(self) -> None:
        conf = IcetlConf(
            {
                "spark.sql.defaultCatalog": "prod",
                "spark.sql.catalog.prod.uri": "http://prod",
                "spark.sql.catalog.prod.header.X-Tenant": "acme",
            }
        )
        settings = resolve(conf)
        assert settings.catalog.extra == {"header.X-Tenant": "acme"}
        assert "header.X-Tenant" in settings.catalog.pyiceberg_properties(S3Settings())


class TestS3:
    def test_aws_names_are_accepted_as_fallback(self) -> None:
        settings = resolve(
            env={"AWS_ACCESS_KEY_ID": "ak", "AWS_SECRET_ACCESS_KEY": "sk", "AWS_REGION": "ap-1"}
        )
        assert settings.s3.access_key_id == "ak"
        assert settings.s3.region == "ap-1"

    def test_icetl_names_beat_aws_names(self) -> None:
        settings = resolve(env={"ICETL_S3_REGION": "eu-1", "AWS_REGION": "us-1"})
        assert settings.s3.region == "eu-1"

    def test_path_style_maps_to_pyicebergs_inverse_property(self) -> None:
        """PyIceberg has no path-style flag -- it has `force-virtual-addressing`."""
        assert (
            S3Settings(path_style_access=True).pyiceberg_properties()["s3.force-virtual-addressing"]
            == "false"
        )
        assert (
            S3Settings(path_style_access=False).pyiceberg_properties()[
                "s3.force-virtual-addressing"
            ]
            == "true"
        )

    @pytest.mark.parametrize(
        ("endpoint", "expected"),
        [
            ("http://localhost:9000", ("localhost:9000", False)),
            ("https://minio.internal", ("minio.internal", True)),
            # DuckDB's ENDPOINT excludes the scheme, so a bare host must pass through.
            ("localhost:9000", ("localhost:9000", False)),
        ],
    )
    def test_duckdb_endpoint_split(self, endpoint: str, expected: tuple[str, bool]) -> None:
        assert S3Settings(endpoint=endpoint).duckdb_endpoint() == expected

    def test_no_endpoint_means_none(self) -> None:
        assert S3Settings().duckdb_endpoint() is None

    def test_configured_is_false_without_credentials(self) -> None:
        """Drives whether httpfs loads at all, so an empty config must stay quiet."""
        assert S3Settings().configured is False
        assert S3Settings(region="us-east-1").configured is False
        assert S3Settings(endpoint="http://minio:9000").configured is True

    @pytest.mark.parametrize("value", ["true", "TRUE", "yes", "1", "on"])
    def test_truthy_spellings(self, value: str) -> None:
        assert resolve(env={"ICETL_S3_PATH_STYLE_ACCESS": value}).s3.path_style_access is True

    @pytest.mark.parametrize("value", ["false", "no", "0", "off"])
    def test_falsey_spellings(self, value: str) -> None:
        assert resolve(env={"ICETL_S3_PATH_STYLE_ACCESS": value}).s3.path_style_access is False

    def test_nonsense_boolean_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="not a boolean"):
            resolve(env={"ICETL_S3_PATH_STYLE_ACCESS": "maybe"})


class TestRedaction:
    def test_debug_pairs_never_expose_secrets(self) -> None:
        settings = resolve(
            env={
                "ICETL_S3_ACCESS_KEY_ID": "AKIAREALKEY",
                "ICETL_S3_SECRET_ACCESS_KEY": "s3cr3t-value",
                "ICETL_S3_ENDPOINT": "http://minio:9000",
            }
        )
        rendered = dict(settings.debug_pairs())
        assert rendered["s3.access_key_id"] == "***"
        assert rendered["s3.secret_access_key"] == "***"
        # Non-secret values stay readable -- redaction must not blind the operator.
        assert rendered["s3.endpoint"] == "http://minio:9000"

        blob = "\n".join(f"{k}={v}" for k, v in settings.debug_pairs())
        assert "AKIAREALKEY" not in blob
        assert "s3cr3t-value" not in blob


class TestEngine:
    def test_threads_and_memory_default_to_unset(self) -> None:
        """P7: DuckDB already uses every core. We only ever narrow it deliberately."""
        settings = resolve()
        assert settings.engine.threads is None
        assert settings.engine.memory_limit is None

    def test_temp_directory_always_resolves(self) -> None:
        """Spill needs a directory; DuckDB will not spill without one."""
        assert resolve().engine.resolved_temp_directory()

    def test_non_integer_thread_count_is_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="not an integer"):
            resolve(env={"ICETL_DUCKDB_THREADS": "lots"})


def test_default_namespace_splits_on_dots() -> None:
    assert resolve(env={"ICETL_DEFAULT_NAMESPACE": "a.b"}).default_namespace == ("a", "b")
    assert resolve().default_namespace == ()

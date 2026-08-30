"""Table-reference parsing and resolution."""

from __future__ import annotations

import pytest

from icetl.catalog import TableResolver, parse_table_ref
from icetl.catalog.registry import CatalogRegistry
from icetl.errors import AnalysisException, CatalogNotFoundError, TableNotFoundError


class TestParsing:
    def test_bare_name_uses_the_default_namespace(self) -> None:
        ref = parse_table_ref("trips", default_namespace=("nyc",))
        assert (ref.catalog, ref.namespace, ref.name) == (None, ("nyc",), "trips")

    def test_two_parts_are_namespace_and_table(self) -> None:
        ref = parse_table_ref("nyc.trips", default_namespace=("ignored",))
        assert (ref.catalog, ref.namespace, ref.name) == (None, ("nyc",), "trips")

    def test_three_parts_are_a_multi_level_namespace_by_default(self) -> None:
        """Without a registry the leading part cannot be a catalog, so it is a namespace.

        Iceberg allows nested namespaces, so this is the reading that cannot be wrong.
        """
        ref = parse_table_ref("a.b.trips")
        assert (ref.catalog, ref.namespace, ref.name) == (None, ("a", "b"), "trips")

    def test_three_parts_split_on_a_known_catalog(self) -> None:
        ref = parse_table_ref("prod.nyc.trips", is_known_catalog=lambda n: n == "prod")
        assert (ref.catalog, ref.namespace, ref.name) == ("prod", ("nyc",), "trips")

    def test_unknown_leading_part_stays_a_namespace(self) -> None:
        ref = parse_table_ref("a.nyc.trips", is_known_catalog=lambda n: n == "prod")
        assert (ref.catalog, ref.namespace, ref.name) == (None, ("a", "nyc"), "trips")

    def test_backticks_quote_a_part_containing_dots(self) -> None:
        ref = parse_table_ref("`odd.name`.trips")
        assert (ref.namespace, ref.name) == (("odd.name",), "trips")

    def test_backticks_quote_spaces(self) -> None:
        assert parse_table_ref("`my db`.`my table`").name == "my table"

    def test_doubled_backtick_is_a_literal(self) -> None:
        assert parse_table_ref("ns.`we``ird`").name == "we`ird"

    @pytest.mark.parametrize("bad", ["`unclosed.trips", "nyc..trips", ".trips", "nyc."])
    def test_malformed_references_are_rejected(self, bad: str) -> None:
        with pytest.raises(AnalysisException):
            parse_table_ref(bad)

    def test_identifier_and_str_roundtrip(self) -> None:
        ref = parse_table_ref("prod.a.b.trips", is_known_catalog=lambda n: n == "prod")
        assert ref.identifier == ("a", "b", "trips")
        assert str(ref) == "prod.a.b.trips"


class TestResolution:
    def test_resolves_against_the_local_catalog(self, resolver: TableResolver) -> None:
        resolved = resolver.resolve("fx.plain")
        assert resolved.ref.name == "plain"
        assert resolved.catalog_name == "test"
        assert [f.name for f in resolved.table.schema().fields] == ["id", "vendor", "amount"]

    def test_bare_name_uses_the_current_namespace(self, resolver: TableResolver) -> None:
        assert resolver.current_namespace == ("fx",)
        assert resolver.resolve("plain").ref.identifier == ("fx", "plain")

    def test_missing_table_raises_table_not_found(self, resolver: TableResolver) -> None:
        with pytest.raises(TableNotFoundError, match="no_such_table"):
            resolver.resolve("fx.no_such_table")

    def test_unconfigured_catalog_raises(self, registry: CatalogRegistry) -> None:
        with pytest.raises(CatalogNotFoundError, match="ghost"):
            registry.get("ghost")

    def test_no_namespace_anywhere_is_an_analysis_error(self, registry: CatalogRegistry) -> None:
        """A bare name with no current namespace must say so, not guess."""
        bare = TableResolver(registry, default_namespace=())
        with pytest.raises(AnalysisException, match="no current namespace"):
            bare.resolve("plain")

    def test_setting_a_missing_namespace_is_rejected(self, resolver: TableResolver) -> None:
        from icetl.errors import NamespaceNotFoundError

        with pytest.raises(NamespaceNotFoundError):
            resolver.set_current_namespace("nope")


class TestRegistry:
    def test_is_known_does_not_connect(self) -> None:
        """Resolution calls this on every three-part reference, so it must stay cheap.

        The settings here point at a URI that would fail to connect; the assertion is
        that we never find out.
        """
        from icetl.conf import CatalogSettings, IcetlSettings

        registry = CatalogRegistry(
            IcetlSettings(catalog=CatalogSettings(name="prod", uri="http://127.0.0.1:1"))
        )
        assert registry.is_known("prod") is True
        assert registry.is_known("other") is False

    def test_registered_catalogs_are_known_and_returned(
        self, registry: CatalogRegistry, catalog: object
    ) -> None:
        assert registry.get("test") is catalog
        assert "test" in list(registry.names())

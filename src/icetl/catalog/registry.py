"""Named Iceberg catalogs, built lazily from settings and cached per session."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyiceberg.catalog import Catalog, load_catalog

from icetl.conf import IcetlSettings
from icetl.errors import CatalogNotFoundError, ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["CatalogRegistry"]


class CatalogRegistry:
    """Resolves catalog names to live PyIceberg `Catalog` objects.

    Construction is lazy: nothing connects until a catalog is actually asked for, so
    importing icetl never blocks on a network call.
    """

    def __init__(self, settings: IcetlSettings) -> None:
        self._settings = settings
        self._catalogs: dict[str, Catalog] = {}

    @property
    def default_name(self) -> str:
        return self._settings.catalog.name

    @property
    def settings(self) -> IcetlSettings:
        return self._settings

    def register(self, name: str, catalog: Catalog) -> None:
        """Inject an already-built catalog.

        This is how the test suite supplies a local `SqlCatalog` and how a user can
        hand us a catalog they configured themselves.
        """
        self._catalogs[name] = catalog

    def is_known(self, name: str) -> bool:
        """True if `name` names a catalog we could produce.

        Used by the resolver to decide whether the leading part of `a.b.c` is a
        catalog or a namespace, so it must not trigger a connection.
        """
        return name == self.default_name or name in self._catalogs

    def names(self) -> Iterable[str]:
        return sorted({self.default_name, *self._catalogs})

    def get(self, name: str | None = None) -> Catalog:
        """Return the named catalog, building and caching it on first use."""
        resolved = name or self.default_name
        if resolved in self._catalogs:
            return self._catalogs[resolved]

        if resolved != self.default_name:
            known = ", ".join(self.names())
            raise CatalogNotFoundError(
                f"Catalog {resolved!r} is not configured. Known catalogs: {known}. "
                f"Configure it with `icetl.catalog.{resolved}.uri=...`."
            )

        catalog = self._build(resolved)
        self._catalogs[resolved] = catalog
        return catalog

    def _build(self, name: str) -> Catalog:
        properties = self._settings.catalog.pyiceberg_properties(self._settings.s3)
        try:
            return load_catalog(name, **properties)
        except Exception as exc:
            # Surface the endpoint we actually tried; "connection refused" with no URI
            # is the single most common time-waster when bringing a catalog up.
            raise ConfigurationError(
                f"Could not build catalog {name!r} "
                f"(type={self._settings.catalog.type!r}, uri={self._settings.catalog.uri!r}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def close(self) -> None:
        """Drop cached catalogs. Next `get()` rebuilds."""
        self._catalogs.clear()

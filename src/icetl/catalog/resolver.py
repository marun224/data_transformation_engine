"""Turning a table reference string into a live PyIceberg table.

Spark's rules, which we follow:

    "trips"                  -> default catalog, current namespace
    "nyc.trips"              -> default catalog, namespace `nyc`
    "prod.nyc.trips"         -> catalog `prod` if it is configured,
                                otherwise default catalog, namespace `prod.nyc`
    "`odd name`.trips"       -> backticks quote a part containing dots or spaces

The three-part case is genuinely ambiguous -- Iceberg allows multi-level namespaces,
so `a.b.c` could be catalog `a` or namespace `a.b`. Spark resolves it by asking
whether a catalog named `a` is registered, and so do we. `CatalogRegistry.is_known`
answers without connecting, so resolution stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from icetl.catalog.registry import CatalogRegistry
from icetl.errors import AnalysisException, NamespaceNotFoundError, TableNotFoundError

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table

__all__ = ["ResolvedTable", "TableRef", "TableResolver", "parse_table_ref"]


def _split_identifier(ref: str) -> list[str]:
    """Split a dotted identifier, honouring backtick quoting.

    A doubled backtick inside a quoted part is a literal backtick, as in Spark.
    """
    parts: list[str] = []
    current: list[str] = []
    in_quotes = False
    index = 0
    while index < len(ref):
        char = ref[index]
        if char == "`":
            if in_quotes and index + 1 < len(ref) and ref[index + 1] == "`":
                current.append("`")
                index += 2
                continue
            in_quotes = not in_quotes
        elif char == "." and not in_quotes:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
        index += 1

    if in_quotes:
        raise AnalysisException(f"Unclosed backtick in table reference {ref!r}.")
    parts.append("".join(current))

    if any(part == "" for part in parts):
        raise AnalysisException(f"Table reference {ref!r} has an empty identifier part.")
    return parts


@dataclass(frozen=True)
class TableRef:
    """A parsed table reference. `catalog` is None when the default applies."""

    namespace: tuple[str, ...]
    name: str
    catalog: str | None = None

    @property
    def identifier(self) -> tuple[str, ...]:
        """The `(namespace..., name)` tuple PyIceberg's catalog API expects."""
        return (*self.namespace, self.name)

    def __str__(self) -> str:
        prefix = f"{self.catalog}." if self.catalog else ""
        return prefix + ".".join(self.identifier)


@dataclass(frozen=True)
class ResolvedTable:
    """A table reference bound to the catalog and PyIceberg table it names."""

    ref: TableRef
    catalog_name: str
    catalog: Catalog
    table: Table

    @property
    def qualified_name(self) -> str:
        return f"{self.catalog_name}.{'.'.join(self.ref.identifier)}"


def parse_table_ref(
    ref: str,
    *,
    default_namespace: tuple[str, ...] = (),
    is_known_catalog: object = None,
) -> TableRef:
    """Parse `ref` into a `TableRef`.

    `is_known_catalog` is an optional `(str) -> bool` predicate used to disambiguate
    the three-part case. Without it, the leading part is always treated as a
    namespace, which is the safe reading when no registry is available.
    """
    parts = _split_identifier(ref.strip())

    if len(parts) == 1:
        return TableRef(namespace=default_namespace, name=parts[0])

    if len(parts) >= 3 and callable(is_known_catalog) and is_known_catalog(parts[0]):
        return TableRef(catalog=parts[0], namespace=tuple(parts[1:-1]), name=parts[-1])

    return TableRef(namespace=tuple(parts[:-1]), name=parts[-1])


class TableResolver:
    """Resolves reference strings against a `CatalogRegistry`.

    Holds the current namespace, which `spark.catalog.setCurrentDatabase` mutates.
    """

    def __init__(
        self, registry: CatalogRegistry, default_namespace: tuple[str, ...] | None = None
    ) -> None:
        self._registry = registry
        self._current_namespace: tuple[str, ...] = (
            default_namespace
            if default_namespace is not None
            else registry.settings.default_namespace
        )

    @property
    def current_namespace(self) -> tuple[str, ...]:
        return self._current_namespace

    def set_current_namespace(self, namespace: str | tuple[str, ...]) -> None:
        parts = tuple(_split_identifier(namespace)) if isinstance(namespace, str) else namespace
        catalog = self._registry.get()
        if not catalog.namespace_exists(parts):
            raise NamespaceNotFoundError(f"Namespace {'.'.join(parts)!r} does not exist.")
        self._current_namespace = parts

    def parse(self, ref: str) -> TableRef:
        return parse_table_ref(
            ref,
            default_namespace=self._current_namespace,
            is_known_catalog=self._registry.is_known,
        )

    def resolve(self, ref: str) -> ResolvedTable:
        """Load the table `ref` names, or raise `TableNotFoundError`."""
        table_ref = self.parse(ref)

        if not table_ref.namespace:
            raise AnalysisException(
                f"Table reference {ref!r} has no namespace and no current namespace is set. "
                f"Use a qualified name like 'nyc.{table_ref.name}', or set "
                f"ICETL_DEFAULT_NAMESPACE."
            )

        catalog_name = table_ref.catalog or self._registry.default_name
        catalog = self._registry.get(table_ref.catalog)

        try:
            table = catalog.load_table(table_ref.identifier)
        except Exception as exc:
            raise TableNotFoundError(
                f"Table {'.'.join(table_ref.identifier)!r} was not found in catalog "
                f"{catalog_name!r}: {type(exc).__name__}: {exc}"
            ) from exc

        return ResolvedTable(ref=table_ref, catalog_name=catalog_name, catalog=catalog, table=table)

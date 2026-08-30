"""Catalog access: named catalogs and table-reference resolution.

Everything that talks to PyIceberg's catalog API lives behind this package, so the
PyIceberg 0.11 -> 0.12 API churn risk stays contained to two modules.
"""

from icetl.catalog.registry import CatalogRegistry
from icetl.catalog.resolver import ResolvedTable, TableRef, TableResolver, parse_table_ref

__all__ = [
    "CatalogRegistry",
    "ResolvedTable",
    "TableRef",
    "TableResolver",
    "parse_table_ref",
]

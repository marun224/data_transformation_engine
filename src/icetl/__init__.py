"""icetl — a DataFrame and SQL surface over Apache Iceberg, executed by DuckDB.

See PLAN.md for the architecture. The short version:

    sqlglot    is the single IR         (both `Session.sql()` and DataFrame calls)
    PyIceberg  plans and commits        (catalog, schema, file pruning, snapshots)
    DuckDB     executes                 (parquet over httpfs, Arrow out)
"""

__version__ = "0.0.1"

__all__ = ["__version__"]

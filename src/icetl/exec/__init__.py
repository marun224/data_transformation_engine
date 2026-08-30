"""Execution: the DuckDB connection, and turning a planned scan into SQL for it."""

from icetl.exec.engine import DuckDBEngine
from icetl.exec.scan_planner import ColumnAlias, FileGroup, ScanPlan, plan_scan
from icetl.exec.source_sql import build_source

__all__ = [
    "ColumnAlias",
    "DuckDBEngine",
    "FileGroup",
    "ScanPlan",
    "build_source",
    "plan_scan",
]

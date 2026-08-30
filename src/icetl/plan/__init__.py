"""The logical plan: one sqlglot expression tree, shared by both user surfaces.

`spark.sql("...")` and `df.filter(...)` produce the *same* kind of object (P1), so
everything downstream is written once and covers both:

    build  ->  analyse  ->  bind a schema  ->  optimize  ->  read the pruning facts
           ->  plan the scans  ->  substitute sources  ->  generate DuckDB SQL

`schema` and `optimizer` are the middle of that pipeline, `pushdown` and
`annotations` are how its conclusions reach `icetl.exec.scan_planner`, and every one
of them degrades to "read everything" rather than failing.
"""

from icetl.plan.analysis import PlanAnalyzer, arrow_to_spark_schema, arrow_to_spark_type
from icetl.plan.annotations import PlanAnnotations, ScanRequest
from icetl.plan.builder import (
    ScanSource,
    as_expression,
    collect_source_keys,
    source_key,
    source_table,
    substitute_sources,
    wrap_as_subquery,
)
from icetl.plan.describe import describe_predicate
from icetl.plan.optimizer import RULES, OptimizedPlan, optimize_plan
from icetl.plan.pushdown import extract_scan_requests, translate_predicate
from icetl.plan.schema import SchemaBinder, iceberg_to_duckdb_type

__all__ = [
    "RULES",
    "OptimizedPlan",
    "PlanAnalyzer",
    "PlanAnnotations",
    "ScanRequest",
    "ScanSource",
    "SchemaBinder",
    "arrow_to_spark_schema",
    "arrow_to_spark_type",
    "as_expression",
    "collect_source_keys",
    "describe_predicate",
    "extract_scan_requests",
    "iceberg_to_duckdb_type",
    "optimize_plan",
    "source_key",
    "source_table",
    "substitute_sources",
    "translate_predicate",
    "wrap_as_subquery",
]

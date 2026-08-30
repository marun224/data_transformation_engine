"""Shadow `pyspark.sql.utils`.

Spark scripts import the exception types from here as often as from `pyspark.errors`,
so both spellings resolve.
"""

from icetl.errors import (
    AnalysisException,
    IllegalArgumentException,
    ParseException,
    PythonException,
    QueryExecutionException,
    UnsupportedOperationException,
)

__all__ = [
    "AnalysisException",
    "IllegalArgumentException",
    "ParseException",
    "PythonException",
    "QueryExecutionException",
    "UnsupportedOperationException",
]

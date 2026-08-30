"""Shadow `pyspark` package -- this is icetl, not Apache Spark.

Installing this distribution puts `pyspark` on the import path so existing scripts
run unchanged (decision 8 in PLAN.md). Real PySpark therefore cannot be installed
into the same environment.

Everything here is a thin re-export of `icetl`. No logic lives in this package; if
you are looking for behaviour, it is in `icetl.*`.
"""

from icetl import __version__ as __icetl_version__
from icetl.conf import IcetlConf as SparkConf
from icetl.errors import UnsupportedFeatureError
from icetl.sql import Row, SparkSession

# The PySpark version whose semantics we target. Scripts branch on this, so it
# reports a Spark version rather than icetl's own.
__version__ = "3.5.0"


class SparkContext:
    """Present so `from pyspark import SparkContext` resolves; using it does not.

    There is no RDD layer here and there will not be one -- the whole point is that
    DuckDB does the compute. Scripts that only *mention* `SparkContext` keep
    importing; scripts that build one get told why.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise UnsupportedFeatureError(
            "SparkContext", hint="icetl has no RDD layer; use SparkSession for DataFrames"
        )

    @classmethod
    def getOrCreate(cls, *args: object, **kwargs: object) -> "SparkContext":
        return cls()


__all__ = [
    "Row",
    "SparkConf",
    "SparkContext",
    "SparkSession",
    "__icetl_version__",
    "__version__",
]

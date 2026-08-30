"""Shadow `pyspark.sql.session` -- re-exports of `icetl.sql.session`."""

from icetl.sql.session import RuntimeConfig, SparkSession

__all__ = ["RuntimeConfig", "SparkSession"]

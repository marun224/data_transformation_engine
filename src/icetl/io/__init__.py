"""Table upkeep: the jobs an Iceberg table needs that are not reads or writes.

Compaction, snapshot expiry and orphan-file cleanup are normally somebody else's
Spark job. On a single node they are ordinary Python, and PLAN.md 3.6 makes them
load-bearing rather than optional: the securities table is only prunable by column
statistics if its files are large and sorted, which is what compaction produces.
"""

from icetl.io.maintenance import (
    CompactionResult,
    ExpiryResult,
    OrphanScan,
    TableMaintenance,
)

__all__ = ["CompactionResult", "ExpiryResult", "OrphanScan", "TableMaintenance"]

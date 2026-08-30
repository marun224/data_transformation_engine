# icetl — a drop-in PySpark surface over Apache Iceberg, executed by DuckDB

Single-node ETL library. Your existing PySpark scripts run unchanged: the distribution
ships a top-level `pyspark` package that shadows the real one.

```python
from pyspark.sql import SparkSession, functions as F  # <- this is us

spark = SparkSession.builder.appName("etl").getOrCreate()
df = spark.table("nyc.yellow_tripdata").filter(F.col("VendorID") == 1)
df.show()
```

**sqlglot** is the single IR, **PyIceberg** plans and commits, **DuckDB** executes.
See [PLAN.md](PLAN.md) for the full design, phases, and the decisions behind them.

## Status

**Phase 0 — scaffolding.** Config, catalog resolution, DuckDB engine, and the
connectivity smoke test. The DataFrame API arrives in Phase 1.

## Setup

```bash
uv venv --python 3.12
uv sync --all-extras
cp .env.example .env      # then fill in your catalog + MinIO details
```

## Proof of life

Verifies REST catalog connectivity, MinIO access, and a DuckDB read end to end:

```bash
uv run python scripts/smoke_catalog.py
```

Point it at a different table with `--namespace` / `--table`, and use `--verbose`
to see the generated SQL and the resolved (secret-redacted) configuration.

## Development

```bash
uv run pytest                    # unit + local-fixture tests, no network
uv run pytest -m integration     # runs against your real REST catalog + MinIO
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests marked `integration` are deselected by default; everything else runs against a
local PyIceberg `SqlCatalog` (sqlite metadata, local-filesystem warehouse) with
generated fixture tables.

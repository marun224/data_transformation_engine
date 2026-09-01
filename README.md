# icetl — a DataFrame and SQL surface over Apache Iceberg, executed by DuckDB

Single-node ETL library. Read and transform Iceberg tables with a lazy DataFrame API or
plain SQL — the two are one code path, not two.

```python
from icetl.sql import Session, functions as F

session = Session.builder.appName("etl").getOrCreate()
df = session.table("nyc.yellow_tripdata").filter(F.col("VendorID") == 1)
df.show()
```

**sqlglot** is the single IR, **PyIceberg** plans and commits, **DuckDB** executes.
See [PLAN.md](PLAN.md) for the full design, phases, and the decisions behind them.

Where DuckDB's behaviour is a defensible choice rather than the only one, icetl follows
**Apache Spark 3.5** as a written specification — `1/0` is NULL, a failed cast is NULL,
`ORDER BY` puts nulls first ascending. That is a spec reference, not a dependency:
nothing here runs on, links against, or requires Spark. Every place the two engines
disagree is recorded in [divergence.md](src/icetl/compat/divergence.md), and every place a
dependency turned out to be wrong is in [FINDINGS.md](FINDINGS.md).

## Status

**Phases 0–8 complete** — the read side, writing back, and changing rows in place.
Config and catalog resolution, the DataFrame and SQL surfaces, the plan IR with predicate
and projection pushdown, a 314-name function library, relational breadth (joins,
`groupBy().agg()`, grouping sets, set operations, ordering, `na.*`/`stat.*`, caching,
temporary views), window functions, complex types, the write path — `df.write`,
`insertInto`, SQL `INSERT`, partitioned creation and dynamic partition overwrite — and
row-level `DELETE`, `UPDATE` and `MERGE`, the last with the full Spark merge grammar and
all copy-on-write. Phase 9 adds schema, DDL and snapshots. See [STATUS.md](STATUS.md).

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
uv run tox                       # lint, mypy, tests against the built wheel, then dist/
uv run tox -e py312              # just the tests
uv run tox -e dist               # just build dist/*.whl and *.tar.gz
uv run tox -e integration        # opt-in; needs the REST catalog + MinIO up
uv run tox -- -k pushdown        # arguments after `--` go to pytest
```

`tox` installs the built wheel and runs the suite against *that*, not against `src/`,
so anything missing from the package fails here rather than after a release. The
underlying tools are still available directly:

```bash
uv run pytest                    # unit + local-fixture tests, no network
uv run pytest -m integration     # runs against your real REST catalog + MinIO
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

Tests marked `integration` are deselected by default; everything else runs against a
local PyIceberg `SqlCatalog` (sqlite metadata, local-filesystem warehouse) with
generated fixture tables.

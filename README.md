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

**Phases 0–11 complete** — the read side, writing back, changing rows in place, the
catalog itself, and performance. Config and catalog resolution, the DataFrame and SQL
surfaces, the plan IR with predicate and projection pushdown, a 273-name function
library, relational breadth (joins, `groupBy().agg()`, grouping sets, set operations,
ordering, `na.*`/`stat.*`, caching, temporary views), window functions, complex types,
the write path — `df.write`, `insertInto`, SQL `INSERT`, partitioned creation and
dynamic partition overwrite — row-level `DELETE`, `UPDATE` and `MERGE` with the full
merge grammar, `session.catalog.*`, SQL DDL with partition and sort-order evolution,
`mergeSchema`, time travel (`VERSION AS OF` / `TIMESTAMP AS OF`) and Iceberg's metadata
tables, Python UDFs, single-node table maintenance, and convenience readers for
parquet/CSV/JSON. What remains is deferred by decision — see [STATUS.md](STATUS.md).

**Start here:** [GUIDE.md](GUIDE.md), or run
[notebooks/00_quickstart.ipynb](notebooks/00_quickstart.ipynb), which builds its own
warehouse and needs no catalog.

### On a wide table, ask for Arrow

`collect()` builds a `Row` per result row, and on the 200-column benchmark table that is
**95% of the wall time** — 25.4 s against 1.3 s for the same query as Arrow. Reach for
`toArrow()`, or stream it when the result will not fit:

```python
for batch in df.toArrowBatches():  # peak memory is one batch
    ...
```

Numbers, and how to read a regression, in [BENCHMARKS.md](BENCHMARKS.md).

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

Performance is tracked rather than assumed. `scripts/benchmark.py` times nine queries
over a generated 200-column table, checking each one's answer on every repeat; commit
the diff to [BENCHMARKS.md](BENCHMARKS.md) when you change anything on the read path:

```bash
uv run python scripts/benchmark.py                          # the default suite
uv run python scripts/benchmark.py --table nyc.yellow_tripdata
uv run python scripts/benchmark.py --markdown BENCHMARKS.md
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

### Two test tiers

Tests marked `integration` are deselected by default; everything else runs against a
local PyIceberg `SqlCatalog` (sqlite metadata, local-filesystem warehouse) with
generated fixture tables.

```bash
uv run pytest                                  # 1679 local tests, offline, ~2 min
uv run pytest -m "integration and not slow"    #  724 against the real catalog, ~3 min
uv run pytest -m integration                   #  737, adding the 41M-row scans
uv run pytest -m integration --it-reseed       #  rebuild the seed tables first
```

The integration tier talks to the REST catalog and object store named in `.env`. It
builds its own tables in `icetl_it` — replicas of the six local fixtures, plus slices of
real data carved out of `nyc.yellow_tripdata` — and never writes anywhere else: a
namespace guard refuses it, and a session-scoped witness reads every protected table's
snapshot id before and after the run and fails if either moved.

Both tiers assert the same behaviours. What differs is everything underneath — a REST
catalog rather than sqlite, MinIO rather than a temp directory, `s3://` paths, real
NULLs, real cardinality, 62-file scans. That is where the defects have been: five of
them are recorded in [FINDINGS.md](FINDINGS.md) §1.14, §1.15, §2.11, §2.12 and §2.13,
and none was reachable offline.

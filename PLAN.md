# Transformation Engine — Design & Implementation Plan

**Working name:** `icetl` (Iceberg ETL) — rename freely, it only appears in `pyproject.toml` and the import path.
**Status:** Draft v1.0 — 2026-08-17
**Owner:** Arun

---

## 1. What we are building

A **pip-installable, single-node ETL / transformation library** over **Apache
Iceberg** tables, executed by **DuckDB**, with a lazy **DataFrame API** and a **SQL
surface** that are one code path rather than two.

```python
from icetl.sql import Session, functions as F

session = Session.builder.appName("etl").getOrCreate()

df = (
    session.table("nyc.yellow_tripdata")
    .filter(F.col("tpep_pickup_datetime") >= "2023-01-01")
    .groupBy("VendorID")
    .agg(F.sum("total_amount").alias("revenue"))
)

df.write.mode("overwrite").saveAsTable("nyc.revenue_by_vendor")
```

**Semantics come from a written specification.** Where DuckDB's behaviour is a
defensible choice rather than the only one, icetl follows **Apache Spark 3.5** — called
*the reference* throughout, and exposed as `REFERENCE_SEMANTICS_VERSION`. That is a spec
reference, not a dependency: nothing here runs on, links against, or requires Spark, and
the public API is icetl's own. See decision 15.

### The three pillars

| Library | Role | Never used for |
|---|---|---|
| **sqlglot** | The **single intermediate representation**. Both the DataFrame API and `Session.sql()` produce the *same* sqlglot AST. Optimizes it, then generates DuckDB SQL. | Execution |
| **DuckDB** | The **execution engine**. Reads parquet directly from MinIO/S3 via `httpfs`, does all compute, returns Arrow. | Metadata, catalog |
| **PyIceberg** | The **catalog + metadata + write layer**. Resolves tables, gives schemas, prunes manifests/partitions to a concrete file list, commits new snapshots. | Compute |

### Design principles

| # | Principle |
|---|---|
| **P1** | **One IR.** SQL text and DataFrame calls converge on one sqlglot tree. No second code path, ever. |
| **P2** | **PyIceberg plans, DuckDB executes.** Metadata work never touches DuckDB; compute never touches PyIceberg. |
| **P3** | **Lazy by default.** Every transformation is tree-building. Only actions (`collect`, `show`, `count`, `write`, `toPandas`, …) execute. |
| **P4** | **Prune early.** Partition + file pruning happens in PyIceberg *before* a byte is read. Projection pruning happens before the SQL is generated. |
| **P5** | **The reference semantics are the spec.** Where DuckDB and the reference disagree, we bend DuckDB to the reference and write the divergence down. Silent behavioural drift is a bug. |
| **P6** | **Correctness over speed.** A slow correct path (Arrow fallback) always exists behind every fast path. |
| **P7** | **Single node, all cores.** No cluster, no config knobs. Use every core, spill to disk when memory runs out. |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  USER SURFACE  (pyspark-compatible)                              │
│  Session · DataFrame · Column · functions · Window · types  │
│  DataFrameReader / DataFrameWriter · Catalog · GroupedData       │
└──────────────┬──────────────────────────────┬────────────────────┘
               │ df.filter(...)               │ spark.sql("SELECT ...")
               │                              │
               │                    sqlglot.parse_one(sql, read="spark")
               ▼                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  LOGICAL PLAN  —  a single sqlglot expression tree                │
│  (exp.Select / exp.Join / exp.Group / exp.Window / exp.Union …)   │
└──────────────┬───────────────────────────────────────────────────┘
               │
               │  ◄── schema binding: PyIceberg Schema → sqlglot MappingSchema
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  OPTIMIZER                                                        │
│  qualify · normalize · unnest subqueries · simplify               │
│  predicate pushdown · projection pushdown · column pruning        │
│  ── extract per-table conjuncts → PyIceberg BooleanExpression ──  │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  SCAN PLANNER  (per Iceberg table reference)                      │
│  table.scan(row_filter=…, selected_fields=…, snapshot_id=…)       │
│         .plan_files()  →  [FileScanTask]                          │
│                                                                   │
│   copy-on-write only (decision 11): a scan is its data files.     │
│   A task carrying delete files is refused, never approximated.     │
│                          │                                        │
│                          ▼                                        │
│                 read_parquet([...])  ← streaming                  │
└──────────────┬───────────────────────────────────────────────────┘
               │  sqlglot.generate(dialect="duckdb")
               ▼
┌──────────────────────────────────────────────────────────────────┐
│  DUCKDB  — httpfs → MinIO, N threads, memory_limit, temp spill    │
└──────────────┬───────────────────────────────────────────────────┘
               │ Arrow RecordBatchReader
        ┌──────┴───────┐
        ▼              ▼
   results            WRITE PATH
   (arrow/pandas/     PyIceberg append · overwrite · dynamic partition
    polars/rows)      overwrite · upsert/MERGE · delete · DDL
```

### 2.1 Why sqlglot as the IR (and not custom plan nodes)

- `spark.sql("...")` and `df.filter(...)` become **literally the same object**, so
  every optimizer rule, every pushdown, and every test covers both surfaces at once.
- sqlglot ships a working optimizer (`qualify`, `pushdown_predicates`,
  `pushdown_projections`, `optimize_joins`, `eliminate_subqueries`, `simplify`) —
  we get the boring 80% for free and write only Iceberg-specific rules.
- Dialect translation Spark→DuckDB is sqlglot's core competency: `Session.sql()` can
  accept Spark SQL, Trino, Snowflake, or anything else, and it all lands in one tree.
- **The cost:** sqlglot's tree is a *SQL* tree, not a relational algebra tree. Some
  things (scan-level metadata like "this table ref resolves to 412 parquet files") do
  not fit. **Mitigation:** we attach our own metadata to nodes via a side-table keyed
  by node id (`plan/annotations.py`), never by mutating sqlglot internals.

### 2.2 Package layout

```
Transformation_Engine_08172026/
├── pyproject.toml                # uv-managed, all deps declared here
├── uv.lock
├── README.md
├── PLAN.md                       # this file
├── .env.example                  # catalog/MinIO credentials
├── src/icetl/
│   ├── __init__.py
│   ├── conf.py                   # SparkConf-alike + env/.env loading
│   ├── errors.py                 # AnalysisException, ParseException, …
│   ├── sql/                      # ← the pyspark-compatible surface
│   │   ├── __init__.py           #   Session, DataFrame, Row, …
│   │   ├── session.py
│   │   ├── dataframe.py
│   │   ├── column.py
│   │   ├── functions.py          #   ~350 F.* functions
│   │   ├── window.py
│   │   ├── group.py              #   GroupedData, pivot, rollup, cube
│   │   ├── readwriter.py         #   DataFrameReader / DataFrameWriter / V2
│   │   ├── catalog.py            #   spark.catalog.*
│   │   ├── types.py              #   StructType, LongType, … + StructType.fromDDL
│   │   └── udf.py                #   udf / pandas_udf  (later phase)
│   ├── plan/
│   │   ├── builder.py            #   DataFrame op → sqlglot node
│   │   ├── annotations.py        #   node-id → scan metadata side-table
│   │   ├── schema_binding.py     #   Iceberg Schema → sqlglot MappingSchema
│   │   ├── optimizer.py          #   rule pipeline
│   │   └── pushdown.py           #   sqlglot predicate → pyiceberg BooleanExpression
│   ├── catalog/
│   │   ├── registry.py           #   named catalogs, REST/SQL/Glue/Hive
│   │   └── resolver.py           #   "cat.ns.table" → pyiceberg Table
│   ├── exec/
│   │   ├── engine.py             #   DuckDB connection lifecycle, S3/httpfs setup
│   │   ├── scan_planner.py       #   plan_files → read_parquet / arrow fallback
│   │   ├── deletes.py            #   positional/equality delete handling
│   │   ├── evolution.py          #   field-id ↔ name reconciliation
│   │   └── result.py             #   Arrow → pandas/polars/Row/show()
│   ├── io/
│   │   ├── writer.py             #   append / overwrite / dynamic partition overwrite
│   │   ├── rowlevel.py           #   MERGE INTO / UPDATE / DELETE
│   │   ├── ddl.py                #   create/alter/drop table, schema & spec evolution
│   │   └── maintenance.py        #   compaction, expire snapshots, rewrite manifests
│   └── compat/
│       └── divergence.md         #   documented Spark↔DuckDB behaviour differences
├── src/pyspark/                  # ← the shadow package: thin re-exports only
│   ├── __init__.py               #   SparkConf, SparkContext stubs
│   ├── sql/__init__.py           #   from icetl.sql import *   (Session, DataFrame, …)
│   ├── sql/functions.py          #   from icetl.sql.functions import *
│   ├── sql/types.py · window.py · column.py · utils.py
│   └── errors/__init__.py        #   AnalysisException, ParseException, …
├── tests/
│   ├── conftest.py               #   local SqlCatalog fixture (sqlite + local FS)
│   ├── fixtures/                 #   generated tables: plain, partitioned, MoR-deletes,
│   │                             #   schema-evolved, wide-200-col, nested-types
│   ├── unit/                     #   plan building, pushdown, type mapping
│   ├── conformance/              #   golden results vs real PySpark (optional extra)
│   └── integration/              #   marked; runs against your REST catalog
├── notebooks/
│   └── 00_quickstart.ipynb
└── scripts/
    └── smoke_catalog.py          #   verify REST + MinIO connectivity
```

### 2.3 Dependencies (`pyproject.toml`, Python 3.12, uv)

```toml
[project]
name = "icetl"
requires-python = ">=3.12,<3.13"
dependencies = [
  "pyiceberg[s3fs,pyarrow,sql-sqlite,glue,hive]>=0.11.1,<0.12",
  "duckdb>=1.5.5,<2",
  "sqlglot>=30.17.0,<31",
  "pyarrow>=25.0.1",
  "pandas>=2.2",
  "python-dotenv>=1.0",
  "rich>=13.0",           # .show() rendering + error formatting
]

[project.optional-dependencies]
polars   = ["polars>=1.0"]
notebook = ["ipykernel>=6.29", "jupyterlab>=4.2"]
dev      = ["pytest>=8", "pytest-cov", "ruff", "mypy", "hypothesis"]

[tool.uv]
package = true

[tool.hatch.build.targets.wheel]
packages = ["src/icetl", "src/pyspark"]   # ← ships the shadow package
```

> `sqlglot` is pinned deliberately — its optimizer internals move between minors.
> Everything installs into a project-local `.venv` created by `uv venv --python 3.12`.

**Consequence of shadowing `icetl` (decided 2026-08-17):** real PySpark cannot be
installed in the same environment — the import name collides. So differential testing
against a live Spark is off the table, and Spark fidelity is pinned down by
**hand-written golden files** instead (§5). To keep that honest, every conformance
rule in §3.5 ships with a golden case whose expected value is derived from Spark's
documented semantics and cited in a comment. If we ever want a live cross-check, it
runs in a *separate* throwaway venv, never this one.

### 2.4 A note on how we develop this

Your Iceberg REST catalog and MinIO are on your machine and are not reachable
from my sandbox. So:

- I build and unit/integration-test against a **local PyIceberg `SqlCatalog`**
  (sqlite metadata + local-filesystem warehouse) with generated fixture tables that
  reproduce the interesting cases — partitioned, merge-on-read deletes, renamed
  columns, 200+ columns, nested types.
- Code lands in `D:\workspace\Transformation_Engine_08172026`; **you** run
  `uv sync` and the `-m integration` tests against the real REST catalog on Windows.
- `scripts/smoke_catalog.py` is the first thing to run — it proves REST + MinIO
  connectivity before anything else is debugged.

---

## 3. The hard parts (design decisions worth arguing about now)

### 3.1 Schema binding is the keystone

sqlglot's optimizer is only as good as the schema you hand it. Without it, `qualify`
can't resolve `col("x")` to a table, and predicate pushdown silently does nothing.

```
PyIceberg Schema  →  { "nyc.yellow_tripdata": {"VendorID": "BIGINT", ...} }  →  MappingSchema
```

Built once per table per session, cached, invalidated on DDL. **Every optimizer rule
depends on this being right**, so it gets tested first and hardest.

### 3.2 Predicate pushdown, in two directions

After the optimizer runs, we split the `WHERE` clause into conjuncts:

| Conjunct | Goes to PyIceberg? | Stays in SQL? |
|---|---|---|
| `as_at_date = '2026-08-17'` (partition col, translatable) | ✅ prunes manifests + files | ✅ yes |
| `cusip IN (...)` (data col, translatable) | ✅ prunes via column stats | ✅ yes |
| `upper(name) LIKE 'A%'` (not translatable) | ❌ | ✅ yes |

**The filter is always kept in the generated SQL.** PyIceberg pruning is a
performance optimization, never a correctness mechanism — file-level pruning is
approximate (stats-based), so DuckDB must re-apply the predicate regardless. This
rule alone eliminates a whole class of wrong-results bugs.

Translation covers: `= != < <= > >=`, `IN`, `IS [NOT] NULL`, `AND/OR/NOT`,
`STARTS_WITH`, `BETWEEN`, and literal-castable comparisons. Anything else → skipped.

### 3.3 Merge-on-read deletes — the hybrid split *(DEFERRED — see Phase 12)*

> **Status: deferred, not dropped.** Decision 11 (2026-08-30) fixes the tables in
> scope as copy-on-write, so a scan is exactly its data files and none of the code
> below was built. Merge-on-read reading — delete files, positional deletes and
> equality deletes — is **planned work, owned by Phase 12**.
>
> The design in this section stands as written; it is what Phase 12 implements.
>
> **Groundwork already in place**, so picking this up is additive rather than a
> rewrite:
>
> | Piece | Where | State |
> |---|---|---|
> | The guard that refuses a MoR table today | `exec/scan_planner.py::_assert_copy_on_write` | Becomes the branch point |
> | A real MoR table to test against | `tests/fixtures/generator.py::build_mor` | Built by hand; PyIceberg cannot write delete files |
> | Proof the guard fires | `tests/fixture/test_pushdown.py::TestCopyOnWriteInvariant` | Becomes the correctness suite |
> | The `UNION ALL` the split needs | `exec/source_sql.py::build_source` | Already there, serving the rename path |
> | `ScanPlan` carrying an Arrow half | removed in decision 11 | Restore `delete_table` / `delete_files` |
>
> Until then the guard is what keeps this honest: `read_parquet` cannot see a delete
> file, so an unhandled MoR table would return the deleted rows and report success.
> Refusing the scan is the only acceptable interim behaviour.

`plan_files()` returns `FileScanTask`s, each possibly carrying `delete_files`.

```python
clean, dirty = partition_tasks(scan.plan_files())

if not dirty:                       # ~99% of real tables, the fast path
    sql_source = read_parquet([t.file.file_path for t in clean])
else:
    arrow = table.scan(...).to_arrow()  # PyIceberg applies deletes correctly
    conn.register("_icetl_dirty_0", arrow)
    sql_source = read_parquet(clean_paths) UNION ALL BY NAME _icetl_dirty_0
```

Correct by delegation, fast when it can be. A later phase can replace the dirty
branch with a DuckDB `ANTI JOIN` on `(file_name, file_row_number)` for positional
deletes — but only once the fast path is proven and benchmarked.

### 3.4 Schema evolution vs `read_parquet` — the sharp edge

Iceberg tracks columns by **field-id**; parquet files on disk carry whatever **name**
they had when written. `read_parquet(union_by_name=true)` matches by *name*, so:

| Evolution | Naive `read_parquet` | Our handling |
|---|---|---|
| Column added | ✅ works (NULLs) | fast path |
| Column dropped | ✅ works | fast path |
| Column **renamed** | ❌ **silently wrong** (old files → NULLs) | detect + per-file alias projection, else Arrow fallback |
| Type promoted (int→long) | ⚠️ may error | explicit CAST in projection |
| Column reordered | ✅ (by name) | fast path |

**Detection:** compare each data file's parquet `field_id` metadata against the
current schema. If names and ids agree for every selected column → fast path.
Otherwise generate `SELECT old_name AS new_name, ...` per file group, or fall back
to Arrow. This is subtle and gets its own test fixture and its own phase.

### 3.5 Spark ≠ DuckDB — the conformance layer

These are the ones that bite. Each becomes a translation rule *and* a test.

| Behaviour | Spark | DuckDB | Our rule |
|---|---|---|---|
| `ORDER BY x ASC` nulls | nulls **first** | nulls **last** | always emit explicit `NULLS FIRST/LAST` |
| `ORDER BY x DESC` nulls | nulls **last** | nulls first | idem |
| `CAST('abc' AS INT)` | `NULL` (non-ANSI) | **error** | emit `TRY_CAST` by default; `ansi_mode=true` opts into strict |
| Integer overflow | wraps | error | documented divergence (we do **not** emulate wrapping) |
| `a <=> b` | null-safe eq | — | → `IS NOT DISTINCT FROM` |
| `1/0` | `NULL` | error | → guarded division |
| Decimal promotion | Spark's precision rules | DuckDB's | explicit cast per Spark's rules in arithmetic |
| `explode(arr)` | row per element, drops empty | — | → `UNNEST` |
| `explode_outer(arr)` | keeps empty as NULL | — | → `LEFT JOIN UNNEST` |
| Agg output col names | `sum(total_amount)` | `sum(total_amount)` | assert equality in tests |
| Window default frame | `RANGE UNBOUNDED PRECEDING → CURRENT ROW` | same | ✅ no action |
| Identifier case | insensitive resolve, preserved | insensitive | ✅ no action, but tested |
| `df.join(o, "k")` | one `k` column | — | rewrite to `USING` + explicit projection |
| Empty `count()` on no rows | `0` | `0` | ✅ |

Every row here goes into `src/icetl/compat/divergence.md` with a runnable example.
Where we can't match Spark, we say so loudly rather than quietly differ.

### 3.6 Wide-table strategy (your 200-column, 250M-row case)

Your securities table is the stress case, so it drives design rather than being
retrofitted:

- **Projection pushdown is mandatory, not opportunistic.** Column set is computed
  from the final tree and passed to `read_parquet` explicitly — never `SELECT *`.
- `selected_fields` also goes to `table.scan()` so PyIceberg fetches fewer stats.
- `as_at_date` filter → partition pruning at the manifest level (never opens files).
- Since queries never filter `security_group`, that partition level gives no pruning —
  so we lean on **column-stats file pruning** for `cusip`/`isin`, which requires the
  table to be sorted/clustered. `io/maintenance.py` (compaction with sort order) is
  therefore in scope, not a nice-to-have.

---

## 4. Phases

Each phase ends with green tests and something you can run. Phases 0–3 are the
critical path; after that, breadth phases can be reordered or parallelised.

**Phases 12–15 are deferred**, not optional-forever. 12–13 hold the merge-on-read
work decision 11 took out of Phase 2; 14 holds the decimal promotion and 15 the
SQL-surface function resolution that decisions 14 and 16 took out of Phase 3. Nothing
in scope needs them today. Where a cheap guard exists it is in place — the scan planner
refuses a merge-on-read table rather than guessing — and where one does not (14 and 15,
where detecting the problem costs as much as fixing it), the divergence is written down
instead.

Phase 15 is the one with a **live wrong-answer path**: `weekday` and `dayofweek` are off
by one through `Session.sql()`, silently. Prefer `F.*` for date-part work until it
lands.

### Phase 0 — Scaffolding *(small)*
- `uv` project, `.venv` on Python 3.12, `pyproject.toml` with all deps + `ipykernel`.
- Package skeleton, `ruff` + `mypy` + `pytest` configured.
- `conf.py`: catalog config from `.env` / `Session.builder.config(...)`.
- `scripts/smoke_catalog.py`: connect to REST catalog, list `nyc`, describe
  `yellow_tripdata`, read 10 rows via DuckDB. **First proof of life.**
- Local `SqlCatalog` test fixture + fixture-table generator.

**Done when:** `uv run python scripts/smoke_catalog.py` prints 10 rows from your table.

### Phase 1 — End-to-end thin slice *(the important one)*
- `Session.builder…getOrCreate()`, `Session.table()`, `Session.sql()`.
- `DataFrame`: `select`, `filter`/`where`, `limit`, `withColumn`, `withColumnRenamed`,
  `drop`, `alias`, `printSchema`, `schema`, `columns`, `explain`.
- `Column` basics + `functions.col/lit/expr` and comparison/arithmetic operators.
- Actions: `show`, `collect`, `count`, `toPandas`, `first`, `take`, `toArrow`.
- Minimal scan planner: `plan_files` → `read_parquet` list (clean tasks only).
- DuckDB engine: httpfs/S3 config for MinIO, threads, memory limit, temp dir.
- `explain()` prints the generated DuckDB SQL + the file count + pushed filters.

**Done when:** the example at the top of this document runs (read side), and
`spark.sql("SELECT VendorID, count(*) FROM nyc.yellow_tripdata GROUP BY 1")`
returns the same thing as the DataFrame equivalent.

### Phase 2 — Plan IR, optimizer, pushdown *(the foundation)*
- Schema binding from PyIceberg → `MappingSchema`, cached.
- Optimizer rule pipeline; `explain(mode="extended")` shows before/after trees.
- `pushdown.py`: sqlglot predicate → PyIceberg `BooleanExpression` (+ the "always
  keep the filter in SQL" invariant, tested).
- Projection pushdown → explicit column lists.
- Annotations side-table so scan metadata rides along with nodes.
- Delete-file hybrid split (§3.3) with a merge-on-read fixture.

**Done when:** pruning is observable — `explain()` on the wide fixture reports
"scanned 3 of 4096 files, 12 of 214 columns".

### Phase 3 — Types, expressions, function library *(the breadth grind)*
- `types.py`: full Spark type hierarchy, `StructType.fromDDL`, `fromJson`,
  Iceberg↔Spark↔DuckDB↔Arrow mapping matrix (tested exhaustively both directions).
- `functions.py`: the ~350 `F.*` functions, grouped and translated —
  string, math, date/time, conditional, hashing, aggregate, collection, casting.
- `Column`: `cast`, `alias`, `when/otherwise`, `isin`, `between`, `like`, `rlike`,
  `substr`, `getItem`, `getField`, `asc_nulls_first`, `eqNullSafe`, `over`, …
- The conformance rules from §3.5 implemented as translation rules.
- `compat/divergence.md` written as we go, not after.

**Done when:** the conformance suite passes for every implemented function.

### Phase 4 — Relational breadth
- Joins: inner, left/right/full outer, left semi, left anti, cross; `on` as
  string / list / Column; self-joins with alias disambiguation.
- Aggregations: `groupBy().agg()`, all agg functions, `distinct`, `dropDuplicates`,
  `rollup`, `cube`, `pivot`, `agg` on the DataFrame directly.
- Set ops: `union`, `unionByName` (incl. `allowMissingColumns`), `intersect`,
  `exceptAll`, `subtract`.
- `orderBy`/`sort`, `distinct`, `sample`, `randomSplit`, `repartition` (no-op),
  `cache`/`persist` (→ DuckDB temp table), `crossJoin`, `na.fill/drop/replace`,
  `stat.*` (`approxQuantile`, `corr`, `cov`, `crosstab`, `freqItems`).
- SQL side: CTEs, subqueries, `EXISTS`, set operations, temp views
  (`createOrReplaceTempView`).

### Phase 5 — Window functions
- `Window.partitionBy().orderBy().rowsBetween()/rangeBetween()`, unbounded markers.
- `row_number`, `rank`, `dense_rank`, `percent_rank`, `ntile`, `cume_dist`,
  `lag`, `lead`, `first`, `last`, `nth_value`, running/moving aggregates.
- Frame-semantics conformance tests (this is where Spark/DuckDB drift is likeliest).

### Phase 6 — Complex types
- `StructType`/`ArrayType`/`MapType` columns end-to-end (read, project, write).
- `explode`, `explode_outer`, `posexplode`, `posexplode_outer`, `inline`.
- `F.struct`, `F.array`, `F.create_map`, `F.get_json_object`, `from_json`, `to_json`.
- `array_*` (~40 fns) and `map_*` families; `transform`, `filter`, `aggregate`,
  `exists`, `forall`, `zip_with` higher-order functions (→ DuckDB lambdas).
- Nested field access `col("a.b.c")`, `withField`, `dropFields`.

### Phase 7 — Write path v1
- `df.write.format("iceberg").mode(...)`: `append`, `overwrite`, `error`, `ignore`.
- `saveAsTable` (create-if-missing with inferred Iceberg schema) and `insertInto`.
- **Dynamic partition overwrite** (`partitionOverwriteMode=dynamic`) — overwrite only
  the partitions present in the incoming data.
- `partitionedBy` / `sortBy` / `tableProperty` on the writer.
- Streaming Arrow batches from DuckDB → PyIceberg, so writes don't materialise
  the whole result in memory.
- Transactional semantics: one Iceberg snapshot per write, retry on commit conflict.

### Phase 8 — Row-level operations
- `MERGE INTO` — full Spark SQL grammar: `WHEN MATCHED [AND cond] THEN UPDATE SET
  … / DELETE`, `WHEN NOT MATCHED THEN INSERT`, `WHEN NOT MATCHED BY SOURCE …`.
- `UPDATE … SET … WHERE …` and `DELETE FROM … WHERE …`.
- Strategy: copy-on-write. DuckDB computes the result set for affected files;
  PyIceberg `overwrite(overwrite_filter=…)` commits the rewritten files. Use
  PyIceberg's native `upsert()` / `delete()` where it matches the semantics exactly,
  otherwise do the rewrite ourselves.
- File-scope minimisation: only touch files whose stats can match the condition.
- Concurrency: optimistic commit with conflict detection and bounded retry.
- Heavy correctness testing — this is the riskiest phase in the plan.

### Phase 9 — Schema, DDL, snapshots
- `spark.catalog.*`: `listDatabases`, `listTables`, `tableExists`, `createTable`,
  `dropTable`, `currentDatabase`, `setCurrentDatabase`.
- SQL DDL: `CREATE [OR REPLACE] TABLE … USING iceberg PARTITIONED BY (…)`,
  `ALTER TABLE … ADD/DROP/RENAME/ALTER COLUMN`, `SET TBLPROPERTIES`, `DROP TABLE`.
- Partition-spec evolution and sort-order changes.
- Schema evolution on write (`mergeSchema`), plus the field-id reconciliation of §3.4.
- Time travel: `spark.read.option("snapshot-id"|"as-of-timestamp")`, `VERSION AS OF`,
  `TIMESTAMP AS OF`; metadata tables (`.snapshots`, `.files`, `.history`,
  `.manifests`, `.partitions`).

### Phase 10 — Performance & scale *(DONE 2026-09-02)*
- ~~Memory: `memory_limit` sized from available RAM~~ — **declined, decision 17.** Spill
  is configured and now measured (FINDINGS §4.8): the same query fails without a temp
  directory and succeeds with one.
- ~~Parallelism: threads = physical cores~~ — **declined, decision 17.** Measured
  indistinguishable from DuckDB's default; oversubscribing is the only thing that hurts.
- Wide-table benchmark harness — `scripts/benchmark.py`, results in `BENCHMARKS.md`.
- Result streaming — `toArrowBatches()` / `toLocalIterator()`, with a guard against the
  silent truncation DuckDB performs when a stream's cursor runs another query.
- Schema caching, keyed on Iceberg's `schema_id` rather than a TTL (17 ms → 0.4 ms). The
  **query cache is not built**; STATUS.md says why.
- Benchmarks committed as a tracked file — `BENCHMARKS.md`.

Also closed here: FINDINGS §3.3, §3.4 and §3.5, carried in from Phases 2 and 9. Found
here: §1.12 and §1.13, two silent wrong answers that predated the phase.

### Phase 11 — Extras *(DONE 2026-09-02)*
- Python UDFs + vectorised UDFs through DuckDB's native and Arrow registration, on
  both surfaces. The null handling took measuring: FINDINGS §2.9.
- `io/maintenance.py` — `compact()`, `expireSnapshots()`, `removeOrphanFiles()`.
  PyIceberg 0.11 provides only the second (§4.9), so the rest are built.
  ~~`rewrite_manifests`~~ — **refused**: a hand-written manifest fails silently by
  hiding data files.
- Non-Iceberg readers — `session.read.parquet/csv/json`. Object-store paths refused,
  because the schema binds on a connection without S3 credentials.
- `notebooks/00_quickstart.ipynb` and `GUIDE.md`. Every notebook cell is run, which
  is what found §2.10.

### Phase 12 — Merge-on-read reads *(deferred from Phase 2 by decision 11)*

Everything needed to read a table another engine wrote with merge-on-read. Deferred
because no table in scope has delete files; **not** dropped, because the moment one
does — a Flink or Trino writer, a Spark job with
`write.delete.mode=merge-on-read`, an upstream default changing — the guard in
`exec/scan_planner.py` starts refusing queries that used to work, and this is what
un-refuses them.

- **The hybrid split of §3.3.** Partition `plan_files()` into clean and dirty tasks;
  clean go to `read_parquet`, dirty go through PyIceberg's
  `ArrowScan(...).to_table(tasks)` — which takes an explicit task list, so only the
  affected files are re-read — and `UNION ALL` the two halves. Correct by delegation,
  and the fast path stays fast.
- **Positional deletes** (`file_path`, `pos`), the common case.
- **Equality deletes**, which PyIceberg applies through the same call, so they arrive
  with the same code — but need their own fixture, since `build_mor` writes positional
  deletes only.
- Restore `ScanPlan.delete_table` / `delete_files`, the `UNION ALL` branch in
  `build_source`, and the `merge-on-read: N file(s)` line in `explain()`.
- Turn `TestCopyOnWriteInvariant` inside out: it currently asserts the scan is
  refused; it should assert the deleted rows do not come back.
- **Only then**, and only if benchmarks justify it, replace the dirty branch with a
  DuckDB `ANTI JOIN` on `(file_name, file_row_number)`. Correctness first (P6).

**Done when:** `fx.mor` reads back 6 rows through icetl where its files hold 8, and
agrees with PyIceberg on every fixture.

### Phase 13 — Row-level writes in merge-on-read mode *(deferred, further out)*

Phase 8 writes copy-on-write, which is what PyIceberg does natively. Emitting delete
files instead — so `DELETE`/`UPDATE`/`MERGE` do not rewrite whole data files — is a
separate and larger piece of work, and PyIceberg has no API for it today (see
carry-over note 5: `build_mor` had to hand-write a delete manifest, a manifest list
and a snapshot). Needs Phase 12 landed first, so we can read what we write.

### Phase 14 — Decimal promotion *(deferred from Phase 3 by decision 14)*

Spark derives the result type of decimal arithmetic from its operand types by a fixed
rule; DuckDB has its own. Both were measured — the table is in
`compat/divergence.md`, "Decimal promotion":

| Operation on `DECIMAL(10,2)` | Spark | DuckDB | |
|---|---|---|---|
| `a + b` | `DECIMAL(11,2)` | `DECIMAL(11,2)` | already agrees |
| `a * b` | `DECIMAL(21,4)` | `DECIMAL(18,4)` | precision differs |
| `a / b` | `DECIMAL(16,6)` | **`DOUBLE`** | *type* differs |

Division is the one that matters. Falling to `DOUBLE` loses exactness, which is the
entire reason a monetary column is a decimal in the first place.

**The work:**

- Add sqlglot's `annotate_types` to the optimizer pipeline in `plan/optimizer.py`.
  It is deliberately omitted today; the schema binding of §3.1 is already in place, so
  it has what it needs.
- Emit an explicit `CAST` around decimal arithmetic per Spark's formulas
  (`DecimalPrecision` in Spark's analyzer), clamping precision to 38 as Spark does.
- Extend `sql/conformance.py`, so both surfaces get it from one pass — a rule in
  `Column.__truediv__` would cover the DataFrame API and miss `spark.sql`.
- Decide what to do when Spark's rule would exceed precision 38. Spark clamps and can
  lose scale; that behaviour needs matching, not improving on.

**Done when:** the three rows above agree with Spark, and a decimal column divided by
a decimal column stays a decimal.

**Why it was not done in Phase 3.** The rule needs *operand* types, and the type of a
sub-expression is only known after binding. Casting without them would produce a
confidently wrong precision — worse than the current divergence, which is at least
honest and documented. There is also no cheap guard here, unlike the merge-on-read
case: detecting "this query would have promoted differently" requires the same type
information as fixing it, so a warning would cost as much as the fix.

**Interim behaviour:** decimal `+` and `-` already match Spark. `*` keeps the right
scale with a smaller precision, which overflows only at extreme magnitudes. `/`
returns `DOUBLE` — accurate to ~15 significant digits, so wrong only where exact
decimal semantics were the point. That is the case to watch.

### Phase 15 — Function resolution on the SQL surface *(deferred from Phase 3 by decision 16)*

P1 says the two surfaces converge on one tree. That holds for *relational* structure and
for the conformance rules in `sql/conformance.py`, which are a tree pass both surfaces
cross. It does **not** hold for functions whose reference behaviour is produced by
**composition inside `sql/functions.py`**: `rint` picking the even neighbour on a tie,
`weekday` shifting Monday to 0, `overlay` built out of `substring`. Those compositions
exist only on the `F.*` path. A bare name in `Session.sql()` goes straight to DuckDB.

Measured over a 17-case sample of the 273-name surface:

| Class | Count | Example | Danger |
|---|---|---|---|
| **Silently different** | 2 | `weekday(Monday)` → `1` in SQL, `0` via `F.*` | **wrong answers, nothing raises** |
| Missing on the SQL surface | 9 | `rint`, `log1p`, `expm1`, `find_in_set`, `octet_length`, `overlay`, `width_bucket`, `regexp_substr` | raises `AnalysisException` — loud, safe |
| Agree but both wrong | 1 | `size(NULL)` → `NULL`, reference says `-1` | a plain conformance bug, unrelated to the split |

**The work:**

- Resolve function names in the SQL path through the same `F.*` table the DataFrame API
  uses, so `Session.sql("SELECT rint(2.5)")` reaches `functions.rint` rather than
  DuckDB's `rint`. The hook belongs where the parsed tree is first walked, before the
  optimizer, so pushdown sees what executes.
- Leave names that are *already* correct in DuckDB alone — most of the 273 are
  pass-throughs, and routing them through Python would cost without buying anything.
- Fix `size(NULL)` to answer `-1`. `array_size`'s docstring already states the two must
  differ; `size` is implemented as `sg.ArraySize`, which makes them aliases.
- **The durable part: a test over the whole surface**, asserting that every name in
  `F.__all__` gives the same value through both surfaces or is absent from both. The
  sample above found the gap; only an exhaustive check stops it reopening.

**Done when:** no name in `F.__all__` answers differently through `Session.sql()` than
through `F.*`.

**Why it was not done in Phase 3.** Phase 3's tests exercise the `F.*` surface, which is
where the compositions live, so nothing in the suite could see the gap — 821 tests pass
with it present. It was found by running the notebook in section 8 of
`notebooks/01_read_real_table.ipynb`, not by the suite.

**Interim exposure:** `weekday` and `dayofweek` are the ones to watch. Both surfaces
answer, neither raises, and the SQL surface is off by one because DuckDB numbers the week
differently. An off-by-one weekday is invisible in a result set. The other nine raise, so
they cannot produce a wrong answer. Prefer `F.*` over `Session.sql()` for date-part work
until this lands.

---

## 5. Testing strategy

| Layer | What | How |
|---|---|---|
| **Unit** | plan building, pushdown translation, type mapping, SQL generation | pytest, no I/O, golden SQL strings |
| **Fixture** | real Iceberg semantics | local `SqlCatalog` (sqlite) + local FS warehouse; fixtures for partitioned / MoR-deletes / renamed-columns / 200-col-wide / nested-types |
| **Conformance** | *"does it match Spark?"* | Value-level cases asserting what Spark returns, each citing the behaviour it encodes. **No JVM at any stage** (decision 13), so these are asserted rather than machine-verified against a real Spark — the limit is recorded in `compat/divergence.md`. Grows with every function implemented in Phase 3+. |
| **Property** | invariants, not Spark-equivalence | `hypothesis` — e.g. "pushdown never changes the result set", "optimizer on/off produce identical rows", round-trip write→read fidelity |
| **Integration** | your REST catalog + MinIO | `pytest -m integration`, skipped by default, run by you on Windows |
| **Benchmark** | wide-table scans, joins, writes | tracked results file; run manually |

---

## 6. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| **Schema evolution silently returns wrong data** (§3.4) | 🔴 wrong results | field-id verification before every fast-path scan; Arrow fallback; dedicated fixture |
| **MERGE INTO correctness** (Phase 8) | 🔴 data loss | copy-on-write only at first; differential tests vs Spark; conflict retry |
| **A merge-on-read table appears upstream** while Phase 12 is still deferred | 🟠 queries start failing | the scan planner refuses such a table rather than returning deleted rows (decision 11); the error names Phase 12, and `fx.mor` proves the guard fires |
| **sqlglot optimizer edge cases** on complex nested SQL | 🟠 wrong results / crash | keep the un-optimized tree as fallback; `icetl.conf` switch to disable optimizer; golden tests |
| Spark semantics gaps found late | 🟠 rework | conformance suite from Phase 3, not at the end |
| `read_parquet` over many small files is slow | 🟡 perf | compaction utility (Phase 11); file-count warnings in `explain()` |
| Very large results OOM the process | 🟡 | streaming Arrow + spill from Phase 10; never `fetchall()` internally |
| PyIceberg 0.11 API churn | 🟡 | pinned `<0.12`; catalog/write access isolated behind `catalog/` and `io/` |

---

## 7. Decisions locked (2026-08-17)

| # | Decision |
|---|---|
| 1 | ~~**Drop-in PySpark-compatible API**~~ — **superseded by decision 15**. The API is icetl's own; Spark remains the semantic reference only |
| 2 | **sqlglot AST is the single IR** — DataFrame and SQL converge on one tree |
| 3 | **PyIceberg file list → DuckDB `read_parquet`**, Arrow fallback only where delete files exist |
| 4 | **Phase 1 = end-to-end thin slice** against `nyc.yellow_tripdata` |
| 5 | **Write path covers both** bulk (append / overwrite / dynamic partition overwrite) **and** row-level (MERGE / UPDATE / DELETE) |
| 6 | **Priority features:** window functions, schema/DDL + evolution, complex types + explode. Python UDFs deferred to Phase 11 |
| 7 | **Python 3.12**, uv, all deps in `pyproject.toml`, `ipykernel` included |
| 8 | ~~**Shadow the `pyspark` import name**~~ — **superseded by decision 15**; the shadow package is deleted |
| 9 | **Golden-file conformance**, no live PySpark in the dev environment |
| 10 | **MERGE stays at Phase 8**, on top of a proven optimizer and write path |
| 14 | **Decimal promotion deferred to Phase 14** (2026-08-30). Spark's decimal arithmetic result types are not reproduced: `+`/`-` already agree, `*` differs in precision, and `/` returns `DOUBLE` where Spark returns `DECIMAL`. Deferred rather than guessed, because the rule needs operand types and a wrong precision is worse than a documented divergence. No guard is possible without doing the work — detection needs the same type information as the fix. |
| 13 | **No PySpark, ever — not even to generate test fixtures** (2026-08-30). Supersedes carry-over note 9, which wanted a golden corpus generated from real Spark in a throwaway venv. Conformance cases are instead written from Spark's published behaviour, each carrying a citation. **What this costs:** an edge case where Spark's real behaviour differs from its documentation will produce a confidently green test. Undocumented corners — NULL propagation, empty input, overflow, decimal promotion — are the exposure. `compat/divergence.md` marks conformance as *asserted, not machine-verified* so the limit is visible rather than implied. |
| 12 | **Build the function library; use SQLFrame as a reference, not a dependency** (2026-08-30). Settles decision 8. Neither SQLFrame nor `duckdb.experimental.spark` implements Spark conformance — both return `inf` for `1/0`, both raise on `CAST('abc' AS INT)`, neither emits explicit NULLS FIRST/LAST — so §3.5 is ours either way, and neither can read Iceberg through our planner. SQLFrame also pins `sqlglot<30.13` against our 30.17. Its 478 function→sqlglot-node mappings are read as a lookup table (MIT); nothing is imported. |
| 15 | **The PySpark compatibility surface is dropped** (2026-08-30). Supersedes decisions 1 and 8. icetl is an Iceberg + DuckDB library and no longer presents itself as Spark: `SparkSession` → `Session`, `PySpark*Error` → `Engine*Error`, the shadow `src/pyspark` package is deleted, `spark.*` config keys are replaced by `icetl.*` twins, and `F.spark_partition_id` is removed (273 names). **What stays, deliberately:** Spark 3.5 remains the *semantic* reference (P5) under the neutral name *the reference*, because having a written spec beats deciding each edge case ad hoc; and sqlglot's `"spark"` dialect stays, bound to `SQL_DIALECT`, because it names a SQL *grammar* — changing it would change the language accepted, not the branding. **What this costs:** any script doing `from pyspark.sql import SparkSession`, catching `PySparkTypeError`, or passing `spark.*` keys to `.config()` breaks. That is the intent, and no test covered the shadow package, so nothing in the suite regressed. |
| 16 | **SQL-surface function resolution deferred to Phase 15** (2026-08-30). Functions whose reference behaviour comes from composition in `sql/functions.py` are reachable only through `F.*`; a bare name in `Session.sql()` goes to DuckDB. Of a 17-case sample: 2 answer **silently differently** (`weekday`, `dayofweek` — off by one), 9 raise because DuckDB has no such function, 1 (`size(NULL)`) is wrong on both. **Deferred, not dropped** — the fix is a resolution hook plus an exhaustive both-surfaces test, and it is worth doing as its own piece rather than inside Phase 4. **No guard**, unlike decision 11: detecting the divergence means evaluating both surfaces, which is the test, which is the fix. `compat/divergence.md` is the warning instead. **What this costs:** P1 is stated over both surfaces and does not currently hold for these names. |
| 17 | **DuckDB sizes its own threads and memory** (2026-09-02). Supersedes Phase 10's first two bullets. `--compare-threads 2 4 8 16` over the benchmark suite: 2, 4 and 8 are indistinguishable, 16 is worse across every case. Physical cores (4) and DuckDB's default (8) differ by less than the run-to-run spread, so pinning threads would buy a configuration surface and a portable-core-count dependency for no measurable gain. Memory likewise: DuckDB already takes ~80% of total RAM, and the thing that turns an OOM into a completed query is the **spill directory**, which is configured and measured (§4.8). **What this costs:** on a machine where another process holds most of the RAM, 80%-of-total is an over-commitment icetl will not notice — an `Out of Memory` from DuckDB, not a wrong answer, and one line to fix. Documented rather than guarded; revisit with a measurement, which `scripts/benchmark.py` exists to take. |
| 11 | **Copy-on-write for now; merge-on-read deferred** (2026-08-30). Every writer of the tables in scope rewrites data files, so no delete or position files exist and §3.3's hybrid split is not built. **Deferred, not dropped** — MoR *reading* is Phase 12, MoR *writing* is Phase 13. The assumption is asserted at scan time rather than trusted, because `read_parquet` cannot see a delete file and would return deleted rows reporting success. Phase 8 writes COW, which is what PyIceberg does natively. |

### Still open

1. ~~**Catalog host**~~ — settled: `http://localhost:8182`, MinIO on `:9100`. In
   `.env` (gitignored); `.env.example` documents every key.
2. ~~**Package name**~~ — settled by decision 15: `icetl` is the public surface,
   imported as `from icetl.sql import Session, functions as F`.
3. ~~**Golden conformance files**~~ — settled by decision 13: **no PySpark**, at any
   stage. Conformance cases are written from Spark's published behaviour with a
   citation each. See decision 13 for what that does and does not buy.

---

## 8. Immediate next steps once approved

1. Phase 0 scaffolding committed to `D:\workspace\Transformation_Engine_08172026`.
2. You run: `uv venv --python 3.12 && uv sync --all-extras`.
3. You run `uv run python scripts/smoke_catalog.py` and paste the output.
4. Phase 1 begins.

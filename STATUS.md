# Status

Where the build has got to. Update this at the end of each phase.
See [PLAN.md](PLAN.md) for the design and the full phase list.

---

## Resuming — read this first

**Paused:** 2026-08-30, mid Phase 3.

### Where things stand

| | |
|---|---|
| Phases 0, 1, 2 | **done**, green |
| Phase 3 | **in progress** — conformance layer, type parsing and 169 `F.*` names done; the `F.*` tail remains |
| Phases 12, 13, 14 | **deferred by decision**, each with its own section in PLAN.md §4 |
| Tests | 711 local, 11 integration |
| Gate | `ruff check` · `ruff format --check` · `mypy` all clean |

### Everything is uncommitted

⚠️ **The repository has no commits.** All 85 files are staged in the index and
nothing has been committed, so the work exists only in the working tree and index.
That has been true since Phase 0 and is now a lot of work to be holding that way.

Nothing here will commit without being asked — say the word and it is one command.

### Picking it back up

Prerequisites: the REST catalog on `localhost:8182` and MinIO on `localhost:9100`
must be running for the integration tests. `.env` holds the settings and is
gitignored; `.env.example` documents every key. Local tests need neither.

```bash
uv sync --extra dev            # if the venv is cold
uv run pytest -q               # 711 expected
uv run pytest -m integration -q # 11 expected; needs the catalog up
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run python scripts/smoke_catalog.py -v   # proof of life against the real catalog
```

### The next thing to do

Finish Phase 3's `F.*` tail — see "Remaining for Phase 3" below. The machinery is in
place (`_col` for argument coercion, `_fn` for the untyped fallback, typed sqlglot
nodes preferred), so the remaining work is mechanical **with one hard rule**:

> Every function needs a test that asserts on a **value**, not on generated SQL.
> Around a dozen of the first 169 produced perfectly plausible SQL and the wrong
> answer. `tests/fixture/test_functions.py` opens with why.

Much of what is left is complex-type and JSON work that Phase 6 owns anyway, so it is
worth checking PLAN.md §4 before grinding further down the function list.

---

## Phase 0 — Scaffolding · **DONE** (2026-08-18)

Project set up, catalog + DuckDB plumbing working, smoke test passing end to end.

**Built:** `conf.py` (config layering + Spark keys) · `errors.py` (PySpark exception
hierarchy) · `paths.py` (Iceberg→DuckDB path translation) · `catalog/` (registry +
resolver) · `exec/engine.py` (DuckDB) · `diagnostics/smoke.py` + `scripts/smoke_catalog.py` ·
local `SqlCatalog` test fixture with 5 generated tables.

---

## Phase 1 — End-to-end thin slice · **DONE**

`SparkSession` / `spark.table()` / `spark.sql()`, `DataFrame` transformations and
actions, `Column` + `F.col/lit/expr`, eager analysis against zero-row Arrow views,
minimal scan planner, `explain()`.

**Built:** `types.py` (Spark type hierarchy + `Row`) · `sql/session.py` ·
`sql/dataframe.py` · `sql/column.py` · `sql/functions.py` · `plan/builder.py` ·
`plan/analysis.py` · `exec/scan_planner.py` · `exec/result.py` · the shadow
`src/pyspark` package.

---

## Phase 2 — Plan IR, optimizer, pushdown · **DONE** (2026-08-30)

Green on: `uv run pytest` (481 passed) · `uv run pytest -m integration` (11 passed,
**against the real REST catalog + MinIO**) · `ruff check` · `ruff format --check` · `mypy`.

Pruning is now observable, which was the phase's bar:

```
== Scans ==
  test.fx.partitioned: 1 of 3 file(s), 0.0 MB
    columns: 3 of 3: id, as_at_date, amount
    pushed filters: as_at_date = '2026-08-16'
```

**Built**

| Module | What it does |
|---|---|
| `plan/schema.py` | Iceberg `Schema` → sqlglot `MappingSchema`, cached per table per schema-id (§3.1) |
| `plan/optimizer.py` | The rule pipeline, applied one rule at a time so a failure costs only that rule |
| `plan/pushdown.py` | sqlglot predicate → PyIceberg `BooleanExpression`; per-source column extraction |
| `plan/annotations.py` | Scan metadata keyed by plan node, merged to one request per table |
| `plan/describe.py` | Renders a PyIceberg predicate as readable SQL for `explain()` |
| `exec/scan_planner.py` | Rewritten: `row_filter` + `selected_fields`, field-id file grouping (§3.4), pruning counters, the copy-on-write guard |
| `exec/source_sql.py` | Builds the relation that replaces a table reference: projection list, per-group aliasing |

**The compile pipeline, end to end**

```
plan → bind schema → optimize → extract scan requests → plan scans → substitute → SQL
```

Every step degrades to Phase 1 behaviour rather than failing: a plan the optimizer
cannot bind still runs, reading more than it needed to.

**Two guarantees worth knowing about**

- **Copy-on-write is asserted, not assumed** (decision 11). `read_parquet` cannot see
  a delete file, so a table that turns out to carry one is refused rather than read
  wrongly. `fx.mor` exists to prove the guard fires.
- **The pushed filter is always kept in the SQL.** PyIceberg's pruning is stats-based
  and therefore approximate, so DuckDB re-applying the predicate is what makes the
  answer right. Nothing in `pushdown.py` removes a filter.
- **The optimizer may not rename output columns.** `qualify` turns `sum(amount)` into
  `_col_0`, which for Spark is a wrong answer. The optimized tree is adopted only if
  its projections can be re-aliased to the names analysis already computed; otherwise
  the original plan runs.

**Carry-over notes closed**

| # | Note | Outcome |
|---|---|---|
| 2 | `scan().to_arrow()` double-counts rows | Using `ArrowScan(...).to_table(tasks)` with an explicit task list |
| 3 | Renamed columns detectable from `table.schemas()` | That is the detection; footers are opened only when it fires |
| 4 | The §3.4 bug must be flipped | `fx.renamed` reads correctly through icetl; the raw-`read_parquet` characterisation test is kept, retitled, as the reason why |
| 5 | PyIceberg cannot write delete files | `build_mor` writes the delete parquet, a `DELETES` manifest, a manifest list and a snapshot by hand. Now serves decision 11's guard rather than the hybrid split |
| 7 | duckdb 1.5 Arrow API | Already handled in Phase 1 |

**Found and fixed along the way**

- **DuckDB was re-typing partition columns from the directory name.** `read_parquet`
  auto-enables `hive_partitioning` when it sees `key=value` directories, which an
  Iceberg warehouse is full of. `fx.partitioned`'s `string` column was coming back as
  `DATE`, sourced from the path rather than the data. Now `hive_partitioning => false`
  always, with a regression test. This was present in Phase 1 and unnoticed.
- Phase 1 had left `ruff` and `mypy` failing (78 unannotated test functions, several
  `Expr`-vs-`Expression` narrowings). Cleared; `py.typed` markers added so `mypy`
  sees `icetl` as typed from the tests too.

### Validated against the real catalog (2026-08-30)

`nyc.yellow_tripdata` (19 cols, 62 files, 41.2M rows) and `nyc.yellow_tripdata_wide` /
`nyc.wide_smoke` (219 cols), partitioned by `month(tpep_pickup_datetime)`, over a REST
catalog on `localhost:8182` with MinIO on `localhost:9100`.

PLAN.md's "done when", on real data:

```
  rest.nyc.yellow_tripdata: 3 of 62 file(s), 1 of 19 columns
    pushed filters: (tpep_pickup_datetime >= '2024-06-01T00:00:00'
                     AND tpep_pickup_datetime < '2024-07-01T00:00:00')
```

| Check | Result |
|---|---|
| Pruned count vs PyIceberg's own | 3,539,170 == 3,539,170 |
| Pruned (3 files) vs partially-pruned (40 files), same query | same answer, **6.3× faster** |
| Projection pushdown, 2 of 219 columns vs `SELECT *` | **8–14× faster** |
| PLAN.md headline example (filter → group → agg → order) | runs; 25 of 62 files, 3 of 19 columns |
| Mixed-case `VendorID` through the optimizer | preserved |

**Three defects only the real table could reach**

1. **A bare date against a `timestamp` column killed the query.**
   `filter(col >= "2024-01-01")` — the form PLAN.md's own example uses — made
   PyIceberg's literal binding raise from inside `plan_files()`. Phase 2's
   "anything not understood is simply not pushed" promise did not hold, because the
   rejection happened at scan time, long after translation. Fixed twice over: every
   predicate is now **bind-validated** before use (a bad conjunct costs only its own
   pruning), and a date-only literal is **widened** to `T00:00:00` for `timestamp`
   columns so the commonest filter on a time-partitioned table actually prunes.
   `timestamptz` is deliberately left alone — see `divergence.md`.
2. **`ORDER BY <output alias>` disabled projection pushdown entirely.** `qualify`
   leaves such a reference unqualified because it names the projection, not the
   table; the extractor read that as an unattributable column and fell back to "read
   everything". The headline aggregate was scanning 19 of 19 columns (219 of 219 on
   the wide table). Now 3 of 19.
3. The `hive_partitioning` fix found on the local fixtures was confirmed necessary
   here — this warehouse really does lay data out as `.../pickup_month=2024-12/...`.

### Outstanding for Phase 2

- [ ] Rename reconciliation is local-fixture-only — none of the real tables has a
      schema-history rename, and grouping opens one parquet footer per file. Wants a
      benchmark before it meets a 4096-file table.
- [ ] `sum(double)` over parallel scans varies in the last few digits between runs
      (DuckDB's aggregation order). Expected for floats; Spark does the same. Worth a
      documented note in Phase 3 rather than a fix.

---

---

## Decision 11 — copy-on-write for now, merge-on-read deferred (2026-08-30)

Every writer of the tables in scope rewrites data files; no delete or position files
exist. So §3.3's merge-on-read hybrid split is **not built**, and the `UNION ALL` in
`exec/source_sql.py` now serves the rename path alone.

**Deferred, not dropped.** PLAN.md now carries two phases for it:

| Phase | Scope |
|---|---|
| **12 — Merge-on-read reads** | The hybrid split of §3.3: positional deletes, equality deletes, `ArrowScan(...).to_table(tasks)` for the dirty half, `UNION ALL` with the clean half |
| **13 — Merge-on-read writes** | Emitting delete files from `DELETE`/`UPDATE`/`MERGE` instead of rewriting data files. Needs 12 first, and PyIceberg has no API for it today |

What was kept, deliberately: the *detection*. "No delete files" is an assumption about
the writers, not something Iceberg enforces — the table is shared, and another engine
could add merge-on-read deletes to one we only read. `read_parquet` cannot see a delete
file, so the deleted rows would come back and the query would report success. That is
the one failure mode worth spending ten lines to make impossible, so
`_assert_copy_on_write` refuses the scan instead — naming Phase 12 in the error — and
`fx.mor` stays as the fixture that proves it fires.

That guard is also what makes the deferral safe to sit on: if a merge-on-read table
ever does appear upstream, queries start *failing* rather than quietly returning
deleted rows, which is the signal to pick Phase 12 up.

**Groundwork already in place for Phase 12**, so it is additive rather than a rewrite:
the guard becomes the branch point, `build_mor` is a real MoR table to test against
(PyIceberg cannot write one — it had to be built by hand), `TestCopyOnWriteInvariant`
becomes the correctness suite inverted, and `build_source` already emits the
`UNION ALL` the split needs. What has to come back is `ScanPlan.delete_table` /
`delete_files` and the `merge-on-read: N file(s)` line in `explain()`.

Knock-on for **Phase 8** (row-level operations): DELETE / UPDATE / MERGE write
copy-on-write, which is what PyIceberg does natively — so decision 11 removes work
there rather than adding it, and defers the harder half to Phase 13.

---

## Phase 3 — Types, expressions, function library · **IN PROGRESS** (paused 2026-08-30)

Green on: `uv run pytest` (711 passed) · `uv run pytest -m integration` (11 passed) ·
`ruff check` · `ruff format --check` · `mypy`.

### Decision 8 settled → decision 12

Evaluated SQLFrame 4.4.0 and `duckdb.experimental.spark` in a throwaway venv.
**Neither implements Spark conformance**, which is the expensive half of Phases 3–6:

| Probe | Spark | SQLFrame | duckdb.experimental.spark |
|---|---|---|---|
| `1/0` | `NULL` | `inf` | `inf` |
| `CAST('abc' AS INT)` | `NULL` | raises | raises |
| `ORDER BY x` nulls | first | no explicit clause | no explicit clause |

Neither can read Iceberg through our planner either, and SQLFrame pins
`sqlglot<30.13` against our 30.17 (our Phase 2 code *would* survive the downgrade — I
checked every API we use — but it would tie the optimizer to their release cadence).
So: **build, with SQLFrame's 478 mappings as a reference table** (MIT, nothing
imported). Recorded as decision 12.

### Built

| Piece | What |
|---|---|
| `sql/conformance.py` | The §3.5 rules as one tree pass, run before the optimizer so pushdown sees the tree that executes |
| `conf.py::SqlSettings` | `spark.sql.ansi.enabled` / `ICETL_ANSI_MODE`, off by default as in Spark |
| `parse_types.py` | `StructType.fromDDL` (both spellings), `fromJson`, via sqlglot's Spark grammar |
| `sql/column.py` | Ordering, `like`/`rlike`/`ilike`, `contains`/`startswith`/`endswith`, `substr`, `getItem`/`getField`, bitwise, `eqNullSafe`, `isNaN`, `when`/`otherwise` |
| `sql/functions.py` | **169 public names** across conditionals, strings, math, trigonometry, date/time, hashing, aggregates, arrays and maps |
| `compat/naming.py` | Spark's generated column names now lower-case the function (`sum(amount)`), keep keywords upper (`CAST(a AS INT)`), and render `count(*)` as `count(1)` |

**Why the conformance rules are a tree pass, not methods on `Column`.** The two
surfaces build the same tree but from different starting points: `df.orderBy(...)`
constructs an `exp.Ordered`, while `spark.sql("... ORDER BY x")` gets one from
sqlglot's Spark parser. A rule inside `Column` would cover one and miss the other,
and the surfaces would quietly disagree — which is what P1 exists to prevent.

### Conformance now enforced

`1/0` → NULL · `CAST('abc' AS INT)` → NULL (raises under `ansi.enabled`) · `ORDER BY`
nulls first ascending, last descending · `<=>` null-safe · all on **both** surfaces.

### Bugs the value-level tests caught

Generated SQL is not evidence. Each of these produced plausible SQL and the wrong
answer, with nothing raising:

| Function | Wrong behaviour | Cause |
|---|---|---|
| `split` | returned the whole string as one element | `sg.Split` → DuckDB `str_split`, which is *literal*; Spark's separator is a regex |
| `greatest` / `least` / `concat_ws` | NULL poisoned the result | sqlglot wraps its typed nodes in a NULL-propagating `CASE`, right for other dialects and wrong for Spark |
| `log(2, 8)` | returned 1/3 instead of 3 | the typed node renders its operands reversed |
| `date_add` | returned `TIMESTAMP` where Spark gives `DATE` | DuckDB widens |
| `F.exp` | *would* have shadowed the sqlglot module for every function below it | Spark has a function named `exp`; caught before it shipped, module now aliased `sg` |

Also corrected: PLAN.md §3.5 claims DuckDB sorts nulls *first* on `DESC`. That was an
older release — 1.5.5 is nulls-last in both directions, so `ASC` is the only real
divergence. Recorded in `divergence.md`.

### Remaining for Phase 3

- [ ] The rest of the `F.*` surface. **169 public names** are done, each with a
      value-level test; the remainder is mostly complex-type and JSON work that
      Phase 6 owns anyway. Every addition needs a value-level test: around a dozen
      of the first 169 were silently wrong — `split`, `greatest`/`least`,
      `concat_ws`, `log`, `date_add`/`add_months`/`last_day`/`trunc`, `pmod`,
      `array_position`, `array_union`, `sequence` — and `F.hash` produced a type
      Spark cannot represent at all.
- [x] Decimal promotion — **deferred to Phase 14** (decision 14). Measured, with
      Spark's exact formulas, in `divergence.md`. See the deferral note below.
- [x] Aggregate output column naming. Spark spells a generated name with the function
      in *lower* case (`sum(amount)`) and keeps keywords upper (`CAST(a AS INT)`); we
      were emitting `SUM(amount)`. Fixed via `normalize_functions`, plus `count(*)` →
      `count(1)` as Spark names it. Eight cases in `test_functions2.py`.
- [ ] `Column.over()` raises for Phase 5; windows are that phase's work.
- [x] Conformance-corpus question settled — **decision 13: no PySpark, at any stage.**
      Cases cite Spark's published behaviour. The residual risk is real and named: an
      edge case where Spark differs from its own documentation gives a confidently
      green test, and undocumented corners (NULL propagation, empty input, overflow,
      decimal promotion) are where that bites. Recorded in `divergence.md`.

---

## Decision 14 — decimal promotion deferred (2026-08-30)

Spark's decimal arithmetic result types are not reproduced. Measured on both engines:

| Operation on `DECIMAL(10,2)` | Spark | DuckDB | |
|---|---|---|---|
| `a + b` | `DECIMAL(11,2)` | `DECIMAL(11,2)` | already agrees |
| `a * b` | `DECIMAL(21,4)` | `DECIMAL(18,4)` | precision differs |
| `a / b` | `DECIMAL(16,6)` | **`DOUBLE`** | *type* differs |

Division is the one that matters — `DOUBLE` loses exactness, which is the whole
reason a money column is a decimal.

**Deferred rather than guessed.** The rule needs *operand* types, and a
sub-expression's type is only known after binding. Casting without them gives a
confidently wrong precision, which is worse than a documented divergence. **Phase 14**
holds the work: add sqlglot's `annotate_types` to the optimizer pipeline (the §3.1
schema binding it needs is already there), then emit Spark's casts from
`sql/conformance.py` so both surfaces get them from one pass.

**No guard, unlike decision 11.** Detecting "this query would have promoted
differently" needs the same type information as fixing it, so there is nothing cheap
to assert. `divergence.md` is the warning instead.

**Interim exposure:** `+` and `-` already match. `*` keeps the right scale with a
smaller precision, overflowing only at extreme magnitudes. `/` returns `DOUBLE`,
accurate to ~15 significant digits — wrong only where exact decimal semantics were
the point.

---

## Carry-over notes

Things found during earlier phases that later phases need.

| # | Note | Affects |
|---|---|---|
| 1 | Windows: PyIceberg needs `file://C:/x`, DuckDB rejects it. Handled by `icetl.paths`; don't bypass it. | any new file-reading code |
| 6 | `pandas>=2.2` resolves to pandas 3.x, whose default string dtype differs from Spark's `toPandas()` `object` dtype. Still undecided. | Phase 3 |
| 12 | sqlglot wraps some typed nodes (`Greatest`, `Least`, `ConcatWs`) in a NULL-propagating `CASE` for DuckDB, and renders `Log`'s operands reversed. Prefer a plain call and a value-level test over assuming a typed node is conformant. | Phase 3 onward |
| ~~9~~ | ~~Golden conformance files generated from real PySpark~~ — **closed by decision 13**: no PySpark at any stage. Cases are written from Spark's published behaviour with a citation each. | — |
| 10 | The optimizer declines to rewrite a `UNION` whose output names need restoring, so set operations get no pushdown. Fixable by re-aliasing the first branch. | Phase 4 |
| 11 | Two references to one table merge to a single scan (union of columns, OR of predicates). Correct, but a self-join with disjoint filters prunes less than it could. | Phase 4 |

---

## Running against the real catalog

Config lives in `.env` (gitignored; `.env.example` documents every key).

```
ICETL_CATALOG_URI=http://localhost:8182     ICETL_S3_ENDPOINT=http://localhost:9100
ICETL_CATALOG_WAREHOUSE=s3://warehouse/     ICETL_S3_PATH_STYLE_ACCESS=true
ICETL_DEFAULT_NAMESPACE=nyc
```

- [x] `scripts/smoke_catalog.py -v` — passes.
- [x] `uv run pytest -m integration` — 11 passed, including `test_phase2_rest.py`.
- [x] `httpfs`, the S3 secret, and `engine_paths` on `s3://` URLs all exercised.
- [ ] The repo has no commits yet — see "Everything is uncommitted" at the top.

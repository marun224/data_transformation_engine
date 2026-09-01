# Status

Where the build has got to. Update this at the end of each phase.
See [PLAN.md](PLAN.md) for the design and the full phase list.

---

## Resuming — read this first

**Paused:** 2026-09-01, at the end of Phase 9. Phase 10 (performance & scale) is next
and is not started.

### Where things stand

| | |
|---|---|
| Phases 0–9 | **done**, green |
| Phases 12, 13, 14, 15 | **deferred by decision**, each with its own section in PLAN.md §4 |
| ⚠️ Live gap | `weekday`/`dayofweek` are off by one through `Session.sql()` — **use `F.*` for date parts** until Phase 15 |
| Decision 15 | **done** — the PySpark compat surface is removed; see the section below |
| Tests | 1426 local, 11 integration |
| Gate | `ruff check` · `ruff format --check` · `mypy` all clean |

### Committed

| Commit | What |
|---|---|
| `92efdb8` | *"Initial commit: icetl through Phase 3 (in progress)"* — 85 files, 16,408 lines |
| `26c792d` | *"code refactoring"* — 66 files, +5,649/−1,460. Phase 3's completion (the third `F.*` tranche, the optimizer's `ArithmeticError` fix), decision 15's rename, `tox.ini`, and the notebooks |
| `dcb2739` | *"phase 4 chnages are committed"* — `groupBy().agg()` and joins |
| `8c5b945` | *"Phase 8 changes"* — 32 files, +9,630/−118. The rest of Phase 4 and the whole of Phases 5, 6, 7 and 8, plus `FINDINGS.md` |

> ⚠️ **Phase 9 is staged, not committed.** Everything through Phase 8 is in `8c5b945`.
> Phase 9 is in the index — 17 files, 4 of them new — with nothing untracked, so
> `git clean` is safe, but the work exists in exactly one place. `git status` first;
> committing it is the first thing to do.

### Picking it back up

Prerequisites: the REST catalog on `localhost:8182` and MinIO on `localhost:9100`
must be running for the integration tests. `.env` holds the settings and is
gitignored; `.env.example` documents every key. Local tests need neither.

```bash
uv sync --extra dev            # if the venv is cold
uv run tox                     # the whole gate: lint, mypy, 1426 tests, then dist/
uv run pytest -q               # 1426 expected, if you want just the tests
uv run pytest -m integration -q # 11 expected; needs the catalog up
uv run ruff check . && uv run ruff format --check . && uv run mypy .
uv run python scripts/smoke_catalog.py -v   # proof of life against the real catalog
```

### The next thing to do

**Phases 4 through 9 are complete.** The read side, writing back, changing rows in place,
and now the catalog itself: `session.catalog.*`, `CREATE`/`ALTER`/`DROP TABLE` and
namespaces, partition-spec and sort-order evolution, `mergeSchema`, time travel, and
Iceberg's metadata tables.

**Next: Phase 10 — performance & scale.** DuckDB `memory_limit` and spill sized from
available RAM, threads from physical cores, a wide-table benchmark harness against the
200-column fixture and then the real table, result streaming so a large result never
OOMs, a query cache, and benchmarks committed as a tracked file so regressions are
visible.

Four things from Phase 9 to carry into it, three of them already measured:

- **`explain()`'s `bytes_scanned` ignores column pruning** (FINDINGS §3.3) and an
  **unfiltered `count(*)` opens parquet footers** Iceberg's manifests could answer
  (§3.4). Both are Phase 10's, and both are already written down with numbers.
- **An inner join's `WHERE` conjunct is folded into the `ON` clause and stops pruning**
  (§3.5, found in Phase 9). `extract_scan_requests` reads only the scope's `WHERE`;
  reading the `ON` clause too is the fix, and `TestOuterJoinsAreNotPrunedByTheirOwnNullChecks`
  already pins the measurement so a change is visible.
- **Pruning is safe except under an outer join** (§1.10). Whatever Phase 10 adds to
  pushdown has to pass through `is_null_rejecting` for a null-padded table, or it
  reintroduces the anti-join bug in a new place.

Still open from Phase 9 itself: `ALTER COLUMN ... TYPE` is refused because PyIceberg 0.11
exposes no type update, and nested-field assignment (`UPDATE t SET s.field = v`) is still
refused — it wants the schema-aware `withField` machinery on the SQL surface, which
neither Phase 8 nor Phase 9 needed to build.

PLAN.md §4 has the full phase list, and [FINDINGS.md](FINDINGS.md) has every trap found so
far — worth ten minutes before writing anything that generates SQL.

The SQL-surface function gap found at the end of Phase 3 is **deferred to Phase 15**
(decision 16) — it is documented, not forgotten. One thing to carry while working:
`weekday`/`dayofweek` answer differently through `Session.sql()` than through `F.*`, so
reach for `F.*` on date parts.

The `F.*` surface finished at 273 names, each with a value-level test. What is missing
from it belongs to later phases by design, not by omission:

| What is left | Owner |
|---|---|
| JSON, `map_*`, `explode`, higher-order (`transform`, `filter`, `aggregate`, `zip_with`), `arrays_zip`, nested field access | **Phase 6** |
| `monotonically_increasing_id`, `first_value`/`last_value` as window forms, everything reached through `Column.over` | **Phase 5** |
| `grouping`, `grouping_id`, `broadcast` | **Phase 4** |
| Twelve functions with no faithful DuckDB spelling — `crc32`, `soundex`, `typeof`, `try_add` and the rest | each with its reason in `divergence.md` |

The rule that governed all of this still governs the next phase:

> Every function needs a test that asserts on a **value**, not on generated SQL.
> Around a dozen of the first 169 produced perfectly plausible SQL and the wrong
> answer. `tests/fixture/test_functions.py` opens with why.

---

## Phase 0 — Scaffolding · **DONE** (2026-08-18)

Project set up, catalog + DuckDB plumbing working, smoke test passing end to end.

**Built:** `conf.py` (config layering + Spark keys) · `errors.py` (PySpark exception
hierarchy) · `paths.py` (Iceberg→DuckDB path translation) · `catalog/` (registry +
resolver) · `exec/engine.py` (DuckDB) · `diagnostics/smoke.py` + `scripts/smoke_catalog.py` ·
local `SqlCatalog` test fixture with 5 generated tables.

---

## Phase 1 — End-to-end thin slice · **DONE**

`Session` / `Session.table()` / `Session.sql()`, `DataFrame` transformations and
actions, `Column` + `F.col/lit/expr`, eager analysis against zero-row Arrow views,
minimal scan planner, `explain()`.

**Built:** `types.py` (Spark type hierarchy + `Row`) · `sql/session.py` ·
`sql/dataframe.py` · `sql/column.py` · `sql/functions.py` · `plan/builder.py` ·
`plan/analysis.py` · `exec/scan_planner.py` · `exec/result.py` · the shadow
`src/pyspark` package (removed by decision 15).

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

## Phase 3 — Types, expressions, function library · **DONE** (2026-08-30)

Green on: `uv run pytest` (822 passed) · `uv run pytest -m integration` (11 passed) ·
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
| `conf.py::SqlSettings` | `icetl.ansiMode` / `ICETL_ANSI_MODE`, off by default as in the reference |
| `parse_types.py` | `StructType.fromDDL` (both spellings), `fromJson`, via sqlglot's Spark grammar |
| `sql/column.py` | Ordering, `like`/`rlike`/`ilike`, `contains`/`startswith`/`endswith`, `substr`, `getItem`/`getField`, bitwise, `eqNullSafe`, `isNaN`, `when`/`otherwise` |
| `sql/functions.py` | **273 public names** across conditionals, strings, math, trigonometry, date/time, hashing, aggregates, arrays and maps |
| `compat/naming.py` | Spark's generated column names now lower-case the function (`sum(amount)`), keep keywords upper (`CAST(a AS INT)`), and render `count(*)` as `count(1)` |

**Why the conformance rules are a tree pass, not methods on `Column`.** The two
surfaces build the same tree but from different starting points: `df.orderBy(...)`
constructs an `exp.Ordered`, while `Session.sql("... ORDER BY x")` gets one from
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

### The third `F.*` tranche (2026-08-30) — 105 more names, 274 at the time

Four batches, each written against DuckDB probes first and then held to Spark's
documented values:

| Batch | Names | What needed real work rather than a pass-through |
|---|---|---|
| Strings | 26 | `overlay` (composed from `substring`), `regexp_substr`/`regexp_instr` (DuckDB's `''`-for-no-match had to be told apart from a genuine empty match), `split_part` (`''` not NULL out of range), `find_in_set` (refuses to match across fields), `octet_length` (`strlen`, not `length`) |
| Math | 21 | `rint` (HALF_EVEN — `round` is half-away-from-zero and differs on *every* tie), `shiftrightunsigned` (widen to `HUGEINT`, add 2^64, shift, narrow), `width_bucket` (one normalised fraction, so descending bounds work), `csc`/`sec`/`log1p`/`expm1` (absent from DuckDB) |
| Date/time | 20 | `weekday` (Monday 0 — a *different* shift from `dayofweek`'s Sunday 1), `next_day` (strictly after, so a zero delta means 7), the epoch conversions, `date_part` |
| Hashing, session, aggregates, arrays | 38 | `array_size` vs `size` (NULL vs -1), `get` vs `element_at` (0- vs 1-indexed, both Spark's), `array_except` (de-duplicates), `equal_null`, `assert_true`, the nine `regr_*` |

**Two functions refuse their argument rather than ignore it.** `rand(seed)` and
`randn(seed)` raise: DuckDB seeds per *connection*, not per expression, so accepting a
seed would return unseeded values from the one argument that exists for
reproducibility. `date_part('DOW', …)` raises for the same class of reason — Spark
numbers Sunday 1 and DuckDB numbers it 0, and an off-by-one weekday looks perfectly
fine — and the error names `dayofweek` and `weekday` instead.

**Twelve functions were left unimplemented on purpose**, each with its reason written
into `divergence.md`: `try_add`/`try_subtract`/`try_multiply` (NULL-on-overflow is not
reachable — DuckDB raises and has no `TRY(...)`), `crc32`, `soundex`, `typeof`,
`input_file_name`, the three time-zone functions (need the ICU extension, which is a
load-time decision rather than a function), `array_insert`, and `shuffle`.

### A Phase 2 defect the tranche uncovered

`SELECT 1.0 / 0.0` **crashed**, on both surfaces, with `decimal.DivisionByZero` raised
from inside sqlglot's `simplify` rule. Spark answers NULL.

`optimize_plan` caught `OptimizeError`, `KeyError`, `ValueError` and `TypeError` — every
way a rule was known to fail. But `simplify` constant-folds literal arithmetic, and
arithmetic has its own exception tree, so the error escaped and Phase 2's "a rule that
fails costs only that rule" did not hold. The integer spelling `1 / 0` *was* tested on
both surfaces and could never have found it: `simplify` declines to fold integer
division at all. `ArithmeticError` is now in the guard; the rules that ran before the
failure are kept, and the Spark parser's own `NULLIF` still makes the answer NULL.

### Phase 3's checklist, closed

- [x] The rest of the `F.*` surface. **273 public names** (274 until decision 15
      removed `spark_partition_id`), each with a value-level
      test. What is left over is Phase 4/5/6 work — see "The next thing to do" at the
      top for the split. Around a dozen of the first 169 were silently wrong —
      `split`, `greatest`/`least`, `concat_ws`, `log`,
      `date_add`/`add_months`/`last_day`/`trunc`, `pmod`, `array_position`,
      `array_union`, `sequence` — and `F.hash` produced a type Spark cannot represent
      at all. The third tranche added `rint`, `weekday`, `regexp_substr`,
      `array_size` and `split_part` to that list of things generated SQL would not
      have caught.
- [x] Decimal promotion — **deferred to Phase 14** (decision 14). Measured, with
      Spark's exact formulas, in `divergence.md`. See the deferral note below.
- [x] Aggregate output column naming. Spark spells a generated name with the function
      in *lower* case (`sum(amount)`) and keeps keywords upper (`CAST(a AS INT)`); we
      were emitting `SUM(amount)`. Fixed via `normalize_functions`, plus `count(*)` →
      `count(1)` as Spark names it. Eight cases in `test_functions2.py`.
- [x] `Column.over()` raises, naming Phase 5. Windows are that phase's work by
      PLAN.md §4, so raising is the finished state here rather than a gap.
- [x] Carry-over note 6 (pandas 3.x string dtype) — closed. `exec/result.py`
      rebuilds string columns from `to_pylist()`, which fixes the dtype *and* the
      null sentinel at once: pandas 3 would otherwise give `str` dtype holding
      `nan` where Spark gives `object` holding `None`, and `astype(object)` alone
      keeps the `nan`. Two cases in `tests/unit/test_result.py`.
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

## Decision 15 — the PySpark compatibility surface is removed (2026-08-30)

Green on: `uv run pytest` (821 passed) · `ruff check` · `ruff format --check` · `mypy`.

icetl is an Iceberg + DuckDB library, and the naming now says so. What changed:

| Was | Now |
|---|---|
| `SparkSession` | `Session` |
| `PySparkException` / `PySparkTypeError` / `ValueError` / `AttributeError` / `NotImplementedError` | `EngineException` / `EngineTypeError` / … |
| `src/pyspark/**` (9 re-export modules) | deleted, with its `pyproject.toml` entries |
| `spark.sql.catalog.<name>.*`, `spark.sql.ansi.enabled`, `spark.sql.defaultCatalog` | `icetl.catalog.<name>.*`, `icetl.ansiMode`, `icetl.defaultCatalog` |
| `F.spark_partition_id` | removed — meaningless in one process; 273 names |
| `Session.sparkContext`, `Builder.master/enableHiveSupport/remote` | removed — cluster and RDD concepts with nothing to map onto |
| `apply_spark_semantics`, `arrow_to_spark_type/schema`, `spark_output_name`, `SPARK_VERSION` | `apply_compat_semantics`, `arrow_to_datatype`/`arrow_to_struct_type`, `output_name`, `REFERENCE_SEMANTICS_VERSION` |
| `Session.version` → `"3.5.0"` | → icetl's version; `Session.reference_semantics` gives `3.5.0` |

**Spark did not stop being the specification.** P5 still holds; it is now written as
*the reference*, defined once at the top of `compat/divergence.md`. Nothing runs on,
links against, or requires Spark.

**Two things survive on purpose**, and are commented so they do not read as misses:

1. **sqlglot's `"spark"` dialect** — now the named constant `SQL_DIALECT` in
   `icetl/compat/__init__.py`, used at all 12 sites. It is an argument value in
   sqlglot's API naming a SQL *grammar*; changing it would change the language accepted.
2. **The `F.*` vocabulary** — `date_format`, `weekday`, `array_except` and the rest are
   the data vocabulary the library exists to provide. Only `spark_partition_id` was
   brand-named, and it is gone.

**Two hazards found while doing it**, both worth remembering:

- **Rewriting `.py` files with PowerShell corrupted them.** `Get-Content`/`Set-Content
  -Encoding utf8` on PS 5.1 reads as ANSI and writes UTF-8 *with BOM*, double-encoding
  every non-ASCII character — `héllo` became `héllo` and three string-length tests
  failed. Bulk edits go through Python with explicit `encoding="utf-8"`.
- **A blind prose substitution corrupted test data.** `F.lit("Spark SQL")` is an input,
  not prose. The repair distinguishes docstrings from other string literals with `ast`;
  the payloads are now neutral (`"Basic SQL"`), base64 cases included.

**Not covered by any test:** the shadow package had no test importing it, so its
deletion regressed nothing — and nothing ever proved the zero-edit promise worked.

---

## Phase 4 — Relational breadth · **DONE** (2026-08-31)

Green on: `uv run tox` (lint · mypy · 1072 tests against the built wheel · dist) and
`uv run pytest -m integration` (11 passed).

Test count went 821 → 882 → 922 → 1024 → 1072 across the phase.

### `groupBy().agg()` — done

**Built:** `sql/group.py` (`GroupedData`) and `DataFrame.groupBy` / `groupby` / `agg`.

`GroupedData` holds no plan of its own — it is the frame plus the grouping expressions,
and only `agg()` builds anything, so `groupBy` alone stays lazy (P3). Output is
**grouping keys first, then aggregates**, which is what a script indexing `row[0]`
depends on.

| Form | Example |
|---|---|
| expressions | `gd.agg(F.sum("amount").alias("total"))` |
| dict | `gd.agg({"amount": "sum"})` |
| keywords | `gd.agg(total=F.sum("amount"))` |
| shortcuts | `gd.count()`, `gd.sum("a")`, `gd.avg/mean/min/max("a")` |
| whole frame | `df.agg(...)`, same as `df.groupBy().agg(...)` |

**What the 29 tests pin.** `fx.plain` was chosen for its nulls (`vendor` is
`a, b, a, c, NULL`, `amount` is `10.0, 20.5, 30.25, NULL, 50.0`), because that is where
aggregation goes quietly wrong:

- NULL is **its own grouping key** — 4 groups from 5 rows;
- an **all-NULL group sums to NULL, not 0** (vendor `c`);
- `count(*)` counts a row that `count(col)` skips — `("c", 1, 0)`;
- `.count()` names its column `count`, not `count(1)`;
- filtering *after* a group nests the plan, so the predicate hits groups not rows;
- P1: the same query through `Session.sql()` gives the same answer.

Two refusals rather than guesses: `agg("amount")` raises pointing at `F.sum("amount")`
(a bare string is a column, not an aggregate), and `gd.sum()` with no columns raises
naming Phase 4 — the reference aggregates every numeric column there, which needs the
frame's types.

**Pushdown survives grouping**: the headline aggregate on `nyc.yellow_tripdata` still
reads 3 of 62 files and 3 of 19 columns.

### Joins — done

**Built:** `DataFrame.join` / `crossJoin`, plus `_JOIN_KINDS` mapping every spelling of
`how` the reference accepts (`left`, `leftouter`, `left_outer`, `leftsemi`, …).

`inner` · `left` · `right` · `outer`/`full` · `semi` · `anti` · `cross`, with `on` as a
name, a list of names, or a Column. Self-joins work through `.alias()`.

**The two `on` forms differ deliberately**, and this closes a divergence that had been
open since Phase 3:

| `on` | SQL emitted | Key columns out |
|---|---|---|
| `"id"` or `["id"]` | `USING (id)` | **one** |
| `col("a.id") == col("b.id")` | `ON a.id = b.id` | **two** |

Verified against DuckDB before building: `USING` collapses the key, `SEMI`/`ANTI` are
supported natively, and semi/anti emit the left side's columns only.

**A new divergence, recorded:** where a join yields two same-named columns, the reference
keeps both called `id` and *raises* on a later `df["id"]` as ambiguous. Ours names them
`id` and `id_1`, so the select resolves. Different, arguably kinder, written down.

### Two defects found while building

- **Projecting through self-join aliases failed.** `df.alias("x").join(df.alias("y"), …)
  .select(col("x.vendor"))` raised *"Referenced table x not found"*: the projection logic
  nested any joined plan into a subquery, which hid the aliases. Replacing a projection
  over a join is safe — the FROM and its aliases stay put — so `_rebase_projection` now
  allows it via `_projection_is_replaceable`. `filter`'s extend-vs-nest rule is untouched;
  changing it needs its own tests.
- **`args["from"]` is `args["from_"]` in sqlglot 30.** My first `_as_join_relation` read
  the old key, silently got `None`, and wrapped the right side in a generated alias — so
  `b.id` stopped resolving. `_has_from_clause` already documented this exact trap; there
  is now a `_from_clause` companion beside it so the next reader finds the accessor
  rather than the key.

### Set operations — done (2026-08-31)

**Built:** `DataFrame.union` / `unionAll` / `unionByName` / `intersect` / `intersectAll` /
`exceptAll` / `subtract`, driven by one `_SET_OPERATIONS` table, plus `_as_set_branch`
for the nesting rules below. 36 tests in `tests/fixture/test_setops.py`.

| Method | SQL | De-duplicates |
|---|---|---|
| `union` / `unionAll` | `UNION ALL` | **no** |
| `unionByName` | `UNION ALL`, re-projected | **no** |
| `intersect` | `INTERSECT` | yes |
| `intersectAll` | `INTERSECT ALL` | no |
| `subtract` | `EXCEPT` | yes |
| `exceptAll` | `EXCEPT ALL` | no |

DuckDB's multiset and NULL semantics were checked against the reference before building
and agree on every point: `EXCEPT ALL` subtracts multiplicities (3 minus 1 leaves 2),
`INTERSECT ALL` keeps the smaller (min of 3 and 2 is 2), and both `INTERSECT` and `EXCEPT`
match **NULL to NULL** where `=` would not. So no conformance layer was needed here.

**One line of the SQL surface came free.** `INTERSECT` and `EXCEPT` had been rejected as
*"not implemented, scheduled for Phase 4"* — but only because `Session.sql()`'s allowlist
named `exp.Union` rather than its parent `exp.SetOperation`. sqlglot parsed them and
DuckDB ran them correctly the whole time. Widening the isinstance check was the entire fix.

**The nesting rule is a correctness guard, not tidiness.** DuckDB binds `INTERSECT`
tighter than `UNION ALL`, so an inlined `a.union(b).intersect(c)` evaluates
`a UNION ALL (b INTERSECT c)` — a different query that raises nothing and answers wrongly.
Measured before the guard existed: the flat spelling returned 2 rows where the intended
grouping returns 1. `_as_set_branch` therefore nests any branch that is itself a set
operation, and any branch carrying LIMIT/OFFSET/ORDER BY (those bind to the whole set
operation, and DuckDB will not even parse them mid-branch).

**`union` matches by position, and that is the reference's behaviour, not a shortcut.**
Two frames carrying the same names in a different order union into nonsense without
complaint — and the widening makes it worse rather than louder: `bigint` over `string`
settles on `string`, so an `id` of `1` comes back as `'1'`. `test_union_matches_by_
position_not_by_name` pins the trap deliberately. `unionByName` is the safe spelling.

**`unionByName` fills with a bare `NULL`, not a cast one.** A set operation takes each
column's type from the branches that do have it, so DuckDB types a filled column from the
other side and the two agree — no Spark-type-to-SQL-type mapping needed on this path.

### Grouping sets and pivot — done (2026-08-31)

**Built:** `DataFrame.rollup` / `cube`, `GroupedData.pivot`, and the three `F.*`
leftovers `grouping`, `grouping_id`, `broadcast` (276 names now). 27 tests in
`tests/fixture/test_grouping_sets.py`.

`rollup` and `cube` are grouping *sets*: they go in their own `exp.Group` argument rather
than in `expressions`, because a grouping set replaces the plain key list rather than
accompanying one — putting the keys in both would group by them twice.

**The trap they share is a wrong answer, not an error.** A rolled-up key comes back as
NULL, and `fx.plain`'s `vendor` already contains a real NULL. So a rollup over it returns
*two* rows whose vendor is NULL — the group of rows that had no vendor (total 50.0) and
the grand total over every row (110.75) — and nothing in the values tells them apart.
`F.grouping` is the only thing that does, which is why it landed in the same slice rather
than being left as an `F.*` leftover.

**`pivot` compiles to conditional aggregation**, not to DuckDB's `PIVOT` — that is a
statement rather than an expression and would not compose with the rest of a plan. So
`sum(v)` becomes `sum(CASE WHEN k IS NOT DISTINCT FROM 'x' THEN v END)`, which is how the
reference compiles a pivot too. Three details fell out of doing it that way:

- the match is **null-safe**, so a NULL pivot key gets its own column named `null` rather
  than an empty one — a NULL key is a group, not an absence;
- **every aggregate in the expression is rewritten**, not the expression as a whole, so a
  composite like `sum(a) / count(b)` restricts both sides before dividing;
- `count(*)` has no argument to restrict, so it counts a literal instead.

`pivot` is **the one transformation that runs a query**: with `values` omitted the
distinct values must be known before the projection can be written. That is what the
reference does, for the same reason. Pass `values` to keep it lazy. A `pivotMaxValues`
guard at 10,000 is carried for the reference's reason — a pivot on a high-cardinality
column does not fail, it succeeds and returns thousands of columns.

**A live bug found by using the new code.** `F.count(<Column>)` raised
`EngineValueError` — `col == "*"` builds a comparison *expression*, not a bool, so the
`or` guarding it reached `Column.__bool__`. Only the string spellings (`F.count("*")`,
`F.count("amount")`) had tests, which is how 882 tests passed with it broken.
`crosstab` was the first caller to pass a Column and found it immediately. Fixed by
testing the type before the value; `TestCountAcceptsAColumn` covers all three spellings.

### `na.*` and `stat.*` — done (2026-08-31)

**Built:** `sql/na.py` (`DataFrameNaFunctions`) and `sql/stat.py`
(`DataFrameStatFunctions`), plus `df.dropna` / `fillna` / `replace` as the frame-level
spellings. 40 tests in `tests/fixture/test_na_stat.py`.

**The type rule is the whole of `na`.** A fill value only touches columns it could
plausibly belong to: `fill(0)` fills numeric columns and leaves a NULL string alone,
`fill("?")` does the reverse, and `fill(True)` reaches neither here because `fx.plain`
has no boolean. Getting that wrong is a wrong answer that a test on a numeric column
alone would not catch, so every fill test asserts on the column that must **not** have
changed as well as the one that must.

Two smaller precedences, both the reference's and both spelled out where they live:
`thresh` **overrides** `how` rather than combining with it, and a name in `subset` that
the frame does not have is ignored (it is a filter over the columns) while a key in
`fill`'s dict form that the frame does not have is refused (it can only be a mistake).

For `stat`, the choices worth knowing:

| Method | Choice |
|---|---|
| `approxQuantile` | `quantile_disc`, so `relativeError=0` returns an **observed value** rather than an interpolation — which is what the reference's quantiles are. Above zero, DuckDB's bound is fixed, so the answer may be more accurate than asked and never less |
| `cov` | `covar_samp`, not `covar_pop` — the reference divides by `n-1`, and on a small frame the two differ by enough to matter |
| `corr`, `cov` | a **non-numeric column is refused**. DuckDB would cast a date and answer; a number that means nothing is worse than an error |
| `crosstab` | built on the new `pivot`. An absent pair counts **0**, not NULL — it is an observed zero, not a missing measurement — and NULL is labelled `null` on both axes so the row label matches the column header |
| `freqItems` | **exact**, not a sketch: the reference approximates because it counts across a cluster, and one grouped scan per column is strictly inside the same contract. The result is folded into a **literal plan with no source**, so the frame is independent of the table it came from |

### Sampling, caching, partitioning and temp views — done (2026-08-31)

**Built:** `sample`, `randomSplit`, `cache`/`persist`/`unpersist`, `repartition`/
`coalesce`, `createOrReplaceTempView`/`createTempView`, `Session.dropTempView`, and
`Session._materialize`/`_release`. 35 tests in `tests/fixture/test_materialize.py`.

Most of this group has less to do than the names suggest, because the reference is a
distributed engine and this is not: `repartition` and `coalesce` are no-ops with
docstrings, and so is `F.broadcast`. They exist so a script written against the reference
runs unaltered. The three that carry real work:

**`cache` registers with *both* DuckDB connections, and that is the design.** Execution
and analysis run on deliberately separate connections — the analyzer never loads httpfs,
never sees a credential and never opens a file — so a temp table created for execution is
invisible to schema resolution. The engine gets the real rows; the analyzer gets a
zero-row view of the same Arrow schema. Without that, a cached frame would `collect()`
fine and fail the moment you tried to `select` from it, which is the case
`test_a_cached_frame_can_still_be_transformed` exists to hold down.

`cache` is **eager** and returns a **new** frame rather than `self`. Nothing here mutates
a plan in place, so the reference's lazy mark would have nowhere to live. Recorded in
`divergence.md`; the consequence is that `df.cache()` on its own caches nothing reachable.

**`randomSplit` materialises first, and must.** Each split is a filter over one random
number per row, so those numbers have to be the *same* numbers every time a split is
collected — otherwise a row lands in two splits or in none, and nothing about the result
looks wrong. Drawing once into a temp table is what makes the splits disjoint and
complete. The tests assert on membership rather than on sizes, since sizes are the part
allowed to vary.

**`sample` refuses `withReplacement=True`** rather than approximating it. DuckDB draws
each row at most once; quietly handing a without-replacement sample to a caller who asked
for the other is a wrong answer. `sample` also joined `_POST_FILTER_CLAUSES`, because
`USING SAMPLE` draws from what the FROM and WHERE produced — merging a later filter into
a sampled SELECT would filter *before* the draw instead of after it.

**A temp view is a plan, not rows.** `_inline_temp_views` substitutes the registered plan
before source keys are collected, so the optimizer, pushdown and the scan planner never
learn that views exist — and a query through a view still prunes to 1 of 3 files. CTE
names are skipped, so `WITH v AS (...)` shadows a view called `v` rather than colliding
with it.

`CREATE VIEW` through `Session.sql()` is still Phase 9, but it no longer says only
*"scheduled for Phase 9"* — that sent people away from `createOrReplaceTempView`, which
works today. The message now names it.

### De-duplication, ordering and local data — done (2026-08-31)

**Built:** `distinct` / `dropDuplicates` / `drop_duplicates`, `orderBy` / `sort` /
`sortWithinPartitions`, `Session.createDataFrame` and `Session.range`. 48 tests in
`tests/fixture/test_order_local.py`. These four are what closed the phase.

**`orderBy` spells no null placement, deliberately.** `_fix_null_ordering` is a tree pass
over every `exp.Ordered`, so a sort built here and one parsed from `Session.sql()` reach
the same node and come out identical — nulls first ascending, last descending, on both
surfaces. Putting the rule in `orderBy` too would give the two surfaces their own answers
to the same question, which is exactly what P1 exists to prevent. `TestOrderingIsShared`
checks that this is true rather than merely intended.

**Two clause-precedence rules, both guarding wrong answers rather than errors.** SQL
applies DISTINCT and ORDER BY *before* LIMIT, so folding either into a frame that has
already taken a LIMIT changes which rows come back, silently:

| Call | Correct | What merging would have given |
|---|---|---|
| `df.limit(3).orderBy("id", ascending=False)` | `[3, 2, 1]` — those three rows, sorted | `[5, 4, 3]` — the table sorted, then three taken |
| `df.limit(3).distinct()` | 3 rows | the whole table de-duplicated, then three |

So `_rebase_order` and `_rebase_distinct` nest past a LIMIT or OFFSET. A later `orderBy`
with no limit in between still *replaces* the earlier one rather than stacking, which is
the reference's behaviour.

`dropDuplicates(subset)` compiles to DuckDB's `DISTINCT ON`. Which row survives is not
defined — the reference does not promise one either, because on a partitioned engine it
cannot — so the tests assert that the keys are unique afterwards, not which row carried
them.

**`Session.range` is lazy; `createDataFrame` is not.** A range is a DuckDB table
*function*, so `collect_source_keys` skips it, nothing asks the catalog to resolve a
table called `range`, and a billion-row range costs nothing until something reads it.
`createDataFrame` goes through the same `_materialize` path as `cache()`, which is what
makes the frame independent of the Python objects it came from — `test_the_frame_reads_
no_table` clears the source list and still collects.

A typed schema **casts** rather than reinterprets, so it obeys the same conformance rules
as any other cast: `createDataFrame([("abc",), ("7",)], "n bigint")` gives NULL and 7, and
raises under `icetl.ansiMode`. That fell out of reusing `Column.cast` rather than
converting types by hand.

**A bug the tests caught in my own code.** With a `names` list shorter than the data,
the tuple path built the Arrow table *from the names* and silently dropped the extra
column — so the width check downstream compared 1 against 1 and passed. Now the table is
always built with the data's own column count and renamed afterwards, so a wrong-length
`names` is caught instead of truncating the rows to fit.

### Carry-over note 10 — measured, then closed (2026-08-31)

The note as first written said set operations "get no pushdown". Measured on
`fx.partitioned`, that was too broad: the loss was **conditional on the branch names**,
and where it bit, it took everything rather than only the rename.

| `SELECT … UNION ALL SELECT …` | Files | Columns | Pushed filters |
|---|---|---|---|
| `id` (names already match) | 2 of 3 | 2 of 3 | pushed |
| `sum(amount)` — **before** | 3 of 3 | 3 of 3 | **none** |
| `sum(amount)` — **after** | **2 of 3** | **2 of 3** | **pushed** |

**Why it cost so much.** `qualify` renames an unaliased projection to `_col_0`, so the
optimized plan has to be renamed back before it can be adopted. `_restore_output_names`
declined for anything that was not a plain `exp.Select` — and declining means returning
`None`, which discards the *whole* pipeline, not merely the re-aliasing. Predicate and
projection pushdown went with it. The query still answered correctly; it just read every
file and every column to do so, which is the kind of defect nothing fails on.

**The fix, six lines.** A set operation's output names come from its **leftmost branch**,
however deeply nested — `A UNION B UNION C` parses left-heavy, and the other branches
match positionally with their own names never read. `_naming_branch` walks down to that
one `Select` and the existing positional re-aliasing does the rest. The result is then
re-checked against the expected names rather than assumed: a branch shape that did not
take the rename is still declined, or the caller would be promised names the plan does
not produce.

Worth fixing rather than documenting, because the bad row was the ordinary one:
`groupBy().agg(F.sum("amount"))` emits an unaliased `SUM(amount)`, so *any* set operation
over an aggregate landed in it. `TestNoteTen` in `test_setops.py` now pins all four
properties — plan adopted, name preserved, both predicates pushed, unused column pruned —
and one more that matters more than any of them: that the answer is still right. Pruning
that changed the answer would be the only outcome worse than not pruning.

Carry-over note 11 (two references to one table merging into a single scan) is visible in
the generated join SQL — both sides of a self-join read the same
`$icetl_src_0_paths_0`. Correct, but a self-join with disjoint filters prunes less than
it could.

---

## Phase 5 — Window functions · **DONE** (2026-08-31)

Green on: `uv run tox` (lint · mypy · 1115 tests against the built wheel · dist) and
`uv run pytest -m integration` (11 passed). Test count went 1072 → 1115.

**Built:** `sql/window.py` (`Window`, `WindowSpec`), `Column.over()`, and twelve `F.*`
names — `row_number`, `rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile`, `lag`,
`lead`, `nth_value`, `first_value`, `last_value`, `monotonically_increasing_id`. 288
names now. 43 tests in `tests/fixture/test_window.py`.

### The frame default is the whole phase

PLAN.md called frame semantics "where Spark/DuckDB drift is likeliest". Probed before
building, they **agree on every point** — which is the useful finding, because it means
the conformance layer needed nothing. But the default frame is still the thing that will
bite a caller, and it is not a divergence, it is SQL:

| Query | Result on `x = 10, 20, 20, 40` |
|---|---|
| `sum(x).over(Window.orderBy("x"))` | `10, 50, 50, 90` |
| the same, `.rowsBetween(unboundedPreceding, currentRow)` | `10, 30, 50, 90` |

With an ordering and no explicit frame the default is
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which includes every row **tying**
with the current one — so a running total jumps over a tie rather than climbing through
it. The two spellings agree on any column without duplicates, which is exactly what
makes it easy to ship the wrong one. `TestFrameDefaults` pins both.

The same shape produces the `last_value` surprise: over the default frame it returns the
**current row**, not the last row of the partition, because the frame ends at the current
row. SQL's rule, and the reference has it too. `first_value` looks right by coincidence —
the frame's start really is the partition's start. Both are documented on the functions
and tested side by side against the explicit unbounded frame.

### Two things inherited rather than built

- **Null ordering inside `OVER`.** `window.py` spells no null placement at all, and gets
  the reference's anyway: `_fix_null_ordering` is a tree pass over every `exp.Ordered`,
  and a window's ordering is made of the same nodes as a top-level one. Verified in the
  generated SQL (`ORDER BY "vendor" NULLS FIRST`) and against `Session.sql()` (P1).
- **Pushdown.** A window reads its whole partition, which is easy to mistake for reading
  everything. It does not: `row_number() OVER (ORDER BY id)` on `fx.wide` reads **1 of
  200 columns**, and a filter outside the window still prunes to 1 of 3 files.

### Design notes

`WindowSpec` is **immutable and frame-independent** — it holds no DataFrame, so a column
name inside it stays unresolved until the plan it lands in is analysed, and one spec can
serve several frames. Every method returns a new spec, so a spec handed to two columns
cannot be changed by either.

Boundaries follow the reference's sign convention — negative preceding, positive
following, zero the current row — with `Window.unboundedPreceding`/`unboundedFollowing`
carrying the same signed-64-bit extremes the reference uses, so a script passing those
numbers directly still works. `abs()` is applied on the way out because SQL states
distance and direction separately.

`monotonically_increasing_id` is `row_number() OVER () - 1`. The reference only promises
monotonic and unique, not consecutive; here there is one partition, so they come out
consecutive, and anything relying on that is relying on more than either engine promises.

---

## Phase 6 — Complex types · **DONE** (2026-08-31)

Green on: `uv run tox` (lint · mypy · 1177 tests against the built wheel · dist) and
`uv run pytest -m integration` (11 passed). Test count went 1115 → 1177.

**Built:** `sql/generators.py` (`SchemaAwareColumn`, `GeneratorColumn`,
`StructPutColumn`, `StructDropColumn`), `Column.withField` / `dropFields`,
`printSchema(level=...)`, and 26 more `F.*` names — the `explode` family, the JSON
family, the map family, and the higher-order functions. 314 names now. 62 tests in
`tests/fixture/test_complex_types.py`.

### Reading already worked; generating rows did not

The first thing the phase found was how little of it was left to do. Struct, array and
map columns already read, projected and round-tripped end to end — `fx.nested` came back
with a `Row` for the struct, a list for the array and a dict for the map, and
`col("person.name")`, `getField` and `getItem` all resolved. Phases 1–3 had built enough.

What was missing was everything that **produces** rather than consumes.

### Generators, and why they live in `select`

`explode` yields one column for a list and **two** for a map; `inline` yields one per
field of the exploded struct. So the number of output columns depends on the *type* of
what is being exploded, which is not knowable in `F` — a `Column` there has no frame. So
generators are a `Column` subclass that `DataFrame.select` expands, resolving the type
through the same zero-row bind the schema property uses.

Two findings made this far cheaper than expected:

- **Rows came free.** DuckDB's `unnest` in a select list already turns one row into one
  per element, so `explode` needed no lateral join and no plan surgery. It is an
  ordinary expression that happens to change the cardinality.
- **Repeating `unnest(x)` in one select list unnests once, not twice.** DuckDB
  correlates the copies. That is what lets `posexplode` emit
  `generate_subscripts(x, 1) - 1` beside `unnest(x)` and get matching pairs rather than
  a cross product — `test_the_positions_pair_with_their_elements` is the guard, because
  a cross product would have been four plausible-looking rows instead of two.

`select` refuses **two** generators in one projection: they would have to agree on how
many rows to produce, and there is no answer to that. The reference refuses it too.

### Three defects the phase uncovered, two of them pre-existing

| What | Detail |
|---|---|
| **`F.struct` with an aliased column emitted invalid SQL** | `F.struct(F.lit(1).alias("a"))` produced `{'a': 1 AS "a"}`, which will not parse. The alias names the field and then has to be *dropped from the value*. Nothing caught it because every earlier caller passed plain column names |
| **`struct_insert` does not overwrite** | `withField` was written on the assumption that it replaces a field of the same name. It refuses one — `Duplicate struct entry name` — so `withField` had to become schema-aware and rebuild the struct, exactly as `dropFields` does |
| **sqlglot's `Bracket` is 0-based** | `p[1]` generated `p[2]` for DuckDB: the node treats its subscript as 0-based and adds one. The error then names an index the query never mentioned. `struct_extract(p, 1)` is unambiguous and is what is used. Carry-over note 12's hazard, in a new place |

### Conformance work that was actually needed

Most of the DuckDB spellings agreed with the reference and needed nothing. Two did not,
and both are on empty input — the case a test over a populated fixture never reaches:

- **`exists` over an empty list.** `list_bool_or([])` is NULL; the reference says false.
- **`forall` over an empty list.** `list_bool_and([])` is NULL; the reference says true,
  vacuously.

Both are now spelled out as three-valued `CASE`s — true / NULL when a NULL element is
present / false — rather than left to the aggregate. `fx.nested`'s second row has an
empty `tags`, which is why the suite reaches this at all.

**`aggregate`** needed the same care for the same reason: DuckDB's `list_reduce` has no
initial value and starts from the first element, so the zero is **prepended to the
list**. That is the same fold, and it is also what makes an empty list return the zero
rather than NULL.

### One divergence recorded

`map_concat` refuses maps whose value types differ (`map<string,int>` with
`map<string,bigint>`) where the reference widens. It refuses **loudly**, so the exposure
is an error rather than a wrong answer; cast the narrower side.

---

## Phase 7 — Write path v1 · **DONE** (2026-08-31)

Green on: `uv run tox` (lint · mypy · 1218 tests against the built wheel · dist) and
`uv run pytest -m integration` (11 passed). Test count went 1177 → 1218.

**Built:** `sql/writer.py` (`DataFrameWriter`), `df.write`, `Session._insert` for SQL
`INSERT INTO` / `INSERT OVERWRITE`, and `Session._invalidate_source`. 41 tests in
`tests/fixture/test_write.py`, each writing into its own throwaway table in a `wr`
namespace and dropping it afterwards — the `fx.*` fixtures are session-scoped and
deliberately immutable.

| Piece | Behaviour |
|---|---|
| `saveAsTable` | creates the table when missing, in **every** mode — the mode is about existing data, and there is none |
| modes | `append`, `overwrite`, `ignore`, and `error`/`errorifexists` as the default |
| `insertInto` | never creates; matches columns **by position**, not by name |
| `partitionBy` / `partitionedBy` | identity partitioning on a **newly created** table |
| `sortBy`, `tableProperty` | likewise, on creation only |
| `partitionOverwriteMode=dynamic` | replaces only the partitions the incoming data touches |
| SQL `INSERT` | routed through the same writer, so both surfaces agree (P1) |

### Two findings that changed the design

**A write was invisible to the session that made it.** Every count in the first smoke
test came back unchanged — create 5, append 5, overwrite 5 — which reads like the write
silently failing. It was the *read* that was stale: a `ScanSource` holds a PyIceberg
table pinned to the snapshot it was loaded at, and the session caches sources for its
whole lifetime. `_invalidate_source` now drops the entry after a write, matching on the
resolved identifier so `nyc.trips` and `trips` both go. A frame built *before* the write
keeps its snapshot, which is read-your-plan rather than a leak, and has its own test.

**PyIceberg 0.11 already implements dynamic partition overwrite.** The hardest item on
PLAN.md's list — "overwrite only the partitions present in the incoming data" — is
`Table.dynamic_partition_overwrite`, found while reading the API. It detects the
partition values in the incoming Arrow table itself, so nothing here has to translate a
partition spec into an overwrite predicate. That removed the one part of this phase that
would have needed a transform-aware boundary calculation, and with it the risk of a
wrong `overwrite_filter` deleting data it should not have.

### Two of PLAN.md's Phase 7 goals turned out to be wrong about the world

Both are measured, not assumed, and both have a test that will say so if the world
changes.

**"Streaming Arrow batches from DuckDB → PyIceberg."** Not possible on 0.11.1:
`Transaction.append` opens with `if not isinstance(df, pa.Table): raise ValueError`, so
there is no reader or batch form to hand it. The whole result is materialised as one
Arrow table. Chunking it into several appends would have bought streaming at the price of
the atomicity below, which is the worse trade. `TestStreamingIsBlockedUpstream` asserts
the refusal, so if a later PyIceberg accepts a reader, that test fails and the failure is
the signal that this became buildable.

**"One Iceberg snapshot per write."** True for an append; an **overwrite is two** —
a `delete` then an `append` — because that is how Iceberg models replacing rows. What
does hold is the property the goal was reaching for: `Table.overwrite` wraps both in a
single transaction, so it is **one commit**, and no reader ever sees the table
mid-overwrite. `TestSnapshotShape` pins both halves.

### The nullability worry, resolved

Phase 6's closing note warned that the analysed schema calls every column nullable, and
that writing would not tolerate it. Half right:

- **Creating** a table here does produce all-optional fields, because the schema comes
  from the executed Arrow result. Recorded in `divergence.md`.
- **Writing into** an existing table is safe regardless — PyIceberg's
  `_check_pyarrow_schema_compatible` validates the incoming schema against the table's
  own and rejects a mismatch before anything is written. That is also what turns a
  by-position insert of mismatched types into a loud error rather than scrambled data,
  which `test_a_type_mismatch_is_caught_rather_than_scrambled` pins.

### Two unused exceptions put to work

`TableAlreadyExistsException` and `TempTableAlreadyExistsException` had been in
`errors.py`'s `__all__` since Phase 0 with no caller. The writer needed the first; the
second replaced an `EngineValueError` that Phase 4's temp views should have used.

---

## Phase 8 — Row-level operations · **DONE** (2026-09-01)

Green on: `uv run tox` (lint · mypy · **1302 tests against the built wheel** · dist) and
`uv run pytest -m integration` (11 passed, against the real REST catalog + MinIO).
Test count went 1218 → 1302.

**Built:** `sql/rowlevel.py` — `DELETE`, `UPDATE` and `MERGE` on the SQL surface, which
is the only surface the reference has for them. `Session.sql()` routes all three;
`plan/pushdown.py` grew `scope_predicate` and `is_exactly_translatable`; `writer.py`'s
`_commit` became the shared `commit_with_retry`. 72 tests in
`tests/fixture/test_rowlevel.py`, 12 more across the two unit files.

Every operation is **copy-on-write**, as decision 11 fixed. `MERGE` covers the full
Spark grammar: `WHEN MATCHED [AND c] THEN UPDATE SET ... / SET * / DELETE`, `WHEN NOT
MATCHED [AND c] THEN INSERT (cols) VALUES (...) / VALUES (...) / INSERT *`, and `WHEN
NOT MATCHED BY SOURCE [AND c] THEN UPDATE / DELETE`, in clause order, first match wins.

### The whole phase is one predicate written twice

A row-level operation is a `SELECT` whose result *is* the new contents of the rows it
touches, committed as `Table.overwrite(rows, overwrite_filter=P)`. PyIceberg deletes the
rows `P` matches and appends `rows`, so `P` and the `WHERE` that produced `rows` must
select **exactly** the same rows:

| If `P` is | Then |
|---|---|
| wider than the `WHERE` | rows are deleted that were never written back — **data loss** |
| narrower | rows survive the delete *and* arrive in the append — **duplication** |

Phase 2's pushdown has no such exposure, and that is the thing worth carrying: there the
SQL re-applies the filter, so an over-wide `P` costs I/O and nothing else. The whole of
`translate_predicate` was written under that licence.

So this phase added a **second, stricter gate** rather than tightening the first.
`is_exactly_translatable` is a whitelist of node shapes whose translation has been read
and found row-for-row exact; `LIKE` is the deliberate omission, since `StartsWith` is a
fine pruning approximation and a bad deletion. And the trick that makes it safe is that
both languages come from **one** set of sqlglot nodes: `scope_predicate` returns the
PyIceberg expression *and* the nodes it was built from, and those nodes are what goes
into the generated `WHERE`. There is no second translation to drift. A conjunct that
fails the gate is dropped from both at once, which only ever widens the scope — more
rows read and written back untouched, never a different answer.

`TestScopeIsExact` is the class that would notice either failure: every case mixes a
translatable conjunct with an untranslatable one, and asserts on the rows *outside* the
scope, which are the ones a wrong predicate takes away or duplicates.

### Two defects found by building it, both silent

**1. A marker column cannot survive subquery merging — and the failure empties the table.**

The obvious shape for `MERGE` is one `LEFT JOIN` with a column saying whether the join
found a source row, since every real column can be NULL on its own account:

```sql
LEFT JOIN (SELECT s.*, TRUE AS __matched FROM src AS s) AS s ON ...
```

`merge_subqueries` flattens that, and `__matched` becomes the literal `TRUE` in the outer
scope — where it no longer means "the join matched" but "there is a row here". Every
target row then looks matched, the `WHEN MATCHED ... DELETE` branch fires for all of
them, and the statement **empties the table and reports success**. It was caught by a
by-source test returning `[]`.

A window-function marker survives the merge, and that is exactly the wrong fix: it works
by accident of which subqueries sqlglot declines to flatten. So the merge was
restructured into three queries whose matchedness is a *predicate* — an inner join, where
every row is matched by construction, and `NOT EXISTS`, which no rewrite can turn into a
constant. It costs one more scan of the target and buys a property that does not depend
on the optimizer's rule list.

| Query | Rows |
|---|---|
| matched | target rows joining a source row, after the first `WHEN MATCHED` that fires |
| unmatched | target rows joining none, after the first `WHEN NOT MATCHED BY SOURCE` |
| inserted | source rows joining no target row, expanded by `WHEN NOT MATCHED` |

**2. sqlglot's `simplify` gives the wrong answer for `CASE ... WHEN TRUE`.**

A `CASE` whose always-true branch is not the first one folds to that branch's value, and
every branch before it is discarded:

```
CASE WHEN a = 1 THEN 'one' WHEN a <= 2 THEN 'two' WHEN TRUE THEN 'rest' END  ->  'rest'
```

DuckDB answers `'one'`. This is a **live wrong-answer path on the read side too** — any
hand-written `Session.sql()` carrying that shape was affected, and nothing raised. Found
because the merge's clause chains generate exactly it: an unconditional `WHEN MATCHED`
is a `WHEN TRUE` branch.

Fixed in two places. `optimize_plan` now normalises the shape away before the rules run —
an always-true branch becomes the `ELSE`, and what follows it is dropped, which no
reachable row can tell apart — so `simplify` never meets the case it mishandles. And the
generator no longer produces it: an unconditional clause *is* the `ELSE`, which is the
honest reading anyway. `TestAlwaysTrueCaseBranches` pins the first, and would fail if a
later sqlglot fixes it and the normalisation is removed carelessly.

### Where the file-scope minimisation comes from

| Statement | Scope |
|---|---|
| `DELETE` with a fully exact predicate | none needed — PyIceberg's own `delete(P)`, which can drop a wholly-matching file without reading it |
| `DELETE` / `UPDATE`, otherwise | the exactly-translatable conjuncts of the `WHERE` |
| `MERGE`, no by-source clause | the `ON`'s target-only conjuncts, **plus** an `IN` list of the distinct join keys the source actually holds |
| `MERGE` with a by-source clause | the whole table — there is no predicate for "everything the source does not name" |
| `MERGE`, `WHEN NOT MATCHED` only | no rewrite at all: it is an `append`, and one snapshot |

The `IN`-list narrowing is the one that matters at scale — it is the difference between
merging ten rows into a 41M-row table and rewriting the table. It covers `integer`,
`long`, `string` and `date` keys, and gives up past 1000 distinct values; a float,
decimal or timestamp key is left alone rather than reasoned about, because the literal is
the thing that has to mean the same in both languages. Giving up always widens the scope,
never changes the answer. `TestMergeScope` measures it on the partitioned fixture: a
one-key merge leaves two of the three partition files untouched.

### Concurrency

PLAN.md asked for optimistic commit with conflict detection and bounded retry. The
window that matters is between the **read** and the commit, and PyIceberg's own
`CommitFailedException` does not cover it: an `overwrite` computed from stale rows is a
perfectly valid commit that happens to erase someone else's. So the table is re-read at
the start of every attempt, its snapshot id is checked immediately before committing, and
a statement whose table moved is planned again from scratch — re-run, not replayed, since
the rows to write back are a function of the rows that were there. Four attempts, then
`QueryExecutionException`. `TestConcurrency` opens the window by hand with an interloping
append, and asserts the replan happened *and* that the interloper's row survived.

### Refused rather than half-done

`DELETE ... USING` and `UPDATE ... FROM` (use a subquery), and `UPDATE t SET s.field = v`
— nested-field assignment needs the schema-aware `withField` machinery Phase 6 built for
the DataFrame surface, reachable from SQL only once Phase 9 has done the same for DDL.
Each names what to do instead. Telling `SET t.vendor` from `SET person.name` is not a
question the parse tree can answer — sqlglot spells both as two parts — so it is decided
by asking the table whether the first part is one of its columns.

### Carried forward

- **PyIceberg's native `delete` handles NULL correctly.** Checked rather than assumed:
  `DELETE ... WHERE amount > 20` keeps the NULL-amount row on both the native and the
  rewrite path, and `test_the_rewrite_path_agrees_with_the_native_one` runs the two
  side by side on the same data.
- **`_invalidate_source` again.** Phase 7's note held: every attempt drops the cached
  source before reading, which is also what makes the snapshot it validates the snapshot
  it read.
- Merge-on-read is still Phase 13, and nothing here got closer to it — but nothing here
  got in its way either: the delete-file path would replace the commit, not the planning.

---

## Phase 9 — Schema, DDL, snapshots · **DONE** (2026-09-01)

Green on: `uv run tox` (lint · mypy · **1426 tests against the built wheel** · dist) and
`uv run pytest -m integration` (11 passed, against the real REST catalog + MinIO).
Test count went 1302 → 1426.

**Built:** `sql/catalog.py` (`session.catalog`), `sql/ddl.py` (`CREATE` / `DROP` /
`ALTER`), `sql/reader.py` (`session.read`), time travel through the source key, Iceberg's
metadata tables, and `mergeSchema` on write. 119 tests across
`tests/fixture/test_catalog_api.py`, `test_ddl.py` and `test_time_travel.py`, plus five
more in `test_pushdown.py` for the anti-join fix below.

| Piece | What |
|---|---|
| `session.catalog` | `listCatalogs` / `listDatabases` / `listTables` / `listColumns` / `listFunctions`, the `*exists` and `get*` pairs, `createTable`, `dropTable`, `setCurrentDatabase`, `refreshTable`, `clearCache` |
| SQL DDL | `CREATE [OR REPLACE] TABLE`, `IF NOT EXISTS`, column lists with `NOT NULL`, `USING iceberg`, `PARTITIONED BY` with transforms, `TBLPROPERTIES`, `COMMENT`, CTAS, `DROP TABLE`, `CREATE`/`DROP NAMESPACE\|DATABASE\|SCHEMA` |
| `ALTER TABLE` | `ADD`/`DROP`/`RENAME COLUMN`, `ALTER COLUMN` (comment, `DROP NOT NULL`), `RENAME TO`, `SET`/`UNSET TBLPROPERTIES` |
| Evolution | `ADD`/`DROP`/`REPLACE PARTITION FIELD`, `WRITE ORDERED BY`, `WRITE UNORDERED`, and `mergeSchema` on write |
| Time travel | `VERSION AS OF` / `TIMESTAMP AS OF`, and `session.read.option("snapshot-id"\|"as-of-timestamp")` |
| Metadata tables | `t.snapshots`, `.history`, `.files`, `.manifests`, `.partitions`, `.refs` and the rest of PyIceberg's `inspect` |

### Three silent wrong answers, two of them older than this phase

This is the phase that found them, not the phase that caused them. All three are in
FINDINGS.md with the full account.

**1. A source view name was reused, so a table was described as another one**
(FINDINGS §1.9, reachable since Phase 7). Source view names were numbered
`icetl_src_{len(self._sources)}` — from how many sources are *cached*, a count that goes
back **down** when `_invalidate_source` removes one. The next source took a name an
earlier one had used, the analyser registers a view name only once, and the reused name
kept the earlier table's schema. Read A, write to A, read B, and `B.columns` returned A's.

Phases 7 and 8 could reach it and never showed it: the *snapshot* changed but the
*schema* did not, so the stale view had the right columns. A schema change is what makes
it visible, which is why it waited for this phase. View names now come from the session's
monotonic counter, and invalidation unregisters the view.

**2. An anti-join returned every row instead of none** (FINDINGS §1.10, reachable since
Phase 4). `LEFT JOIN b … WHERE b.id IS NULL` selects the rows where the join found
nothing. The conjunct was pushed into `b`'s Iceberg scan like any other, PyIceberg pruned
every file — no data file holds a NULL id — and every left row survived.

This is **the one case where pruning changed the answer**, and it is worth being precise
about why §3.2's invariant did not cover it. "The pushed filter is always kept in the
SQL" holds for the filter; the rows it prunes were needed so the join would *not* match.
A row of NULLs on the padded side is manufactured by the join, not read from the table.

`null_padded_aliases` now finds the aliases an outer join can fill with NULLs, and only
conjuncts that `is_null_rejecting` — that a row of NULLs could not satisfy — are pushed
into them. `WHERE b.date = X` still prunes to one file; only `IS NULL` and its
kin are held back, so the fix costs nothing on the ordinary outer join.

**3. A column added by `ALTER TABLE` made the table unreadable** (FINDINGS §1.11). The
machinery existed and the gate did not: `ColumnAlias(stored=None)` has projected a typed
NULL since Phase 2, but the reconciliation path was entered only when a column had been
**renamed**, and an added column has one name and no history. `read_parquet` cannot
produce a column no file has, so it raised. `_late_columns` reads the schema *history* —
a field id missing from any earlier schema is one some file can predate — and enters the
same path, still O(schemas) and still opening no footers on a table that never changed.

### And one loud failure that had made the writer unusable for real data

`df.write.saveAsTable(...)` could not create a table from a frame carrying a **timestamp**
column, and had not been able to since Phase 7. DuckDB stamps a `TIMESTAMP WITH TIME ZONE`
with the *session's own* zone — `timestamp[us, tz=Asia/Calcutta]` here — and Iceberg's
`timestamptz` is UTC by definition, so PyIceberg refuses anything else outright. No `fx.*`
fixture had a timestamp column and the real `nyc` table is only ever read, so nothing
reached it. `writer.iceberg_ready` now relabels the type on the way in — zone to UTC,
nanoseconds to microseconds — which moves no value, because both sides are instants.

### Why a created table is not built by the writer

`CREATE TABLE t (id BIGINT NOT NULL)` says something DuckDB cannot: the analysed schema
calls every field nullable, which is the Phase 7 divergence. Handing the frame to
`saveAsTable` would have dropped the constraint silently.

So the DataType goes through the **existing** `createDataFrame` path to an Arrow schema —
no second type mapping to keep in step — and the declared columns are marked non-nullable
before PyIceberg is asked. `TestNullability` pins both halves, including the contrast: a
table created by a *write* is still all-optional, and that is still correct.

### Time travel is a property of the source key

`VERSION AS OF` is folded into the key a table reference resolves under, so the same table
at two snapshots is two sources that join to each other:

```sql
SELECT o.id FROM t VERSION AS OF 8271497619288662701 AS o
LEFT JOIN t AS n ON o.id = n.id WHERE n.id IS NULL   -- the rows a delete removed
```

Nothing downstream needed a flag threaded through it, and `session.read.option(...)`
builds the same node the SQL does, so the two surfaces cannot drift (P1). The *schema*
stays the current one, which is what PyIceberg's own scan does and the reference does not
— recorded in `divergence.md`. And a snapshot is history, so `assert_no_version` refuses
DELETE, UPDATE, MERGE and DDL against one.

### sqlglot parses most of the DDL, and says so when it does not

`ALTER TABLE t DROP COLUMN a` (singular), `UNSET TBLPROPERTIES`, and the whole of
Iceberg's Spark SQL extensions come back as `exp.Command` — sqlglot's explicit "I did not
understand this, here is the text". `run_alter_command` reads those few forms itself, a
deliberately small grammar with everything outside it refused by name.

`DROP PARTITION FIELD` is the exception: it gets far enough into sqlglot's own
`DROP PARTITION` rule to fail outright rather than fall back, so `Session.sql()` offers
the text to that grammar before letting the parse error stand.

### Refused rather than half-done

| | Why |
|---|---|
| `ALTER COLUMN … TYPE` | Iceberg allows only widening promotions and PyIceberg 0.11 exposes no type update; doing it here would rewrite every file |
| `ADD COLUMN … NOT NULL` | existing rows would have no value for it |
| `catalog.cacheTable` | caching here is per-frame and eager; a name-level cache would have to shadow the table for both surfaces and go stale behind a write |
| `CREATE TABLE … LOCATION` | the catalog owns the location |
| global temporary views | there is one session, so there is no one to share with |

### Carried forward

- **A monotonic counter, not a collection's size.** §1.9's root cause. `_materialize` had
  always used `self._counter` and was never affected; `_source_for` did not.
- **Pruning is safe *except* under an outer join.** §1.10 is the only exception found so
  far, and `is_null_rejecting` is where the next one would go.
- **An inner join's `WHERE` conjunct is folded into `ON` and stops pruning** — measured,
  a gap rather than a bug, and Phase 10's. FINDINGS §3.5.

---

## Decision 16 — SQL-surface function resolution deferred to Phase 15 (2026-08-30)

**P1 holds for the plan, not yet for the function library.** Relational structure and the
conformance rules in `sql/conformance.py` are shared — that pass is a tree walk both
surfaces cross. But a function whose reference behaviour comes from *composition* in
`sql/functions.py` (`rint` picking the even neighbour on a tie, `weekday` shifting Monday
to 0, `overlay` built from `substring`) exists only on the `F.*` path. `Session.sql()`
hands the bare name to DuckDB.

Measured over a 17-case sample of the 273-name surface, re-verified after the decision-15
rename:

| Class | Count | Names | Danger |
|---|---|---|---|
| **Silently different** | 2 | `weekday`, `dayofweek` — off by one | **wrong answers, nothing raises** |
| Missing in SQL | 9 | `rint` (×2 cases), `log1p`, `expm1`, `find_in_set`, `octet_length`, `overlay`, `width_bucket`, `regexp_substr` | raises — loud, safe |
| Agree, both wrong | 1 | `size(NULL)` → `NULL`, reference says `-1` | a plain conformance bug |
| Agree | 5 | `array_size`, `split_part`, `next_day`, `element_at`, `equal_null` | — |

**Why the suite did not catch it.** Phase 3's tests exercise the `F.*` surface, which is
where the compositions live. 821 tests pass with the gap present. It was found by running
`notebooks/01_read_real_table.ipynb` section 8, which probes both surfaces side by side.

**No guard, unlike decision 11.** Detecting the divergence means evaluating a name on both
surfaces and comparing — which is the test, which is the fix. There is nothing cheaper to
assert, so `compat/divergence.md` carries the warning instead.

**Interim exposure:** `weekday` and `dayofweek` only. Both surfaces answer, neither raises,
and the SQL surface carries DuckDB's week numbering — an off-by-one weekday is invisible in
a result set. **Prefer `F.*` over `Session.sql()` for date-part work.** The other nine
raise, so they cannot mislead.

**Phase 15 holds the work:** route SQL function names through the `F.*` table before the
optimizer, fix `size(NULL)`, and — the durable part — assert over all of `F.__all__` that
both surfaces agree or neither resolves. The sample found the gap; only an exhaustive test
keeps it closed.

---

## Carry-over notes

Things found during earlier phases that later phases need. [FINDINGS.md](FINDINGS.md) is the
fuller register: every dependency bug and wrong-answer trap found so far, grouped by kind,
with the rule each one taught.

| # | Note | Affects |
|---|---|---|
| 1 | Windows: PyIceberg needs `file://C:/x`, DuckDB rejects it. Handled by `icetl.paths`; don't bypass it. | any new file-reading code |
| ~~6~~ | ~~`pandas>=2.2` resolves to pandas 3.x, whose default string dtype differs from Spark's `toPandas()` `object` dtype~~ — **closed in Phase 3**: `exec/result.py` rebuilds string columns from `to_pylist()`, correcting the dtype and the null sentinel together. | — |
| 12 | sqlglot wraps some typed nodes (`Greatest`, `Least`, `ConcatWs`) in a NULL-propagating `CASE` for DuckDB, and renders `Log`'s operands reversed. Prefer a plain call and a value-level test over assuming a typed node is conformant. | Phase 3 onward |
| ~~9~~ | ~~Golden conformance files generated from real PySpark~~ — **closed by decision 13**: no PySpark at any stage. Cases are written from Spark's published behaviour with a citation each. | — |
| ~~10~~ | ~~A set operation whose branches need their output names restored loses **all** pushdown~~ — **closed in Phase 4**: `_naming_branch` re-aliases the leftmost branch, so a set operation over an unaliased aggregate now prunes like any other query. Measured before and after in the Phase 4 section. | — |
| 11 | Two references to one table merge to a single scan (union of columns, OR of predicates). Correct, but a self-join with disjoint filters prunes less than it could. | Phase 4 |
| 13 | Functions built by *composition* in `sql/functions.py` are reachable only through `F.*`; `Session.sql()` hands the bare name to DuckDB. `weekday`/`dayofweek` answer **silently differently**; 8 more raise. Deferred by decision 16. | **Phase 15** |
| 14 | `size(NULL)` gives `NULL` on both surfaces; the reference says `-1`. `array_size`'s own docstring says the two must differ, but `size` is `sg.ArraySize`, which makes them aliases. | **Phase 15** |
| 15 | `explain()`'s `bytes_scanned` is the *selected files'* size, not bytes read — it ignores column pruning, so a narrow query looks far more expensive than it is. | Phase 10 |
| 16 | An unfiltered `count(*)` reads parquet footers when Iceberg's manifests already hold the row count (~2× on a 357-file table, and it grows with file count). | Phase 10 |
| 17 | A generated plan may not carry meaning in a **constant** column: `merge_subqueries` flattens the subquery and the constant folds, so `TRUE AS __matched` past a LEFT JOIN stops meaning “the join matched”. Express it as a predicate (`EXISTS`, or an inner join) instead. | any phase generating SQL |

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
- [x] Committed as `92efdb8` — see "Committed" at the top.

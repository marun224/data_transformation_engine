# Reference ↔ DuckDB divergences

## What "the reference" means

icetl is not Spark and does not present a Spark API. But its semantics are not invented
either: where DuckDB's behaviour is a defensible choice rather than the only one, this
library follows **Apache Spark 3.5.0** — referred to throughout as *the reference*, *the
reference engine* or *the reference semantics*, and exposed in code as
`REFERENCE_SEMANTICS_VERSION`.

That choice is about having *a* specification rather than deciding each edge case ad
hoc. Nothing here executes on Spark, links against it, or requires it installed.

One related survivor is worth naming so it does not read as an oversight: sqlglot's SQL
grammar for parsing and rendering is its `"spark"` dialect, bound to the constant
`SQL_DIALECT` in `icetl/compat/__init__.py`. That string is an argument value in
sqlglot's API naming a *grammar* — changing it would change the SQL accepted, which is a
behaviour change rather than a rename.

## The rules

Where the reference and DuckDB disagree, the reference is the spec (P5). Each row below
is either a **translation rule** we implement, or a **documented divergence** we do not
emulate and say so loudly. Silent behavioural drift is a bug.

This file is written as the rules land, not afterwards. Anything marked *planned* has
no code behind it yet.

**How these were established.** Every claim about DuckDB below was executed and
observed. Every claim about the *reference* is taken from Spark 3.5's published
behaviour and cited, **not** verified against a running Spark — decision 13 rules out a
JVM at any stage. So a case where its real behaviour differs from its documentation
would show up here as a confident ✅. The exposure is the undocumented corners: NULL
propagation on unusual inputs, empty input, overflow, and decimal promotion.

| Status | Meaning |
|---|---|
| ✅ | Translation rule implemented and covered by a conformance case |
| 📋 | Planned — the phase that will deliver it is named |
| ⚠️ | Documented divergence: we do **not** match the reference, deliberately |

## Ordering and nulls

| Behaviour | Reference | DuckDB | Rule | Status |
|---|---|---|---|---|
| `ORDER BY x ASC` nulls | first | **last** | always emit explicit `NULLS FIRST` | ✅ |
| `ORDER BY x DESC` nulls | last | **last** | emit explicit `NULLS LAST` anyway | ✅ |

> PLAN.md §3.5 records DuckDB as nulls-*first* on `DESC`. That was true of an older
> release; DuckDB 1.5.5 is nulls-last in **both** directions, so `ASC` is the only
> real divergence. `DESC` is still made explicit, because "happens to agree" is a
> property of this release rather than a promise.

## Casting and arithmetic

| Behaviour | Reference | DuckDB | Rule | Status |
|---|---|---|---|---|
| `CAST('abc' AS INT)` | `NULL` (non-ANSI) | error | `TRY_CAST` by default; `spark.sql.ansi.enabled=true` opts into strict. An explicit `try_cast(...)` stays lenient in both modes | ✅ |
| `1/0` | `NULL` | `inf` | guarded division. sqlglot's the reference parser already emits the `NULLIF`, so both surfaces agree with no rule of ours | ✅ |
| `x % 0` | `NULL` | `NULL` | no rule needed | ✅ |
| Integer overflow | wraps | error | not emulated — we error. `ansi_mode` does not change this; DuckDB cannot be asked to wrap | ⚠️ |
| Decimal promotion | the reference's precision rules | DuckDB's, and division falls to `DOUBLE` | explicit cast per the reference's rules | 📋 **deferred to Phase 14** — see below |

### Decimal promotion — measured, not yet fixed

the reference derives the result type of decimal arithmetic from the operand types by a fixed
rule, then clamps precision to 38. DuckDB has its own rules. Both were executed; this
is what they give for `DECIMAL(10,2) op DECIMAL(10,2)`:

| Operation | the reference's rule | the reference result | DuckDB result | Same? |
|---|---|---|---|---|
| `a + b` | precision `max(s1,s2) + max(p1-s1, p2-s2) + 1`, scale `max(s1,s2)` | `DECIMAL(11,2)` | `DECIMAL(11,2)` | ✅ |
| `a * b` | precision `p1+p2+1`, scale `s1+s2` | `DECIMAL(21,4)` | `DECIMAL(18,4)` | ❌ precision |
| `a / b` | precision `p1-s1+s2+max(6, s1+s2+1)`, scale `max(6, s1+s2+1)` | `DECIMAL(16,6)` | `DOUBLE` | ❌ **type** |

Division is the one that matters: falling to `DOUBLE` loses exactness, which is the
whole reason a monetary column is a decimal. Addition already agrees.

**Deferred to Phase 14** (decision 14). The rule needs the *operand* types, and the
type of a sub-expression is only known after binding. The vehicle is sqlglot's
`annotate_types`, which the optimizer pipeline currently omits; adding it and then
emitting an explicit cast per the reference's formula is the shape of the work. Doing it by
guesswork -- casting without knowing the operand types -- would produce a confidently
wrong precision, which is worse than an honest divergence.

Note there is **no guard** for this one, unlike merge-on-read: detecting "this query
would have promoted differently" needs the same type information as fixing it. So
until Phase 14 lands, this table is the warning.

**What to watch:** a decimal column divided by a decimal column comes back `DOUBLE`
-- accurate to about 15 significant digits, and wrong only where exact decimal
semantics were the reason for the column. Addition and subtraction already agree.

## Operators and functions

| Behaviour | Reference | DuckDB | Rule | Status |
|---|---|---|---|---|
| `a <=> b` | null-safe equality | — | `IS NOT DISTINCT FROM`; `Column.eqNullSafe` builds the same node | ✅ |
| `concat(a, NULL)` | `NULL` | skips the NULL | generate `a \|\| b`, which propagates | ✅ |
| `concat_ws(s, a, NULL)` | skips the NULL | skips the NULL | plain call — **not** sqlglot's typed node, which wraps it in a NULL-propagating `CASE` | ✅ |
| `greatest` / `least` with NULL | ignore NULLs | ignore NULLs | plain call, for the same reason as `concat_ws` | ✅ |
| `split(s, p)` | `p` is a **regex** | `str_split` is literal | `str_split_regex`. The literal form returns the whole string as one element, silently | ✅ |
| `regexp_replace` | replaces every match | replaces the first | always pass the `g` flag | ✅ |
| `regexp_extract` no match | `""` | `NULL` | coalesce to `""` | ✅ |
| `dayofweek` | Sunday = 1 | Sunday = 0 | `+ 1` | ✅ |
| `date_add` / `date_sub` | returns `DATE` | returns `TIMESTAMP` | cast back to `DATE` | ✅ |
| `log(base, x)` | log to a base | same order | plain call — sqlglot's typed node renders the operands reversed | ✅ |
| `round(2.5)` | `3` (HALF_UP) | `3` | no rule needed | ✅ |
| `substring` / `locate` | 1-indexed | 1-indexed | no rule needed | ✅ |
| Array element access | 0-indexed | 1-indexed | `+ 1` on an integer index | ✅ |
| `date_format` patterns | Java (`yyyy-MM-dd`) | strftime (`%Y-%m-%d`) | **not translated** — two different pattern languages | ⚠️ |
| `first`/`last(ignorenulls=True)` | skips NULLs | no equivalent | refused rather than silently keeping them | ⚠️ |
| `explode(arr)` | row per element, drops empty | — | `UNNEST` | 📋 Phase 6 |
| `explode_outer(arr)` | keeps empty as NULL | — | `LEFT JOIN UNNEST` | 📋 Phase 6 |
| `df.join(o, "k")` | one `k` column | two | the name form emits `USING (k)`, which collapses it; the Column form emits `ON` and keeps both | ✅ |
| Duplicate column names after a join | two columns both named `id`; `df["id"]` is then ambiguous and raises | — | ours are disambiguated to `id` and `id_1`, so a later `select` resolves instead of raising | ⚠️ |
| `rint(2.5)` | `2.0` (HALF_EVEN) | `round` gives `3` | pick the even neighbour on an exact tie, defer to `round` otherwise | ✅ |
| `weekday` | Monday = 0 | `dayofweek` is Sunday = 0 | `(dayofweek + 6) % 7`, a *different* shift from `dayofweek`'s | ✅ |
| `next_day(d, day)` | strictly after `d` | — | `(target - dow + 7) % 7`, mapping a zero delta to 7 | ✅ |
| `array_size(NULL)` | `NULL` | `len` gives `NULL` | no rule needed — but `size(NULL)` is `-1`, so the two are not aliases | ✅ |
| `get(arr, i)` | 0-indexed | 1-indexed | `+ 1`; `element_at` stays 1-indexed, as the reference has it both ways | ✅ |
| `array_except` | de-duplicates | `list_filter` does not | wrap in `list_distinct` | ✅ |
| `split_part` out of range | `""` | `NULL` | coalesce to `""` | ✅ |
| `regexp_substr` no match | `NULL` | `regexp_extract` gives `""` | guard with `regexp_matches`, so an empty *match* stays distinguishable from no match | ✅ |
| `find_in_set(x, s)` where `x` holds a comma | `0` | would match a run of fields | explicit `CASE` | ✅ |
| `shiftrightunsigned` | Java `>>>` | no unsigned shift; `UBIGINT` rejects a negative | widen to `HUGEINT`, add 2^64, shift, narrow back | ✅ |
| `octet_length` | bytes | `length` counts characters | `strlen`, which is DuckDB's byte count | ✅ |
| `1.0 / 0.0` in the optimizer | `NULL` | rule raised `decimal.DivisionByZero` | widen `optimize_plan`'s guard to `ArithmeticError` — see below | ✅ |
| `date_part('DOW', d)` | Sunday = 1 | Sunday = 0 | **refused**, naming `dayofweek`/`weekday` instead of translating | ⚠️ |
| `xxhash64` / `hash` | specific seeded algorithms | DuckDB's own | values are stable within a query and do **not** match the reference's | ⚠️ |
| `log1p` / `expm1` | `Math.log1p` / `Math.expm1` | absent | computed as `ln(1 + x)` / `exp(x) - 1`, which drift in the last digits near zero | ⚠️ |
| `rand(seed)` / `randn(seed)` | reproducible | seeds per *connection*, not per expression | **refused** — accepting it would silently lose the reproducibility the argument is for | ⚠️ |
| `try_add` / `try_subtract` / `try_multiply` | `NULL` on overflow | raises, and has no `TRY` expression | not implemented; see below | 📋 |
| `crc32`, `soundex`, `typeof`, `input_file_name` | — | no DuckDB equivalent whose output matches | not implemented; see below | 📋 |
| `to_utc_timestamp` / `from_utc_timestamp` / `current_timezone` | — | needs the ICU extension | not implemented; see below | 📋 |
| `monotonically_increasing_id` | — | needs a window function | 📋 Phase 5, with `Column.over` |

### Functions left unimplemented, and why

Each of these has a DuckDB spelling that *looks* close enough to use. Writing one down
here was cheaper than shipping a function that returns a plausible wrong answer.

| Function | What stops it |
|---|---|
| `try_add`, `try_subtract`, `try_multiply` | Their whole contract is NULL on overflow. DuckDB raises on overflow and has no `TRY(...)` expression to catch it — only `TRY_CAST`. Computing in `HUGEINT` and range-checking works for integers and silently truncates a `DOUBLE`, so it would fix the rare case by breaking the common one. `try_divide` **is** implemented: division by zero is detectable without any of that. |
| `crc32` | A specific checksum polynomial. DuckDB has no `crc32`, and no other hash is a substitute for a value that is usually compared against one some other system computed. |
| `soundex` | Not in DuckDB. A phonetic key is only useful if it is *the same* phonetic key. |
| `typeof` | DuckDB answers `INTEGER`/`VARCHAR`; the reference answers `int`/`string`. A mapping table would cover the common types and quietly mislabel the rest. |
| `input_file_name` | DuckDB exposes a filename only as a `read_parquet` option, and the scan planner builds that relation itself. Reachable, but it is scan-planner work rather than a function. |
| `to_utc_timestamp`, `from_utc_timestamp`, `current_timezone` | Need DuckDB's ICU extension, which is a new load-time dependency — and `exec/engine.py` deliberately loads extensions lazily and offline-safely. Worth doing as a decision, not as a side effect of a function. |
| `array_insert`, `shuffle` | No DuckDB primitive. `array_insert` also pads with NULLs and accepts a negative position, which is enough behaviour to want a fixture rather than a one-liner. |
| `monotonically_increasing_id` | Needs `row_number() over ()`. Phase 5 owns windows, and `Column.over` already raises for the same reason. |

### The optimizer's rule guard and `ArithmeticError` (2026-08-30)

`SELECT 1.0 / 0.0` crashed, on **both** surfaces, with `decimal.DivisionByZero` raised
from inside sqlglot's `simplify` rule. the reference's answer is NULL.

The cause is worth keeping: `optimize_plan` catches `OptimizeError`, `KeyError`,
`ValueError` and `TypeError`, which is every way a rule was known to fail. But
`simplify` *constant-folds literal arithmetic*, and arithmetic has its own exception
tree — `decimal.DivisionByZero` is an `ArithmeticError`, so it escaped the guard and
Phase 2's "a rule that fails costs only that rule" did not hold.

The integer spelling `1 / 0` — which the conformance suite did test, on both surfaces —
never reached it: `simplify` declines to fold integer division at all, calling it unsafe
across engines. So the bug needed a *decimal* literal to appear, and the obvious test
was the one shape that could not find it.

`ArithmeticError` is now in the guard, which also covers `OverflowError` and
`decimal.InvalidOperation` arriving the same way. Degrading is safe here because the
the reference parser's own `NULLIF` guard is still in the tree, so the unoptimized plan returns
the reference's NULL.

## Confirmed as already matching

Verified, so no rule is needed — but each keeps a conformance case so a future DuckDB
release cannot drift without the suite noticing.

| Behaviour | Status |
|---|---|
| Window default frame (`RANGE UNBOUNDED PRECEDING → CURRENT ROW`) | 📋 assert in Phase 5 |
| Aggregate output column names (`sum(total_amount)`) | 📋 assert in Phase 3 (remaining) |
| Identifier case: insensitive resolve, preserved spelling | 📋 assert in Phase 3 |
| `count()` over zero rows returns `0` | 📋 assert in Phase 4 |

## Reading Iceberg through DuckDB

Not the reference-vs-DuckDB but engine-vs-format: places where handing DuckDB an Iceberg
table's files would quietly answer a different question than Iceberg does. Each is a
silent wrong answer if unhandled, which is why each has a fixture rather than a note.

| Behaviour | What goes wrong | Rule | Status |
|---|---|---|---|
| Renamed column | `read_parquet(union_by_name = true)` matches on *name*; files written before a rename read back as NULL, with no error | detect from the schema history (O(schemas)), group files by their stored names from the parquet footers, alias each group back by field-id | ✅ |
| Added column | files written before the column existed have no such column | `union_by_name` fills NULL, which is what Iceberg does — fast path | ✅ |
| Merge-on-read deletes | `read_parquet` cannot honour delete files, so deleted rows would come back with no error | **deferred to Phase 12** (decision 11 — the tables in scope are copy-on-write). Asserted at scan time, so such a table is refused rather than answered wrongly | 📋 Phase 12 |
| Hive partitioning | DuckDB auto-detects `key=value` directories, which an Iceberg warehouse is full of, and synthesises the column from the *path* with a type-cast — a `string` partition column comes back `DATE`. Iceberg's directory holds the *transformed* value, not the column's | `hive_partitioning => false`, always | ✅ |
| Type promotion (int→long) | branch types must agree across a UNION | explicit `CAST` in the per-group projection | ✅ |
| Stats-based file pruning | PyIceberg prunes on min/max, which is approximate: a straddling file is kept and its non-matching rows come back | the pushed filter is **always** left in the generated SQL as well | ✅ |
| Bare date vs `timestamp` | SQL, the reference and DuckDB all read `ts >= '2024-01-01'` as midnight that day. PyIceberg demands full ISO-8601 and *raises* from inside `plan_files()` rather than declining | widen a date-only literal to `T00:00:00` for `timestamp` columns, then bind-validate every predicate before use, so one PyIceberg dislikes costs only its own conjunct | ✅ |
| Bare date vs `timestamptz` | a bare date carries no zone; DuckDB resolves it against the session's, so assuming UTC could move the boundary and prune away wanted rows | not widened — the conjunct stays in SQL and prunes nothing | ⚠️ |
| `ORDER BY <output alias>` | `qualify` leaves such a reference unqualified because it names the projection, not the table. Read as an unattributable column it disabled projection pushdown for the whole query | unqualified names matching an output alias are recognised as projection references; ones that also name a real column are included in the scan (a safe superset) | ✅ |

## Optimizer-level divergences

| Behaviour | Note | Status |
|---|---|---|
| Unaliased projection names | sqlglot's `qualify` renames `sum(amount)` to `_col_0`. The optimized plan is adopted only if its output columns can be re-aliased to the names analysis already computed; otherwise the unoptimized plan runs instead | ✅ |
| Identifier case in the optimizer | the optimizer runs in DuckDB's dialect, which is case-insensitive, so `VendorID` is normalised internally. Output names are restored from the analysed schema, so the user-visible spelling is unaffected | ✅ |
| Nullability | the analysed schema comes from DuckDB, which has no non-nullable expression, so every field reports `nullable = true` even where Iceberg marked it required | ⚠️ |
| `WHERE b.c IS NULL` over an outer join's null-padded side | the conjunct is **not** pushed to Iceberg. A row of NULLs there is manufactured by the join, not read from the table, so pushing it pruned away exactly the files that make an anti-join right and the query returned every left row instead of none. Only conjuncts a row of NULLs could not satisfy are pushed into a null-padded table; `WHERE b.date = X` still prunes. The one exception to “pruning is only ever a way to read less” — FINDINGS.md §1.10 | ✅ |
| `CASE ... WHEN TRUE THEN v ...` | sqlglot 30.17's `simplify` folds a `CASE` whose always-true branch is **not the first** down to that branch's value, discarding every branch before it — `CASE WHEN a = 1 THEN 'one' WHEN TRUE THEN 'rest' END` becomes `'rest'`. DuckDB answers `'one'`, so it is a silent wrong answer rather than a missed optimisation. The plan is normalised before the rules run — an always-true branch becomes the `ELSE` and what follows it is dropped, which no reachable row can tell apart — so `simplify` never meets the shape. Found in Phase 8 | ✅ |

## Single-node divergences — Phase 4

The reference is a distributed engine and icetl is not. Most of these are cases where a
method's whole purpose is to move data between machines, and there are no machines to
move it between; the methods exist so a script written against the reference runs
unaltered, rather than failing on an attribute that will never mean anything here.

| Behaviour | Reference | icetl | Status |
|---|---|---|---|
| `repartition`, `coalesce` | redistribute across partitions | **no-op**, returns the same frame | ⚠️ |
| `F.broadcast(df)` | hints a join to ship the frame to every executor | **no-op**, returns the frame | ⚠️ |
| `persist(storageLevel=...)` | picks memory/disk/serialised | refused — there is one storage level, a DuckDB temp table | ⚠️ |
| `cache()` / `persist()` | lazy; marks the frame and materialises at the next action, returning `self` | **eager**, and returns a **new** frame. `cached = df.cache()` is the spelling that works | ⚠️ |
| `sample(withReplacement=True)` | supported | **refused**. DuckDB draws each row at most once, and quietly returning a without-replacement sample to a caller who asked for the other is a wrong answer | ⚠️ |
| `df.stat.approxQuantile(relativeError>0)` | error bound is the argument | DuckDB's `approx_quantile` bound is fixed, so the answer may be *more* accurate than asked and never less — which is what the argument promises | ✅ |
| `df.stat.freqItems` | approximate (a sketch, because it counts across a cluster) | **exact** — one grouped scan per column. Strictly inside the same contract | ✅ |
| `F.grouping_id()` with no arguments | means "every grouping column" | refused; the columns must be named. Nothing in `functions.py` can see what the query groups by, and DuckDB has no such spelling either | ⚠️ |
| `catalog.dropTempView` | on `spark.catalog` | on the **session** (`session.dropTempView`) until `Session.catalog` lands in Phase 9 | ⚠️ |
| `sortWithinPartitions` | sorts inside each partition | `sort` — there is one partition, so sorting within it sorts the frame | ⚠️ |
| `range(numPartitions=...)` | splits the range across partitions | accepted and ignored | ⚠️ |
| `dropDuplicates(subset)` | which row survives is undefined | also undefined (`DISTINCT ON`). Rely on the keys being unique, not on which row carried them | ✅ |

`cache()` being eager is the one to know about. Nothing here mutates a plan in place, so
a lazy mark would have nowhere to live; running now and handing back a new frame says
the same thing without the mutation. The consequence is that `df.cache()` on its own
caches nothing you can reach.

## Set operations and grouping sets — Phase 4

Checked against DuckDB before building, and **agreeing on every point**, which is why
there is no conformance rule for any of it:

| Behaviour | Reference | DuckDB | Status |
|---|---|---|---|
| `union` keeps duplicates | yes (`union` is `UNION ALL`) | yes | ✅ |
| `EXCEPT ALL` subtracts multiplicities | 3 minus 1 leaves 2 | same | ✅ |
| `INTERSECT ALL` keeps the smaller | min(3, 2) is 2 | same | ✅ |
| `INTERSECT` / `EXCEPT` match NULL to NULL | yes, unlike `=` | same | ✅ |
| `GROUPING(col)` on a rolled-up key | `1` | `1` | ✅ |
| `GROUPING_ID(a, b)` bit order | most significant first | same | ✅ |

Two behaviours are **deliberately preserved traps**, not defects — the reference has
them, and hiding them would be a bigger surprise than keeping them:

| Behaviour | Note |
|---|---|
| `union` matches **by position**, not by name | Two frames with the same names in a different order union into nonsense without complaint. The type widening makes it quieter rather than louder: `bigint` over `string` settles on `string`, so an `id` of `1` comes back as `'1'`. `unionByName` is the safe spelling |
| A rolled-up key comes back as **NULL** | Indistinguishable from a real NULL in the data. A rollup over a column that already contains NULL produces two NULL rows meaning different things; `F.grouping` is the only thing that tells them apart |

Two implementation choices worth recording because they are invisible from the outside:

- **`pivot` compiles to conditional aggregation** (`sum(CASE WHEN k = 'x' THEN v END)`),
  not to DuckDB's own `PIVOT`, which is a statement rather than an expression and would
  not compose with the rest of a plan. The match is null-safe, so a NULL pivot key gets
  its own column named `null` rather than an empty one.
- **A set-operation branch is nested when it is itself a set operation.** DuckDB binds
  `INTERSECT` tighter than `UNION ALL`, so an inlined `a.union(b).intersect(c)` would
  evaluate `a UNION ALL (b INTERSECT c)` — a different query that raises nothing and
  answers wrongly.

## The write path — Phase 7

| Behaviour | Reference | icetl | Status |
|---|---|---|---|
| `df.write.save(path)` | writes files to a path | **refused** — data lives in Iceberg tables here, so `saveAsTable` is the only way in | ⚠️ |
| `format(...)` | many | `iceberg` only | ⚠️ |
| Nullability of a **created** table | inferred from the frame's schema | every field is optional, because the analysed schema comes from DuckDB, which has no non-nullable expression. Writing *into* an existing table is unaffected — PyIceberg validates against the table's own schema | ⚠️ |
| `insertInto` column matching | by position | by position | ✅ |
| `partitionBy` on an **existing** table | re-partitions | **ignored** — an existing table's partitioning is its own, and changing it on a write is a much larger act than a write. `Table.update_spec()` is the deliberate way | ⚠️ |
| `partitionOverwriteMode` default | `static`, which replaces every row | the same. The dangerous one is the default in both, so it is the one you have to *not* ask for | ✅ |

Two upstream limits, measured rather than assumed, each with a characterisation test
that will fail when it lifts:

| Limit | Detail |
|---|---|
| **No streaming writes** | PyIceberg 0.11.1's `Transaction.append` opens `if not isinstance(df, pa.Table): raise ValueError`, so batches and readers are refused and the whole result is materialised. Chunking into several appends would trade atomicity for it, which is the worse deal |
| **An overwrite is two snapshots** | A `delete` then an `append`, which is how Iceberg models replacing rows — so "one snapshot per write" holds only for appends. Both land in **one transaction**, so no reader sees the table mid-overwrite; the atomicity the goal was after does hold |

## Schema, DDL and snapshots — Phase 9

| Behaviour | Reference | icetl | Status |
|---|---|---|---|
| `CREATE TABLE t (c T NOT NULL)` | the column is required | required — the schema is built as Arrow and the column marked non-nullable before PyIceberg sees it, rather than going through the writer, which cannot say it | ✅ |
| A table created by a **write** | nullability inferred | every field optional, as recorded under Phase 7. `catalog.listColumns` reports the difference honestly | ⚠️ |
| `CREATE OR REPLACE TABLE` | keeps the table and adds a snapshot, so time travel survives the replace | **drops and rebuilds**, so the snapshot history goes with it. The data outcome is the same; the history is not | ⚠️ |
| `ALTER TABLE ... ALTER COLUMN c TYPE t` | widening promotions | **refused** — Iceberg allows only widening, and PyIceberg 0.11 exposes no type update, so doing it here would mean rewriting every file | ⚠️ |
| `ALTER TABLE ... ADD COLUMN c T NOT NULL` | refused | refused — existing rows would have no value for it | ✅ |
| `UPDATE t SET s.field = v` | rewrites the one struct field | **refused**; rebuild the struct with `named_struct(...)`. Needs the schema-aware `withField` machinery on the SQL surface | 📋 later |
| `catalog.cacheTable` / `uncacheTable` / `isCached` | caches by table name | **refused**, pointing at `frame.cache()`. Caching here is per-frame and eager; a name-level cache would have to shadow the table for both surfaces and go stale behind a write | ⚠️ |
| Global temporary views | shared across sessions | **refused** — there is one session, so there is no one to share with | ⚠️ |
| `catalog.recoverPartitions` | re-scans the filesystem for directories the metastore does not know about | **accepted and does nothing** — Iceberg tracks files in manifests, so there is no gap to recover. It checks the table exists and returns | ⚠️ |
| `listTables().tableType` | `MANAGED` / `EXTERNAL` / `VIEW` / `TEMPORARY` | `MANAGED` for a catalog table, `TEMPORARY` for a view. The catalog holds the metadata pointer and a `DROP` removes it, which is what `MANAGED` means | ⚠️ |
| `listColumns().isBucket` | Hive bucketing | always `false`. An Iceberg `bucket(n, c)` is a *partition* transform and is reported through `isPartition` | ⚠️ |
| `listFunctions()` | the catalog's functions | the `F.*` surface — which is **not** yet what `Session.sql()` resolves. See the surface divergences below; decision 16 | ⚠️ |
| A namespace whose name contains a dot | addressable | **not addressable** — nested Iceberg namespaces are joined with dots, so `nyc.raw` is two levels and a literal dot cannot be told apart | ⚠️ |
| Time travel and the **schema** | reads the snapshot's own schema | reads the **current** schema, which is what PyIceberg's own scan does. It differs only where the schema changed after the snapshot: a renamed column still resolves through the §3.4 reconciliation, an added one reads NULL, a dropped one is not selectable | ⚠️ |
| `VERSION AS OF n` | a snapshot id, for an Iceberg table | a snapshot id | ✅ |
| `TIMESTAMP AS OF t` | the newest snapshot at or before `t` | the same; a time before the first snapshot is refused rather than answered empty | ✅ |
| Writing to a snapshot | refused | refused for `DELETE`, `UPDATE`, `MERGE` and DDL. `INSERT INTO t VERSION AS OF ...` does not parse at all, so that one is closed earlier and by sqlglot | ✅ |
| Metadata tables (`t.snapshots`, `t.files`, …) | computed on demand | **materialised at plan time** — there are no data files to prune and no predicate to push down, so the rows are read when the frame is built. A frame therefore holds the metadata as it was then | ⚠️ |
| `mergeSchema` on write | adds the columns the frame has | the same, and **additive only**: a column the table has and the frame lacks is untouched and lands NULL | ✅ |

### Two notes on the implementation

**A timestamp is relabelled on the way in.** DuckDB stamps a `TIMESTAMP WITH TIME ZONE`
with the session's own zone and Iceberg's `timestamptz` is UTC by definition, so
`writer.iceberg_ready` retypes it — zone to UTC, nanoseconds to microseconds — before the
table is created or written to. It moves no value: both sides are instants. Without it,
creating a table from a frame carrying a timestamp column was impossible, and had been
since Phase 7 (FINDINGS.md §2.7).

**PyIceberg's schema mismatches are translated.** A required field given a NULL, a column
the table does not have, a type that will not fit — PyIceberg reports each as a bare
`ValueError`. `commit_with_retry` re-raises it as `AnalysisException`, keeping PyIceberg's
own message, so a caller catching this engine's hierarchy hears about it.

## Row-level operations — Phase 8

`DELETE`, `UPDATE` and `MERGE` are **SQL-surface only**, as they are in the reference
3.5 — there is no `df.delete()` to diverge from. All three are copy-on-write: the rows
in scope are recomputed and their files rewritten. Emitting delete files instead is
Phase 13, and decision 11 is why.

| Behaviour | Reference | icetl | Status |
|---|---|---|---|
| `DELETE FROM t WHERE c` with `c` NULL for a row | the row survives | the row survives — survivors are `NOT COALESCE(c, FALSE)`, not `NOT c` | ✅ |
| `UPDATE ... SET c = e WHERE p` with `p` NULL | the row is not updated | not updated — `CASE WHEN p THEN e ELSE c END` falls to `ELSE` on NULL | ✅ |
| Right-hand sides in a multi-column `SET` | all see the row's **old** values | the same — every assignment is a projection of one input row | ✅ |
| `DELETE ... USING` | supported | **refused**, with a pointer to a subquery in the `WHERE` | ⚠️ |
| `UPDATE ... FROM` | supported | **refused**, likewise | ⚠️ |
| `UPDATE t SET s.field = v` | rewrites the one struct field | **refused** — rebuild the struct with `named_struct(...)`. Doing it properly needs the schema-aware `withField` machinery on the SQL surface | 📋 Phase 9 |
| `MERGE` cardinality violation | raises when one target row matches several source rows | raises — checked **whenever the statement rewrites target rows**, which includes a merge whose only clauses are `WHEN NOT MATCHED BY SOURCE`. The reference documents the check for matched clauses; we have not measured whether it also fires there, so this may be the stricter of the two | ⚠️ |
| `WHEN NOT MATCHED THEN INSERT (a) VALUES (...)` | unnamed columns get NULL | the same | ✅ |
| `WHEN NOT MATCHED BY SOURCE ... UPDATE SET *` | refused — there is no source row | refused | ✅ |
| A source row matching no `WHEN NOT MATCHED` condition | inserts nothing | inserts nothing — not a row of NULLs | ✅ |
| Snapshots per statement | — | `DELETE` is **one**; `UPDATE` and a rewriting `MERGE` are **two** (a delete then an append) in **one commit**, for the reason recorded under Phase 7 | ⚠️ |

### Two things worth knowing before reading the code

**The predicate is generated in two languages and they must agree row for row.** The
commit is `Table.overwrite(rows, overwrite_filter=P)`: PyIceberg deletes the rows `P`
matches, then appends the rows the SQL kept. A `P` wider than the SQL's `WHERE` deletes
rows that are never written back; a narrower one leaves rows in place *and* appends them
again. Read pushdown has no such exposure — there the SQL re-applies the filter, so an
over-wide `P` costs only I/O — which is why `plan/pushdown.py` grew a second, stricter
gate for this path:

| | Read pushdown | Row-level writes |
|---|---|---|
| Requirement on `P` | a **superset** of the SQL's rows | **exactly** the SQL's rows |
| Gate | `translate_predicate` alone | `translate_predicate` **and** `is_exactly_translatable` |
| `LIKE 'a%'` | pushed as `StartsWith` | **not** pushed — the two spell escapes differently, and "close enough" is a data-loss bug here |
| A conjunct that fails the gate | stays in the SQL, prunes less | dropped from *both* forms at once, so the scope widens and the answer does not change |

Both forms come from one set of sqlglot nodes — `scope_predicate` returns the PyIceberg
expression together with the very nodes it was built from, and those nodes go into the
`SELECT`'s `WHERE`. They cannot drift because there is no second translation.

**A `MERGE`'s scope comes from the source's own keys.** With no `WHEN NOT MATCHED BY
SOURCE` clause, a target row can only match if its join key is one the source actually
holds, so the distinct source keys become an `IN` list and the rewrite touches a few
files instead of the table. It applies to `integer`, `long`, `string` and `date` keys
only — a float, decimal or timestamp key is left un-narrowed rather than reasoned about,
since its SQL literal and its PyIceberg literal are the two things that must agree — and
it gives up past 1000 distinct values, where binding the list costs more than the pruning
buys. Giving up always means a wider scope, never a different answer.

## Complex types — Phase 6

Most DuckDB spellings agree with the reference and needed no rule. The ones that did are
all on **empty input**, which is the case a test over populated data never reaches:

| Behaviour | Reference | DuckDB raw | Rule | Status |
|---|---|---|---|---|
| `exists(f)` over an empty list | `false` | `list_bool_or([])` is NULL | three-valued `CASE`: true / NULL when a NULL element is present / false | ✅ |
| `forall(f)` over an empty list | `true` (vacuously) | `list_bool_and([])` is NULL | the same shape, inverted | ✅ |
| `aggregate(col, zero, merge)` over an empty list | the zero | `list_reduce` has no initial value and starts from the first element | the zero is **prepended to the list**, which is the same fold | ✅ |
| `explode` of an empty or NULL collection | no row | `unnest` drops the row | correct as-is; `explode_outer` substitutes `[NULL]` to keep one | ✅ |
| `posexplode` position | 0-based | `generate_subscripts` is 1-based | one is subtracted | ✅ |

| Behaviour | Reference | icetl | Status |
|---|---|---|---|
| `map_concat` over differing value types | widens `map<string,int>` to meet `map<string,bigint>` | **refuses** — DuckDB will not merge them. Loud, so it is an error rather than a wrong answer; cast the narrower side | ⚠️ |
| `schema_of_json` type names | the reference's own (`BIGINT`, `STRING`) | DuckDB's (`UBIGINT`, `VARCHAR`). The *shape* agrees; the type words do not, so the result reads well and does not feed back into `from_json` | ⚠️ |
| `arrays_zip` field names | named after the input columns | the same, falling back to positions (`"0"`, `"1"`) when two inputs would claim one name. `list_zip` returns an *unnamed* struct, and two unnamed fields cannot be told apart on the way back into Arrow | ⚠️ |

Three implementation notes, invisible from the outside but worth recording:

- **`from_json` is not a `CAST`.** A cast refuses a JSON object carrying a key the
  target type has no room for; the reference ignores extra keys. DuckDB's `from_json`
  with a structure argument has the reference's behaviour, so that is what is generated.
- **`withField` rebuilds the struct.** DuckDB's `struct_insert` *refuses* a field name
  the struct already has rather than overwriting it, so add-or-replace cannot be one
  call. The struct is rebuilt from its field list instead — which is also why `withField`
  and `dropFields` both need the frame's schema.
- **sqlglot's `Bracket` treats its subscript as 0-based** and adds one for DuckDB, so a
  literal `p[1]` arrives as `p[2]` and the error names an index the query never
  mentioned. Positional struct access is spelled `struct_extract(p, 1)` instead. This is
  carry-over note 12's hazard — a typed node doing something reasonable and unexpected —
  in a new place.

## Window frames — Phase 5

PLAN.md named frame semantics as the likeliest place for the two engines to drift.
Probed before building, they **agree on every point**, so no conformance rule was needed:

| Behaviour | Reference | DuckDB | Status |
|---|---|---|---|
| Default frame with an ordering | `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` | same | ✅ |
| Default frame with no ordering | whole partition | same | ✅ |
| `rank` leaves gaps, `dense_rank` does not | 1,2,2,4 / 1,2,2,3 | same | ✅ |
| Ranking functions ignore the frame | yes | same | ✅ |
| `lag`/`lead` offset and default | supported | same | ✅ |
| `IGNORE NULLS` on value functions | supported | same | ✅ |
| Null placement inside `OVER (ORDER BY ...)` | nulls first ascending | nulls last | ✅ — the existing `_fix_null_ordering` pass covers it, because a window's ordering is made of the same `exp.Ordered` nodes |

Two behaviours are **SQL's, not divergences**, and are documented on the functions
because they surprise people rather than because they differ:

| Behaviour | Note |
|---|---|
| The default frame is `RANGE`, not `ROWS` | It includes every row tying with the current one, so a running total jumps over a tie rather than climbing through it. The two agree on any column without duplicates, which is what makes shipping the wrong one easy |
| `last_value` over the default frame returns the **current row** | The frame ends at the current row, so its last value is this one. `rowsBetween(unboundedPreceding, unboundedFollowing)` is the fix. `first_value` looks right only by coincidence — the frame's start really is the partition's start |

| Single-node behaviour | Reference | icetl | Status |
|---|---|---|---|
| `monotonically_increasing_id` | monotonic and unique, not consecutive (a partition number is encoded in the high bits) | `row_number() OVER () - 1`, so consecutive here. Relying on that relies on more than either engine promises | ⚠️ |

## Surface divergences — `Session.sql()` vs `F.*` ⚠️

**P1 does not currently hold for function names** (decision 16, deferred to Phase 15).
The conformance rules in `sql/conformance.py` are a tree pass both surfaces cross, so
casting, division and null ordering agree. But a function whose reference behaviour is
produced by *composition* in `sql/functions.py` exists only on the `F.*` path — a bare
name in `Session.sql()` is handed to DuckDB unchanged.

| Behaviour | `Session.sql()` | `F.*` | Reference | Status |
|---|---|---|---|---|
| `weekday(Monday)` | `1` | `0` | `0` | ⚠️ **silently different** |
| `dayofweek(Monday)` | `1` | `2` | `2` | ⚠️ **silently different** |
| `rint`, `log1p`, `expm1`, `find_in_set`, `octet_length`, `overlay`, `width_bucket`, `regexp_substr` | raises `AnalysisException` | correct | — | ⚠️ missing in SQL, but loud |
| `size(NULL)` | `NULL` | `NULL` | `-1` | ⚠️ wrong on **both** — a plain conformance bug |

`weekday` and `dayofweek` are the ones that can produce a wrong answer: both surfaces
return a value, neither raises, and the SQL surface carries DuckDB's week numbering. An
off-by-one weekday does not look like a defect in a result set. **Prefer `F.*` over
`Session.sql()` for date-part work** until Phase 15 lands.

The remaining eight raise, so they cannot silently mislead — they simply are not
available through SQL yet.

Measured over a 17-case sample of the 273-name surface, not the whole of it; the
exhaustive both-surfaces test is Phase 15's deliverable. `notebooks/01_read_real_table.ipynb`
section 8 runs the sample live.

## Environment-level divergences

Not reference-vs-DuckDB, but differences worth knowing at the environment level.

| Behaviour | Note | Status |
|---|---|---|
| `Session.version` | reports icetl's own version. `Session.reference_semantics` reports the release the rules are checked against (`3.5.0`). | ✅ |
| `getSqlState()` | always returns `None`; we do not model SQLSTATE codes | ⚠️ |
| `pandas` 3.x | the dependency floor `>=2.2` resolves to pandas 3.x, whose default string dtype differs from the `object` dtype `toPandas()` is expected to yield | ✅ `exec/result.py` rebuilds string columns from `to_pylist()`, which corrects the dtype and the `nan`-vs-`None` sentinel in one pass |

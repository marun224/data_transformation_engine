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

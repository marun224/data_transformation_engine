# Spark ↔ DuckDB divergences

Where Spark and DuckDB disagree, Spark is the spec (P5). Each row below is either a
**translation rule** we implement, or a **documented divergence** we do not emulate and
say so loudly. Silent behavioural drift is a bug.

This file is written as the rules land, not afterwards. Anything marked *planned* has
no code behind it yet.

**How these were established.** Every claim about DuckDB below was executed and
observed. Every claim about *Spark* is taken from its published behaviour and cited,
**not** verified against a running Spark — decision 13 rules out a JVM at any stage.
So a case where Spark's real behaviour differs from its documentation would show up
here as a confident ✅. The exposure is the undocumented corners: NULL propagation on
unusual inputs, empty input, overflow, and decimal promotion.

| Status | Meaning |
|---|---|
| ✅ | Translation rule implemented and covered by a conformance case |
| 📋 | Planned — the phase that will deliver it is named |
| ⚠️ | Documented divergence: we do **not** match Spark, deliberately |

## Ordering and nulls

| Behaviour | Spark | DuckDB | Rule | Status |
|---|---|---|---|---|
| `ORDER BY x ASC` nulls | first | **last** | always emit explicit `NULLS FIRST` | ✅ |
| `ORDER BY x DESC` nulls | last | **last** | emit explicit `NULLS LAST` anyway | ✅ |

> PLAN.md §3.5 records DuckDB as nulls-*first* on `DESC`. That was true of an older
> release; DuckDB 1.5.5 is nulls-last in **both** directions, so `ASC` is the only
> real divergence. `DESC` is still made explicit, because "happens to agree" is a
> property of this release rather than a promise.

## Casting and arithmetic

| Behaviour | Spark | DuckDB | Rule | Status |
|---|---|---|---|---|
| `CAST('abc' AS INT)` | `NULL` (non-ANSI) | error | `TRY_CAST` by default; `spark.sql.ansi.enabled=true` opts into strict. An explicit `try_cast(...)` stays lenient in both modes | ✅ |
| `1/0` | `NULL` | `inf` | guarded division. sqlglot's Spark parser already emits the `NULLIF`, so both surfaces agree with no rule of ours | ✅ |
| `x % 0` | `NULL` | `NULL` | no rule needed | ✅ |
| Integer overflow | wraps | error | not emulated — we error. `ansi_mode` does not change this; DuckDB cannot be asked to wrap | ⚠️ |
| Decimal promotion | Spark's precision rules | DuckDB's, and division falls to `DOUBLE` | explicit cast per Spark's rules | 📋 **deferred to Phase 14** — see below |

### Decimal promotion — measured, not yet fixed

Spark derives the result type of decimal arithmetic from the operand types by a fixed
rule, then clamps precision to 38. DuckDB has its own rules. Both were executed; this
is what they give for `DECIMAL(10,2) op DECIMAL(10,2)`:

| Operation | Spark's rule | Spark result | DuckDB result | Same? |
|---|---|---|---|---|
| `a + b` | precision `max(s1,s2) + max(p1-s1, p2-s2) + 1`, scale `max(s1,s2)` | `DECIMAL(11,2)` | `DECIMAL(11,2)` | ✅ |
| `a * b` | precision `p1+p2+1`, scale `s1+s2` | `DECIMAL(21,4)` | `DECIMAL(18,4)` | ❌ precision |
| `a / b` | precision `p1-s1+s2+max(6, s1+s2+1)`, scale `max(6, s1+s2+1)` | `DECIMAL(16,6)` | `DOUBLE` | ❌ **type** |

Division is the one that matters: falling to `DOUBLE` loses exactness, which is the
whole reason a monetary column is a decimal. Addition already agrees.

**Deferred to Phase 14** (decision 14). The rule needs the *operand* types, and the
type of a sub-expression is only known after binding. The vehicle is sqlglot's
`annotate_types`, which the optimizer pipeline currently omits; adding it and then
emitting an explicit cast per Spark's formula is the shape of the work. Doing it by
guesswork -- casting without knowing the operand types -- would produce a confidently
wrong precision, which is worse than an honest divergence.

Note there is **no guard** for this one, unlike merge-on-read: detecting "this query
would have promoted differently" needs the same type information as fixing it. So
until Phase 14 lands, this table is the warning.

**What to watch:** a decimal column divided by a decimal column comes back `DOUBLE`
-- accurate to about 15 significant digits, and wrong only where exact decimal
semantics were the reason for the column. Addition and subtraction already agree.

## Operators and functions

| Behaviour | Spark | DuckDB | Rule | Status |
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
| `df.join(o, "k")` | one `k` column | two | rewrite to `USING` + explicit projection | 📋 Phase 4 |

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

Not Spark-vs-DuckDB but engine-vs-format: places where handing DuckDB an Iceberg
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
| Bare date vs `timestamp` | SQL, Spark and DuckDB all read `ts >= '2024-01-01'` as midnight that day. PyIceberg demands full ISO-8601 and *raises* from inside `plan_files()` rather than declining | widen a date-only literal to `T00:00:00` for `timestamp` columns, then bind-validate every predicate before use, so one PyIceberg dislikes costs only its own conjunct | ✅ |
| Bare date vs `timestamptz` | a bare date carries no zone; DuckDB resolves it against the session's, so assuming UTC could move the boundary and prune away wanted rows | not widened — the conjunct stays in SQL and prunes nothing | ⚠️ |
| `ORDER BY <output alias>` | `qualify` leaves such a reference unqualified because it names the projection, not the table. Read as an unattributable column it disabled projection pushdown for the whole query | unqualified names matching an output alias are recognised as projection references; ones that also name a real column are included in the scan (a safe superset) | ✅ |

## Optimizer-level divergences

| Behaviour | Note | Status |
|---|---|---|
| Unaliased projection names | sqlglot's `qualify` renames `sum(amount)` to `_col_0`. The optimized plan is adopted only if its output columns can be re-aliased to the names analysis already computed; otherwise the unoptimized plan runs instead | ✅ |
| Identifier case in the optimizer | the optimizer runs in DuckDB's dialect, which is case-insensitive, so `VendorID` is normalised internally. Output names are restored from the analysed schema, so the user-visible spelling is unaffected | ✅ |
| Nullability | the analysed schema comes from DuckDB, which has no non-nullable expression, so every field reports `nullable = true` even where Iceberg marked it required | ⚠️ |

## Environment-level divergences

Not Spark-vs-DuckDB, but differences a migrating script will notice.

| Behaviour | Note | Status |
|---|---|---|
| `pyspark.__version__` | reports `3.5.0` (the semantics we target), not icetl's version. `pyspark.__icetl_version__` gives ours. | ✅ |
| Real PySpark cannot be co-installed | the shadow package owns the `pyspark` import name (decision 8) | ⚠️ |
| `getSqlState()` | always returns `None`; we do not model SQLSTATE codes | ⚠️ |
| `pandas` 3.x | the dependency floor `>=2.2` resolves to pandas 3.x, whose default string dtype differs from the `object` dtype Spark's `toPandas()` yields | 📋 still open — carried to Phase 3 |

# Findings

Things that were **not true of the world in the way we assumed**, found by building
icetl. One entry per finding, grouped by the kind of trap it is rather than by the phase
that hit it, because the next one will arrive in a different phase and look like an old
one.

## Where this sits among the other documents

| Document | Answers |
|---|---|
| [PLAN.md](PLAN.md) | what we are building and why, phase by phase |
| [STATUS.md](STATUS.md) | how far it got, in the order it happened |
| [divergence.md](src/icetl/compat/divergence.md) | where icetl and the reference engine deliberately differ |
| **this file** | where a **dependency, or our own generated SQL, was wrong** — and what each one taught |

A finding is not a divergence. A divergence is a decision; a finding is a discovery, and
usually a bug. Findings that ended in a permanent behavioural difference are cross-linked
to `divergence.md` rather than duplicated.

## Severity

| | Meaning |
|---|---|
| 🔴 | **Silent wrong answer** — a plausible result, no error, nothing to notice |
| 🟠 | Loud failure — a crash or a refusal, so it could not mislead |
| 🟡 | Performance or ergonomics only |
| 🟢 | Measured and found **fine** — recorded so it is not re-litigated |

---

## 1. Silent wrong answers

The class that matters. Every one of these returned a result a reviewer would have
accepted.

### 1.1 🔴 sqlglot's `simplify` mis-folds `CASE … WHEN TRUE`

*Phase 8.* A `CASE` whose always-true branch is **not the first** folds to that branch's
value, discarding every branch before it:

```
CASE WHEN a = 1 THEN 'one' WHEN a <= 2 THEN 'two' WHEN TRUE THEN 'rest' END  ->  'rest'
```

DuckDB answers `'one'`. sqlglot 30.17.

**Reach:** not just generated SQL — any hand-written `Session.sql()` carrying that shape
was affected, on the ordinary read path, and nothing raised.

**Found by:** a `MERGE` clause chain, where an unconditional `WHEN MATCHED` becomes a
`WHEN TRUE` branch. A three-clause merge assigned the last clause's value to every row.

**Guard:** `plan/optimizer.py::_close_always_true_branches` normalises the shape away
before the rules run — an always-true branch becomes the `ELSE`, and what follows it is
dropped, which no reachable row can tell apart. `simplify` never meets the case it
mishandles. `tests/unit/test_optimizer.py::TestAlwaysTrueCaseBranches`.

**Rule:** a rewrite rule in a dependency is not a correctness boundary. When a rule
changes an answer rather than a plan, normalise the input rather than trusting the rule.

### 1.2 🔴 A constant column stops meaning anything once the subquery is merged

*Phase 8.* The natural shape for `MERGE` is one `LEFT JOIN` carrying a marker that says
whether the join found a source row — it cannot be one of the source's own columns,
because every real column can legitimately be NULL:

```sql
LEFT JOIN (SELECT s.*, TRUE AS __matched FROM src AS s) AS s ON …
```

`merge_subqueries` flattens that subquery, and `__matched` becomes the literal `TRUE` in
the outer scope — where it means "there is a row here", not "the join matched".

**Consequence:** every target row looks matched, `WHEN MATCHED … DELETE` fires for all of
them, and **the statement empties the table and reports success.**

**Found by:** a `WHEN NOT MATCHED BY SOURCE` test returning `[]`.

**The fix that would have been wrong:** a window-function marker survives the merge — and
only because sqlglot declines to flatten a subquery containing one. That is an accident of
the rule list, not a property.

**Guard:** matchedness is expressed as a *predicate* instead — an inner join, where every
row is matched by construction, and `NOT EXISTS`, which no rewrite can turn into a
constant. `MERGE` became three queries rather than one join; it costs one extra scan of
the target. `sql/rowlevel.py`, and carry-over note 17.

**Rule:** never carry meaning in a constant. An optimizer is entitled to fold it, and
folding is invisible.

### 1.3 🔴 DuckDB re-typed partition columns from the directory name

*Phase 2, present unnoticed since Phase 1.* `read_parquet` auto-enables
`hive_partitioning` when it sees `key=value` directories — which an Iceberg warehouse is
full of. `fx.partitioned`'s `as_at_date` column is a **string** in the table and was
coming back as `DATE`, sourced from the path rather than from the data.

**Guard:** `hive_partitioning => false`, always, with a regression test. Confirmed
necessary against the real warehouse, which lays data out as `…/pickup_month=2024-12/…`.

**Rule:** a reader with helpful defaults will infer things the catalog already knows.
Turn the inference off rather than hope the two agree.

### 1.4 🔴 Around a dozen `F.*` functions produced perfect SQL and the wrong answer

*Phase 3.* Of the first 169 names: `split`, `greatest`/`least`, `concat_ws`, `log`,
`date_add`/`add_months`/`last_day`/`trunc`, `pmod`, `array_position`, `array_union`,
`sequence` — and `F.hash` produced a type Spark cannot represent at all. The third tranche
added `rint`, `weekday`, `regexp_substr`, `array_size` and `split_part`.

Several came from sqlglot's *typed nodes*: it wraps `Greatest`, `Least` and `ConcatWs` in
a NULL-propagating `CASE` for DuckDB, and renders `Log`'s operands reversed. The generated
SQL reads correctly in every case.

**Guard:** every function has a test asserting on a **value**, never on generated SQL.
`tests/fixture/test_functions.py` opens with why. Carry-over note 12.

**Rule:** this is the origin of the project's central testing rule. Generated SQL that
looks right is the failure mode, not the evidence.

### 1.5 🔴 sqlglot's `Bracket` is 0-based

*Phase 6.* `p[1]` generated `p[2]` for DuckDB — the node treats its subscript as 0-based
and adds one. The resulting error names an index the query never mentioned.

**Guard:** `struct_extract(p, 1)`, which is unambiguous. Carry-over note 12's hazard in a
new place.

### 1.6 🔴 A write was invisible to the session that made it

*Phase 7.* Create 5 rows, append 5, overwrite 5 — every count came back unchanged, which
reads exactly like the write silently failing. It was the **read** that was stale: a
`ScanSource` pins a PyIceberg table to the snapshot it was loaded at, and the session
caches sources for its whole lifetime.

**Guard:** `Session._invalidate_source`, matched on the resolved identifier so `nyc.trips`
and `trips` both go. A frame built *before* the write keeps its snapshot, which is
read-your-plan rather than a leak, and has its own test.

**Recurs:** Phase 8 needed it again, and for a second reason — dropping the cached source
before each attempt is also what makes the snapshot it validates the snapshot it read.

**Rule:** anything that changes a table must invalidate. It is the first thing to check
when a write "does nothing".

### 1.7 🔴 `exists` and `forall` over an empty list

*Phase 6.* `list_bool_or([])` and `list_bool_and([])` are both NULL; the reference says
`false` and (vacuously) `true`. Both are now spelled as three-valued `CASE`s.

**Why nothing caught it earlier:** a test over populated data never reaches an empty
collection. `fx.nested`'s second row has an empty `tags`, which is the only reason the
suite gets there at all.

**Rule:** the empty case is a different function. Fixture data has to contain one.

### 1.8 🔴 `weekday` / `dayofweek` disagree between the two surfaces — **still live**

*Phase 3, deferred to Phase 15 by decision 16.* A function whose reference behaviour comes
from *composition* in `sql/functions.py` exists only on the `F.*` path; `Session.sql()`
hands the bare name to DuckDB. For these two both surfaces answer, neither raises, and the
SQL surface carries DuckDB's week numbering.

**Exposure today:** `weekday` and `dayofweek` only. Nine other composed functions raise,
so they are loud. **Prefer `F.*` over `Session.sql()` for date-part work.**

**Why there is no guard:** detecting the divergence means evaluating a name on both
surfaces and comparing — which is the test, which is the fix. There was nothing cheaper to
assert, so it is written down instead. See `divergence.md`.

**Found by:** running `notebooks/01_read_real_table.ipynb` section 8, which probes both
surfaces side by side. 821 tests passed with the gap present.

---

## 2. Loud failures — bugs that could not mislead

### 2.1 🟠 A bare date against a `timestamp` column killed the query

*Phase 2, found only against the real table.* `filter(col >= "2024-01-01")` — the form
PLAN.md's own example uses — made PyIceberg's literal binding raise from **inside**
`plan_files()`. Phase 2's promise that "anything not understood is simply not pushed" did
not hold, because the rejection happened at scan time, long after translation.

**Guard, twice over:** every predicate is bind-validated before use, so a bad conjunct
costs only its own pruning; and a date-only literal is widened to `T00:00:00` for
`timestamp` columns, so the commonest filter on a time-partitioned table actually prunes.
`timestamptz` is deliberately left alone — a bare date has no zone.

**Rule:** a translation layer that declines gracefully must decline *at translation time*.
Validate where you can still change your mind.

### 2.2 🟠 `1.0 / 0.0` crashed the optimizer

*Phase 3.* `simplify` constant-folds literal arithmetic, and the arithmetic can fail:
Python's decimal division raised `decimal.DivisionByZero` from inside the rule.
`optimize_plan` caught `OptimizeError`, `KeyError`, `ValueError` and `TypeError` — every
way a rule was *known* to fail — and arithmetic has its own exception tree.

**Not findable by the obvious test:** `1 / 0` was tested on both surfaces and passes,
because `simplify` declines to fold integer division at all.

**Guard:** `ArithmeticError` is in the catch list. The rules that ran before the failure
are kept, and the Spark parser's own `NULLIF` still makes the answer NULL.

### 2.3 🟠 `args["from"]` is `args["from_"]` in sqlglot 30

*Phase 4, and again in Phase 8.* Reading the old key silently returns `None`. In Phase 4
that wrapped a join's right side in a generated alias, so `b.id` stopped resolving; in
Phase 8 it made `UPDATE … FROM` look absent and pass the refusal check.

**Guard:** `_from_clause` beside `_has_from_clause`, so the next reader finds the accessor
rather than the key. Phase 8 checks both spellings.

**Rule:** the same trap will be found twice by different people. Leave an accessor, not a
comment.

### 2.4 🟠 `struct_insert` refuses rather than overwrites

*Phase 6.* `withField` was written assuming it replaces a field of the same name. DuckDB
raises `Duplicate struct entry name`. So `withField` had to become schema-aware and
rebuild the struct, exactly as `dropFields` already did — which is also why both need the
frame's schema.

### 2.5 🟠 `F.struct` with an aliased column emitted invalid SQL

*Phase 6.* `F.struct(F.lit(1).alias("a"))` produced `{'a': 1 AS "a"}`, which will not
parse: the alias names the field and then has to be dropped from the value. Nothing caught
it because every earlier caller passed plain column names.

### 2.6 🟠 `scan().to_arrow()` double-counts rows

*Phase 2, carry-over note 2.* Closed at the time by `ArrowScan(...).to_table(tasks)`,
which takes an explicit task list.

**Do not go looking for that call today** — decision 11 took the merge-on-read split out
of Phase 2, and the copy-on-write scan path hands file paths to DuckDB's `read_parquet`
instead, so `scan_planner.py` only calls `plan_files()`. `ArrowScan` survives as **Phase
12's** plan for the dirty half of the hybrid split (PLAN.md §3.3), which is where the
double-counting trap will be waiting again.

---

## 3. Performance findings

### 3.1 🟡 `ORDER BY <output alias>` disabled projection pushdown entirely

*Phase 2, found against the real table.* `qualify` leaves such a reference unqualified
because it names the projection, not the table; the extractor read that as an
unattributable column and fell back to "read everything". PLAN.md's own headline aggregate
was scanning **19 of 19** columns — and 219 of 219 on the wide table. Now 3 of 19.

### 3.2 🟡 A set operation lost all pushdown to output-name restoration

*Phase 4, carry-over note 10, now closed.* `_naming_branch` re-aliases the leftmost
branch, so a set operation over an unaliased aggregate prunes like any other query.

### 3.3 🟡 `explain()`'s `bytes_scanned` ignores column pruning — **open**

It reports the *selected files'* size, not bytes read, so a narrow query looks far more
expensive than it is. Carry-over note 15, Phase 10.

### 3.4 🟡 An unfiltered `count(*)` opens parquet footers — **open**

Iceberg's manifests already hold the row count. Roughly 2× on a 357-file table, and it
grows with file count. Carry-over note 16, Phase 10.

### 3.5 🟡 Two references to one table merge to a single scan — **open**

Correct, and a union of columns with an OR of predicates, so a self-join with disjoint
filters prunes less than it could. Carry-over note 11.

### 3.6 🟡 `sum(double)` varies in its last digits between runs

DuckDB's aggregation order over parallel scans. Expected for floats; Spark does the same.
Documented rather than fixed.

---

## 4. Measured and found fine

Recorded so nobody spends the day again.

### 4.1 🟢 Window frame semantics agree on every point

*Phase 5.* PLAN.md called frames "where Spark/DuckDB drift is likeliest". Probed before
building: they agree, and the conformance layer needed nothing.

What *will* still bite a caller is not a divergence but SQL itself — with an ordering and
no explicit frame the default is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`,
which includes every row tying with the current one, so a running total jumps over a tie.
The two spellings agree on any column without duplicates, which is exactly what makes it
easy to ship the wrong one. `TestFrameDefaults`.

### 4.2 🟢 PyIceberg's native `delete` handles NULL correctly

*Phase 8.* Checked rather than assumed, because the whole delete path depends on it:
`DELETE … WHERE amount > 20` keeps the NULL-amount row on both the native and the rewrite
path. `test_the_rewrite_path_agrees_with_the_native_one` runs the two side by side on the
same data, so a future PyIceberg that changes its mind fails the suite.

### 4.3 🟢 PyIceberg 0.11 already implements dynamic partition overwrite

*Phase 7.* The hardest item on that phase's list — "overwrite only the partitions present
in the incoming data" — is `Table.dynamic_partition_overwrite`, found by reading the API.
It detects the partition values in the incoming Arrow table itself, which removed the one
part of the phase needing a transform-aware boundary calculation, and with it the risk of
a wrong `overwrite_filter` deleting data it should not have.

### 4.4 🟢 Reading complex types already worked

*Phase 6.* Struct, array and map columns already read, projected and round-tripped end to
end before the phase started; Phases 1–3 had built enough. What was missing was everything
that **produces** rather than consumes.

### 4.5 🟢 DuckDB's `unnest` gives rows for free, and repeated copies correlate

*Phase 6.* `unnest` in a select list already turns one row into one per element, so
`explode` needed no lateral join and no plan surgery. And repeating `unnest(x)` in one
select list unnests **once**, not twice — DuckDB correlates the copies, which is what lets
`posexplode` emit `generate_subscripts(x, 1) - 1` beside `unnest(x)` and get matching
pairs rather than a cross product. `test_the_positions_pair_with_their_elements` guards it,
because a cross product would have been four plausible-looking rows instead of two.

### 4.6 🟢 Neither SQLFrame nor `duckdb.experimental.spark` implements Spark conformance

*Phase 3, decision 12.* Evaluated in a throwaway venv, which is the expensive half of
Phases 3–6:

| Probe | Spark | SQLFrame | duckdb.experimental.spark |
|---|---|---|---|
| `1/0` | `NULL` | `inf` | `inf` |
| `CAST('abc' AS INT)` | `NULL` | raises | raises |
| `ORDER BY x` nulls | first | no explicit clause | no explicit clause |

Neither can read Iceberg through our planner either, and SQLFrame pins `sqlglot<30.13`
against our 30.17. So: build, with SQLFrame's 478 mappings as a reference table.

### 4.7 🟢 Two upstream limits, with characterisation tests that will fail when they lift

*Phase 7.* **No streaming writes** — `Transaction.append` opens with
`if not isinstance(df, pa.Table): raise ValueError`, so there is no reader or batch form,
and the whole result is materialised. Chunking would trade atomicity for it, which is the
worse deal. **An overwrite is two snapshots** — a `delete` then an `append`, which is how
Iceberg models replacing rows — but **one commit**, so no reader sees the table
mid-overwrite. `TestStreamingIsBlockedUpstream` and `TestSnapshotShape` assert both, so
the failure is the signal that something became buildable.

---

## 5. Process hazards

Not about the engine, and both cost real time.

### 5.1 🟠 Rewriting `.py` files with PowerShell corrupts them

*Phase 3.* `Get-Content` / `Set-Content -Encoding utf8` on PS 5.1 reads as ANSI and writes
UTF-8 **with BOM**, double-encoding every non-ASCII character. `héllo` became `héllo`
and three string-length tests failed.

**Rule:** bulk edits go through Python with an explicit `encoding="utf-8"`.

### 5.2 🔴 A blind prose substitution corrupted test data

*Phase 3.* A project-wide rename treated `F.lit("Spark SQL")` as prose. It is an **input**.

**Rule:** distinguish docstrings from other string literals with `ast`, not with a regex.
The payloads are now neutral (`"Basic SQL"`), base64 cases included.

---

## 6. The rules these produced

The findings are worth less than what they taught. In rough order of how often they have
since paid off:

1. **Assert on a value, never on generated SQL.** §1.4 is the origin; §1.1, §1.3 and §1.5
   would all have passed a generated-SQL test.
2. **An over-approximation is free on read and fatal on write.** Read pushdown may match
   more rows than the SQL — the SQL re-applies the filter. A row-level write has no second
   chance, so it needs a second, stricter gate rather than a loosened first one. See
   `divergence.md`, "Row-level operations".
3. **Never carry meaning in a constant** (§1.2), and do not rely on which subqueries an
   optimizer declines to flatten.
4. **Where a cheap guard exists, build it; where it does not, write the divergence down.**
   Decision 11's copy-on-write assertion costs ten lines and turns a silent wrong answer
   into a refusal naming Phase 12. Decisions 14 and 16 had no such guard — detecting the
   problem costs as much as fixing it — so they are documented instead.
5. **Fixture data must contain the empty case, the NULL case and the tie.** §1.7 and §4.1.
6. **Measure the dependency rather than reasoning about it** (§4.2, §4.3, §4.6), and leave
   the measurement as a test so the answer stays true.
7. **The real table finds what fixtures cannot.** §1.3, §2.1 and §3.1 were all reachable
   only against `nyc.yellow_tripdata`.

---

## 7. Still open

| # | Finding | Owner |
|---|---|---|
| §1.8 | `weekday` / `dayofweek` differ between the surfaces, silently | **Phase 15** |
| — | `size(NULL)` is `NULL` on both surfaces; the reference says `-1` | **Phase 15** |
| §3.3 | `explain()`'s `bytes_scanned` ignores column pruning | Phase 10 |
| §3.4 | An unfiltered `count(*)` reads parquet footers | Phase 10 |
| §3.5 | A self-join with disjoint filters prunes less than it could | Phase 4 note 11 |
| §3.6 | `sum(double)` varies in its last digits | documented, not fixed |
| — | Rename reconciliation is local-fixture-only and opens one footer per file; wants a benchmark before it meets a 4096-file table | Phase 2 leftover |
| — | The `MERGE` cardinality check fires for a by-source-only merge; whether the reference does too is unmeasured | Phase 8 |

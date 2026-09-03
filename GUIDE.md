# icetl user guide

A DataFrame and SQL surface over Apache Iceberg, executed by DuckDB, on one machine.

New here? Run [`notebooks/00_quickstart.ipynb`](notebooks/00_quickstart.ipynb) first —
it builds its own warehouse and needs no catalog. This guide is the reference to come
back to.

**Contents**

1. [The shape of the thing](#1-the-shape-of-the-thing)
2. [Getting a session](#2-getting-a-session)
3. [Reading](#3-reading)
4. [Transforming](#4-transforming)
5. [Reading `explain()`](#5-reading-explain)
6. [Getting results out](#6-getting-results-out)
7. [Writing](#7-writing)
8. [Changing rows in place](#8-changing-rows-in-place)
9. [Schema, DDL and time travel](#9-schema-ddl-and-time-travel)
10. [Your own Python](#10-your-own-python)
11. [Keeping tables fast](#11-keeping-tables-fast)
12. [Where icetl differs from the reference](#12-where-icetl-differs-from-the-reference)
13. [When something is slow](#13-when-something-is-slow)
14. [When something is wrong](#14-when-something-is-wrong)

---

## 1. The shape of the thing

Three libraries, each doing the one thing it is best at:

| | |
|---|---|
| **sqlglot** | the single plan IR — both surfaces build the same tree |
| **PyIceberg** | plans scans and commits writes — manifests, snapshots, metadata |
| **DuckDB** | executes — every row that moves, moves through DuckDB |

The consequence worth internalising: **the DataFrame API and `session.sql()` are one
code path, not two.** They converge on the same sqlglot tree before anything is
optimized, so they prune identically and return identical answers. Neither wraps the
other.

```python
api = session.table("nyc.trips").filter(F.col("vendor_id") == 1).select("fare")
sql = session.sql("SELECT fare FROM nyc.trips WHERE vendor_id = 1")
# same plan, same scan, same rows
```

**The semantic reference is Apache Spark 3.5**, as a written specification. `1/0` is
NULL, a failed cast is NULL, `ORDER BY` puts nulls first ascending. Nothing here runs
on, links against, or requires Spark. Every place the two differ on purpose is in
[`src/icetl/compat/divergence.md`](src/icetl/compat/divergence.md).

---

## 2. Getting a session

```python
from icetl.sql import Session, functions as F

session = Session.builder.appName("etl").getOrCreate()
```

Configuration comes from four layers, highest first: `.config(...)`, the process
environment, `.env`, then built-in defaults. Conf keys and environment variables are
two spellings of one setting:

| conf key | environment |
|---|---|
| `icetl.catalog.<name>.uri` | `ICETL_CATALOG_URI` |
| `icetl.catalog.<name>.warehouse` | `ICETL_CATALOG_WAREHOUSE` |
| `icetl.catalog.<name>.s3.endpoint` | `ICETL_S3_ENDPOINT` |
| `icetl.defaultCatalog` | `ICETL_CATALOG_NAME` |
| `icetl.defaultNamespace` | `ICETL_DEFAULT_NAMESPACE` |
| `icetl.ansiMode` | `ICETL_ANSI_MODE` |

`.env.example` documents every key. Several catalogs can be configured at once —
`icetl.catalog.prod.uri` and `icetl.catalog.staging.uri` coexist, and
`icetl.defaultCatalog` picks which one an unqualified name resolves to.

**Threads and memory are DuckDB's to choose.** Measured, not assumed: 2, 4 and 8
threads are indistinguishable on the benchmark suite and oversubscribing is worse
(decision 17). Narrow them only if you have a reason:

```python
Session.builder.config("icetl.duckdb.memoryLimit", "4GB").getOrCreate()
```

Spill is always configured, and it is the setting that turns an out-of-memory error
into a completed query.

---

## 3. Reading

```python
session.table("nyc.trips")  # the whole table, lazily
session.sql("SELECT * FROM nyc.trips")  # the same thing
session.read.table("nyc.trips")  # the same thing, with reader options
```

Files that are not in a table yet — for cleaning up and appending, or joining against:

```python
session.read.parquet("dump/trips.parquet")
session.read.csv("dump/trips.csv", header=True, sep=",")
session.read.json("dump/events.json")
session.read.format("parquet").load("dump/trips.parquet")
```

A file read is an ordinary frame; it just gets no pruning, because there are no
manifests to prune with. Two things to watch:

- **Type inference is DuckDB's sniffer**, and it is good enough to surprise you:
  `2026-08-17` in a CSV becomes a `date`, and appending that to a table storing
  `string` is refused. Cast to the table's schema deliberately.
- **Object-store paths are refused.** The schema is bound on a connection without S3
  credentials, so an `s3://` path would fail confusingly. Iceberg data in object
  storage is read by name with `session.table()`.

---

## 4. Transforming

The `F.*` namespace carries 273 functions, each with a value-level test. The
DataFrame surface covers joins, `groupBy().agg()`, `rollup`/`cube`/grouping sets, set
operations, ordering, `na.*`/`stat.*`, window functions, and the complex types.

```python
from icetl.sql import functions as F, Window

busy = (
    session.table("nyc.trips")
    .filter(F.col("as_at_date") == "2026-08-17")
    .groupBy("vendor_id")
    .agg(F.sum("fare").alias("total"), F.countDistinct("trip_id").alias("trips"))
    .orderBy(F.col("total").desc())
)

ranked = session.table("nyc.trips").select(
    "vendor_id",
    "fare",
    F.rank().over(Window.partitionBy("vendor_id").orderBy(F.col("fare").desc())).alias("r"),
)
```

**A string argument names a column**, which is the opposite of what a string means to
a `Column` operator:

```python
F.upper("name")  # upper-cases the column `name`
F.col("a") == "name"  # compares against the string "name"
```

**One caveat, until Phase 15.** Functions built by composition are reachable only
through `F.*`; a bare name in `Session.sql()` goes to DuckDB. `weekday` and
`dayofweek` answer **differently** between the two surfaces — silently, off by one. Use
`F.*` for date parts.

`cache()` is eager and returns a **new** frame, so `cached = df.cache()` is the
spelling that works — `df.cache()` alone caches nothing you can reach.

---

## 5. Reading `explain()`

The habit worth forming. Every number is a ratio, because a number with nothing to
compare it to cannot tell you whether pushdown worked.

```
== Scans ==
  nyc.trips: 3 of 4096 file(s), 1.5 KB of 76.7 KB
    columns: 2 of 214: vendor_id, fare
    pushed filters: as_at_date = '2026-08-17'
    kept in SQL only: upper(vendor) = 'A'
```

| line | what it tells you |
|---|---|
| `3 of 4096 file(s)` | partition and statistics pruning — how much never opened |
| `1.5 KB of 76.7 KB` | bytes of the **selected columns**, against the files' size |
| `columns: 2 of 214` | projection pushdown reaching the reader |
| `pushed filters` | what Iceberg evaluated against manifests |
| `kept in SQL only` | filters DuckDB applied — right answer, no pruning |

`explain(mode="extended")` adds the before-and-after optimizer trees, which is how you
find out *why* a query did not prune.

**`kept in SQL only` is not an error.** It means the predicate bought no pruning — a
function Iceberg cannot evaluate, a UDF, a comparison it cannot translate. The answer
is still right.

---

## 6. Getting results out

| call | gives you | use when |
|---|---|---|
| `collect()` | `list[Row]` | small results, or you want Python objects |
| `toArrow()` | `pyarrow.Table` | **anything wide** — see below |
| `toArrowBatches()` | an iterator of `RecordBatch` | the result may not fit in memory |
| `toLocalIterator()` | an iterator of `Row` | same, but you want Rows |
| `toPandas()` | `pandas.DataFrame` | handing to something that wants pandas |
| `show()` / `count()` | printed table / `int` | looking, and counting |

**On a wide table, ask for Arrow.** `collect()` builds a `Row` per result row, and on
the 200-column benchmark table that is **95% of the wall time** — 25.4 s against 1.3 s
for the same query (`FINDINGS.md` §3.8). It is the single most useful thing to know
when working a wide table.

Streaming has one rule:

```python
for batch in df.toArrowBatches():
    ...  # do not run another query on this session in here
```

DuckDB ends a result when its cursor runs the next query, and it does so *silently* —
a half-read stream would report its prefix as the whole answer. icetl notices and
**refuses** instead. Collect first with `toArrow()` if you need to interleave.

`count()` on an unfiltered scan is answered from Iceberg's manifests and opens no file
at all. Any filter, join, limit, generator or aggregate takes the ordinary path.

---

## 7. Writing

```python
df.write.mode("append").saveAsTable("nyc.trips")
df.write.mode("overwrite").partitionBy("as_at_date").saveAsTable("nyc.trips")
df.write.insertInto("nyc.trips")  # by position, not by name
session.sql("INSERT INTO nyc.trips SELECT * FROM staging")
```

**`overwrite` replaces every row in the table**, not the rows resembling the incoming
data. To replace only the partitions the new data touches:

```python
df.write.mode("overwrite").option("partitionOverwriteMode", "dynamic").saveAsTable(t)
```

The two differ by an option string and by the entire contents of the table. An
overwrite is two snapshots — a delete then an append — but **one commit**, so no
reader sees the table mid-overwrite.

`mergeSchema` allows a write whose frame has columns the table does not:

```python
df.write.option("mergeSchema", "true").mode("append").saveAsTable("nyc.trips")
```

---

## 8. Changing rows in place

```sql
DELETE FROM nyc.trips WHERE as_at_date < '2026-01-01';

UPDATE nyc.trips SET fare = fare * 1.1 WHERE vendor_id = 2;

MERGE INTO nyc.trips AS t
USING staging AS s ON t.trip_id = s.trip_id
WHEN MATCHED AND s.void THEN DELETE
WHEN MATCHED THEN UPDATE SET fare = s.fare
WHEN NOT MATCHED THEN INSERT *;
```

**Copy-on-write.** The files holding matched rows are rewritten; no delete files are
produced. A table another engine wrote with merge-on-read deletes is **refused** rather
than answered wrongly — `read_parquet` cannot see a delete file, so it would return
the deleted rows and report success. The refusal names Phase 12.

Row-level writes minimise the files they touch by translating the predicate twice: once
for Iceberg, to choose files, and once for SQL, to choose rows. The Iceberg half is
held to a **stricter** standard than a read predicate, because a read re-applies the
filter in SQL and a write has no second chance.

Nested-field assignment (`UPDATE t SET s.field = v`) is refused; it needs machinery
that has not been built.

---

## 9. Schema, DDL and time travel

```sql
CREATE TABLE nyc.trips (trip_id BIGINT, fare DOUBLE, as_at_date STRING)
  PARTITIONED BY (as_at_date);
ALTER TABLE nyc.trips ADD COLUMN vendor STRING;
ALTER TABLE nyc.trips RENAME COLUMN vendor TO vendor_name;
DROP TABLE nyc.trips;
```

Partition-spec and sort-order evolution are supported. `ALTER COLUMN ... TYPE` is
refused: PyIceberg 0.11 exposes no type update.

```python
session.catalog.listTables("nyc")
session.catalog.tableExists("nyc.trips")
```

Time travel, three spellings of one thing:

```sql
SELECT * FROM nyc.trips VERSION AS OF 8271497619288662701;
SELECT * FROM nyc.trips TIMESTAMP AS OF '2026-08-16T00:00:00';
```
```python
session.read.option("snapshot-id", 8271497619288662701).table("nyc.trips")
```

The *schema* used is the current one, not the snapshot's — which is what PyIceberg's
own scan does, and is recorded as a divergence.

Iceberg's metadata tables are queryable: `nyc.trips.snapshots`, `.files`,
`.manifests`, `.history`, `.partitions`.

```python
session.sql("SELECT snapshot_id, operation FROM nyc.trips.snapshots ORDER BY committed_at").show()
```

---

## 10. Your own Python

```python
band = session.udf.register("band", lambda fare: "high" if fare > 50 else "low", "string")

df.select(band("fare"))  # DataFrame surface
session.sql("SELECT band(fare) FROM nyc.trips")  # SQL surface, same function
```

The **return type is declared, not inferred**: DuckDB needs it before the first row,
and a wrong guess would be a silently mistyped column. It defaults to `string`.
Complex types work: `"array<bigint>"`, `"map<string,bigint>"`, `"struct<a:bigint>"`.

Vectorised, for a function taking and returning a `pandas.Series`:

```python
session.udf.registerVectorised("pct", lambda s: s / 100.0, "double")
```

**NULL does not reach the function by default.** This is a divergence from the
reference, arrived at by measurement — DuckDB's two null-handling modes are each
unusable on their own (`FINDINGS.md` §2.9), and the mode that lets a UDF *return* NULL
also hands it a row of NULLs the data never contained. Opt in where you want it:

```python
session.udf.register("or_zero", lambda v: 0.0 if v is None else v, "double", callOnNull=True)
```

A UDF is never pushed to Iceberg — it cannot be — so a predicate over one reads every
file and shows up under *kept in SQL only*. That is correct, not a bug.

---

## 11. Keeping tables fast

**Compaction is not tidying.** The read design leans on column-statistics file pruning,
and statistics only prune when files are large and sorted. Many small unsorted files
defeat it.

```python
m = session.maintenance("nyc.trips")

m.compact()  # per partition, sorted by the table's sort order
m.expireSnapshots(retainLast=5)  # or olderThan=..., or snapshotIds=[...]
m.removeOrphanFiles()  # reports; pass dryRun=False to delete
```

That order matters: compaction leaves the replaced files referenced by older snapshots,
expiry unreferences them, and only then are they orphans.

- `compact()` skips partitions that are already fine, so running it repeatedly costs
  nothing. It wants a quiet table: a writer landing rows between the read and the
  commit would have them replaced.
- `removeOrphanFiles()` **reports by default** and ignores anything written in the last
  three days, because a commit in flight writes its data before referencing it.
- `rewriteManifests()` is refused. PyIceberg 0.11 exposes none, and a hand-written
  manifest fails silently by hiding data files.

---

## 12. Where icetl differs from the reference

The full register is [`src/icetl/compat/divergence.md`](src/icetl/compat/divergence.md).
The ones most likely to bite:

| | |
|---|---|
| `weekday` / `dayofweek` | **off by one** through `Session.sql()`. Use `F.*` — Phase 15 |
| `cache()` | eager, and returns a new frame |
| UDFs and NULL | the function is not called for an all-NULL call unless `callOnNull=True` |
| `repartition()` / `coalesce()` | no-ops — there is one partition and always will be |
| `toLocalIterator()` | must be finished before the session runs another query |
| Nullability | every analysed field is nullable; DuckDB has no non-nullable expression |
| Decimal arithmetic | `*` differs in precision, `/` returns `DOUBLE` — Phase 14 |
| `size(NULL)` | `NULL` here, `-1` in the reference — Phase 15 |

Conformance cases are written from the reference's *published* behaviour, with a
citation each, and are **asserted rather than machine-verified** — no Spark is
installed at any stage (decision 13). An edge case where the reference's real
behaviour differs from its documentation will produce a confidently green test.

---

## 13. When something is slow

1. **`explain()` first.** If `columns: 214 of 214`, projection pushdown failed. If
   `pushed filters: none` on a partitioned table, predicate pushdown failed.
2. **Are you calling `collect()` on a wide result?** That is usually the answer —
   `toArrow()` is up to 19× faster on the benchmark table.
3. **How many files?** `3 of 4096` is healthy; `4096 of 4096` means nothing pruned.
   Many small files means `compact()`.
4. **Is the filter translatable?** Anything under *kept in SQL only* prunes nothing —
   a UDF, a function Iceberg has no equivalent for, a comparison against an expression.
5. **Measure it.** `uv run python scripts/benchmark.py` runs a fixed suite;
   [`BENCHMARKS.md`](BENCHMARKS.md) has a table for reading which row moved and what
   that means.

Known pruning gaps, written down rather than surprising: a self-join with disjoint
filters prunes less than it could (§3.6), and a query with no filter still lists every
manifest.

---

## 14. When something is wrong

**[`FINDINGS.md`](FINDINGS.md) is the first place to look**, and ten minutes in it is
worth it before writing anything that generates SQL. It is a register of every time a
dependency turned out to be confidently wrong — thirteen silent wrong answers so far,
each with the guard that now prevents it.

The rules those findings produced, which are also good advice for using icetl:

1. **Assert on a value, never on generated SQL.** Around a dozen of the first 169
   functions produced perfectly plausible SQL and the wrong answer.
2. **Ask the same frame two different questions and compare.** `count()` disagreed with
   `len(collect())` for two whole phases, because every test asked one question.
3. **An over-approximation is free on read and fatal on write.**
4. **Fixture data must contain the empty case, the NULL case and the tie.**
5. **A fix verified on one spelling of a construct is not verified.** The anti-join bug
   was fixed for `LEFT JOIN` and stayed broken for `RIGHT` for five phases.

Errors are typed, and the type is the first clue:

| exception | means |
|---|---|
| `AnalysisException` | the plan does not make sense — a missing column, a schema mismatch |
| `ParseException` | the SQL did not parse |
| `QueryExecutionException` | DuckDB failed while running it |
| `UnsupportedFeatureError` | deliberately not built — the message names the phase that owns it |
| `EngineTypeError` / `EngineValueError` | an argument was wrong |

An `UnsupportedFeatureError` naming a phase is a decision, not an oversight. PLAN.md §4
says what that phase covers and STATUS.md says why it was deferred.

# Benchmarks

Wall-clock times for a fixed set of queries over a 200-column Iceberg table, so a
change that costs performance shows up as a number rather than as a feeling. This
file is tracked; re-run it and commit the diff when you change anything on the read
path.

```bash
uv run python scripts/benchmark.py                            # what produced the table below
uv run python scripts/benchmark.py --rows 2000000             # bigger
uv run python scripts/benchmark.py --table nyc.yellow_tripdata  # the real catalog
uv run python scripts/benchmark.py --compare-threads 2 4 8 16 # settle a config question
uv run python scripts/benchmark.py --markdown BENCHMARKS.md   # rewrite this file
```

Every case checks its own answer on every repeat. A benchmark that got fast by
getting wrong is the failure a timing harness is most likely to hide.

---

## The run of record

**2026-09-02** · Windows 10, 8 logical cores (4 physical), 15.8 GB RAM · Python 3.12 ·
DuckDB 1.5.5 · default threads 8, default `memory_limit` 12.5 GiB · 200,000 rows ×
200 columns, identity-partitioned into 8 files · best of 3.

| benchmark | best (s) | median (s) | rows | what it measures |
|---|---:|---:|---:|---|
| `narrow` | 0.440 | 0.532 | 200000 | 2 of 200 columns — projection pushdown |
| `narrow_arrow` | 0.035 | 0.036 | 200000 | the same 2 columns as Arrow — `narrow` minus this is Row building |
| `wide` | 25.445 | 30.964 | 200000 | all 200 columns — the anti-pattern, for the ratio |
| `wide_arrow` | 1.312 | 1.472 | 200000 | all 200 columns as Arrow — the scan alone |
| `filtered` | 0.119 | 0.152 | 25000 | 2 columns behind a partition predicate — file pruning |
| `count` | 0.042 | 0.043 | 200000 | `count(*)` — answered from manifests, opens no file |
| `aggregate` | 0.070 | 0.072 | 8 | group by a partition column, sum a measure |
| `stream` | 0.050 | 0.069 | 200000 | 2 columns streamed in batches — peak memory is one batch |
| `join` | 1.177 | 1.235 | 200000 | self-join on `id`, 2 columns each |

---

## What the numbers say

### Projection pushdown is worth 13×, and it is the scan that gets faster

`wide_arrow` 1.312 s against `narrow_arrow` 0.035 s is the honest ratio for reading
2 columns instead of 200 — measured on Arrow, where nothing but the scan is being
timed. PLAN.md §3.6 sizes the whole design around this and the number supports it.

### `collect()` on a wide result is 95% Python, not DuckDB

`wide` 25.4 s against `wide_arrow` 1.3 s. The scan takes a second and a half; the
other twenty-four seconds are building 200,000 `Row` objects of 200 fields each — 40
million Python values. The same gap in miniature is `narrow` 0.440 s against
`narrow_arrow` 0.035 s.

This makes `toArrow()` and `toArrowBatches()` a **speed** feature as much as a memory
one, and it is the single most useful thing to know when working the 250M-row table:
`collect()` a wide result and the engine is barely involved in what you are waiting
for. Recorded as FINDINGS.md §3.8.

### `count(*)` costs nothing, because it reads nothing

0.042 s over a table whose full scan is 1.3 s. Iceberg's manifests carry the row
count, so the query opens no data file at all — FINDINGS.md §3.4, closed in Phase 10.

### Thread count: DuckDB's default is right, and 16 is worse

`--compare-threads 2 4 8 16`, same table, best of 3:

| benchmark | 2 | 4 | 8 (default) | 16 |
|---|---:|---:|---:|---:|
| `narrow` | 0.478 | 0.532 | 0.449 | 0.478 |
| `wide` | 14.068 | 14.257 | 14.601 | 14.602 |
| `filtered` | 0.045 | 0.048 | 0.045 | 0.087 |
| `count` | 0.022 | 0.019 | 0.017 | 0.038 |
| `aggregate` | 0.035 | 0.029 | 0.032 | 0.057 |
| `stream` | 0.039 | 0.034 | 0.033 | 0.065 |
| `join` | 0.595 | 0.537 | 0.538 | 1.293 |

*(Run before `wide_arrow` was added, so `wide` here is the `collect()` figure.)*

2, 4 and 8 are indistinguishable — every difference is inside the run-to-run spread.
16, twice this machine's logical cores, is clearly worse: the join more than doubles.

**PLAN.md Phase 10 proposed `threads = physical cores`. The measurement does not
support it** — 4 (physical) and 8 (logical, DuckDB's default) are the same to within
noise, so the change would add code and a configuration surface to buy nothing. The
existing behaviour stands: threads are left to DuckDB, and icetl only ever *narrows*
them when asked. Oversubscribing is the only thing that hurts, and nothing in icetl
does that.

### Spill works, and it has a floor

The engine always configures `temp_directory`, because DuckDB will not spill without
one. Measured at a 400 MB `memory_limit`, on a sort with roughly 600 MB of working
set:

| `temp_directory` | result |
|---|---|
| set | **OK**, 1.77 s |
| `''` (DuckDB's "do not spill") | `Out of Memory` |

So the setting is doing exactly what it claims. Below roughly 400 MB the same query
fails *either way* — DuckDB needs a working set of buffers before it has anything to
spill from — so a temp directory buys a query too big for memory, not a query with
no memory. `preserve_insertion_order` made no difference at any limit tried.

Pinned in `tests/fixture/test_engine_memory.py`, as a difference rather than as a
setting: a test asserting only that the option was applied would still pass on a
DuckDB that had stopped honouring it.

---

## Reading a regression

The cases are chosen so that a regression lands on one row and that row names the
cause. A single "scan the table" number could not separate these:

| row moved | look at |
|---|---|
| `narrow_arrow` up, `wide_arrow` flat | projection pushdown stopped narrowing the column list |
| `filtered` up, `narrow` flat | predicate pushdown stopped pruning files |
| `count` up from ~0.04 s | the metadata fast path stopped firing — `plan/counting.py` |
| `stream` approaching `narrow` | streaming started materialising the whole result |
| everything up together | the engine, the machine, or the table |

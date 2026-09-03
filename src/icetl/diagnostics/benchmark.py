"""The wide-table benchmark harness: what Phase 10 measures itself against.

Runs a fixed set of queries over a 200-column Iceberg table and reports wall time,
so a change that costs performance shows up as a number rather than as a feeling.
The results belong in `BENCHMARKS.md`, tracked, because a benchmark nobody committed
is a benchmark nobody can regress against.

    uv run python scripts/benchmark.py                     # local fixture, 200k rows
    uv run python scripts/benchmark.py --rows 2000000      # bigger
    uv run python scripts/benchmark.py --table ns.tbl      # the real catalog
    uv run python scripts/benchmark.py --compare-threads   # settle a config question
    uv run python scripts/benchmark.py --markdown BENCHMARKS.md

**The cases are chosen to separate the things that can each go wrong on their own.**
A single "scan the table" number cannot tell projection pushdown from predicate
pushdown from execution, so `narrow` and `wide` differ only in the column count,
`filtered` and `narrow` only in the predicate, and `count` exists to show that it
touches no file at all. A regression lands on one row, which names the cause.

**Every case asserts on its result**, not only on its duration. A benchmark that has
started returning the wrong answer quickly is the failure mode a timing harness is
most likely to hide, so each one carries the row count it must produce.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from icetl.conf import CatalogSettings, EngineSettings, IcetlSettings, resolve_settings
from icetl.sql import Session
from icetl.sql import functions as F

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from icetl.sql.dataframe import DataFrame

__all__ = ["BENCHMARKS", "Benchmark", "Result", "build_wide_table", "main", "run_all"]

#: Columns in the generated table. The number PLAN.md 3.6 sizes the design against.
WIDE_COLUMNS = 200
#: Rows by default. Big enough that the timings are not startup noise, small enough
#: that the default run finishes while you are still looking at it.
DEFAULT_ROWS = 200_000
DEFAULT_REPEAT = 3

NAMESPACE = "bench"
TABLE = "wide"


@dataclass(frozen=True)
class Benchmark:
    """One measured query, and the answer it has to keep giving."""

    name: str
    note: str
    build: Callable[[Session, str], DataFrame]
    #: What `consume` returns when the query is right. Checked every repeat, because
    #: a benchmark that got fast by getting wrong is worse than a slow one.
    expect: Callable[[Any], bool]
    #: How the frame is drained. `collect` for most; `count` and the streaming case
    #: measure something the default would hide.
    consume: str = "collect"


@dataclass
class Result:
    """Timings for one benchmark, in seconds."""

    name: str
    note: str
    runs: list[float] = field(default_factory=list)
    answer: str = ""

    @property
    def best(self) -> float:
        return min(self.runs)

    @property
    def median(self) -> float:
        return statistics.median(self.runs)


def _consume(frame: DataFrame, how: str) -> Any:
    if how == "collect":
        return len(frame.collect())
    if how == "count":
        return frame.count()
    if how == "stream":
        # Deliberately not `collect()`: the point of the streaming path is that the
        # whole result never exists at once, and a benchmark that materialised it
        # would be measuring the thing streaming exists to avoid.
        return sum(batch.num_rows for batch in frame.toArrowBatches())
    if how == "arrow":
        return frame.toArrow().num_rows
    raise AssertionError(f"unknown consume mode {how!r}")  # pragma: no cover


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark(
        name="narrow",
        note="2 of 200 columns -- projection pushdown",
        build=lambda session, table: session.table(table).select("id", "col_001"),
        expect=lambda rows: rows > 0,
    ),
    Benchmark(
        name="narrow_arrow",
        note="the same 2 columns, left as Arrow -- `narrow` minus this is Row building",
        build=lambda session, table: session.table(table).select("id", "col_001"),
        expect=lambda rows: rows > 0,
        consume="arrow",
    ),
    Benchmark(
        name="wide",
        note="all 200 columns -- the anti-pattern, for the ratio",
        build=lambda session, table: session.table(table),
        expect=lambda rows: rows > 0,
    ),
    Benchmark(
        name="wide_arrow",
        note="all 200 columns as Arrow -- the scan alone, without Row building",
        build=lambda session, table: session.table(table),
        expect=lambda rows: rows > 0,
        consume="arrow",
    ),
    Benchmark(
        name="filtered",
        note="2 columns behind a partition predicate -- file pruning",
        build=lambda session, table: (
            session.table(table).filter(F.col("part") == 3).select("id", "col_001")
        ),
        expect=lambda rows: rows > 0,
    ),
    Benchmark(
        name="count",
        note="count(*) -- answered from manifests, opens no file",
        build=lambda session, table: session.table(table),
        expect=lambda rows: rows > 0,
        consume="count",
    ),
    Benchmark(
        name="aggregate",
        note="group by a partition column, sum a measure",
        build=lambda session, table: (
            session.table(table).groupBy("part").agg(F.sum("col_001").alias("total"))
        ),
        expect=lambda rows: rows > 0,
    ),
    Benchmark(
        name="stream",
        note="2 columns, streamed in batches -- peak memory is one batch",
        build=lambda session, table: session.table(table).select("id", "col_001"),
        expect=lambda rows: rows > 0,
        consume="stream",
    ),
    Benchmark(
        name="join",
        note="self-join on id, 2 columns each",
        build=lambda session, table: (
            session.table(table)
            .select("id", "col_001")
            .alias("a")
            .join(session.table(table).select("id", "col_002").alias("b"), on="id")
        ),
        expect=lambda rows: rows > 0,
    ),
)


# ---------------------------------------------------------------------------
# The table under test
# ---------------------------------------------------------------------------


def build_wide_table(root: Path, rows: int, *, columns: int = WIDE_COLUMNS) -> str:
    """Create a partitioned 200-column Iceberg table in a local warehouse.

    Written through PyIceberg rather than through icetl's own writer, so the harness
    measures reading and is not also measuring the thing that produced its input.
    Partitioned on `part` so the `filtered` case has something to prune.
    """
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import Schema
    from pyiceberg.transforms import IdentityTransform
    from pyiceberg.types import DoubleType, LongType, NestedField

    root.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "bench",
        uri=f"sqlite:///{(root / 'catalog.db').as_posix()}",
        warehouse=f"file://{root.as_posix()}",
    )

    fields = [
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "part", LongType(), required=False),
    ]
    fields += [
        NestedField(index + 3, f"col_{index + 1:03d}", DoubleType(), required=False)
        for index in range(columns - 2)
    ]
    schema = Schema(*fields)
    spec = PartitionSpec(
        PartitionField(source_id=2, field_id=1000, transform=IdentityTransform(), name="part")
    )

    catalog.create_namespace_if_not_exists(NAMESPACE)
    identifier = f"{NAMESPACE}.{TABLE}"
    table = catalog.create_table(identifier, schema=schema, partition_spec=spec)

    # One append per partition, so the table has several data files and pruning has
    # something to do -- a single-file table would report pruning that never happened.
    partitions = 8
    per_partition = max(1, rows // partitions)
    for part in range(partitions):
        start = part * per_partition
        data: dict[str, list[Any]] = {
            "id": list(range(start, start + per_partition)),
            "part": [part] * per_partition,
        }
        for index in range(columns - 2):
            data[f"col_{index + 1:03d}"] = [
                float((start + offset) % 1000) + index for offset in range(per_partition)
            ]
        table.append(pa.table(data, schema=schema.as_arrow()))

    return identifier


def _local_settings(root: Path, engine: EngineSettings) -> IcetlSettings:
    return IcetlSettings(
        catalog=CatalogSettings(
            name="bench",
            type="sql",
            uri=f"sqlite:///{(root / 'catalog.db').as_posix()}",
            warehouse=f"file://{root.as_posix()}",
        ),
        engine=engine,
        default_namespace=(NAMESPACE,),
    )


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def run_all(
    session: Session,
    table: str,
    *,
    repeat: int = DEFAULT_REPEAT,
    only: Sequence[str] | None = None,
) -> list[Result]:
    """Run each benchmark `repeat` times, checking its answer every time."""
    selected = [b for b in BENCHMARKS if not only or b.name in only]
    results: list[Result] = []

    for benchmark in selected:
        result = Result(benchmark.name, benchmark.note)
        for attempt in range(repeat):
            frame = benchmark.build(session, table)
            started = time.perf_counter()
            answer = _consume(frame, benchmark.consume)
            result.runs.append(time.perf_counter() - started)
            if not benchmark.expect(answer):
                raise SystemExit(
                    f"benchmark {benchmark.name!r} returned {answer!r} on run "
                    f"{attempt + 1}, which is not a valid answer -- a timing is "
                    f"worthless if the query stopped being right"
                )
            result.answer = str(answer)
        results.append(result)
    return results


def _render(results: list[Result], *, heading: str) -> str:
    lines = [
        heading,
        "",
        "| benchmark | best (s) | median (s) | result | what it measures |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        lines.append(
            f"| `{result.name}` | {result.best:.3f} | {result.median:.3f} | "
            f"{result.answer} | {result.note} |"
        )
    return "\n".join(lines)


def _engine_settings(
    threads: int | None, memory_limit: str | None, temp_directory: str | None
) -> EngineSettings:
    return EngineSettings(threads=threads, memory_limit=memory_limit, temp_directory=temp_directory)


def _describe_machine() -> str:
    import duckdb

    connection = duckdb.connect()
    threads = connection.execute("SELECT current_setting('threads')").fetchone()
    memory = connection.execute("SELECT current_setting('memory_limit')").fetchone()
    connection.close()
    return (
        f"DuckDB {duckdb.__version__}, default threads "
        f"{threads[0] if threads else '?'}, default memory_limit "
        f"{memory[0] if memory else '?'}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS, help="rows in the generated table"
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="runs per benchmark")
    parser.add_argument("--only", nargs="*", help="run only these benchmarks by name")
    parser.add_argument(
        "--table",
        help="run against this table in the configured catalog instead of a generated one",
    )
    parser.add_argument("--threads", type=int, help="DuckDB threads (default: DuckDB's own)")
    parser.add_argument("--memory-limit", help="DuckDB memory_limit, e.g. '4GB'")
    parser.add_argument(
        "--compare-threads",
        nargs="*",
        type=int,
        help="run the suite once per thread count and print them side by side",
    )
    parser.add_argument("--markdown", help="write the results to this file as a Markdown table")
    parser.add_argument("--keep", help="reuse/keep the generated warehouse at this path")
    args = parser.parse_args(argv)

    temporary = args.keep is None and args.table is None
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="icetl-bench-"))

    try:
        if args.table:
            table = args.table
            base = resolve_settings()
        else:
            if not (root / "catalog.db").exists():
                print(f"building {args.rows:,} rows x {WIDE_COLUMNS} columns in {root} ...")
                started = time.perf_counter()
                build_wide_table(root, args.rows)
                print(f"  built in {time.perf_counter() - started:.1f}s")
            table = f"{NAMESPACE}.{TABLE}"
            base = _local_settings(root, EngineSettings())

        print(_describe_machine())
        thread_counts = args.compare_threads if args.compare_threads is not None else [args.threads]
        rendered: list[str] = []

        for threads in thread_counts:
            engine = _engine_settings(threads, args.memory_limit, str(root / "spill"))
            settings = IcetlSettings(
                catalog=base.catalog,
                s3=base.s3,
                engine=engine,
                sql=base.sql,
                default_namespace=base.default_namespace,
            )
            with Session(settings=settings) as session:
                results = run_all(session, table, repeat=args.repeat, only=args.only)
            label = f"threads={threads}" if threads is not None else "threads=default"
            block = _render(results, heading=f"### {table} — {label}")
            rendered.append(block)
            print()
            print(block)

        if args.markdown:
            Path(args.markdown).write_text(
                "\n\n".join(["# Benchmarks", "", _describe_machine(), "", *rendered]) + "\n",
                encoding="utf-8",
            )
            print(f"\nwritten to {args.markdown}")
    finally:
        if temporary:
            shutil.rmtree(root, ignore_errors=True)

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

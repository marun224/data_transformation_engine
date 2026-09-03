"""How to assert something true about data you did not author.

The default suite knows every row of every fixture, so it asserts values. Against a real
table that option is gone: `assert count == 3539170` is true until someone re-seeds the
catalog, and then it is a failing test that has found nothing. Worse, it is the kind of
failure people learn to ignore.

So the integration suite asserts in five ways instead, and each test picks the one that
actually holds:

1. **Exact values, on a replica we built** -- `tests/integration/seed.py` layer 1. Where
   this is available it is the strongest option and it is used.
2. **Differential against PyIceberg** (`pyiceberg_count`). PyIceberg owns the metadata,
   so for anything about rows, files or partitions it is the authority. This is the
   pattern `test_phase2_rest.py` established: *pruning must change speed, not answers.*
3. **Differential across the two surfaces** (`agree_across_surfaces`). The same question
   asked through `session.sql()` and through the DataFrame API must return the same
   answer, because P1 says they are one code path. This is not a formality -- `count()`
   disagreed with `len(collect())` for two whole phases, and it survived because every
   test asked one question.
4. **Differential against raw DuckDB** (`duckdb_answer`). Read the same parquet files
   with plain `read_parquet`, bypassing icetl's plan entirely. Any disagreement is in
   the plan, the pushdown or the conformance layer -- which is precisely what we want
   told about.
5. **Algebraic invariants** (`assert_sorted`, `assert_partition_of`, ...). Properties
   that hold whatever the data is, so they survive a re-seed and need no golden value.

The rule the whole suite follows: **no expected value derived from `nyc` is ever written
into a test.** Where a count is needed it is computed, in the same test, by one of the
above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from icetl.paths import engine_paths

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa
    from pyiceberg.expressions import BooleanExpression
    from pyiceberg.table import Table

    from icetl.exec.scan_planner import ScanPlan
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session

__all__ = [
    "agree_across_surfaces",
    "assert_partition_of",
    "assert_sorted",
    "column",
    "compiled_sql",
    "duckdb_answer",
    "pyiceberg_count",
    "scan_of",
    "scans_of",
]


# ---------------------------------------------------------------------------
# Reaching into the plan
# ---------------------------------------------------------------------------


def scans_of(df: DataFrame) -> list[ScanPlan]:
    """Every scan the frame compiles to.

    Compiling rather than executing: these assertions are about what icetl *decided to
    read*, which is settled before a byte moves.
    """
    return df._session._compile(df._plan, df._sources, df.columns).scans


def scan_of(df: DataFrame) -> ScanPlan:
    """The single scan the frame compiles to, asserting there is exactly one.

    Shared rather than redefined per module -- `test_phase2_rest.py` had its own copy,
    and a helper that drifts between files is a helper that stops meaning one thing.
    """
    scans = scans_of(df)
    assert len(scans) == 1, f"expected one scan, got {len(scans)}"
    return scans[0]


def compiled_sql(df: DataFrame) -> str:
    """The DuckDB SQL the frame generates, for the few tests that must read it."""
    return df._session._compile(df._plan, df._sources, df.columns).sql


# ---------------------------------------------------------------------------
# The differentials
# ---------------------------------------------------------------------------


def pyiceberg_count(
    table: Table,
    *,
    row_filter: BooleanExpression | None = None,
    selected_fields: tuple[str, ...] = ("*",),
) -> int:
    """The row count PyIceberg itself reports for a filtered scan.

    The authority for anything metadata-shaped. Note `selected_fields` defaults to
    everything: narrowing it is a speed choice, never a correctness one, and a caller
    that narrows it wrongly would be comparing against the wrong number.
    """
    from pyiceberg.expressions import AlwaysTrue

    scan = table.scan(
        row_filter=row_filter if row_filter is not None else AlwaysTrue(),
        selected_fields=selected_fields,
    )
    return int(scan.to_arrow().num_rows)


def agree_across_surfaces(session: Session, sql: str, frame: DataFrame) -> list[tuple[Any, ...]]:
    """Assert the SQL surface and the DataFrame surface give the same rows, and return them.

    P1 is the claim that these are one code path. A test that only ever asks one of them
    cannot notice when they stop being.
    """
    from_sql = _rows(session.sql(sql))
    from_frame = _rows(frame)
    assert from_sql == from_frame, (
        f"the two surfaces disagree, which means they are not one code path:\n"
        f"  sql:   {from_sql[:5]}{' ...' if len(from_sql) > 5 else ''}\n"
        f"  frame: {from_frame[:5]}{' ...' if len(from_frame) > 5 else ''}"
    )
    return from_sql


def duckdb_answer(session: Session, df: DataFrame, sql_over_files: str) -> pa.Table:
    """Run `sql_over_files` against the frame's own parquet files, bypassing icetl's plan.

    `$paths` is bound to the files the frame's scan selected, so the comparison is over
    exactly the same bytes: any difference is icetl's planning, pushdown or conformance
    layer, not a different input. Use `read_parquet($paths)` as the relation.
    """
    paths = engine_paths([p for s in scans_of(df) for g in s.groups for p in g.paths])
    engine = session._engine
    engine.ensure_object_store(paths)
    return engine.arrow(sql_over_files, {"paths": paths})


# ---------------------------------------------------------------------------
# The invariants
# ---------------------------------------------------------------------------


def column(df: DataFrame, name: str) -> list[Any]:
    """One column of a frame, as a Python list, in row order.

    Through Arrow rather than `collect()`: on a wide result `collect()` is 95% `Row`
    building (FINDINGS 3.8), and a test that reads one column should not pay for it.
    """
    return list(df.toArrow().column(name).to_pylist())


def assert_sorted(
    values: Sequence[Any], *, ascending: bool = True, nulls_first: bool = True
) -> None:
    """Assert an ordered result really is ordered, with NULLs where the reference puts them.

    An invariant rather than a golden list: it holds for any data, so it keeps working
    when the catalog is re-seeded. `nulls_first` defaults to the reference engine's
    ascending behaviour, which is the divergence the conformance layer exists to
    preserve -- DuckDB would put them last.
    """
    nulls = [i for i, v in enumerate(values) if v is None]
    present = [v for v in values if v is not None]

    ordered = sorted(present, reverse=not ascending)
    assert present == ordered, (
        f"not sorted {'ascending' if ascending else 'descending'}: "
        f"{present[:8]}{' ...' if len(present) > 8 else ''}"
    )

    if nulls:
        expected_at = (
            list(range(len(nulls)))
            if nulls_first
            else list(range(len(values) - len(nulls), len(values)))
        )
        assert nulls == expected_at, (
            f"NULLs are at {nulls[:8]}, expected "
            f"{'first' if nulls_first else 'last'} ({expected_at[:8]})"
        )


def assert_partition_of(parts: Sequence[Sequence[Any]], whole: Sequence[Any]) -> None:
    """Assert the pieces partition the whole -- disjoint, and covering it exactly.

    The workhorse invariant for filters: a predicate and its complement must re-union
    to the unfiltered table, whatever the data holds. It catches a filter that drops
    rows and a filter that duplicates them, without anyone knowing the row count.
    """
    combined: list[Any] = [value for part in parts for value in part]
    assert sorted(combined) == sorted(whole), (
        f"the pieces do not reconstruct the whole: "
        f"{len(combined)} rows across {len(parts)} parts vs {len(whole)} in total"
    )


def _rows(df: DataFrame) -> list[tuple[Any, ...]]:
    """A frame as a sorted list of tuples, so two results compare independent of order.

    Sorted because neither surface promises an order without `ORDER BY`, and a
    difference in row order is not the difference these helpers exist to find.
    """
    table = df.toArrow()
    columns = [table.column(i).to_pylist() for i in range(table.num_columns)]
    return sorted(zip(*columns, strict=True), key=_sort_key) if columns else []


def _sort_key(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Order rows with NULLs and mixed types present, without raising.

    `sorted()` on tuples containing `None` raises as soon as it compares one against a
    number, and real data has NULLs everywhere -- so each value becomes
    `(is_null, type_name, value)` and only comparable things are ever compared.
    """
    return tuple((v is None, type(v).__name__, "" if v is None else str(v)) for v in row)

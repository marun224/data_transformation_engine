"""Proof of life: catalog -> namespace -> table -> plan -> DuckDB read.

Six steps, each reported pass or fail, so a broken setup names the layer that broke
instead of surfacing one stack trace from somewhere deep in pyarrow. This is the
first thing to run against a new environment.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.table import Table as RichTable

from icetl.catalog import CatalogRegistry, TableResolver
from icetl.conf import IcetlConf, IcetlSettings, resolve_settings
from icetl.exec import DuckDBEngine
from icetl.paths import engine_paths

if TYPE_CHECKING:
    from pyiceberg.catalog import Catalog

__all__ = ["SmokeResult", "main", "run_smoke_test"]

_DEFAULT_NAMESPACE = "nyc"
_DEFAULT_TABLE = "yellow_tripdata"


@dataclass
class SmokeResult:
    """What the run found. `ok` is the process exit condition."""

    ok: bool = True
    steps: list[tuple[str, bool, str]] = field(default_factory=list)
    rows: Any = None
    sql: str | None = None

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.steps.append((name, ok, detail))
        self.ok = self.ok and ok


class _Reporter:
    """Prints each step as it runs and turns exceptions into failed steps."""

    def __init__(self, console: Console, result: SmokeResult, *, verbose: bool) -> None:
        self._console = console
        self._result = result
        self._verbose = verbose

    def step(self, name: str, action: Callable[[], tuple[str, Any]]) -> Any:
        """Run `action`, print the outcome, and return its value (None on failure)."""
        try:
            detail, value = action()
        except Exception as exc:
            self._result.record(name, False, f"{type(exc).__name__}: {exc}")
            self._console.print(f"  [bold red]FAIL[/] {name}")
            self._console.print(f"       [red]{type(exc).__name__}: {exc}[/]")
            if self._verbose:
                self._console.print(f"[dim]{traceback.format_exc()}[/]")
            return None
        self._result.record(name, True, detail)
        self._console.print(
            f"  [bold green]OK[/]   {name}" + (f" [dim]{detail}[/]" if detail else "")
        )
        return value


def _render_settings(console: Console, settings: IcetlSettings) -> None:
    table = RichTable(title="Resolved configuration", title_justify="left", show_lines=False)
    table.add_column("key", style="cyan", no_wrap=True)
    table.add_column("value")
    for key, value in settings.debug_pairs():
        table.add_row(key, value or "[dim](unset)[/]")
    console.print(table)


def _render_rows(console: Console, arrow_table: Any, limit: int) -> None:
    table = RichTable(
        title=f"First {min(limit, arrow_table.num_rows)} row(s)", title_justify="left"
    )
    for name in arrow_table.column_names:
        table.add_column(name, overflow="fold")
    for row in arrow_table.to_pylist():
        table.add_row(
            *("[dim]NULL[/]" if row[c] is None else str(row[c]) for c in arrow_table.column_names)
        )
    console.print(table)


def run_smoke_test(
    *,
    namespace: str = _DEFAULT_NAMESPACE,
    table_name: str = _DEFAULT_TABLE,
    limit: int = 10,
    settings: IcetlSettings | None = None,
    catalog: Catalog | None = None,
    console: Console | None = None,
    verbose: bool = False,
) -> SmokeResult:
    """Run the six checks. `catalog` lets tests supply a local catalog instead."""
    console = console or Console()
    settings = settings or resolve_settings()
    result = SmokeResult()
    reporter = _Reporter(console, result, verbose=verbose)

    console.print()
    console.rule("[bold]icetl connectivity smoke test")
    if verbose:
        _render_settings(console, settings)
    console.print()

    registry = CatalogRegistry(settings)
    if catalog is not None:
        registry.register(settings.catalog.name, catalog)

    # 1 -- catalog
    def connect() -> tuple[str, Catalog]:
        built = registry.get()
        return f"{settings.catalog.type} @ {settings.catalog.uri or 'n/a'}", built

    live_catalog = reporter.step("catalog connection", connect)
    if live_catalog is None:
        return result

    # 2 -- namespaces
    def list_namespaces() -> tuple[str, list[tuple[str, ...]]]:
        namespaces = list(live_catalog.list_namespaces())
        names = [".".join(n) for n in namespaces]
        shown = ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
        return f"{len(names)} found: {shown}", namespaces

    if reporter.step("list namespaces", list_namespaces) is None:
        return result

    # 3 -- tables in the target namespace
    def list_tables() -> tuple[str, list[tuple[str, ...]]]:
        identifiers = list(live_catalog.list_tables(namespace))
        names = [ident[-1] for ident in identifiers]
        if table_name not in names:
            raise LookupError(
                f"table {table_name!r} not in namespace {namespace!r}. Present: "
                + (", ".join(sorted(names)[:20]) or "(none)")
            )
        return f"{len(names)} in {namespace!r}, including {table_name!r}", identifiers

    if reporter.step(f"list tables in {namespace!r}", list_tables) is None:
        return result

    # 4 -- load and describe
    resolver = TableResolver(registry, default_namespace=tuple(namespace.split(".")))

    def describe() -> tuple[str, Any]:
        resolved = resolver.resolve(f"{namespace}.{table_name}")
        iceberg_table = resolved.table
        schema = iceberg_table.schema()
        snapshot = iceberg_table.current_snapshot()
        partition_fields = [f.name for f in iceberg_table.spec().fields]
        # Parentheses, not square brackets: rich would read `[...]` as markup and
        # swallow it.
        detail = (
            f"{len(schema.fields)} columns, "
            f"partitioned by ({', '.join(partition_fields) or 'nothing'}), "
            f"snapshot {snapshot.snapshot_id if snapshot else 'none'}"
        )
        if verbose:
            console.print(f"[dim]{schema}[/]")
        return detail, iceberg_table

    iceberg_table = reporter.step("load table metadata", describe)
    if iceberg_table is None:
        return result

    # 5 -- plan the scan
    def plan() -> tuple[str, list[Any]]:
        tasks = list(iceberg_table.scan(limit=limit).plan_files())
        total_bytes = sum(task.file.file_size_in_bytes for task in tasks)
        with_deletes = sum(1 for task in tasks if task.delete_files)
        detail = f"{len(tasks)} file(s), {total_bytes / 1e6:.1f} MB"
        if with_deletes:
            # Phase 0 reads the parquet directly, which ignores delete files. Say so
            # rather than quietly returning deleted rows.
            detail += f", [yellow]{with_deletes} carry delete files (not applied yet)[/]"
        return detail, tasks

    tasks = reporter.step("plan files", plan)
    if tasks is None:
        return result
    if not tasks:
        result.record("read via DuckDB", False, "no data files to read; the table is empty")
        console.print("  [bold yellow]SKIP[/] read via DuckDB [dim](table has no data files)[/]")
        return result

    # 6 -- read through DuckDB
    def read() -> tuple[str, Any]:
        paths = engine_paths([task.file.file_path for task in tasks])
        engine = DuckDBEngine(settings)
        engine.ensure_object_store(paths)
        # `union_by_name` is the naive read: it matches columns by name, which is
        # wrong for renamed columns (PLAN.md 3.4). Phase 2 replaces it.
        sql = f"SELECT * FROM read_parquet($paths, union_by_name = true) LIMIT {int(limit)}"
        result.sql = sql
        if verbose:
            console.print(f"[dim]{sql}[/]")
            console.print(f"[dim]paths[0] = {paths[0]}[/]")
        arrow_table = engine.arrow(sql, {"paths": paths})
        engine.close()
        return f"{arrow_table.num_rows} row(s), {arrow_table.num_columns} column(s)", arrow_table

    rows = reporter.step("read via DuckDB", read)
    if rows is not None:
        result.rows = rows
        console.print()
        _render_rows(console, rows, limit)

    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="icetl-smoke",
        description="Verify Iceberg REST catalog + object store + DuckDB connectivity.",
    )
    parser.add_argument("--namespace", default=None, help=f"default: {_DEFAULT_NAMESPACE}")
    parser.add_argument("--table", default=_DEFAULT_TABLE)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config key, e.g. --set icetl.catalog.rest.uri=http://host:8182",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="show config, SQL, tracebacks")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    console = Console()

    conf = IcetlConf()
    for override in args.set:
        if "=" not in override:
            console.print(f"[red]--set expects KEY=VALUE, got {override!r}[/]")
            return 2
        key, value = override.split("=", 1)
        conf.set(key, value)

    settings = resolve_settings(conf)
    namespace = args.namespace or (
        ".".join(settings.default_namespace) if settings.default_namespace else _DEFAULT_NAMESPACE
    )

    result = run_smoke_test(
        namespace=namespace,
        table_name=args.table,
        limit=args.limit,
        settings=settings,
        console=console,
        verbose=args.verbose,
    )

    console.print()
    if result.ok:
        console.print("[bold green]All checks passed.[/] Phase 1 can begin.")
        return 0

    failed = [name for name, ok, _ in result.steps if not ok]
    console.print(f"[bold red]Failed:[/] {', '.join(failed)}")
    console.print("[dim]Re-run with --verbose for configuration and tracebacks.[/]")
    return 1


if __name__ == "__main__":
    sys.exit(main())

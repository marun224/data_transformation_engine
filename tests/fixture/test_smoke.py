"""The smoke test itself, run against the local catalog.

Phase 0's deliverable is a diagnostic that either passes or names the layer that
failed. Both halves of that need coverage, so the failure paths are tested too.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from pyiceberg.catalog.sql import SqlCatalog
from rich.console import Console

from icetl.conf import EngineSettings, IcetlSettings
from icetl.diagnostics.smoke import run_smoke_test


@pytest.fixture
def quiet_console() -> Console:
    """A console that renders to nothing, so test output stays readable."""
    return Console(quiet=True, width=200)


@pytest.fixture
def smoke_settings(local_settings: IcetlSettings, tmp_path: Path) -> IcetlSettings:
    return replace(local_settings, engine=EngineSettings(temp_directory=str(tmp_path)))


def test_all_steps_pass_against_the_local_catalog(
    smoke_settings: IcetlSettings, catalog: SqlCatalog, quiet_console: Console
) -> None:
    result = run_smoke_test(
        namespace="fx",
        table_name="plain",
        limit=10,
        settings=smoke_settings,
        catalog=catalog,
        console=quiet_console,
    )

    assert result.ok, [step for step in result.steps if not step[1]]
    assert [name for name, _, _ in result.steps] == [
        "catalog connection",
        "list namespaces",
        "list tables in 'fx'",
        "load table metadata",
        "plan files",
        "read via DuckDB",
    ]
    assert result.rows is not None
    assert result.rows.num_rows == 5


def test_limit_is_respected(
    smoke_settings: IcetlSettings, catalog: SqlCatalog, quiet_console: Console
) -> None:
    result = run_smoke_test(
        namespace="fx",
        table_name="wide",
        limit=3,
        settings=smoke_settings,
        catalog=catalog,
        console=quiet_console,
    )
    assert result.ok
    assert result.rows.num_rows == 3


def test_missing_table_fails_at_the_right_step(
    smoke_settings: IcetlSettings, catalog: SqlCatalog, quiet_console: Console
) -> None:
    """A wrong table name must be reported as a table problem, not a catalog one."""
    result = run_smoke_test(
        namespace="fx",
        table_name="does_not_exist",
        settings=smoke_settings,
        catalog=catalog,
        console=quiet_console,
    )

    assert not result.ok
    failed = [name for name, ok, _ in result.steps if not ok]
    assert failed == ["list tables in 'fx'"]
    # The message should list what *is* there, so the fix is obvious.
    detail = next(d for name, ok, d in result.steps if not ok)
    assert "plain" in detail


def test_missing_namespace_fails_at_the_right_step(
    smoke_settings: IcetlSettings, catalog: SqlCatalog, quiet_console: Console
) -> None:
    result = run_smoke_test(
        namespace="no_such_namespace",
        table_name="plain",
        settings=smoke_settings,
        catalog=catalog,
        console=quiet_console,
    )
    assert not result.ok
    assert [name for name, ok, _ in result.steps if not ok] == [
        "list tables in 'no_such_namespace'"
    ]


def test_unreachable_catalog_fails_fast(quiet_console: Console) -> None:
    """No catalog injected and a dead URI: the first step must fail, not hang."""
    from icetl.conf import CatalogSettings

    settings = IcetlSettings(
        catalog=CatalogSettings(name="rest", type="rest", uri="http://127.0.0.1:1")
    )
    result = run_smoke_test(settings=settings, console=quiet_console)

    assert not result.ok
    assert result.steps[0][0] == "catalog connection"
    assert result.steps[0][1] is False
    assert len(result.steps) == 1, "must stop at the first failure, not cascade"

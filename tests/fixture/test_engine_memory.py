"""Phase 10: memory, spill, and the parallelism question that measurement settled.

P7 says "use every core, spill when memory runs out". The second half is the one that
needed checking, because DuckDB will not spill at all without a temp directory and
says nothing when it has none -- it simply raises `Out of Memory` on a query it could
have finished.

`TestSpillIsWhatMakesTheQueryFinish` is the test that earns its runtime: the same
query at the same memory limit **fails** without a temp directory and **succeeds**
with one. A test that only asserted the setting was applied would still pass on a
DuckDB that had stopped honouring it.

There is a floor, and it is recorded here rather than discovered later: below roughly
400 MB this workload raises whether or not spill is configured, because DuckDB needs
a working set of buffers before it has anything to spill *from*. So the temp
directory buys a query that is too big for memory, not a query with no memory.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from icetl.conf import EngineSettings, IcetlSettings
from icetl.errors import QueryExecutionException
from icetl.exec import DuckDBEngine

if TYPE_CHECKING:
    from collections.abc import Iterator

#: 3M rows of ~200 bytes sorted inside a 400 MB limit: about 600 MB of working set,
#: so it cannot be done in memory, and it takes under two seconds when it can spill.
SPILLING_QUERY = (
    "SELECT count(*) FROM (SELECT i, repeat('x', 200) AS pad FROM range(3000000) t(i) "
    "ORDER BY pad, i)"
)
SPILL_LIMIT = "400MB"


def engine_with(tmp_path: Path, **engine: object) -> DuckDBEngine:
    return DuckDBEngine(IcetlSettings(engine=EngineSettings(**engine)))  # type: ignore[arg-type]


class TestSpillIsWhatMakesTheQueryFinish:
    """The claim, tested as a difference rather than as a setting."""

    @pytest.fixture
    def spilling(self, tmp_path: Path) -> Iterator[DuckDBEngine]:
        built = engine_with(
            tmp_path, memory_limit=SPILL_LIMIT, temp_directory=str(tmp_path / "spill")
        )
        yield built
        built.close()

    def test_a_query_too_big_for_memory_finishes(self, spilling: DuckDBEngine) -> None:
        assert spilling.arrow(SPILLING_QUERY).column(0)[0].as_py() == 3_000_000

    def test_the_same_query_fails_with_spill_turned_off(self, tmp_path: Path) -> None:
        """The control. Without it the test above proves only that 400 MB is enough.

        DuckDB treats an empty `temp_directory` as "do not spill", which is the state
        the engine would be in if it stopped configuring one.
        """
        engine = engine_with(tmp_path, memory_limit=SPILL_LIMIT)
        try:
            engine.connection.execute("SET temp_directory=''")
            with pytest.raises(QueryExecutionException, match="Out of Memory"):
                engine.arrow(SPILLING_QUERY)
        finally:
            engine.close()


class TestTheEngineConfiguresSpill:
    def test_a_temp_directory_is_always_set(self, tmp_path: Path) -> None:
        """Even with nothing configured -- the default is a real path, not ''."""
        engine = engine_with(tmp_path)
        try:
            setting = engine.connection.execute(
                "SELECT current_setting('temp_directory')"
            ).fetchone()
            assert setting is not None and setting[0]
        finally:
            engine.close()

    def test_the_configured_directory_is_used_and_created(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "spill"
        engine = engine_with(tmp_path, temp_directory=str(target))
        try:
            setting = engine.connection.execute(
                "SELECT current_setting('temp_directory')"
            ).fetchone()
            assert setting is not None
            assert Path(setting[0]) == target
            assert target.is_dir()
        finally:
            engine.close()

    def test_a_memory_limit_is_applied_when_given(self, tmp_path: Path) -> None:
        engine = engine_with(tmp_path, memory_limit="512MB", temp_directory=str(tmp_path / "s"))
        try:
            setting = engine.connection.execute("SELECT current_setting('memory_limit')").fetchone()
            assert setting is not None and "MiB" in setting[0]
        finally:
            engine.close()

    def test_threads_are_left_to_duckdb_unless_asked(self, tmp_path: Path) -> None:
        """P7, and now measured: 2, 4 and 8 threads are indistinguishable on the
        benchmark suite, and 16 -- twice this machine's logical cores -- is worse. So
        the default is DuckDB's own, and icetl only ever narrows it. See BENCHMARKS.md.
        """
        default = engine_with(tmp_path, temp_directory=str(tmp_path / "a"))
        narrowed = engine_with(tmp_path, threads=2, temp_directory=str(tmp_path / "b"))
        try:
            unset = default.connection.execute("SELECT current_setting('threads')").fetchone()
            asked = narrowed.connection.execute("SELECT current_setting('threads')").fetchone()
            assert unset is not None and asked is not None
            assert int(asked[0]) == 2
            assert int(unset[0]) >= 1
        finally:
            default.close()
            narrowed.close()


class TestSettingsResolution:
    def test_the_default_temp_directory_is_stable(self) -> None:
        settings = EngineSettings()
        assert settings.resolved_temp_directory() == EngineSettings().resolved_temp_directory()
        assert settings.resolved_temp_directory().endswith("icetl-duckdb-spill")

    def test_an_explicit_temp_directory_wins(self) -> None:
        assert EngineSettings(temp_directory="/tmp/x").resolved_temp_directory() == "/tmp/x"

    def test_the_debug_view_says_auto_where_nothing_is_set(self) -> None:
        pairs = dict(IcetlSettings().debug_pairs())
        assert pairs["engine.threads"] == "(auto)"
        assert pairs["engine.memory_limit"] == "(auto)"

    def test_the_debug_view_shows_what_was_set(self) -> None:
        settings = replace(IcetlSettings(), engine=EngineSettings(threads=4, memory_limit="2GB"))
        pairs = dict(settings.debug_pairs())
        assert pairs["engine.threads"] == "4"
        assert pairs["engine.memory_limit"] == "2GB"

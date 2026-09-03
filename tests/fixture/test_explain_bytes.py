"""Phase 10: `explain()` reporting the bytes a query reads, not the bytes it selects.

`bytes_scanned` used to be the selected *files'* size, so a query reading 2 of 200
columns looked exactly as expensive as one reading all 200 -- the number was there to
show whether pushdown worked and could not show it (FINDINGS.md 3.3).

Iceberg's manifests carry a compressed size per column, so the honest answer is a sum
over the selected columns. Both numbers are now printed, because either alone is
unreadable: bytes read says how much work the query does, and the file total says how
much of the table pruning failed to remove.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icetl.exec.scan_planner import format_bytes
from icetl.sql import functions as F

if TYPE_CHECKING:
    from icetl.exec.scan_planner import ScanPlan
    from icetl.sql.dataframe import DataFrame
    from icetl.sql.session import Session


def scan_of(session: Session, frame: DataFrame, key: str) -> ScanPlan:
    compiled = session._compile(frame._plan, frame._sources, frame.columns)
    return next(scan for scan in compiled.scans if scan.source.key == key)


class TestColumnPruningIsVisible:
    """The 200-column fixture, which is what 3.6 built it for."""

    def test_two_columns_cost_a_fraction_of_two_hundred(self, session: Session) -> None:
        narrow = scan_of(session, session.table("fx.wide").select("id", "col_001"), "fx.wide")
        wide = scan_of(session, session.table("fx.wide"), "fx.wide")
        assert narrow.bytes_scanned < wide.bytes_scanned / 10

    def test_the_file_total_is_the_same_either_way(self, session: Session) -> None:
        """The distinction the old number could not make: same files, different read."""
        narrow = scan_of(session, session.table("fx.wide").select("id", "col_001"), "fx.wide")
        wide = scan_of(session, session.table("fx.wide"), "fx.wide")
        assert narrow.bytes_total == wide.bytes_total
        assert narrow.files_scanned == wide.files_scanned

    def test_bytes_read_never_exceeds_the_files(self, session: Session) -> None:
        for reference in ("fx.plain", "fx.wide", "fx.partitioned", "fx.nested"):
            scan = scan_of(session, session.table(reference), reference)
            assert 0 < scan.bytes_scanned <= scan.bytes_total

    def test_more_columns_never_cost_less(self, session: Session) -> None:
        one = scan_of(session, session.table("fx.wide").select("id"), "fx.wide")
        three = scan_of(
            session, session.table("fx.wide").select("id", "col_001", "col_002"), "fx.wide"
        )
        assert one.bytes_scanned < three.bytes_scanned

    def test_a_nested_column_counts_its_leaves(self, session: Session) -> None:
        """A struct is several parquet columns, sized by their own field-ids.

        Reading the top-level id alone would report zero bytes for `person`, which is
        the failure mode this walk exists to avoid.
        """
        person = scan_of(session, session.table("fx.nested").select("person"), "fx.nested")
        just_id = scan_of(session, session.table("fx.nested").select("id"), "fx.nested")
        assert person.bytes_scanned > 0
        assert person.bytes_scanned != just_id.bytes_scanned

    def test_a_list_and_a_map_are_counted_too(self, session: Session) -> None:
        for column in ("tags", "scores"):
            scan = scan_of(session, session.table("fx.nested").select(column), "fx.nested")
            assert scan.bytes_scanned > 0


class TestFilePruningIsStillVisible:
    def test_a_partition_filter_lowers_both_numbers(self, session: Session) -> None:
        whole = scan_of(session, session.table("fx.partitioned"), "fx.partitioned")
        pruned = scan_of(
            session,
            session.table("fx.partitioned").filter(F.col("as_at_date") == "2026-08-16"),
            "fx.partitioned",
        )
        assert pruned.files_scanned < whole.files_scanned
        assert pruned.bytes_total < whole.bytes_total
        assert pruned.bytes_scanned < whole.bytes_scanned


class TestWhatExplainPrints:
    def test_both_numbers_appear_when_they_differ(self, session: Session) -> None:
        scan = scan_of(session, session.table("fx.wide").select("id", "col_001"), "fx.wide")
        assert " of " in scan.describe()
        assert scan.describe().count(" of ") == 2  # files, and bytes

    def test_the_scan_line_reaches_explain(self, session: Session, capsys) -> None:  # type: ignore[no-untyped-def]
        session.table("fx.wide").select("id", "col_001").explain()
        printed = capsys.readouterr().out
        assert "2 of 200" in printed
        assert "KB of " in printed

    def test_an_empty_scan_still_describes_itself(self, session: Session) -> None:
        pruned = scan_of(
            session,
            session.table("fx.partitioned").filter(F.col("as_at_date") == "1999-01-01"),
            "fx.partitioned",
        )
        assert pruned.describe() == "no data files"


class TestFormatBytes:
    """A fixed `MB` printed a 16x difference as '0.0 MB of 0.1 MB'."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1557, "1.5 KB"),
            (1024 * 1024, "1.0 MB"),
            (5 * 1024 * 1024, "5.0 MB"),
            (3 * 1024 * 1024 * 1024, "3.0 GB"),
        ],
    )
    def test_it_scales_to_the_number(self, count: int, expected: str) -> None:
        assert format_bytes(count) == expected

    def test_a_very_large_count_stays_in_gb(self) -> None:
        """No unit above GB, so a huge number reads as a huge number of GB."""
        assert format_bytes(4096 * 1024 * 1024 * 1024).endswith(" GB")

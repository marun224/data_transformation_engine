"""The annotations side table, and the merge that folds it back to one scan per table.

Two references to the same table can each carry their own filter, so each gets its
own annotation. But `substitute_sources` replaces every reference to a table with the
*same* relation, so the requests have to be merged before planning. The merge must
always widen -- union the columns, OR the predicates -- because a widened scan reads
rows it did not need and the SQL filter throws them away (PLAN.md 3.2), whereas a
narrowed scan loses them for good.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyiceberg.expressions import AlwaysTrue, Or
from sqlglot import exp

from icetl.plan.annotations import PlanAnnotations, ScanRequest
from tests.predicates import EqualTo, GreaterThan

if TYPE_CHECKING:
    from icetl.plan.builder import ScanSource


class TestScanRequest:
    def test_no_predicate_by_default(self, sources: dict[str, ScanSource]) -> None:
        request = ScanRequest(source=sources["fx.plain"])
        assert not request.has_predicate
        assert request.columns is None

    def test_merging_columns_takes_the_union(self, sources: dict[str, ScanSource]) -> None:
        source = sources["fx.plain"]
        left = ScanRequest(source, columns=("id",), predicate=EqualTo("id", 1))
        right = ScanRequest(source, columns=("vendor",), predicate=EqualTo("id", 2))
        assert left.merge(right).columns == ("id", "vendor")

    def test_merging_with_all_columns_gives_all_columns(
        self, sources: dict[str, ScanSource]
    ) -> None:
        source = sources["fx.plain"]
        left = ScanRequest(source, columns=("id",), predicate=EqualTo("id", 1))
        right = ScanRequest(source, columns=None, predicate=EqualTo("id", 2))
        assert left.merge(right).columns is None

    def test_merging_predicates_takes_the_disjunction(self, sources: dict[str, ScanSource]) -> None:
        """A file is needed if *either* reference might want it."""
        source = sources["fx.plain"]
        left = ScanRequest(source, predicate=EqualTo("id", 1))
        right = ScanRequest(source, predicate=GreaterThan("id", 5))
        assert left.merge(right).predicate == Or(EqualTo("id", 1), GreaterThan("id", 5))

    def test_an_unfiltered_reference_disables_pruning_for_the_table(
        self, sources: dict[str, ScanSource]
    ) -> None:
        """One reference reading the whole table means no file may be skipped."""
        source = sources["fx.plain"]
        filtered = ScanRequest(source, predicate=EqualTo("id", 1))
        unfiltered = ScanRequest(source)
        assert isinstance(filtered.merge(unfiltered).predicate, AlwaysTrue)
        assert isinstance(unfiltered.merge(filtered).predicate, AlwaysTrue)

    def test_unpushed_filters_are_carried_and_deduplicated(
        self, sources: dict[str, ScanSource]
    ) -> None:
        source = sources["fx.plain"]
        left = ScanRequest(source, unpushed=("UPPER(vendor) = 'A'",))
        right = ScanRequest(source, unpushed=("UPPER(vendor) = 'A'", "id % 2 = 0"))
        assert left.merge(right).unpushed == ("UPPER(vendor) = 'A'", "id % 2 = 0")


class TestPlanAnnotations:
    def test_annotations_are_keyed_by_node_identity(self, sources: dict[str, ScanSource]) -> None:
        root = exp.select("*").from_("fx.plain")
        annotations = PlanAnnotations(root)
        first, second = exp.to_table("fx.plain"), exp.to_table("fx.plain")

        annotations.annotate(first, ScanRequest(sources["fx.plain"], columns=("id",)))
        annotations.annotate(second, ScanRequest(sources["fx.plain"], columns=("vendor",)))

        assert len(annotations) == 2
        assert annotations.get(first) is not annotations.get(second)

    def test_an_unannotated_node_is_none(self, sources: dict[str, ScanSource]) -> None:
        annotations = PlanAnnotations(exp.select("*"))
        assert annotations.get(exp.to_table("fx.plain")) is None

    def test_merged_folds_both_references_into_one_request(
        self, sources: dict[str, ScanSource]
    ) -> None:
        root = exp.select("*").from_("fx.plain")
        annotations = PlanAnnotations(root)
        source = sources["fx.plain"]
        annotations.annotate(
            exp.to_table("fx.plain"),
            ScanRequest(source, columns=("id",), predicate=EqualTo("id", 1)),
        )
        annotations.annotate(
            exp.to_table("fx.plain"),
            ScanRequest(source, columns=("vendor",), predicate=EqualTo("id", 2)),
        )

        merged = annotations.merged()
        assert set(merged) == {"fx.plain"}
        assert merged["fx.plain"].columns == ("id", "vendor")

    def test_empty_annotations_are_falsey(self, sources: dict[str, ScanSource]) -> None:
        assert not PlanAnnotations(exp.select("*"))
        assert PlanAnnotations(exp.select("*")).merged() == {}

"""A side table carrying scan metadata alongside the plan, without mutating it.

sqlglot nodes are shared, copied, and rewritten freely by the optimizer, so hanging
our own attributes off them would be fragile and would leak our concepts into trees
we hand to sqlglot. Instead the pruning facts -- which columns a scan needs, which
predicate can be pushed to PyIceberg -- live here, keyed by node identity, with the
root expression held so the nodes cannot be garbage collected out from under the
keys.

Two references to the same table (a self-join, say) get an entry each, because each
can carry a different filter. `merged()` folds them back to one request per table for
the compile step, which substitutes one source expression per reference:

    columns    union      -- reading a column nobody asked for is merely wasteful
    predicate  OR         -- a file is needed if *either* reference might want it

Both directions are the safe one. Widening the scan can never drop a row, and the
filter is re-applied in SQL regardless (3.2), so a merge that over-reads costs time
and never correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyiceberg.expressions import AlwaysTrue, Or

if TYPE_CHECKING:
    from pyiceberg.expressions import BooleanExpression
    from sqlglot import exp

    from icetl.plan.builder import ScanSource

__all__ = ["PlanAnnotations", "ScanRequest"]


@dataclass(frozen=True)
class ScanRequest:
    """What one table reference in the plan actually needs read.

    `columns` of `None` means "every column" -- the honest answer when the optimizer
    could not bind the plan, and the one that cannot lose data. An empty tuple is a
    different thing entirely: a `count(*)` needs no column values at all.
    """

    source: ScanSource
    columns: tuple[str, ...] | None = None
    predicate: BooleanExpression = field(default_factory=AlwaysTrue)
    # Conjuncts we could not translate, kept as SQL text so `explain()` can say what
    # was left behind rather than just how much was pushed.
    unpushed: tuple[str, ...] = ()

    @property
    def has_predicate(self) -> bool:
        return not isinstance(self.predicate, AlwaysTrue)

    def merge(self, other: ScanRequest) -> ScanRequest:
        if self.columns is None or other.columns is None:
            columns = None
        else:
            columns = tuple(dict.fromkeys((*self.columns, *other.columns)))

        if not self.has_predicate or not other.has_predicate:
            # One reference reads the table unfiltered, so nothing may be pruned.
            predicate: BooleanExpression = AlwaysTrue()
        else:
            predicate = Or(self.predicate, other.predicate)

        return ScanRequest(
            source=self.source,
            columns=columns,
            predicate=predicate,
            unpushed=tuple(dict.fromkeys((*self.unpushed, *other.unpushed))),
        )


class PlanAnnotations:
    """Scan requests keyed by the plan node that produced them."""

    def __init__(self, root: exp.Expression) -> None:
        # Held only to keep the nodes -- and therefore the identity keys -- alive.
        self._root = root
        self._by_node: dict[int, ScanRequest] = {}

    def annotate(self, node: exp.Expression, request: ScanRequest) -> None:
        self._by_node[id(node)] = request

    def get(self, node: exp.Expression) -> ScanRequest | None:
        return self._by_node.get(id(node))

    def merged(self) -> dict[str, ScanRequest]:
        """One request per table reference, folded across every node naming it."""
        out: dict[str, ScanRequest] = {}
        for request in self._by_node.values():
            key = request.source.key
            existing = out.get(key)
            out[key] = request if existing is None else existing.merge(request)
        return out

    def __len__(self) -> int:
        return len(self._by_node)

    def __bool__(self) -> bool:
        return bool(self._by_node)

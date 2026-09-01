"""`Window` and `WindowSpec` -- the frame a window function is evaluated over.

A `WindowSpec` is a **partition**, an **ordering**, and a **frame**, and it holds no
reference to any DataFrame. That is deliberate and matches the reference: one spec can
be reused across frames, so a column name inside it stays unresolved until the plan it
lands in is analysed.

Every method returns a *new* spec. `Window.partitionBy("g").orderBy("x")` builds two
objects rather than mutating one, so a spec passed to two different columns cannot be
changed by one of them.

**The frame default is the thing to know.** When a spec has an ordering but no explicit
frame, SQL -- and the reference, and DuckDB -- use
`RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, which is *not* the same as the
`ROWS` equivalent: `RANGE` includes every row that ties with the current one, so a
running total over a column with duplicates jumps in steps rather than climbing one row
at a time. With no ordering the frame is the whole partition. Neither default is spelled
here -- both are left to SQL, so the DataFrame surface and `Session.sql()` inherit the
same behaviour from the same place (P1).

Boundaries follow the reference's sign convention, which reads oddly until you see it
written down: **negative is preceding, positive is following, zero is the current row.**
So `rowsBetween(-1, 1)` is a three-row sliding window centred on the current row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlglot import exp

from icetl.errors import EngineTypeError, EngineValueError
from icetl.sql.column import Column

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["Window", "WindowSpec"]

#: The reference spells its unbounded markers as the extremes of a signed 64-bit
#: integer, and scripts written against it pass those numbers directly. Any offset at or
#: beyond one of them means "unbounded", rather than a frame nine quintillion rows wide.
_UNBOUNDED_PRECEDING = -(2**63)
_UNBOUNDED_FOLLOWING = 2**63 - 1


class WindowSpec:
    """A partition, an ordering and a frame. Immutable; every method returns a new one."""

    def __init__(
        self,
        partition: Sequence[exp.Expression] = (),
        order: Sequence[exp.Expression] = (),
        frame: tuple[str, Any, Any] | None = None,
    ) -> None:
        self._partition = list(partition)
        self._order = list(order)
        self._frame = frame

    def __repr__(self) -> str:
        parts = []
        if self._partition:
            parts.append(f"partitionBy={[e.sql() for e in self._partition]}")
        if self._order:
            parts.append(f"orderBy={[e.sql() for e in self._order]}")
        if self._frame:
            parts.append(f"{self._frame[0].lower()}Between={self._frame[1]}..{self._frame[2]}")
        return f"WindowSpec({', '.join(parts) or 'unbounded'})"

    # -- building ------------------------------------------------------------

    def partitionBy(self, *cols: Any) -> WindowSpec:
        """Split the rows into independent groups, each windowed on its own."""
        return WindowSpec(_columns("partitionBy", cols), self._order, self._frame)

    def orderBy(self, *cols: Any) -> WindowSpec:
        """Order the rows within each partition.

        Null placement is not spelled here. `_fix_null_ordering` in `sql/conformance.py`
        is a tree pass over every `exp.Ordered`, and a window's ordering is made of the
        same nodes as a top-level one -- so this inherits nulls-first-ascending without
        knowing about it, exactly as `DataFrame.orderBy` does.
        """
        ordered = [_as_ordered(item) for item in _columns("orderBy", cols)]
        return WindowSpec(self._partition, ordered, self._frame)

    def rowsBetween(self, start: int, end: int) -> WindowSpec:
        """A frame counted in **rows**: `rowsBetween(-1, 1)` is the row either side.

        Ties are irrelevant here -- two equal values are still two rows. Use this when
        you mean "the previous three rows"; use `rangeBetween` when you mean "everything
        within three of this value".
        """
        return WindowSpec(
            self._partition, self._order, ("ROWS", *_frame("rowsBetween", start, end))
        )

    def rangeBetween(self, start: int, end: int) -> WindowSpec:
        """A frame counted in **values** of the ordering column.

        `rangeBetween(0, 0)` is every row tying with this one, which is why it is not the
        same as `rowsBetween(0, 0)`. Offsets other than the unbounded markers and zero
        need a single ordering column that they can be added to.
        """
        return WindowSpec(
            self._partition, self._order, ("RANGE", *_frame("rangeBetween", start, end))
        )

    # -- applying ------------------------------------------------------------

    def _apply(self, function: exp.Expression) -> exp.Window:
        """`function OVER (this spec)`."""
        arguments: dict[str, Any] = {"this": function, "over": "OVER"}
        if self._partition:
            arguments["partition_by"] = [item.copy() for item in self._partition]
        if self._order:
            arguments["order"] = exp.Order(expressions=[item.copy() for item in self._order])
        if self._frame is not None:
            kind, start, end = self._frame
            arguments["spec"] = _window_spec(kind, start, end)
        return exp.Window(**arguments)


class Window:
    """Entry point for building a `WindowSpec`.

    Never instantiated -- every member is a class-level constant or a static factory, as
    in the reference, so `Window.partitionBy("g")` is the whole of the API.
    """

    #: The reference's frame boundary markers, and the same values it uses for them.
    unboundedPreceding = _UNBOUNDED_PRECEDING
    unboundedFollowing = _UNBOUNDED_FOLLOWING
    currentRow = 0

    def __init__(self) -> None:  # pragma: no cover - defensive
        raise EngineTypeError("Window is a namespace, not something to instantiate.")

    @staticmethod
    def partitionBy(*cols: Any) -> WindowSpec:
        return WindowSpec().partitionBy(*cols)

    @staticmethod
    def orderBy(*cols: Any) -> WindowSpec:
        return WindowSpec().orderBy(*cols)

    @staticmethod
    def rowsBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec().rowsBetween(start, end)

    @staticmethod
    def rangeBetween(start: int, end: int) -> WindowSpec:
        return WindowSpec().rangeBetween(start, end)


def _columns(method: str, cols: tuple[Any, ...]) -> list[exp.Expression]:
    """Column names or Columns, accepting a single list as the reference does."""
    items = list(cols)
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = list(items[0])
    if not items:
        raise EngineValueError(f"{method}() needs at least one column.")

    expressions: list[exp.Expression] = []
    for item in items:
        if isinstance(item, str):
            # Unresolved on purpose: a spec belongs to no frame, so the name is bound
            # when the plan it lands in is analysed.
            expressions.append(exp.column(item, quoted=True))
        elif isinstance(item, Column):
            expressions.append(item._expression.copy())
        else:
            raise EngineTypeError(
                f"{method}() takes column names or Column objects, got {type(item).__name__}."
            )
    return expressions


def _as_ordered(expression: exp.Expression) -> exp.Expression:
    if isinstance(expression, exp.Ordered):
        return expression
    return exp.Ordered(this=expression)


def _frame(method: str, start: Any, end: Any) -> tuple[Any, Any]:
    for label, value in (("start", start), ("end", end)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise EngineTypeError(
                f"{method}() expects {label} as an int, got {type(value).__name__}."
            )
    if start > end:
        raise EngineValueError(
            f"{method}({start}, {end}) is empty: the frame's start must not be after its end."
        )
    return start, end


def _boundary(value: int) -> tuple[Any, str | None]:
    """One frame edge as sqlglot spells it: (bound, side).

    The reference's sign convention: negative preceding, positive following, zero the
    current row. `abs()` is applied because SQL states the distance and the direction
    separately, where the reference packs both into the sign.
    """
    if value <= _UNBOUNDED_PRECEDING:
        return "UNBOUNDED", "PRECEDING"
    if value >= _UNBOUNDED_FOLLOWING:
        return "UNBOUNDED", "FOLLOWING"
    if value == 0:
        return "CURRENT ROW", None
    side = "PRECEDING" if value < 0 else "FOLLOWING"
    return exp.Literal.number(str(abs(value))), side


def _window_spec(kind: str, start: Any, end: Any) -> exp.WindowSpec:
    start_bound, start_side = _boundary(start)
    end_bound, end_side = _boundary(end)
    arguments: dict[str, Any] = {"kind": kind, "start": start_bound, "end": end_bound}
    if start_side is not None:
        arguments["start_side"] = start_side
    if end_side is not None:
        arguments["end_side"] = end_side
    return exp.WindowSpec(**arguments)

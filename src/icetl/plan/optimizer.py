"""The optimizer pipeline: the same rules for `Session.sql()` and the DataFrame API.

The DataFrame builder nests a subquery for every operation it cannot safely merge,
so a five-call chain arrives here as five levels of `SELECT * FROM (...)`. That shape
is correct but opaque: nothing can tell which columns the query really needs, or
which conjunct belongs to which scan, until it has been flattened. Flattening it is
what makes pushdown possible at all, which is why this runs before
`icetl.plan.pushdown` rather than instead of it.

**Rules are applied one at a time, not through sqlglot's `optimize()`.** Two reasons:
the sequence is the design and deserves to be readable, and a rule that fails on an
exotic plan should cost us only that rule. Each stage is individually
semantics-preserving, so keeping the last stage that succeeded is always sound.

**The output-name guarantee.** `qualify` renames unaliased projections to `_col_0`,
which for us would be a wrong answer -- the reference engine calls that column `sum(amount)` and
scripts index on the name. So the optimized tree is adopted only if its output
columns can be made to match the names analysis already computed, and the top-level
projections are re-aliased to those names before it is used. Anything that cannot be
reconciled is discarded and the original plan runs instead: a slower plan is a cost,
a renamed column is a bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.merge_subqueries import merge_subqueries
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.pushdown_predicates import pushdown_predicates
from sqlglot.optimizer.pushdown_projections import pushdown_projections
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.simplify import simplify

from icetl.plan.schema import DIALECT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlglot.schema import MappingSchema

__all__ = ["RULES", "OptimizedPlan", "optimize_plan"]


@dataclass(frozen=True)
class OptimizedPlan:
    """What the pipeline produced, and whether it was fit to use.

    `applied` is False when the original plan is being returned unchanged, and
    `note` then says why -- which is what `explain(mode="extended")` prints, so a
    query that mysteriously fails to prune can be diagnosed without a debugger.
    """

    original: exp.Expression
    optimized: exp.Expression
    applied: bool
    note: str | None = None
    # Rules that ran cleanly, in order. Empty when binding failed at the first step.
    stages: tuple[str, ...] = ()


# The pipeline, in the order it runs. The order is the design; each entry says what
# the rule buys us and therefore what is lost if it is the one that fails.
RULES: tuple[str, ...] = (
    # Resolve every column to a table and expand `SELECT *` into real columns.
    # Nothing after this achieves anything without it -- see PLAN.md 3.1.
    "qualify",
    # Drop columns no ancestor consumes. This is projection pushdown (3.6).
    "pushdown_projections",
    # Rewrite to CNF, so `pushdown_predicates` sees conjuncts it can move singly.
    "normalize",
    # Move each conjunct as close to its scan as it can legally go.
    "pushdown_predicates",
    # Collapse the subquery nesting the DataFrame builder created, so filters and
    # projections end up adjacent to the table they apply to.
    "merge_subqueries",
    # Constant folding and predicate tidying, last so it sees the merged tree.
    "simplify",
)


def _apply(name: str, expression: exp.Expression, schema: MappingSchema) -> exp.Expression:
    if name == "qualify":
        return qualify(expression, schema=schema, dialect=DIALECT)
    if name == "pushdown_projections":
        return pushdown_projections(expression, schema=schema)
    if name == "normalize":
        # sqlglot annotates this one with its looser `Expr` base.
        return cast(exp.Expression, normalize(expression))
    if name == "pushdown_predicates":
        return pushdown_predicates(expression, dialect=DIALECT)
    if name == "merge_subqueries":
        return merge_subqueries(expression)
    if name == "simplify":
        return simplify(expression, dialect=DIALECT)
    raise AssertionError(f"unknown rule {name!r}")  # pragma: no cover


def _named_selects(expression: exp.Expression) -> list[str]:
    """The output column names of a query node, or `[]` for anything else."""
    return list(expression.named_selects) if isinstance(expression, exp.Query) else []


def _restore_output_names(
    expression: exp.Expression, expected: Sequence[str]
) -> exp.Expression | None:
    """Re-alias top-level projections to `expected`, or return None if it cannot be done.

    Positional re-aliasing is sound because every rule in the pipeline preserves the
    number and order of output columns -- and the length check below is what makes
    that an assertion rather than an assumption.
    """
    if _named_selects(expression) == list(expected):
        return expression

    # A UNION's names come from its first branch, several levels down; rather than
    # reach in and rewrite that, decline and let the original plan run.
    if not isinstance(expression, exp.Select):
        return None

    projections = expression.expressions
    if len(projections) != len(expected):
        return None
    if any(isinstance(projection, exp.Star) for projection in projections):
        return None

    for index, name in enumerate(expected):
        projection = projections[index]
        if projection.alias_or_name == name:
            continue
        projections[index] = exp.alias_(projection.unalias(), name, quoted=True)
    return expression


def optimize_plan(
    plan: exp.Expression, schema: MappingSchema, output_names: Sequence[str]
) -> OptimizedPlan:
    """Optimize `plan`, falling back to it untouched if the result cannot be trusted.

    `output_names` is the analysed schema's column list -- the names the caller has
    already promised the user -- and is what the result is held to.
    """
    current = plan.copy()
    stages: list[str] = []
    note: str | None = None

    for name in RULES:
        try:
            current = _apply(name, current, schema)
        except (OptimizeError, KeyError, ValueError, TypeError, ArithmeticError) as exc:
            # A rule that cannot handle this plan is not an error the user caused.
            # Keep what the earlier rules achieved and stop.
            #
            # `ArithmeticError` is here because `simplify` constant-folds literal
            # arithmetic, and the arithmetic can fail: `SELECT 1.0 / 0.0` reaches
            # Python's decimal division and raises `decimal.DivisionByZero`. The reference engine
            # answers NULL, and does so here too once the rule is skipped, because
            # the reference engine parser's own `NULLIF` guard is still in the tree. The integer
            # spelling `1 / 0` never showed this: `simplify` declines to fold integer
            # division at all, which is why the rule was reachable only with a decimal
            # literal. Overflow and `decimal.InvalidOperation` arrive the same way.
            note = f"stopped at {name}: {type(exc).__name__}: {exc}"
            break
        stages.append(name)

    if not stages:
        return OptimizedPlan(plan, plan, applied=False, note=note or "no rule applied")

    restored = _restore_output_names(current, output_names)
    if restored is None:
        return OptimizedPlan(
            plan,
            plan,
            applied=False,
            note=(
                "discarded: the optimized plan's output columns "
                f"{_named_selects(current)} could not be reconciled with "
                f"{list(output_names)}"
            ),
            stages=tuple(stages),
        )

    return OptimizedPlan(plan, restored, applied=True, note=note, stages=tuple(stages))

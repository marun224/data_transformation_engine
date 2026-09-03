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
columns can be made to match the names analysis already computed, and the projections
that name the output are re-aliased before it is used -- the top-level ones for a
SELECT, the leftmost branch's for a set operation. Anything that cannot be reconciled
is discarded and the original plan runs instead: a slower plan is a cost, a renamed
column is a bug.
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

from icetl.plan.cardinality import contains_generator
from icetl.plan.schema import DIALECT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlglot.schema import MappingSchema

__all__ = ["RULES", "UNSAFE_WITH_GENERATORS", "OptimizedPlan", "optimize_plan"]


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

#: Rules that move a projection out of the scope that defines it. Each is sound for a
#: scalar expression and unsound for a generator, and each was caught doing something
#: different with the same `unnest`:
#:
#:     pushdown_projections   replaced the unreferenced generator with `1 AS _`, so
#:                            `count()` over an exploded frame returned the *table's*
#:                            row count -- silently
#:     pushdown_predicates    substituted the generator's alias with the generator and
#:                            pushed it into a WHERE, which DuckDB rejects outright
#:     merge_subqueries       merged the defining scope away entirely, dropping the
#:                            generator with it
#:
#: Skipped together rather than singly: they are one assumption, and a plan that has
#: lost two of the three is not a shape worth reasoning about separately. What this
#: costs is subquery flattening on exploded queries -- scan pruning survives, because
#: `plan/pushdown.py` reads the qualified tree itself rather than relying on these.
#: See `plan/cardinality.py`.
UNSAFE_WITH_GENERATORS = frozenset(
    {"pushdown_projections", "pushdown_predicates", "merge_subqueries"}
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


def _close_always_true_branches(expression: exp.Expression) -> exp.Expression:
    """Turn `... WHEN TRUE THEN v ...` into `... ELSE v END`, dropping what follows.

    Semantics-preserving on its own -- no branch after an always-true one can ever be
    reached -- and it has to run **before** `simplify`, because sqlglot 30.17 folds a
    `CASE` whose always-true branch is not the first one down to that branch's value
    and throws the earlier branches away:

        CASE WHEN a = 1 THEN 'one' WHEN TRUE THEN 'rest' END   ->   'rest'

    DuckDB answers `'one'`, so the rule is a wrong answer rather than a missed
    optimisation, and it is silent. Normalising the shape away first means `simplify`
    never sees the case it mishandles. Only a `CASE` with no operand is touched: in
    `CASE x WHEN TRUE THEN ...` the branch is the comparison `x = TRUE`, which is not
    always true at all.
    """

    def rewrite(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Case) or node.args.get("this") is not None:
            return node
        branches = list(node.args.get("ifs") or [])
        for index, branch in enumerate(branches):
            condition = branch.this
            if not (isinstance(condition, exp.Boolean) and condition.this is True):
                continue
            value = branch.args.get("true")
            if value is None:  # pragma: no cover - defensive
                return node
            if index == 0:
                # Nothing is conditional any more, so there is no CASE left to write.
                return value.copy()
            closed = node.copy()
            closed.set("ifs", [item.copy() for item in branches[:index]])
            closed.set("default", value.copy())
            return closed
        return node

    return expression.transform(rewrite, copy=True)


def _named_selects(expression: exp.Expression) -> list[str]:
    """The output column names of a query node, or `[]` for anything else."""
    return list(expression.named_selects) if isinstance(expression, exp.Query) else []


def _naming_branch(expression: exp.Expression) -> exp.Select | None:
    """The SELECT whose projections name `expression`'s output columns.

    For a plain SELECT that is the node itself. For a set operation it is the
    **leftmost branch**, however deeply nested: `A UNION B UNION C` parses left-heavy,
    and SQL takes the output names from the first branch alone. Re-aliasing that one
    branch renames the whole set operation, which is the whole of carry-over note 10 --
    before this, any set operation needing a rename lost every optimization, predicate
    and projection pushdown included, not merely the rename.

    Only the first branch is touched: the others match positionally and their own
    names are never read, so rewriting them would be noise.
    """
    node = expression
    while isinstance(node, exp.SetOperation):
        node = node.this
    return node if isinstance(node, exp.Select) else None


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

    branch = _naming_branch(expression)
    if branch is None:
        return None

    projections = branch.expressions
    if len(projections) != len(expected):
        return None
    if any(isinstance(projection, exp.Star) for projection in projections):
        return None

    for index, name in enumerate(expected):
        projection = projections[index]
        if projection.alias_or_name == name:
            continue
        projections[index] = exp.alias_(projection.unalias(), name, quoted=True)

    # `branch` was mutated in place, so a set operation is renamed by this too. Verify
    # rather than assume: a branch shape that did not take the rename must still be
    # declined, or the caller would promise names the plan does not produce.
    if _named_selects(expression) != list(expected):
        return None
    return expression


def optimize_plan(
    plan: exp.Expression, schema: MappingSchema, output_names: Sequence[str]
) -> OptimizedPlan:
    """Optimize `plan`, falling back to it untouched if the result cannot be trusted.

    `output_names` is the analysed schema's column list -- the names the caller has
    already promised the user -- and is what the result is held to.
    """
    current = _close_always_true_branches(plan.copy())
    stages: list[str] = []
    note: str | None = None
    skipped = UNSAFE_WITH_GENERATORS if contains_generator(current) else frozenset()

    for name in RULES:
        if name in skipped:
            note = (
                "the plan contains a row-generating function, so the rules that move "
                f"projections between scopes were skipped ({', '.join(sorted(skipped))})"
            )
            continue
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

"""Rendering a PyIceberg predicate as something a person can read.

`str()` on a PyIceberg expression gives its constructor repr --
`GreaterThan(term=Reference(name='id'), literal=LongLiteral(10))` -- which is fine in
a debugger and useless in `explain()` output, where the question being asked is
"did my filter get pushed, and which one". This renders `id > 10` instead.

Purely cosmetic: nothing here feeds back into planning, so an expression this does
not recognise falls back to `repr` rather than raising.
"""

from __future__ import annotations

from pyiceberg.expressions import (
    AlwaysFalse,
    AlwaysTrue,
    And,
    BooleanExpression,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    IsNaN,
    IsNull,
    LessThan,
    LessThanOrEqual,
    LiteralPredicate,
    Not,
    NotEqualTo,
    NotIn,
    NotNaN,
    NotNull,
    NotStartsWith,
    Or,
    SetPredicate,
    StartsWith,
    UnaryPredicate,
    UnboundPredicate,
)

__all__ = ["describe_predicate"]

_BINARY_OPS: dict[type[BooleanExpression], str] = {
    EqualTo: "=",
    NotEqualTo: "!=",
    GreaterThan: ">",
    GreaterThanOrEqual: ">=",
    LessThan: "<",
    LessThanOrEqual: "<=",
    StartsWith: "STARTS WITH",
    NotStartsWith: "NOT STARTS WITH",
    In: "IN",
    NotIn: "NOT IN",
}

_UNARY_OPS: dict[type[BooleanExpression], str] = {
    IsNull: "IS NULL",
    NotNull: "IS NOT NULL",
    IsNaN: "IS NaN",
    NotNaN: "IS NOT NaN",
}


def _value(literal: object) -> str:
    """One literal, spelled the way it would appear in SQL."""
    inner = getattr(literal, "value", literal)
    return f"'{inner}'" if isinstance(inner, str) else str(inner)


def describe_predicate(expression: BooleanExpression) -> str:
    """`expression` as readable SQL-ish text."""
    if isinstance(expression, AlwaysTrue):
        return "true"
    if isinstance(expression, AlwaysFalse):
        return "false"
    if isinstance(expression, And):
        return f"({describe_predicate(expression.left)} AND {describe_predicate(expression.right)})"
    if isinstance(expression, Or):
        return f"({describe_predicate(expression.left)} OR {describe_predicate(expression.right)})"
    if isinstance(expression, Not):
        return f"NOT {describe_predicate(expression.child)}"

    if isinstance(expression, UnboundPredicate):
        term = getattr(expression.term, "name", str(expression.term))
        unary = _UNARY_OPS.get(type(expression))
        if unary is not None and isinstance(expression, UnaryPredicate):
            return f"{term} {unary}"
        operator = _BINARY_OPS.get(type(expression), type(expression).__name__)
        if isinstance(expression, SetPredicate):
            values = ", ".join(sorted(_value(literal) for literal in expression.literals))
            return f"{term} {operator} ({values})"
        if isinstance(expression, LiteralPredicate):
            return f"{term} {operator} {_value(expression.literal)}"

    return repr(expression)  # pragma: no cover - defensive

"""The reference engine's rules for naming an unaliased output column.

`df.select(F.col("amount") * 2)` produces a column called `(amount * 2)` in the reference engine,
and scripts index results by that name. DuckDB would call it something else, so the
name is decided here -- from the expression tree, in the reference engine's spelling -- and emitted
as an explicit alias.

Phase 1 covered references, literals, casts and the operators. Phase 3 adds the rule
for *function* calls, which is the one the function library needs: the reference engine spells a
generated name with the function in **lower case** -- `sum(amount)`, not
`SUM(amount)` -- while keeping SQL keywords like `CAST` upper. `normalize_functions`
is sqlglot's switch for exactly that distinction.
"""

from __future__ import annotations

from sqlglot import exp

from icetl.compat import SQL_DIALECT

__all__ = ["output_name"]


def output_name(expression: exp.Expression) -> str:
    """The name the reference engine would give `expression` in a `select`.

    >>> from sqlglot import exp, parse_one
    >>> output_name(parse_one("amount * 2"))
    '(amount * 2)'
    >>> output_name(exp.column("vendor"))
    'vendor'
    """
    if isinstance(expression, exp.Alias):
        return expression.alias
    if isinstance(expression, exp.Column):
        # The reference engine drops the qualifier: `t.amount` is named `amount`.
        return expression.name
    if isinstance(expression, exp.Literal):
        # A string literal names the column after its *value*, without quotes:
        # `F.lit("x")` is column `x`, not column `'x'`.
        return expression.this if expression.is_string else expression.sql(dialect=SQL_DIALECT)
    if isinstance(expression, exp.Star):
        return "*"
    if isinstance(expression, exp.Count) and isinstance(expression.this, exp.Star):
        # The reference engine rewrites `count(*)` to `count(1)` before naming the column.
        return "count(1)"

    return _parenthesised(expression).sql(
        dialect=SQL_DIALECT, copy=False, normalize_functions="lower"
    )


def _parenthesised(expression: exp.Expression) -> exp.Expression:
    """Wrap every binary operator in parentheses, the way the reference engine names them.

    The reference engine's rendering is recursive -- `(col("a") > 1) & (col("b") > 2)` is named
    `((a > 1) AND (b > 2))`, with each operand parenthesised in its own right, not
    just the outermost expression.

    Written as an explicit post-order walk rather than `Expression.transform`, which
    is top-down and stops descending as soon as the callback returns a new node --
    so it would wrap the outermost operator and never reach its operands.

    Any parentheses sqlglot already put in the tree are dropped and reinstated by
    this rule, so parsing `(a + 1) * 2` does not come back doubly wrapped.
    """
    return _rewrite(expression.copy())


def _rewrite(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Paren):
        return _rewrite(node.this)

    for key, child in list(node.args.items()):
        if isinstance(child, exp.Expression):
            node.set(key, _rewrite(child))
        elif isinstance(child, list):
            node.set(
                key,
                [_rewrite(item) if isinstance(item, exp.Expression) else item for item in child],
            )

    # `Dot` is struct-field access (`person.name`), which the reference engine leaves bare.
    if isinstance(node, exp.Binary) and not isinstance(node, exp.Dot):
        return exp.Paren(this=node)
    return node

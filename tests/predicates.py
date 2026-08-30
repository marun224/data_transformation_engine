"""PyIceberg predicate constructors, spelled so mypy accepts the positional form.

PyIceberg's predicates are pydantic models. The positional `__init__(term, literal)`
that Python actually calls lives on their base classes, but every leaf subclass
declares an extra field, so mypy follows `dataclass_transform` and synthesises a
keyword-only `__init__` from the fields instead -- one that is never used at runtime
and whose argument names are the hyphenated JSON aliases.

`src/icetl/plan/pushdown.py` sidesteps this the same way and for the same reason.
Naming them once here keeps the assertions in the tests readable, rather than
scattering `type: ignore` over every expected value.
"""

from __future__ import annotations

from typing import Any

from pyiceberg import expressions as _expressions

__all__ = [
    "EqualTo",
    "GreaterThan",
    "GreaterThanOrEqual",
    "In",
    "IsNull",
    "LessThan",
    "LessThanOrEqual",
    "NotEqualTo",
    "StartsWith",
]

EqualTo: Any = _expressions.EqualTo
NotEqualTo: Any = _expressions.NotEqualTo
GreaterThan: Any = _expressions.GreaterThan
GreaterThanOrEqual: Any = _expressions.GreaterThanOrEqual
LessThan: Any = _expressions.LessThan
LessThanOrEqual: Any = _expressions.LessThanOrEqual
In: Any = _expressions.In
IsNull: Any = _expressions.IsNull
StartsWith: Any = _expressions.StartsWith

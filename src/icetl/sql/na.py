"""`df.na` -- the null-handling surface: `drop`, `fill`, `replace`.

Every method here compiles to an ordinary projection or filter over the frame it came
from, so nothing in this module executes anything or knows about DuckDB. What it does
know about is the frame's **types**, and that is the whole reason it is a separate
module rather than three methods on `DataFrame`.

**The type rule is the thing to get right.** In the reference engine a fill value only
touches columns it could plausibly belong to: `fill(0)` fills numeric columns and leaves
a NULL string alone, `fill("")` does the reverse, and `fill(False)` only reaches
booleans. Silently filling a string column with `0` would be a wrong answer that no
test on a numeric column could catch, so the matching is explicit and tested per type.

`subset` narrows *which* columns are considered; the type rule still applies inside it.
A name in `subset` that the frame does not have is ignored rather than refused, which is
the reference's behaviour -- `subset` is a filter over the columns, not an assertion
about them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlglot import exp

from icetl.errors import EngineTypeError, EngineValueError
from icetl.plan.builder import as_expression
from icetl.sql.column import Column, to_literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from icetl.sql.dataframe import DataFrame

__all__ = ["DataFrameNaFunctions"]

#: `simpleString()` names of the types a numeric fill or replace may touch. `decimal`
#: is matched by prefix because it carries its precision and scale (`decimal(10,2)`).
_NUMERIC_TYPES = frozenset({"tinyint", "smallint", "int", "bigint", "float", "double"})
_NUMERIC_PREFIX = "decimal"

_STRING_TYPES = frozenset({"string"})
_BOOLEAN_TYPES = frozenset({"boolean"})


def _is_numeric(type_name: str) -> bool:
    return type_name in _NUMERIC_TYPES or type_name.startswith(_NUMERIC_PREFIX)


def _accepts(type_name: str, value: Any) -> bool:
    """True when a column of `type_name` is one the reference would apply `value` to.

    `bool` is checked before the numeric branch: it is a subclass of `int` in Python,
    and `fill(True)` must not reach a `bigint` column.
    """
    if isinstance(value, bool):
        return type_name in _BOOLEAN_TYPES
    if isinstance(value, (int, float)):
        return _is_numeric(type_name)
    if isinstance(value, str):
        return type_name in _STRING_TYPES
    return False


class DataFrameNaFunctions:
    """What `df.na` returns. Holds the frame; builds a new one per call."""

    def __init__(self, df: DataFrame) -> None:
        self._df = df

    def __repr__(self) -> str:
        return f"DataFrameNaFunctions[{', '.join(self._df.columns)}]"

    # -- drop ---------------------------------------------------------------

    def drop(
        self,
        how: str = "any",
        thresh: int | None = None,
        subset: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Drop rows with nulls in `subset` (default: every column).

        `how="any"` drops a row if any considered column is null; `how="all"` drops it
        only when they all are. **`thresh` overrides `how`** rather than combining with
        it -- `thresh=2` keeps rows with at least two non-null values, whatever `how`
        says. That precedence is the reference's, and it is the kind of thing that reads
        as a bug when you meet it in someone else's code, so it is spelled out here.
        """
        considered = self._subset(subset)
        if not considered:
            return self._df

        if thresh is not None:
            if not isinstance(thresh, int) or isinstance(thresh, bool):
                raise EngineTypeError(
                    f"drop() expects thresh as an int, got {type(thresh).__name__}."
                )
            if thresh < 0:
                raise EngineValueError(f"drop() expects thresh >= 0, got {thresh}.")
            non_null = [
                exp.Case(
                    ifs=[exp.If(this=_not_null(name), true=exp.Literal.number(1))],
                    default=exp.Literal.number(0),
                )
                for name in considered
            ]
            total: exp.Expression = non_null[0]
            for term in non_null[1:]:
                total = exp.Add(this=total, expression=term)
            enough = exp.GTE(this=exp.Paren(this=total), expression=exp.Literal.number(thresh))
            return self._df.filter(Column(enough))

        if how not in ("any", "all"):
            raise EngineValueError(f"drop() expects how as 'any' or 'all', got {how!r}.")
        # "any null drops the row"  -> every column must be non-null   -> AND
        # "all null drops the row"  -> some column must be non-null    -> OR
        combine = exp.and_ if how == "any" else exp.or_
        condition = as_expression(combine(*[_not_null(name) for name in considered]))
        return self._df.filter(Column(condition))

    # -- fill ---------------------------------------------------------------

    def fill(
        self,
        value: Any,
        subset: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Replace nulls with `value`, in the columns whose type it suits.

        `value` may be a scalar, or a `{column: value}` dict -- and the dict form
        **ignores `subset`**, because it already names its columns. A dict entry naming
        a column the frame does not have is an error rather than a no-op, since unlike
        `subset` it can only be a mistake.
        """
        if isinstance(value, dict):
            if subset is not None:
                raise EngineValueError(
                    "fill() takes either a {column: value} dict or a subset, not both."
                )
            return self._fill_mapping(value)

        if not isinstance(value, (int, float, str, bool)):
            raise EngineTypeError(
                f"fill() expects a number, string, bool or dict, got {type(value).__name__}."
            )
        considered = set(self._subset(subset))
        mapping = {
            field.name: value
            for field in self._df.schema.fields
            if field.name in considered and _accepts(field.dataType.simpleString(), value)
        }
        return self._project(mapping)

    def _fill_mapping(self, mapping: dict[Any, Any]) -> DataFrame:
        resolved: dict[str, Any] = {}
        for name, value in mapping.items():
            if not isinstance(name, str):
                raise EngineTypeError(
                    f"fill() expects column names as strings, got {type(name).__name__}."
                )
            actual = self._df._resolve_name(name)
            if actual is None:
                raise EngineValueError(
                    f"fill() was given the column {name!r}, which this frame does not "
                    f"have. Columns: {', '.join(self._df.columns)}."
                )
            resolved[actual] = value
        return self._project(resolved)

    def _project(self, mapping: dict[str, Any]) -> DataFrame:
        """Re-project the frame, coalescing each named column with its fill value."""
        if not mapping:
            return self._df
        projections: list[Any] = []
        for field in self._df.schema.fields:
            reference = exp.column(field.name, quoted=True)
            if field.name in mapping:
                filled = exp.Coalesce(this=reference, expressions=[to_literal(mapping[field.name])])
                projections.append(
                    Column(as_expression(exp.alias_(filled, field.name, quoted=True)))
                )
            else:
                projections.append(Column(reference))
        return self._df.select(*projections)

    # -- replace ------------------------------------------------------------

    def replace(
        self,
        to_replace: Any,
        value: Any = None,
        subset: str | Sequence[str] | None = None,
    ) -> DataFrame:
        """Swap values for other values, within the columns whose type suits them.

        Three shapes, all the reference's:

            replace(10, 0)                 one value for another
            replace([10, 20], [0, 1])      pairwise, and the lists must be the same length
            replace({10: 0, 20: 1})        the same thing spelled as a mapping

        A pair is only applied to a column whose type accepts **both** sides, so
        `replace("a", "b")` cannot touch a numeric column and `replace(1, 2)` cannot
        touch a string one. The match is null-safe, so a NULL replacement value works;
        a NULL *key* does not, and `fill` is the method for that.
        """
        pairs = _replacement_pairs(to_replace, value)
        if not pairs:
            return self._df

        considered = set(self._subset(subset))
        projections: list[Any] = []
        changed = False
        for field in self._df.schema.fields:
            reference = exp.column(field.name, quoted=True)
            type_name = field.dataType.simpleString()
            applicable = [
                (old, new)
                for old, new in pairs
                if field.name in considered
                and _accepts(type_name, old)
                and (new is None or _accepts(type_name, new))
            ]
            if not applicable:
                projections.append(Column(reference))
                continue
            changed = True
            case = exp.Case(
                ifs=[
                    exp.If(
                        this=exp.NullSafeEQ(this=reference.copy(), expression=to_literal(old)),
                        true=to_literal(new),
                    )
                    for old, new in applicable
                ],
                default=reference.copy(),
            )
            projections.append(Column(as_expression(exp.alias_(case, field.name, quoted=True))))
        return self._df.select(*projections) if changed else self._df

    # -- shared -------------------------------------------------------------

    def _subset(self, subset: str | Sequence[str] | None) -> list[str]:
        """The columns to consider, in schema order, spelled as the schema spells them.

        A name the frame does not carry is dropped rather than refused: `subset` is a
        filter over the columns, and the reference treats it as one.
        """
        if subset is None:
            return list(self._df.columns)
        names = [subset] if isinstance(subset, str) else list(subset)
        for name in names:
            if not isinstance(name, str):
                raise EngineTypeError(f"subset takes column names, got {type(name).__name__}.")
        wanted = {name.casefold() for name in names}
        return [column for column in self._df.columns if column.casefold() in wanted]


def _not_null(name: str) -> exp.Expression:
    return exp.Not(this=exp.Is(this=exp.column(name, quoted=True), expression=exp.Null()))


def _replacement_pairs(to_replace: Any, value: Any) -> list[tuple[Any, Any]]:
    """Normalise the three spellings of `replace` into (old, new) pairs."""
    if isinstance(to_replace, dict):
        if value is not None:
            raise EngineValueError("replace() takes either a {old: new} dict or a value, not both.")
        return list(to_replace.items())

    if isinstance(to_replace, (list, tuple)):
        olds = list(to_replace)
        if isinstance(value, (list, tuple)):
            news = list(value)
            if len(olds) != len(news):
                raise EngineValueError(
                    f"replace() got {len(olds)} value(s) to replace but {len(news)} "
                    f"replacement(s); the lists must be the same length."
                )
            return list(zip(olds, news, strict=True))
        return [(old, value) for old in olds]

    if isinstance(value, (list, tuple)):
        raise EngineValueError(
            "replace() got a single value to replace but a list of replacements."
        )
    return [(to_replace, value)]

"""`Column` -- a named expression, backed by a sqlglot node.

A Column is a thin wrapper: every operator builds a sqlglot expression and hands back
a new Column. Nothing is evaluated, nothing is bound to a DataFrame, and the same
node types come out as `spark.sql()` produces for the same operation (P1).

Nodes are copied on the way in and out. sqlglot expressions carry parent pointers, so
sharing one between two trees would corrupt both -- and `c = col("a"); c + 1; c * 2`
has to keep working.
"""

from __future__ import annotations

import datetime
import decimal
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from icetl.compat.naming import spark_output_name
from icetl.errors import PySparkTypeError, PySparkValueError, UnsupportedFeatureError
from icetl.plan.builder import as_expression
from icetl.types import DataType

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["Column", "to_expression", "to_literal"]

# The Python values that map onto a SQL literal without ceremony.
_LITERAL_TYPES = (
    bool,
    int,
    float,
    str,
    bytes,
    decimal.Decimal,
    datetime.date,
    datetime.datetime,
)


def to_literal(value: Any) -> exp.Expression:
    """Turn a Python value into a literal node, as `F.lit` does."""
    if value is None:
        return exp.null()
    if isinstance(value, Column):
        raise PySparkTypeError("A Column is already an expression; it cannot be made a literal.")
    if isinstance(value, bool):
        # Checked before int: bool is an int subclass and `exp.convert` would emit 1/0.
        return exp.true() if value else exp.false()
    if isinstance(value, _LITERAL_TYPES):
        return as_expression(exp.convert(value))
    if isinstance(value, (list, tuple)):
        return exp.Array(expressions=[to_literal(item) for item in value])
    raise PySparkTypeError(
        f"Cannot make a literal from {type(value).__name__}. Supported: "
        f"{', '.join(t.__name__ for t in _LITERAL_TYPES)}, None, and lists of those."
    )


def to_expression(value: Any) -> exp.Expression:
    """Coerce a Column, a column name, or a plain value into an expression node.

    Note the asymmetry Spark has: in an *operator* position a bare string is a
    literal (`col("a") == "b"` compares against the string `b`), while in a
    *projection* position it names a column. Callers pick; this function is the
    operator-position rule.
    """
    if isinstance(value, Column):
        return value._expression.copy()
    return to_literal(value)


class Column:
    """An expression over a DataFrame's columns."""

    __slots__ = ("_expression",)

    def __init__(self, expression: exp.Expression) -> None:
        if not isinstance(expression, exp.Expression):
            raise PySparkTypeError(
                f"Column expects a sqlglot expression, got {type(expression).__name__}. "
                f"Build columns with F.col(), F.lit(), or F.expr()."
            )
        self._expression = expression

    # -- internals ---------------------------------------------------------

    def _copy(self) -> exp.Expression:
        return self._expression.copy()

    def _binary(self, node: type[exp.Binary], other: Any, *, reverse: bool = False) -> Column:
        left, right = self._copy(), to_expression(other)
        if reverse:
            left, right = right, left
        return Column(as_expression(node(this=left, expression=right)))

    @property
    def _output_name(self) -> str:
        """The name Spark would give this column in a projection."""
        return spark_output_name(self._expression)

    # -- comparison --------------------------------------------------------

    def __eq__(self, other: Any) -> Column:  # type: ignore[override]
        return self._binary(exp.EQ, other)

    def __ne__(self, other: Any) -> Column:  # type: ignore[override]
        return self._binary(exp.NEQ, other)

    def __lt__(self, other: Any) -> Column:
        return self._binary(exp.LT, other)

    def __le__(self, other: Any) -> Column:
        return self._binary(exp.LTE, other)

    def __gt__(self, other: Any) -> Column:
        return self._binary(exp.GT, other)

    def __ge__(self, other: Any) -> Column:
        return self._binary(exp.GTE, other)

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: Any) -> Column:
        return self._binary(exp.Add, other)

    def __radd__(self, other: Any) -> Column:
        return self._binary(exp.Add, other, reverse=True)

    def __sub__(self, other: Any) -> Column:
        return self._binary(exp.Sub, other)

    def __rsub__(self, other: Any) -> Column:
        return self._binary(exp.Sub, other, reverse=True)

    def __mul__(self, other: Any) -> Column:
        return self._binary(exp.Mul, other)

    def __rmul__(self, other: Any) -> Column:
        return self._binary(exp.Mul, other, reverse=True)

    def __truediv__(self, other: Any) -> Column:
        return self._division(other)

    def __rtruediv__(self, other: Any) -> Column:
        return self._division(other, reverse=True)

    def __mod__(self, other: Any) -> Column:
        # DuckDB already returns NULL for `x % 0`, matching Spark. No guard needed.
        return self._binary(exp.Mod, other)

    def __rmod__(self, other: Any) -> Column:
        return self._binary(exp.Mod, other, reverse=True)

    def __neg__(self) -> Column:
        return Column(exp.Neg(this=self._copy()))

    def _division(self, other: Any, *, reverse: bool = False) -> Column:
        """Spark division: `x / 0` is NULL, not an error and not infinity.

        `safe=True` is how sqlglot spells that, and it is the same flag its Spark
        parser sets, so `df` and `spark.sql` produce an identical node. Generating
        for DuckDB turns it into `x / NULLIF(y, 0)`; without it DuckDB 1.5 would
        return `inf`, which is a wrong answer rather than a loud one.

        A divisor that is literally non-zero skips the guard -- it cannot divide by
        zero, and `NULLIF(2, 0)` in every generated query makes `explain()` harder to
        read for no gain.
        """
        left, right = self._copy(), to_expression(other)
        if reverse:
            left, right = right, left
        return Column(
            exp.Div(this=left, expression=right, typed=False, safe=not _is_nonzero_literal(right))
        )

    # -- logical -----------------------------------------------------------

    def __and__(self, other: Any) -> Column:
        return self._binary(exp.And, other)

    def __rand__(self, other: Any) -> Column:
        return self._binary(exp.And, other, reverse=True)

    def __or__(self, other: Any) -> Column:
        return self._binary(exp.Or, other)

    def __ror__(self, other: Any) -> Column:
        return self._binary(exp.Or, other, reverse=True)

    def __invert__(self) -> Column:
        return Column(exp.Not(this=exp.paren(self._copy())))

    def __bool__(self) -> bool:
        raise PySparkValueError(
            "Cannot convert a Column into a bool. Use `&` for and, `|` for or, and "
            "`~` for not, and parenthesise each operand -- Python binds `&` tighter "
            "than `==`, so `df.a == 1 & df.b == 2` is not what it looks like."
        )

    # -- null checks -------------------------------------------------------

    def isNull(self) -> Column:
        return Column(exp.Is(this=self._copy(), expression=exp.null()))

    def isNotNull(self) -> Column:
        return Column(exp.Not(this=exp.Is(this=self._copy(), expression=exp.null())))

    def isNaN(self) -> Column:
        """True for a floating-point NaN. Distinct from NULL in both Spark and DuckDB."""
        return Column(exp.Anonymous(this="isnan", expressions=[self._copy()]))

    def eqNullSafe(self, other: Any) -> Column:
        """Spark's `<=>`: equality where two NULLs compare equal.

        `IS NOT DISTINCT FROM` is the SQL spelling, and it is what sqlglot's Spark
        parser produces for `<=>`, so both surfaces build the same node (P1).
        """
        return Column(exp.NullSafeEQ(this=self._copy(), expression=to_expression(other)))

    # -- naming and typing -------------------------------------------------

    def alias(self, *alias: str, **kwargs: Any) -> Column:
        """Rename the column. Multiple names are for exploding functions (Phase 6)."""
        if kwargs:
            raise PySparkValueError(
                f"alias() got unexpected keyword(s): {', '.join(sorted(kwargs))}."
            )
        if len(alias) != 1:
            raise PySparkValueError(
                f"alias() takes exactly one name in Phase 1, got {len(alias)}. "
                f"Multiple aliases are for generator functions, which arrive in Phase 6."
            )
        return Column(as_expression(exp.alias_(self._copy(), alias[0], quoted=True)))

    # Spark exposes both spellings; `name` is the older one.
    name = alias

    def cast(self, dataType: DataType | str) -> Column:
        """Cast to a Spark type, given as a `DataType` or a DDL string like `"int"`.

        Uses SQL `CAST`, so an unparseable value raises where Spark's default
        (non-ANSI) mode would return NULL. Loud rather than wrong; `TRY_CAST` is
        Phase 3's conformance work. Recorded in divergence.md.
        """
        if isinstance(dataType, DataType):
            ddl = dataType.simpleString()
        elif isinstance(dataType, str):
            ddl = dataType
        else:
            raise PySparkTypeError(
                f"cast() expects a DataType or a DDL string, got {type(dataType).__name__}."
            )
        try:
            target = exp.DataType.build(ddl, dialect="spark")
        except Exception as exc:
            raise PySparkValueError(f"{ddl!r} is not a recognised Spark type.") from exc
        return Column(exp.Cast(this=self._copy(), to=target))

    astype = cast

    # -- ordering ----------------------------------------------------------

    def _ordered(self, *, desc: bool, nulls_first: bool | None = None) -> Column:
        """One `ORDER BY` term.

        When `nulls_first` is left unstated, `icetl.sql.conformance` fills in Spark's
        default at compile time -- nulls first ascending, last descending. Doing it
        there rather than here is what makes `df.orderBy(col("x"))` and
        `spark.sql("... ORDER BY x")` agree, since only one of them comes through
        this method.
        """
        node = exp.Ordered(this=self._copy(), desc=desc)
        if nulls_first is not None:
            node.set("nulls_first", nulls_first)
        return Column(node)

    def asc(self) -> Column:
        return self._ordered(desc=False)

    def desc(self) -> Column:
        return self._ordered(desc=True)

    def asc_nulls_first(self) -> Column:
        return self._ordered(desc=False, nulls_first=True)

    def asc_nulls_last(self) -> Column:
        return self._ordered(desc=False, nulls_first=False)

    def desc_nulls_first(self) -> Column:
        return self._ordered(desc=True, nulls_first=True)

    def desc_nulls_last(self) -> Column:
        return self._ordered(desc=True, nulls_first=False)

    # -- string matching ---------------------------------------------------

    def like(self, other: str) -> Column:
        """SQL `LIKE`: `%` matches any run, `_` any single character."""
        return Column(exp.Like(this=self._copy(), expression=to_literal(other)))

    def ilike(self, other: str) -> Column:
        """Case-insensitive `LIKE`."""
        return Column(exp.ILike(this=self._copy(), expression=to_literal(other)))

    def rlike(self, other: str) -> Column:
        """Java-regex match, as Spark's `rlike`.

        DuckDB's `regexp_matches` is a *search*, not an anchored match, which is the
        same semantics Spark gives `rlike` -- both find the pattern anywhere in the
        string unless it is anchored.
        """
        return Column(exp.RegexpLike(this=self._copy(), expression=to_literal(other)))

    def contains(self, other: Any) -> Column:
        return Column(
            exp.Anonymous(this="contains", expressions=[self._copy(), to_expression(other)])
        )

    def startswith(self, other: Any) -> Column:
        return Column(
            exp.Anonymous(this="starts_with", expressions=[self._copy(), to_expression(other)])
        )

    def endswith(self, other: Any) -> Column:
        return Column(
            exp.Anonymous(this="ends_with", expressions=[self._copy(), to_expression(other)])
        )

    def substr(self, startPos: int | Column, length: int | Column) -> Column:
        """Spark's `substr`, which is **1-indexed**, as SQL `SUBSTRING` is.

        Both arguments may be Columns, which is why they go through `to_expression`
        rather than being inlined as literals.
        """
        if isinstance(startPos, Column) != isinstance(length, Column):
            raise PySparkTypeError(
                "substr() takes either two ints or two Columns, not one of each."
            )
        return Column(
            exp.Substring(
                this=self._copy(),
                start=to_expression(startPos),
                length=to_expression(length),
            )
        )

    # -- element access ----------------------------------------------------

    def getItem(self, key: Any) -> Column:
        """`col[i]` for an array (0-based, as Spark) or `col[k]` for a map."""
        return self.__getitem__(key)

    def getField(self, name: str) -> Column:
        """`col.field` for a struct."""
        if not isinstance(name, str):
            raise PySparkTypeError(f"getField() expects a name, got {type(name).__name__}.")
        return Column(exp.Dot(this=self._copy(), expression=exp.to_identifier(name, quoted=True)))

    def __getitem__(self, key: Any) -> Column:
        """`col[0]`, `col["k"]`, and `col["field"]` for a struct.

        Spark's arrays are 0-based while DuckDB's `list[i]` is 1-based, so an integer
        index is shifted. A string key is left alone: it addresses a map entry or a
        struct field, and neither is positional.
        """
        if isinstance(key, Column):
            return Column(exp.Bracket(this=self._copy(), expressions=[key._copy()]))
        if isinstance(key, bool):
            raise PySparkTypeError("A bool is not a valid index or key.")
        if isinstance(key, int):
            return Column(
                exp.Bracket(this=self._copy(), expressions=[to_literal(key + 1)], offset=1)
            )
        if isinstance(key, str):
            return Column(exp.Bracket(this=self._copy(), expressions=[to_literal(key)]))
        raise PySparkTypeError(f"Cannot index a Column with {type(key).__name__}.")

    def __getattr__(self, name: str) -> Column:
        """`col.field`, the attribute spelling of `getField`."""
        if name.startswith("_"):
            raise AttributeError(name)
        return self.getField(name)

    # -- bitwise -----------------------------------------------------------

    def bitwiseAND(self, other: Any) -> Column:
        return self._binary(exp.BitwiseAnd, other)

    def bitwiseOR(self, other: Any) -> Column:
        return self._binary(exp.BitwiseOr, other)

    def bitwiseXOR(self, other: Any) -> Column:
        return self._binary(exp.BitwiseXor, other)

    # -- conditionals ------------------------------------------------------

    def when(self, condition: Column, value: Any) -> Column:
        """Add a branch to a `CASE` started by `F.when`."""
        if not isinstance(self._expression, exp.Case):
            raise PySparkTypeError("when() can only be chained onto a Column built by F.when().")
        if self._expression.args.get("default") is not None:
            raise PySparkTypeError("when() cannot follow otherwise().")
        if not isinstance(condition, Column):
            raise PySparkTypeError(
                f"when() expects a Column condition, got {type(condition).__name__}."
            )
        case = self._copy()
        case.args.setdefault("ifs", []).append(
            exp.If(this=condition._copy(), true=to_expression(value))
        )
        return Column(case)

    def otherwise(self, value: Any) -> Column:
        """The `ELSE` of a `CASE`. Without it an unmatched row is NULL, as in Spark."""
        if not isinstance(self._expression, exp.Case):
            raise PySparkTypeError(
                "otherwise() can only be chained onto a Column built by F.when()."
            )
        if self._expression.args.get("default") is not None:
            raise PySparkTypeError("otherwise() can only be given once.")
        case = self._copy()
        case.set("default", to_expression(value))
        return Column(case)

    # -- deferred ----------------------------------------------------------

    def over(self, window: Any) -> Column:
        raise UnsupportedFeatureError("Column.over()", phase="Phase 5")

    # -- membership and ranges --------------------------------------------

    def isin(self, *values: Any) -> Column:
        """True when the column matches any of `values`.

        Accepts either varargs or a single iterable, as Spark does.
        """
        items: Iterable[Any]
        if len(values) == 1 and isinstance(values[0], (list, tuple, set, frozenset)):
            items = values[0]
        else:
            items = values
        return Column(exp.In(this=self._copy(), expressions=[to_expression(v) for v in items]))

    def between(self, lowerBound: Any, upperBound: Any) -> Column:
        return Column(
            exp.Between(
                this=self._copy(),
                low=to_expression(lowerBound),
                high=to_expression(upperBound),
            )
        )

    # -- rendering ---------------------------------------------------------

    def __repr__(self) -> str:
        # Quoting is stripped first: internally we quote every identifier so that a
        # column named `order` still works, but PySpark reprs `df["order"]` as
        # `Column<'order'>`, not with backticks around it.
        rendered = self._expression.copy()
        for identifier in rendered.find_all(exp.Identifier):
            identifier.set("quoted", False)
        return f"Column<'{rendered.sql(dialect='spark')}'>"

    def __hash__(self) -> int:
        # `__eq__` builds a predicate rather than comparing, so identity is the only
        # hash that stays consistent. PySpark's Column behaves the same way.
        return id(self)

    def __iter__(self) -> Any:
        raise PySparkTypeError("A Column is not iterable.")


def _is_nonzero_literal(node: exp.Expression) -> bool:
    """True when `node` is a numeric literal that is definitely not zero."""
    if not isinstance(node, exp.Literal) or node.is_string:
        return False
    try:
        return float(node.this) != 0.0
    except (TypeError, ValueError):
        return False


def _column_from_name(name: str) -> Column:
    """`col("a")`, `col("t.a")`, `col("*")` -- parsed so qualifiers survive."""
    if name == "*":
        return Column(exp.Star())
    if name.endswith(".*"):
        qualifier = name[:-2]
        return Column(exp.Column(this=exp.Star(), table=exp.to_identifier(qualifier)))
    parsed = sqlglot.maybe_parse(name, into=exp.Column, dialect="spark")
    return Column(parsed)

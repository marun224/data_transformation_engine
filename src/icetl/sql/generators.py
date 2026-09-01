"""Generator columns -- the ones that produce more than they are handed.

`explode` produces more **rows**; `posexplode`, `inline` and `json_tuple` also produce
more **columns**. Neither fits the one-Column-one-output-column shape the rest of the
surface is built on, so they are a `Column` subclass that `DataFrame.select` knows to
expand before it builds a projection list.

**Rows come for free.** DuckDB's `unnest` in a select list already expands one row into
one per element, so `explode` needs no lateral join and no plan surgery -- it is an
ordinary expression that happens to change the cardinality.

Columns take a little more. The useful discovery is that repeating an identical
`unnest(x)` in one select list unnests **once**, not twice -- DuckDB correlates the
copies -- so `posexplode` can emit `generate_subscripts(x, 1) - 1` beside `unnest(x)`
and get matching pairs rather than a cross product. Every multi-column form here is
built on that.

**The shape depends on the type**, and the type is not known until the frame is: the
reference's `explode` yields one column for a list and two for a map, and `inline` yields
one per field of the exploded struct. So expansion happens in `select`, where the
DataFrame is in hand and its schema can be resolved -- which is also why these classes
carry the *source* expression rather than a finished one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlglot import exp

from icetl.errors import AnalysisException, EngineValueError
from icetl.plan.builder import as_expression
from icetl.sql.column import Column
from icetl.types import ArrayType, MapType, StructType

if TYPE_CHECKING:
    from icetl.sql.dataframe import DataFrame

__all__ = [
    "GeneratorColumn",
    "SchemaAwareColumn",
    "StructDropColumn",
    "StructPutColumn",
]

#: The reference's default output names, per generator shape.
_LIST_NAME = "col"
_POSITION_NAME = "pos"
_MAP_NAMES = ("key", "value")


class SchemaAwareColumn(Column):
    """A column whose projection cannot be written until the frame's types are known.

    `DataFrame.select` expands one of these instead of projecting it. Two kinds exist:
    generators, which change how many rows and columns come out, and `dropFields`, which
    changes neither but has to know a struct's field list to rebuild it.
    """

    #: True when expanding this changes the row count, which `select` allows only once.
    _is_generator = False

    def _expand(self, df: DataFrame) -> list[exp.Expression]:  # pragma: no cover - abstract
        raise NotImplementedError


class GeneratorColumn(SchemaAwareColumn):
    """A column that expands to one or more projections, and sometimes to more rows."""

    _is_generator = True

    def __init__(
        self,
        kind: str,
        source: exp.Expression,
        *,
        outer: bool = False,
        fields: list[str] | None = None,
        names: list[str] | None = None,
    ) -> None:
        # The base `Column` still needs a usable expression: an un-expanded generator
        # used somewhere that cannot expand it should fail as SQL, not as an attribute
        # error, and `unnest(x)` on its own is exactly the single-column case.
        super().__init__(_unnest(source, outer=outer))
        self._kind = kind
        self._source = source
        self._outer = outer
        self._fields = fields
        self._names = names

    # -- constructors --------------------------------------------------------

    @classmethod
    def explode(cls, source: exp.Expression, *, outer: bool, position: bool) -> GeneratorColumn:
        return cls("posexplode" if position else "explode", source, outer=outer)

    @classmethod
    def inline(cls, source: exp.Expression, *, outer: bool) -> GeneratorColumn:
        return cls("inline", source, outer=outer)

    @classmethod
    def json_tuple(cls, source: exp.Expression, fields: list[str]) -> GeneratorColumn:
        return cls("json_tuple", source, fields=fields)

    # -- naming --------------------------------------------------------------

    def alias(self, *alias: str, **kwargs: Any) -> Column:
        """Name the generated columns. A multi-column generator takes one name each.

        This is what `Column.alias`'s several-names form was reserved for: `explode` on a
        map produces two columns, so `explode(m).alias("k", "v")` is the only way to name
        them.
        """
        if kwargs:
            raise EngineValueError(f"alias() got unexpected keyword(s): {', '.join(kwargs)}.")
        if not alias:
            raise EngineValueError("alias() needs at least one name.")
        return GeneratorColumn(
            self._kind,
            self._source,
            outer=self._outer,
            fields=self._fields,
            names=list(alias),
        )

    def __repr__(self) -> str:
        return f"GeneratorColumn[{self._kind}]"

    # -- expansion -----------------------------------------------------------

    def _expand(self, df: DataFrame) -> list[exp.Expression]:
        """The projections this generator stands for, in output order."""
        if self._kind == "json_tuple":
            return self._expand_json_tuple()

        data_type = df._type_of(self._source)
        if self._kind == "inline":
            return self._expand_inline(data_type)
        if isinstance(data_type, MapType):
            return self._expand_map()
        if isinstance(data_type, ArrayType):
            return self._expand_list()
        raise AnalysisException(
            f"{self._kind}() needs an array or map column, got {data_type.simpleString()}."
        )

    def _expand_list(self) -> list[exp.Expression]:
        names = self._resolved_names([_LIST_NAME], position_first=True)
        projections = [_aliased(_unnest(self._source, outer=self._outer), names[-1])]
        if self._kind == "posexplode":
            projections.insert(0, _aliased(self._position(), names[0]))
        return projections

    def _expand_map(self) -> list[exp.Expression]:
        """A map explodes to two columns, which is why `explode` is not one-to-one."""
        entries = exp.Anonymous(this="map_entries", expressions=[self._source.copy()])
        unnested = _unnest(entries, outer=self._outer)
        names = self._resolved_names(list(_MAP_NAMES), position_first=True)
        offset = 1 if self._kind == "posexplode" else 0
        projections = [
            _aliased(_field(unnested, part), names[offset + index])
            for index, part in enumerate(_MAP_NAMES)
        ]
        if self._kind == "posexplode":
            # Positions come from the *entry list*, not the map: a map has no subscripts
            # of its own, and `generate_subscripts` refuses one.
            projections.insert(0, _aliased(self._position(entries), names[0]))
        return projections

    def _expand_inline(self, data_type: Any) -> list[exp.Expression]:
        """One column per field of the exploded struct."""
        if not isinstance(data_type, ArrayType) or not isinstance(
            data_type.elementType, StructType
        ):
            raise AnalysisException(
                f"inline() needs an array of structs, got {data_type.simpleString()}."
            )
        fields = [field.name for field in data_type.elementType.fields]
        names = self._resolved_names(fields, position_first=False)
        unnested = _unnest(self._source, outer=self._outer)
        return [
            _aliased(_field(unnested, field), names[index]) for index, field in enumerate(fields)
        ]

    def _expand_json_tuple(self) -> list[exp.Expression]:
        assert self._fields is not None
        # The reference names them `c0`, `c1`, ... rather than after the fields, because
        # a path is not always a usable column name.
        defaults = [f"c{index}" for index in range(len(self._fields))]
        names = self._resolved_names(defaults, position_first=False)
        return [
            _aliased(
                exp.Anonymous(
                    this="json_extract_string",
                    expressions=[
                        exp.Cast(this=self._source.copy(), to=exp.DataType.build("JSON")),
                        exp.Literal.string(field),
                    ],
                ),
                names[index],
            )
            for index, field in enumerate(self._fields)
        ]

    def _position(self, over: exp.Expression | None = None) -> exp.Expression:
        """The 0-based index of the exploded element.

        `generate_subscripts` counts from 1 and the reference counts from 0, so one is
        subtracted. It is paired with the `unnest` beside it by DuckDB, not by us.
        """
        subscripts = exp.Anonymous(
            this="generate_subscripts",
            expressions=[(over or self._source).copy(), exp.Literal.number("1")],
        )
        return exp.Sub(this=subscripts, expression=exp.Literal.number("1"))

    def _resolved_names(self, defaults: list[str], position_first: bool) -> list[str]:
        expected = list(defaults)
        if self._kind == "posexplode" and position_first:
            expected = [_POSITION_NAME, *expected]
        if self._names is None:
            return expected
        if len(self._names) != len(expected):
            raise EngineValueError(
                f"{self._kind}() produces {len(expected)} column(s) "
                f"({', '.join(expected)}), so alias() needs {len(expected)} name(s), "
                f"got {len(self._names)}."
            )
        return self._names


def _unnest(source: exp.Expression, *, outer: bool) -> exp.Expression:
    """`unnest(source)`, or a form that still yields one NULL row when it is empty.

    The outer variants exist because the plain one drops the row entirely: a row whose
    list is empty simply disappears, which is right for `explode` and wrong for
    `explode_outer`. Substituting a one-element list of NULL puts the row back.
    """
    argument = source.copy()
    if outer:
        empty = exp.EQ(
            this=exp.Anonymous(this="len", expressions=[source.copy()]),
            expression=exp.Literal.number("0"),
        )
        missing = exp.Is(this=source.copy(), expression=exp.Null())
        argument = exp.Case(
            ifs=[
                exp.If(
                    this=exp.Or(this=missing, expression=empty),
                    true=exp.Array(expressions=[exp.Null()]),
                )
            ],
            default=source.copy(),
        )
    return exp.Anonymous(this="unnest", expressions=[argument])


def _field(source: exp.Expression, name: str) -> exp.Expression:
    """`(source).name` -- parenthesised, because `unnest(x).y` will not parse."""
    return exp.Dot(this=exp.Paren(this=source.copy()), expression=exp.to_identifier(name))


def _aliased(expression: exp.Expression, name: str) -> exp.Expression:
    return as_expression(exp.alias_(expression, name, quoted=True))


class StructDropColumn(SchemaAwareColumn):
    """`Column.dropFields` -- a struct rebuilt without the named fields.

    DuckDB has `struct_insert` but no field removal, so the only way to drop one is to
    build a new struct from the fields that stay. That needs the field list, which comes
    from the frame's schema -- hence the deferred expansion.

    A name that is not a field of the struct is **ignored**, as it is in the reference:
    dropping what is not there is not an error.
    """

    def __init__(self, source: exp.Expression, fields: list[str], name: str | None = None) -> None:
        # Deliberately un-runnable if it is never expanded: DuckDB has no such function,
        # so a `dropFields` used where a schema cannot be resolved fails loudly rather
        # than quietly returning the struct unchanged.
        super().__init__(exp.Anonymous(this="icetl_drop_fields", expressions=[source.copy()]))
        self._source = source
        self._fields = fields
        self._name = name

    def alias(self, *alias: str, **kwargs: Any) -> Column:
        if kwargs:
            raise EngineValueError(f"alias() got unexpected keyword(s): {', '.join(kwargs)}.")
        if len(alias) != 1:
            raise EngineValueError(f"alias() takes exactly one name, got {len(alias)}.")
        return StructDropColumn(self._source, self._fields, alias[0])

    def __repr__(self) -> str:
        return f"StructDropColumn[drop {', '.join(self._fields)}]"

    def _expand(self, df: DataFrame) -> list[exp.Expression]:
        data_type = df._type_of(self._source)
        if not isinstance(data_type, StructType):
            raise AnalysisException(
                f"dropFields() needs a struct column, got {data_type.simpleString()}."
            )
        dropped = {name.casefold() for name in self._fields}
        keep = [f.name for f in data_type.fields if f.name.casefold() not in dropped]
        if not keep:
            raise AnalysisException(
                "dropFields() would leave an empty struct, which has no SQL spelling."
            )
        rebuilt = exp.Struct(
            expressions=[
                exp.PropertyEQ(
                    this=exp.to_identifier(field),
                    expression=_field(self._source, field),
                )
                for field in keep
            ]
        )
        return [_aliased(rebuilt, self._name or self._output_name)]


class StructPutColumn(SchemaAwareColumn):
    """`Column.withField` -- a struct with one field added or replaced.

    DuckDB's `struct_insert` looks like the obvious tool and is not: it **refuses** a
    field name the struct already has (`Duplicate struct entry name`) rather than
    overwriting it, where the reference replaces. So adding and replacing are one
    operation here only because the struct is rebuilt from its field list, which is why
    this needs the frame's schema like `dropFields` does.
    """

    def __init__(
        self,
        source: exp.Expression,
        field: str,
        value: exp.Expression,
        name: str | None = None,
    ) -> None:
        super().__init__(
            exp.Anonymous(this="icetl_with_field", expressions=[source.copy(), value.copy()])
        )
        self._source = source
        self._field = field
        self._value = value
        self._name = name

    def alias(self, *alias: str, **kwargs: Any) -> Column:
        if kwargs:
            raise EngineValueError(f"alias() got unexpected keyword(s): {', '.join(kwargs)}.")
        if len(alias) != 1:
            raise EngineValueError(f"alias() takes exactly one name, got {len(alias)}.")
        return StructPutColumn(self._source, self._field, self._value, alias[0])

    def __repr__(self) -> str:
        return f"StructPutColumn[with {self._field}]"

    def _expand(self, df: DataFrame) -> list[exp.Expression]:
        data_type = df._type_of(self._source)
        if not isinstance(data_type, StructType):
            raise AnalysisException(
                f"withField() needs a struct column, got {data_type.simpleString()}."
            )
        target = self._field.casefold()
        entries: list[tuple[str, exp.Expression]] = []
        replaced = False
        for field in data_type.fields:
            if field.name.casefold() == target:
                # The struct's own spelling of the name is kept, as the reference keeps
                # it: `withField("NAME", ...)` replaces `name` without renaming it.
                entries.append((field.name, self._value.copy()))
                replaced = True
            else:
                entries.append((field.name, _field(self._source, field.name)))
        if not replaced:
            entries.append((self._field, self._value.copy()))

        rebuilt = exp.Struct(
            expressions=[
                exp.PropertyEQ(this=exp.to_identifier(name), expression=node)
                for name, node in entries
            ]
        )
        return [_aliased(rebuilt, self._name or self._output_name)]

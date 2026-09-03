"""Python UDFs: a callable registered with DuckDB, reachable from both surfaces.

    slen = session.udf.register("slen", len, "int")
    df.select(slen("name"))                     # the DataFrame surface
    session.sql("SELECT slen(name) FROM t")     # the SQL surface, same function

One registration serves both, because a UDF is an ordinary function call in the plan
and neither surface can tell which one made it (P1).

**Registration goes on two connections, not one.** The engine executes on its own
DuckDB connection and `plan/analysis.py` binds schemas on a *separate* one. A function
registered only on the first analyses as *Catalog Error: Scalar Function with name
slen does not exist* -- before a single row is read, and from a call that plainly
exists. Both, therefore, and `unregister` removes it from both.

**Registered on the root connection, not on a cursor.** Unlike a registered Arrow
table or a `CREATE TEMP TABLE`, which are per-cursor, a DuckDB function created on the
root connection is visible to every cursor derived from it. That is what lets a UDF
survive the thread-local cursor the engine hands out.

**NULL handling is registered as SPECIAL and guarded here**, which took measuring to
arrive at. DuckDB offers two modes and each is unusable on its own (FINDINGS.md 2.9):

| | a UDF that returns NULL | `udf(lambda x: x * 2)` over a scan |
|---|---|---|
| `DEFAULT` | **raises** -- "the UDF is not expected to return NULL values" | fine |
| `SPECIAL` | fine | **raises** -- see below |

Returning NULL is not negotiable: any UDF mapping onto a nullable column does it. So
SPECIAL it is -- and SPECIAL brings DuckDB's own quirk, which is that a UDF over
`read_parquet` is handed **one extra row of NULLs that the data never contained**.
Not an edge case here: `read_parquet` is every Iceberg scan.

So the call is wrapped. With `callOnNull` off -- the default -- a call whose arguments
are *all* None returns None without reaching the function, which absorbs DuckDB's
invented row and gives NULL in, NULL out. `callOnNull=True` removes the wrapper for
the case that wants the reference's behaviour: a UDF whose job is turning NULL into
something. Such a function handles `None` by construction, which is exactly what makes
the invented row harmless for it.

**The vectorised path is different, and closer to the reference.** A vector cannot
skip an element, so a `pandas_udf` always sees NULLs as missing values in the Series
-- as the reference's does. It sees DuckDB's extra row too; per-element pandas
operations propagate NaN through it and DuckDB discards the result. Both are recorded
in `compat/divergence.md`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import sqlglot
from sqlglot import exp

from icetl.compat import SQL_DIALECT
from icetl.errors import AnalysisException, EngineTypeError, EngineValueError
from icetl.parse_types import parse_datatype_string
from icetl.sql.column import Column
from icetl.types import DataType, StringType

if TYPE_CHECKING:
    from collections.abc import Callable

    import duckdb

    from icetl.sql.session import Session

__all__ = ["UDFRegistration", "UserDefinedFunction", "duckdb_type_of"]

#: Names for the anonymous functions `F.udf` produces. A caller who never names a UDF
#: still needs one in the SQL, and this cannot collide with anything a user would write.
_ANONYMOUS = itertools.count()

#: The reference's default return type for a UDF whose author did not give one.
#: A module-level singleton rather than a default argument, which ruff rejects.
_DEFAULT_RETURN_TYPE = StringType()


def duckdb_type_of(data_type: DataType | str) -> str:
    """The DuckDB type text for a return type, given as a `DataType` or DDL string.

    Routed through sqlglot rather than a hand-written table, so the complex types come
    for free and stay consistent with how every other type in the codebase is
    translated: `array<string>` -> `TEXT[]`, `map<string,bigint>` ->
    `MAP(TEXT, BIGINT)`, `struct<a:bigint>` -> `STRUCT(a BIGINT)`.
    """
    if isinstance(data_type, str):
        data_type = parse_datatype_string(data_type)
    if not isinstance(data_type, DataType):
        raise EngineTypeError(
            f"A UDF's returnType must be a DataType or a DDL string, got "
            f"{type(data_type).__name__}."
        )
    parsed = sqlglot.parse_one(data_type.simpleString(), into=exp.DataType, read=SQL_DIALECT)
    return parsed.sql(dialect="duckdb")


class UserDefinedFunction:
    """A registered Python function, callable to build a `Column`.

    Calling it does not run anything: it builds the same function-call node the SQL
    parser builds for the same name, so the two surfaces converge on one plan.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., Any],
        returnType: DataType | str,
        *,
        vectorised: bool = False,
        callOnNull: bool = False,
    ) -> None:
        self.name = name
        self.func = func
        self.returnType = (
            parse_datatype_string(returnType) if isinstance(returnType, str) else returnType
        )
        self.vectorised = vectorised
        self.callOnNull = callOnNull

    def __call__(self, *cols: Any) -> Column:
        from icetl.sql.functions import _col

        return Column(exp.Anonymous(this=self.name, expressions=[_col(c) for c in cols]))

    def __repr__(self) -> str:
        kind = "pandas_udf" if self.vectorised else "udf"
        return f"<{kind} {self.name!r} -> {self.returnType.simpleString()}>"


def _null_guard(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return NULL, without calling `func`, when every argument is NULL.

    Two things at once, which is why it is worth one wrapper rather than two. It gives
    NULL in / NULL out -- DuckDB's `DEFAULT` semantics, which we cannot ask DuckDB for
    because `DEFAULT` also forbids *returning* NULL. And it absorbs the row of NULLs
    DuckDB invents for a SPECIAL UDF over `read_parquet`, which would otherwise reach
    a function with no reason to expect one.

    All arguments rather than any: that is the shape of the invented row, and a
    partially-NULL call is a real row the function should see.
    """

    def guarded(*args: Any) -> Any:
        if args and all(arg is None for arg in args):
            return None
        return func(*args)

    return guarded


def _arrow_adapter(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a pandas-style vectorised function for DuckDB's Arrow UDF protocol.

    DuckDB hands each argument in as a `pyarrow.ChunkedArray` and wants one back; the
    reference hands a `pandas.Series` and wants one back. Converting at the boundary
    is what makes the *reference's* signature the one people write, which is the whole
    point of `pandas_udf` -- a function taking Arrow directly would just be a UDF with
    a different argument type.
    """

    def adapted(*columns: Any) -> Any:
        import pyarrow as pa

        series = [column.to_pandas() for column in columns]
        result = func(*series)
        if hasattr(result, "to_numpy") and not isinstance(result, pa.Array):
            return pa.Array.from_pandas(result)
        return result

    return adapted


class UDFRegistration:
    """What `session.udf` returns: register a Python function under a SQL name."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._registered: dict[str, UserDefinedFunction] = {}

    def __repr__(self) -> str:
        return f"UDFRegistration[{', '.join(sorted(self._registered)) or 'none'}]"

    @property
    def registered(self) -> dict[str, UserDefinedFunction]:
        """The UDFs this session knows, by name. A copy -- mutating it changes nothing."""
        return dict(self._registered)

    def register(
        self,
        name: str,
        f: Callable[..., Any],
        returnType: DataType | str | None = None,
        *,
        vectorised: bool = False,
        callOnNull: bool = False,
    ) -> UserDefinedFunction:
        """Register `f` under `name`, returning it as a callable that builds a Column.

        `returnType` defaults to `StringType()`, as the reference's does. It is not
        inferred from the function: DuckDB needs the type before the first row, and a
        guess that was wrong would be a silently mistyped column.

        `callOnNull` opts into the reference's null handling -- the function is called
        with `None` rather than answered with NULL on its behalf. Off by default, for
        a reason this module's docstring measures rather than asserts.

        Re-registering a name **replaces** it, which the reference allows and DuckDB
        does not -- it raises *A function by the name of 'x' is already created*. So
        the old one is removed first.
        """
        if not isinstance(name, str) or not name:
            raise EngineTypeError(f"register() expects a function name, got {name!r}.")
        if not callable(f):
            raise EngineTypeError(f"register() expects a callable, got {type(f).__name__}.")

        udf = UserDefinedFunction(
            name,
            f,
            _DEFAULT_RETURN_TYPE if returnType is None else returnType,
            vectorised=vectorised,
            callOnNull=callOnNull,
        )
        self._install(udf)
        return udf

    def registerVectorised(
        self,
        name: str,
        f: Callable[..., Any],
        returnType: DataType | str | None = None,
        *,
        callOnNull: bool = False,
    ) -> UserDefinedFunction:
        """`register`, for a function taking and returning `pandas.Series`.

        A vectorised function always receives whole vectors, NULLs included, as
        missing values in the Series -- `callOnNull` decides only whether a vector
        that is *entirely* NULL is passed at all.
        """
        return self.register(name, f, returnType, vectorised=True, callOnNull=callOnNull)

    def unregister(self, name: str) -> bool:
        """Drop a UDF from both connections. True if it was there."""
        if name not in self._registered:
            return False
        for connection in self._connections():
            with _ignoring_missing():
                connection.remove_function(name)
        del self._registered[name]
        self._session._analyzer.invalidate()
        return True

    # -- internals ---------------------------------------------------------

    def _connections(self) -> list[duckdb.DuckDBPyConnection]:
        """The engine's root connection and the analyzer's, in that order.

        Both, because a function missing from the analyzer's connection fails at
        *analysis* -- so the query never runs, and the error names a catalog rather
        than a UDF.
        """
        return [self._session._engine.connection, self._session._analyzer.connection]

    def _install(self, udf: UserDefinedFunction) -> None:
        return_type = duckdb_type_of(udf.returnType)
        if udf.vectorised:
            function = _arrow_adapter(udf.func)
        elif udf.callOnNull:
            function = udf.func
        else:
            function = _null_guard(udf.func)

        for connection in self._connections():
            with _ignoring_missing():
                connection.remove_function(udf.name)
            try:
                connection.create_function(
                    udf.name,
                    function,
                    None,  # argument types inferred per call site
                    _duckdb_dtype(return_type),
                    **_udf_flags(vectorised=udf.vectorised),
                )
            except Exception as exc:  # pragma: no cover - defensive
                raise AnalysisException(
                    f"Could not register the UDF {udf.name!r} returning "
                    f"{udf.returnType.simpleString()}: {exc}"
                ) from exc

        self._registered[udf.name] = udf
        # A name re-registered with a different return type produces the *same* SQL
        # with a different schema, which a cache keyed on SQL would answer from the
        # previous registration. Bumping the epoch is what stops that.
        self._session._analyzer.invalidate()


def _udf_flags(*, vectorised: bool) -> dict[str, Any]:
    """`create_function`'s `type` and `null_handling`, spelled for this DuckDB.

    duckdb 1.5 moved both enums out of the documented `duckdb.functional` -- which no
    longer exists, along with `duckdb.typing` -- and into a private `_duckdb._func`.
    Every tutorial still imports the old paths. The plain strings work at runtime and
    are what this falls back to; the enums are preferred when they can be found,
    because a string that stopped being accepted would fail at registration rather
    than quietly.

    Always SPECIAL, whatever `callOnNull` says: `DEFAULT` forbids a UDF from returning
    NULL at all, which rules it out for every UDF rather than for some. What
    `callOnNull` selects is whether `_null_guard` wraps the function, not what DuckDB
    is told. See this module's docstring.
    """
    try:
        from _duckdb._func import FunctionNullHandling, PythonUDFType

        kind: Any = PythonUDFType.ARROW if vectorised else PythonUDFType.NATIVE
        nulls: Any = FunctionNullHandling.SPECIAL
    except ImportError:  # pragma: no cover - depends on the installed duckdb
        kind = "arrow" if vectorised else "native"
        nulls = "special"
    return {"type": kind, "null_handling": nulls}


def _duckdb_dtype(type_text: str) -> Any:
    import duckdb

    try:
        return duckdb.dtype(type_text)
    except Exception as exc:
        raise EngineValueError(
            f"{type_text!r} is not a type DuckDB recognises, so a UDF cannot return it."
        ) from exc


class _ignoring_missing:
    """`remove_function` on a name that is not there raises; that is not an error here."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, *_: object) -> bool:
        return exc_type is not None


def udf(
    f: Callable[..., Any] | DataType | str | None = None,
    returnType: DataType | str | None = None,
    *,
    vectorised: bool = False,
    callOnNull: bool = False,
) -> Any:
    """Build a UDF against the **active** session, as a call or a decorator.

        square = F.udf(lambda x: x * x, "bigint")

        @F.udf(returnType="bigint")
        def square(x): ...

    Registers on the active session, which is what makes the same function reachable
    from `Session.sql()` too. A session built directly rather than through
    `Session.builder` is not the active one -- use `session.udf.register(...)` there,
    which needs no ambient state and is the form every test here uses.
    """
    from icetl.sql.session import Session

    # `F.udf("bigint")` and `F.udf(returnType="bigint")` are both the decorator form.
    declared_default: DataType | str = _DEFAULT_RETURN_TYPE if returnType is None else returnType

    if f is None or isinstance(f, (DataType, str)):
        declared = declared_default if f is None else f

        def decorator(func: Callable[..., Any]) -> UserDefinedFunction:
            return _register_anonymous(
                Session.active(),
                func,
                declared,
                vectorised=vectorised,
                callOnNull=callOnNull,
            )

        return decorator

    if not callable(f):
        raise EngineTypeError(f"udf() expects a callable, got {type(f).__name__}.")
    return _register_anonymous(
        Session.active(), f, declared_default, vectorised=vectorised, callOnNull=callOnNull
    )


def pandas_udf(
    f: Callable[..., Any] | DataType | str | None = None,
    returnType: DataType | str | None = None,
    *,
    callOnNull: bool = False,
) -> Any:
    """`udf`, for a function taking and returning `pandas.Series`.

    Runs through DuckDB's Arrow UDF protocol, so the function is called once per
    vector rather than once per row.
    """
    return udf(f, returnType, vectorised=True, callOnNull=callOnNull)


def _register_anonymous(
    session: Session,
    func: Callable[..., Any],
    returnType: DataType | str,
    *,
    vectorised: bool,
    callOnNull: bool = False,
) -> UserDefinedFunction:
    """Register a function nobody named, under a name nobody would write."""
    given = getattr(func, "__name__", "lambda")
    safe = "".join(char if char.isalnum() else "_" for char in given).strip("_") or "udf"
    name = f"icetl_udf_{safe}_{next(_ANONYMOUS)}"
    return session.udf.register(
        name, func, returnType, vectorised=vectorised, callOnNull=callOnNull
    )

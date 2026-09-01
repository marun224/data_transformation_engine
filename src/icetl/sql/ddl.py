"""`CREATE`, `DROP` and `ALTER` -- the DDL surface. Phase 9.

Everything here changes *metadata*: a schema, a partition spec, a sort order, a property,
or the existence of a table or namespace. Nothing here reads or writes rows, except
`CREATE TABLE ... AS SELECT`, which hands the rows to the Phase 7 writer rather than
inventing a second way in.

**Two things shape the module.**

*Nullability is honoured, which means the table is built from an Arrow schema rather than
handed to `saveAsTable`.* A table created by a write has all-optional fields, because its
schema comes from DuckDB, which has no non-nullable expression -- recorded in
`divergence.md` since Phase 7. `CREATE TABLE t (id BIGINT NOT NULL)` says something
DuckDB cannot, and silently dropping the constraint is the failure mode this project
spends its effort avoiding. So the DataType is turned into an Arrow schema through the
existing `createDataFrame` path -- no second type mapping -- and the declared columns are
then marked non-nullable before PyIceberg is asked to create the table.

*sqlglot does not parse every spelling Spark accepts, and says so.* `ALTER TABLE t DROP
COLUMN a`, `UNSET TBLPROPERTIES`, and the whole of Iceberg's Spark SQL extensions --
`ADD PARTITION FIELD`, `WRITE ORDERED BY` -- come back as `exp.Command`, sqlglot's
explicit "I did not understand this, here is the text" escape hatch. `run_alter_command`
reads those few forms itself. It is a small grammar, kept deliberately small, and
anything outside it is refused by name rather than guessed at.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlglot import exp

from icetl.errors import (
    AnalysisException,
    EngineValueError,
    NamespaceNotFoundError,
    ParseException,
    QueryExecutionException,
    TableAlreadyExistsException,
    TableNotFoundError,
    UnsupportedFeatureError,
)
from icetl.plan.builder import as_expression, assert_no_version, source_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pyarrow as pa
    from pyiceberg.catalog import Catalog
    from pyiceberg.table import Table
    from pyiceberg.transforms import Transform

    from icetl.catalog.resolver import TableRef
    from icetl.sql.session import Session

__all__ = [
    "drop_table",
    "run_alter",
    "run_alter_command",
    "run_create",
    "run_drop",
]

#: `CREATE NAMESPACE`, `CREATE DATABASE` and `CREATE SCHEMA` all name the same thing.
_NAMESPACE_KINDS = frozenset({"NAMESPACE", "DATABASE", "SCHEMA"})


# -- CREATE ----------------------------------------------------------------------


def run_create(session: Session, plan: exp.Create) -> None:
    kind = str(plan.args.get("kind") or "").upper()
    if kind in _NAMESPACE_KINDS:
        _create_namespace(session, plan)
        return
    if kind != "TABLE":
        raise UnsupportedFeatureError(f"CREATE {kind or '<unknown>'} statements", phase="Phase 11")
    _create_table(session, plan)


def _create_namespace(session: Session, plan: exp.Create) -> None:
    target = plan.this
    if not isinstance(target, exp.Table):
        raise ParseException("CREATE NAMESPACE needs a name.")
    namespace = _namespace_of(target)
    catalog = session._registry.get()
    if catalog.namespace_exists(namespace):
        if plan.args.get("exists"):
            return
        raise AnalysisException(
            f"Namespace {'.'.join(namespace)!r} already exists. Use CREATE NAMESPACE IF NOT EXISTS."
        )
    catalog.create_namespace(namespace, _string_properties(plan))


def _create_table(session: Session, plan: exp.Create) -> None:
    """`CREATE TABLE`, with or without a column list, with or without a query."""
    options = _CreateOptions.read(plan)
    target, columns = _create_target(plan)
    ref = _resolve_ref(session, target)
    catalog = session._registry.get(ref.catalog)
    exists = catalog.table_exists(ref.identifier)

    if exists:
        if plan.args.get("exists"):
            return
        if not plan.args.get("replace"):
            raise TableAlreadyExistsException(
                f"Table {'.'.join(ref.identifier)!r} already exists. Use CREATE TABLE "
                f"IF NOT EXISTS, or CREATE OR REPLACE TABLE to rebuild it."
            )

    query = plan.expression
    if columns is None and query is None:
        raise AnalysisException(
            "CREATE TABLE needs either a column list or AS SELECT ... to take its schema from."
        )

    from icetl.sql.writer import iceberg_ready

    frame = session._frame_for(query) if query is not None else None
    if columns is not None:
        schema = _arrow_schema_from_columns(session, columns)
    else:
        assert frame is not None
        schema = iceberg_ready(frame.toArrow()).schema

    if exists:
        # OR REPLACE: the table is dropped and rebuilt, so its snapshot history goes with
        # it. Recorded in divergence.md -- the reference keeps the table and adds a
        # snapshot, which preserves time travel across the replace.
        catalog.drop_table(ref.identifier)

    _ensure_namespace(catalog, ref)
    try:
        table = catalog.create_table(ref.identifier, schema=schema, properties=options.properties)
    except Exception as exc:
        raise QueryExecutionException(
            f"Could not create table {'.'.join(ref.identifier)!r}: {exc}"
        ) from exc

    if options.partitioned_by:
        _apply_partitioning(table, options.partitioned_by)
    session._invalidate_source(ref)

    if frame is not None:
        # By position, which is what the column list means when both are given.
        frame.write.insertInto(".".join(ref.identifier))


@dataclass(frozen=True)
class _CreateOptions:
    """The `USING` / `PARTITIONED BY` / `TBLPROPERTIES` / `COMMENT` a CREATE carries."""

    properties: dict[str, str]
    partitioned_by: tuple[exp.Expression, ...]

    @classmethod
    def read(cls, plan: exp.Create) -> _CreateOptions:
        container = plan.args.get("properties")
        items = list(container.expressions) if isinstance(container, exp.Properties) else []
        properties: dict[str, str] = {}
        partitioned: tuple[exp.Expression, ...] = ()

        for item in items:
            if isinstance(item, exp.FileFormatProperty):
                fmt = item.this.name if hasattr(item.this, "name") else str(item.this)
                if fmt.lower() != "iceberg":
                    raise UnsupportedFeatureError(
                        f"USING {fmt}",
                        hint="Iceberg is the only table format this engine creates",
                    )
            elif isinstance(item, exp.PartitionedByProperty):
                inner = item.this
                partitioned = tuple(inner.expressions if isinstance(inner, exp.Schema) else [inner])
            elif isinstance(item, exp.SchemaCommentProperty):
                properties["comment"] = _text(item.this)
            elif isinstance(item, exp.LocationProperty):
                raise UnsupportedFeatureError(
                    "CREATE TABLE ... LOCATION",
                    hint="The catalog owns the location; set it on the namespace instead",
                )
            elif isinstance(item, exp.Property):
                properties[_text(item.this)] = _text(item.args.get("value"))
            else:
                raise UnsupportedFeatureError(
                    f"CREATE TABLE ... {type(item).__name__}",
                    hint="USING, PARTITIONED BY, TBLPROPERTIES and COMMENT are supported",
                )
        return cls(properties=properties, partitioned_by=partitioned)


def _create_target(plan: exp.Create) -> tuple[exp.Table, list[exp.ColumnDef] | None]:
    """The table being created, and its declared columns when it has a column list."""
    target = plan.this
    if isinstance(target, exp.Schema):
        inner = target.this
        if not isinstance(inner, exp.Table):
            raise ParseException("CREATE TABLE needs a table name.")
        columns = [item for item in target.expressions if isinstance(item, exp.ColumnDef)]
        if len(columns) != len(target.expressions):
            raise UnsupportedFeatureError(
                "CREATE TABLE with a table-level constraint",
                hint="Column definitions only; NOT NULL on a column is supported",
            )
        return inner, columns
    if isinstance(target, exp.Table):
        return target, None
    raise ParseException(f"CREATE TABLE needs a table name, got {type(target).__name__}.")


def _arrow_schema_from_columns(session: Session, columns: Sequence[exp.ColumnDef]) -> pa.Schema:
    """The Arrow schema a declared column list describes, `NOT NULL` included.

    The types go through `createDataFrame`, so a DDL type means here exactly what it
    means everywhere else -- there is no second mapping to drift. Only the nullability is
    applied afterwards, because that is the one thing the DuckDB round trip cannot carry.
    """
    from icetl.parse_types import parse_struct_ddl

    declared = ", ".join(
        f"{_quoted(column.name)} {column.args['kind'].sql(dialect='spark')}" for column in columns
    )
    try:
        struct = parse_struct_ddl(declared)
    except Exception as exc:
        raise AnalysisException(f"Could not read the column list: {exc}") from exc

    from icetl.sql.writer import iceberg_ready

    schema = iceberg_ready(session.createDataFrame([], struct).toArrow()).schema
    for index, column in enumerate(columns):
        if _is_not_null(column):
            schema = schema.set(index, schema.field(index).with_nullable(False))
    return schema


def _is_not_null(column: exp.ColumnDef) -> bool:
    for constraint in column.args.get("constraints") or []:
        if isinstance(constraint.kind, exp.NotNullColumnConstraint):
            return not constraint.kind.args.get("allow_null")
    return False


# -- DROP ------------------------------------------------------------------------


def run_drop(session: Session, plan: exp.Drop) -> None:
    kind = str(plan.args.get("kind") or "").upper()
    target = plan.this
    if not isinstance(target, exp.Table):
        raise ParseException(f"DROP needs a name, got {type(target).__name__}.")

    if kind in _NAMESPACE_KINDS:
        _drop_namespace(session, plan, target)
        return
    if kind not in ("TABLE", ""):
        raise UnsupportedFeatureError(f"DROP {kind} statements", phase="Phase 11")

    name = source_key(target)
    dropped = drop_table(
        session,
        name,
        if_exists=bool(plan.args.get("exists")),
        purge=bool(plan.args.get("purge")),
    )
    if not dropped and not plan.args.get("exists"):  # pragma: no cover - drop_table raises
        raise TableNotFoundError(f"Table {name!r} does not exist.")


def drop_table(session: Session, name: str, *, if_exists: bool, purge: bool) -> bool:
    """Drop `name`. Returns False when it was not there and `if_exists` allowed that."""
    ref = session._resolver.parse(name)
    if not ref.namespace:
        raise AnalysisException(
            f"Table reference {name!r} has no namespace and no current namespace is set."
        )
    catalog = session._registry.get(ref.catalog)
    if not catalog.table_exists(ref.identifier):
        if if_exists:
            return False
        raise TableNotFoundError(f"Table {name!r} does not exist.")

    # The cached source has to go *before* the table does: it holds a loaded PyIceberg
    # table, and a later read finding it would query metadata that is no longer there.
    session._invalidate_source(ref)
    if purge:
        catalog.purge_table(ref.identifier)
    else:
        catalog.drop_table(ref.identifier)
    return True


def _drop_namespace(session: Session, plan: exp.Drop, target: exp.Table) -> None:
    namespace = _namespace_of(target)
    catalog = session._registry.get()
    if not catalog.namespace_exists(namespace):
        if plan.args.get("exists"):
            return
        raise NamespaceNotFoundError(f"Namespace {'.'.join(namespace)!r} does not exist.")

    tables = list(catalog.list_tables(namespace))
    if tables and not plan.args.get("cascade"):
        raise AnalysisException(
            f"Namespace {'.'.join(namespace)!r} is not empty: it holds "
            f"{len(tables)} table(s). Use DROP NAMESPACE ... CASCADE to drop them too."
        )
    for identifier in tables:
        session._invalidate_source(session._resolver.parse(".".join(identifier)))
        catalog.drop_table(identifier)
    catalog.drop_namespace(namespace)


# -- ALTER -----------------------------------------------------------------------


def run_alter(session: Session, plan: exp.Alter) -> None:
    kind = str(plan.args.get("kind") or "").upper()
    if kind not in ("TABLE", ""):
        raise UnsupportedFeatureError(f"ALTER {kind} statements", phase="Phase 11")
    target = plan.this
    if not isinstance(target, exp.Table):
        raise ParseException(f"ALTER TABLE needs a table, got {type(target).__name__}.")

    ref = _resolve_ref(session, target)
    table = _load(session, ref)
    for action in plan.args.get("actions") or []:
        _apply_action(session, ref, table, action)
        if isinstance(action, exp.AlterRename):
            # The table no longer answers to `ref`, so there is nothing left to reload
            # and nothing a later action in the same statement could address.
            break
        # Reloaded between actions so each one sees the last one's commit -- a type
        # change on a column the previous action renamed has to find the new name.
        table = _load(session, ref)
    session._invalidate_source(ref)


def _apply_action(session: Session, ref: TableRef, table: Table, action: exp.Expression) -> None:
    """One `ALTER TABLE` action, applied and committed on its own.

    One at a time rather than batched into a single `UpdateSchema`, so a statement that
    fails halfway says which action failed -- and so the two-step actions (a rename that
    is followed by a type change on the new name) see each other.
    """
    if isinstance(action, exp.Schema):  # ADD COLUMNS (a INT, b STRING)
        for column in action.expressions:
            _add_column(session, table, column)
        return
    if isinstance(action, exp.ColumnDef):  # ADD COLUMN a INT
        _add_column(session, table, action)
        return
    if isinstance(action, exp.Drop):  # DROP COLUMNS (a, b)
        for name in _dropped_names(action):
            _drop_column(table, name)
        return
    if isinstance(action, exp.RenameColumn):
        _rename_column(table, action.this.name, action.args["to"].name)
        return
    if isinstance(action, exp.AlterColumn):
        _alter_column(table, action)
        return
    if isinstance(action, exp.AlterRename):
        _rename_table(session, ref, action.this)
        return
    if isinstance(action, exp.AlterSet):
        _set_properties(table, action)
        return
    raise UnsupportedFeatureError(
        f"ALTER TABLE ... {type(action).__name__}",
        hint=(
            "ADD/DROP/RENAME/ALTER COLUMN, RENAME TO, SET/UNSET TBLPROPERTIES, "
            "ADD/DROP/REPLACE PARTITION FIELD and WRITE ORDERED BY are supported"
        ),
    )


def _add_column(session: Session, table: Table, column: exp.ColumnDef) -> None:
    from icetl.parse_types import parse_datatype_string

    kind = column.args.get("kind")
    if kind is None:
        raise ParseException(f"ADD COLUMN {column.name!r} needs a type.")
    if _is_not_null(column):
        raise AnalysisException(
            f"ADD COLUMN {column.name!r} cannot be NOT NULL: existing rows would have no "
            f"value for it. Add it optional, backfill, then make it required."
        )
    datatype = parse_datatype_string(kind.sql(dialect="spark"))
    field = _arrow_field(session, column.name, datatype)
    # `union_by_name` takes an Arrow schema and does the type mapping itself, so this
    # needs no Arrow-to-Iceberg conversion of its own. `add_column` would: it wants an
    # `IcebergType`, and the public Arrow converter refuses a schema without field ids.
    import pyarrow as arrow

    with table.update_schema() as update:
        update.union_by_name(arrow.schema([field]))
    doc = _doc_of(column)
    if doc is not None:
        with table.update_schema() as update:
            update.update_column(column.name, doc=doc)


def _drop_column(table: Table, name: str) -> None:
    _require_column(table, name)
    with table.update_schema() as update:
        update.delete_column(name)


def _rename_column(table: Table, name: str, new_name: str) -> None:
    _require_column(table, name)
    with table.update_schema() as update:
        update.rename_column(name, new_name)


def _alter_column(table: Table, action: exp.AlterColumn) -> None:
    """`ALTER COLUMN c TYPE t`, `... DROP NOT NULL`, `... COMMENT '...'`."""
    name = action.this.name if hasattr(action.this, "name") else str(action.this)
    _require_column(table, name)

    if action.args.get("dtype") is not None:
        raise UnsupportedFeatureError(
            f"ALTER COLUMN {name!r} TYPE",
            hint=(
                "Iceberg allows only widening promotions (int->long, float->double, "
                "decimal scale). PyIceberg 0.11 exposes no type update, so this would "
                "have to rewrite every file"
            ),
        )
    if action.args.get("comment") is not None:
        with table.update_schema() as update:
            update.update_column(name, doc=_text(action.args["comment"]))
        return
    if action.args.get("drop") and action.args.get("allow_null"):
        with table.update_schema() as update:
            update.make_column_optional(name)
        return
    raise UnsupportedFeatureError(
        f"ALTER COLUMN {name!r} with this change",
        hint="DROP NOT NULL and COMMENT are the supported column changes",
    )


def _rename_table(session: Session, ref: TableRef, target: exp.Expression) -> None:
    if not isinstance(target, exp.Table):
        raise ParseException("RENAME TO needs a table name.")
    catalog = session._registry.get(ref.catalog)
    new_ref = session._resolver.parse(source_key(target))
    if not new_ref.namespace:
        # `RENAME TO t2` with no namespace means the table's own, not the session's.
        new_ref = session._resolver.parse(".".join([*ref.namespace, new_ref.name]))
    session._invalidate_source(ref)
    catalog.rename_table(ref.identifier, new_ref.identifier)


def _set_properties(table: Table, action: exp.AlterSet) -> None:
    updates: dict[str, Any] = {}
    for item in action.expressions:
        if isinstance(item, exp.Properties):
            for prop in item.expressions:
                updates[_text(prop.this)] = _text(prop.args.get("value"))
        elif isinstance(item, exp.Property):
            updates[_text(item.this)] = _text(item.args.get("value"))
    if not updates:
        raise UnsupportedFeatureError(
            "ALTER TABLE ... SET with no TBLPROPERTIES",
            hint="SET TBLPROPERTIES ('k'='v') is the supported form",
        )
    with table.transaction() as transaction:
        transaction.set_properties(**updates)


def _dropped_names(action: exp.Drop) -> list[str]:
    inner = action.this
    if isinstance(inner, exp.Schema):
        return [item.name for item in inner.expressions]
    return [inner.name]


# -- the forms sqlglot hands back as `Command` -----------------------------------
#
# `ALTER TABLE t DROP COLUMN a` (singular), `UNSET TBLPROPERTIES`, and Iceberg's Spark
# SQL extensions. sqlglot 30.17 parses none of them; it returns the raw text instead,
# which is its documented escape hatch rather than a failure.

_TABLE_NAME = r"(?P<table>[A-Za-z_][\w$]*(?:\.[A-Za-z_][\w$]*)*|`[^`]+`(?:\.`[^`]+`)*)"

_DROP_COLUMN = re.compile(
    rf"^\s*TABLE\s+{_TABLE_NAME}\s+DROP\s+COLUMNS?\s+(?:IF\s+EXISTS\s+)?(?P<column>[\w$.]+)\s*$",
    re.IGNORECASE,
)
_UNSET_PROPERTIES = re.compile(
    rf"^\s*TABLE\s+{_TABLE_NAME}\s+UNSET\s+TBLPROPERTIES\s*"
    r"(?:IF\s+EXISTS\s*)?\((?P<keys>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_ADD_PARTITION = re.compile(
    rf"^\s*TABLE\s+{_TABLE_NAME}\s+ADD\s+PARTITION\s+FIELD\s+(?P<spec>.+?)"
    r"(?:\s+AS\s+(?P<alias>[\w$]+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_DROP_PARTITION = re.compile(
    rf"^\s*TABLE\s+{_TABLE_NAME}\s+DROP\s+PARTITION\s+FIELD\s+(?P<spec>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_REPLACE_PARTITION = re.compile(
    rf"^\s*TABLE\s+{_TABLE_NAME}\s+REPLACE\s+PARTITION\s+FIELD\s+(?P<old>.+?)\s+WITH\s+"
    r"(?P<new>.+?)(?:\s+AS\s+(?P<alias>[\w$]+))?\s*$",
    re.IGNORECASE | re.DOTALL,
)
_WRITE_ORDERED = re.compile(
    rf"^\s*TABLE\s+{_TABLE_NAME}\s+WRITE\s+(?:LOCALLY\s+)?ORDERED\s+BY\s+(?P<order>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_WRITE_UNORDERED = re.compile(rf"^\s*TABLE\s+{_TABLE_NAME}\s+WRITE\s+UNORDERED\s*$", re.IGNORECASE)


def run_alter_command(session: Session, plan: exp.Command) -> None:
    """An `ALTER` sqlglot declined to parse. Only the known forms; the rest are refused."""
    tail = str(plan.args.get("expression") or "")

    for pattern, handler in (
        (_DROP_COLUMN, _command_drop_column),
        (_UNSET_PROPERTIES, _command_unset_properties),
        (_REPLACE_PARTITION, _command_replace_partition),
        (_ADD_PARTITION, _command_add_partition),
        (_DROP_PARTITION, _command_drop_partition),
        (_WRITE_ORDERED, _command_write_ordered),
        (_WRITE_UNORDERED, _command_write_unordered),
    ):
        match = pattern.match(tail)
        if match is None:
            continue
        ref = session._resolver.parse(match.group("table").replace("`", ""))
        handler(session, ref, _load(session, ref), match)
        session._invalidate_source(ref)
        return

    raise UnsupportedFeatureError(
        f"ALTER{tail.rstrip()}",
        hint=(
            "Supported: ADD/DROP/RENAME/ALTER COLUMN, RENAME TO, SET/UNSET TBLPROPERTIES, "
            "ADD/DROP/REPLACE PARTITION FIELD, WRITE ORDERED BY, WRITE UNORDERED"
        ),
    )


def _command_drop_column(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    _drop_column(table, match.group("column"))


def _command_unset_properties(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    keys = [key.strip().strip("'\"`") for key in match.group("keys").split(",")]
    keys = [key for key in keys if key]
    if not keys:
        raise ParseException("UNSET TBLPROPERTIES needs at least one key.")
    with table.transaction() as transaction:
        transaction.remove_properties(*keys)


def _command_add_partition(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    _add_partition_field(table, match.group("spec"), match.groupdict().get("alias"))


def _command_drop_partition(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    name = _partition_field_name(table, match.group("spec"))
    with table.update_spec() as update:
        update.remove_field(name)


def _command_replace_partition(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    name = _partition_field_name(table, match.group("old"))
    with table.update_spec() as update:
        update.remove_field(name)
    _add_partition_field(_reload(session, ref), match.group("new"), match.groupdict().get("alias"))


def _command_write_ordered(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    from pyiceberg.table.sorting import NullOrder
    from pyiceberg.transforms import IdentityTransform

    with table.update_sort_order() as update:
        for term in _split_top_level(match.group("order")):
            column, descending, nulls_first = _parse_sort_term(term)
            _require_column(table, column)
            order = NullOrder.NULLS_FIRST if nulls_first else NullOrder.NULLS_LAST
            if descending:
                update.desc(column, IdentityTransform(), order)
            else:
                update.asc(column, IdentityTransform(), order)


def _command_write_unordered(
    session: Session, ref: TableRef, table: Table, match: re.Match[str]
) -> None:
    with table.update_sort_order():
        # Committing an update that adds no field clears the order, which is what
        # WRITE UNORDERED means.
        pass


def _parse_sort_term(term: str) -> tuple[str, bool, bool]:
    """`amount DESC NULLS FIRST` -> `("amount", True, True)`.

    Nulls default the way the reference orders them, not the way SQL does: nulls first
    ascending, last descending. The conformance layer makes the same choice for
    `ORDER BY`, and a sort order that disagreed with it would write files ordered one way
    and read them expecting the other.
    """
    words = term.split()
    if not words:
        raise ParseException("WRITE ORDERED BY needs a column.")
    column = words[0].strip('`"')
    rest = " ".join(words[1:]).upper()
    descending = "DESC" in rest
    if "NULLS FIRST" in rest:
        return column, descending, True
    if "NULLS LAST" in rest:
        return column, descending, False
    return column, descending, not descending


# -- partitioning ----------------------------------------------------------------

_TRANSFORMS: dict[str, str] = {
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "day": "day",
    "days": "day",
    "date": "day",
    "hour": "hour",
    "hours": "hour",
    "date_hour": "hour",
}


def _apply_partitioning(table: Table, terms: Sequence[exp.Expression]) -> None:
    with table.update_spec() as update:
        for term in terms:
            column, transform = _partition_term(as_expression(term))
            _require_column(table, column)
            if transform is None:
                update.add_identity(column)
            else:
                update.add_field(column, transform)


def _add_partition_field(table: Table, text: str, alias: str | None) -> None:
    import sqlglot

    from icetl.compat import SQL_DIALECT

    try:
        node = sqlglot.parse_one(text.strip(), read=SQL_DIALECT)
    except Exception as exc:
        raise ParseException(f"Could not read the partition field {text.strip()!r}: {exc}") from exc
    column, transform = _partition_term(as_expression(node))
    _require_column(table, column)
    with table.update_spec() as update:
        if transform is None:
            if alias:
                update.add_field(column, _identity(), alias)
            else:
                update.add_identity(column)
        else:
            update.add_field(column, transform, alias)


def _partition_field_name(table: Table, text: str) -> str:
    """The spec field a `DROP`/`REPLACE PARTITION FIELD` names, by field name or by term."""
    stripped = text.strip().strip('`"')
    existing = {field.name for field in table.spec().fields}
    if stripped in existing:
        return stripped
    import sqlglot

    from icetl.compat import SQL_DIALECT

    try:
        node = sqlglot.parse_one(stripped, read=SQL_DIALECT)
    except Exception:
        node = None
    if node is not None:
        column, transform = _partition_term(as_expression(node))
        wanted = str(transform) if transform is not None else "identity"
        for field in table.spec().fields:
            source = table.schema().find_field(field.source_id).name
            if source == column and str(field.transform) == wanted:
                return field.name
    raise AnalysisException(
        f"No partition field {stripped!r} on this table. "
        f"Fields: {', '.join(sorted(existing)) or '(none)'}."
    )


def _partition_term(term: exp.Expression) -> tuple[str, Transform[Any, Any] | None]:
    """One `PARTITIONED BY` term as `(column, transform)`; None means identity."""
    from pyiceberg.transforms import (
        BucketTransform,
        DayTransform,
        HourTransform,
        MonthTransform,
        TruncateTransform,
        YearTransform,
    )

    if isinstance(term, exp.Paren):
        return _partition_term(term.this)
    # A bare column arrives as an `Identifier` inside `PARTITIONED BY` and as a `Column`
    # everywhere else; both name the same thing.
    if isinstance(term, (exp.Column, exp.Identifier)):
        return term.name, None
    if isinstance(term, exp.PartitionedByBucket):
        return _bucket_or_truncate(term, BucketTransform)
    if isinstance(term, exp.PartitionByTruncate):
        return _bucket_or_truncate(term, TruncateTransform)

    if isinstance(term, exp.Anonymous):
        name = str(term.this).lower()
        arguments = list(term.expressions)
        builders = {
            "year": YearTransform,
            "month": MonthTransform,
            "day": DayTransform,
            "hour": HourTransform,
        }
        if name in _TRANSFORMS and len(arguments) == 1:
            return _column_name(arguments[0]), builders[_TRANSFORMS[name]]()
        if name in ("bucket", "truncate") and len(arguments) == 2:
            width, column = _width_and_column(arguments)
            sized: Transform[Any, Any] = (
                BucketTransform(width) if name == "bucket" else TruncateTransform(width)
            )
            return column, sized

    # A typed node sqlglot produced for one of the date transforms.
    for node_type, builder in (
        (exp.Year, YearTransform),
        (exp.Month, MonthTransform),
        (exp.Day, DayTransform),
    ):
        if isinstance(term, node_type):
            return _column_name(term.this), builder()

    raise UnsupportedFeatureError(
        f"the partition transform {term.sql(dialect='spark')!r}",
        hint=(
            "Supported: a bare column, years/months/days/hours(col), "
            "bucket(n, col) and truncate(n, col)"
        ),
    )


def _bucket_or_truncate(term: exp.Expression, builder: Any) -> tuple[str, Transform[Any, Any]]:
    """sqlglot models `bucket(16, c)` and `truncate(10, c)` as nodes of their own.

    Both put the column in `this` and the width in `expression`, whichever order they
    were written in.
    """
    column, width = term.args.get("this"), term.args.get("expression")
    if width is None or column is None:
        raise ParseException(f"Could not read {term.sql(dialect='spark')!r}.")
    return _column_name(column), builder(int(width.name))


def _width_and_column(arguments: Sequence[exp.Expression]) -> tuple[int, str]:
    first, second = arguments
    if isinstance(first, exp.Literal):
        return int(first.name), _column_name(second)
    return int(second.name), _column_name(first)


def _column_name(node: exp.Expression) -> str:
    if isinstance(node, exp.Column):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    raise ParseException(f"Expected a column, got {node.sql(dialect='spark')!r}.")


def _identity() -> Transform[Any, Any]:
    from pyiceberg.transforms import IdentityTransform

    return IdentityTransform()


# -- shared helpers --------------------------------------------------------------


def _resolve_ref(session: Session, target: exp.Table) -> TableRef:
    assert_no_version(target, "DDL against")
    ref = session._resolver.parse(source_key(target))
    if not ref.namespace:
        raise AnalysisException(
            f"Table reference {source_key(target)!r} has no namespace and no current "
            f"namespace is set. Use a qualified name like 'nyc.{ref.name}'."
        )
    return ref


def _load(session: Session, ref: TableRef) -> Table:
    catalog = session._registry.get(ref.catalog)
    if not catalog.table_exists(ref.identifier):
        raise TableNotFoundError(f"Table {'.'.join(ref.identifier)!r} does not exist.")
    return catalog.load_table(ref.identifier)


def _reload(session: Session, ref: TableRef) -> Table:
    return session._registry.get(ref.catalog).load_table(ref.identifier)


def _ensure_namespace(catalog: Catalog, ref: TableRef) -> None:
    # Best effort: some catalogs manage namespaces themselves and refuse to be told
    # about them, which is not a reason to fail the create.
    import contextlib

    with contextlib.suppress(Exception):
        catalog.create_namespace_if_not_exists(ref.namespace)


def _require_column(table: Table, name: str) -> None:
    available = [field.name for field in table.schema().fields]
    if name not in available:
        raise AnalysisException(
            f"No column {name!r} on this table. Columns: {', '.join(available)}."
        )


def _arrow_field(session: Session, name: str, datatype: Any) -> pa.Field:
    """One column's Arrow field, typed through the same path every other schema takes."""
    from icetl.sql.writer import iceberg_ready
    from icetl.types import StructField, StructType

    frame = session.createDataFrame([], StructType([StructField(name, datatype, True)]))
    return iceberg_ready(frame.toArrow()).schema.field(0)


def _doc_of(column: exp.ColumnDef) -> str | None:
    for constraint in column.args.get("constraints") or []:
        if isinstance(constraint.kind, exp.CommentColumnConstraint):
            return _text(constraint.kind.this)
    return None


def _namespace_of(target: exp.Table) -> tuple[str, ...]:
    parts = [part for part in (target.catalog, target.db, target.name) if part]
    if not parts:
        raise ParseException("A namespace needs a name.")
    return tuple(parts)


def _string_properties(plan: exp.Create) -> dict[str, str]:
    container = plan.args.get("properties")
    if not isinstance(container, exp.Properties):
        return {}
    return {
        _text(item.this): _text(item.args.get("value"))
        for item in container.expressions
        if isinstance(item, exp.Property) and item.args.get("value") is not None
    }


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside brackets."""
    out, depth, current = [], 0, ""
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        out.append(current.strip())
    return out


def _text(node: Any) -> str:
    if node is None:
        raise EngineValueError("Expected a value, got nothing.")
    if isinstance(node, exp.Literal):
        return node.name
    if isinstance(node, exp.Identifier):
        return node.name
    if isinstance(node, exp.Expression):
        return node.name or node.sql(dialect="spark").strip("'\"")
    return str(node)


def _quoted(name: str) -> str:
    escaped = name.replace("`", "``")
    return f"`{escaped}`"

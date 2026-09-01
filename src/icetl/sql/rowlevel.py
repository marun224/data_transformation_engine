"""`DELETE`, `UPDATE` and `MERGE` -- row-level operations, copy-on-write.

Phase 8. The reference engine exposes all three on the SQL surface only, and so does
this: `Session.sql("DELETE FROM ...")` is the whole API, and every statement returns
the empty frame a statement returns.

**One idea does the work.** A row-level operation is a `SELECT` whose result *is* the
new contents of the rows it touches, plus an Iceberg commit that swaps those rows for
it. Nothing here re-implements filtering, expression evaluation or joins -- the plan
goes through the same compile pipeline every query does (P1), so conformance,
pushdown and the optimizer all apply to it unchanged.

**The predicate is the dangerous part, and it is where most of this file goes.** The
commit is `Table.overwrite(rows, overwrite_filter=P)`: PyIceberg deletes the rows
matching `P` and appends `rows`. So `P` and the `WHERE` clause of the SELECT that
produced `rows` must select **exactly** the same rows. Wider, and rows are deleted
that were never written back -- data loss. Narrower, and rows survive the delete *and*
come back in `rows` -- duplication. Read pushdown never faces this, because there the
SQL re-applies the filter and an over-wide PyIceberg predicate only costs I/O.

The gate is `pushdown.is_exactly_translatable`, and the trick that makes it safe is
that both languages are generated from **one** set of sqlglot nodes:
`scope_predicate` returns the PyIceberg expression *and* the very nodes it was built
from, and those nodes are what goes into the SELECT's `WHERE`. A conjunct that cannot
be translated exactly is dropped from both at once, which only ever widens the scope
-- more rows read and written back untouched, never a wrong answer.

**Three-valued logic is the other place this goes quietly wrong.** `DELETE ... WHERE
c` deletes the rows where `c` is *true*; a row where `c` is NULL survives, and
`NOT c` is NULL for exactly those rows, so `WHERE NOT c` would delete them too. The
survivor filter is therefore `NOT COALESCE(c, FALSE)`. `UPDATE` needs no such care --
`CASE WHEN c THEN new ELSE old END` already falls to `ELSE` on NULL, which is the rule
itself.

**Concurrency.** Iceberg commits are optimistic. The table is re-read at the start of
every attempt, its snapshot id is checked immediately before the commit, and a
statement whose table moved underneath it is planned again from scratch rather than
committed against metadata it never read. `commit_with_retry` handles the narrower
race inside PyIceberg's own commit.

Merge-on-read is Phase 13: every operation here rewrites data files, which is what
PyIceberg does natively and what decision 11 fixed as the mode for now.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pyiceberg.expressions import AlwaysTrue, BooleanExpression
from pyiceberg.types import DateType, IntegerType, LongType, StringType
from sqlglot import exp

from icetl.errors import (
    AnalysisException,
    ParseException,
    QueryExecutionException,
    UnsupportedFeatureError,
)
from icetl.plan.builder import assert_no_version, source_key
from icetl.plan.pushdown import ColumnResolver, conjuncts, scope_predicate
from icetl.sql.writer import commit_with_retry

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pyarrow as pa
    from pyiceberg.schema import Schema as IcebergSchema
    from pyiceberg.table import Table

    from icetl.catalog.resolver import TableRef
    from icetl.plan.builder import ScanSource
    from icetl.sql.session import Session

__all__ = ["run_delete", "run_merge", "run_update"]

#: How many times a statement is re-planned after finding the table moved under it.
_ATTEMPTS = 4
_BACKOFF_SECONDS = 0.1

#: The most distinct join-key values that will be turned into an `IN` list to narrow a
#: MERGE's scope. Past this the list costs more to bind and evaluate than the pruning
#: it buys, so the scope widens to the whole table instead.
_KEY_VALUE_LIMIT = 1000

#: Key types whose SQL literal and whose PyIceberg literal are the same value. The scope
#: `IN` list is generated in both languages, so a type whose two spellings could disagree
#: -- floats, decimals, timestamps with their zone questions -- is left out rather than
#: reasoned about.
_NARROWABLE_KEY_TYPES = (IntegerType, LongType, StringType, DateType)


# -- the target table ------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """The table a statement writes to, loaded at the snapshot it will be planned on."""

    key: str
    node: exp.Table
    alias: str
    source: ScanSource
    columns: tuple[str, ...]
    resolver: ColumnResolver
    snapshot_id: int | None

    @property
    def table(self) -> Table:
        return self.source.resolved.table

    @property
    def ref(self) -> TableRef:
        return self.source.resolved.ref

    @property
    def schema(self) -> IcebergSchema:
        return self.source.resolved.table.schema()

    @property
    def name(self) -> str:
        return self.source.resolved.qualified_name

    def column_named(self, name: str) -> str | None:
        """This table's spelling of `name`, or None -- matched case-insensitively."""
        lowered = name.lower()
        return next((column for column in self.columns if column.lower() == lowered), None)


def _resolve_target(session: Session, node: exp.Table) -> _Target:
    """Load the target table fresh, and hand the plan the same snapshot we validate.

    The session caches a `ScanSource` per reference, pinned to the snapshot it was
    loaded at -- which is what makes repeated reads cheap and what would make this read
    stale. Dropping the entry first means the frame built below and the table committed
    to are the same object at the same snapshot, so the check before the commit is
    checking the thing that was actually read.
    """
    assert_no_version(node, "DELETE / UPDATE / MERGE against")
    key = source_key(node)
    probe = session._resolver.resolve(key)
    session._invalidate_source(probe.ref)
    source = session._source_for(key)
    schema = source.resolved.table.schema()
    alias = node.alias or node.name
    current = source.resolved.table.current_snapshot()
    return _Target(
        key=key,
        node=node,
        alias=alias,
        source=source,
        columns=tuple(field.name for field in schema.fields),
        resolver=ColumnResolver(alias, {f.name: f.field_type for f in schema.fields}),
        snapshot_id=None if current is None else current.snapshot_id,
    )


def _run_optimistic(
    session: Session,
    node: exp.Table,
    plan_one: Callable[[_Target], Callable[[], None] | None],
) -> None:
    """Plan the statement against a freshly loaded table, then commit if it is still current.

    `plan_one` does the reading and returns the commit to make, or `None` when the
    statement turns out to have nothing to do. The whole of it is re-run -- not
    replayed -- when the snapshot moved in between, because the rows to write back are
    a function of the rows that were there.
    """
    for attempt in range(_ATTEMPTS):
        target = _resolve_target(session, node)
        commit = plan_one(target)
        if commit is None:
            session._invalidate_source(target.ref)
            return
        fresh = target.source.resolved.catalog.load_table(target.ref.identifier)
        current = fresh.current_snapshot()
        moved = (None if current is None else current.snapshot_id) != target.snapshot_id
        if not moved:
            commit_with_retry(commit)
            session._invalidate_source(target.ref)
            return
        time.sleep(_BACKOFF_SECONDS * (2**attempt))
    raise QueryExecutionException(
        f"Gave up after {_ATTEMPTS} attempts: another writer committed to "
        f"{node.sql(dialect='spark')} each time this statement was planned."
    )


def _conform(data: pa.Table, target: _Target) -> pa.Table:
    """Cast the computed rows to the table's own Arrow schema before writing them.

    DuckDB types the result, and it types by its own rules: `SET n = 1` produces a
    32-bit integer for a 64-bit column, and a `CASE` over two branches widens to
    whichever is larger. PyIceberg validates the incoming schema against the table's
    and rejects a mismatch, so without this an ordinary `UPDATE` would fail on a type
    difference that has nothing to do with what the user asked for.

    Names come from the table, positionally: the projection is built in the table's
    column order, so this is a relabelling rather than a match.
    """
    from pyiceberg.io.pyarrow import schema_to_pyarrow

    expected = schema_to_pyarrow(target.schema)
    if len(data.schema.names) != len(expected.names):  # pragma: no cover - defensive
        raise AnalysisException(
            f"{target.name} has {len(expected.names)} column(s) but the rewritten rows "
            f"have {len(data.schema.names)}."
        )
    data = data.rename_columns(list(expected.names))
    try:
        return data.cast(expected)
    except Exception as exc:
        raise AnalysisException(
            f"The rows computed for {target.name} do not fit its schema: {exc}"
        ) from exc


# -- small sqlglot builders ------------------------------------------------------


def _ident(name: str) -> exp.Identifier:
    return exp.to_identifier(name, quoted=True)


def _qualified(alias: str, column: str) -> exp.Column:
    return exp.Column(this=_ident(column), table=_ident(alias))


def _all_of(terms: Sequence[exp.Expression]) -> exp.Expression | None:
    """`a AND b AND ...`, or None when there is nothing to require."""
    if not terms:
        return None
    # sqlglot annotates its builders with the looser `Condition` base.
    return cast(exp.Expression, exp.and_(*(term.copy() for term in terms)))


def _is_true(condition: exp.Expression) -> exp.Expression:
    """`COALESCE(condition, FALSE)` -- the condition, with NULL read as false.

    SQL's `WHERE` already reads NULL as false; this is for the places that need the
    condition as a *value*, where NULL would propagate instead.
    """
    return exp.Coalesce(this=exp.paren(condition.copy()), expressions=[exp.false()])


def _select(session: Session, plan: exp.Select) -> pa.Table:
    """Run a generated plan through the ordinary compile pipeline."""
    return session._frame_for(plan).toArrow()


# -- DELETE ----------------------------------------------------------------------


def run_delete(session: Session, plan: exp.Delete) -> None:
    """`DELETE FROM t WHERE ...`.

    Two routes. When every conjunct translates exactly, PyIceberg's own `delete` runs
    it -- and it can drop a file whose every row matches without reading the file at
    all, which no rewrite of ours could. Otherwise the surviving rows are computed here
    and swapped in for the scope the translatable conjuncts describe.
    """
    node = plan.this
    if not isinstance(node, exp.Table):
        raise ParseException(f"DELETE needs a table, got {type(node).__name__}.")
    if plan.args.get("using"):
        raise UnsupportedFeatureError(
            "DELETE ... USING",
            hint="Express the join as a subquery in the WHERE clause instead",
        )
    condition = _where(plan)

    def planned(target: _Target) -> Callable[[], None] | None:
        predicate, kept, dropped = scope_predicate(
            conjuncts(condition), target.resolver, target.schema, exact_only=True
        )
        if condition is None:
            table = target.table
            return lambda: table.delete()
        if not dropped:
            # The predicate *is* the WHERE clause, so PyIceberg can be asked for the
            # delete rather than told the answer.
            table = target.table
            return lambda: table.delete(predicate)

        survivors = exp.Select(
            expressions=[
                exp.alias_(_qualified(target.alias, column), column, quoted=True)
                for column in target.columns
            ]
        ).from_(target.node.copy())
        survives = exp.not_(_is_true(condition))
        scope = _all_of(kept)
        survivors = survivors.where(exp.and_(scope, survives) if scope else survives)

        data = _conform(_select(session, survivors), target)
        table = target.table
        return lambda: table.overwrite(data, overwrite_filter=predicate)

    _run_optimistic(session, node, planned)


# -- UPDATE ----------------------------------------------------------------------


def run_update(session: Session, plan: exp.Update) -> None:
    """`UPDATE t SET c = e, ... WHERE ...`.

    Every row in scope is rewritten as `CASE WHEN <where> THEN <new> ELSE <old> END`,
    column by column, which gets two rules for free: a row the `WHERE` does not select
    keeps its value, and every right-hand side sees the row's **old** values, because
    they are all projections of the same input row.
    """
    node = plan.this
    if not isinstance(node, exp.Table):
        raise ParseException(f"UPDATE needs a table, got {type(node).__name__}.")
    # sqlglot spells the key `from_` on an UPDATE and `from` elsewhere.
    if plan.args.get("from_") or plan.args.get("from"):
        raise UnsupportedFeatureError(
            "UPDATE ... FROM",
            hint="Express the other table as a subquery in the SET expression or WHERE",
        )
    condition = _where(plan)

    def planned(target: _Target) -> Callable[[], None] | None:
        assignments = _assignments(plan.expressions, target)
        predicate, kept, _ = scope_predicate(
            conjuncts(condition), target.resolver, target.schema, exact_only=True
        )
        projections = []
        for column in target.columns:
            old = _qualified(target.alias, column)
            new = assignments.get(column)
            if new is None:
                value: exp.Expression = old
            elif condition is None:
                value = new.copy()
            else:
                value = exp.Case(ifs=[exp.If(this=condition.copy(), true=new.copy())], default=old)
            projections.append(exp.alias_(value, column, quoted=True))

        updated = exp.Select(expressions=projections).from_(target.node.copy())
        scope = _all_of(kept)
        if scope is not None:
            updated = updated.where(scope)

        data = _conform(_select(session, updated), target)
        table = target.table
        return lambda: table.overwrite(data, overwrite_filter=predicate)

    _run_optimistic(session, node, planned)


def _assignments(items: Sequence[exp.Expression], target: _Target) -> dict[str, exp.Expression]:
    """The `SET` list as `{column: expression}`, checked against the table."""
    out: dict[str, exp.Expression] = {}
    for item in items:
        if not isinstance(item, exp.EQ):
            raise ParseException(f"UPDATE ... SET expects assignments, got {item.sql()!r}.")
        left = item.this
        if not isinstance(left, exp.Column):
            raise ParseException(
                f"UPDATE ... SET expects a column on the left, got {left.sql()!r}."
            )
        column = _assignment_column(left, target)
        if column in out:
            raise AnalysisException(f"Column {column!r} is assigned more than once.")
        out[column] = item.expression
    if not out:
        raise ParseException("UPDATE needs at least one assignment.")
    return out


def _assignment_column(left: exp.Column, target: _Target) -> str:
    """The table column a `SET` left-hand side names, with any alias stripped.

    `t.vendor` and `vendor` are the same column; `person.name` is a *field* of the
    column `person` -- and sqlglot spells both as two parts, so the shape cannot tell
    them apart. The table can: a first part that is one of its own columns is a column,
    not an alias. Getting this backwards would turn a refused nested assignment into a
    silent one on the wrong column.
    """
    parts = [part.name for part in left.parts]
    if len(parts) > 1 and target.column_named(parts[0]) is None and parts[0] == target.alias:
        parts = parts[1:]
    if len(parts) > 1 and target.column_named(parts[0]) is not None:
        raise UnsupportedFeatureError(
            f"assigning to the nested field {left.sql()!r}",
            phase="Phase 9",
            hint="Rebuild the whole struct instead: SET s = named_struct(...)",
        )
    column = target.column_named(parts[0]) if len(parts) == 1 else None
    if column is None:
        raise AnalysisException(
            f"{target.name} has no column {left.sql()!r}. Columns: {', '.join(target.columns)}."
        )
    return column


def _where(plan: exp.Expression) -> exp.Expression | None:
    where = plan.args.get("where")
    return where.this if isinstance(where, exp.Where) else None


# -- MERGE -----------------------------------------------------------------------


@dataclass(frozen=True)
class _When:
    """One `WHEN ...` clause, read off the parse tree."""

    kind: str  # "matched" | "not_matched" | "by_source"
    action: str  # "update" | "delete" | "insert"
    condition: exp.Expression | None
    #: `{column: expression}` for UPDATE; `None` means `SET *`.
    assignments: dict[str, exp.Expression] | None = None
    #: Target-column-ordered expressions for INSERT; `None` means `INSERT *`.
    inserts: list[exp.Expression] | None = None


def run_merge(session: Session, plan: exp.Merge) -> None:
    """`MERGE INTO t USING s ON ... WHEN ...`.

    Copy-on-write, in three queries whose results are written as one commit:

    | Query | Rows it produces |
    |---|---|
    | matched | target rows that join a source row, after the first `WHEN MATCHED` that fires |
    | unmatched | target rows that join none, after the first `WHEN NOT MATCHED BY SOURCE` |
    | inserted | source rows that join no target row, expanded by `WHEN NOT MATCHED` |

    **Three queries rather than one join, deliberately.** The obvious shape is a single
    LEFT JOIN with a marker column saying whether the join found anything -- and it
    works, right up until the optimizer merges the marker's subquery into the outer
    query and constant-folds the marker to `TRUE`. Every row then looks matched, the
    delete branch fires for all of them, and the statement empties the table while
    reporting success. Splitting the halves puts matchedness in an `EXISTS`, which is a
    predicate no rewrite can quietly reinterpret, and an inner join, where every row is
    matched by construction. It costs one more scan and buys a correctness property
    that does not depend on which rules the optimizer happens to run.

    A merge with **only** `WHEN NOT MATCHED` clauses changes no existing row, so it
    skips the rewrite and appends -- which is the common upsert-into-new-partitions
    shape, and turns the whole statement into one snapshot.
    """
    node = plan.this
    if not isinstance(node, exp.Table):
        raise ParseException(f"MERGE INTO needs a table, got {type(node).__name__}.")
    using = plan.args.get("using")
    if not isinstance(using, (exp.Table, exp.Subquery)):
        raise ParseException("MERGE ... USING needs a table or a subquery.")
    on = plan.args.get("on")
    if not isinstance(on, exp.Expression):
        raise ParseException("MERGE needs an ON condition.")

    source_alias = using.alias or (using.name if isinstance(using, exp.Table) else "")
    if not source_alias:
        raise AnalysisException(
            "MERGE ... USING (subquery) needs an alias, so the WHEN clauses can name "
            "its columns: USING (SELECT ...) AS s"
        )
    source = _aliased(using, source_alias)
    source_columns = _columns_of(session, source)

    def planned(target: _Target) -> Callable[[], None] | None:
        whens = _whens(plan, target, source_alias, source_columns)
        matched_clauses = [when for when in whens if when.kind == "matched"]
        by_source_clauses = [when for when in whens if when.kind == "by_source"]
        insert_clauses = [when for when in whens if when.kind == "not_matched"]

        predicate, scope, matchable = _merge_scope(
            session, target, on, source, source_alias, by_source_clauses
        )
        # Nothing in the target can satisfy the ON condition, so no clause that acts on
        # a target row can fire however many of them there are.
        rewrites = matchable and bool(matched_clauses or by_source_clauses)

        added = (
            _merge_inserts(session, target, source, source_alias, on, scope, insert_clauses)
            if insert_clauses
            else None
        )
        if not rewrites:
            if added is None or added.num_rows == 0:
                return None
            table = target.table
            return lambda: table.append(added)

        _assert_one_match_per_target_row(session, target, source, on, scope)
        rows = [
            _merge_matched(session, target, source, on, scope, matched_clauses),
            _merge_unmatched(session, target, source, on, scope, by_source_clauses),
        ]
        if added is not None and added.num_rows:
            rows.append(added)

        import pyarrow as pa

        combined = pa.concat_tables(rows)
        table = target.table
        return lambda: table.overwrite(combined, overwrite_filter=predicate)

    _run_optimistic(session, node, planned)


# -- reading the WHEN clauses ----------------------------------------------------


def _whens(
    plan: exp.Merge, target: _Target, source_alias: str, source_columns: Sequence[str]
) -> list[_When]:
    """The `WHEN` clauses in the order they were written, which is the order they fire."""
    container = plan.args.get("whens")
    clauses = list(container.expressions) if isinstance(container, exp.Whens) else []
    if not clauses:
        raise ParseException("MERGE needs at least one WHEN clause.")

    out: list[_When] = []
    for clause in clauses:
        kind = (
            "by_source"
            if clause.args.get("source")
            else ("matched" if clause.args.get("matched") else "not_matched")
        )
        condition = clause.args.get("condition")
        then = clause.args.get("then")

        if isinstance(then, exp.Var) and str(then.this).upper() == "DELETE":
            _refuse_on_target(kind, "DELETE")
            out.append(_When(kind=kind, action="delete", condition=condition))
        elif isinstance(then, exp.Update):
            _refuse_on_target(kind, "UPDATE")
            out.append(
                _When(
                    kind=kind,
                    action="update",
                    condition=condition,
                    assignments=_merge_update_set(then, target, source_alias, source_columns, kind),
                )
            )
        elif isinstance(then, exp.Insert):
            if kind != "not_matched":
                raise AnalysisException(
                    f"{_spelling(kind)} cannot INSERT -- the target row already exists."
                )
            out.append(
                _When(
                    kind=kind,
                    action="insert",
                    condition=condition,
                    inserts=_merge_insert_values(then, target, source_alias, source_columns),
                )
            )
        else:
            spelled = then.sql(dialect="spark") if then is not None else "<missing>"
            raise UnsupportedFeatureError(
                f"MERGE action {spelled!r}",
                hint="UPDATE SET ..., DELETE and INSERT ... are the actions",
            )
        if kind == "by_source":
            _refuse_source_references(out[-1], source_alias)
    return out


def _refuse_source_references(when: _When, source_alias: str) -> None:
    """`WHEN NOT MATCHED BY SOURCE` acts on rows no source row matched, so there is none.

    Without this the statement still fails -- the query is built without the source
    relation, and binding reports `Referenced table "s" not found`, which is true and
    says nothing about why. Only *qualified* references are checked, because an
    unqualified name in a clause that has one relation in scope is that relation's.
    """
    nodes = [when.condition] if when.condition is not None else []
    nodes += list((when.assignments or {}).values())
    for node in nodes:
        for column in node.find_all(exp.Column):
            if column.table == source_alias:
                raise AnalysisException(
                    f"WHEN NOT MATCHED BY SOURCE cannot read {column.sql()!r}: it acts on "
                    f"target rows that matched no source row, so there is no {source_alias!r} "
                    f"row to read from."
                )


def _refuse_on_target(kind: str, action: str) -> None:
    if kind == "not_matched":
        raise AnalysisException(
            f"WHEN NOT MATCHED can only INSERT -- there is no target row to {action}."
        )


def _spelling(kind: str) -> str:
    return {
        "matched": "WHEN MATCHED",
        "not_matched": "WHEN NOT MATCHED",
        "by_source": "WHEN NOT MATCHED BY SOURCE",
    }[kind]


def _merge_update_set(
    then: exp.Update,
    target: _Target,
    source_alias: str,
    source_columns: Sequence[str],
    kind: str,
) -> dict[str, exp.Expression] | None:
    """`SET a = ..., b = ...` as a mapping, with `SET *` expanded to the source's columns."""
    items = list(then.expressions)
    if len(items) == 1 and isinstance(items[0], exp.Star):
        if kind == "by_source":
            raise AnalysisException(
                "WHEN NOT MATCHED BY SOURCE ... UPDATE SET * has no source row to copy from."
            )
        # Expanded here rather than left as a marker, so a source missing one of the
        # target's columns is a named error now instead of a binding failure later.
        values = _star_values(target, source_alias, source_columns, "SET *")
        return dict(zip(target.columns, values, strict=True))
    return _assignments(items, target)


def _merge_insert_values(
    then: exp.Insert, target: _Target, source_alias: str, source_columns: Sequence[str]
) -> list[exp.Expression]:
    """The insert expressions, in the **target's** column order."""
    columns = then.this
    values = then.expression
    if isinstance(columns, exp.Star):
        return _star_values(target, source_alias, source_columns, "INSERT *")
    if not isinstance(values, exp.Tuple):
        raise ParseException("WHEN NOT MATCHED ... INSERT needs VALUES (...) or INSERT *.")
    supplied = list(values.expressions)

    if columns is None:
        # `INSERT VALUES (...)` with no column list -- positional, as `insertInto` is.
        if len(supplied) != len(target.columns):
            raise AnalysisException(
                f"INSERT VALUES matches columns by position, and gives {len(supplied)} "
                f"value(s) where {target.name} has {len(target.columns)} column(s)."
            )
        return supplied
    if not isinstance(columns, exp.Tuple):
        raise ParseException(f"INSERT expects a column list, got {columns.sql()!r}.")

    names = []
    for item in columns.expressions:
        resolved = target.resolver.name(item) if isinstance(item, exp.Column) else None
        if resolved is None:
            raise AnalysisException(
                f"INSERT names a column {item.sql()!r} that {target.name} does not have."
            )
        names.append(resolved)
    if len(names) != len(supplied):
        raise AnalysisException(
            f"INSERT lists {len(names)} column(s) and {len(supplied)} value(s)."
        )

    by_name = dict(zip(names, supplied, strict=True))
    # A column the INSERT does not name gets NULL, as it does in the reference.
    return [by_name.get(column, exp.Null()) for column in target.columns]


def _star_values(
    target: _Target, source_alias: str, source_columns: Sequence[str], spelling: str
) -> list[exp.Expression]:
    """`*`: the source column of the same name, for each of the target's columns."""
    available = {name.lower(): name for name in source_columns}
    values: list[exp.Expression] = []
    for column in target.columns:
        found = available.get(column.lower())
        if found is None:
            raise AnalysisException(
                f"{spelling} needs a source column for every target column, and the "
                f"source has none called {column!r}. "
                f"Source columns: {', '.join(source_columns) or '(none)'}."
            )
        values.append(_qualified(source_alias, found))
    return values


# -- the three halves ------------------------------------------------------------


def _merge_matched(
    session: Session,
    target: _Target,
    source: exp.Expression,
    on: exp.Expression,
    scope: Sequence[exp.Expression],
    clauses: Sequence[_When],
) -> pa.Table:
    """Target rows that join a source row, after whichever clause fired for each.

    An **inner** join, so every row here is matched by definition and the clause
    conditions are the only thing left to test. One `CASE` chain per column, in clause
    order, so the first matching clause wins -- which is the rule -- and a row no clause
    claims falls through to its own value.
    """
    select = exp.Select(expressions=_clause_projections(target, clauses)).from_(target.node.copy())
    select.set("joins", [exp.Join(this=source.copy(), kind="INNER", on=on.copy())])
    return _conform(_select(session, _restrict(select, [*scope, *_survives(clauses)])), target)


def _merge_unmatched(
    session: Session,
    target: _Target,
    source: exp.Expression,
    on: exp.Expression,
    scope: Sequence[exp.Expression],
    clauses: Sequence[_When],
) -> pa.Table:
    """Target rows the source never mentions, after any `WHEN NOT MATCHED BY SOURCE`.

    With no such clause this is just "everything else in scope, unchanged" -- which
    still has to be selected, because the commit replaces the whole scope.
    """
    select = exp.Select(expressions=_clause_projections(target, clauses)).from_(target.node.copy())
    unmatched = exp.not_(_exists(source, on))
    return _conform(
        _select(session, _restrict(select, [*scope, unmatched, *_survives(clauses)])), target
    )


def _merge_inserts(
    session: Session,
    target: _Target,
    source: exp.Expression,
    source_alias: str,
    on: exp.Expression,
    scope: Sequence[exp.Expression],
    clauses: Sequence[_When],
) -> pa.Table:
    """Source rows that match no target row, expanded by the `WHEN NOT MATCHED` clauses.

    The `EXISTS` looks at the target through the same scope the rewrite uses. That is
    safe because the scope is built to contain every row the `ON` condition could match:
    a target row outside it could not have matched anyway.
    """
    projections: list[exp.Expression] = []
    for position, column in enumerate(target.columns):
        ifs: list[exp.If] = []
        default: exp.Expression = exp.Null()
        for when in clauses:
            assert when.inserts is not None
            value = when.inserts[position].copy()
            if when.condition is None:
                default = value  # nothing after an unconditional clause can fire
                break
            ifs.append(exp.If(this=when.condition.copy(), true=value))
        projections.append(
            cast(
                exp.Expression,
                exp.alias_(
                    exp.Case(ifs=ifs, default=default) if ifs else default, column, quoted=True
                ),
            )
        )

    select = exp.Select(expressions=projections).from_(source.copy())
    conditions: list[exp.Expression] = [
        exp.not_(_exists(_restrict(exp.Select().from_(target.node.copy()), scope), on))
    ]
    claimed = [when.condition.copy() for when in clauses if when.condition is not None]
    if len(claimed) == len(clauses):
        # Every clause is conditional, so a source row matching none of them is not
        # inserted at all -- rather than inserted as a row of NULLs.
        conditions.append(
            cast(exp.Expression, exp.or_(*claimed)) if len(claimed) > 1 else claimed[0]
        )
    return _conform(_select(session, _restrict(select, conditions)), target)


def _clause_projections(target: _Target, clauses: Sequence[_When]) -> list[exp.Expression]:
    """One projection per target column: the `CASE` chain the clauses build for it.

    An **unconditional** clause becomes the `ELSE` rather than a `WHEN TRUE` branch.
    That is the honest reading -- nothing after it can fire -- and it also avoids a
    `simplify` defect that folds a whole `CASE` down to the value of a non-leading
    always-true branch, discarding the branches before it. `optimizer` normalises the
    same shape away for hand-written SQL; not generating it is the cheaper half.
    """
    projections: list[exp.Expression] = []
    for column in target.columns:
        old = _qualified(target.alias, column)
        ifs: list[exp.If] = []
        default: exp.Expression = old
        for when in clauses:
            value = _merge_value(when, column, old)
            if when.condition is None:
                if value is not None:
                    default = value
                break
            if value is not None:
                ifs.append(exp.If(this=when.condition.copy(), true=value))
        projections.append(
            cast(
                exp.Expression,
                exp.alias_(
                    exp.Case(ifs=ifs, default=default) if ifs else default, column, quoted=True
                ),
            )
        )
    return projections


def _merge_value(when: _When, column: str, old: exp.Expression) -> exp.Expression | None:
    """What `column` becomes when `when` fires, or None when the clause leaves it alone.

    A `DELETE` clause still contributes a branch, holding the column's own value. The
    row is filtered out afterwards so the value is never read -- but without the branch,
    a *later* clause would fire for a row this one already claimed.
    """
    if when.action == "delete":
        return old.copy()
    assert when.assignments is not None
    assigned = when.assignments.get(column)
    return None if assigned is None else assigned.copy()


def _survives(clauses: Sequence[_When]) -> list[exp.Expression]:
    """The filter that drops the rows a `DELETE` clause claimed, if there are any."""
    if not any(when.action == "delete" for when in clauses):
        return []
    ifs: list[exp.If] = []
    fired: exp.Expression = exp.false()
    for when in clauses:
        deletes = cast(exp.Expression, exp.convert(when.action == "delete"))
        if when.condition is None:
            fired = deletes
            break
        ifs.append(exp.If(this=when.condition.copy(), true=deletes))
    if ifs:
        fired = exp.Case(ifs=ifs, default=fired)
    return [cast(exp.Expression, exp.not_(exp.paren(fired)))]


def _restrict(select: exp.Select, conditions: Sequence[exp.Expression]) -> exp.Select:
    combined = _all_of(conditions)
    return select if combined is None else select.where(combined)


def _exists(relation: exp.Expression, condition: exp.Expression) -> exp.Exists:
    """`EXISTS (SELECT 1 FROM relation WHERE condition)`.

    Correlated: `condition` is the `ON` clause, which names the outer row's columns.
    This is what carries "matched" through the optimizer intact -- a rule may move it,
    but no rule can turn it into a constant the way it can a marker column.
    """
    inner = relation if isinstance(relation, exp.Select) else exp.Select().from_(relation.copy())
    inner = inner.copy()
    inner.set("expressions", [exp.Literal.number(1)])
    return exp.Exists(this=inner.where(condition.copy()))


def _assert_one_match_per_target_row(
    session: Session,
    target: _Target,
    source: exp.Expression,
    on: exp.Expression,
    scope: Sequence[exp.Expression],
) -> None:
    """Refuse a merge where one target row matches several source rows.

    The rewrite joins, so a target row matching twice would be written back twice --
    and even deduplicated, which clause won would depend on which of the two source rows
    the engine happened to look at. The reference refuses it for the same reason.

    The check is a count against a count: the inner join emits one row per (target,
    source) pair, so it exceeds the number of *matched target rows* exactly when some
    target row has more than one partner.
    """
    matched = _restrict(
        exp.Select(expressions=[exp.alias_(exp.Literal.number(1), "n", quoted=True)]).from_(
            target.node.copy()
        ),
        [*scope, _exists(source, on)],
    )
    joined = _restrict(
        exp.Select(expressions=[exp.alias_(exp.Literal.number(1), "n", quoted=True)]).from_(
            target.node.copy()
        ),
        scope,
    )
    joined.set("joins", [exp.Join(this=source.copy(), kind="INNER", on=on.copy())])

    if session._frame_for(joined).count() > session._frame_for(matched).count():
        raise AnalysisException(
            f"MERGE INTO {target.name}: a row of the target matched more than one row of "
            f"the source. Which one would win is undefined, so the statement is refused. "
            f"Deduplicate the source on the ON columns first."
        )


# -- how much of the target a merge can touch ------------------------------------


def _merge_scope(
    session: Session,
    target: _Target,
    on: exp.Expression,
    source: exp.Expression,
    source_alias: str,
    by_source_clauses: Sequence[_When],
) -> tuple[BooleanExpression, list[exp.Expression], bool]:
    """The scope, in both languages, and whether any target row can match at all.

    Two things narrow it. The `ON` condition's target-only conjuncts restrict which rows
    can match, exactly as a `WHERE` would. And an equality between a target column and a
    source column restricts it much further: a target row can only match if its key is
    one of the values the source actually holds, so the distinct source keys become an
    `IN` list. That is what turns "merge 10 rows into a 41M-row table" from a full
    rewrite into a handful of files.

    A `WHEN NOT MATCHED BY SOURCE` clause acts on rows the source never mentions, so it
    widens the scope back to the whole table and neither narrowing applies.
    """
    if by_source_clauses:
        return AlwaysTrue(), [], True

    terms = conjuncts(on)
    candidates = [term for term in terms if target.resolver.owns_every_column(term)]
    for column, expression in _key_pairs(terms, target, source_alias):
        values = _distinct_values(session, source, expression)
        if values is None:
            continue
        if not values:
            # The source holds no usable value for this key, and `ON` is a conjunction,
            # so no target row can match -- whatever the other conjuncts say.
            return AlwaysTrue(), [], False
        candidates.append(
            exp.In(
                this=_qualified(target.alias, column),
                expressions=[exp.convert(value) for value in values],
            )
        )

    predicate, kept, _ = scope_predicate(
        candidates, target.resolver, target.schema, exact_only=True
    )
    return predicate, kept, True


def _key_pairs(
    terms: Sequence[exp.Expression], target: _Target, source_alias: str
) -> list[tuple[str, exp.Column]]:
    """`t.k = s.k` conjuncts, as `(target column, source column)`."""
    pairs = []
    for term in terms:
        if not isinstance(term, exp.EQ):
            continue
        for first, second in ((term.this, term.expression), (term.expression, term.this)):
            column = target.resolver.name(first) if isinstance(first, exp.Column) else None
            if column is None or not isinstance(second, exp.Column):
                continue
            if second.table != source_alias:
                continue
            if isinstance(target.resolver.field_type(column), _NARROWABLE_KEY_TYPES):
                pairs.append((column, second))
            break
    return pairs


def _distinct_values(
    session: Session, source: exp.Expression, expression: exp.Column
) -> list[Any] | None:
    """The distinct non-NULL values of a join key in the source, or None to not narrow.

    NULLs are dropped rather than lost: a NULL key equals nothing, so a target row whose
    key is NULL matches no source row and belongs outside the scope -- which is exactly
    where `IN` puts it.
    """
    probe = (
        exp.Select(expressions=[exp.alias_(expression.copy(), "k", quoted=True)])
        .from_(source.copy())
        .distinct()
        .limit(_KEY_VALUE_LIMIT + 1)
    )
    try:
        rows = session._frame_for(probe).toArrow().column(0).to_pylist()
    except Exception:
        # A key this engine cannot evaluate on its own is a reason to prune less, not a
        # reason to fail the merge.
        return None
    if len(rows) > _KEY_VALUE_LIMIT:
        return None
    return [value for value in rows if value is not None]


def _columns_of(session: Session, source: exp.Expression) -> list[str]:
    """The source's output columns, from analysis alone -- no rows are read."""
    probe = exp.Select(expressions=[exp.Star()]).from_(source.copy())
    return session._frame_for(probe).columns


def _aliased(node: exp.Expression, alias: str) -> exp.Expression:
    """`node` under the alias the WHEN clauses name it by."""
    copied = node.copy()
    copied.set("alias", exp.TableAlias(this=_ident(alias)))
    return copied

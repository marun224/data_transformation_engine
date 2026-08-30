"""Turning an Arrow result into what the caller asked for: Rows, pandas, or text.

Arrow's Python conversion and the reference engine's disagree on the nested types, so `to_pylist`
output is walked against the schema rather than handed back raw:

    struct  ->  dict            the reference engine yields a nested `Row`
    map     ->  [(k, v), ...]   the reference engine yields a `dict`
    list    ->  list            already right, but its elements may not be

The walk is skipped entirely for schemas with no struct or map in them, which is
almost all of them, so the common path stays a straight `to_pylist`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa

from icetl.types import ArrayType, DataType, MapType, Row, StructType

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["NULL_DISPLAY", "format_show", "to_pandas", "to_rows"]

# How `show()` renders a NULL. The reference engine switched from `null` to `NULL` in 3.4
# (matching how it renders the literal elsewhere), and `the reference API.__version__` here
# claims 3.5.0, so `NULL` is the consistent choice. Recorded in divergence.md;
# this constant is the single place to change it.
NULL_DISPLAY = "NULL"

_MIN_TRUNCATE = 4  # below this the "..." suffix would be longer than the cell

# The reference engine pads every show() column out to at least three characters, so a table of
# short names still reads as a table. Narrower columns would be a visible divergence.
_MIN_COLUMN_WIDTH = 3


def _convert(value: Any, data_type: DataType) -> Any:
    """Reshape one Arrow-derived Python value into the reference engine's form."""
    if value is None:
        return None
    if isinstance(data_type, StructType):
        return Row._from_fields(
            tuple(field.name for field in data_type.fields),
            tuple(_convert(value.get(field.name), field.dataType) for field in data_type.fields),
        )
    if isinstance(data_type, MapType):
        return {key: _convert(item, data_type.valueType) for key, item in value}
    if isinstance(data_type, ArrayType):
        return [_convert(item, data_type.elementType) for item in value]
    return value


def to_rows(table: pa.Table, schema: StructType) -> list[Row]:
    """Materialise an Arrow table as a list of `Row`, matching `df.collect()`."""
    fields = tuple(field.name for field in schema.fields)
    records = table.to_pylist()
    if not schema.needConversion():
        return [
            Row._from_fields(fields, tuple(record[name] for name in fields)) for record in records
        ]
    return [
        Row._from_fields(
            fields,
            tuple(_convert(record[f.name], f.dataType) for f in schema.fields),
        )
        for record in records
    ]


def _is_text_like(arrow_type: pa.DataType) -> bool:
    return (
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or pa.types.is_string_view(arrow_type)
        or pa.types.is_binary(arrow_type)
        or pa.types.is_large_binary(arrow_type)
    )


def to_pandas(table: pa.Table) -> pd.DataFrame:
    """Convert to pandas with the reference engine's dtypes.

    The reference engine's `toPandas()` yields `object` columns holding `None` for missing strings.
    pandas 3 instead converts Arrow strings to its new `str` dtype, whose missing
    value is `nan` -- and `astype(object)` keeps the `nan`, so the sentinel has to be
    fixed as well as the dtype. Rebuilding from `to_pylist()` gets both right at once.

    Columns that are already `object` are left alone: that is the pandas 2 path,
    where Arrow's own conversion already produces exactly what the reference engine would.

    Without this, the same script would see a different dtype and a different null
    sentinel depending on which pandas happened to be installed. Recorded in
    divergence.md.
    """
    import pandas as pd

    frame = table.to_pandas()
    for index, field in enumerate(table.schema):
        if _is_text_like(field.type) and frame[field.name].dtype != object:
            frame[field.name] = pd.Series(
                table.column(index).to_pylist(), dtype=object, index=frame.index
            )
    return frame


def _cell(value: Any, truncate: int) -> str:
    """Render one value as `show()` would."""
    if value is None:
        return NULL_DISPLAY
    if isinstance(value, bool):
        # The reference engine prints Scala booleans lowercase; Python's str() would capitalise.
        text = "true" if value else "false"
    elif isinstance(value, Row):
        text = "{" + ", ".join(_cell(v, 0) for v in value) + "}"
    elif isinstance(value, dict):
        text = "{" + ", ".join(f"{_cell(k, 0)} -> {_cell(v, 0)}" for k, v in value.items()) + "}"
    elif isinstance(value, list):
        text = "[" + ", ".join(_cell(v, 0) for v in value) + "]"
    else:
        text = str(value)

    if truncate >= _MIN_TRUNCATE and len(text) > truncate:
        return text[: truncate - 3] + "..."
    return text


def format_show(
    rows: list[Row],
    schema: StructType,
    *,
    n: int,
    truncate: int,
    vertical: bool,
    has_more: bool,
) -> str:
    """Render `show()`'s output.

    Matches the reference engine's layout, including its alignment rule: cells are right-justified
    when truncation is on and left-justified when it is off (`truncate=0`), which is
    what makes untruncated wide values readable.
    """
    names = [field.name for field in schema.fields]
    body = [[_cell(value, truncate) for value in row] for row in rows[:n]]

    if vertical:
        return _format_vertical(names, body, has_more, n)

    widths = [
        max([_MIN_COLUMN_WIDTH, len(name), *(len(cells[index]) for cells in body)])
        for index, name in enumerate(names)
    ]
    justify = str.rjust if truncate > 0 else str.ljust
    separator = "+" + "+".join("-" * width for width in widths) + "+"

    lines = [
        separator,
        "|"
        + "|".join(justify(name, width) for name, width in zip(names, widths, strict=True))
        + "|",
        separator,
    ]
    lines += [
        "|"
        + "|".join(justify(cell, width) for cell, width in zip(cells, widths, strict=True))
        + "|"
        for cells in body
    ]
    lines.append(separator)
    return "\n".join(lines) + "\n" + _footer(has_more, n)


def _format_vertical(names: list[str], body: list[list[str]], has_more: bool, n: int) -> str:
    if not body:
        return "(0 rows)\n"
    label_width = max(len(name) for name in names)
    blocks = []
    for index, cells in enumerate(body):
        header = f"-RECORD {index}"
        blocks.append(
            "\n".join(
                [header]
                + [
                    f" {name.rjust(label_width)} | {cell}"
                    for name, cell in zip(names, cells, strict=True)
                ]
            )
        )
    return "\n".join(blocks) + "\n" + _footer(has_more, n)


def _footer(has_more: bool, n: int) -> str:
    return f"only showing top {n} row{'' if n == 1 else 's'}\n" if has_more else ""

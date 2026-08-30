"""Compatibility notes and, from Phase 3, the translation rules themselves.

`divergence.md` is the running record of every place the reference semantics and DuckDB
disagree. It is written as rules land, not afterwards (P5).
"""

#: sqlglot's identifier for the SQL grammar this library reads and renders names in.
#:
#: This is **not** branding left over from an older name -- it is an argument value in
#: sqlglot's own API, naming a grammar. It is what makes backtick-quoted identifiers,
#: `CAST(x AS INT)` type spellings and the DDL forms in `parse_types` parse the way the
#: reference semantics require. Changing it would change the language accepted, which is
#: a behaviour change rather than a rename.
SQL_DIALECT = "spark"

__all__ = ["SQL_DIALECT"]
